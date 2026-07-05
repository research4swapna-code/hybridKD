#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cell 1: Environment Setup & Imports

print("Step 1: Setting up the environment...")
# --- Install/Upgrade Libraries ---
#!pip install -q --upgrade pip
#!pip install -q --upgrade transformers accelerate einops Pillow requests matplotlib datasets sentencepiece timm torch torchvision torchaudio "transformers[torch]"

# --- Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoConfig,
    get_scheduler
)
from datasets import load_dataset

# --- Utility Imports ---
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import os
from tqdm.auto import tqdm
import gc
import traceback
import shutil

# --- Initial Setup ---
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"Initial CUDA memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB reserved")
print("Environment setup complete.")


# In[2]:


# Cell 2: Configuration

print("Step 2: Defining configuration...")

# --- GPU Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() and DEVICE == 'cuda' else torch.float16 if DEVICE == 'cuda' else torch.float32
print(f"Using Device: {DEVICE}, Compute Dtype: {COMPUTE_DTYPE}")

# --- Model Paths & IDs ---
TEACHER_MODEL_ID = "F2La"
STUDENT_CODE_PATH = "F2Li"

# --- Dataset Configuration ---
COCO_DATASET_ID = "COCO_2017"
TRAIN_SAMPLES = 1000
VAL_SAMPLES = 200

# --- Training Configuration (OOM OPTIMIZED) ---
NUM_EPOCHS = 50
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
KD_ALPHA = 0.5
KD_TEMPERATURE = 2.0
MAX_GRAD_NORM = 1.0
WARMUP_STEPS_RATIO = 0.1

# --- Task Configuration ---
TASK_PROMPTS_MAP = {"CAPTION": "<CAPTION>"}
MAX_TARGET_LENGTH = 128

# --- Output Configuration ---
OUTPUT_DIR = "D-Florence"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Model outputs will be saved to: {OUTPUT_DIR}")
print(f"OOM MITIGATION: Batch Size set to {BATCH_SIZE}, Gradient Accumulation set to {GRADIENT_ACCUMULATION_STEPS}")


# In[3]:


# Cell 3: Load Models and Processor

# --- User Action Prerequisite ---
os.makedirs(STUDENT_CODE_PATH, exist_ok=True)
if not os.path.exists(os.path.join(STUDENT_CODE_PATH, 'modeling_florence2.py')):
    raise FileNotFoundError(f"Student code directory '{STUDENT_CODE_PATH}' is missing or does not contain your modified modeling files. Please create and populate it first.")

processor, teacher_model, student_model = None, None, None

# --- 1. Load SHARED Processor ---
try:
    print(f"\n--- Loading Shared Processor for: {TEACHER_MODEL_ID} ---")
    processor = AutoProcessor.from_pretrained(TEACHER_MODEL_ID, trust_remote_code=True, local_files_only=True)
    print("Shared Florence-2 processor loaded successfully.")
except Exception as e:
    raise RuntimeError(f"CRITICAL ERROR: Failed to load processor. {e}")

# --- 2. Load Teacher Model ---
try:
    print(f"\n--- Loading Teacher Model from Hub: {TEACHER_MODEL_ID} ---")
    teacher_model = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32)
    teacher_model = teacher_model.to(dtype=COMPUTE_DTYPE, device=DEVICE).eval()
    print(f"Teacher Model loaded. Parameters: {sum(p.numel() for p in teacher_model.parameters()):,}")
except Exception as e:
    raise RuntimeError(f"CRITICAL ERROR: Failed to load teacher model. {e}")

# --- 3. Instantiate Student Model ---
try:
    print(f"\n--- Instantiating Student Model from Modified Code in: {STUDENT_CODE_PATH} ---")
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_CODE_PATH, trust_remote_code=True, torch_dtype=torch.float32, local_files_only=True
    )
    
    # Enable Gradient Checkpointing for memory savings
    if hasattr(student_model, "gradient_checkpointing_enable"):
        student_model.gradient_checkpointing_enable()
        print("Gradient Checkpointing enabled for the student model.")
    
    student_model = student_model.to(dtype=COMPUTE_DTYPE, device=DEVICE)
    print(f"Student 'Florence-2 Light' Model Instantiated. Parameters: {sum(p.numel() for p in student_model.parameters()):,}")

    untrained_student_dir = os.path.join(OUTPUT_DIR, "untrained_student_model_checkpoint")
    print(f"\nSaving UNTRAINED student model to demonstrate size reduction...")
    student_model.save_pretrained(untrained_student_dir)
    processor.save_pretrained(untrained_student_dir)
    weight_file = "model.safetensors" if os.path.exists(os.path.join(untrained_student_dir, "model.safetensors")) else "pytorch_model.bin"
    if os.path.exists(os.path.join(untrained_student_dir, weight_file)):
        file_size_mb = os.path.getsize(os.path.join(untrained_student_dir, weight_file)) / (1024 * 1024)
        print(f"✅ Visible size reduction confirmed: Untrained student's weight file is ~{file_size_mb:.2f} MB.")
except Exception as e:
    raise RuntimeError(f"CRITICAL ERROR: Failed to instantiate student model from your modified code. {e}")

if torch.cuda.is_available():
    print(f"\nCUDA memory after loading both models: {torch.cuda.memory_allocated(DEVICE) / 1024**2:.2f}MB")


# In[4]:


# Cell 4: Prepare Dataset and DataLoaders

class OnlineDistillationDataset(Dataset):
    def __init__(self, hf_dataset_iterable, num_samples):
        print(f"Streaming and collecting {num_samples} samples...")
        self.data = list(tqdm(hf_dataset_iterable.take(num_samples), total=num_samples, desc="Streaming Samples"))
        print(f"Finished collecting {len(self.data)} samples.")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            return {"image_pil": item['image'].convert("RGB")}
        except Exception as e:
            print(f"Warning: Could not load image at index {idx}. Error: {e}. Skipping."); return None

def collate_fn_distill(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    return [item['image_pil'] for item in batch]

train_dataloader, val_dataloader = None, None
print(f"\n--- Preparing COCO Dataset via Streaming from Hub ---")
try:
    train_ds_iterable = load_dataset(COCO_DATASET_ID, split="train", streaming=True)
    val_ds_iterable = load_dataset(COCO_DATASET_ID, split="validation", streaming=True)
    
    train_dataset = OnlineDistillationDataset(train_ds_iterable, TRAIN_SAMPLES)
    val_dataset = OnlineDistillationDataset(val_ds_iterable, VAL_SAMPLES)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_distill, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_distill, num_workers=2)
    print(f"DataLoaders created. Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")
except Exception as e:
    raise RuntimeError(f"CRITICAL ERROR loading COCO dataset via streaming: {e}")


# In[ ]:


# Cell 5: Training Loop

# Check if all prerequisites for training are met
proceed_to_training = True

if proceed_to_training:
    print("\n--- Setting up Training Components ---")
    optimizer = optim.AdamW(student_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    num_training_steps = NUM_EPOCHS * len(train_dataloader) // GRADIENT_ACCUMULATION_STEPS
    if num_training_steps == 0: num_training_steps = 1
    lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=int(WARMUP_STEPS_RATIO * num_training_steps), num_training_steps=num_training_steps)
    scaler = GradScaler(enabled=(DEVICE == 'cuda' and COMPUTE_DTYPE != torch.float32))
    kl_div_loss_fn = nn.KLDivLoss(reduction="none")
    best_val_loss = float('inf')
    
    print("\n--- Starting Distillation Training Loop ---")
    for epoch in range(NUM_EPOCHS):
        student_model.train(); teacher_model.eval()
        total_train_loss, total_task_loss, total_kd_loss = 0, 0, 0
        
        progress_bar_train = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Train")
        for step, image_pil_batch in enumerate(progress_bar_train):
            if image_pil_batch is None or not image_pil_batch: continue
            
            if step % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.zero_grad(set_to_none=True)

            try:
                # Randomly select a task for the entire batch
                task_prompt = random.choice(list(TASK_PROMPTS_MAP.values()))
                batch_prompts = [task_prompt] * len(image_pil_batch)
                
                # Use the processor on the batch of images and prompts
                model_inputs = processor(images=image_pil_batch, text=batch_prompts, return_tensors="pt").to(DEVICE)
                pixel_values = model_inputs.pixel_values.to(dtype=COMPUTE_DTYPE)

                # ---- 1. Get Teacher's Targets ----
                with torch.no_grad(), autocast(enabled=scaler.is_enabled()):
                    teacher_gen_ids = teacher_model.generate(
                        pixel_values=pixel_values, input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask,
                        max_new_tokens=MAX_TARGET_LENGTH, num_beams=1, early_stopping=True
                    )
                    teacher_target_mask = (teacher_gen_ids != processor.tokenizer.pad_token_id).long()
                    teacher_outputs = teacher_model(
                        pixel_values=pixel_values, input_ids=teacher_gen_ids,
                        attention_mask=teacher_target_mask, labels=teacher_gen_ids
                    )
                    teacher_logits = teacher_outputs.logits

                # ---- 2. Student Forward Pass and Loss Calculation ----
                with autocast(enabled=scaler.is_enabled()):
                    student_target_mask = (teacher_gen_ids != processor.tokenizer.pad_token_id).long()
                    student_outputs = student_model(
                        pixel_values=pixel_values, input_ids=teacher_gen_ids,
                        attention_mask=student_target_mask, labels=teacher_gen_ids
                    )
                    student_logits = student_outputs.logits
                    hard_loss = student_outputs.loss

                    # KL Divergence Distillation Loss
                    len_to_match = min(student_logits.shape[1], teacher_logits.shape[1])
                    mask_for_loss = (teacher_gen_ids[:, :len_to_match] != processor.tokenizer.pad_token_id).unsqueeze(-1)
                    
                    s_log_probs = F.log_softmax(student_logits[:, :len_to_match, :] / KD_TEMPERATURE, dim=-1)
                    t_probs = F.softmax(teacher_logits[:, :len_to_match, :] / KD_TEMPERATURE, dim=-1)
                    
                    kl_unreduced = kl_div_loss_fn(s_log_probs, t_probs).sum(dim=-1)
                    kl_masked = kl_unreduced.masked_fill(~mask_for_loss.squeeze(-1), 0.0)
                    
                    num_active_tokens = mask_for_loss.sum()
                    soft_loss = (kl_masked.sum() / num_active_tokens) * (KD_TEMPERATURE ** 2) if num_active_tokens > 0 else torch.tensor(0.0, device=DEVICE)
                    
                    combined_loss = KD_ALPHA * soft_loss + (1.0 - KD_ALPHA) * hard_loss
                    
                # --- Backpropagation ---
                scaler.scale(combined_loss / GRADIENT_ACCUMULATION_STEPS).backward()
                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(student_model.parameters(), MAX_GRAD_NORM)
                    scaler.step(optimizer); scaler.update()
                    if lr_scheduler: lr_scheduler.step()

                total_train_loss += combined_loss.item(); total_task_loss += hard_loss.item(); total_kd_loss += soft_loss.item()
                progress_bar_train.set_postfix({"L": f"{total_train_loss/(step+1):.3f}", "T": f"{total_task_loss/(step+1):.3f}", "KL": f"{total_kd_loss/(step+1):.3f}"})
            
            except Exception as e_step:
                print(f"Error in training step {step}: {e_step}")
                traceback.print_exc()
                gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue
        
        # --- End of Epoch Validation & Checkpointing ---
        avg_train_loss = total_train_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        print(f"\nEpoch {epoch+1} Training Avg Loss: {avg_train_loss:.4f}")

        student_model.eval()
        total_val_loss = 0
        if val_dataloader:
            with torch.no_grad():
                for batch_val_images in tqdm(val_dataloader, desc=f"Epoch {epoch+1} Val"):
                    if batch_val_images is None: continue
                    val_task_prompt = random.choice(list(TASK_PROMPTS_MAP.values()))
                    teacher_inputs_val = processor(images=batch_val_images, text=val_task_prompt, return_tensors="pt").to(DEVICE)
                    with autocast(enabled=scaler.is_enabled()):
                        teacher_gen_ids_val = teacher_model.generate(**teacher_inputs_val, max_new_tokens=MAX_TARGET_LENGTH)
                        student_val_outputs = student_model(
                            pixel_values=teacher_inputs_val.pixel_values, input_ids=teacher_gen_ids_val,
                            attention_mask=(teacher_gen_ids_val != processor.tokenizer.pad_token_id).long(),
                            labels=teacher_gen_ids_val
                        )
                        if student_val_outputs.loss is not None: total_val_loss += student_val_outputs.loss.item()
            
            avg_val_loss = total_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else float('inf')
            print(f"Epoch {epoch+1} Validation Avg Task Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_dir = os.path.join(OUTPUT_DIR, "best_checkpoint")
                print(f"New best val loss: {best_val_loss:.4f}. Saving model to {checkpoint_dir}")
                student_model.save_pretrained(checkpoint_dir)
                processor.save_pretrained(checkpoint_dir)
                # Copy custom code to checkpoint dir for self-contained loading
                for fname in ['modeling_florence2.py', 'modeling_davit.py', 'configuration_florence2.py']:
                    if os.path.exists(os.path.join(STUDENT_CODE_PATH, fname)):
                        shutil.copy(os.path.join(STUDENT_CODE_PATH, fname), os.path.join(checkpoint_dir, fname))
        
        gc.collect(); torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None

    print("\n--- Distillation Training Finished ---")
else:
    print("Skipping training loop: Prerequisites from previous cells were not met.")


# In[ ]:


import time

print("\n--- Testing the Best Distilled Student Model ---")
best_checkpoint_dir = os.path.join(OUTPUT_DIR, "best_checkpoint")
if os.path.isdir(best_checkpoint_dir):
    try:
        print(f"Loading best model from: {best_checkpoint_dir}")
        
        # Fix 1: Update config
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(best_checkpoint_dir, trust_remote_code=True)
        
        print(config)
        
        # Ensure vision config has the right model type
        if hasattr(config, 'vision_config') and hasattr(config.vision_config, 'model_type'):
            config.vision_config.model_type = 'davit'
            config.save_pretrained(best_checkpoint_dir)
            print("Updated vision config to use DaViT")
        
        # Fix 2: Modify modeling file if needed
        modeling_file = os.path.join(best_checkpoint_dir, "modeling_florence2.py")
        if os.path.exists(modeling_file):
            with open(modeling_file, 'r') as f:
                content = f.read()
            
            if "assert config.vision_config.model_type == 'davit'" in content:
                content = content.replace(
                    "assert config.vision_config.model_type == 'davit', 'only DaViT is supported for now'",
                    "# assert config.vision_config.model_type == 'davit', 'only DaViT is supported for now'"
                )
                
                with open(modeling_file, 'w') as f:
                    f.write(content)
                print("Fixed modeling file assertion")

        # Now load the model
        loaded_processor = AutoProcessor.from_pretrained(
            best_checkpoint_dir, 
            trust_remote_code=True, 
            local_files_only=True
        )
        
        loaded_student_model = AutoModelForCausalLM.from_pretrained(
            best_checkpoint_dir, 
            trust_remote_code=True, 
            local_files_only=True,
            torch_dtype=COMPUTE_DTYPE
        )

        
        loaded_student_model = loaded_student_model.to(device=DEVICE).eval()
        print("Model reloaded successfully.")
        
        test_url = "COCO_test2014_000000004778.jpg"
        test_image = Image.open(test_url)
        start = time.time()
        for task_name, test_prompt in TASK_PROMPTS_MAP.items():
            inputs = loaded_processor(images=test_image, text=test_prompt, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                generated_ids = loaded_student_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024, 
                    num_beams=3
                )
            decoded_text = loaded_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_output = loaded_processor.post_process_generation(
                decoded_text, 
                task=test_prompt, 
                image_size=(test_image.width, test_image.height)
            )
            print(f"\n--- Inference for Task: {task_name} ---")
            print(f"Parsed Output: {parsed_output}")
        end = time.time()
        time_taken = end-start
        print(time_taken)
    except Exception as e:
        print(f"Error during test inference of saved model: {e}")
        traceback.print_exc()
else:
    print("No best checkpoint was saved to test. Skipping final inference test.")



