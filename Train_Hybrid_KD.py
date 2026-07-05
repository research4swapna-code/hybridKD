import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
from PIL import Image
import os, json, time
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if DEVICE.type=="cuda" else torch.float32

TEACHER_MODEL_ID = "F2La"
STUDENT_MODEL_PATH = "F2Li"

COCO_PATH = "COCO_2017"

TRAIN_SAMPLES = 10000
VAL_SAMPLES = 2000

BATCH_SIZE = 16
GRAD_ACC = 4
EPOCHS = 100
LR = 1e-5

KD_T = 2.0
L_GT = 0
L_SEQ = 0.2
L_KL = 0.8

PATIENCE = 10

OUTPUT_DIR = "train_outputs_25may_sweep1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# DATASET
# -------------------------
class CocoDataset(Dataset):
    def __init__(self, image_dir, ann_file, max_samples=None):
        with open(ann_file, "r") as f:
            data = json.load(f)

        id2file = {img["id"]: img["file_name"] for img in data["images"]}

        self.samples = []
        for ann in data["annotations"]:
            path = os.path.join(image_dir, id2file[ann["image_id"]])
            self.samples.append((path, ann["caption"]))

        if max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, caption = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return {"image": image, "caption": caption}

def collate_fn(batch):
    return {
        "images": [b["image"] for b in batch],
        "captions": [b["caption"] for b in batch]
    }

train_loader = DataLoader(
    CocoDataset(f"{COCO_PATH}/train/images/train2017",
                f"{COCO_PATH}/train/annotations/captions_train2017.json",
                TRAIN_SAMPLES),
    batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

val_loader = DataLoader(
    CocoDataset(f"{COCO_PATH}/train/images/val2017",
                f"{COCO_PATH}/train/annotations/captions_val2017.json",
                VAL_SAMPLES),
    batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

# -------------------------
# MODELS
# -------------------------
processor = AutoProcessor.from_pretrained(TEACHER_MODEL_ID, trust_remote_code=True, local_files_only=True)

teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
).to(DEVICE).eval()

for p in teacher.parameters():
    p.requires_grad = False

student = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_PATH, torch_dtype=DTYPE, trust_remote_code=True, local_files_only=True
).to(DEVICE)

if hasattr(student.config, "use_cache"):
    student.config.use_cache = False

optimizer = torch.optim.AdamW(student.parameters(), lr=LR)

pad_id = processor.tokenizer.pad_token_id

# -------------------------
# TRAINING
# -------------------------
best_val = float("inf")
counter = 0

train_losses, val_losses = [], []

for epoch in range(EPOCHS):

    # =====================
    # TRAIN
    # =====================
    student.train()
    total_train = 0
    start = time.time()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for step, batch in enumerate(pbar):

        images = batch["images"]
        captions = batch["captions"]

        inputs = processor(images=images, text=["<CAPTION>"]*len(images), return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k,v in inputs.items()}
        pixel_values = inputs["pixel_values"].to(DTYPE).contiguous()

        # ---- TEACHER ----
        with torch.no_grad():
            teacher_ids = teacher.generate(
                pixel_values=pixel_values,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=128,
                num_beams=3
            ).to(DEVICE).contiguous()

            teacher_out = teacher(
                pixel_values=pixel_values,
                decoder_input_ids=teacher_ids,
                decoder_attention_mask=(teacher_ids != pad_id)
            )
            teacher_logits = teacher_out.logits.clamp(-50, 50)

        # ---- GT ----
        gt = processor(images=images, text=captions, return_tensors="pt", padding=True)
        gt = {k: v.to(DEVICE) for k,v in gt.items()}

        gt_ids = gt["input_ids"].contiguous()
        gt_mask = gt["attention_mask"].contiguous()

        student_gt = student(
            pixel_values=pixel_values,
            decoder_input_ids=gt_ids,
            decoder_attention_mask=gt_mask,
            labels=gt_ids
        )
        loss_gt = student_gt.loss

        # ---- SEQ ----
        student_seq = student(
            pixel_values=pixel_values,
            decoder_input_ids=teacher_ids,
            decoder_attention_mask=(teacher_ids != pad_id),
            labels=teacher_ids
        )
        loss_seq = student_seq.loss
        student_logits = student_seq.logits.clamp(-50, 50)

        # ---- KL ----
        mask = (teacher_ids != pad_id).unsqueeze(-1)

        s_log_probs = F.log_softmax(student_logits / KD_T, dim=-1)
        t_probs = F.softmax(teacher_logits / KD_T, dim=-1)

        kl = F.kl_div(s_log_probs, t_probs, reduction="none").sum(-1)
        kl = kl * mask.squeeze(-1)

        num_tokens = mask.sum().clamp(min=1)
        loss_kl = (kl.sum() / num_tokens) * (KD_T**2)

        # ---- FINAL ----
        loss = L_GT*loss_gt + L_SEQ*loss_seq + L_KL*loss_kl

        loss.backward()

        if (step+1) % GRAD_ACC == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_train += loss.item()

        pbar.set_postfix({
            "loss": total_train/(step+1),
            "gt": loss_gt.item(),
            "seq": loss_seq.item(),
            "kl": loss_kl.item()
        })

    # =====================
    # VALIDATION (COMBINED KD)
    # =====================
    student.eval()

    total_val_gt = 0
    total_val_seq = 0
    total_val_kl = 0

    with torch.no_grad():
        for batch in val_loader:

            images = batch["images"]
            captions = batch["captions"]

            inputs = processor(images=images, text=["<CAPTION>"]*len(images), return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k,v in inputs.items()}
            pixel_values = inputs["pixel_values"].to(DTYPE).contiguous()

            # TEACHER
            teacher_ids = teacher.generate(
                pixel_values=pixel_values,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=128,
                num_beams=3
            ).to(DEVICE).contiguous()

            teacher_out = teacher(
                pixel_values=pixel_values,
                decoder_input_ids=teacher_ids,
                decoder_attention_mask=(teacher_ids != pad_id)
            )
            teacher_logits = teacher_out.logits.clamp(-50, 50)

            # GT
            gt = processor(images=images, text=captions, return_tensors="pt", padding=True)
            gt = {k: v.to(DEVICE) for k,v in gt.items()}

            student_gt = student(
                pixel_values=pixel_values,
                decoder_input_ids=gt["input_ids"],
                decoder_attention_mask=gt["attention_mask"],
                labels=gt["input_ids"]
            )
            loss_gt = student_gt.loss

            # SEQ
            student_seq = student(
                pixel_values=pixel_values,
                decoder_input_ids=teacher_ids,
                decoder_attention_mask=(teacher_ids != pad_id),
                labels=teacher_ids
            )
            loss_seq = student_seq.loss
            student_logits = student_seq.logits.clamp(-50, 50)

            # KL
            mask = (teacher_ids != pad_id).unsqueeze(-1)

            s_log_probs = F.log_softmax(student_logits / KD_T, dim=-1)
            t_probs = F.softmax(teacher_logits / KD_T, dim=-1)

            kl = F.kl_div(s_log_probs, t_probs, reduction="none").sum(-1)
            kl = kl * mask.squeeze(-1)

            loss_kl = (kl.sum() / mask.sum().clamp(min=1)) * (KD_T**2)

            total_val_gt += loss_gt.item()
            total_val_seq += loss_seq.item()
            total_val_kl += loss_kl.item()

    avg_val = (
        L_GT * (total_val_gt / len(val_loader)) +
        L_SEQ * (total_val_seq / len(val_loader)) +
        L_KL * (total_val_kl / len(val_loader))
    )

    avg_train = total_train / len(train_loader)

    train_losses.append(avg_train)
    val_losses.append(avg_val)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {avg_train:.4f}")
    print(f"Val Loss:   {avg_val:.4f}")
    print(f"Time: {time.time()-start:.2f}s")

    # =====================
    # EARLY STOP + SAVE
    # =====================
    if avg_val < best_val - 1e-4:
        best_val = avg_val
        counter = 0

        save_path = os.path.join(OUTPUT_DIR, "best_model")
        os.makedirs(save_path, exist_ok=True)

        student.save_pretrained(save_path)
        processor.save_pretrained(save_path)

        print("Saved best model")

    else:
        counter += 1
        print(f"No improvement ({counter}/{PATIENCE})")

        if counter >= PATIENCE:
            print("Early stopping triggered")
            break

# -------------------------
# LOSS CURVE
# -------------------------
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(os.path.join(OUTPUT_DIR, "loss.png"))
plt.close()
