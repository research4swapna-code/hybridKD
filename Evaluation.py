import os
import json
from PIL import Image
import torch
from torch.profiler import record_function
from transformers import AutoProcessor, AutoModelForCausalLM
import time
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
#dtype = torch.float16 if device == "cuda" else torch.float32 
torch_dtype = torch.float32

model_path = "best_model"


model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True,
    device_map=device,
    torch_dtype=torch_dtype
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True,torch_dtype =torch_dtype)

print("Florence Model and Processor Loaded for Inference.")
#model = model.half()
model.eval()

print(model.dtype)
val_image_path = "/COCO_2017/train/images/val2017"

image_files = sorted(os.listdir(val_image_path))

results = []

for img in tqdm(image_files):
	img_path = os.path.join(val_image_path, img)
	image = Image.open(img_path).convert('RGB')
	
	
	text_prompt = "<CAPTION>" 
	inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device) 
	
	with torch.no_grad():
        	with record_function("model_generate"):
        		generated_ids = model.generate(
	    			input_ids=inputs["input_ids"],
	    			pixel_values=inputs["pixel_values"],
	    			max_new_tokens=128,
	    			num_beams=3,
	    			do_sample=False
			)
        		end_time = time.time()

	generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
	parsed_answer = processor.post_process_generation(generated_text, task="<CAPTION>", image_size=image.size)
	#print(img)
	
	image_id = int(img.split('.')[0])
	#iid = img.split('.')[1]
	
	results.append({
		"image_id": image_id,
		"caption": parsed_answer['<CAPTION>']
	})
	
with open("KD_coco_eval_caption_128_sweep1.json", 'w') as f:
	json.dump(results, f)
	
#with open("florence2_base_coco_eval_caption.json", 'w') as f:#
#	json.dump(results, f)
	




#from pycocoevalcap.eval import COCOEvalCap
#from pycocotools.coco import COCO
#from coco_caption.pycocoevalcap.eval import COCOEvalCap

#coco = COCO('COCO_2017/train/annotations/captions_val2017.json')
#cocoRes = coco.loadRes("KD_coco_eval_caption_float32.json")
#cocoEval = COCOEvalCap(coco, cocoRes)
#cocoEval.evaluate()

#for metric, score in cocoEval.eval.items():
#	print(f"{metric}: {score}")


