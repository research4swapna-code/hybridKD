from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

coco = COCO('/home/rcideais04/VLM_workspace/COCO_2017/train/annotations/captions_val2017.json')

cocoRes = coco.loadRes("KD_coco_eval_caption_sweep8.json")
cocoEval = COCOEvalCap(coco, cocoRes)
#cocoEval = COCOEvalCap(coco, cocoRes)

print("before evaluation")
cocoEval.evaluate()


for metric, score in cocoEval.eval.items():
	print(f"{metric}: {score}")


#############################################################################################################################

