### REQUIRES REPVIT SAM AND RF GROUNDING DINO. 
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from repvit_sam import sam_model_registry, SamPredictor
from torchvision.ops import box_convert
from utils import image_loader
class GroundingSAM:
    def __init__(self, grounding_config="GroundingDINO_SwinT_OGC.py", grounding_path="groundingdino_swint_ogc.pth", repvit_path="repvit_sam.pt", device="cuda:0"):
        self.grounding_model = load_model(grounding_config, grounding_path)## LOADS GROUNDING MODEL
        self.sam = sam_model_registry["repvit"](checkpoint=repvit_path).to(device).eval()## LOADS SEGMENT ANYTHING MODEL
        self.predictor = SamPredictor(sam)
    
    def show_mask(self, mask, ax, random_color=False):
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
      
    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
      
    def segment(self, boxes):
        input_box = np.array(boxes)
        masks, _, _ = self.predictor.predict(point_coords=None,point_labels=None, box=input_box[None, :],multimask_output=False)
        return masks
      
    def grounding(self, text, image):
        image_source, image = load_image(image)

        boxes, logits, phrases = predict(model=self.grounding_model,image=image,caption=text,box_threshold=0.35,text_threshold=0.35)
        return boxes, logits, phrases, image_source
    def sam_dino(self, text, image, binary=False):
        boxes, logits, phrases, image_source = grounding(text, image)
        masks = []
        cimage = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        source_h, source_w, _ = cimage.shape
        self.predictor.set_image(cimage)
        boxes3 = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = box_convert(boxes=boxes3, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        for box in xyxy:
            box = np.array(box)
            mask = segment(box)### MIGHT HAVE TO CHANGE!!!!
            masks.append(mask)
        if binary == True:
            binary_mask = masks[0].astype(np.uint8)*255
            return binary_mask, xyxy, logits, phrases, image_source
        else:
            return masks, xyxy, logits, phrases, image_source
