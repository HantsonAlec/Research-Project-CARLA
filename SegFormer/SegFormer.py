from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch
from torch import nn
import numpy as np


class SegFormer:
    def __init__(self, confidence=0.85):
        torch.cuda.empty_cache()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        self.model_name = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(
            self.model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.model_name)
        self.model.to(self.device)
        self.confidence = confidence

    def predict(self, image):
        # Making prediction
        pixel_values = self.feature_extractor(
            image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values)
        logits = outputs.logits
        # reshape
        logits = nn.functional.interpolate(outputs.logits.detach().cpu(),
                                           size=image.size[::-1],
                                           mode='bilinear',
                                           align_corners=False)
        pred = logits.argmax(dim=1)[0]
        return pred

    def panoptic_detection(self, image):
        # predict
        seg = self.predict(image)
        seg_mask = np.zeros(
            (seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # creating mask
        palette = np.array(self.color_palette())
        for label, color in enumerate(palette):
            seg_mask[seg == label, :] = color
        # Convert to BGR
        seg_mask = seg_mask[..., ::-1]
        return seg_mask

    def color_palette(self):
        return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                [143, 255, 140], [204, 255, 4], [255, 51, 7]]
