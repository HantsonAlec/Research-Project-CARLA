import detrCustom
import pytorch_lightning as pl
from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
import json
import random


class DETR_CUSTOM:
    def __init__(self, model_path, confidence):
        self.model_path = model_path
        self.confidence = confidence
        self.feature_extractor = DetrFeatureExtractor.from_pretrained(
            "facebook/detr-resnet-50")
        # model
        self.model = DetrLightning(
            lr=1e-5, lr_backbone=1e-4, weight_decay=1e-5)
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.device = torch.device("cuda")
        self.model.to(self.device)
        self.model.eval()

        # Attributes
        self.id2label = self.load_labels()
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.id2label.values()]

    def load_labels(self):
        with open("detrCustom/labels.json", "r") as outfile:
            id2label = json.load(outfile)
        return id2label

    def get_attributes(self):
        return self.id2label, self.colors

    def predict_outputs(self, image):
        encoding = self.feature_extractor(image, return_tensors="pt")
        # remove batch dimension
        pixel_values = encoding["pixel_values"].squeeze()
        pixel_values = pixel_values.unsqueeze(0).to(self.device)
        outputs = self.model(pixel_values=pixel_values, pixel_mask=None)
        return outputs

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        #img_w, img_h = size
        img_h, img_w, dim = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def detect_objects(self, image, outputs):
        # keep only predictions with confidence >= threshold
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > self.confidence

        # convert predicted boxes from [0; 1] to image scales
        bboxes_scaled = self.rescale_bboxes(
            outputs.pred_boxes[0, keep].cpu(), image.shape)

        return probas[keep], bboxes_scaled


class DetrLightning(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        id2label = self.load_labels()
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                            num_labels=len(
                                                                id2label),
                                                            ignore_mismatched_sizes=True)
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def load_labels(self):
        with open("detrCustom/labels.json", "r") as outfile:
            id2label = json.load(outfile)
        return id2label

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs
