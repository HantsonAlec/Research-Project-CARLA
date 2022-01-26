import torch
import torchvision.transforms as T 
import random, sys

class DETR:
    def __init__(self, confidence):
        sys.modules.pop('models')
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
        sys.modules.pop('models')
        self.model.eval()
        self.model = self.model.cuda()
        self.classes = ['N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
                            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
                            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
                            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
                            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
                            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                            'toothbrush'
                        ]
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.classes]
        self.confidence=confidence

    def get_attributes(self):
        return self.classes, self.colors

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b


    def detect_objects(self, image):
        # mean-std normalize the input image (batch-size: 1)
        img = self.transform(image).unsqueeze(0)
        img = img.cuda()
        #Get output
        outputs = self.model(img)

        # keep only predictions with confidence >= threshold
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > self.confidence

        # convert predicted boxes from [0; 1] to image scales
        bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'][0, keep].cpu(), image.size)
        return probas[keep], bboxes_scaled