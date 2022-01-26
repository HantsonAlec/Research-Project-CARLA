import json
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from LSTR.config import system_configs
from LSTR.nnet.py_factory import NetworkFactory


class LSTRPredict:
    def __init__(self):
        self.load_configs()
        # Load model
        self.nnet = NetworkFactory()
        test_iter = system_configs.max_iter
        self.nnet.load_params(test_iter)
        self.nnet.cuda()
        self.nnet.eval_mode()

    def load_configs(self):
        print('loaded')
        with open('./LSTR/config/LSTR.json', "r") as f:
            self.configs = json.load(f)
        self.configs["system"]["snapshot_name"] = "LSTR"
        system_configs.update_config(self.configs["system"])
        # info
        self.input_size = self.configs['db']['input_size']

    def predict(self, image):
        self.height, self.width = image.shape[0:2]

        # Image
        images = np.zeros(
            (1, 3, self.input_size[0], self.input_size[1]), dtype=np.float32)
        pad_image = image.copy()
        resized_image = cv2.resize(
            pad_image, (self.input_size[1], self.input_size[0]))
        resized_image = resized_image / 255.
        resized_image = cv2.normalize(
            resized_image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        resized_image = resized_image.transpose(2, 0, 1)
        images[0] = resized_image
        images = torch.from_numpy(images).cuda(non_blocking=True)

        # Masks
        masks = np.ones(
            (1, 1, self.input_size[0], self.input_size[1]), dtype=np.float32)
        pad_mask = np.zeros((self.height, self.width, 1), dtype=np.float32)
        resized_mask = cv2.resize(
            pad_mask, (self.input_size[1], self.input_size[0]))
        masks[0][0] = resized_mask.squeeze()
        masks = torch.from_numpy(masks).cuda(non_blocking=True)
        # results
        outputs, weights = self.nnet.test([images, masks])
        return outputs

    def detect_lanes(self, image):
        # getting results from model
        outputs = self.predict(image)
        out_logits, out_curves = outputs['pred_logits'], outputs['pred_curves']
        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)
        labels[labels != 1] = 0
        results = torch.cat([labels.unsqueeze(-1).float(), out_curves], dim=-1)
        # making predictions
        pred = results[0].cpu().numpy()
        pred = pred[pred[:, 0].astype(int) == 1]
        lane_points = []
        for i, lane in enumerate(pred):
            lane = lane[1:]  # remove conf
            lower, upper = lane[0], lane[1]  # Get lower, upper positions
            lane = lane[2:]  # remove upper, lower positions

        # lane point generation
            ys = np.linspace(lower, upper, num=100)
            points = np.zeros((len(ys), 2), dtype=np.int32)
            points[:, 1] = (ys * self.height).astype(int)
            points[:, 0] = ((lane[0] / (ys - lane[1]) ** 2 + lane[2] / (ys - lane[1]) + lane[3] + lane[4] * ys -
                            lane[5]) * self.width).astype(int)
            points = points[(points[:, 0] > 0) & (points[:, 0] < self.width)]
            lane_points.append(points)

        return lane_points
