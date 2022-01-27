import torch
from transformers import DetrConfig, DetrForObjectDetection
import pytorch_lightning as pl
import json
from roboflow import Roboflow
from transformers import DetrFeatureExtractor
import torchvision
import os
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from detr.datasets import get_coco_api_from_dataset
from detr.datasets.coco_eval import CocoEvaluator
from tqdm.notebook import tqdm

# DETR MUST BE CLONED TO THIS DIRECTORY -- git clone https://github.com/facebookresearch/detr.git


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder, "_annotations.coco.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        # preprocess image and target
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(
            images=img, annotations=target, return_tensors="pt")
        # remove batch dimension
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target


# DETR MODEL, turned into lightning module
class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay, train_dataloader, val_dataloader):
        super().__init__()
        id2label = self.load_labels()
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                            num_labels=len(
                                                                id2label),
                                                            ignore_mismatched_sizes=True)
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.train_dataloader, self.val_dataloader = train_dataloader, val_dataloader

    def load_labels(self):
        with open("../../detrCustom/labels.json", "r") as outfile:
            id2label = json.load(outfile)
        return id2label

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()}
                  for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values,
                             pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        # Logging
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())
        return loss

    def validation_step(self, batch, batch_idx):
        # Logging
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters(
            ) if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                      weight_decay=self.weight_decay)

        return optimizer

    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        return self.val_dataloader


def evaluate(model, val_dataset, val_dataloader, feature_extractor):
    # Get ground truths
    base_ds = get_coco_api_from_dataset(val_dataset)
    iou_types = ['bbox']
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # init device and model
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    # Loop evaluation
    for idx, batch in enumerate(tqdm(val_dataloader)):
        # get the inputs
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        # get labels from the batch
        labels = [{k: v.to(device) for k, v in t.items()}
                  for t in batch["labels"]]
        # predict
        outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        # Output to coco format
        orig_target_sizes = torch.stack(
            [target["orig_size"] for target in labels], dim=0)
        # convert outputs of model to COCO api
        results = feature_extractor.post_process(outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target,
               output in zip(labels, results)}
        coco_evaluator.update(res)
    # Run evaluator
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()


def load_data(feature_extractor):
    train_dataset = CocoDetection(
        img_folder='CARLA-17/train', feature_extractor=feature_extractor)
    val_dataset = CocoDetection(img_folder='CARLA-17/valid',
                                feature_extractor=feature_extractor, train=False)
    return train_dataset, val_dataset


def create_batch(batch, feature_extractor):
    pixel_values = [item[0] for item in batch]
    encoding = feature_extractor.pad_and_create_pixel_mask(
        pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch


def main():
    # Download dataset in coco format - or download from kaggle (link)
    rf = Roboflow(api_key="ruQpTRXWHhPyIXpWAIh5")
    project = rf.workspace().project("carla-izloa")
    dataset = project.version(17).download("coco")

    # load feature extractor
    feature_extractor = DetrFeatureExtractor.from_pretrained(
        "facebook/detr-resnet-50")

    # load data
    train_dataset, val_dataset = load_data(feature_extractor)

    # Get labels
    cats = train_dataset.coco.cats
    id2label = {k: v['name'] for k, v in cats.items()}

    # safe to use in predictions
    with open("../../detrCustom/labels.json", "w") as outfile:
        json.dump(id2label, outfile)

    # Create batch based on dataloaders - resizes images
    train_dataloader = DataLoader(
        train_dataset, collate_fn=create_batch, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, collate_fn=create_batch, batch_size=2)
    batch = next(iter(train_dataloader))

    # create model
    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4,
                 train_dataloader=train_dataloader, val_dataloader=val_dataloader)
    # init outputs
    outputs = model(
        pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
    # start training
    trainer = Trainer(gpus=1, max_epochs=100, gradient_clip_val=0.1)
    trainer.fit(model)
    # Logs and model checkpoint can be found on lightning logs folder
    evaluate(model, val_dataset, val_dataloader, feature_extractor)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
