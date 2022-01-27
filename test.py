from transformers.models.detr.feature_extraction_detr import rgb_to_id, id_to_rgb
from copy import deepcopy
from SegFormer.SegFormer import SegFormer
from PIL import Image
import numpy, torch, io
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import time
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

#torch.cuda.empty_cache()
segmentator= SegFormer()
im=cv2.imread('Town02_002640_png.rf.ac6bee6a9e4c4b2a1e149670c97196c1.jpg')
im = im[:, :, ::-1]
image=Image.fromarray(numpy.uint8(im))
start = time.time()
seg_mask=segmentator.panoptic_detection(image)
end = time.time()
print(end - start)
# Show image + mask
img = np.array(image) * 0.5 + seg_mask * 0.5
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(seg_mask)
plt.show()