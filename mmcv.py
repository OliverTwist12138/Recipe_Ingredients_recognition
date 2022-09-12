# -*- coding: utf-8 -*-
"""MMCV.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VVz-ZWpMpo_ZR08QUSwT2cKelda2shzX
"""

# Check nvcc version
!nvcc -V
# Check GCC version
!gcc --version

# Commented out IPython magic to ensure Python compatibility.
# install dependencies: (use cu111 because colab has CUDA 11.1)
!pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# install mmcv-full thus we could use CUDA operators
!pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# Install mmdetection
!rm -rf mmdetection
!git clone https://github.com/open-mmlab/mmdetection.git
# %cd mmdetection

!pip install -e .

pip install --upgrade mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA}/torch${PYTORCH}/index.html

# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

!mkdir checkpoints
!wget -c https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth \
      -O checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth

import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector

# Choose to use a config and initialize the detector
config = 'configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py'
# Setup a checkpoint file to load
checkpoint = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# Set the device to be used for evaluation
device='cuda:0'

# Load the config
config = mmcv.Config.fromfile(config)
# Set pretrained to be None since we do not need pretrained model here
config.model.pretrained = None

# Initialize the detector
model = build_detector(config.model)

# Load checkpoint
checkpoint = load_checkpoint(model, checkpoint, map_location=device)

# Set the classes of models for inference
model.CLASSES = checkpoint['meta']['CLASSES']

# We need to set the model's cfg for inference
model.cfg = config

# Convert the model to GPU
model.to(device)
# Convert the model into evaluation mode
model.eval()

# Use the detector to do inference
img = 'demo/demo.jpg'
result = inference_detector(model, img)
show_result_pyplot(model, img, result, score_thr=0.3)

# Let's plot the result
show_result_pyplot(model, img, result, score_thr=0.3)

from google.colab import drive
drive.mount('/content/drive')
path = '/content/drive/My Drive/images/'

img_path = path + 'pineapple.jpg'
result = inference_detector(model, img_path)

import numpy as np
from PIL import Image
img = mmcv.imread(img_path)
img = img.copy()
if isinstance(result, tuple):
    bbox_result, segm_result = result
    if isinstance(segm_result, tuple):
        segm_result = segm_result[0]  # ms rcnn
else:
    bbox_result, segm_result = result, None
bboxes = np.vstack(bbox_result)
labels = [
    np.full(bbox.shape[0], i, dtype=np.int32)
    for i, bbox in enumerate(bbox_result)
]
labels = np.concatenate(labels)
# draw segmentation masks
segms = None
if segm_result is not None and len(labels) > 0:  # non empty
    segms = mmcv.concat_list(segm_result)
    if isinstance(segms[0], torch.Tensor):
        segms = torch.stack(segms, dim=0).detach().cpu().numpy()
    else:
        segms = np.stack(segms, axis=0)
score_thr = 0.3
if score_thr > 0:
    assert bboxes is not None and bboxes.shape[1] == 5
    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]
    labels = labels[inds]
    if segms is not None:
        segms = segms[inds, ...]
segms = np.where(segms == 1, 255, segms)

image = Image.open(img_path)
b = np.array(b)
b = np.transpose(b,(2,0,1))
for j in range(len(segms)):
  a = segms[j]
  for i in range(3):
    b[i] = b[i] * a / 255
  b = np.transpose(b,(1,2,0))
  black_pixels = np.where(
      (b[:, :, 0] == 0) & 
      (b[:, :, 1] == 0) & 
      (b[:, :, 2] == 0)
  )
  # set those pixels to white
  b[black_pixels] = [255, 255, 255]
  new_image = Image.fromarray(b)
  new_image.save('/content/drive/My Drive/images/result{}.jpg'.format(j))