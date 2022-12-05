```python
from google.colab import drive 
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).



```python
!python -m pip install pyyaml==5.1
import sys, os, distutils.core


!git clone 'https://github.com/facebookresearch/detectron2'
dist = distutils.core.run_setup("./detectron2/setup.py")
!python -m pip install {' '.join([f"'{x}'" for x in dist.install_requires])}
sys.path.insert(0, os.path.abspath('./detectron2'))
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: pyyaml==5.1 in /usr/local/lib/python3.8/dist-packages (5.1)
    fatal: destination path 'detectron2' already exists and is not an empty directory.
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: Pillow>=7.1 in /usr/local/lib/python3.8/dist-packages (7.1.2)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.8/dist-packages (3.2.2)
    Requirement already satisfied: pycocotools>=2.0.2 in /usr/local/lib/python3.8/dist-packages (2.0.6)
    Requirement already satisfied: termcolor>=1.1 in /usr/local/lib/python3.8/dist-packages (2.1.1)
    Requirement already satisfied: yacs>=0.1.8 in /usr/local/lib/python3.8/dist-packages (0.1.8)
    Requirement already satisfied: tabulate in /usr/local/lib/python3.8/dist-packages (0.8.10)
    Requirement already satisfied: cloudpickle in /usr/local/lib/python3.8/dist-packages (1.5.0)
    Requirement already satisfied: tqdm>4.29.0 in /usr/local/lib/python3.8/dist-packages (4.64.1)
    Requirement already satisfied: tensorboard in /usr/local/lib/python3.8/dist-packages (2.9.1)
    Requirement already satisfied: fvcore<0.1.6,>=0.1.5 in /usr/local/lib/python3.8/dist-packages (0.1.5.post20221122)
    Requirement already satisfied: iopath<0.1.10,>=0.1.7 in /usr/local/lib/python3.8/dist-packages (0.1.9)
    Requirement already satisfied: future in /usr/local/lib/python3.8/dist-packages (0.16.0)
    Requirement already satisfied: pydot in /usr/local/lib/python3.8/dist-packages (1.3.0)
    Requirement already satisfied: omegaconf>=2.1 in /usr/local/lib/python3.8/dist-packages (2.2.3)
    Requirement already satisfied: hydra-core>=1.1 in /usr/local/lib/python3.8/dist-packages (1.2.0)
    Requirement already satisfied: black==22.3.0 in /usr/local/lib/python3.8/dist-packages (22.3.0)
    Requirement already satisfied: timm in /usr/local/lib/python3.8/dist-packages (0.6.12)
    Requirement already satisfied: fairscale in /usr/local/lib/python3.8/dist-packages (0.4.12)
    Requirement already satisfied: packaging in /usr/local/lib/python3.8/dist-packages (21.3)
    Requirement already satisfied: typing-extensions>=3.10.0.0 in /usr/local/lib/python3.8/dist-packages (from black==22.3.0) (4.1.1)
    Requirement already satisfied: platformdirs>=2 in /usr/local/lib/python3.8/dist-packages (from black==22.3.0) (2.5.4)
    Requirement already satisfied: click>=8.0.0 in /usr/local/lib/python3.8/dist-packages (from black==22.3.0) (8.1.3)
    Requirement already satisfied: mypy-extensions>=0.4.3 in /usr/local/lib/python3.8/dist-packages (from black==22.3.0) (0.4.3)
    Requirement already satisfied: tomli>=1.1.0 in /usr/local/lib/python3.8/dist-packages (from black==22.3.0) (2.0.1)
    Requirement already satisfied: pathspec>=0.9.0 in /usr/local/lib/python3.8/dist-packages (from black==22.3.0) (0.10.2)
    Requirement already satisfied: numpy in /usr/local/lib/python3.8/dist-packages (from pycocotools>=2.0.2) (1.23.5)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.8/dist-packages (from matplotlib) (3.0.9)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.8/dist-packages (from matplotlib) (0.11.0)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.8/dist-packages (from matplotlib) (2.8.2)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.8/dist-packages (from matplotlib) (1.4.4)
    Requirement already satisfied: PyYAML in /usr/local/lib/python3.8/dist-packages (from yacs>=0.1.8) (5.1)
    Requirement already satisfied: portalocker in /usr/local/lib/python3.8/dist-packages (from iopath<0.1.10,>=0.1.7) (2.6.0)
    Requirement already satisfied: antlr4-python3-runtime==4.9.* in /usr/local/lib/python3.8/dist-packages (from omegaconf>=2.1) (4.9.3)
    Requirement already satisfied: importlib-resources in /usr/local/lib/python3.8/dist-packages (from hydra-core>=1.1) (5.10.0)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.8/dist-packages (from python-dateutil>=2.1->matplotlib) (1.15.0)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.8/dist-packages (from tensorboard) (1.8.1)
    Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.8/dist-packages (from tensorboard) (2.23.0)
    Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.8/dist-packages (from tensorboard) (2.14.1)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.8/dist-packages (from tensorboard) (0.4.6)
    Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.8/dist-packages (from tensorboard) (57.4.0)
    Requirement already satisfied: absl-py>=0.4 in /usr/local/lib/python3.8/dist-packages (from tensorboard) (1.3.0)
    Requirement already satisfied: protobuf<3.20,>=3.9.2 in /usr/local/lib/python3.8/dist-packages (from tensorboard) (3.19.6)
    Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /usr/local/lib/python3.8/dist-packages (from tensorboard) (0.6.1)
    Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.8/dist-packages (from tensorboard) (1.0.1)
    Requirement already satisfied: grpcio>=1.24.3 in /usr/local/lib/python3.8/dist-packages (from tensorboard) (1.50.0)
    Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.8/dist-packages (from tensorboard) (3.4.1)
    Requirement already satisfied: wheel>=0.26 in /usr/local/lib/python3.8/dist-packages (from tensorboard) (0.38.4)
    Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.8/dist-packages (from google-auth<3,>=1.6.3->tensorboard) (4.9)
    Requirement already satisfied: cachetools<6.0,>=2.0.0 in /usr/local/lib/python3.8/dist-packages (from google-auth<3,>=1.6.3->tensorboard) (5.2.0)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.8/dist-packages (from google-auth<3,>=1.6.3->tensorboard) (0.2.8)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.8/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard) (1.3.1)
    Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/lib/python3.8/dist-packages (from markdown>=2.6.8->tensorboard) (4.13.0)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.8/dist-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard) (3.10.0)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.8/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard) (0.4.8)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.8/dist-packages (from requests<3,>=2.21.0->tensorboard) (1.24.3)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.8/dist-packages (from requests<3,>=2.21.0->tensorboard) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.8/dist-packages (from requests<3,>=2.21.0->tensorboard) (2022.9.24)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.8/dist-packages (from requests<3,>=2.21.0->tensorboard) (2.10)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.8/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard) (3.2.2)
    Requirement already satisfied: huggingface-hub in /usr/local/lib/python3.8/dist-packages (from timm) (0.11.1)
    Requirement already satisfied: torchvision in /usr/local/lib/python3.8/dist-packages (from timm) (0.13.1+cu113)
    Requirement already satisfied: torch>=1.7 in /usr/local/lib/python3.8/dist-packages (from timm) (1.12.1+cu113)
    Requirement already satisfied: filelock in /usr/local/lib/python3.8/dist-packages (from huggingface-hub->timm) (3.8.0)



```python
import torch, detectron2
!nvcc --version
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)
```

    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2021 NVIDIA Corporation
    Built on Sun_Feb_14_21:12:58_PST_2021
    Cuda compilation tools, release 11.2, V11.2.152
    Build cuda_11.2.r11.2/compiler.29618528_0
    torch:  1.12 ; cuda:  cu113
    detectron2: 0.6



```python
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
```


```python
cfg = get_cfg()
model = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(model))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
predictor = DefaultPredictor(cfg)
```


```python
def get_mappings():
  mappings = {}
  mappings[1] = 59
  mappings[7] = 57
  mappings[8] = 62
  mappings[10] = 75
  return mappings
```


```python
import json

with open('/content/drive/MyDrive/Segmentation/ann/annotations/instances_default.json', 'r') as f:
    data = f.read()

ann = json.loads(data)
categories = ann["categories"]
images = ann["images"]
annotations = ann["annotations"]

cat_images_data = {}
img_data = {}

for img in images:
  img_data[img["id"]] = img["file_name"]

for ann_data in annotations:
  cat_id = ann_data["category_id"]
  img_id = ann_data["image_id"]
  if cat_id not in cat_images_data:
    cat_images_data[cat_id] = []
  cat_images_data[cat_id].append(img_data[img_id])

print(categories)
```

    [{'id': 1, 'name': 'Bed', 'supercategory': ''}, {'id': 2, 'name': 'Cabinet', 'supercategory': ''}, {'id': 3, 'name': 'Carpets', 'supercategory': ''}, {'id': 4, 'name': 'Floor', 'supercategory': ''}, {'id': 5, 'name': 'Generator', 'supercategory': ''}, {'id': 6, 'name': 'Room heaters', 'supercategory': ''}, {'id': 7, 'name': 'Sofa', 'supercategory': ''}, {'id': 8, 'name': 'Television', 'supercategory': ''}, {'id': 9, 'name': 'Treadmill', 'supercategory': ''}, {'id': 10, 'name': 'Vase', 'supercategory': ''}]



```python
# for cat in [10]:
#   count = 0
#   print('Cat: ', cat)
#   for img_file_name in cat_images_data[cat]:
#     print(img_file_name)
#     im = cv2.imread("/content/drive/MyDrive/Segmentation/ann/images/" + str(img_file_name))
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2_imshow(out.get_image()[:, :, ::-1])
#     print(outputs["instances"].pred_classes)
#     print(outputs["instances"].pred_boxes)
#     count += 1
#     if count > 10:
#       break
```


```python
correct = 0
total = 0
for cat in cat_images_data:
  print('Running for category: ', cat)
  for img_file_name in cat_images_data[cat]:
    im = cv2.imread("/content/drive/MyDrive/Segmentation/ann/images/" + str(img_file_name))
    outputs = predictor(im)
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2_imshow(out.get_image()[:, :, ::-1])
    if cat in get_mappings() and get_mappings()[cat] in outputs["instances"].pred_classes:
      correct += 1
    total += 1
```

    Running for category:  6
    Running for category:  10
    Running for category:  4
    Running for category:  7
    Running for category:  3
    Running for category:  9
    Running for category:  8
    Running for category:  2
    Running for category:  1
    Running for category:  5



```python
print('Accuracy: ', (correct*100)/total)
```

    Accuracy:  35.476190476190474



```python
# Visual output

DIR_PATH = '/content/drive/MyDrive/Segmentation/ann/images/'
directory = os.fsencode(DIR_PATH)

selected_files = ['victoria-black-vase.jpg', 'viljestark-vase-clear-glass__0640433_pe699813_s5.jpg', 'vinliden-sofa-hakebo-beige__0852744_pe780233_s5.jpg', 'vintage-tv_23-2147503075.jpg']
cnt = 0
for file in selected_files:
  file_name = os.fsdecode(file)
  print(file_name)
  im = cv2.imread(DIR_PATH + file_name)
  outputs = predictor(im)
  v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  cv2_imshow(out.get_image()[:, :, ::-1])
  if cnt > 10:
    break;
  cnt += 1
```

    victoria-black-vase.jpg



    
![png](output_10_1.png)
    


    viljestark-vase-clear-glass__0640433_pe699813_s5.jpg



    
![png](output_10_3.png)
    


    vinliden-sofa-hakebo-beige__0852744_pe780233_s5.jpg



    
![png](output_10_5.png)
    


    vintage-tv_23-2147503075.jpg



    
![png](output_10_7.png)
    



```python

```
