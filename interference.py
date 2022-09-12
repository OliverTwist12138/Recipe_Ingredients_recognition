import torch
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
# Inference
results = model(img)
# Results
#
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'crop', 'display', 'files', 'imgs', 'n', 'names', 'pandas', 'pred', 'print', 'render', 's', 'save', 'show', 't', 'times', 'tolist', 'xywh', 'xywhn', 'xyxy', 'xyxyn']
a = results.pandas()  # or .show(), .save(), .crop(), .pandas(), etc.
print(a)