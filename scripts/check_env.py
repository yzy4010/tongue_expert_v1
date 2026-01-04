# 环境测试脚本
import numpy as np
import cv2
import albumentations as A
import torch
import matplotlib

print("NumPy:", np.__version__)
print("OpenCV:", cv2.__version__)
print("Albumentations:", A.__version__)
print("Torch:", torch.__version__)
print("Matplotlib:", matplotlib.__version__)
print("CUDA:", torch.cuda.is_available())

# NumPy: 1.24.4
# OpenCV: 4.8.1
# Albumentations: 1.3.1
# Torch: 2.0.1+cpu
# Matplotlib: 3.7.5
# CUDA: False