import cv2
import numpy as np
import os
import torch
import glob
from depth_anything_3.api import DepthAnything3

# -------------------------- 1. 配置参数 --------------------------
OUTPUT_DIR = "/home/user/ljx/FileStore/DA3_MegaSaM_DYNAPO/DA3_output/temple_2"
image_dir = "/home/user/ljx/FileStore/Sintel/temple_2/rgb"  # input帧目录
MODEL_NAME = "depth-anything/DA3NESTED-GIANT-LARGE"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 创建目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------- 2. Get all png files and sort them alphabetically/numerically --------------------------
frame_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
print(f"Total frames found: {len(frame_paths)}")
if len(frame_paths) == 0:
    print("Error: No images found! Check your directory path and file extensions.")

# -------------------------- 3. 初始化模型 --------------------------
model = DepthAnything3.from_pretrained(MODEL_NAME)
model = model.to(DEVICE)
print("模型初始化完成")

# -------------------------- 4. 模型推理（传入文件路径列表，支持导出） --------------------------
prediction = model.inference(
    image=frame_paths,  # 传入文件路径列表（核心修改）
    export_dir=OUTPUT_DIR,
    export_format="npz"  # 支持 mini_npz/npz/glb 等格式
)


print(f"结果保存完成，输出目录：{OUTPUT_DIR}")