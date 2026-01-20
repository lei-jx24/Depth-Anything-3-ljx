import cv2
import numpy as np
import os
import torch
from depth_anything_3.api import DepthAnything3
import tempfile  # 用于创建临时目录

# -------------------------- 1. 配置参数 --------------------------
VIDEO_PATH = "/home/user/ljx/FileStore/raw_video/soccer.mp4"
OUTPUT_DIR = "./assets/videos/video_output"
TEMP_FRAME_DIR = "/home/user/ljx/FileStore/preprocessed_video/soccer/"  # 临时帧保存目录
FPS = 5
MODEL_NAME = "depth-anything/DA3NESTED-GIANT-LARGE"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 创建目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_FRAME_DIR, exist_ok=True)

# -------------------------- 2. 视频解码并保存为临时图像文件 --------------------------
def decode_video_to_temp_files(video_path, temp_dir, target_fps=15):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件：{video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(original_fps / target_fps))
    
    frame_paths = []  # 保存临时文件路径列表
    frame_idx = 0
    temp_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip == 0:
            # 保存为临时JPG文件
            frame_path = os.path.join(temp_dir, f"temp_frame_{temp_idx:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            temp_idx += 1
        
        frame_idx += 1
    
    cap.release()
    return frame_paths  # 返回文件路径列表

# 执行解码并保存临时文件
frame_paths = decode_video_to_temp_files(VIDEO_PATH, TEMP_FRAME_DIR, target_fps=FPS)
print(f"视频解码完成，共抽取 {len(frame_paths)} 帧，临时文件保存在：{TEMP_FRAME_DIR}")
