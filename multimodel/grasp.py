import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import cv2
from inference.post_process import post_process_output
from utils.dataset_processing.grasp import detect_grasps
from utils.visualisation.plot import plot_grasp
from utils.data.camera_data import CameraData
class GraspAnything:
    def __init__(self, model_path, device="cuda:0"):
        self.grasp_model = torch.load(model_path).to(device).eval()
        self.device = device
    
    def grasp(self, image, crop=(256, 256):
        # Define behavior for method1
        rgb = cv2.imread(image)
        rgb = cv2.resize(rgb, crop)
        height, width = rgb.shape[:2]
        cam_data = CameraData(width=width, height=height, include_depth=False, include_rgb=True, output_size=max(crop))
        x, depth, rgb_image, = cam_data.get_data(rgb=rgb, depth=None)
        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.grasp_model.predict(xc)
        fig = plt.figure(figsize=(10, 10))
        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
        grasps = detect_grasps(q_img, ang_img, width_img, no_grasps=10)
        plot_grasp(fig=fig, rgb_img=cam_data.get_rgb(rgb, False), grasps=grasps, save=False)
