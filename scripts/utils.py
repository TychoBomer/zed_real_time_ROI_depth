import time
import torch
import cv2
import numpy as np

import time




# Function for applying Non-Maximum Suppression for dino
def apply_nms(detections, nms_threshold):
    nms_idx = torch.ops.torchvision.nms(
        torch.from_numpy(detections.xyxy).float(),
        torch.from_numpy(detections.confidence).float(),
        nms_threshold
    ).numpy().tolist()

    if len(nms_idx) == 0:
        raise ValueError("No boxes left after NMS")
    print(f"After NMS: {len(detections.xyxy)} boxes")
    

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]
    
    return detections


# Mask post-processing helper
def process_masks(out_obj_ids, out_mask_logits, height, width):
    all_mask = np.zeros((height, width, 1), dtype=np.uint8)
    for i in range(len(out_obj_ids)):
        out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
        all_mask = cv2.bitwise_or(all_mask, out_mask)
    return all_mask
