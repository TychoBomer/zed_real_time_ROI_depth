import time
import torch
import cv2
import numpy as np
import torch.nn as nn
from typing import Tuple
from GroundingDINO.groundingdino.util.inference import Model
from sam2.build_sam import build_sam2_camera_predictor
import queue
import seaborn as sns
import os


from concurrent.futures import ThreadPoolExecutor
from deprecated import deprecated



def build_models(cfg) -> Tuple[nn.Module, Model]:
    """
    Builds and returns the SAM2 camera predictor and GroundingDINO models.

    Args:
        cfg: Configuration object that contains paths and parameters for the models.

    Returns:
        Tuple[nn.Module, Model]: SAM2 camera predictor and GroundingDINO model.
    """
    grounding_dino_model = Model(
        model_config_path=cfg.grounding_dino.config_path,
        model_checkpoint_path=cfg.grounding_dino.checkpoint_path
    )
    predictor = build_sam2_camera_predictor(cfg.sam2.model_cfg, cfg.sam2.checkpoint)

    return predictor, grounding_dino_model

def apply_nms(detections, nms_threshold: float):
    """
    Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.

    Args:
        detections: Detected objects with bounding boxes, confidence, and class IDs.
        nms_threshold (float): Threshold for NMS to filter out overlapping boxes.

    Returns:
        Detections after NMS filtering.
    """
    nms_idx = torch.ops.torchvision.nms(
        torch.from_numpy(detections.xyxy).float(),
        torch.from_numpy(detections.confidence).float(),
        nms_threshold
    ).numpy().tolist()

    if len(nms_idx) == 0:
        raise ValueError("No boxes left after NMS")
    print(f"After NMS: {len(detections.xyxy)} boxes")
    
    # Filter the detections
    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]
    
    return detections

 


@deprecated('Function has been replaced')
def process_masks_old(out_obj_ids, out_mask_logits, height: int, width: int) -> np.ndarray:
    """
    Process mask logits to generate final binary masks for object segmentation.

    Args:
        out_obj_ids: Object IDs from the mask logits.
        out_mask_logits: Mask logits for the segmented objects.
        height (int): Height of the input image.
        width (int): Width of the input image.

    Returns:
        np.ndarray: Combined mask of all objects.
    """
    all_mask = np.zeros((height, width, 1), dtype=np.uint8)

    def process_single_mask(mask_logit) -> np.ndarray:
        out_mask = (mask_logit > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
        return out_mask

    with ThreadPoolExecutor() as executor:
        mask_futures = [executor.submit(process_single_mask, mask_logit) for mask_logit in out_mask_logits]
        
        # Combine masks once processed
        for future in mask_futures:
            out_mask = future.result()
            all_mask = cv2.bitwise_or(all_mask, out_mask)

    return all_mask

def process_masks(out_obj_ids, out_mask_logits) -> dict:
    """
    Process mask logits to generate individual binary masks for each object.

    Args:
        out_obj_ids: Object IDs from the mask logits.
        out_mask_logits: Mask logits for the segmented objects.

    Returns:
        dict: A dictionary where keys are object IDs and values are binary masks.
    """
    out_masks = {}
    for obj_id, mask_logit in zip(out_obj_ids, out_mask_logits):
        out_masks[obj_id] = (mask_logit > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
    return out_masks






def update_caption(caption_queue: queue.Queue, user_caption: str):
    """
    Continuously update the object detection caption based on user input.

    Args:
        caption_queue (queue.Queue): Queue for holding dynamic captions.
        user_caption (str): Initial object caption to detect.
    """
    while True:
        new_caption = input("\t <-- Insert new object or enter 'tr' to toggle depth refinement.").strip()
        if new_caption:
            if new_caption.lower() == 'tr':
                caption_queue.put("tr")  
                print("Depth refinement toggle requested.")
                time.sleep(0.5)  # Prevent multiple rapid toggles
            else :
                caption_queue.put(new_caption)
                print(f"Updated detection target: {new_caption}")


import time

def fps_decorator(func):
    """
    Decorator to measure and print the FPS of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f"\rCurrent FPS: {fps:.2f}", end="")
        return result
    return wrapper

def create_mask_color_palette(n_colors:int=10, palette_type:str ='hsv') -> np.array:
    palette = sns.color_palette(palette_type, n_colors)
    return [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in palette]

def setup_output_folder(folder_name: str) -> str:
    #* utils.py nested two time so root is two above
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    if os.path.isabs(folder_name):
        output_folder = folder_name
    else:
        output_folder = os.path.join(project_root, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder
    

