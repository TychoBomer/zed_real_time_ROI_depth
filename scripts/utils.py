import time
import torch
import cv2
import numpy as np
import torch.nn as nn
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor
from GroundingDINO.groundingdino.util.inference import Model
from sam2.build_sam import build_sam2_camera_predictor
import queue

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

def process_masks(out_obj_ids, out_mask_logits, height: int, width: int) -> np.ndarray:
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

def mask_guided_filter(depth_map: np.ndarray, guidance_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply guided filtering to the depth map only within the masked region.

    Args:
        depth_map (np.ndarray): Input depth map.
        guidance_img (np.ndarray): Guidance image (can be the mask or an RGB image).
        mask (np.ndarray): Binary mask for the object of interest.

    Returns:
        np.ndarray: Refined depth map with filtering applied within the mask.
    """
    # Ensure depth map is uint8
    depth_map = depth_map.astype(np.uint8)
    
    # Handle zero values in the depth map (treat zero as invalid depth)
    zero_mask = (depth_map == 0)  # Find zero values in the depth map
    depth_map_filled = np.copy(depth_map)

    # Inpaint zero values but only within the mask
    if np.any(zero_mask):
        depth_map_filled[zero_mask & (mask > 0)] = 0  # Set zeros inside mask to zero for inpainting
        inpaint_radius = 5
        depth_map_filled = cv2.inpaint(depth_map_filled.astype(np.uint8), 
                                       (zero_mask & (mask > 0)).astype(np.uint8), 
                                       inpaint_radius, cv2.INPAINT_TELEA)

    # Normalize the mask and guidance image
    mask = mask.astype(np.float32) / 255.0  # Normalize mask to [0, 1] range
    guidance_img = guidance_img.astype(np.float32) / 255.0

    # Apply the guided filter only within the mask
    filtered_region = np.copy(depth_map_filled)
    filtered_region[mask == 0] = 0  # Set pixels outside the mask to zero

    # Apply guided filter locally to the region inside the mask
    refined_region = cv2.ximgproc.guidedFilter(guide=guidance_img, src=filtered_region, radius=6, eps=1e-4)

    # Return the refined depth map
    refined_depth = np.where(mask > 0, refined_region, depth_map)
    
    return refined_depth

def preprocess_depth_map(depth_map, kernel_size=5):
    # Fill small gaps by median filtering
    filled_depth_map = cv2.medianBlur(depth_map, kernel_size)
    return filled_depth_map
def inpaint_depth_map(depth_map, inpaint_radius=3):
    # Create a mask where depth is zero or NaN
    mask = (depth_map == 0).astype(np.uint8)
    # Use inpainting to fill zero or NaN values
    inpainted_depth_map = cv2.inpaint(depth_map, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return inpainted_depth_map

def refine_depth_with_postprocessing(depth_map, left_image, mask, lmbda=8000, sigma=1.5, inpaint_radius=3):
    """
    Refine depth map by combining inpainting, WLS filtering, and mask-guided blending.

    Parameters:
        depth_map (np.ndarray): Input depth map.
        left_image (np.ndarray): Left RGB image from the stereo pair (used as guidance).
        mask (np.ndarray): SAM2 segmentation mask (binary mask).
        lmbda (float): Regularization parameter for WLS filter.
        sigma (float): Smoothness parameter for WLS filter.
        inpaint_radius (int): Radius for inpainting gaps in the depth map.

    Returns:
        np.ndarray: Refined depth map.
    """
    # Step 1: Inpaint large missing regions
    depth_map_inpainted = inpaint_depth_map(depth_map, inpaint_radius)
    
    # Step 2: Convert to 16-bit for WLS filtering
    depth_map_inpainted = (depth_map_inpainted * 16).astype(np.int16)
    
    # Step 3: Apply WLS filtering for edge-preserving refinement
    wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    refined_depth = wls_filter.filter(depth_map_inpainted, left_image)
    refined_depth = refined_depth.astype(np.float32) / 16.0  # Normalize back

    # Step 4: Weighted blending to enhance masked areas
    mask = mask.astype(np.float32) / 255.0
    final_depth_map = depth_map * (1 - mask) + refined_depth * mask

    return final_depth_map
1   


def update_caption(caption_queue: queue.Queue, user_caption: str):
    """
    Continuously update the object detection caption based on user input.

    Args:
        caption_queue (queue.Queue): Queue for holding dynamic captions.
        user_caption (str): Initial object caption to detect.
    """
    while True:
        new_caption = input("\nEnter new object to detect (e.g., 'bottle'): ")
        caption_queue.put(new_caption)

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