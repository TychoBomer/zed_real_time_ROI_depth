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

from deprecated import deprecated

import numpy as np

import numpy as np
import cv2

def compute_slope_compensation(depth_map, mask):
    """
    Apply slope depth compensation to handle missing or noisy depth values.

    Args:
        depth_map (np.ndarray): Input depth map (H x W).
        mask (np.ndarray): Binary mask for the region of interest (H x W).

    Returns:
        np.ndarray: Depth map after slope depth compensation.
    """
    # Compute depth gradients
    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
    slope = np.sqrt(grad_x**2 + grad_y**2)

    # Limit slope values to prevent extreme changes
    slope = np.clip(slope, 0, np.percentile(slope[mask > 0], 95))

    # Fill missing depth values based on slope
    filled_depth = depth_map.copy()
    missing = (depth_map == 0) & (mask > 0)
    filled_depth[missing] = slope[missing]

    return filled_depth

def refine_depth_with_wjbf_and_sdcf(depth_map, obj_masks, guidance_img, sigma_spatial=15, sigma_range=30):
    """
    Refine the depth map using Weighted Joint Bilateral Filter and Slope Depth Compensation.

    Args:
        depth_map (np.ndarray): Raw depth map (H x W).
        obj_masks (dict): Dictionary of binary masks for each object.
        guidance_img (np.ndarray): Guidance image for the joint bilateral filter.
        sigma_spatial (float): Spatial sigma for joint bilateral filter.
        sigma_range (float): Range sigma for joint bilateral filter.

    Returns:
        np.ndarray: Refined depth map.
    """
    refined_depth_map = depth_map.copy()

    # Normalize depth map and guidance image
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    guidance_img_normalized = cv2.normalize(guidance_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Process each segmentation mask
    for obj_id, mask in obj_masks.items():
        # Ensure the mask is binary
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        obj_mask = mask.astype(np.uint8)

        # Apply slope depth compensation
        compensated_depth = compute_slope_compensation(depth_map_normalized, obj_mask)

        # Weighted Joint Bilateral Filter
        if np.any(obj_mask > 0):  # Only process if the mask has valid regions
            joint_bilateral_filtered = cv2.ximgproc.jointBilateralFilter(
                joint=guidance_img_normalized,  # Guidance image
                src=compensated_depth,         # Source depth map
                d=-1,                          # Automatically determine kernel size
                sigmaColor=sigma_range,        # Range sigma for joint bilateral filter
                sigmaSpace=sigma_spatial       # Spatial sigma for joint bilateral filter
            )

            # Merge the refined region back into the depth map
            refined_depth_map[obj_mask > 0] = joint_bilateral_filtered[obj_mask > 0]

    # Convert back to original depth map scale
    refined_depth_map = cv2.normalize(refined_depth_map, None, depth_map.min(), depth_map.max(), cv2.NORM_MINMAX)

    return refined_depth_map





class Sam2PromptType:
    valid_prompt_types = {"g_dino_bbox", "bbox", "point", "mask"} # all types SAM2 could handle

    def __init__(self, prompt_type,user_caption=None, **kwargs) -> None:
        self._prompt_type = None
        self.user_caption = user_caption
        self.prompt_type = prompt_type # attempts the set function @prompt_type.setter
        self.params = kwargs
        self.validate_prompt()

    def validate_prompt(self)->None:
        if self.prompt_type == "g_dino_bbox":
            if not self.user_caption or not isinstance(self.user_caption, str):
                raise ValueError("For 'g_dino_bbox' prompt, 'user_caption' must be provided as a non-empty string.")
            
            # prepare uswer caption to use dots as separators (advised to use in GroundinDINO)
            self.user_caption = self.format_user_caption(self.user_caption)


        elif self.prompt_type == "point":
            if "point_coords" not in self.params or not isinstance(self.params["point_coords"], (list,tuple)):
                raise ValueError("For sam2 prompt 'point' prompt, 'point_coords' must be provided as a list or tuple of (x, y) coordinates.")            
            point_coords = np.array(self.params["point_coords"])
            try:
                if point_coords.ndim != 2 or point_coords.shape[1] != 2:
                    raise ValueError("Each point in 'point_coords' must have exactly two values (x, y).")
                self.params["point_coords"] = point_coords

                # Ensure labels are provided for multiple points
                if "labels" not in self.params or not isinstance(self.params["labels"], list) or len(self.params["labels"]) != len(point_coords):
                    raise ValueError("For 'point' prompt, 'labels' must be provided as a list of the same length as 'point_coords'.")
                # Allow convert proper format
                self.params["point_coords"] = point_coords

            except Exception as e:
                raise ValueError(f'Invalid format for point_coords: {e}')
                    

        elif self.prompt_type == "bbox":
            if "bbox_coords" not in self.params or not isinstance(self.params["bbox_coords"], (tuple,list)):
                raise ValueError("For sam2 prompt 'bbox', 'bbox_coords' must be provided as a tuple (x1, y1, x2, y2).")

            try:
                bbox_coords = np.array(self.params["bbox_coords"])
                if bbox_coords.ndim != 2 or bbox_coords.shape[1] != 4:
                    raise ValueError("Each bounding box must have exactly four values (x1, y1, x2, y2).")
                
                # Ensure all coordinates are valid
                for bbox in bbox_coords:
                    x1, y1, x2, y2 = bbox
                    if not (x1 < x2 and y1 < y2):
                        raise ValueError(f"Invalid bbox coordinates: {bbox}. Ensure (x1, y1, x2, y2) with x1 < x2 and y1 < y2.")
                self.params["bbox_coords"] = bbox_coords
            
            except Exception as e:
                raise ValueError(f'Invalid format for bbox_coords: {e}')

                
        elif self.prompt_type == "mask":
            raise NotImplementedError("Not implemented yet and probably will not be used")
        
    @staticmethod
    def format_user_caption(user_cap:str) -> str:
        # Force split with dot
        formatted_caption = user_cap.replace(",", ".").replace("_", ".").replace(" ", ".").replace("+",".").replace("/",".")
        formatted_caption = ".".join(filter(None, formatted_caption.split(".")))
        return formatted_caption

    @property 
    def prompt_type(self) -> str:
        return self._prompt_type

    @prompt_type.setter
    def prompt_type(self, selected_prompt_type) -> None:
        if selected_prompt_type not in self.valid_prompt_types:
            raise ValueError(f"Invalid prompt type for SAM2! Valid promt types are: {self.valid_prompt_types}")
        self._prompt_type = selected_prompt_type


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

def mask_guided_filter(depth_map: np.ndarray, guidance_img: np.ndarray, obj_masks: dict) -> np.ndarray:
    """
    Apply guided filtering to the depth map for each object mask.

    Args:
        depth_map (np.ndarray): Input depth map.
        guidance_img (np.ndarray): Guidance image (can be the mask or an RGB image).
        obj_masks (dict): Dictionary of binary masks for each object.

    Returns:
        np.ndarray: Refined depth map with filtering applied to each mask region.
    """
    # Ensure depth map is uint8
    depth_map = depth_map.astype(np.uint8)

    # Handle zero values in the depth map (treat zero as invalid depth)
    zero_mask = (depth_map == 0)  # Find zero values in the depth map
    depth_map_filled = np.copy(depth_map)

    # Inpaint zero values globally
    if np.any(zero_mask):
        inpaint_radius = 5
        depth_map_filled = cv2.inpaint(depth_map_filled, zero_mask.astype(np.uint8), inpaint_radius, cv2.INPAINT_TELEA)

    # Normalize the guidance image
    if guidance_img.ndim == 3:  # RGB image
        guidance_img = guidance_img.astype(np.float32) / 255.0
    elif guidance_img.ndim == 2:  # Grayscale image
        guidance_img = cv2.cvtColor(guidance_img, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0

    # Initialize refined depth map
    refined_depth = np.copy(depth_map_filled)

    # Process each mask individually
    for obj_id, mask in obj_masks.items():
        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Normalize mask
        mask = mask.astype(np.float32) / 255.0

        # Filtered region
        filtered_region = np.copy(depth_map_filled)
        filtered_region[mask == 0] = 0  # Set pixels outside the mask to zero

        # Apply guided filter locally to the region inside the mask
        refined_region = cv2.ximgproc.guidedFilter(guide=guidance_img, src=filtered_region, radius=6, eps=1e-4)

        # Update refined depth map only within the mask
        refined_depth = np.where(mask > 0, refined_region, refined_depth)

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

def refine_depth_with_postprocessing(depth_map, left_image, obj_masks, lmbda=8000, sigma=1.5, inpaint_radius=3):
    """
    Refine depth map by combining inpainting, WLS filtering, and mask-guided blending for each object mask.

    Parameters:
        depth_map (np.ndarray): Input depth map.
        left_image (np.ndarray): Left RGB image from the stereo pair (used as guidance).
        obj_masks (dict): Dictionary of binary masks for each object.
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

    # Step 4: Apply mask-guided blending for each object mask
    final_depth_map = np.copy(depth_map)  # Start with the original depth map

    for obj_id, mask in obj_masks.items():
        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Normalize mask to [0, 1]
        mask = mask.astype(np.float32) / 255.0

        # Blend refined depth map with original depth map using the mask
        final_depth_map = final_depth_map * (1 - mask) + refined_depth * mask

    return final_depth_map





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