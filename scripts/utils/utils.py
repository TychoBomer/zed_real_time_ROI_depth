import time
import torch
import numpy as np
import torch.nn as nn
from typing import Tuple
from GroundingDINO.groundingdino.util.inference import Model
from sam2.build_sam import build_sam2_camera_predictor
import queue

from concurrent.futures import ThreadPoolExecutor
from deprecated import deprecated




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