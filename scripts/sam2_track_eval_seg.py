import os,sys
import torch
import numpy as np
import random
import cv2
import time
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

import threading
import queue
from skimage.measure import find_contours
from pycocotools import mask as mask_utils
import json

sys.path.insert(0, os.getcwd())


# Initialize GPU settings
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()


from wrappers.pyzed_wrapper import pyzed_wrapper_v2 as pw
from scripts.utils.utils import *
from scripts.utils.depth_utils import (
    depth_refinement_RANSAC_plane_fitting,
    )
from utils.logger import Log
from utils.sam2prty import Sam2PromptType


# Boundary IoU functions
def mask_to_boundary(mask, dilation_ratio=0.02):
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1

    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    return mask - mask_erode

def compute_boundary_iou(gt_mask, pred_mask, dilation_ratio=0.02):
    gt_boundary = mask_to_boundary(gt_mask, dilation_ratio)
    pred_boundary = mask_to_boundary(pred_mask, dilation_ratio)
    intersection = np.logical_and(gt_boundary, pred_boundary).sum()
    union = np.logical_or(gt_boundary, pred_boundary).sum()
    return intersection / union if union > 0 else 0

def compute_iou(gt_mask, pred_mask):
    """Computes Intersection over Union (IoU)."""
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return intersection / union if union > 0 else 0



def load_ground_truth_mask(json_path, image_shape):
    """Loads ground truth segmentation mask from Roboflow COCO JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)

    annotation = data["annotations"][0]
    rle = annotation["segmentation"]
    gt_mask = mask_utils.decode(rle)

    # Resize if necessary
    if gt_mask.shape != image_shape:
        gt_mask = cv2.resize(gt_mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)

    return gt_mask.astype(np.uint8)



# Function to decode COCO RLE segmentation mask
def decode_coco_rle(segmentation, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    rle_counts = segmentation["counts"]
    rle_counts = [int(x) for x in rle_counts.split()]
    
    current_pos = 0
    for i in range(0, len(rle_counts), 2):
        start = rle_counts[i]
        length = rle_counts[i + 1]
        mask.ravel()[current_pos + start : current_pos + start + length] = 1
        current_pos += start + length
    
    return mask


def run(cfg, sam2_prompt: Sam2PromptType) -> None:
    depth_refinement_enabled = cfg.depth.refine_depth

    caption_queue = queue.Queue()

    Log.info("Initializing the pipeline...", tag="pipeline_init")
    if sam2_prompt.prompt_type == "g_dino_bbox":
        user_caption = sam2_prompt.user_caption

        
    caption_thread = threading.Thread(target=update_caption, args=(caption_queue, sam2_prompt.user_caption), daemon=True)
    caption_thread.start()

    


    # Build needed models
    Log.info("Building models...", tag="building_models")
    try:
        sam2_predictor, grounding_dino_model = build_models(cfg)
        Log.info("Models successfully built and loaded.", tag="model_building")
    except Exception as e:
        Log.error(f"Failed to build models: {e}", tag="model_build_error")
        return
    
    # Use Nakama Pyzed Wrapper for acces to ZED camera
    Log.info("Initializing ZED camera...", tag="zed_camera_init")
    wrapper = pw.Wrapper(cfg.camera)
    try:
        wrapper.open_input_source()
        Log.info("ZED camera initialized.", tag="camera_init")
    except Exception as e:
        Log.error(f"Failed to initialize ZED camera: {e}", tag="camera_init_error")
        return

    l_intr, r_intr = wrapper.get_intrinsic()
    K_l = np.array([[l_intr.fx, 0, l_intr.cx],
                    [0, l_intr.fy, l_intr.cy],
                    [0, 0, 1]])
    

    # Fixed color palette for sam mask ids
    colors = create_mask_color_palette(10,'hsv')
    # Simple save folder for output images
    output_dir = setup_output_folder(cfg.results.output_dir)

    wrapper.start_stream()
    Log.info('Camera stream started.', tag = "camera_stream")

    if_init = False
    ann_frame_idx = 0
    ann_obj_id = 1
    depth_refinement_enabled = cfg.depth.refine_depth



    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/home/nakama/Documents/TychoMSC/models/sam2_track_test/segment-anything-2-real-time/output/output.mp4', fourcc, 20, (1280, 720), True)
    out2 = cv2.VideoWriter('/home/nakama/Documents/TychoMSC/models/sam2_track_test/segment-anything-2-real-time/output/output_depth.mp4', fourcc, 20, (1280, 720), False)
    
    framecount = 0

    # Extract the base name of the SVO file (e.g., "bottle_aruco_1040" from "./output/bottle_aruco_1040.svo2")
    svo_filename = os.path.basename(cfg.camera.svo_input_filename)  # "bottle_aruco_1040.svo2"
    svo_base, _ = os.path.splitext(svo_filename)  # "bottle_aruco_1040"

    # Create ground truth directory inside the output folder
    gt_output_dir = os.path.join(output_dir, "ground_truth")
    gt_ann_output_dir = os.path.join(output_dir, "ground_truth_ann")
    os.makedirs(gt_output_dir, exist_ok=True)  # Ensure the directory exists
    
    ## MAIN LOOP
    try:
        while True:
            ts = time.time()

            if wrapper.retrieve(is_image=True, is_measure=True):
                # Extract from ZED camera
                left_image = wrapper.output_image
                depth_map = wrapper.output_measure


                depth_map_og = depth_map.astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, 'depth_map_og.png'),depth_map_og)


                norm_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, 'norm_depth.png'), norm_depth_map)


                left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_RGBA2RGB)
                height, width, _ = left_image_rgb.shape
                if framecount == 150:
                    unique_name = f"{svo_base}_frame_{framecount}.png"
                    save_path = os.path.join(gt_output_dir, unique_name)
                    cv2.imwrite(save_path, left_image_rgb)
                    json_path = os.path.join(gt_ann_output_dir, "bottle_aruco_720_frame_200.json")
                    if os.path.exists(json_path):
                        gt_mask = load_ground_truth_mask(json_path, image_shape=(720, 1280))


                # Handling of the queue
                if not caption_queue.empty():
                    msg = caption_queue.get()
                    if msg == "tr":
                        depth_refinement_enabled = not depth_refinement_enabled
                        status_text = f"{GREEN}ENABLED{RESET}" if depth_refinement_enabled else f"{RED}DISABLED{RESET}"
                        Log.info(f"Depth refinement toggled: {status_text}", tag="toggle_refinement")
                    else:
                        user_caption = Sam2PromptType.format_user_caption(msg)
                        Log.info(f"Updated object to detect: {user_caption}", tag="caption_update")
                        if_init = False  
                        ann_frame_idx = 0
                        ann_obj_id = 1
                    


                # Re-initialize with new object detection if a new caption is provided
                if not if_init:
                    sam2_predictor.load_first_frame(left_image_rgb)
                    ann_obj_id = 1  # Reset object ID counter for new object if pipeline restarted

                    #* Use GroundingDINO box(es) as initial prompt
                    if sam2_prompt.prompt_type == "g_dino_bbox":
                        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                            detections = grounding_dino_model.predict_with_classes(
                                image=left_image_rgb,
                                classes=[user_caption],
                                box_threshold=cfg.grounding_dino.box_threshold,
                                text_threshold=cfg.grounding_dino.text_threshold
                            )

                        if not detections or len(detections.xyxy) == 0:
                            Log.warn(f"No detections found for '{user_caption}', Consider changing it.")
                            cv2.imwrite(os.path.join(output_dir, 'test.png'), left_image_rgb)
                            framecount+=1

                            continue  # Skip the rest of the loop and go to the next frame
                        else:
                            Log.info(f"Detections found for '{user_caption}'")
                            detections = apply_nms(detections, cfg.grounding_dino.nms_threshold)

                            for box in detections.xyxy:
                                x1, y1, x2, y2 = map(int, box)
                                # cv2.rectangle(left_image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                input_boxes = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                                _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_prompt(
                                    frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=input_boxes
                                )
                                ann_obj_id += 1
                            if_init=True


                    #* Use point(s) as initial prompt
                    elif sam2_prompt.prompt_type == "point":
                        # NOTE: if using points you NEED a label. Label = 1 is foreground point, label = 0 is backgroiund point. 
                        # We will probanly only need foreground points
                        point_coords = sam2_prompt.params["point_coords"]
                        labels = np.array(sam2_prompt.params["labels"])  
                        for point, label in zip(point_coords, labels):
                            x ,y =map(int, point)
                            color = (0, 255, 0) if label == 1 else (0, 0, 255)
                            cv2.circle(left_image_rgb, (x, y), 3, color, 2)

                        with torch.inference_mode():
                            _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_prompt(
                                    frame_idx=ann_frame_idx, obj_id=ann_obj_id, points = point_coords, labels = labels
                                )
                            # ann_obj_id += len(point_coords)
                            ann_obj_id += 1
                        if_init=True
                        

                    #* Use pre defined box(es) as initial prompt
                    elif sam2_prompt.prompt_type == "bbox":
                        bbox_coords = sam2_prompt.params["bbox_coords"]
                        for box_id, bbox in enumerate(bbox_coords):
                            x1, y1, x2, y2 = map(int, bbox)
                            color = colors[(box_id+1)%len(colors)+1]
                            cv2.rectangle(left_image_rgb, (x1, y1), (x2, y2), (color[0], color[2], color[1]) , 2)


                        with torch.inference_mode():
                            for bbox in bbox_coords: 
                                x1, y1, x2, y2 = map(int, bbox)
                                input_bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                                _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_prompt(
                                    frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=input_bbox
                                )
                                ann_obj_id += 1 
                        if_init=True

                    
                else:
                    #* Continue tracking with the current object
                    out_obj_ids, out_mask_logits = sam2_predictor.track(left_image_rgb)


                #* Mask processing
                all_mask = np.zeros_like(left_image_rgb, dtype=np.uint8)
                obj_masks = process_masks(out_obj_ids, out_mask_logits)
                for obj_id, mask in obj_masks.items():
                    color = colors[obj_id%len(colors)+1]
                    colored_mask = np.zeros_like(left_image_rgb, dtype=np.uint8)
                    for c in range(3):
                        colored_mask[:, :, c] = mask[:, :, 0] * color[c]
                    all_mask = cv2.add(all_mask,colored_mask)

                
                left_image_rgb = cv2.addWeighted(left_image_rgb, 1, all_mask, 0.8, 0)
                


                ### SEGMENT EVALUATION
                if framecount ==200:
                    pred_mask = np.zeros_like(gt_mask, dtype=np.uint8)
                    for obj_id, mask in obj_masks.items():
                        pred_mask = np.maximum(pred_mask, mask[:, :, 0])

                        iou = compute_iou(gt_mask, pred_mask)
                        boundary_iou = compute_boundary_iou(gt_mask, pred_mask)

                        Log.info(f"IoU: {iou:.4f}, Boundary IoU: {boundary_iou:.4f}", tag="iou_results")


                # For video capture
                if framecount < 5000:
                    out.write(left_image_rgb)
                    out2.write(norm_depth_map)
                    framecount+=1
                else:
                    out.write(left_image_rgb)
                    out.release()
                    out2.write(norm_depth_map)
                    out2.release()
                    cv2.destroyAllWindows()
                    break
                
                # *Refine the depth mask using masks
                if depth_refinement_enabled:
                    # !NOTE: plane fiting best one so far
                    refined_depth_map = depth_refinement_RANSAC_plane_fitting(depth_map, obj_masks, max_occ_percentage=cfg.depth.max_occlusion_percentage)
                    refined_depth_map = cv2.normalize(refined_depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    # previous_depth = norm_depth_map
                    cv2.imwrite(os.path.join(output_dir, 'refined_depth.png'), refined_depth_map)







                ann_frame_idx+=1


            else:
                # svo END REACHED
                break
        

            # ann_frame_idx += 1
            te = time.time()

    
            print(f"\rCurrent FPS: {1 / (te - ts):.2f}", end="")

            cv2.imwrite(os.path.join(output_dir, 'test.png'), left_image_rgb)
            pass

    except Exception as e:
        Log.error(f"An error occurred during execution: {e}", tag="runtime_error")

    finally:
        Log.info("Shutting down and closing stream...", tag ='Close_stream')
        wrapper.stop_stream()
        wrapper.close_input_source()

# Main function using Hydra to load config
if __name__ == "__main__":
    if GlobalHydra.is_initialized:
        GlobalHydra.instance().clear()

    with initialize(config_path="../configurations"):
        cfg = compose(config_name="sam2_zed_small")
        sam2_prompt = Sam2PromptType('g_dino_bbox',user_caption='bottle')
        

        # point_coords = [(390, 200)]
        # labels = [1]  # 1 = foreground, 0 = background
        # sam2_prompt = Sam2PromptType('point', point_coords = point_coords, labels=labels)

        # bbox_coords = [(601, 350, 680, 500)]
        # bbox_coords = [(50, 50, 150, 150), (200, 200, 300, 300)] #! NOTE: 3+ boxes make it really inaccurate
        # sam2_prompt = Sam2PromptType('bbox', bbox_coords = bbox_coords)

        run(cfg, sam2_prompt=sam2_prompt)
