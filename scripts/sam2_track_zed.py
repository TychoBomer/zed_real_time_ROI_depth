import os,sys
import torch
import numpy as np
import random
import cv2
import time
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import imageio

import threading
import queue

sys.path.insert(0, os.getcwd())

# Initialize GPU settings
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()


from wrappers.pyzed_wrapper import pyzed_wrapper as pw
from scripts.utils.utils import *
from scripts.utils.depth_utils import (
    mask_guided_filter,
    refine_depth_with_postprocessing,
    refine_depth_with_wjbf_and_sdcf, 
    refine_depth_with_cross_bilateral,
    refine_depth_with_plane_fitting,
    refine_depth_with_mhmf,
    refine_depth_with_hole_filling
    )
from scripts.utils.depth_utils import visualize_depth, to_numpy_func, resize_to_multiple

from utils.logger import Log



def run(cfg, sam2_prompt: Sam2PromptType) -> None:

    Log.info("Initializing the pipeline...", tag="pipeline_init")
    if sam2_prompt.prompt_type == "g_dino_bbox":
        # Setup the caption queue
        caption_queue = queue.Queue()
        user_caption = sam2_prompt.user_caption
        caption_thread = threading.Thread(target=update_caption, args=(caption_queue, user_caption))
        caption_thread.daemon = True
        caption_thread.start()
    else:
        caption_queue = None 

    


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
    wrapper = pw.Wrapper(cfg.camera.connection_type)
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

    try:
        while True:
            ts = time.time()

            if wrapper.retrieve(is_image=True, is_measure=True):
                # Extract from ZED camera
                left_image = wrapper.output_image
                depth_map = wrapper.output_measure
                norm_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                depth_resized = resize_to_multiple(depth_map, 14)
                depth_tensor = torch.tensor(depth_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to('cuda')
                
                prompt_depth = to_numpy_func(depth_tensor)
                prompt_depth_vis,depth_min, depth_max= visualize_depth(prompt_depth, ret_minmax=True)
                prompt_depth_vis,depth_min, depth_max= visualize_depth(prompt_depth, ret_minmax=True, depth_min=depth_min, depth_max=depth_max)

                imageio.imwrite(os.path.join(output_dir, 'norm_depth_heatmap.jpg'), prompt_depth_vis)


                cv2.imwrite(os.path.join(output_dir, 'norm_depth.png'), norm_depth_map)

                heatmap_depth_map = visualize_depth(norm_depth_map, cmap='Spectral')
                cv2.imwrite(os.path.join(output_dir, 'norm_depth_heatmap.jpg'), heatmap_depth_map)


                left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_RGBA2RGB)
                height, width, _ = left_image_rgb.shape
                cv2.imwrite(os.path.join(output_dir, 'left_img_og.png'), left_image_rgb)

                # Check if there is a new caption in the queue
                if caption_queue and not caption_queue.empty():
                    user_caption = Sam2PromptType.format_user_caption(caption_queue.get())
                    Log.info(f"Updated object to detect: {user_caption}", tag="caption_update")
                    # Force re-initialization with new object
                    if_init = False  
                    ann_frame_idx = 0
                    ann_obj_id = 1


                # Re-initialize with new object detection if a new caption is provided
                if not if_init:
                    previous_depth = norm_depth_map
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
                            continue  # Skip the rest of the loop and go to the next frame
                        else:
                            Log.info(f"Detections found for '{user_caption}'")
                            detections = apply_nms(detections, cfg.grounding_dino.nms_threshold)

                            for box in detections.xyxy:
                                x1, y1, x2, y2 = map(int, box)
                                cv2.rectangle(left_image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

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

                
                left_image_rgb = cv2.addWeighted(left_image_rgb, 1, all_mask, 0.5, 0)
                
                # *Refine the depth mask using masks
                if cfg.depth.refine_depth:
                    # refined_depth_map = mask_guided_filter(depth_map, left_image_rgb, obj_masks)
                    # refined_depth_map = refine_depth_with_postprocessing(depth_map, left_image_rgb, obj_masks)     
                    # refined_depth_map = refine_depth_with_wjbf_and_sdcf(depth_map, obj_masks, guidance_img, sigma_spatial=15, sigma_range=30)      
                    # refined_depth_map = refine_depth_with_plane_fitting(depth_map, obj_masks)
                    # refined_depth_map = refine_depth_with_mhmf(depth_map, obj_masks, depth_threshold=(1, 255))            
                    guidance_img = np.sum([mask for mask in obj_masks.values()], axis=0).astype(np.uint8)
                    refined_depth_map = refine_depth_with_hole_filling(
                    current_depth=norm_depth_map,
                    previous_depth=previous_depth,
                    obj_masks=obj_masks,
                    guidance_img=guidance_img,
                    max_distance=5,
                    alpha=0.5,
                    kernel_size=5
                    )
                    
                    
                    refined_depth_map = cv2.normalize(refined_depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    previous_depth = norm_depth_map
                    cv2.imwrite(os.path.join(output_dir, 'refined_depth.png'), refined_depth_map)
                ann_frame_idx+=1

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
        # sam2_prompt = Sam2PromptType('g_dino_bbox',user_caption='apple')
        

        # point_coords = [(390, 200)]
        # labels = [1]  # 1 = foreground, 0 = background
        # sam2_prompt = Sam2PromptType('point', point_coords = point_coords, labels=labels)

        bbox_coords = [(320, 120, 470, 280)]
        # bbox_coords = [(50, 50, 150, 150), (200, 200, 300, 300)] #! NOTE: 3+ boxes make it really inaccurate
        sam2_prompt = Sam2PromptType('bbox', bbox_coords = bbox_coords)

        run(cfg, sam2_prompt=sam2_prompt)
