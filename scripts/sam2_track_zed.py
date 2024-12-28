import warnings
import os,sys
import torch
import numpy as np
import random
import cv2
import numpy as np
import time
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

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
from utils import process_masks, apply_nms, build_models, mask_guided_filter, update_caption, refine_depth_with_postprocessing
from utils import refine_depth_with_wjbf_and_sdcf
from utils import Sam2PromptType



def run(cfg, sam2_prompt: Sam2PromptType) -> None:
    
    if sam2_prompt.prompt_type == "g_dino_bbox":
        # Setup the caption queue
        caption_queue = queue.Queue()
        user_caption = sam2_prompt.user_caption
        caption_thread = threading.Thread(target=update_caption, args=(caption_queue, user_caption))
        caption_thread.daemon = True
        caption_thread.start()
    else:
        caption_queue = None 

    import seaborn as sns

    # Create a global color palette
    num_colors = 10  # Define a sufficiently large number of colors
    palette = sns.color_palette("hsv", num_colors)
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in palette]

    # Simple save folder ffor output images
    output_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output_img')
    os.makedirs(output_folder, exist_ok=True)

    # Build all necessary models
    sam2_predictor, grounding_dino_model = build_models(cfg)
    # Use Nakama Pyzed Wrapper for acces to ZED camera
    wrapper = pw.Wrapper(cfg.camera.connection_type)
    wrapper.open_input_source()

    l_intr, r_intr = wrapper.get_intrinsic()
    K_l = np.array([[l_intr.fx, 0, l_intr.cx],
                    [0, l_intr.fy, l_intr.cy],
                    [0, 0, 1]])

    wrapper.start_stream()

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
                cv2.imwrite(os.path.join(output_folder, 'norm_depth.png'), norm_depth_map)

                left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_RGBA2RGB)
                height, width, _ = left_image_rgb.shape
                cv2.imwrite(os.path.join(output_folder, 'left_img_og.png'), left_image_rgb)

                # Check if there is a new caption in the queue
                if caption_queue and not caption_queue.empty():
                    user_caption = Sam2PromptType.format_user_caption(caption_queue.get())
                    print(f"\nUpdated object to detect: {user_caption}")
                    # Force re-initialization with new object
                    if_init = False  
                    ann_frame_idx = 0
                    ann_obj_id = 1


                # Re-initialize with new object detection if a new caption is provided
                if not if_init:
                    sam2_predictor.load_first_frame(left_image_rgb)
                    ann_obj_id = 1  # Reset object ID counter for new object

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
                            warnings.warn(f"No detections found for '{user_caption}'. The current tracked object is '{user_caption}'. Consider changing it.")
                            cv2.imwrite(os.path.join(output_folder, 'test.png'), left_image_rgb)
                            continue  # Skip the rest of the loop and go to the next frame
                        else:
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
                        for idx, bbox in enumerate(bbox_coords):
                            x1, y1, x2, y2 = map(int, bbox)
                            color = [random.randint(0, 255) for _ in range(3)]
                            cv2.rectangle(left_image_rgb, (x1, y1), (x2, y2), color, 2)


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
                    # Continue tracking with the current object
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
                # refined_depth_map = mask_guided_filter(depth_map, left_image_rgb, obj_masks)
                # refined_depth_map = refine_depth_with_postprocessing(depth_map, left_image_rgb, obj_masks)     
                guidance_img = np.sum([mask for mask in obj_masks.values()], axis=0).astype(np.uint8)  # Combine all masks               
                refined_depth_map = refine_depth_with_wjbf_and_sdcf(depth_map, obj_masks, guidance_img, sigma_spatial=15, sigma_range=30)      
                cv2.imwrite(os.path.join(output_folder, 'refined_depth.png'), refined_depth_map)
                ann_frame_idx+=1

            # ann_frame_idx += 1
            te = time.time()

    
            print(f"\rCurrent FPS: {1 / (te - ts):.2f}", end="")

            cv2.imwrite(os.path.join(output_folder, 'test.png'), left_image_rgb)
            pass

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        wrapper.stop_stream()
        wrapper.close_input_source()

# Main function using Hydra to load config
if __name__ == "__main__":
    if GlobalHydra.is_initialized:
        GlobalHydra.instance().clear()

    with initialize(config_path="../configurations"):
        cfg = compose(config_name="sam2_zed_small")
        sam2_prompt = Sam2PromptType('g_dino_bbox',user_caption='keyboard')
        

        # point_coords = [(390, 200)]
        # labels = [1]  # 1 = foreground, 0 = background
        # sam2_prompt = Sam2PromptType('point', point_coords = point_coords, labels=labels)

        # bbox_coords = [(50, 50, 150, 150), (200, 200, 300, 300)] #! NOTE: 3+ boxes make it really inaccurate
        # sam2_prompt = Sam2PromptType('bbox', bbox_coords = bbox_coords)

        run(cfg, sam2_prompt=sam2_prompt)
