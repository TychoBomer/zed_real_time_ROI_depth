import warnings
import os,sys
import torch
import numpy as np
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
from utils import process_masks, apply_nms, build_models, mask_guided_filter, update_caption, refine_depth_with_preprocessing

# External input queue for queued caption inputs
caption_queue = queue.Queue()
user_caption = "keyboard"
fps_print_active = True

# Start the input handler in a separate thread
caption_thread = threading.Thread(target=update_caption, args=(caption_queue, user_caption))
caption_thread.daemon = True
caption_thread.start()


def run(cfg) -> None:
    global user_caption, fps_print_active
    if_init = False
    
    output_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output_img')
    os.makedirs(output_folder, exist_ok=True)

    sam2_predictor, grounding_dino_model = build_models(cfg)

    wrapper = pw.Wrapper(cfg.camera.connection_type)
    wrapper.open_input_source()

    l_intr, r_intr = wrapper.get_intrinsic()
    K_l = np.array([[l_intr.fx, 0, l_intr.cx],
                    [0, l_intr.fy, l_intr.cy],
                    [0, 0, 1]])

    wrapper.start_stream()
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
                if not caption_queue.empty():
                    user_caption = caption_queue.get()
                    print(f"\nUpdated object to detect: {user_caption}")
                    # Force re-initialization with new object
                    if_init = False  
                    ann_frame_idx = 0
                    ann_obj_id = 1


                # Re-initialize with new object detection if a new caption is provided
                if not if_init:
                    sam2_predictor.load_first_frame(left_image_rgb)
                    ann_obj_id = 1  # Reset object ID counter for new object
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                        detections = grounding_dino_model.predict_with_classes(
                            image=left_image_rgb,
                            classes=[user_caption],
                            box_threshold=cfg.grounding_dino.box_threshold,
                            text_threshold=cfg.grounding_dino.text_threshold
                        )

                    if not detections or len(detections.xyxy) == 0:
                        warnings.warn(f"No detections found for '{user_caption}'. The current tracked object is '{user_caption}'. Consider changing it.")
                        # continue  # Skip the rest of the loop and go to the next frame
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

                        all_mask = process_masks(out_obj_ids, out_mask_logits, height, width)
                        mask_colored = np.zeros_like(left_image_rgb)
                        mask_colored[:, :, 1] = all_mask
                        left_image_rgb = cv2.addWeighted(left_image_rgb, 1, mask_colored, 0.5, 0)
                        if_init = True  # Re-initialization done, start tracking
                        
                         # Apply mask-guided filtering to the depth map
                        # refined_depth_map = mask_guided_filter(depth_map, left_image_rgb, all_mask)
                        refined_depth_map = refine_depth_with_preprocessing(depth_map, left_image_rgb, all_mask)                        
                        cv2.imwrite(os.path.join(output_folder, 'refined_depth.png'), refined_depth_map)
                        ann_frame_idx+=1
                else:
                    # Continue tracking with the current object
                    out_obj_ids, out_mask_logits = sam2_predictor.track(left_image_rgb)
                    all_mask = process_masks(out_obj_ids, out_mask_logits, height, width)
                    mask_colored = np.zeros_like(left_image_rgb)
                    mask_colored[:, :, 1] = all_mask
                    left_image_rgb = cv2.addWeighted(left_image_rgb, 1, mask_colored, 0.5, 0)
                    # refined_depth_map = mask_guided_filter(depth_map, left_image_rgb, all_mask)
                    refined_depth_map = refine_depth_with_preprocessing(depth_map, left_image_rgb, all_mask)                    
                    cv2.imwrite(os.path.join(output_folder, 'refined_depth.png'), refined_depth_map)
                    
                    ann_frame_idx+=1

            # ann_frame_idx += 1
            te = time.time()

            # Print FPS without interrupting user input
            if fps_print_active:
                print(f"\rCurrent FPS: {1 / (te - ts):.2f}", end="")

            cv2.imwrite(os.path.join(output_folder, 'test.png'), left_image_rgb)

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
        run(cfg)
