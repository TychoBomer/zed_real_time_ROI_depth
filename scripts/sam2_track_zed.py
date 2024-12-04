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
from utils import process_masks, apply_nms, build_models, mask_guided_filter, update_caption, refine_depth_with_postprocessing

# External input queue for queued caption inputs
caption_queue = queue.Queue()
user_caption = "keyboard"
fps_print_active = True

# Start the input handler in a separate thread
caption_thread = threading.Thread(target=update_caption, args=(caption_queue, user_caption))
caption_thread.daemon = True
caption_thread.start()


class Sam2PromptType:
    valid_prompt_types = {"g_dino_bbox", "bbox", "point", "mask"} # all types SAM2 could handle

    def __init__(self, prompt_type, **kwargs) -> None:
        self._prompt_type = None
        self.prompt_type = prompt_type # attempts the set function @prompt_type.setter
        self.params = kwargs
        self.validate()

    def validate(self)->None:
        if self.prompt_type == "point":
            if "point_coords" not in self.params or not isinstance(self.params["point_coords"], tuple) or len(self.params["point_coords"])!=2:
                raise ValueError("For sam2 prompt 'point', 'point_coords' must be provided as a tuple (x,y).")
            
            point_coords = self.params["point_coords"]
            try:
                # convert to np.array for prompting sam
                point_coords = np.array(point_coords)
                if point_coords.ndim == 1:
                    point_coords = np.expand_dims(point_coords,axis=0)
                if point_coords.shape[1] !=2:
                    raise ValueError("point in point_coords must have two values (x,y)")

            except Exception as e:
                ValueError(f"Invalid format for point_coords: {e}")
            # Allow convert proper format
            self.params["point_coords"] = point_coords
                

        # TODO: BBOX PROPER FORMAT FOR SAM 2 PROMPTING
        elif self.prompt_type == "bbox":
            if "bbox_coords" not in self.params or not isinstance(self.params["bbox_coords"],tuple) or len(self.params["bbox_coords"])!=4:
                raise ValueError("For sam2 prompt 'bbox', 'bbox_coords' must be provided as a tuple (x1,y1,x2,y2).")
        
        elif self.prompt_type == "mask":
            raise NotImplementedError("Not implemented yet and probably will not be used")

    @property 
    def prompt_type(self):
        return self._prompt_type

    @prompt_type.setter
    def prompt_type(self, selected_prompt_type):
        if selected_prompt_type not in self.valid_prompt_types:
            raise ValueError(f"Invalid prompt type for SAM2! Valid promt types are: {self.valid_prompt_types}")
        self._prompt_type = selected_prompt_type



def run(cfg, sam2_prompt: Sam2PromptType) -> None:
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

                    elif sam2_prompt.prompt_type == "point":
                        # NOTE: if using points you NEED a label. Label = 1 is foreground point, label = 0 is backgroiund point. 
                        # We will probanly only need foreground points
                        point_coords = sam2_prompt.params["point_coords"]
                        label = np.array([1])
                        with torch.inference_mode():
                            _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_prompt(
                                    frame_idx=ann_frame_idx, obj_id=ann_obj_id, points = point_coords, labels = label
                                )
                            ann_obj_id += 1

                    # elif sam2_prompt.prompt_type == "bbox":
                    #     bbox_coords = sam2_prompt


                    all_mask = process_masks(out_obj_ids, out_mask_logits, height, width)
                    mask_colored = np.zeros_like(left_image_rgb)
                    mask_colored[:, :, 1] = all_mask
                    left_image_rgb = cv2.addWeighted(left_image_rgb, 1, mask_colored, 0.5, 0)
                    if_init = True  # Re-initialization done, start tracking
                    
                        # Apply mask-guided filtering to the depth map
                    # refined_depth_map = mask_guided_filter(depth_map, left_image_rgb, all_mask)
                    refined_depth_map = refine_depth_with_postprocessing(depth_map, left_image_rgb, all_mask)                        
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
                    refined_depth_map = refine_depth_with_postprocessing(depth_map, left_image_rgb, all_mask)                    
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

        sam2_prompt = Sam2PromptType('point', point_coords = (300,200))
        # sam2_prompt = Sam2PromptType('g_dino_bbox')

        run(cfg, sam2_prompt=sam2_prompt)
