import os,sys
import torch, torchvision
import numpy as np
import cv2
import numpy as np
import time
from hydra import initialize, compose

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
#     # set to bfloat for entire file
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()


from sam2.build_sam import build_sam2_camera_predictor
sys.path.insert(0, os.getcwd())
from wrappers.pyzed_wrapper import pyzed_wrapper as pw

from GroundingDINO.groundingdino.util.inference import Model

from hydra.core.global_hydra import GlobalHydra
from utils import process_masks, apply_nms


def run(cfg) -> None:
    # Define the path to the output folder
    output_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output_img')
    os.makedirs(output_folder, exist_ok=True)



    if_init = False
    grounding_dino_model = Model(model_config_path=cfg.grounding_dino.config_path,
                                            model_checkpoint_path=cfg.grounding_dino.checkpoint_path)
    predictor = build_sam2_camera_predictor(cfg.sam2.model_cfg, cfg.sam2.checkpoint)

    # Create an instance of the Wrapper class
    wrapper = pw.Wrapper('id')
    # Open the input source (camera, stream, or file)
    wrapper.open_input_source()

    # Extracting intrinsics
    l_intr, r_intr = wrapper.get_intrinsic()
    K_l = np.array([[l_intr.fx, 0, l_intr.cx],
                    [0, l_intr.fy, l_intr.cy],
                    [0, 0, 1]])

    wrapper.start_stream()
    try:
        while True:
            ts = time.time()
            # Retrieve a frame (image and depth)
            if wrapper.retrieve(is_image=True, is_measure=True):

                # Extraction from zed camera
                left_image = wrapper.output_image
                depth_map = wrapper.output_measure
                norm_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite(os.path.join(output_folder, 'norm_depth.png'), norm_depth_map)


                # Convert left image to RGB format
                left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_RGBA2RGB)
                height, width, _ = left_image_rgb.shape
                cv2.imwrite(os.path.join(output_folder, 'left_img_og.png'), left_image_rgb)

                if not if_init:
                    predictor.load_first_frame(left_image_rgb)
                    if_init = True

                    ann_frame_idx = 0 
                    ann_obj_id = 1  
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):

                        detections =grounding_dino_model.predict_with_classes(
                        image=left_image_rgb,

                        classes=[cfg.grounding_dino.caption],
                        box_threshold=cfg.grounding_dino.box_threshold,
                        text_threshold=cfg.grounding_dino.text_threshold)

                    if not detections or len(detections.xyxy) == 0:
                        raise ValueError("No detections found")

                    detections = apply_nms(detections, cfg.grounding_dino.nms_threshold)

                    # input_boxes  = np.zeros((len(detections.xyxy), 2, 2), dtype=np.float32)
                    for box in (detections.xyxy):
                        ts_box = time.time()
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(left_image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

                
                        # input_boxes = detections.xyxy.reshape(2,2)
                        input_boxes = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                        # with torch.inference_mode(),torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                            frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=input_boxes
                        )
                        ann_obj_id+=1
            

                    # process masks
                    all_mask = process_masks(out_obj_ids, out_mask_logits, height, width)
                    mask_colored = np.zeros_like(left_image_rgb)
                    mask_colored[:, :, 1] = all_mask
                    left_image_rgb = cv2.addWeighted(left_image_rgb, 1, mask_colored, 0.5, 0)


            
                else:
                    # Use tracking
                    out_obj_ids, out_mask_logits = predictor.track(left_image_rgb)
                    # Process masks
                    all_mask = process_masks(out_obj_ids, out_mask_logits, height, width)
                    mask_colored = np.zeros_like(left_image_rgb)
                    mask_colored[:, :, 1] = all_mask
                    left_image_rgb = cv2.addWeighted(left_image_rgb, 1, mask_colored, 0.5, 0)
                
            ann_frame_idx+=1
            te = time.time()
            print(f"Current FPS:  {1/(te-ts)}")

            cv2.imwrite(os.path.join(output_folder, 'test.png'), left_image_rgb)


    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Stop streaming and close the input source
        wrapper.stop_stream()
        wrapper.close_input_source()


# Main function that uses Hydra to load the config and runs the main logic
if __name__ == "__main__":
    # Initialize Hydra configuration file
    if GlobalHydra.is_initialized:
        GlobalHydra.instance().clear()
    with initialize(config_path="../configurations"):
        cfg = compose(config_name="sam2_zed_small")  
    
        # Run the main pipeline with the loaded config
        run(cfg)