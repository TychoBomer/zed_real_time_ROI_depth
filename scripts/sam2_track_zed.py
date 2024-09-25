import os,sys
import torch, torchvision

import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import numpy as np



# from google.colab.patches import cv2_imshow

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor
import time
sys.path.insert(0, os.getcwd())
from wrappers.pyzed_wrapper import pyzed_wrapper as pw

from GroundingDINO.groundingdino.util.inference import Model, box_convert, predict

 # Define default variables
 # CAPTION : str  = 'thumb
BOX_THRESHOLD : float = 0.35
TEXT_THRESHOLD : float = 0.25
NMS_THRESHOLD : float = 0.6
CAPTION : str  = 'thumb'

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH : str = "/home/nakama/Documents/TychoMSC/models/sam2_track_test/segment-anything-2-real-time/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH: str = "/home/nakama/Documents/TychoMSC/models/sam2_track_test/segment-anything-2-real-time/GroundingDINO/weights/groundingdino_swint_ogc.pth"


grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                                           model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)




sam2_checkpoint = "/home/nakama/Documents/TychoMSC/models/sam2_track_test/segment-anything-2-real-time/checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)



    # Create an instance of the Wrapper class
wrapper = pw.Wrapper('id')
    # Open the input source (camera, stream, or file)
wrapper.open_input_source()

# Extracting intrinsics
l_intr, r_intr = wrapper.get_intrinsic()
K_l = np.array([[l_intr.fx, 0, l_intr.cx],
                [0, l_intr.fy, l_intr.cy],
                [0, 0, 1]])

    # from scripts.wrappers.all_sam_wrapper import AllSAMWrapper
    # # sam_wrap = AllSAMWrapper("MobileSAM")
    # # sam_wrap = AllSAMWrapper("EdgeSAM")
    # # sam_wrap = AllSAMWrapper("RepViTSAM")
    # sam_wrap = AllSAMWrapper('LightHQSAM')


wrapper.start_stream()





# cap = cv2.VideoCapture("/home/nakama/Documents/TychoMSC/models/sam2_track_test/segment-anything-2-real-time/notebooks/videos/aquarium/aquarium.mp4")
if_init = False

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    try:
        while True:
            # Retrieve a frame (image and depth)
            if wrapper.retrieve(is_image=True, is_measure=True):
                ts = time.time()

                # ts = time.time()
                left_image = wrapper.output_image
                depth_map = wrapper.output_measure
                # norm_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        # depth_edges = cv2.Canny(norm_depth_map, threshold1=5, threshold2=80)
                    

                # Convert left image to RGB format

                left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_RGBA2RGB)
                cv2.imwrite('og_img_rgb.png', left_image_rgb)
                
                width, height = left_image_rgb.shape[:2][::-1]

                if not if_init:

                    predictor.load_first_frame(left_image_rgb)
                    if_init = True

                    ann_frame_idx = 0  # the frame index we interact with
                    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        detections =grounding_dino_model.predict_with_classes(
                        image=left_image_rgb,
                        classes=[CAPTION],
                        box_threshold=BOX_THRESHOLD,
                        text_threshold=TEXT_THRESHOLD)

                    if not detections or len(detections.xyxy) == 0:
                        raise ValueError("No detections found")
                    
                    print(f"Type of detections.xyxy: {type(detections.xyxy)}")
                    print(f"Type of detections.confidence: {type(detections.confidence)}")

                    # NMS post-process
                    print(f"Before NMS: {len(detections.xyxy)} boxes")
                    nms_idx = torchvision.ops.nms(
                        torch.from_numpy(detections.xyxy).float(), 
                        torch.from_numpy(detections.confidence).float(), 
                        NMS_THRESHOLD
                    ).numpy().tolist()

                    if len(nms_idx) == 0:
                        raise ValueError("No boxes left after NMS")

                    detections.xyxy = detections.xyxy[nms_idx]
                    detections.confidence = detections.confidence[nms_idx]
                    detections.class_id = detections.class_id[nms_idx]

                    print(f"After NMS: {len(detections.xyxy)} boxes")
                    input_boxes  = np.zeros((len(detections.xyxy), 2, 2), dtype=np.float32)
                    for i, box in enumerate(detections.xyxy):
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(left_image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

                
                        # input_boxes = detections.xyxy.reshape(2,2)
                        input_boxes[i] = np.array([[x1, y1], [x2, y2]], dtype=np.float32)

                    # bbox = np.array(bbox, dtype=np.float32)
                    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                        frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=input_boxes
                    )
                    all_mask = np.zeros((height, width, 1), dtype=np.uint8)
                    # print(all_mask.shape)
                    for i in range(0, len(out_obj_ids)):
                        out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                            np.uint8
                        ) * 255

                        all_mask = cv2.bitwise_or(all_mask, out_mask)
                    mask_colored = np.zeros_like(left_image_rgb)
                    mask_colored[:, :, 1] = all_mask
                    left_image_rgb = cv2.addWeighted(left_image_rgb, 1, mask_colored, 0.5, 0)
            


                # Else use tracking
                else:
                    out_obj_ids, out_mask_logits = predictor.track(left_image_rgb)

                    all_mask = np.zeros((height, width, 1), dtype=np.uint8)
                    # print(all_mask.shape)
                    for i in range(0, len(out_obj_ids)):
                        out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                            np.uint8
                        ) * 255

                        all_mask = cv2.bitwise_or(all_mask, out_mask)

                    mask_colored = np.zeros_like(left_image_rgb)
                    mask_colored[:, :, 1] = all_mask
                    left_image_rgb = cv2.addWeighted(left_image_rgb, 1, mask_colored, 0.5, 0)
                
                te = time.time()
            print(f' Current FPS: {1/(te-ts)}' )
            cv2.imwrite('test.png', left_image_rgb)
            pass


    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Stop streaming and close the input source
        wrapper.stop_stream()
        wrapper.close_input_source()
        # cv2.destroyAllWindows()

        wrapper.stop_stream()
        wrapper.close_input_source()

