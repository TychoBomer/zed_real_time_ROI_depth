from utils.logger import Log
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
from utils.sam2prty import Sam2PromptType
sys.path.insert(0, os.getcwd())


# Initialize GPU settings
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()


from wrappers.pyzed_wrapper import pyzed_wrapper as pw
from utils.utils import *
from scripts.utils.depth_utils import (
    depth_refinement_RANSAC_plane_fitting,
    )


def run(cfg, sam2_prompt: Sam2PromptType, record: bool = False, svo_file: str = None) -> None:
    depth_refinement_enabled = cfg.depth.refine_depth
    caption_queue = queue.Queue()

    Log.info("Initializing the pipeline...", tag="pipeline_init")

    # Build needed models
    Log.info("Building models...", tag="building_models")
    wrapper = pw.Wrapper("svo" if svo_file else cfg.camera.connection_type)
    
    # If using playback, set SVO file path
    if svo_file:
        wrapper.input_parameters.svo_input_filename = svo_file

    try:
        wrapper.open_input_source()
        Log.info("ZED camera initialized.", tag="camera_init")
    except Exception as e:
        Log.error(f"Failed to initialize ZED camera: {e}", tag="camera_init_error")
        return

    # Start recording if enabled
    if record:
        wrapper.start_recording("./output/output.svo")
        Log.info("Recording started.", tag="recording")

    wrapper.start_stream()
    Log.info('Camera stream started.', tag = "camera_stream")

    framecount = 0

    try:
        while True:
            ts = time.time()

            if wrapper.retrieve(is_image=True, is_measure=True):
                # Extract images
                left_image = wrapper.output_image
                depth_map = wrapper.output_measure

                # norm_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_RGBA2RGB)

                # # Display and save
                # cv2.imshow("ZED Left", left_image_rgb)
                # cv2.imshow("Depth Map", norm_depth_map)

                cv2.imwrite(f"output/VID_WATCHER_.png", left_image_rgb)
                # cv2.imwrite(f"output/depth_{framecount}.png", norm_depth_map)

                if framecount > 1000:
                    break

                framecount += 1

    except Exception as e:
        Log.error(f"An error occurred: {e}", tag="runtime_error")

    finally:
        # Stop recording if enabled
        if record:
            wrapper.stop_recording()
            Log.info("Recording stopped.", tag="recording_stop")

        Log.info("Shutting down and closing stream...", tag='Close_stream')
        wrapper.stop_stream()
        wrapper.close_input_source()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ensure GlobalHydra is properly cleared before initializing
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize(config_path="../configurations"):
        cfg = compose(config_name="sam2_zed_small")
        sam2_prompt = Sam2PromptType('g_dino_bbox', user_caption='bottle')

    # Run after Hydra is fully initialized
    run(cfg, sam2_prompt, record=True)