import os, sys
import torch
import numpy as np
import random
import cv2
from cv2 import aruco
import time
import csv
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

from wrappers.pyzed_wrapper import pyzed_wrapper_v2 as pw
from scripts.utils.utils import *
from scripts.utils.depth_utils import depth_refinement_RANSAC_plane_fitting
from utils.logger import Log
from utils.sam2prty import Sam2PromptType

# Define Aruco dictionary
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

def estimate_aruco_depth(image, corners, marker_ids, marker_size_mm, camera_matrix, dist_coeffs):
    """
    Estimates the real-world depth of Aruco markers using pose estimation.
    """
    marker_depths = {}

    if corners is not None and len(corners) > 0:
        marker_size_m = marker_size_mm / 1000.0  # Convert mm to meters
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, marker_size_m, camera_matrix, dist_coeffs
        )

        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            marker_id = marker_ids[i]  # Use actual Aruco ID from detection
            z_depth_mm = tvec[0][2] * 1000  # Convert meters to mm
            marker_depths[marker_id] = z_depth_mm

            Log.info(f"📏 Aruco Marker {marker_id}: Estimated Depth = {z_depth_mm:.2f} mm", tag="aruco_depth")

    return marker_depths

def detect_aruco_markers(image):
    """
    Detects Aruco markers and estimates their size based on pixel distance.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT)

    marker_data = {}  # {marker_id: (center_x, center_y)}

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            c = corners[i][0]
            center_x = int(np.mean(c[:, 0]))
            center_y = int(np.mean(c[:, 1]))
            marker_data[marker_id] = (center_x, center_y, 20)  # Assume 20mm marker size

    return marker_data, corners

def denormalize_depth(normalized_depth, d_min, d_max):
    """
    Converts a normalized depth map (0-255) back to millimeters.
    """
    return normalized_depth.astype(np.float32) * (d_max - d_min) / 255.0 + d_min

def run(cfg, sam2_prompt: Sam2PromptType) -> None:
    depth_refinement_enabled = cfg.depth.refine_depth
    caption_queue = queue.Queue()

    Log.info("Initializing the pipeline...", tag="pipeline_init")
    caption_thread = threading.Thread(target=update_caption, args=(caption_queue, sam2_prompt.user_caption), daemon=True)
    caption_thread.start()

    # Build models
    Log.info("Building models...", tag="building_models")
    try:
        sam2_predictor, grounding_dino_model = build_models(cfg)
        Log.info("Models successfully built and loaded.", tag="model_building")
    except Exception as e:
        Log.error(f"Failed to build models: {e}", tag="model_build_error")
        return
    
    # Initialize ZED camera
    Log.info("Initializing ZED camera...", tag="zed_camera_init")
    wrapper = pw.Wrapper(cfg.camera)
    try:
        wrapper.open_input_source()
        Log.info("ZED camera initialized.", tag="camera_init")
    except Exception as e:
        Log.error(f"Failed to initialize ZED camera: {e}", tag="camera_init_error")
        return

    output_dir = setup_output_folder(cfg.results.output_dir)
    wrapper.start_stream()
    Log.info('Camera stream started.', tag="camera_stream")

    # Define VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{output_dir}/output.mp4", fourcc, 20, (1280, 720), True)
    out2 = cv2.VideoWriter(f"{output_dir}/output_depth.mp4", fourcc, 20, (1280, 720), False)

    framecount = 0
    depth_evaluations = []  # Store depth comparisons for CSV

    ## MAIN LOOP
    try:
        while True:
            ts = time.time()

            if wrapper.retrieve(is_image=True, is_measure=True):
                left_image = wrapper.output_image
                depth_map = wrapper.output_measure
                d_min = np.min(depth_map[depth_map > 0])  # Exclude invalid depths (0)
                d_max = np.max(depth_map)

                norm_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_RGBA2RGB)

                # Get Aruco marker detections
                marker_positions, corners = detect_aruco_markers(left_image_rgb)

                if not marker_positions:
                    Log.warn("⚠️ No Aruco markers detected in this frame.", tag="aruco_detection")
                    marker_positions, corners = {}, None  # Assign empty values to avoid errors
                else:
                    Log.info("✅ Aruco markers detected in this frame.", tag="aruco_detection")

                # Define Camera Matrix (Intrinsic Parameters)
                l_intr, _ = wrapper.get_intrinsic()
                fx, fy, cx, cy = l_intr.fx, l_intr.fy, l_intr.cx, l_intr.cy
                camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion
                
                aruco_ground_truth_depths = {}

                # Estimate Aruco Depth (Ground Truth)
                if corners:
                    marker_ids = list(marker_positions.keys())  # Convert dict_keys to a list
                    aruco_ground_truth_depths = estimate_aruco_depth(left_image_rgb, corners, marker_ids, 20, camera_matrix, dist_coeffs)

                # Retrieve ZED depth at Aruco locations
                if marker_positions:
                    for marker_id, (x, y, marker_size) in marker_positions.items():
                        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:  # Ensure within bounds
                            zed_depth = depth_map[y, x]  # ZED raw depth
                            refined_depth = denormalize_depth(norm_depth_map, d_min, d_max)[y, x]  # Convert refined depth back to mm
                            ground_truth_depth = aruco_ground_truth_depths.get(marker_id, None)  # Aruco GT depth

                            if ground_truth_depth:
                                raw_error = abs(zed_depth - ground_truth_depth)
                                refined_error = abs(refined_depth - ground_truth_depth)

                                Log.info(f"🔍 Marker {marker_id}: Aruco Depth={ground_truth_depth:.2f}mm, ZED Depth={zed_depth:.2f}mm, Refined Depth={refined_depth:.2f}mm, Raw Error={raw_error:.2f}mm, Refined Error={refined_error:.2f}mm", tag="depth_comparison")

                                # Store results
                                depth_evaluations.append([marker_id, marker_size, ground_truth_depth, zed_depth, refined_depth, raw_error, refined_error])

                if depth_evaluations:
                    csv_filename = os.path.join(output_dir, "depth_comparison.csv")
                    with open(csv_filename, "w", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["Marker ID", "Size (mm)", "Aruco Depth (mm)", "ZED Depth (mm)", "Refined Depth (mm)", "Raw Error (mm)", "Refined Error (mm)"])
                        writer.writerows(depth_evaluations)

                    Log.info(f"✅ Depth comparison saved to {csv_filename}", tag="csv_output")

            else:
                break

    except Exception as e:
        Log.error(f"An error occurred: {e}", tag="runtime_error")

    finally:
        Log.info("Shutting down and closing stream...", tag="Close_stream")
        wrapper.stop_stream()
        wrapper.close_input_source()



# Main function using Hydra to load config
if __name__ == "__main__":
    if GlobalHydra.is_initialized:
        GlobalHydra.instance().clear()

    with initialize(config_path="../configurations"):
        cfg = compose(config_name="sam2_zed_small")
        sam2_prompt = Sam2PromptType('g_dino_bbox',user_caption='apple')
        

        # point_coords = [(390, 200)]
        # labels = [1]  # 1 = foreground, 0 = background
        # sam2_prompt = Sam2PromptType('point', point_coords = point_coords, labels=labels)

        # bbox_coords = [(320, 120, 470, 280)]
        # bbox_coords = [(50, 50, 150, 150), (200, 200, 300, 300)] #! NOTE: 3+ boxes make it really inaccurate
        # sam2_prompt = Sam2PromptType('bbox', bbox_coords = bbox_coords)

        run(cfg, sam2_prompt=sam2_prompt)

