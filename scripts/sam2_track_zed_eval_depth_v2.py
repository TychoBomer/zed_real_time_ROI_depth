import os
import sys
import torch
import numpy as np
import cv2
from cv2 import aruco
import time
import csv
import threading
import queue
import matplotlib.pyplot as plt
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

sys.path.insert(0, os.getcwd())

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

    if corners is not None and len(corners) > 0 and marker_ids is not None:
        marker_size_m = marker_size_mm / 1000.0  # Convert mm to meters
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, marker_size_m, camera_matrix, dist_coeffs
        )

        # Ensure marker_ids is an iterable array
        if isinstance(marker_ids, np.ndarray):
            marker_ids = marker_ids.flatten()  # Convert shape (N,1) -> (N,)

        if len(marker_ids) != len(corners):
            Log.warn("⚠️ Mismatch in detected Aruco markers: corners and marker IDs have different lengths.", tag="aruco_depth")
            return marker_depths  # Return empty dictionary to avoid error

        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            if i < len(marker_ids):  # Prevent index out of range
                marker_id = marker_ids[i]  # Now it's always a valid integer
                z_depth_mm = tvec[0][2] * 1000  # Convert meters to mm
                marker_depths[marker_id] = z_depth_mm

    return marker_depths



def detect_aruco_markers(image):
    """
    Detects Aruco markers in an image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT)

    marker_data = {}

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

def calculate_rmse_mae(depth_errors):
    """
    Calculates RMSE and MAE for depth errors grouped by depth bin.
    """
    bins = np.arange(0, 3000, 100)  # Bin depths every 100mm up to 3000mm
    rmse_values, mae_values, z_values = [], [], []

    for i in range(len(bins) - 1):
        z_min, z_max = bins[i], bins[i + 1]
        bin_errors = [err for z, err in depth_errors if z_min <= z < z_max]

        if bin_errors:
            rmse = np.sqrt(np.mean(np.array(bin_errors) ** 2))
            mae = np.mean(np.abs(np.array(bin_errors)))
            rmse_values.append(rmse)
            mae_values.append(mae)
            z_values.append((z_min + z_max) / 2)  # Midpoint of bin

    return z_values, rmse_values, mae_values

def run(cfg, sam2_prompt: Sam2PromptType) -> None:
    """
    Runs the depth evaluation pipeline.
    """
    output_dir = setup_output_folder(cfg.results.output_dir)
    depth_evaluations = []
    depth_errors = []

    # Initialize ZED
    wrapper = pw.Wrapper(cfg.camera)
    wrapper.open_input_source()
    wrapper.start_stream()

    try:
        while True:
            if wrapper.retrieve(is_image=True, is_measure=True):
                left_image = wrapper.output_image
                depth_map = wrapper.output_measure
                d_min, d_max = np.min(depth_map[depth_map > 0]), np.max(depth_map)
                norm_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_RGBA2RGB)

                marker_positions, corners = detect_aruco_markers(left_image_rgb)

                if not marker_positions:
                    continue

                # Camera intrinsic parameters
                l_intr, _ = wrapper.get_intrinsic()
                camera_matrix = np.array([[l_intr.fx, 0, l_intr.cx], [0, l_intr.fy, l_intr.cy], [0, 0, 1]])
                dist_coeffs = np.zeros((5, 1))

                aruco_gt_depths = estimate_aruco_depth(left_image_rgb, corners, list(marker_positions.keys()), 10, camera_matrix, dist_coeffs)

                for marker_id, (x, y, _) in marker_positions.items():
                    if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
                        zed_depth = depth_map[y, x]
                        refined_depth = denormalize_depth(norm_depth_map, d_min, d_max)[y, x]
                        ground_truth_depth = aruco_gt_depths.get(marker_id)

                        if ground_truth_depth:
                            raw_error = abs(zed_depth - ground_truth_depth)
                            refined_error = abs(refined_depth - ground_truth_depth)

                            depth_evaluations.append([marker_id, ground_truth_depth, zed_depth, refined_depth, raw_error, refined_error])
                            depth_errors.append((ground_truth_depth, refined_error))

            else:
                break

        # Save CSV
        csv_filename = os.path.join(output_dir, "depth_comparison.csv")
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Marker ID", "GT Depth (mm)", "ZED Depth (mm)", "Refined Depth (mm)", "Raw Error (mm)", "Refined Error (mm)"])
            writer.writerows(depth_evaluations)

        # Compute RMSE and MAE
        z_values, rmse_values, mae_values = calculate_rmse_mae(depth_errors)
        rmse_csv = os.path.join(output_dir, "depth_error_analysis.csv")
        with open(rmse_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Depth Bin (mm)", "RMSE (mm)", "MAE (mm)"])
            for z, rmse, mae in zip(z_values, rmse_values, mae_values):
                writer.writerow([z, rmse, mae])

        Log.info(f"Saved RMSE and MAE to {rmse_csv}")

    finally:
        wrapper.stop_stream()
        wrapper.close_input_source()
        Log.info("Pipeline finished.")

    # Plot results
    plot_results(csv_filename, rmse_csv)

def plot_results(depth_csv, error_csv):
    """
    Reads CSV files and generates plots for depth error analysis.
    """
    z_vals, raw_errs, ref_errs = [], [], []
    rmse_z, rmse_vals, mae_vals = [], [], []

    # Read depth error CSV
    with open(depth_csv, "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            z_vals.append(float(row[1]))
            raw_errs.append(float(row[4]))
            ref_errs.append(float(row[5]))

    # Read RMSE & MAE CSV
    with open(error_csv, "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            rmse_z.append(float(row[0]))
            rmse_vals.append(float(row[1]))
            mae_vals.append(float(row[2]))

    # Plot depth errors with 'x' markers
    plt.figure()
    plt.scatter(z_vals, raw_errs, marker='x', label="Raw Error", color="red", alpha=0.5)
    # plt.scatter(z_vals, ref_errs, marker='x', label="Refined Error", color="blue", alpha=0.5)
    plt.xlabel("Depth (mm)")
    plt.ylabel("Error (mm)")
    plt.legend()
    plt.title("Depth Error vs. Z (Using 'x' Markers)")
    plt.show()

    # Plot RMSE and MAE with 'x' markers
    plt.figure()
    plt.plot(rmse_z, rmse_vals, marker='x', label="RMSE", color="purple")
    plt.plot(rmse_z, mae_vals, marker='x', label="MAE", color="green")
    plt.xlabel("Depth (mm)")
    plt.ylabel("Error (mm)")
    plt.legend()
    plt.title("RMSE & MAE vs. Z (Using 'x' Markers)")
    plt.show()

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