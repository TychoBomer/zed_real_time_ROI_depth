import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import concurrent.futures
import torch
from typing import Dict
from utils.logger import Log



def weighted_median_filter(depth_map: np.ndarray, mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    filtered = cv2.medianBlur(depth_map.astype(np.float32), kernel_size)
    return np.where(mask > 0, filtered, depth_map)

def fit_plane_to_segment(depth_map: np.ndarray, mask: np.ndarray, max_occ_percentage) -> np.ndarray:
    """
    Fit a plane to the depth values within a given segment using RANSAC.

    Args:
        depth_map (np.ndarray): Depth map (H x W).
        mask (np.ndarray): Binary mask for the segment (H x W).

    Returns:
        np.ndarray: Depth map with plane fitted values within the mask.
    """
    num_holes = np.count_nonzero((depth_map == 0) & (mask > 0))
    total_pixels = np.count_nonzero(mask)
    hole_percentage = num_holes / total_pixels if total_pixels > 0 else 1.0

    if hole_percentage > max_occ_percentage:
        Log.warn(f"Skipping segment: too much holes")
        return depth_map  # Not enough data to fit a plane


    y, x = np.where(mask > 0)
    z = depth_map[y, x]
    # Exclude invalid depth values
    valid = z > 0
    x, y, z = x[valid], y[valid], z[valid]

    if len(z) < 3:
        Log.warn(f"Skipping segment: Not enough valid points.")
        return depth_map  #

    coords = np.column_stack((x, y))

    if hole_percentage> 0.7:
        Log.info(f"Using RANSAC plane fitting. Occlusion Ratio: {hole_percentage:.2f}")


        # Fit a plane using RANSAC
        poly = PolynomialFeatures(degree=1)
        coords_poly = poly.fit_transform(coords)  # Polynomial features for plane fitting
        model = RANSACRegressor(LinearRegression(), residual_threshold=2.0)
        model.fit(coords_poly, z)

    else: 
        # Weighted Least Squares Estimation (LSE) when occlusion ratio > 0.7
        Log.info(f"Using Weighted LSE for plane fitting. Occlusion Ratio: {hole_percentage:.2f}")
        poly = PolynomialFeatures(degree=1)
        coords_poly = poly.fit_transform(coords)

        # Assign weights: More weight for valid points, less for occluded ones
        weights = np.ones(len(z))
        weights[z == 0] = 0.1  # Occluded points get a low weight

        model = LinearRegression()
        model.fit(coords_poly, z, sample_weight=weights)  # Weighted fit


    # Predict depth values for all pixels in the segment
    H, W = depth_map.shape
    full_x, full_y = np.meshgrid(np.arange(W), np.arange(H))
    full_coords = np.column_stack((full_x.ravel(), full_y.ravel()))
    full_coords_poly = poly.transform(full_coords)
    
    fitted_depth = model.predict(full_coords_poly).reshape(H, W)

    # Apply median filtering to refine depth
    fitted_depth = weighted_median_filter(fitted_depth, mask, kernel_size=5)

    # add segment to original depth map
    refined_segment = np.zeros_like(depth_map)
    refined_segment[mask > 0] = fitted_depth[mask > 0]

    return refined_segment

def depth_refinement_RANSAC_plane_fitting(depth_map: np.ndarray, obj_masks: Dict[int, np.ndarray], max_occ_percentage: float = 0.7) -> np.ndarray:
    """
    Multi-threaded depth refinement using RANSAC-based plane fitting. The following
    papers: http://www.apsipa.org/proceedings_2012/papers/302.pdf
            https://www.honda-ri.de/pubs/pdf/1938.pdf
            https://www.researchgate.net/publication/322812111_Disparity_refinement_process_based_on_RANSAC_plane_fitting_for_machine_vision_applications

    Args:
        depth_map (np.ndarray): Original disparity map (H x W).
        obj_masks (dict): Dictionary of binary masks for each detected object.

    Returns:
        np.ndarray: Refined depth map.
    """
    refined_depth_map = depth_map.copy()

    # Process masks in parallel using threading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}

        for obj_id, mask in obj_masks.items():
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            obj_mask = mask.astype(np.uint8)

            if np.count_nonzero(obj_mask) > 100:  # Process only significant objects
                futures[executor.submit(fit_plane_to_segment, depth_map, obj_mask, max_occ_percentage = max_occ_percentage)] = obj_id

    # Collect results
    for future in concurrent.futures.as_completed(futures):
        obj_id = futures[future]
        refined_segment = future.result()
        obj_mask = obj_masks[obj_id]
        if obj_mask.ndim == 3:
            obj_mask = obj_mask[:, :, 0]
        refined_depth_map[obj_mask > 0] = refined_segment[obj_mask > 0]

    return refined_depth_map



def visualize_depth(depth: np.ndarray, 
                    depth_min=None, 
                    depth_max=None, 
                    percentile=2, 
                    ret_minmax=False,
                    cmap='Spectral'):
    if depth_min is None: depth_min = np.percentile(depth, percentile)
    if depth_max is None: depth_max = np.percentile(depth, 100 - percentile)
    if depth_min == depth_max:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
    img_colored_np = cm(depth[None], bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = (img_colored_np[0] * 255.0).astype(np.uint8)
    if ret_minmax:
        return img_colored_np, depth_min, depth_max
    else:
        return img_colored_np
    
def to_numpy_func(tensor):
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr

def resize_to_multiple(image, multiple):
    H, W, _ = image.shape if len(image.shape) == 3 else image.shape + (1,)
    new_H = (H + multiple - 1) // multiple * multiple
    new_W = (W + multiple - 1) // multiple * multiple
    resized = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    return resized
