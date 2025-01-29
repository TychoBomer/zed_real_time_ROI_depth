
import numpy as np
import cv2
from deprecated import deprecated

@deprecated
def spatial_hole_filling(depth_map, mask, max_distance=5):
    """
    Perform spatial hole filling on the depth map using valid neighboring pixels.

    Args:
        depth_map (np.ndarray): Raw depth map (H x W).
        mask (np.ndarray): Binary mask for the segment (H x W).
        max_distance (int): Maximum distance for neighboring pixels.

    Returns:
        np.ndarray: Depth map with holes filled spatially.
    """
    filled_depth = depth_map.copy()
    missing_mask = (depth_map == 0).astype(np.uint8)

    # Perform inpainting only within the mask
    inpaint_mask = missing_mask & mask

    # Use OpenCV's inpainting method
    filled_depth = cv2.inpaint(filled_depth, inpaint_mask, inpaintRadius=max_distance, flags=cv2.INPAINT_TELEA)

    return filled_depth

@deprecated
def localized_weighted_median_filter(depth_map, guidance_img, mask, kernel_size=5):
    """
    Apply weighted median filtering only within the specified mask.

    Args:
        depth_map (np.ndarray): Depth map (H x W).
        guidance_img (np.ndarray): Guidance image for edge preservation (H x W).
        mask (np.ndarray): Binary mask (H x W) indicating the regions to refine.
        kernel_size (int): Kernel size for the median filter.

    Returns:
        np.ndarray: Depth map with localized filtering applied.
    """
    # Normalize inputs
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    guidance_img_normalized = cv2.normalize(guidance_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply joint bilateral filtering
    filtered_depth = cv2.ximgproc.jointBilateralFilter(
        joint=guidance_img_normalized,
        src=depth_map_normalized,
        d=-1,
        sigmaColor=25,
        sigmaSpace=kernel_size
    )

    # Combine original and filtered depth maps, applying changes only within the mask
    refined_depth = depth_map.copy()
    refined_depth[mask > 0] = filtered_depth[mask > 0]

    return refined_depth


@deprecated
def temporal_hole_filling(current_depth, previous_depth, alpha=0.5):
    """
    Fill holes in the current depth map using the previous depth map.

    Args:
        current_depth (np.ndarray): Current depth map (H x W).
        previous_depth (np.ndarray): Previous depth map (H x W).
        alpha (float): Weight for blending current and previous depth maps.

    Returns:
        np.ndarray: Depth map with temporal hole filling applied.
    """
    filled_depth = current_depth.copy()

    # Use the previous frame's depth values to fill missing values in the current frame
    missing_mask = (current_depth == 0).astype(np.uint8)
    filled_depth[missing_mask > 0] = alpha * previous_depth[missing_mask > 0] + (1 - alpha) * filled_depth[missing_mask > 0]

    return filled_depth

@deprecated
def refine_depth_with_hole_filling(current_depth, previous_depth, obj_masks, guidance_img, max_distance=5, alpha=0.5, kernel_size=5):
    """
    Refine the depth map using spatio-temporal hole filling and localized weighted median filtering.

    Args:
        current_depth (np.ndarray): Current depth map (H x W).
        previous_depth (np.ndarray): Previous depth map (H x W).
        obj_masks (dict): Dictionary of binary masks for each object.
        guidance_img (np.ndarray): Guidance image for edge preservation (H x W).
        max_distance (int): Maximum distance for spatial hole filling.
        alpha (float): Weight for blending current and previous depth maps.
        kernel_size (int): Kernel size for the weighted median filter.

    Returns:
        np.ndarray: Refined depth map.
    """
    refined_depth = current_depth.copy()

    # Step 1: Spatial Hole Filling for each object mask
    combined_mask = np.zeros_like(current_depth, dtype=np.uint8)
    for obj_id, mask in obj_masks.items():
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        obj_mask = mask.astype(np.uint8)

        if np.any(obj_mask > 0):
            refined_depth = spatial_hole_filling(refined_depth, obj_mask, max_distance=max_distance)
            combined_mask |= obj_mask  # Combine masks for localized filtering

    # Step 2: Temporal Hole Filling
    if previous_depth is not None:
        refined_depth = temporal_hole_filling(refined_depth, previous_depth, alpha=alpha)

    # Step 3: Localized Weighted Median Filtering
    refined_depth = localized_weighted_median_filter(refined_depth, guidance_img, combined_mask, kernel_size=kernel_size)

    return refined_depth



@deprecated
def hybrid_median_filter(depth_map, mask, kernel_size=5):
    """
    Apply hybrid median filtering to the depth map within a given mask.

    Args:
        depth_map (np.ndarray): Depth map (H x W).
        mask (np.ndarray): Binary mask for the segment (H x W).
        kernel_size (int): Size of the kernel for median filtering.

    Returns:
        np.ndarray: Depth map with hybrid median filtering applied within the mask.
    """
    # Extract depth values within the mask
    depth_within_mask = depth_map.copy()
    depth_within_mask[mask == 0] = 0  # Set values outside the mask to 0

    # Apply median filtering in different directions
    median_horizontal = cv2.medianBlur(depth_within_mask, ksize=kernel_size)
    median_vertical = cv2.medianBlur(depth_within_mask.T, ksize=kernel_size).T
    median_diagonal1 = cv2.medianBlur(cv2.rotate(depth_within_mask, cv2.ROTATE_90_CLOCKWISE), ksize=kernel_size)
    median_diagonal1 = cv2.rotate(median_diagonal1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    median_diagonal2 = cv2.medianBlur(cv2.rotate(depth_within_mask, cv2.ROTATE_90_COUNTERCLOCKWISE), ksize=kernel_size)
    median_diagonal2 = cv2.rotate(median_diagonal2, cv2.ROTATE_90_CLOCKWISE)

    # Combine the median values within the mask
    refined_depth = np.zeros_like(depth_map)
    refined_depth[mask > 0] = np.median(
        np.stack([median_horizontal, median_vertical, median_diagonal1, median_diagonal2], axis=-1),
        axis=-1
    )[mask > 0]

    return refined_depth

@deprecated
def refine_depth_with_mhmf(depth_map, obj_masks, depth_threshold=(1, 255)):
    """
    Apply guided filtering to the depth map for each object mask.

    Args:
        depth_map (np.ndarray): Input depth map.
        guidance_img (np.ndarray): Guidance image (can be the mask or an RGB image).
        obj_masks (dict): Dictionary of binary masks for each object.

    Returns:
        np.ndarray: Refined depth map with filtering applied to each mask region.
    """
    # Ensure depth map is uint8
    depth_map = depth_map.astype(np.uint8)

    # Handle zero values in the depth map (treat zero as invalid depth)
    zero_mask = (depth_map == 0)  # Find zero values in the depth map
    depth_map_filled = np.copy(depth_map)

    # Inpaint zero values globally
    if np.any(zero_mask):
        inpaint_radius = 5
        depth_map_filled = cv2.inpaint(depth_map_filled, zero_mask.astype(np.uint8), inpaint_radius, cv2.INPAINT_TELEA)

    # Normalize the guidance image
    if guidance_img.ndim == 3:  # RGB image
        guidance_img = guidance_img.astype(np.float32) / 255.0
    elif guidance_img.ndim == 2:  # Grayscale image
        guidance_img = cv2.cvtColor(guidance_img, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0

    # Initialize refined depth map
    refined_depth = np.copy(depth_map_filled)

    # Process each mask individually
    for obj_id, mask in obj_masks.items():
        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Normalize mask
        mask = mask.astype(np.float32) / 255.0

        # Filtered region
        filtered_region = np.copy(depth_map_filled)
        filtered_region[mask == 0] = 0  # Set pixels outside the mask to zero

        # Apply guided filter locally to the region inside the mask
        refined_region = cv2.ximgproc.guidedFilter(guide=guidance_img, src=filtered_region, radius=6, eps=1e-4)

        # Update refined depth map only within the mask
        refined_depth = np.where(mask > 0, refined_region, refined_depth)

    return refined_depth


@deprecated
def compute_slope_compensation(depth_map, mask):
    """
    Apply slope depth compensation to handle missing or noisy depth values.

    Args:
        depth_map (np.ndarray): Input depth map (H x W).
        mask (np.ndarray): Binary mask for the region of interest (H x W).

    Returns:
        np.ndarray: Depth map after slope depth compensation.
    """
    # Compute depth gradients
    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
    slope = np.sqrt(grad_x**2 + grad_y**2)

    # Limit slope values to prevent extreme changes
    slope = np.clip(slope, 0, np.percentile(slope[mask > 0], 95))

    # Fill missing depth values based on slope
    filled_depth = depth_map.copy()
    missing = (depth_map == 0) & (mask > 0)
    filled_depth[missing] = slope[missing]

    return filled_depth

@deprecated
def refine_depth_with_wjbf_and_sdcf(depth_map, obj_masks, guidance_img, sigma_spatial=15, sigma_range=30):
    """
    Refine the depth map using Weighted Joint Bilateral Filter and Slope Depth Compensation.

    Args:
        depth_map (np.ndarray): Raw depth map (H x W).
        obj_masks (dict): Dictionary of binary masks for each object.
        guidance_img (np.ndarray): Guidance image for the joint bilateral filter.
        sigma_spatial (float): Spatial sigma for joint bilateral filter.
        sigma_range (float): Range sigma for joint bilateral filter.

    Returns:
        np.ndarray: Refined depth map.
    """
    refined_depth_map = depth_map.copy()

    # Normalize depth map and guidance image
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    guidance_img_normalized = cv2.normalize(guidance_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Process each segmentation mask
    for obj_id, mask in obj_masks.items():
        # Ensure the mask is binary
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        obj_mask = mask.astype(np.uint8)

        # Apply slope depth compensation
        compensated_depth = compute_slope_compensation(depth_map_normalized, obj_mask)

        # Weighted Joint Bilateral Filter
        if np.any(obj_mask > 0):  # Only process if the mask has valid regions
            joint_bilateral_filtered = cv2.ximgproc.jointBilateralFilter(
                joint=guidance_img_normalized,  # Guidance image
                src=compensated_depth,         # Source depth map
                d=-1,                          # Automatically determine kernel size
                sigmaColor=sigma_range,        # Range sigma for joint bilateral filter
                sigmaSpace=sigma_spatial       # Spatial sigma for joint bilateral filter
            )

            # Merge the refined region back into the depth map
            refined_depth_map[obj_mask > 0] = joint_bilateral_filtered[obj_mask > 0]

    # Convert back to original depth map scale
    refined_depth_map = cv2.normalize(refined_depth_map, None, depth_map.min(), depth_map.max(), cv2.NORM_MINMAX)

    return refined_depth_map

@deprecated
def refine_depth_with_cross_bilateral(depth_map, obj_masks, guidance_img, sigma_spatial=15, sigma_range=30):
    """
    Refine the depth map using localized Cross-Bilateral Filtering for each segmentation mask.

    Args:
        depth_map (np.ndarray): Raw depth map (H x W).
        obj_masks (dict): Dictionary of binary masks for each object.
        guidance_img (np.ndarray): Guidance image for the joint bilateral filter.
        sigma_spatial (float): Spatial sigma for bilateral filter.
        sigma_range (float): Range sigma for bilateral filter.

    Returns:
        np.ndarray: Refined depth map.
    """
    # Normalize depth map and guidance image
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    guidance_img_normalized = cv2.normalize(guidance_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Initialize the refined depth map with the original depth values
    refined_depth_map = depth_map.copy()

    # Process each segmentation mask
    for obj_id, mask in obj_masks.items():
        # Ensure the mask is binary and 2D
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        obj_mask = mask.astype(np.uint8)

        if np.any(obj_mask > 0):  # Process only if the mask has valid regions
            # Extract depth values within the mask
            depth_within_mask = depth_map_normalized * (obj_mask > 0)

            # Apply cross-bilateral filtering only within the mask
            refined_region = cv2.ximgproc.jointBilateralFilter(
                joint=guidance_img_normalized,
                src=depth_within_mask,
                d=-1,
                sigmaColor=sigma_range,
                sigmaSpace=sigma_spatial
            )

            # Merge the refined region back into the depth map
            refined_depth_map[obj_mask > 0] = refined_region[obj_mask > 0]

    # Convert back to the original depth map scale
    refined_depth_map = cv2.normalize(refined_depth_map, None, depth_map.min(), depth_map.max(), cv2.NORM_MINMAX)

    return refined_depth_map

@deprecated
def preprocess_depth_map(depth_map, kernel_size=5):
    # Fill small gaps by median filtering
    filled_depth_map = cv2.medianBlur(depth_map, kernel_size)
    return filled_depth_map


@deprecated
def inpaint_depth_map(depth_map, inpaint_radius=3):
    # Create a mask where depth is zero or NaN
    mask = (depth_map == 0).astype(np.uint8)
    # Use inpainting to fill zero or NaN values
    inpainted_depth_map = cv2.inpaint(depth_map, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return inpainted_depth_map

@deprecated
def refine_depth_with_postprocessing(depth_map, left_image, obj_masks, lmbda=8000, sigma=1.5, inpaint_radius=3):
    """
    Refine depth map by combining inpainting, WLS filtering, and mask-guided blending for each object mask.

    Parameters:
        depth_map (np.ndarray): Input depth map.
        left_image (np.ndarray): Left RGB image from the stereo pair (used as guidance).
        obj_masks (dict): Dictionary of binary masks for each object.
        lmbda (float): Regularization parameter for WLS filter.
        sigma (float): Smoothness parameter for WLS filter.
        inpaint_radius (int): Radius for inpainting gaps in the depth map.

    Returns:
        np.ndarray: Refined depth map.
    """
    # Step 1: Inpaint large missing regions
    depth_map_inpainted = inpaint_depth_map(depth_map, inpaint_radius)

    # Step 2: Convert to 16-bit for WLS filtering
    depth_map_inpainted = (depth_map_inpainted * 16).astype(np.int16)

    # Step 3: Apply WLS filtering for edge-preserving refinement
    wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    refined_depth = wls_filter.filter(depth_map_inpainted, left_image)
    refined_depth = refined_depth.astype(np.float32) / 16.0  # Normalize back

    # Step 4: Apply mask-guided blending for each object mask
    final_depth_map = np.copy(depth_map)  # Start with the original depth map

    for obj_id, mask in obj_masks.items():
        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Normalize mask to [0, 1]
        mask = mask.astype(np.float32) / 255.0

        # Blend refined depth map with original depth map using the mask
        final_depth_map = final_depth_map * (1 - mask) + refined_depth * mask

    return final_depth_map

@deprecated
def mask_guided_filter(depth_map: np.ndarray, guidance_img: np.ndarray, obj_masks: dict) -> np.ndarray:    
    """
    Refine the depth map using the Multistage Hybrid Median Filter (MHMF) method.

    Args:
        depth_map (np.ndarray): Raw depth map (H x W).
        obj_masks (dict): Dictionary of binary masks for each object.
        depth_threshold (tuple): Min and max depth values to filter.

    Returns:
        np.ndarray: Refined depth map.
    """
    refined_depth_map = depth_map.copy()

    # Process each segment (mask)
    for obj_id, mask in obj_masks.items():
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        obj_mask = mask.astype(np.uint8)

        # Combine mask with depth threshold
        valid_depth = (depth_map > depth_threshold[0]) & (depth_map < depth_threshold[1])
        combined_mask = obj_mask & valid_depth.astype(np.uint8)

        if np.any(combined_mask > 0):  # Process only if the mask has valid regions
            print(f"Processing Object {obj_id} - Mask Non-Zero Count: {np.count_nonzero(combined_mask)}")

            # Apply hybrid median filtering within the combined mask
            refined_segment = hybrid_median_filter(refined_depth_map, combined_mask, kernel_size=5)

            # Merge the refined segment into the final depth map
            refined_depth_map[combined_mask > 0] = refined_segment[combined_mask > 0]

    return refined_depth_map