import numpy as np
import cv2
import sys,os
import time

sys.path.insert(0, os.getcwd())
from wrappers.pyzed_wrapper import pyzed_wrapper as pw

from fast_guided_filter import guided_filter as gf


def refine_depth_map(depth_map, mask, radius=2, eps=1e-3, smooth_edges=True, subsample = 5):
    """
    Refine the depth map only within the masked region using guided filtering, with optional mask smoothing.
    
    Parameters:
    - depth_map: np.ndarray, the original depth map.
    - mask: np.ndarray, the segmentation mask.
    - radius: int, the radius of the guided filter.
    - eps: float, regularization parameter for the guided filter.
    - smooth_edges: bool, whether to smooth the mask edges.
    
    Returns:
    - refined_depth_map: np.ndarray, depth map where only masked regions are refined.
    """

    # Normalize the mask to [0, 1] if necessary
    if mask.max() > 1:
        mask = mask.astype(np.float32) / 255.0

    # Optional: Smooth the mask to handle edge artifacts
    # if smooth_edges:
    #     kernel_size = (3, 3)  # Adjust this for more or less smoothing
    #     mask = cv2.GaussianBlur(mask, kernel_size, sigmaX=3)

    # Apply dilation to cover more edge pixels and reduce edge artifacts
    mask_dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    # Apply guided filtering to the masked (dilated) region only
    refined_region = cv2.ximgproc.guidedFilter(guide=mask_dilated, src=depth_map, radius=radius, eps=eps)
    # refined_region = gf(I= mask, p = depth_map, r = radius, eps=eps, s=subsample)

    # Keep the original depth map values for regions outside the mask
    refined_depth_map = np.where(mask > 0, refined_region, depth_map)

    return refined_depth_map

def real_time_image_and_depth(input_type):

    # Create an instance of the Wrapper class
    wrapper = pw.Wrapper(input_type)
    # Open the input source (camera, stream, or file)
    wrapper.open_input_source()

    # Extracting intrinsics
    l_intr, r_intr = wrapper.get_intrinsic()
    K_l = np.array([[l_intr.fx, 0, l_intr.cx],
                  [0, l_intr.fy, l_intr.cy],
                  [0, 0, 1]])

    from scripts.wrappers.all_sam_wrapper import AllSAMWrapper
    # sam_wrap = AllSAMWrapper("MobileSAM")
    # sam_wrap = AllSAMWrapper("EdgeSAM")
    # sam_wrap = AllSAMWrapper("RepViTSAM")
    sam_wrap = AllSAMWrapper('LightHQSAM')


    wrapper.start_stream()

    try:
        while True:
            # Retrieve a frame (image and depth)
            if wrapper.retrieve(is_image=True, is_measure=True):
                ts = time.time()
                left_image = wrapper.output_image
                depth_map = wrapper.output_measure
                norm_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                # depth_edges = cv2.Canny(norm_depth_map, threshold1=5, threshold2=80)
                


                # Convert left image to RGB format for displaying
                left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_RGBA2RGB)
                # Predict using dino+mobilesam
                annotated_image, binary_mask, annotated_frame = sam_wrap.predict(left_image_rgb)
                # annotated_image = sam_wrap.predict_everything(left_image_rgb)
                te = time.time()

                cv2.imwrite('annotated_image.jpg', annotated_image)
                cv2.imwrite('annotated_frame.jpg', annotated_frame)
                cv2.imwrite('binarymask.jpg', binary_mask)
                cv2.imwrite('norm_depth_map.jpg',norm_depth_map)

                # cv2.imshow('annotated image',annotated_image)

                # Apply binary mask to the normalized depth map
                masked_depth_map = cv2.bitwise_and(norm_depth_map, norm_depth_map, mask=binary_mask)
                cv2.imwrite(f'masked_depth_map.jpg', masked_depth_map)

                refined_depth_map = refine_depth_map(norm_depth_map, binary_mask)
                cv2.imwrite(f'refined_depth_map.jpg', refined_depth_map)


                # Exit if 'q' is pressed
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                print(f' time needed: {te-ts} seconds')
            else:
                print("Failed to retrieve frame from input source")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Stop streaming and close the input source
        wrapper.stop_stream()
        wrapper.close_input_source()
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    # Choose the input type (id, serial, svo, or stream)
    input_type : str = "id"  # Change this to the desired input type
    # weights_path : str = '/home/nakama/Documents/TychoMSC/main_project2/fastsam/weights/FastSAM-x.pt'
    # Start real-time extraction and display
    real_time_image_and_depth(input_type)