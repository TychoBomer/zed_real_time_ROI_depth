import numpy as np
import cv2
import sys,os
import time

sys.path.insert(0, os.getcwd())
from wrappers.pyzed_wrapper import pyzed_wrapper as pw



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