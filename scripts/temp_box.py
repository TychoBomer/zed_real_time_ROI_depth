
import cv2

# Load the image
image_path = "/home/nakamalab/Documents/zed_real_time_ROI_depth/output_img/left_img_og.png" 
output_path = image_path+'_box.png'  # Replace with the desired output path
image = cv2.imread(image_path)

# Define the bounding box coordinates (x1, y1, x2, y2)
bbox_coords = (250, 150, 500, 250)  # Replace with your desired bounding box coordinates

# Draw the bounding box
# Parameters: (image, (x1, y1), (x2, y2), color (BGR), thickness)
cv2.rectangle(image, (bbox_coords[0], bbox_coords[1]), (bbox_coords[2], bbox_coords[3]), (0, 255, 0), 2)

# Save the image with the bounding box
cv2.imwrite(output_path, image)

print(f"Image with bounding box saved at {output_path}")
