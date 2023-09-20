import cv2
import numpy as np
import os

# Path to the directory containing the images
image_directory = r"C:\Users\Muyeed\Desktop\Python\dataset"
output_directory = r"C:\Users\Muyeed\Desktop\Python\output_yellow_plates"  # Path to output folder

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith(".jpg")]

# Initialize a count for detected yellow number plates
total_yellow_plate_count = 0

# Loop through each image file
for image_file in image_files:
    # Load an image
    image_path = os.path.join(image_directory, image_file)
    image = cv2.imread(image_path)

    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create a mask for yellow regions
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours in the yellow mask
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through detected contours
    for contour in contours:
        # Check if the contour area is reasonable for a number plate
        area = cv2.contourArea(contour)

        if area > 1000:  # Adjust the area threshold based on your images
            # Fit a rotated bounding box to the contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # Calculate the aspect ratio of the bounding box
            width = np.linalg.norm(box[0] - box[1])
            height = np.linalg.norm(box[1] - box[2])
            aspect_ratio = max(width, height) / min(width, height)

            # Filter out contours with extreme aspect ratios
            if 0.5 < aspect_ratio < 2.5:  # Adjust the aspect ratio range based on your images
                # Save the yellow plate region as a separate image
                mask = np.zeros_like(yellow_mask)
                cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)
                yellow_plate = cv2.bitwise_and(image, image, mask=mask)
                output_path = os.path.join(output_directory, f"yellow_plate_{total_yellow_plate_count}.jpg")
                cv2.imwrite(output_path, yellow_plate)
                total_yellow_plate_count += 1

    print(f"Yellow number plates detected in {image_file}: {total_yellow_plate_count}")

# Print the total yellow number plates detected across all images
print("Total yellow number plates detected in all images:", total_yellow_plate_count)
