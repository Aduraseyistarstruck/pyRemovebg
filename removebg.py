# Background removal in Python can be achieved using various libraries, including OpenCV, Pillow, and scikit-image. Here's a general overview of the steps involved:
# Step 1: Install Required Libraries
# You'll need to install OpenCV and numpy using pip:
# ```
# bash
# pip install opencv-python numpy
# ```
# Step 2: Load the Image
# Use OpenCV to load the image:
# ```
# import cv2
# image = cv2.imread('image.jpg')
# ```
# Step 3: Convert to HSV Color Space
# Convert the image to HSV (Hue, Saturation, Value) color space:
# ```
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# ```
# Step 4: Define Background Color Range
# Define the range of HSV values for the background color:
# ```
# lower_background = (0, 0, 0)
# upper_background = (255, 255, 100)
# ```
# Step 5: Threshold the Image
# Threshold the HSV image to separate the background:
# ```
# mask = cv2.inRange(hsv_image, lower_background, upper_background)
# ```
# Step 6: Apply Morphological Operations
# Apply morphological operations to refine the mask:
# ```
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# ```
# Step 7: Remove Background
# Use the mask to remove the background:
# ```
# result = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
# ```
# Step 8: Save the Result
# Save the resulting image:
# ```
# cv2.imwrite('result.jpg', result)
# ```
# Here's the complete code:

# import cv2
# import numpy as np
# # Load the image
# image = cv2.imread('image1.jpg')
# # Convert to HSV color space
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# # Define background color range
# lower_background = (0, 0, 0)
# upper_background = (255, 255, 100)
# # Threshold the image
# mask = cv2.inRange(hsv_image, lower_background, upper_background)
# # Apply morphological operations
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# # Remove background
# result = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
# # Save the result
# cv2.imwrite('result.jpg', result)
# print("Done")




# # Here's a simple Python code using OpenCV and NumPy to remove the background from an image:

# import cv2
# import numpy as np
# # Load the image
# image = cv2.imread('image1.jpg')
# # Convert the image to HSV
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# # Define the lower and upper bounds for the background
# lower_background = np.array([0, 0, 0])
# upper_background = np.array([150, 150, 150])
# # Threshold the HSV image to get the background
# mask = cv2.inRange(hsv_image, lower_background, upper_background)
# # Invert the mask to get the foreground
# mask_inv = cv2.bitwise_not(mask)
# # Apply the mask to the original image
# foreground = cv2.bitwise_and(image, image, mask=mask_inv)
# # Save the foreground image
# cv2.imwrite('foreground.jpg', foreground)
# # ```
# # This code works by:
# # 1. Converting the image to HSV (Hue, Saturation, Value) color space.
# # 2. Defining the lower and upper bounds for the background in HSV.
# # 3. Thresholding the HSV image to get the background.
# # 4. Inverting the mask to get the foreground.
# # 5. Applying the mask to the original image to get the foreground.
# # You can adjust the `lower_background` and `upper_background` variables to change the background detection.
# # Note: This code assumes that the background is relatively uniform and can be detected using a simple threshold. For more complex backgrounds, you may need to use more advanced techniques, such as edge detection or deep learning-based methods.


# CODE TO CONVERT AN IMAGE TO JPEG format
# from PIL import Image
# import os

# def convert_to_jpeg(input_path, output_path=None):
#     """
#     Converts an image to JPEG format.

#     :param input_path: Path to the input image file.
#     :param output_path: Path to save the converted JPEG image. If None, saves in the same directory as input with .jpeg extension.
#     :return: Path to the saved JPEG image.
#     """
#     try:
#         # Open the input image
#         with Image.open(input_path) as img:
#             # Define the output path if not provided
#             if output_path is None:
#                 base, _ = os.path.splitext(input_path)
#                 output_path = f"{base}.jpeg"
            
#             # Convert and save the image as JPEG
#             img.convert('RGB').save(output_path, 'JPEG')
#             print(f"Image successfully converted to {output_path}")
#             return output_path
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None

# input_image_path = "image.jpg"
# convert_to_jpeg(input_image_path)


from rembg import remove
from PIL import Image
imagePath = input("Enter Image Name: ")
imageRemovedName = input("file should be saved as what?\n:")
inp = Image.open(imagePath)
output = remove(inp)
output.save(imageRemovedName)
Image.open("RemovedBg.png")
print("BACKGROUND REMOVED.")
#must also install onnxruntime i.e. pip install onnxruntime
#PIL is pip install pillow
#OSError: cannot write mode RGBA as JPEG is due to saving it as JPEG instead of png
#even .jpg file can be converted.