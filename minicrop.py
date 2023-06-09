import cv2

# Load the images
img = cv2.imread('./0_frame0.jpg')
img_thermal = cv2.imread('./1_frame0.jpg')

# crop out a portion of both image and show them side by side
crop_width = 285  # Set the width of the crop
crop_height = 285  # Set the height of the crop
img_thermal_cropped = img_thermal[crop_height:-crop_height, crop_width:-crop_width]  # Crop the image
img_cropped = img[crop_height:-crop_height, crop_width:-crop_width]  # Crop the image

# Show the images
cv2.imshow('Cropped Thermal', img_thermal_cropped)
cv2.imshow('Cropped RGB', img_cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()