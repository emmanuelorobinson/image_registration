import os

import cv2
import numpy as np
from PIL import Image




VIDEO_FOLDER = './DJI Dataset/'
CAPTURE_FOLDER = './Capture/'
RESULTS_FOLDER = './Results/'
CROP_FOLDER = './Crop/'

# check if those folders exist
if not os.path.exists(VIDEO_FOLDER):
    os.makedirs(VIDEO_FOLDER)
if not os.path.exists(CAPTURE_FOLDER):
    os.makedirs(CAPTURE_FOLDER)
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)
if not os.path.exists(CROP_FOLDER):
    os.makedirs(CROP_FOLDER)


#0_frame0.jpg = RGB Image
#1_frame0.jpg = Thermal Image with a cropo and black side bars

# place the thermal image on the RGB image and align them
import cv2

def cropAndFit(img, img_thermal, i):
    # # Load the images
    # img = cv2.imread('Capture/0_frame260.jpg')
    # img_thermal = cv2.imread('Capture/1_frame260.jpg')

    # crop out the black side bars from the thermal image

    crop_width = 285  # Set the width of the crop
    img_cropped = img_thermal[:, crop_width:-crop_width]  # Crop the image
    # save the cropped image
    # cv2.imwrite('./1_frame0.jpg', img_cropped)
    cv2.imwrite('Crop/1_frame%d.jpg' % i, img_cropped)

    # print size of thermal image
    print(img_cropped.shape)

    # zoom in to the rgb image

    zoom_factor = 1.52  # Set the zoom factor
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    
    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 1.9, max(0, new_width - width) // 1.95
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]
    
    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)
    
    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    # assert result.shape[0] == height and result.shape[1] == width
    assert result.shape[0] == height and result.shape[1] == width

    crop_width = 285  # Set the width of the crop
    result = result[:, crop_width:-crop_width]  # Crop the image
    print(result.shape)
    # save the zoomed in image
    # cv2.imwrite('./0_frame0.jpg', result)
    cv2.imwrite('Crop/0_frame%d.jpg' % i, result)


    # background = Image.open('./0_frame0.jpg')
    background = Image.open('Crop/0_frame%d.jpg' % i)
    # overlay = Image.open('./1_frame0.jpg')
    overlay = Image.open('Crop/1_frame%d.jpg' % i)

    background.convert('RGBA')
    overlay.convert('RGBA')

    if overlay.mode != 'RGBA':
        overlay = overlay.convert('RGBA')

    # reduce opacity of overlay
    alpha = 0.5
    overlay.putalpha(int(255 * alpha))
    # overlay.show()


    background.paste(overlay, (0,0), overlay)
    # background.save('./final.jpg', 'JPEG', quality=100)
    background.save('Results/final%d.jpg' % i, 'JPEG', quality=100)



# get the number of img in capture folder
num_img = len([name for name in os.listdir(CAPTURE_FOLDER) if os.path.isfile(os.path.join(CAPTURE_FOLDER, name))])
num_img_pair = num_img // 2

# # get the image pair
countB = 0

# for i in range(num_img_pair):
#     img = cv2.imread(CAPTURE_FOLDER + '0_frame%d.jpg' % i)
#     img_thermal = cv2.imread(CAPTURE_FOLDER + '1_frame%d.jpg' % i)
#     cropAndFit(img, img_thermal, i)

#create a d video from the images in the results folder
import cv2
import numpy as np
import glob


# Get a list of all the image file names in the Results folder
img_names = os.listdir('Results')

# Sort the images by name index
img_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

# Load all the images from the Results folder
img_array = []
for filename in img_names:
    img = cv2.imread('Results/' + filename)
    img_array.append(img)

# Get the dimensions of the first image
height, width, layers = img_array[0].shape

# Create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('output.mp4', fourcc, 30, (width, height))

# Write each image to the video writer object
for i in range(len(img_array)):
    video.write(img_array[i])

# Release the video writer object and close all windows
video.release()
cv2.destroyAllWindows()


