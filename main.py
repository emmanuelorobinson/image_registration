import os

import cv2
import numpy as np


VIDEO_FOLDER = './DJI Dataset/'
CAPTURE_FOLDER = './Capture/'


# # clear the capture folder
# for file in os.listdir(CAPTURE_FOLDER):
#     os.remove(CAPTURE_FOLDER + file)

# # delete all files in current directory that has string "output in it"

# for file in os.listdir('./'):
#   if 'output' in file: 
#     os.remove(file)
	


# Loop through frame of both video and save image of both video as pair
count = 0
for video in os.listdir(VIDEO_FOLDER):
  
  if count == 3:
    count = 1
  
  if video.endswith('.MP4'):
    cap = cv2.VideoCapture(VIDEO_FOLDER + video)
    countA = 0
    while cap.isOpened():
      ret, frame = cap.read()

      # if countA > 500:
      #   break

      if ret:                                
        cv2.imwrite(CAPTURE_FOLDER + str(count) + '_frame%d.jpg' % countA, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
      else:
        break

      countA += 1
      

    cap.release()
    cv2.destroyAllWindows()
  count = 3


# get the number of img in capture folder
num_img = len([name for name in os.listdir(CAPTURE_FOLDER) if os.path.isfile(os.path.join(CAPTURE_FOLDER, name))])
num_img_pair = num_img // 2

# # get the image pair
countB = 0

############################## Test 1
# for i in range(num_img_pair):
#   img1_color = cv2.imread(CAPTURE_FOLDER + '0_frame%d.jpg' % i)
#   img2_color = cv2.imread(CAPTURE_FOLDER + '1_frame%d.jpg' % i)

	
#   # Convert to grayscale.
#   img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
#   img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

#   # remove black bars on the sides of img2
#   img2 = img2[:, 80:560]
  

#   # invert the grayscale image of img2
#   img2 = cv2.bitwise_not(img2)

#   # Find the ORB features and descriptors.

#   height, width = img2.shape
  
#   # Create ORB detector with 5000 features.
#   orb_detector = cv2.ORB_create(5000)
  
#   # Find keypoints and descriptors.
#   # The first arg is the image, second arg is the mask
#   #  (which is not required in this case).
#   kp1, d1 = orb_detector.detectAndCompute(img1, None)
#   kp2, d2 = orb_detector.detectAndCompute(img2, None)
  
#   # Match features between the two images.
#   # We create a Brute Force matcher with 
#   # Hamming distance as measurement mode.
#   matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
  
#   # Match the two sets of descriptors.
#   matches = matcher.match(d1, d2)
  

#   # Sort matches on the basis of their Hamming distance.
#   # matches.sort(key = lambda x: x.distance)

#   #bard
#   # matches = sorted(matches, key=lambda x: x.distance)

#   #chatgpt
#   matches = list(matches)  # Convert the tuple to a list
#   matches.sort(key=lambda x: x.distance)  # Sort the list based on the 'distance' attribute


  
#   # Take the top 90 % matches forward.
#   matches = matches[:int(len(matches)*0.9)]
#   no_of_matches = len(matches)
  
#   # Define empty matrices of shape no_of_matches * 2.
#   p1 = np.zeros((no_of_matches, 2))
#   p2 = np.zeros((no_of_matches, 2))
  
#   for i in range(len(matches)):
#     p1[i, :] = kp1[matches[i].queryIdx].pt
#     p2[i, :] = kp2[matches[i].trainIdx].pt
  
#   # Find the homography matrix.
#   homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
  
#   # Use this matrix to transform the
#   # colored image wrt the reference image.
#   transformed_img = cv2.warpPerspective(img1_color,
#                     homography, (width, height))
  
#   # Save the output.
#   cv2.imwrite(str(countB) + 'output.jpg', transformed_img)
#   countB += 1



######################################### Test 2
# for i in range(num_img_pair):

# 	# Open the image files.
# 	img1_color = cv2.imread("./Capture/0_frame%d.jpg" % i) # Image to be aligned.
# 	img2_color = cv2.imread("./Capture/1_frame%d.jpg" % i) # Reference image.

# 	# Convert to grayscale.
# 	img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
# 	img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
# 	height, width = img2.shape

# 	# Create ORB detector with 5000 features.
# 	orb_detector = cv2.ORB_create(5000)

# 	# Find keypoints and descriptors.
# 	# The first arg is the image, second arg is the mask
# 	# (which is not required in this case).
# 	kp1, d1 = orb_detector.detectAndCompute(img1, None)
# 	kp2, d2 = orb_detector.detectAndCompute(img2, None)

# 	# match the features
# 	method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
# 	matcher = cv2.DescriptorMatcher_create(method)
# 	matches = matcher.match(d1, d2, None)

# 	# Match features between the two images.
# 	# We create a Brute Force matcher with
# 	# Hamming distance as measurement mode.
# 	matches = sorted(matches, key=lambda x:x.distance)

# 	# keep only the top matches
# 	keep = int(len(matches) * 0.2)
# 	matches = matches[:keep]

# 	# check to see if we should visualize the matched keypoints
# 	# if debug:
# 	#   matchedVis = cv2.drawMatches(img1_color, kp1, img2_color, kp2, matches, None)
# 	#   matchedVis = imutils.resize(matchedVis, width=1000)
# 	#   cv2.imwrite('matched.jpg', matchedVis)
# 	# 	# cv2.imshow("Matched Keypoints", matchedVis)
# 	# 	# cv2.waitKey(0)

# 	# Define empty matrices of shape no_of_matches * 2.

# 	ptsA = np.zeros((len(matches), 2), dtype="float")
# 	ptsB = np.zeros((len(matches), 2), dtype="float")

# 	# loop over the top matches
# 	for (i, m) in enumerate(matches):
# 		# indicate that the two keypoints in the respective images
# 		# map to each other
# 		ptsA[i] = kp1[m.queryIdx].pt
# 		ptsB[i] = kp2[m.trainIdx].pt


# 	(H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
# 		# use the homography matrix to align the images
# 	(h, w) = img2_color.shape[:2]
# 	aligned = cv2.warpPerspective(img1_color, H, (w, h))

# 	# Save the output.
# 	cv2.imwrite(str(countB) + 'output.jpg', aligned)
# 	countB += 1
        
	

