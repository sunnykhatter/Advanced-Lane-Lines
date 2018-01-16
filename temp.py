#Finding Lane lines 
#Lakshay Khatter

#Import neccesary libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output


#Create a list of camera images
camera_cal = []
images = glob.glob('camera_cal/calibration*.jpg')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)


#Arrays to store object points and image points from ALL IMAGES
objpoints = [] #3D points in real world space
imgpoints = [] #2D points in image plane

for idx, fname in enumerate(images):
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
	# print(fname)
	# If found, add object points, image points
	if ret == True:
		objpoints.append(objp)
		imgpoints.append(corners)
		print(fname + " Corners Detected")
	else:
		print(fname + " Corners Not Detected")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
undistort = cv2.undistort(img, mtx, dist, None, mtx)


# Apply a distortion correction to raw images.
raw_image = glob.glob('test_images/*.jpg')
undistorted_images = []
for idx, fname in enumerate(raw_image):
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	undistort = cv2.undistort(gray, mtx, dist, None, mtx)
	undistorted_images.append(undistort)
	# plt.imshow(undistort, cmap='gray')
	# plt.show()



# Use color transforms, gradients, etc., to create a thresholded binary image.
for image in undistorted_images:
	#Calculate the derivative in the x and y direction
	
	grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)

	plt.imshow(grad_binary, cmap='gray')
	plt.show()



# Apply a perspective transform to rectify binary image ("birds-eye view").
# Detect lane pixels and fit to find the lane boundary.
# Determine the curvature of the lane and vehicle position with respect to center.
# Warp the detected lane boundaries back onto the original image.
# Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


