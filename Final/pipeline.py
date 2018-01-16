import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip

image = mpimg.imread('bridge_shadow.jpg')

# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Take both Sobel x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Calculate the gradient magnitude
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	# Rescale to 8 bit
	scale_factor = np.max(gradmag)/255 
	gradmag = (gradmag/scale_factor).astype(np.uint8) 
	# Create a binary image of ones where threshold is met, zeros otherwise
	binary_output = np.zeros_like(gradmag)
	binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

	# Return the binary image
	return binary_output

def undistort_image(img, objp, imgp):
    
    img_size = (img.shape[1], img.shape[0])
    
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp, imgp, img_size, None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    
    return dst

# Edit this function to create your own pipeline.
def colorGradThresholdImage(img):
	# Sobel x
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]

	# Threshold: gradient
	sobelx = mag_thresh(img, sobel_kernel=3, mag_thresh=(75, 255)) # Take the derivative in x

	# Threshold color channel
	s_thresh_min = 180
	s_thresh_max = 255
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
	
	# Stack each channel to view their individual contributions in green and blue respectively
	# This returns a stack of the two binary images, whose components you can see as different colors
	color_binary = np.dstack(( np.zeros_like(sobelx), sobelx, s_binary)) * 255
	
	# Combine the two binary thresholds
	combined_binary = np.zeros_like(sobelx)
	combined_binary[(s_binary == 1) | (sobelx == 1)] = 1
	
	return combined_binary

def showImage(original, transformedImage, caption):
	# Plot the result
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	f.tight_layout()

	ax1.imshow(original)
	ax1.set_title('Original Image', fontsize=40)

	ax2.imshow(transformedImage, cmap='gray')
	ax2.set_title(caption, fontsize=40)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()

def warp_image(img, src, dst, image_size):
	M = cv2.getPerspectiveTransform(src, dst)
	warped = cv2.warpPerspective(img, M, image_size, flags=cv2.INTER_LINEAR)
	Minv = cv2.getPerspectiveTransform(dst, src)
	return warped, M, Minv

def get_src_dest_warp_points(image):

	# Construct source and destination points as the basis for the perspective transform
	center_point = np.uint(image.shape[1]/2)
	y_top = np.uint(image.shape[0]/1.5)
	  
	corners = np.float32([[253, 697], [585, 456], [700, 456], [1061, 690]])
	new_top_left = np.array([corners[0, 0], 0])
	new_top_right = np.array([corners[3, 0], 0])
	offset = [50, 0]
	
	img_size = (image.shape[1], image.shape[0])
	
	src = np.float32([corners[0], corners[1], corners[2], corners[3]])
	dst = np.float32([corners[0] + offset, new_top_left + offset, new_top_right - offset, corners[3] - offset])
	
	return src, dst

def get_curvature(leftx, lefty, rightx, righty, ploty, image_size):
	y_eval = np.max(ploty)
	
	# Calculate curvature in pixel-space.
	# Convert from pixels to metres.
	
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension
	
	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
	
	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	#right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

	# Now our radius of curvature is in meters
	# Example values: 632.1 m    626.2 m
	
	# Calculate Lane Deviation from center of lane:
	# First we calculate the intercept points at the bottom of our image, then use those to 
	# calculate the lane deviation of the vehicle (assuming camera is in center of vehicle)
	scene_height = image_size[0] * ym_per_pix
	scene_width = image_size[1] * xm_per_pix
	
	left_intercept = left_fit_cr[0] * scene_height ** 2 + left_fit_cr[1] * scene_height + left_fit_cr[2]
	right_intercept = right_fit_cr[0] * scene_height ** 2 + right_fit_cr[1] * scene_height + right_fit_cr[2]
	calculated_center = (left_intercept + right_intercept) / 2.0
	
	lane_deviation = (calculated_center - scene_width / 2.0)
	
	return left_curverad, right_curverad, lane_deviation

def find_lane_lines(binary_warped, debug=False):
	
	if debug == True:
		# Create an output image to draw on and  visualize the result
		out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

	
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint
	
	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []
	
	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		
		if debug == True:
			# Draw the windows on the visualization image
			cv2.rectangle(out_img, (win_xleft_low,win_y_low), (win_xleft_high,win_y_high), (0,255,0), 2)
			cv2.rectangle(out_img, (win_xright_low,win_y_low), (win_xright_high,win_y_high), (0,255,0), 2)
		
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
		
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
		
	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)
	
	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	
	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	
	l, r, d = get_curvature(leftx, lefty, rightx, righty, ploty, binary_warped.shape)
	
	if debug == True:
		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
		return left_fitx, right_fitx, ploty, left_fit, right_fit, l, r, d, out_img
	else:
		return left_fitx, right_fitx, ploty, left_fit, right_fit, l, r, d

def draw_lanes_on_image(binary_warped, undistorted_img, Minv, left_fitx, right_fitx, ploty, left_radius, right_radius, lane_deviation):

	# Create a blank image to draw the lines on
	warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
	cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=30)
	cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=30)

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted_img.shape[1], undistorted_img.shape[0])) 

	# Combine the result with the original image
	result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)

	curvature_text = "Curvature: Left = " + str(np.round(left_radius, 2)) + ", Right = " + str(np.round(right_radius, 2)) 
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(result, curvature_text, (10, 30), font, 1, (0,255,255), 2)

	deviation_text = "Lane deviation from center = {:.2f} m".format(lane_deviation)   
	cv2.putText(result, deviation_text, (10, 60), font, 1, (255,0,255), 2)
		
	return result



def pipeline_image(img):
	

	# Undistort Image
	img = img_undistorted = undistort_image(img, objpoints, imgpoints)
	# Apply thresholds to image
	img = colorGradThresholdImage(img)
	# Get source and destination points
	src, dst = get_src_dest_warp_points(img)
	# Warp image to get top down view
	img, _, Minv = warp_image(img, src, dst, (image.shape[1], image.shape[0]))
	# Find lane lines
	left_fitx, right_fitx, ploty, left_fit, right_fit, l, r, d = find_lane_lines(img, debug=False)
	# Draw lanes
	img = draw_lanes_on_image(img, img_undistorted, Minv, left_fitx, right_fitx, ploty, l, r, d)
	
	return img

#Create a list of camera callibration images
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

#Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Take image and apply distortion correction
undistortImage = cv2.undistort(image, mtx, dist, None, mtx)

showImage(image, undistortImage, "Undistored Image")
cv2.imwrite("output_images/undistortedImage.jpg", undistortImage)

#Combine both the threshold and gradient detection.
thresholdImage = colorGradThresholdImage(undistortImage)
showImage(image, thresholdImage, "Thresholded Image") 
mpimg.imsave("out.png", thresholdImage)
cv2.imwrite("output_images/thresholdImage.jpg", thresholdImage)

# Apply a perspective transform to rectify binary image ("birds-eye view").
src, dst = get_src_dest_warp_points(undistortImage)
warped, _, Minv = warp_image(colorGradThresholdImage(image), src, dst, (image.shape[1], image.shape[0]))
showImage(image, warped, "Top-Down View")    
cv2.imwrite("output_images/topDownView.jpg", warped);

left_fitx, right_fitx, ploty, left_fit, right_fit, l, r, d, out_img = find_lane_lines(warped, debug=True)

# Take a histogram of the bottom half of the image
histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
plt.plot(histogram)
plt.show()
cv2.imwrite("output_images/histogramOfLane.jpg", result);

# Assume you now have a new warped binary image 
# from the next frame of video (also called "binary_warped")
# It's now much easier to find line pixels!
nonzero = warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
margin = 100
left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

# Again, extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]
# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
# Generate x and y values for plotting
ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# Create an image to draw on and an image to show the selection window
out_img = np.dstack((warped, warped, warped))*255
window_img = np.zeros_like(out_img)
# Color in left and right line pixels
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

# Generate a polygon to illustrate the search window area
# And recast the x and y points into usable format for cv2.fillPoly()
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))

# Draw the lane onto the warped blank image
cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
plt.imshow(result)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
cv2.imwrite("output_images/warpedWithLaneLine.jpg", result);

#Finally create the final image with the lange lines drawn on.
result = draw_lanes_on_image(warped, undistortImage, Minv, left_fitx, right_fitx, ploty, l, r, d)	
plt.figure(figsize=(16,8))
plt.imshow(result)
plt.axis("off");
plt.show()
cv2.imwrite("output_images/result.jpg", result);

print("Creating Video")
video_challenge_output = "output_images/project_video_output.mp4"	
clip1 = VideoFileClip("project_video.mp4")
clip1_output = clip1.fl_image(pipeline_image)
clip1_output.write_videofile(video_challenge_output, audio=False)



