import cv2
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

 # Thresholding hyper-parameters
sobel_kernel_size = 9
sobel_x_thresholds = (50, 200)
sobel_y_thresholds = (0, 190)
sobel_mag_thresholds = (50, 180)
sobel_dir_thresholds = (0.7, 1.3)
color_cannel_thresholds = (170, 250)


image_height_pixels = 720
image_width_pixels = 1280

ym_per_pix = 30/720  # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension


class Transform():
    ''' Transform is used to both undistort images 
       (from camera distortion) and to to do
       perspective transforms.
    '''
    def __init__(self):
        
        # undistortion
        self.undistortMapX = None
        self.undistortMapY = None
        
        # points used to generate perspective transform
        self.perspectiveSrcPoints = []
        self.perspectiveDestPoints = []
        
        # perspective transform matris (and inverse)
        self.perspectiveMatrix = None
        self.perspectiveMatrixInv = None
    
        # chessboard image file names used for calibration and corners found.
        self.chessboard_imgs_and_corners = []
        
        # termination criteria for "cornerSubPix": max iteratons = 30, epsilon = 0.001
        self._corner_sub_pix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        
    def calibrate_with_chessboards(self, glob_pattern, nx, ny, save_corners=False):
        ''' Camera calibration. 
        
        Args:
            glob_pattern: Patern passed to "glob" to find chessboard images to use for calibration.
            nx: Number of horizontal intescetions in the chessboard images.
            ny: Number of veritical intersections in the chessboard images.
            save_corners: Save the filename and found cornders in the "chessboard_imgs_and_corners" property. Default is False.
        '''
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((nx*ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        
        # Get image file paths matching "glob_pattern"
        images = glob.glob(glob_pattern)
        
        # save shape of first image found
        img_shape = None
        
        # loop over images and find chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
            
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self._corner_sub_pix_criteria)
                imgpoints.append(corners2)
                
                if img_shape is None:
                    img_shape = (img.shape[1], img.shape[0])
                
                if save_corners:
                    self.chessboard_imgs_and_corners.append((fname, corners2))
        
        # calibrate camera using data found processing chessboard images
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
        
        # Get the undistort maps to return.
        self.undistortMapX, self.undistortMapY = cv2.initUndistortRectifyMap(mtx,dist,None,None,img_shape,5)
    
    def init_perspective_transform(self, src_points, dest_points):
        ''' Initialize perspective transformation.
        
        Args:
            src_points:  Points in perspective image that represents a square.
            dest_points: Points in desintaion image that represents that same square.
        '''
        self.perspectiveSrcPoints = src_points
        self.perspectiveDestPoints = dest_points
        self.perspectiveMatrix = cv2.getPerspectiveTransform(self.perspectiveSrcPoints, self.perspectiveDestPoints)
        self.perspectiveMatrixInv = np.linalg.inv(self.perspectiveMatrix)
    
    def remove_distortion(self, image):
        ''' Remove camera distortion from the image.
        '''
        if self.undistortMapX is not None:
            return cv2.remap(image, self.undistortMapX, self.undistortMapY, cv2.INTER_LINEAR)
        else:
            return image
    
    def warp_perspective(self, image):
        ''' Warp image to from "perspective" to "overhead" image.
        '''
        return cv2.warpPerspective(image, self.perspectiveMatrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    
    def unwarp_perspective(self, image):
        ''' Remove warping of image (change from "overhead"  back to "perspective).
        '''
        return cv2.warpPerspective(image, self.perspectiveMatrixInv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    def save(self, filename):
        state = (self.undistortMapX, self.undistortMapY, self.perspectiveMatrix, self.perspectiveMatrixInv)
        pickle.dump( state, open( filename + ".p", "wb" ) )
    
    def load(self, filename):
        state = pickle.load( open( filename + ".p", "rb" ) )
        self.undistortMapX = state[0]
        self.undistortMapY = state[1]
        self.perspectiveMatrix = state[2]
        self.perspectiveMatrixInv = state[3]

def sobel_thresh(gray_img, sobel_kernel=3, x_thresh=(0, 255), y_thresh=(0,255), mag_thresh=(0, 255), dir_thresh=(0, np.pi/2)):
    ''' Threshold an gray scale image based on the gradient (sobel operator)
    
    Args:
        gray_img: gray image to perform sobel operation on and threshold.
        sobel_kernel: size of the sobel kernel to use.  Default is 3.
        x_thresh: lower/upper bound for x-direction threshold. Default is (0, 255), which is "all".
        y_thresh: lower/upper bound for y-direction threshold. Default is (0, 255), which is "all".
        mag_thresh: lower/upper bound for magnitude threshold. Default is (0,255), which is "all".
        dir_thresh: lower/upper bound for gradient direction.  Default is (0, pi/2) which is "all".
    
    Returns:
        4 binary images, one for each threshold.
    '''
    sobelx = np.copy(gray_img)
    sobely = np.copy(gray_img)
    
    # Calculate gradient magnitude
    sobelx = cv2.Sobel(sobelx, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
    sobely = cv2.Sobel(sobely, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # Take the derivative in y
    
    # x-direction absolute value, scaled, threshold
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    grad_x_binary = np.zeros_like(scaled_sobel)
    grad_x_binary[(scaled_sobel >= x_thresh[0]) & (scaled_sobel <= x_thresh[1])] = 1
    
    # y-direction absolute value, scaled, threshold
    abs_sobely = np.absolute(sobely) # Absolute y derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))
    grad_y_binary = np.zeros_like(scaled_sobel)
    grad_y_binary[(scaled_sobel >= y_thresh[0]) & (scaled_sobel <= y_thresh[1])] = 1
    
    # magnitude, scaled, threshold
    sobel_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_sobel = np.uint8(255*sobel_mag/np.max(sobel_mag))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    
    # gradient direction absolute value, scaled, threshold
    absgraddir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1
    
    return grad_x_binary, grad_y_binary, mag_binary, dir_binary


def color_channel_thresh(color_channel, thresh=(0, 255)):
    ''' Threshold a single channel (gray) image.
    
    Args:
        color_channel: The single channel image to threshold.
        thresh: lower/upper bound for pixel threshold. Default is (0, 255), which is "all".
    
    Returns:
        Binary, thresholded image.
    '''
    binary = np.zeros_like(color_channel)
    binary[(color_channel >= thresh[0]) & (color_channel <= thresh[1])] = 1
    return binary



def to_binary_mask(image):
    
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Apply each of the thresholding functions
    gradx, grady, mag_binary, dir_binary = \
        sobel_thresh(l_channel, sobel_kernel_size, \
                     sobel_x_thresholds, sobel_y_thresholds, \
                     sobel_mag_thresholds, sobel_dir_thresholds)

    channel_binary = color_channel_thresh(s_channel, color_cannel_thresholds)
    
    binary_mask = np.zeros_like(dir_binary)
    binary_mask[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (channel_binary == 1)] = 1
    
    return binary_mask



def calc_radius_of_curvature(pixels_x, pixels_y):
    coefficients = np.polyfit(pixels_y * ym_per_pix, pixels_x * xm_per_pix, 2)
    return ((1 + (2 * coefficients[0] * image_height_pixels * ym_per_pix + coefficients[1])**2)**1.5) / np.absolute(2 * coefficients[0])

def x_in_meters(x):
    return x * xm_per_pix

# Number of previous "fits" to keep.
max_fits_to_keep = 5

# Minimum number of pixels that have to be passed for
# a line to be considered "detected".
min_pixels_per_line = 100

# Minimum radius of curvature (in meters) a line can
# have and be considered "detected".
min_radius_of_curvature = 100 # meters

class Line():
    ''' Encapsulates a "lane line" (left or right).
        
        Pass pixel locations to "set_new_line_pixels" to
        initialize (or re-initialize) a line.
    '''
    def __init__(self):
        
        # was the line detected in the last iteration?
        self.detected = False
        
        # polynomial coefficients for the most recent fit
        self.current_fit = None
        
        # polynomial coefficients of the previous n fits
        self.prev_fits = []
        
        #radius of curvature of the line in meters
        self.radius_of_curvature = None
        
        #x values for detected line pixels
        self.allx = None
        
        #y values for detected line pixels
        self.ally = None

    def reset(self):
        ''' Reset all line values to defaults (uninitialize).
        '''
        self.detected = False
        self.current_fit = None
        self.prev_fits = []
        self.radius_of_curvature = None
        self.allx = None
        self.ally = None
    
    def set_not_detected(self):
        ''' Set the "detected" flag to false (also remove "current_fit" coefficients)
        '''
        self.detected = False
        self.current_fit = None
    
    def has_best_fit(self):
        ''' Are there "best fit" coefficients?
        
            There will be a "best fit" if one or more lines have been detected. 
        
        Returns:
            True if there is a "best fit"
        '''
        return (len(self.prev_fits) > 0) or (self.current_fit is not None)
    
    def get_best_fit(self):
        ''' Get the "best fit" which is the average of the current and
            and previous n fit coefficients (for last n detected lines).
        '''
        # best fit is average of the last n fits
        all_fits = self.prev_fits
        if self.current_fit is not None:
            if len(all_fits) > 0:
                all_fits = np.concatenate([self.prev_fits, [self.current_fit]], axis=0)
            else:
                all_fits = [self.current_fit]
        
        if len(all_fits) > max_fits_to_keep:
            all_fits = all_fits[1:]
        
        return np.mean(all_fits, axis=0)
    
    def x_using_best_fit(self, y_vals):
        ''' Calculate the "x" values for the given "y" values using
            the coefficients of the "best fit"
        '''
        if not self.has_best_fit():
            return None
        
        coefficients = self.get_best_fit()
        return (coefficients[0]*(y_vals**2) + coefficients[1]*y_vals + coefficients[2])
    
    def x_using_current_fit(self, y_vals):
        ''' Calculate the "x" values for the given "y" values using
            the coefficients of the "current fit"
        '''
        if self.current_fit == None:
            return None
        
        return (self.current_fit[0]*(y_vals**2) + self.current_fit[1]*y_vals + self.current_fit[2])
    
    def set_new_line_pixels(self, pixel_x_vals, pixel_y_vals):
        ''' Initialize the Line with the specified pixel locations.
        
            Will fit a quadradic to the pixel locations, the results being
            stored as the "current fit".    If there are not a minimum
            number of pixels specified, or the resuling cuve has too
            much curvature, then the "detected" property will be set
            to False, and there will be no "current fit".
        '''
        # save pevious fit to "prev_fits", if it was detected
        if self.current_fit is not None:
            if len(self.prev_fits) >= max_fits_to_keep:
                self.prev_fits = self.prev_fits[1:]
            self.prev_fits.append(self.current_fit)
        
        self.detected = False
        self.current = None
        if len(pixel_x_vals) >= min_pixels_per_line:
            
            rcurv = calc_radius_of_curvature(pixel_x_vals, pixel_y_vals)
            
            if rcurv >= min_radius_of_curvature:
                self.detected = True
                self.allx = pixel_x_vals
                self.ally = pixel_y_vals
                self.radius_of_curvature = rcurv
                self.current_fit = np.polyfit(pixel_y_vals, pixel_x_vals, 2)
        
        return self.detected

# min/max lane width (pixels) to validate lanes
min_lane_width = 500
max_lane_width = 900

class Lane():
    ''' Encapsulates "lane", consiting of left and right "line".
        
        Pass pixel locations to "set_new_lane_pixels" to
        initialize (or re-initialize) a lane.
    '''
    def __init__(self):
        self.left_line = Line()
        self.right_line = Line()
        
        self.detected = False
        self.no_detect_count = 0
        
        
    def __is_reasonable_lane__(self):
        ''' Sanity check to see if current lines make a valid lane.'''
        # were lines even detected?
        if (not self.left_line.detected) and (not self.right_line.detected):
            return False
        
        # plot the left and right line
        ploty = np.linspace(0, image_height_pixels - 1, image_height_pixels)
        left_fitx  = self.left_line.x_using_best_fit(ploty)
        right_fitx = self.right_line.x_using_best_fit(ploty)
        
        # is the left line on the left?
        if left_fitx[image_height_pixels - 1] > ((image_width_pixels//2)):
            return False
        
        # is the right line on the right?
        if right_fitx[image_height_pixels - 1] < ((image_width_pixels//2)):
            return False
        
        # use difference between right and left line
        # x positions as the "lane width"
        diff_fitx = right_fitx - left_fitx
        
        # is the mimimum width a reasonable lane width?
        min_fitx = np.min(diff_fitx)
        if (min_fitx < min_lane_width) or (min_fitx > max_lane_width):
            return False
        
        # is the maximum width a reasonable lane width?
        max_fitx = np.max(diff_fitx)
        if(max_fitx < min_lane_width) or (max_fitx > max_lane_width):
            return False
        
        return True
    
    def reset(self):
        ''' Reset all land (and line) values to defaults (uninitialize).
        '''
        self.detected = False
        self.no_detect_count = 0
        self.left_line.reset()
        self.right_line.reset()
    
    def has_best_fit(self):
        ''' Check if both left and right line have a "best fit"
        '''
        return (self.left_line.has_best_fit()) and (self.right_line.has_best_fit())
    
    def get_radius_of_curvature(self):
        ''' Return radius of curvature in meters (average of left/right line)
        '''
        if not self.has_best_fit():
            return None
        
        return (self.left_line.radius_of_curvature + self.right_line.radius_of_curvature) / 2
    
    def get_offset_from_center(self):
        ''' How from off from the center of the lane is the vehicle (in meters)
        '''
        if not self.has_best_fit():
            return None
        
        x_left = self.left_line.x_using_best_fit(image_height_pixels)
        x_right = self.right_line.x_using_best_fit(image_height_pixels)
        x_center = x_left + ( (x_right - x_left) // 2)
        x_vehicle_offset = (image_width_pixels //2) - x_center
        
        return x_in_meters(x_vehicle_offset)
    
    def set_new_lane_pixels(self, left_pixels, right_pixels):
        ''' Initialize a lane passing left/right line pixels.
        
            Will check if the resulting lines make up a valid lane, and
            if they are not, will set "detected" = False.
            
            Also keeps a count of the number of times "not detected" in a row.
        '''
        left_detected =  self.left_line.set_new_line_pixels(left_pixels[0], left_pixels[1])
        right_detected = self.right_line.set_new_line_pixels(right_pixels[0], right_pixels[1])
        
        self.detected = False
        if (left_detected) and (right_detected):
            self.detected = self.__is_reasonable_lane__()
        
        if not self.detected:
            self.left_line.set_not_detected()
            self.right_line.set_not_detected()
            self.no_detect_count += 1
        else:
            self.no_detect_count = 0
    
        return self.detected

# number of vertical slices to search for lines
nwindows = 9

# margin (left/right) of the pixel search window
window_margin = 100

# number of pixels required to recenter the search window
minpix_for_new_centroid = 50


def find_lane_pixels_initial(binary_warped):
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Take a histogram of the bottom third of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_y_center = (win_y_high - win_y_low)//2;
        
        win_xleft_center = leftx_current
        win_xright_center = rightx_current
        
        win_xleft_low = win_xleft_center - window_margin
        win_xleft_high = win_xleft_center + window_margin
        win_xright_low = win_xright_center - window_margin
        win_xright_high = win_xright_center + window_margin
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix_for_new_centroid:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix_for_new_centroid:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    left_x = nonzerox[left_lane_inds]
    left_y = nonzeroy[left_lane_inds]
    
    right_x = nonzerox[right_lane_inds]
    right_y = nonzeroy[right_lane_inds] 
    
    return (left_x, left_y), (right_x, right_y)

# Margin (left/right) around the "best fit" line to search for pixels
best_fit_margin = 50

def find_lane_pixels_using_previous(binary_warped, lane):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_x_centers = lane.left_line.x_using_best_fit(nonzeroy)
    left_lane_inds = ((nonzerox >  (left_x_centers - best_fit_margin)) & (nonzerox < (left_x_centers + best_fit_margin)))
    
    right_x_centers = lane.right_line.x_using_best_fit(nonzeroy)
    right_lane_inds = ((nonzerox > (right_x_centers - best_fit_margin)) & (nonzerox < (right_x_centers + best_fit_margin)))
    
    left_x = nonzerox[left_lane_inds]
    left_y = nonzeroy[left_lane_inds]
    
    right_x = nonzerox[right_lane_inds]
    right_y = nonzeroy[right_lane_inds] 
    
    return (left_x, left_y), (right_x, right_y)

# maximum number of times "no lane detected" can happen before
# resetting and search for lanes from scratch.
max_no_detects = 5

def find_lane(binary_warped, lane):
    ''' Find a "lane" in the specified binary image.
    
    Args:
        binary_warped: a binary image of just the lines of an image that has been perspective "warped" (top-down image).
        lane: a Lane object, holds any past lanes that were found, and is where the currently found land is stored.
    
    Results:
        lane: returns the same lane object passed in, populated with the found lane (if any).
    '''
    used_previous_fit = False
    
    if lane.has_best_fit():
        # find left/right lines using previous best fit.
        used_previous_fit = True
        left_pixels, right_pixels = find_lane_pixels_using_previous(binary_warped, lane)
    else:
        # Find left and right line from scratch
        left_pixels, right_pixels = find_lane_pixels_initial(binary_warped)
    
    lane.set_new_lane_pixels(left_pixels, right_pixels)
    
    if (not lane.detected) and (lane.no_detect_count >= max_no_detects):
        # too many no-detects, start fresh
        lane.reset()
        if used_previous_fit:
            #last attempt was via previous fit, try again from scratch
            left_pixels, right_pixels = find_lane_pixels_initial(binary_warped)
            lane.set_new_lane_pixels(left_pixels, right_pixels)
    
    return lane

def fill_lane(warped_binary_mask, lane):
    
    warp_zero = np.zeros_like(warped_binary_mask).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    if not lane.has_best_fit():
        return color_warp
    
    ploty = np.linspace(0, warped_binary_mask.shape[0]-1, warped_binary_mask.shape[0])
    
    left_fitx  = lane.left_line.x_using_best_fit(ploty)
    right_fitx = lane.right_line.x_using_best_fit(ploty)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    return color_warp


def get_lane_finder(transform):
    
    lane = Lane()
    
    def process_image(image):
        binary_mask = to_binary_mask(image)
        
        warped_binary_mask = transform.warp_perspective(binary_mask)
        
        find_lane(warped_binary_mask, lane)
        
        lane_colorized = None
        # Combine the result with the original image
        if lane.has_best_fit():
            lane_colorized = fill_lane(warped_binary_mask, lane)
            lane_colorized = transform.unwarp_perspective(lane_colorized)
        
        return lane, lane_colorized
    
    return process_image