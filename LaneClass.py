import numpy as np
import cv2

REGION_OF_INTEREST = (.55, 0.95, 0., 1.)     # (y_start, y_end, x_start, x_end) relative to image size


class Line:
    """
    Line class returns information for the detected lines on the image.
    """
    def __init__(self):
        self.right_detected = False
        self.left_detected = False
        self.left_inside_detected = False
        self.left_outside_detected = False
        self.right_inside_detected = False
        self.right_out_detected = False

        self.lef_radius = 0.0
        self.right_radius = 0.0
        self.turn_side = 0
        self.left_fit = 0
        self.right_fit = 0
        self.mov_avg_left = np.array([])
        self.mov_avg_right = np.array([])

        self.ym_per_pix = 30 / 720.  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    def start2fit(self, image_data):
        """
        :param image_data: ImageLine Object
        :return:
        """

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(image_data.warped_binary.shape[0] / nwindows)
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        if not self.left_detected or not self.right_detected:
            histogram = np.sum(image_data.warped_binary[int(image_data.shape_roi_h / 2):, :], axis=0)
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0] / 2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image_data.warped_binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        if not self.left_detected or not self.right_detected:
            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = image_data.warped_binary.shape[0] - (window + 1) * window_height
                win_y_high = image_data.warped_binary.shape[0] - window * window_height

                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin

                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin

                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)).nonzero()[0]

                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        else:
            left_lane_inds = (
            (nonzerox > (self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] - margin)) & (
                nonzerox < (self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] + margin)))
            right_lane_inds = (
                (nonzerox > (self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] - margin)) & (
                    nonzerox < (self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] + margin)))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)


class ImageLine:
    """
    ImageLine Class performs image processing on the input image.
    """
    def __init__(self, image, ret, mtx, dist, rvecs, tvecs):

        self.image = image
        self.hls = 0
        self.yuv = 0

        self.shape_h = image.shape[0]
        self.shape_w = image.shape[1]

        # REGION OF INTEREST

        self.roi_y_start = int(REGION_OF_INTEREST[0] * self.shape_h)
        self.roi_y_end = int(REGION_OF_INTEREST[1] * self.shape_h)
        self.roi_x_start = int(REGION_OF_INTEREST[2] * self.shape_w)
        self.roi_x_end = int(REGION_OF_INTEREST[3] * self.shape_w)
        self.image_roi = self.image[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end, :]
        self.shape_roi_h = self.image_roi.shape[0]
        self.shape_roi_w = self.image_roi.shape[1]

        self.binary_output_s = np.zeros(shape=(self.shape_roi_h, self.shape_roi_w), dtype=np.float)
        self.binary_output_b = np.zeros(shape=(self.shape_roi_h, self.shape_roi_w), dtype=np.float)
        self.binary_output = self.binary_output_s + self.binary_output_b
        self.binary_sobel_s = np.zeros(shape=(self.shape_roi_h, self.shape_roi_w), dtype=np.float)
        self.binary_sobel_b = np.zeros(shape=(self.shape_roi_h, self.shape_roi_w), dtype=np.float)
        self.binary_hls_s = np.zeros(shape=(self.shape_roi_h, self.shape_roi_w), dtype=np.float)
        self.binary_lab_b = np.zeros(shape=(self.shape_roi_h, self.shape_roi_w), dtype=np.float)
        self.binary_mask = np.zeros(shape=(self.shape_roi_h, self.shape_roi_w), dtype=np.uint8)

        self.warped_binary = np.zeros_like(self.image_roi)
        self.unwarped_lines = np.zeros_like(self.image_roi, dtype=np.int8)

        self.ploty = np.linspace(0, self.shape_h - 1, self.shape_h)

        self.ret = ret
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs

        self.line_dst_offset = 100

    def to_bgr(self):
        return cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

    def reset_binary_images(self):
        self.binary_output_s = np.zeros(shape=(self.shape_roi_h, self.shape_roi_w), dtype=np.float)
        self.binary_output_b = np.zeros(shape=(self.shape_roi_h, self.shape_roi_w), dtype=np.float)
        self.binary_output = self.binary_output_s + self.binary_output_b
        self.binary_sobel_s = np.zeros(shape=(self.shape_roi_h, self.shape_roi_w), dtype=np.float)
        self.binary_sobel_b = np.zeros(shape=(self.shape_roi_h, self.shape_roi_w), dtype=np.float)
        self.binary_hls_s = np.zeros(shape=(self.shape_roi_h, self.shape_roi_w), dtype=np.float)
        self.binary_lab_b = np.zeros(shape=(self.shape_roi_h, self.shape_roi_w), dtype=np.float)
        self.binary_mask = np.zeros(shape=(self.shape_roi_h, self.shape_roi_w), dtype=np.uint8)

        self.warped_binary = np.zeros_like(self.image_roi)
        self.unwarped_lines = np.zeros_like(self.image_roi, dtype=np.int8)
        
    def undistort(self):
        self.image_roi = cv2.undistort(self.image_roi, self.mtx, self.dist, None, self.mtx)

    def binary(self, sobel_kernel=11, mag_thresh=(15, 255), s_thresh=(190, 255), debug=False):

        # --------------------------- Binary Thresholding ----------------------------
        # Binary Thresholding is an intermediate step to improve lane line perception
        # it includes image transformation to gray scale to apply sobel transform and
        # binary slicing to output 0,1 type images according to pre-defined threshold.
        #
        # Also it's performed RGB to HLS transformation to get S information which in-
        # tensifies lane line detection.
        #
        # The output is a binary image combined with best of both S transform and mag-
        # nitude thresholding.

        self.hls = cv2.cvtColor(self.image_roi, cv2.COLOR_BGR2HLS)
        self.lab = cv2.cvtColor(self.image_roi, cv2.COLOR_BGR2LAB)

        # HLS COMPUTATION - WHILE LINES DETECTION

        # Sobel Transform
        sobelx_s = cv2.Sobel(self.hls[:, :, 1], cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely_s = 0 #cv2.Sobel(self.hls[:, :, 1], cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        sobel_abs_s = np.abs(sobelx_s ** 2 + sobely_s ** 2)
        sobel_abs_s = np.uint8(255 * sobel_abs_s / np.max(sobel_abs_s))

        self.binary_sobel_s[(sobel_abs_s > mag_thresh[0]) & (sobel_abs_s <= mag_thresh[1])] = 1

        # Threshold color channel
        self.binary_hls_s[(self.hls[:, :, 1] >= s_thresh[0]) & (self.hls[:, :, 1] <= s_thresh[1])] = 1

        # Combine the two binary thresholds

        self.binary_output_s[(self.binary_hls_s == 1) | (self.binary_sobel_s == 1)] = 1
        self.binary_output_s = np.uint8(255 * self.binary_output_s / np.max(self.binary_output_s))

        # LAB COMPUTATION - YELLOW LINES DETECTION
        # Sobel Transform
        sobelx_b = cv2.Sobel(self.lab[:, :, 2], cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely_b = cv2.Sobel(self.lab[:, :, 2], cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        sobel_abs_b = np.abs(sobelx_b ** 2 + sobely_b ** 2)
        sobel_abs_b = np.uint8(255 * sobel_abs_b / np.max(sobel_abs_b))

        #self.binary_sobel_b[(sobel_abs_b > mag_thresh[0]) & (sobel_abs_b <= mag_thresh[1])] = 1
        self.binary_sobel_b[(sobel_abs_b > 1) & (sobel_abs_b <= mag_thresh[1])] = 1

        # Threshold color channel
        self.binary_lab_b[(self.lab[:, :, 2] >= s_thresh[0]) & (self.lab[:, :, 2] <= s_thresh[1])] = 1

        # Combine the two binary thresholds

        self.binary_output_b[(self.binary_lab_b == 1) | (self.binary_sobel_b == 1)] = 1
        self.binary_output_b = np.uint8(255 * self.binary_output_b / np.max(self.binary_output_b))

        self.binary_output = self.binary_output_b + self.binary_output_s


    def mask(self):
        # ---------------- MASKED IMAGE --------------------
        offset = 100
        mask_polyg = np.array([[(0 + offset, self.shape_roi_h),
                                (self.shape_roi_w // 2.5, 50),
                                (self.shape_roi_w // 1.8, 50),
                                (self.shape_roi_w, self.shape_roi_h)]],
                              dtype=np.int)

        # This time we are defining a four sided polygon to mask
        # Applying polygon

        # Next we'll create a masked edges image using cv2.fillPoly()
        mask_img = np.zeros_like(self.binary_output)

        # This time we are defining a four sided polygon to mask
        # Applying polygon
        cv2.fillPoly(mask_img, mask_polyg, 255)
        masked_edges = cv2.bitwise_and(self.binary_output, mask_img)
        self.binary_output = masked_edges
        cv2.imshow('binary', self.binary_output)

        #cv2.imwrite('mask_out.jpg', self.binary_output)

    def warp(self, inverse_warp=False):

        line_dst_offset = 200

        # src = [595, 452], \
        #       [685, 452], \
        #       [1110, self.shape_h], \
        #       [220, self.shape_h]
        src = [573, 43], \
              [659, 43], \
              [1155, self.shape_roi_h], \
              [325, self.shape_roi_h]

        dst = [src[3][0] + line_dst_offset, 0], \
              [src[2][0] - line_dst_offset, 0], \
              [src[2][0] - line_dst_offset, src[2][1]], \
              [src[3][0] + line_dst_offset, src[3][1]]

        src = np.float32([src])
        dst = np.float32([dst])

        if not inverse_warp:
            self.warped_binary = cv2.warpPerspective(self.binary_output, cv2.getPerspectiveTransform(src, dst),
                                                     dsize=(self.shape_roi_w, self.shape_roi_h), flags=cv2.INTER_LINEAR)
        else:
            return cv2.warpPerspective(self.unwarped_lines, cv2.getPerspectiveTransform(dst, src),
                                                      dsize=(self.shape_roi_w, self.shape_roi_h), flags=cv2.INTER_LINEAR)


