import numpy as np
import cv2
import glob
import os
import pickle
import time

out_examples = False
MOV_AVG_LENGTH = 5


def line_detector(image_data, line_info):

    image_data.undistort()
    image_data.binary()
    image_data.mask()
    image_data.warp()

    if not line_info.right_detected or not line_info.left_detected:
        line_info.start2fit(image_data)
    else:
        line_info.fit(image_data)

    try:
        line_info.mov_avg_left = np.append(line_info.mov_avg_left, np.array([line_info.left_fit]), axis=0)
        line_info.mov_avg_right = np.append(line_info.mov_avg_right, np.array([line_info.right_fit]), axis=0)

    except:
        line_info.mov_avg_left = np.array([line_info.left_fit])
        line_info.mov_avg_right = np.array([line_info.right_fit])

    line_info.left_fit = np.array([np.mean(line_info.mov_avg_left[::-1][:, 0][0:MOV_AVG_LENGTH]),
                         np.mean(line_info.mov_avg_left[::-1][:, 1][0:MOV_AVG_LENGTH]),
                         np.mean(line_info.mov_avg_left[::-1][:, 2][0:MOV_AVG_LENGTH])])

    line_info.right_fit = np.array([np.mean(line_info.mov_avg_right[::-1][:, 0][0:MOV_AVG_LENGTH]),
                         np.mean(line_info.mov_avg_right[::-1][:, 1][0:MOV_AVG_LENGTH]),
                         np.mean(line_info.mov_avg_right[::-1][:, 2][0:MOV_AVG_LENGTH])])

    if line_info.mov_avg_left.shape[0] > 1000:
        line_info.mov_avg_left = line_info.mov_avg_left[0:MOV_AVG_LENGTH]
    if line_info.mov_avg_right.shape[0] > 1000:
        line_info.mov_avg_right = line_info.mov_avg_right[0:MOV_AVG_LENGTH]

    if abs(line_info.left_fit[0]) < 5e-5:
        line_info.turn_side = 0
    elif line_info.left_fit[0] > 0:
        line_info.turn_side = 1
    else:
        line_info.turn_side = -1

    draw_lines_new(image_data, line_info)

    return image_data, line_info


def load_camera():
    # ------------------------ Camera Calibration ------------------------
    # As calibration may take some time, save calibration data into pickle file to speed up testing
    if not os.path.exists('camera_files/calibration.p'):
        # Read all jpg files from calibration image folder
        images = glob.glob('camera_files/camera_cal/*.jpg')

        with open('camera_files/calibration.p', mode='wb') as f:
            ret, mtx, dist, rvecs, tvecs = calibrate_camera(images, nx=9, ny=6)
            pickle.dump([ret, mtx, dist, rvecs, tvecs], f)
            f.close()
    else:
        with open('camera_files/calibration.p', mode='rb') as f:
            ret, mtx, dist, rvecs, tvecs = pickle.load(f)
            f.close()

    return ret, mtx, dist, rvecs, tvecs


def calibrate_camera(image_files, nx, ny):
    objpoints = []
    imgpoints = []

    objp = np.zeros(shape=(nx * ny, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for i in image_files:
        img = cv2.imread(i)
        if img.shape[0] != 720:
            img = cv2.resize(img,(1280, 720))
        cv2.imshow('image',img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny))

        if ret:
            print("Calibrated!")
            imgpoints.append(corners)
            objpoints.append(objp)

    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def fit_from_lines(left_fit, right_fit, img_w):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = img_w.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
    nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
    (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
    nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit


def draw_lines(img, img_w, left_fit, right_fit, perspective):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img_w).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    #color_warp_center = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    #cv2.fillPoly(color_warp_center, np.int_([pts]), (0, 255, 0))
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    #newwarp = warp(color_warp, perspective[1], perspective[0])
    # Combine the result with the original image
    #result = cv2.addWeighted(img, 1, newwarp, 0.2, 0)

    color_warp_lines = np.dstack((warp_zero, warp_zero, warp_zero))
    cv2.polylines(color_warp_lines, np.int_([pts_right]), isClosed=False, color=(255, 255, 0), thickness=25)
    cv2.polylines(color_warp_lines, np.int_([pts_left]), isClosed=False, color=(0, 0, 255), thickness=25)

    # ----- Radius Calculation ------ #

    img_height = img.shape[0]
    y_eval = img_height

    ym_per_pix = 30 / 720.  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    ploty = np.linspace(0, img_height - 1, img_height)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])

    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    radius = round((float(left_curverad) + float(right_curverad))/2.,2)

    # ----- Off Center Calculation ------ #

    lane_width = (right_fit[2] - left_fit[2]) * xm_per_pix
    center = (right_fit[2] - left_fit[2]) / 2
    off_left = (center - left_fit[2]) * xm_per_pix
    off_right = -(right_fit[2] - center) * xm_per_pix
    off_center = round((center - img.shape[0] / 2.) * xm_per_pix,2)

    # --- Print text on screen ------ #
    #if radius < 5000.0:
    text = "radius = %s [m]\noffcenter = %s [m]" % (str(radius), str(off_center))
    #text = "radius = -- [m]\noffcenter = %s [m]" % (str(off_center))

    for i, line in enumerate(text.split('\n')):
        i = 50 + 20 * i
    #    cv2.putText(result, line, (0,i), cv2.FONT_HERSHEY_DUPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
    #return result


def draw_lines_new(image_data, line_info, perspective):

    left_fitx = line_info.left_fit[0] * image_data.ploty ** 2 + line_info.left_fit[1] * image_data.ploty +\
                line_info.left_fit[2]
    right_fitx = line_info.right_fit[0] * line_info.right_fit ** 2 + line_info.right_fit[1] * image_data.ploty + \
                 line_info.right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, image_data.ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, image_data.ploty])))])

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    #newwarp = warp(color_warp, perspective[1], perspective[0])
    # Combine the result with the original image
    #result = cv2.addWeighted(img, 1, newwarp, 0.2, 0)

    cv2.polylines(image_data.warped_binary, np.int_([pts_right]), isClosed=False, color=(255, 255, 0), thickness=25)
    cv2.polylines(image_data.warped_binary, np.int_([pts_left]), isClosed=False, color=(0, 0, 255), thickness=25)

    # ----- Radius Calculation ------ #

    ym_per_pix = 30 / 720.  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(image_data.ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(image_data.ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    line_info.left_radius = ((1 + (2 * left_fit_cr[0] * line_info.shape_w * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])

    line_info.right_radius = (
                         (1 + (2 * right_fit_cr[0] * line_info.shape_w * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    radius = round((float(line_info.left_radius) + float(line_info.right_radius))/2.,2)

    # ----- Off Center Calculation ------ #

    lane_width = (line_info.right_fit[2] - line_info.left_fit[2]) * xm_per_pix
    center = (line_info.right_fit[2] - line_info.left_fit[2]) / 2
    off_left = (center - line_info.left_fit[2]) * xm_per_pix
    off_right = -(line_info.right_fit[2] - center) * xm_per_pix
    off_center = round((center - image_data.shape_h / 2.) * xm_per_pix,2)

    # --- Print text on screen ------ #
    text = "radius = %s [m]\noffcenter = %s [m]" % (str(radius), str(off_center))

    for i, line in enumerate(text.split('\n')):
        i = 50 + 20 * i
    #    cv2.putText(result, line, (0,i), cv2.FONT_HERSHEY_DUPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
    #return result


