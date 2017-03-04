import numpy as np
import cv2
import glob
import os
import pickle
import time

OUT_EXAMPLES = False
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

    if line_info.mov_avg_left.shape[0] > 100:
        line_info.mov_avg_left = line_info.mov_avg_left[0:MOV_AVG_LENGTH]
    if line_info.mov_avg_right.shape[0] > 100:
        line_info.mov_avg_right = line_info.mov_avg_right[0:MOV_AVG_LENGTH]

    if abs(line_info.left_fit[0]) < 5e-5:
        line_info.turn_side = 0
    elif line_info.left_fit[0] > 0:
        line_info.turn_side = 1
    else:
        line_info.turn_side = -1

    draw_lines(image_data, line_info)

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
            img = cv2.resize(img, (1280, 720))
        cv2.imshow('image', img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny))

        if ret:
            print("Calibrated!")
            imgpoints.append(corners)
            objpoints.append(objp)

    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def draw_lines(image_data, line_info):

    left_fitx = line_info.left_fit[0] * image_data.ploty ** 2 + line_info.left_fit[1] * image_data.ploty +\
                line_info.left_fit[2]
    right_fitx = line_info.right_fit[0] * image_data.ploty ** 2 + line_info.right_fit[1] * image_data.ploty +\
                 line_info.right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, image_data.ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, image_data.ploty])))])

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    # Combine the result with the original image
    warp_zero = np.zeros_like(image_data.warped_binary).astype(np.uint8)
    image_data.unwarped_lines = np.dstack((warp_zero, warp_zero, warp_zero))
    cv2.polylines(image_data.unwarped_lines, np.int_([pts_right]), isClosed=False, color=(0, 255, 0), thickness=25)
    cv2.polylines(image_data.unwarped_lines, np.int_([pts_left]), isClosed=False, color=(0, 255, 0), thickness=25)
    unwarped = image_data.warp(inverse_warp=True)
    image_data.image = cv2.addWeighted(image_data.image, 1, unwarped, 2, 1)

    # ----- Radius Calculation ------ #

    line_info.ym_per_pix = 30 / 720.  # meters per pixel in y dimension
    line_info.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(image_data.ploty * line_info.ym_per_pix, left_fitx * line_info.xm_per_pix, 2)
    right_fit_cr = np.polyfit(image_data.ploty * line_info.ym_per_pix, right_fitx * line_info.xm_per_pix, 2)

    # Calculate the new radii of curvature
    line_info.left_radius = ((1 + (2 * left_fit_cr[0] * image_data.shape_w * line_info.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) \
                            / np.absolute(2 * left_fit_cr[0])

    line_info.right_radius = ((1 + (2 * right_fit_cr[0] * image_data.shape_w * line_info.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) \
                             / np.absolute(2 * right_fit_cr[0])

    radius = round((float(line_info.left_radius) + float(line_info.right_radius))/2.,2)

    # ----- Off Center Calculation ------ #

    #lane_width = (line_info.right_fit[2] - line_info.left_fit[2]) * line_info.xm_per_pix
    center = (line_info.right_fit[2] - line_info.left_fit[2]) / 2
    #off_left = (center - line_info.left_fit[2]) * line_info.xm_per_pix
    #off_right = -(line_info.right_fit[2] - center) * line_info.xm_per_pix
    off_center = round((center - image_data.shape_h / 2.) * line_info.xm_per_pix,2)

    # --- Print text on screen ------ #
    text = "radius = %s [m]\noffcenter = %s [m]" % (str(radius), str(off_center))

    for i, line in enumerate(text.split('\n')):
        i = 50 + 20 * i
        cv2.putText(image_data.image, line, (0, i), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
