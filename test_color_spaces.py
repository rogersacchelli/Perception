import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.cvtColor(cv2.imread(filename=sys.argv[1]), cv2.COLOR_BGR2RGB)
#img = cv2.resize(img, (640, 480))
img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
img_yuv_inv = 255 - img_yuv
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

image_list = [img, img_yuv, img_yuv_inv, img_ycrcb, img_hsv, img_hls, img_lab]
image_space = ['rgb', 'yuv', 'img_yuv_inv', 'ycrcb', 'hsv', 'hls', 'lab']

print(np.equal(img_lab[:, :, 2], img_yuv_inv[:, :, 1]))

for i, (v, cs) in enumerate(zip(image_list, image_space)):
    for j in range(3):
        plt.subplot(len(image_space), 3, 3 * i + j + 1)
        plt.axis('off')
        plt.title('Color Space %s' % cs)
        plt.imshow(v[:, :, j % 3], cmap='gray')

plt.show()