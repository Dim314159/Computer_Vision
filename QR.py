#!/usr/bin/env python
# coding: utf-8

import cv2 as cv
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Parameters
# use your path to images
my_path = 'data/QR/dataset2/'


# # Load data
files = glob.glob(my_path + '*.jpg')

# # Initialize detector
qr_detect = cv.QRCodeDetector()

# # Extract QR
file = np.random.choice(files)

image = cv.imread(file)

_, _, straight_qrcode = qr_detect.detectAndDecode(image)
with_borders = cv.copyMakeBorder(straight_qrcode, 1, 1, 1, 1, borderType = cv.BORDER_CONSTANT, 
                                    value=255)
qr = cv.resize(with_borders, dsize = None, fx = 16, fy = 16, interpolation = cv.INTER_NEAREST)
plt.imshow(cv.cvtColor(qr, cv.COLOR_BGR2RGB))
plt.title('QR')
plt.axis('off')
plt.show()

