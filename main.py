import imutils
import tensorflow as tf
from tensorflow.keras.preprocessing.image import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
import re
import random
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import os
from PIL import Image


def crop_brain_contour(image):
    # import imutils
    # import cv2
    # from matplotlib import pyplot as plt

    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    return new_image
def equalizeHist(img):
  img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
  img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

# convert the YUV image back to RGB format
  img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
  return img_output
def skull_stripping(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    colormask = np.zeros(image.shape, dtype=np.uint8)
    colormask[thresh!=0] = np.array((0,0,255))
    blended = cv2.addWeighted(image,0.7,colormask,0.1,0)
    ret, markers = cv2.connectedComponents(thresh)

    #Get the area taken by each component. Ignore label 0 since this is the background.
    marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0]
    #Get label of largest component by area
    largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above
    #Get pixels which correspond to the brain
    brain_mask = markers==largest_component
    brain_mask = np.uint8(brain_mask)
    kernel = np.ones((6,6),np.uint8)
    closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
    brain_out = img.copy()

    #In a copy of the original image, clear those pixels that don't correspond to the brain
    brain_out[brain_mask==False] = (0,0,0)
    return brain_out
SIZE=224
tumor = []
name = []
count = 0
for image in os.listdir('/content/drive/MyDrive/data/Training/tumor'):
    img_path = os.path.join('/content/drive/MyDrive/data/Training/tumor', image)
    print(img_path)
    img_raw = cv2.imread(img_path)
    img = np.array(img_raw)
    img = crop_brain_contour(img)
    img = skull_stripping(img)
    img = equalizeHist(img)
    img = cv2.resize(img, (SIZE, SIZE))
    img = img.reshape(SIZE, SIZE, 3)
    tumor.append(img)
    name.append(image)
    count = count+1
for i in range(count):
    ssname = 'ss_'+ name[i]
    img_path = os.path.join('/content/drive/MyDrive/data/Training/ss_tumor', ssname)
    print(img_path)
    cv2.imwrite(img_path, tumor[i])
tumor = np.array(tumor)
SIZE=224
tumor = []
name = []
count = 0
for image in os.listdir('/content/drive/MyDrive/data/Training/tumor'):
    img_path = os.path.join('/content/drive/MyDrive/data/Training/tumor', image)
    print(img_path)
    img_raw = cv2.imread(img_path)
    img = np.array(img_raw)
    img = crop_brain_contour(img)
    img = skull_stripping(img)
    img = equalizeHist(img)
    img = cv2.resize(img, (SIZE, SIZE))
    img = img.reshape(SIZE, SIZE, 3)
    tumor.append(img)
    name.append(image)
    count = count+1
for i in range(count):
    ssname = 'ss_'+ name[i]
    img_path = os.path.join('/content/drive/MyDrive/data/Training/ss_tumor', ssname)
    print(img_path)
    cv2.imwrite(img_path, tumor[i])
tumor = np.array(tumor)