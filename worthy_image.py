import pandas as pd
import json
import os
import glob
import datetime
import pytz

# For computing distance and manipulating spatial data
from scipy.spatial import distance as dist
# For perspective and contour manipulation in images
from imutils import perspective, contours
import numpy as np
import imutils
import matplotlib
import matplotlib.pyplot as plt

# OpenCV for image processing
import cv2 as cv
from tabulate import tabulate
from PIL import Image
import base64
import joblib
# import sklearn
from imutils.perspective import order_points
from xgboost import XGBRegressor
import xgboost
from PIL import Image
from io import BytesIO

# CONSTANTS
THRESHOLD_VALUE = 150  # Threshold for creating a binary image
MIN_CONTOUR_AREA = 10000  # Minimum contour area to consider
KERNEL_SIZE = (3, 3)  # Kernel size for morphological operations
REF_DIMENSION_CM = 10.08  # Reference dimension for pixels to metric conversion (in cm)
MIN_CONTOUR_DIMENSION = 200  # Minimum contour dimension for further processing
LS_THRESHOLD = 140  # Threshold for size decision (object size > 140mm)
CROP_BOX = (640, 50, 3950, 3350)  # Coordinates for cropping the image
SPLIT_RATIO = 0.35  # Ratio to split the cropped image into two regions
DECISION_THRESHOLD = 146  # Threshold to make a decision based on object size
TIMEZONE = 'Asia/Kolkata'  # Timezone for timestamp
FILENAME = 'image.jpg'  # Default filename for output

# Function to calculate midpoint between two points
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Function to extract RGB values and object size from an image
def extract_rgb_values(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    _, thresh = cv.threshold(gray, THRESHOLD_VALUE, 255, cv.THRESH_BINARY)
    
    kernel = np.ones(KERNEL_SIZE)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    valid_contours = [cnt for cnt in contours if cv.contourArea(cnt) > MIN_CONTOUR_AREA]
    
    orig = image.copy()
    
    l = []
    ls = 0
    
    for contour in valid_contours:
        if cv.contourArea(contour) < MIN_CONTOUR_DIMENSION:
            continue
        else:
            box = cv.minAreaRect(contour)
            box = cv.boxPoints(box) if not imutils.is_cv2() else cv.cv.BoxPoints(box)
            box = np.array(box, dtype="int")
            box = order_points(box)
            cv.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
            
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            
            pixels_per_metric = dB / float(REF_DIMENSION_CM)
            dimA = dA / pixels_per_metric
            dimB = dB / pixels_per_metric
            
            length = max(dimA, dimB)
            breadth = min(dimA, dimB)
            
            l.append(length)
            
            cv.putText(orig, "{:.1f}mm".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            cv.putText(orig, "{:.1f}mm".format(dimB), (int(trbrX + 10), int(trbrY)), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    
    for i in l:
        if i > LS_THRESHOLD:
            ls = i

    mask = np.zeros_like(image, dtype=np.uint8)
    for cnt in valid_contours:
        cv.drawContours(mask, [cnt], -1, (255, 255, 255), thickness=cv.FILLED)

    result = cv.bitwise_and(image, mask)
    
    valid_mask = mask[:, :, 0] > 0
    rgb_values = image[valid_mask]

    if len(rgb_values) == 0:
        return [0, 0, 0]

    average_rgb = np.mean(rgb_values, axis=0)
    
    return average_rgb, thresh, result, ls

# Function to perform whiteness analysis by splitting the image
def whiteness_analysis(im):
    if im is None:
        raise ValueError("Image could not be loaded.")
    
    im_pil = Image.fromarray(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    
    image_cropped = im_pil.crop(box=CROP_BOX)
    image_copy = image_cropped.copy()
    width, height = image_copy.size
    
    split_point = int(width * SPLIT_RATIO)
    strip_region = image_copy.crop((0, 0, split_point, height))
    bowl_region = image_copy.crop((split_point, 0, width, height))
    
    strip_rgb, _, _, ls = extract_rgb_values(cv.cvtColor(np.array(strip_region), cv.COLOR_RGB2BGR))
    bowl_rgb, _, _, _ = extract_rgb_values(cv.cvtColor(np.array(bowl_region), cv.COLOR_RGB2BGR))
    
    return bowl_rgb, strip_rgb, ls

# Function to process an image from bytes and return structured data
def val_extract(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        im1 = cv.imdecode(nparr, cv.IMREAD_COLOR) 
        
        if im1 is None:
            raise ValueError("Error decoding image from byte stream.")

        bowl_rgb, strip_rgb, ls = whiteness_analysis(im1)

        decision = "YES" if ls > DECISION_THRESHOLD else "NO"
        
        data = {
            'filename': FILENAME,
            'timestamp': datetime.datetime.now(pytz.timezone(TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S"),
            'ls': ls,
            'strip_r': strip_rgb[0],
            'strip_g': strip_rgb[1],
            'strip_b': strip_rgb[2],
            'bowl_r': bowl_rgb[0],
            'bowl_g': bowl_rgb[1],
            'bowl_b': bowl_rgb[2],
            'Decision': decision,
        }
        
        return data
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

