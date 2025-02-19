import pandas as pd
import json
import os
import glob
import datetime
import pytz
import yaml  # Import YAML for configuration
import numpy as np
import imutils
import cv2 as cv
from scipy.spatial import distance as dist
from PIL import Image
from imutils.perspective import order_points

# Load configuration from YAML file
config_path = "D:\\Agsure\\AI\\ML\\config.yml"

if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file {config_path} not found!")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Load constants from YAML
THRESHOLD_VALUE = config["threshold_value"]
MIN_CONTOUR_AREA = config["min_contour_area"]
KERNEL_SIZE = tuple(config["kernel_size"])
REF_DIMENSION_CM = config["ref_dimension_cm"]
MIN_CONTOUR_DIMENSION = config["min_contour_dimension"]
LS_THRESHOLD = config["ls_threshold"]
CROP_BOX = tuple(config["crop_box"])
SPLIT_RATIO = config["split_ratio"]
DECISION_THRESHOLD = config["decision_threshold"]
TIMEZONE = config["timezone"]

# Function to calculate midpoint
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Function to extract RGB values and object size from an image
def extract_rgb_values(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, THRESHOLD_VALUE, 255, cv.THRESH_BINARY)

    kernel = np.ones(KERNEL_SIZE, np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv.contourArea(cnt) > MIN_CONTOUR_AREA]

    lengths = []
    ls = 0

    for contour in valid_contours:
        if cv.contourArea(contour) < MIN_CONTOUR_DIMENSION:
            continue
        box = cv.minAreaRect(contour)
        box = cv.boxPoints(box)
        box = np.array(box, dtype="int")
        box = order_points(box)

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

        lengths.append(length)

    for i in lengths:
        if i > LS_THRESHOLD:
            ls = i

    mask = np.zeros_like(image, dtype=np.uint8)
    for cnt in valid_contours:
        cv.drawContours(mask, [cnt], -1, (255, 255, 255), thickness=cv.FILLED)

    valid_mask = mask[:, :, 0] > 0
    rgb_values = image[valid_mask]

    if len(rgb_values) == 0:
        return [0, 0, 0]

    average_rgb = np.mean(rgb_values, axis=0)
    return average_rgb, ls

# Function to perform whiteness analysis by splitting the image
def whiteness_analysis(image):
    if image is None:
        raise ValueError("Image could not be loaded.")

    im_pil = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    image_cropped = im_pil.crop(box=CROP_BOX)
    width, height = image_cropped.size

    split_point = int(width * SPLIT_RATIO)
    strip_region = image_cropped.crop((0, 0, split_point, height))
    bowl_region = image_cropped.crop((split_point, 0, width, height))

    strip_rgb, ls = extract_rgb_values(cv.cvtColor(np.array(strip_region), cv.COLOR_RGB2BGR))
    bowl_rgb, _ = extract_rgb_values(cv.cvtColor(np.array(bowl_region), cv.COLOR_RGB2BGR))

    return bowl_rgb, strip_rgb, ls

# Function to process an image from bytes
def val_extract(image_bytes, filename):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        im1 = cv.imdecode(nparr, cv.IMREAD_COLOR)

        if im1 is None:
            raise ValueError("Error decoding image from byte stream.")

        bowl_rgb, strip_rgb, ls = whiteness_analysis(im1)
        decision = "YES" if ls > DECISION_THRESHOLD else "NO"

        data = {
            'filename': filename,
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

# Test the function
if __name__ == "__main__":
    image_path = "C:/Users/ACER/OneDrive/Desktop/Testing files/S_1036_25_217_1_20250210_112956.jpg"  # Replace with actual image path
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            image_data = img_file.read()
        filename = os.path.basename(image_path)  # Extract the exact filename
        result = val_extract(image_data, filename)
        print(json.dumps(result, indent=4))
    else:
        print("Test image not found!")
