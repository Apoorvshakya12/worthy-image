# Worthy Images Analyzer

This project provides a Python-based image analysis tool to evaluate the *whiteness index* and *object size* from agricultural product images (e.g., rice grains ). It uses OpenCV and image processing techniques to extract RGB values, measure physical dimensions, and decide the suitability of the image based on certain thresholds.

---

## Features

- ✅ Crops and splits the image into **bowl** and **strip** regions
- ✅ Calculates **average RGB** values for both regions
- ✅ Measures **object size (length in mm)** using contour detection
- ✅ Makes a **decision (YES/NO)** based on object size threshold
- ✅ Supports image input via bytes (e.g., API or UI integration)
- ✅ Returns results in structured dictionary form

---

## How It Works

1. **Preprocessing**:
   - Grayscale conversion, thresholding, morphological filtering.
   - Extracts external contours larger than a minimum area.

2. **Feature Extraction**:
   - Calculates object dimensions using pixel-to-metric conversion.
   - Draws contours and measures length/breadth.

3. **Whiteness Analysis**:
   - Splits the image into two regions: `strip` and `bowl`.
   - Computes mean RGB values for both.

4. **Decision Logic**:
   - If object length (`ls`) > `146mm`, the image is deemed "worthy".


 
