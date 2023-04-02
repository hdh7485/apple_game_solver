#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  import numpy as np
#  import pytesseract
#  import cv2
#  import PIL
#  import numpy as np
#
#  filename = 'apple_game_image.png'
#  img = cv2.imread(filename)
#  lower_bound = (235, 235, 235)
#  upper_bound = (255, 255, 255)
#  img = cv2.inRange(img, lower_bound, upper_bound)
#  cv2.imshow('inRange', img)
#  cv2.waitKey()
#  cv2.destroyAllWindows()
#
#  text = pytesseract.image_to_string(img)
#  print(text)

import cv2

# Load the image
filename = 'apple_game_image.png'
img = cv2.imread(filename)

# Convert to grayscale
lower_bound = (0, 0, 100)
upper_bound = (90, 90, 255)
img = cv2.inRange(img, lower_bound, upper_bound)
cv2.imshow('inRange', img)
cv2.waitKey()
cv2.destroyAllWindows()

# Apply Gaussian blur to reduce noise
#  blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect rectangles using findContours
contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(contours)

# If contours are found
if contours:
    # Loop over the contours
    for contour in contours:
        # Get the area and perimeter of the contour
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Approximate the contour with a polygon
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        # If the polygon has four vertices (a rectangle)
        if len(approx) == 4:
            # Get the bounding box coordinates of the rectangle
            x, y, w, h = cv2.boundingRect(approx)

            # Extract the region of interest (ROI) inside the rectangle
            roi = img[y:y+h, x:x+w]

            # Apply thresholding to binarize the ROI
            _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            # Find contours in the binarized ROI
            digit_contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # If contours are found
            if digit_contours:
                # Sort the contours from left to right
                digit_contours = sorted(digit_contours, key=lambda c: cv2.boundingRect(c)[0])

                # Loop over the contours
                for digit_contour in digit_contours:
                    # Get the bounding box coordinates of the contour
                    digit_x, digit_y, digit_w, digit_h = cv2.boundingRect(digit_contour)

                    # Ignore contours that are too small or too large
                    if digit_w < 10 or digit_h < 10 or digit_w > 50 or digit_h > 50:
                        continue

                    # Extract the digit ROI from the binarized ROI
                    digit_roi = thresh[digit_y:digit_y+digit_h, digit_x:digit_x+digit_w]

                    # Resize the digit ROI to 28x28 pixels
                    resized_digit_roi = cv2.resize(digit_roi, (28, 28), interpolation=cv2.INTER_AREA)

                    # Invert the colors of the digit ROI (to white on black)
                    inverted_digit_roi = cv2.bitwise_not(resized_digit_roi)

                    # Flatten the inverted digit ROI to a 1D numpy array
                    flattened_digit_roi = inverted_digit_roi.flatten()

            # Draw the rectangle in the original image
            cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('inRange', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
