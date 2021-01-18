import cv2
import imutils as imutils
import numpy as np

outputs = []
counter_triangle = 1
counter_rect = 1
counter_cir = 1
counter_line = 1
colors = []


def contour_image(image, colorName):
    contours = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    for c in contours:  ## looping over the contours found to assign shapes
        outshape = shapeDetector(c)
        line = str(colorName) + ' ' + str(outshape)  # creating a string of the output
        outputs.append(line)
        if outshape == "triangle":
            global counter_triangle
            counter_triangle += 1
        elif outshape == "rectangle":
            global counter_rect
            counter_rect += 1
        elif outshape == "line":
            global counter_line
            counter_line += 1
        elif outshape == "circle":
            global counter_cir
            counter_cir += 1
    cv2.waitKey(0)


def shapeDetector(c):
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)

    if len(approx) == 3:
        shape = "triangle"
        # if the shape has 4 vertices, it is either a rectangle or
        # a line
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "rectangle" if ar >= 0.45 and ar <= 1.6 else "line"
        # otherwise, we assume the shape is a circle
    else:
        shape = "circle"
        # return the name of the shape
    return shape


def detectColor(image):
    boundaries = [  ## upper and lower boundaries for colors
        ([0, 50, 50], [10, 255, 255], "yellow"),  ##yellow
        ([0, 255, 0], [0, 255, 255], "green"),  ##green
        ([255, 0, 0], [255, 255, 0], "blue"),  ##blue
        ([0, 0, 255], [255, 0, 255], "red")  ##red
    ]
    for (lower, upper,
         colors) in boundaries:  ## looping over every color and calling the shapeDetector fn. on the masked image
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        colorName = np.array(colors, dtype=str)

        mask = cv2.inRange(image, lower, upper)
        contour_image(mask, colorName)  ## calling contor image to countor the masked shape
    return colorName


image = cv2.imread('test1.png')  ## first we read the image

blurred = cv2.GaussianBlur(image, (5, 5), 0)

detectColor(blurred)  ## call detect color function
for x in range(len(outputs)):  ## printing the output of the image stored in the array
    print(outputs[x])
