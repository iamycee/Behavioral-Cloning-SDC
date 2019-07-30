import cv2
import numpy as np


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  #smoothening
    canny = cv2.Canny(blur, 50, 150)
    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[
        (200, height), (1100, height), (550, 250)
    ]])  #triangle coordinates for region of image with road in it

    mask = np.zeros_like(image)  #dark image the size of test_image
    cv2.fillPoly(mask, polygons,
                 255)  #whiten out the triangular  region of road
    masked_image = cv2.bitwise_and(image,
                                   mask)  #  gives us onlt the lane edges
    return masked_image


'''
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)  #working with a copy of the original image
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image,
                        2,
                        np.pi / 180,
                        100,
                        np.array([]),
                        minLineLength=40,
                        maxLineGap=5)
line_image = display_lines(lane_image, lines)

combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

cv2.imshow('result', combo_image)
cv2.waitKey(0)
'''
cap = cv2.VideoCapture("test2.mp4")

while (cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image,
                            2,
                            np.pi / 180,
                            100,
                            np.array([]),
                            minLineLength=40,
                            maxLineGap=5)
    line_image = display_lines(frame, lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result', combo_image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
