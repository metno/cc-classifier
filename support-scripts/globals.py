import cv2
import numpy as np

BRIGHTNESS_THRESHOLD = 0.45

def get_mean_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2].ravel().mean() / 255
    return brightness


def calc_spread(vector):
    i = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]) / 8
    x = np.array(vector)
    mean = np.sum(x * i)
    variance = sum(i * i * x) - mean*mean
    return variance
