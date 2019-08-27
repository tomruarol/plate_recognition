# import the necessary packages
from __future__ import print_function
from plate_recognition.license_plate import LicensePlateDetector
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to the images to be classified")
args = vars(ap.parse_args())