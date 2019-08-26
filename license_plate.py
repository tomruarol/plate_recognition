# import the necessary packages
from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective
import numpy as np
import imutils
import cv2
 
# namedtuple to store information regarding the detected license plate
LicensePlate = namedtuple("LicensePlateRegion", ["success", "plate", "thresh", "candidates"]) # namedtuple is similar to a struct in C.
                                                                                              # It is very Simple to instantiate new  namedtuples.
                                                                                              # As simple as passing in a list of arguments to the LicensePlate variable, just like instantiating a class.

'''
LicensePlate attributes:
    - success : A boolean indicating whether the license plate detection and character segmentation was successful or not.
    - plate : An image of the detected license plate.
    - thresh : The thresholded license plate region, revealing the license plate characters on the background.
    - candidates : A list of character candidates that should be passed on to our machine learning classifier for final identification.
'''

class LicensePlateDetector:
	
    def __init__(self, image, minPlateW=60, minPlateH=20, numChars=7, minCharW=40): # numChars might need to be modified to the country's plates

        self.image = image # image to detect license plates in
        self.minPlateW = minPlateW # min width of the plate region
        self.minPlateH = minPlateH # min height of the plate region
        self.numChars = numChars # number of characters to be detected in the plate, this is the number of characters our license plate has
        self.minCharW = minCharW # min width of the extracted characters, this is the min number of pixels wide a region must be to be considered a license plate character
