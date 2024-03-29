'''
This script contains all the develop for license paltes recognition. 
Based on segmenting characters with openCV.
'''

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


    def detect(self):
		
        lpRegions = self.detectPlates() # detect plate regions in the image
 
        for lpRegion in lpRegions: # loop over the plate regions
			
            # detect character candidates in the current  plate region
            lp = self.detectCharacterCandidates(lpRegion)
 
            if lp.success: # only continue if characters were successfully detected

                yield (lp, lpRegion) # yield a tuple of the plate object and bounding box: <palte, bouncing_box>


    # TODO
    # We need to define the detectPlates function which detects license plate candidates in an input image.
    
    '''
    This method detects license plates candidates in an input image
    '''
    def detectPlates(self):
       
        # initialize the rectangular and square kernels to be applied to the image
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        regions = [] # initialize the list of license plate regions

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) # convert the image to grayscale
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel) # apply blackhat operation
        cv2.imshow("Blackhat", blackhat)

        # find regions in the image that are light
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKernel)
        light = cv2.threshold(light, 50, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow("Light", light)

        # compute the Scharr gradient representation of the blackhat image in the x-direction,
        # and scale the resulting image into the range [0, 255]
        gradX = cv2.Sobel(blackhat, ddepth=cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        cv2.imshow("Gy", gradX)

        # blur the gradient representation, apply a closing operating, and threshold the
        # image using Otsu's method
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv2.imshow("Thresh", thresh)

        # perform a series of erosions and dilations on the image
        '''
        Morphological transformations are some simple operations based on the image shape.
        Dilatation engorda la imagen (como si el boli/lapiz fuese mas gordo)
        Erosion hace justo lo contrario
        '''
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cv2.imshow("E&D", thresh)

        # take the bitwise 'and' between the 'light' regions of the image, then perform
        # another series of erosions and dilations
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        cv2.imshow("Bitwise AND, E&D", thresh)

        # find contours in the thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        for c in cnts: # loop over the contours

            # grab the bounding box associated with the contour and compute the area and aspect ratio
            (w, h) = cv2.boundingRect(c)[2:]
            aspectRatio = w / float(h)

            # compute the rotated bounding box of the region
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.cv.BoxPoints(rect)) if imutils.is_cv2() else cv2.boxPoints(rect)

            # ensure the aspect ratio, width, and height of the bounding box fall within
            # tolerable limits, then update the list of license plate regions
            if (aspectRatio > 3 and aspectRatio < 6) and h > self.minPlateH and w > self.minPlateW:regions.append(box)

        return regions


    '''
    This function accepts a plate region, applies image processing techniques, and then segments the foreground license plate characters from the background.
    '''
    def detectCharacterCandidates(self, region): # region is a rotated bouncing_box

		# apply a 4-point transform to extract the plate as if we had a 90-degree viewing angle
        plate = perspective.four_point_transform(self.image, region)
        cv2.imshow("Perspective Transform", imutils.resize(plate, width=400))

		# extract the Value component from the HSV color space and apply adaptive thresholding to reveal the characters on the plate
        V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2] # extract the Value channel from the HSV color space
        '''
        Why Value Channel instead of GrayScale?

        The grayscale version of an image is a weighted combination of the RGB channels. 
        The Value channel, however, is given a dedicated dimension in the HSV color space. When performing thresholding to extract dark regions 
        from a light background (or vice versa), better results can often be obtained by using the Value rather than grayscale.
        '''
        T = threshold_local(V, 29, offset=15, method="gaussian") # apply adaptive thresholding to reveal the characters on the plate
        thresh = (V > T).astype("uint8") * 255
        thresh = cv2.bitwise_not(thresh)
        '''
        Image Thresholding: Classify pixels as "dark" or "light"
        Adaptive Thresholding: Form of image thresholding that takes into account spatial variations in illumination
        '''

		# resize the plate region to a canonical size
        plate = imutils.resize(plate, width=400)
        thresh = imutils.resize(thresh, width=400)
        cv2.imshow("Thresh", thresh)

        labels = measure.label(thresh, neighbors=8, background=0) # perform a connected components analysis
        charCandidates = np.zeros(thresh.shape, dtype="uint8") # mask to store the locations of the character candidates

        for label in np.unique(labels): # loop over the unique components

            if label == 0: # label corresponds to the background of the plate, so we can ignore it
                continue
 
			# otherwise, construct the label mask to display only connected components for the
			# current label, then find contours in the label mask.
            # By performing this masking, we are revealing only pixels that are part of the current connected component.
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255 # draw all pixels with the current  label  value as white on a black background
            cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find contours in the label mask
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]

            if len(cnts) > 0: # check that at least one contour was found in the  labelMask

                c = max(cnts, key=cv2.contourArea) # grab the largest contour which corresponds to the component in the mask
                (boxX, boxY, boxW, boxH) = cv2.boundingRect(c) # compute the bounding box for the contour

				# compute the aspect ratio, solidity, and height ratio for the component
                aspectRatio = boxW / float(boxH) #  the ratio of the bounding box width to the bounding box height
                solidity = cv2.contourArea(c) / float(boxW * boxH)
                heightRatio = boxH / float(plate.shape[0]) # the ratio of the bounding box height to the license plate height
                # Large values of  heightRatio  indicate that the height of the (potential) character is similar to the license plate itself (and thus a likely character).

				# determine if the aspect ratio, solidity, and height of the contour pass the rules tests
                keepAspectRatio = aspectRatio < 1.0 # We want aspectRatio to be at most square, ideally taller rather than wide since most characters are taller than they are wide.
                keepSolidity = solidity > 0.15 # We want solidity to be reasonably large, otherwise we could be investigating “noise”, such as dirt, bolts, etc. on the license plate.
                keepHeight = heightRatio > 0.4 and heightRatio < 0.95 # We want our keepHeight  ratio to be just the right size.

                if keepAspectRatio and keepSolidity and keepHeight: # check to see if the component passes all the tests
					
                    # We take the contour, compute the convex hull (to ensure the entire bounding region of the character is included in the contour), 
                    # and draw the convex hull on our  charCandidates  mask. 
                    hull = cv2.convexHull(c) # compute the convex hull of the contour
                    cv2.drawContours(charCandidates, [hull], -1, 255, -1) # draw it on the character candidates mask

        charCandidates = segmentation.clear_border(charCandidates) # clear pixels that touch the borders of the character candidates mask and detect contours in the candidates mask

		# TODO:
		# There will be times when we detect more than the desired number of characters it would be wise to apply a method to 'prune' the unwanted characters

        return LicensePlate(success=True, plate=plate, thresh=thresh, candidates=charCandidates)