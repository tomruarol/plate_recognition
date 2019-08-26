# Plate Recognition

*Plate Recognition based in Python, Deep Learning and OpenCV.*

In this script we will:

 - Apply a perspective transform to extract the license plate region from the car for character segmentation.
 - Perform a connected component analysis on the license plate region to find character-like sections of the image.
 - Utilize contour properties to aide us in segmenting the foreground license plate characters from the background of the license plate.

Files in the script:

 - ***license_plate.py:*** *encapsulates all the methods we need to extract license plates and license plate characters from images.*