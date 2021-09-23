import cv2 as cv 
import numpy as np 
import argparse                     # Argument Parser from Command Line
import posesetup as setup           # Contains Body Parts dictionary and Pose Pairs 
                                    # (for drawing lines)
from posefunctions import *

# Parse command line arguments and store in args
parser = argparse.ArgumentParser() 

parser.add_argument('--input', 
    help='Path to image/video. Leave blank to use frames from camera.')
parser.add_argument('--output', default='out.jpg', 
    help='Output path to save image/video. Leave blank to use frames from camera.')
parser.add_argument('--threshold', default=0.2, type=float,
    help='Threshold value for the heat map') 
parser.add_argument('--video', 
    help='Option to indicate a video has been input.') 
parser.add_argument('--width', default=368, type=int, 
    help='Resize input to specific width')
parser.add_argument('--height', default=368, type=int, 
    help='Resize input to specific height')

args = parser.parse_args()

# Read in pre-trained NN
nnet = cv.dnn.readNetFromTensorflow("graph_opt.pb")

# A VideoCapture object is constructed either by
    # 1) passing in the input (the image/video) given in the command line 
    #    as a parameter to the constructor, or else
    # 2) pass in 0, which denotes the default camera
vidcap = cv.VideoCapture(args.input if args.input else 0)

# The function waitKey waits for a key event indefinitely when the delay <= 0 or
# for delay milliseconds, when it is positive (hence we give ourselves 100 ms to
# press a key here) it returns the code of the pressed key or -1 if no key was
# pressed before the specified time had elapsed 
while cv.waitKey(100) < 0:
    # read grabs, decodes and returns the next video frame (given by image here)
    retval, image = vidcap.read()
    # retval is false if no frames have been grabbed
    if not retval: 
        cv.waitKey() 
        break
    
    # store height and width of image frame
    imageH = image.shape[0] 
    imageW = image.shape[1]
    
    # Obtain a 4-dimensional Mat object 
    # an array of "heatmaps", the probability of a body part being in location x, y
    nnoutput = process(image, nnet, args.width, args.height)

    outputW = nnoutput.shape[3]
    outputH = nnoutput.shape[2]
    
    # Find the locations of the body parts for the input image
    locations = [] 
    findBodyPartPositions(locations, nnoutput, imageW, imageH, 
            outputW, outputH, args.threshold)
   
    # Draw the skeleton lines for each of the POSE_PAIRS:
    drawSkeleton(locations, image)

    # Frame information for image
    addFrameInfo(image, nnet)

    # Write image to output given on command line, and display image.
    cv.imwrite(args.output, image) 
    cv.imshow('Pose Estimation using OpenCV', image)
