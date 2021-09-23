import cv2 as cv 
import numpy as np 
import argparse                     # Argument Parser from Command Line
import posesetup as setup           # Contains Body Parts dictionary and Pose Pairs 

def process(image, nnet, width, height):
    # cv.dnn.blobFromImage() params:
        # image: input image (with 1,3 or 4 channels) 
        # scalefactor: a multipler for image values 
            # (1.0 here but depends on nn that has been trained) 
        # size: spatial size for output image 
        # mean: scalar with mean values which are subtracted from channels 
        # swapRB: True means swap BGR to RGB; False by default
        # crop: indicates whether image will be cropped after resize or not
    nninput = cv.dnn.blobFromImage(image, 1.0, (width, height), 
                (0, 0, 0), swapRB=False, crop=False) 
    nnet.setInput(nninput)
    return nnet.forward()

def findBodyPartPositions(locations, nnoutput, imageW, imageH, outputW, outputH, thresh):
    for i in range(len(setup.BODY_PARTS)):
        # Slice heatmap from output for the body part 
        # first axis is 0 as we input only one image
        heatMap = nnoutput[0, i, :, :]
 
        # Finds the global minimum and maximum element values and their positions 
            # (does not work with a multi-channel array) 
            # (hence only a single pose can be detected this way)
        minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(heatMap)
  
        # Scale to image size
        W = (imageW * maxLoc[0]) / outputW
        H = (imageH * maxLoc[1]) / outputH

        # Add point pair if maxVal is higher than the given threshold
        if maxVal > thresh:
            locations.append((int(W), int(H))) 
        else:
            locations.append(None)

def drawSkeleton(locations, image):
    for pair in setup.POSE_PAIRS:
        # Lookup the index in the BODY_PARTS dictionary 
        idxFrom = setup.BODY_PARTS[pair[0]] 
        idxTo = setup.BODY_PARTS[pair[1]]

        # Only draw lines between locations that are in the locations array
        if locations[idxFrom] and locations[idxTo]:
            # cv.line() params: 
                # frame, pt1, pt2, Scalar(colour:BGR), thick=1,linetype=8, shift=0)
            cv.line(image, locations[idxFrom], locations[idxTo], (0, 255, 255), 4)

            # cv.ellipse() params: 
                # frame, center, axes, angle, startAngle, endAngle, 
                # colour[, thickness[, linetype[, shift]]]
            cv.ellipse(image, locations[idxFrom], (3,3), 0, 0, 360, 
                    (0, 0, 255), cv.FILLED) 
            cv.ellipse(image, locations[idxTo], (3,3), 0, 0, 360, 
                    (0, 0, 255), cv.FILLED)

def addFrameInfo(image, nnet):
    # Frame information for image
    t, _ = nnet.getPerfProfile() 
    freq = cv.getTickFrequency() / 1000
    cv.putText(image, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 0)) 
