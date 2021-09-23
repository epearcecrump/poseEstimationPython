import cv2 as cv 
import numpy as np 
import argparse                     # Argument Parser from Command Line
import posesetup as setup           # Contains Body Parts dictionary and Pose Pairs 
                                    # (for drawing lines)

def process(image, nnet, width, height):
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
    
    # cv.dnn.blobFromImage() params:
        # image: input image (with 1,3 or 4 channels) 
        # scalefactor: a multipler for image values 
            # (1.0 here but depends on nn that has been trained) 
        # size: spatial size for output image 
        # mean: scalar with mean values which are subtracted from channels 
        # swapRB: True means swap BGR to RGB; False by default
        # crop: indicates whether image will be cropped after resize or not
    #nninput = cv.dnn.blobFromImage(image, 1.0, (args.width, args.height), 
    #            (0, 0, 0), swapRB=False, crop=False) 
    #nnet.setInput(nninput)

    # Obtain a 4-dimensional Mat object 
        # an array of "heatmaps", the probability of a body part 
        # being in location x, y
    #nnoutput = nnet.forward()
    nnoutput = process(image, nnet, args.width, args.height)

    outputW = nnoutput.shape[3]
    outputH = nnoutput.shape[2]
    
    # Find the locations of the body parts for the input image
    locations = [] 
    findBodyPartPositions(locations, nnoutput, imageW, imageH, outputW, outputH, args.threshold)

    #for i in range(len(setup.BODY_PARTS)):
        # Slice heatmap from output for the body part 
        # first axis is 0 as we input only one image
        #heatMap = nnoutput[0, i, :, :]
 
        # Finds the global minimum and maximum element values and their positions 
            # (does not work with a multi-channel array) 
            # (hence only a single pose can be detected this way)
        #minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(heatMap)
  
        # Scale to image size
        #W = (imageW * maxLoc[0]) / nnoutput.shape[3] 
        #H = (imageH * maxLoc[1]) / nnoutput.shape[2]

        # Add point pair if maxVal is higher than the given threshold
        #if maxVal > args.threshold: 
        #    locations.append((int(W), int(H))) 
        #else:
        #    locations.append(None)
   
    # Draw the skeleton lines for each of the POSE_PAIRS:
    drawSkeleton(locations, image)

    #for pair in setup.POSE_PAIRS:
        # Lookup the index in the BODY_PARTS dictionary 
    #    idxFrom = setup.BODY_PARTS[pair[0]] 
    #    idxTo = setup.BODY_PARTS[pair[1]]

        # Only draw lines between locations that are in the locations array
    #    if locations[idxFrom] and locations[idxTo]:
            # cv.line() params: 
                # frame, pt1, pt2, Scalar(colour:BGR), thick=1,linetype=8, shift=0)
    #        cv.line(image, locations[idxFrom], locations[idxTo], (0, 255, 255), 4)

            # cv.ellipse() params: 
                # frame, center, axes, angle, startAngle, endAngle, 
                # colour[, thickness[, linetype[, shift]]]
    #        cv.ellipse(image, locations[idxFrom], (3,3), 0, 0, 360, 
    #                (0, 0, 255), cv.FILLED) 
    #        cv.ellipse(image, locations[idxTo], (3,3), 0, 0, 360, 
    #                (0, 0, 255), cv.FILLED)

    # Frame information for image
    t, _ = nnet.getPerfProfile() 
    freq = cv.getTickFrequency() / 1000
    cv.putText(image, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 0)) 

    # Write image to output given on command line, and display image.
    cv.imwrite(args.output, image) 
    cv.imshow('Pose Estimation using OpenCV', image)
