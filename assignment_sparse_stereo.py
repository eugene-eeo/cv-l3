#####################################################################

# Example : load, display and compute SGBM disparity
# for a set of rectified stereo images from a  directory structure
# of left-images / right-images with filesname DATE_TIME_STAMP_{L|R}.png

# basic illustrative python script for use with provided stereo datasets

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2017 Department of Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import os
import numpy as np
# from scipy import stats
from surf import match
from yolo2 import yolov3

# where is the data ? - set this to where you have it

master_path_to_dataset = "/home/eeojun/Downloads/TTBB-durham-02-10-17-sub10"; # ** need to edit this **
directory_to_cycle_left = "left-images";     # edit this if needed
directory_to_cycle_right = "right-images";   # edit this if needed

#####################################################################

# fixed camera parameters for this stereo setup (from calibration)

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 262.0;
image_centre_w = 474.5;

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns

skip_forward_file_pattern = ""; # set to timestamp to skip forward to

pause_playback = False; # pause until key press after each image

#####################################################################

# resolve full directory location of data set for left / right images

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);

# get a list of the left image files and sort them (by timestamp in filename)

left_file_list = sorted(os.listdir(full_path_directory_left));

#####################################################################

## depth_map : compute depth map (in metres) for a given disparity image

def depth_map(shape, imgL, imgR):
    depths = np.full(shape, np.nan, dtype=np.float32)

    B = stereo_camera_baseline_m
    f = camera_focal_length_px

    for left_kp, right_kp in match(imgL, imgR):
        lx, ly = left_kp.pt
        rx, ry = right_kp.pt

        plx = lx - image_centre_w
        ply = ly - image_centre_h
        prx = rx - image_centre_w
        pry = ry - image_centre_h

        d = ((plx - prx)**2 + (ply - pry)**2) ** 0.5
        if d != 0:
            depths[int(ly), int(lx)] = f * B / d
    return depths


def get_distance(depth_map, bounding_box):
    x0, x1, y0, y1 = bounding_box
    return np.nanmean(depth_map[y0:y1, x0:x1])
    # x, _ = stats.mode(depth_map[y0:y1, x0:x1], nan_policy='omit', axis=None)
    # return x[0]

# Does preprocessing of the grayscale images
def preprocess(imgL, imgR):
    imgL = cv2.bilateralFilter(imgL, 5, 50, 20)
    imgR = cv2.bilateralFilter(imgR, 5, 50, 20)

    grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

    # grayL = cv2.equalizeHist(grayL)
    # grayR = cv2.equalizeHist(grayR)
    # grayL = sharpen(grayL)
    # grayR = sharpen(grayR)

    # grayL = np.power(grayL, 0.85).astype('uint8');
    # grayR = np.power(grayR, 0.85).astype('uint8');

    # grayR = hist_match(grayR, grayL).astype(np.uint8)

    return grayL, grayR


for filename_left in left_file_list:

    # skip forward to start a file we specify by timestamp (if this is set)

    if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_left)):
        continue;
    elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left)):
        skip_forward_file_pattern = "";

    # from the left image filename get the correspondoning right image

    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);

    # for sanity print out these filenames

    print(full_path_filename_left);
    print(full_path_filename_right);
    print();

    # check the file is a PNG file (left) and check a correspondoning right image
    # actually exists

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

        imgL = imgL[0:400,:]
        imgR = imgR[0:400,:]

        print("-- files loaded successfully");
        print();

        # remember to convert to grayscale (as the disparity matching works on grayscale)
        # N.B. need to do for both as both are 3-channel images
        grayL, grayR = preprocess(imgL, imgR)

        cv2.imshow('grayL', grayL)
        cv2.imshow('grayR', grayR)

        # scale the disparity to 8-bit for viewing
        # divide by 16 and convert to 8-bit image (then range of values should
        # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
        # so we fix this also using a initial threshold between 0 and max_disparity
        # as disparity=-1 means no disparity available

        depths = depth_map(grayL.shape, grayL, grayR)

        tags = []

        for class_name, confidence, left, top, right, bottom in yolov3(imgL):
            # depth = np.nanmedian(depths[top:bottom,max(left, 0):right])
            depth = get_distance(depths, (max(left, 0), right, top, bottom))
            if np.isnan(depth):
                continue

            tags.append((depth, class_name, confidence, left, top, right, bottom))

        # Sort by z (depth) so we draw the back labels first
        tags.sort(reverse=True)

        for depth, class_name, confidence, left, top, right, bottom in tags:
            # construct label
            label = '%s (%.2fm)' % (class_name, depth)
            # draw a bounding box around matched section
            cv2.rectangle(imgL, (left, top), (right, bottom), (255, 178, 50), 2)
            # display the label at the top of the bounding box
            labelsize, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            top = max(top, labelsize[1])
            cv2.rectangle(
                imgL,
                (left, top - round(1.5 * labelsize[1])),
                (left + round(1.5 * labelsize[0]), top + baseline),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(imgL, label, (left, top), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0,0,0), 1)

        cv2.imshow('result', imgL)

        # keyboard input for exit (as standard), save disparity and cropping
        # exit - x
        # save - s
        # pause - space

        key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):       # exit
            break; # exit
        elif (key == ord('s')):     # save
            cv2.imwrite("left.png", imgL)
            cv2.imwrite("right.png", imgR)
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback);
    else:
        print("-- files skipped (perhaps one is missing or not PNG)");
        print();

# close all windows

cv2.destroyAllWindows()

#####################################################################
