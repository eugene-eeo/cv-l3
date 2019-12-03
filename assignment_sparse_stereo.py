import cv2
import os
import numpy as np
from utils import annotate_image, tiled_histogram_eq, is_valid_match, compute_luma, preprocess_for_object_recognition
from surf import match, find_keypoints_and_descriptors
from yolo2 import yolov3

# where is the data ? - set this to where you have it

master_path_to_dataset = "tt"; # ** need to edit this **
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

def disparity_map(shape, l_keypoints, l_descriptors, r_keypoints, r_descriptors):
    # Computes a disparity map of a given shape in metres, given the ORB feature
    # point matches.
    depths = np.full(shape, np.nan, dtype=np.float32)
    for left_kp, right_kp in match(l_keypoints, l_descriptors, r_keypoints, r_descriptors):
        # Left and right x and y values (indices into the image)
        lx, ly = left_kp.pt
        rx, ry = right_kp.pt

        # Calculate disparity (d = |P_L - P_R|)
        d = ((lx - rx)**2 + (ly - ry)**2) ** 0.5
        if d > 0:
            depths[int(ly), int(lx)] = d
    return depths


def get_distance(disparity_map, bounding_box):
    B = stereo_camera_baseline_m
    f = camera_focal_length_px

    x0, x1, y0, y1 = bounding_box
    disps = disparity_map[y0:y1, x0:x1].ravel()
    disps = disps[~np.isnan(disps)]
    if len(disps) == 0:
        return np.nan
    return (f * B) / np.median(disps)


def preprocess(imgL, imgR):
    # Does preprocessing of the image pairs
    grayL = compute_luma(imgL)
    grayR = compute_luma(imgR)

    grayL = tiled_histogram_eq(grayL)
    grayR = tiled_histogram_eq(grayR)
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

    # check the file is a PNG file (left) and check a correspondoning right image
    # actually exists

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

        # remember to convert to grayscale (as the disparity matching works on grayscale)
        # N.B. need to do for both as both are 3-channel images
        grayL, grayR = preprocess(imgL, imgR)
        grayL = grayL[0:390, :]
        grayR = grayR[0:390, :]

        l_keypoints, l_descriptors, r_keypoints, r_descriptors = find_keypoints_and_descriptors(grayL, grayR)
        disparities = disparity_map(grayL.shape, l_keypoints, l_descriptors, r_keypoints, r_descriptors)

        imgL = preprocess_for_object_recognition(imgL)

        # grayL = cv2.drawKeypoints(grayL, l_keypoints, None, (73, 58, 215))
        # grayR = cv2.drawKeypoints(grayR, r_keypoints, None, (73, 58, 215))

        tags = []
        for class_name, confidence, left, top, right, bottom in yolov3(imgL):
            left = max(left, 0)
            top = max(top, 0)
            if not is_valid_match(class_name, left, top, right, bottom):
                continue
            depth = get_distance(disparities, (left, right, top, bottom))
            # Ignore if we have a nan depth
            if np.isnan(depth):
                continue

            tags.append((depth, class_name, confidence, left, top, right, bottom))

        # Sort by z (depth) so we draw the back labels first
        tags.sort(reverse=True)
        annotate_image(tags, imgL)

        # cv2.imshow('grayL', grayL)
        # cv2.imshow('grayR', grayR)
        cv2.imshow('result', imgL)

        # Find nearest object and print
        nearest = "No detected objects (0.0m)" if len(tags) == 0 else "%s (%.1fm)" % (tags[-1][1], tags[-1][0])
        print(filename_left)
        print(filename_right, ":", nearest)

        # cv2.imwrite("a/left_sparse_%s" % (filename_left.replace("_L", "")), imgL)

        # keyboard input for exit (as standard), save disparity and cropping
        # exit - x
        # save - s
        # pause - space
        # next - n (in pause mode)

        quit = False
        while not quit:
            key = cv2.waitKey(20 * (not(pause_playback))) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
            if (key == ord('x')):       # exit
                quit = True
                break; # exit
            elif (key == ord('s')):     # save
                cv2.imwrite("left_sparse_%s" % (filename_left.replace("_L", "")), imgL)
            elif (key == ord(' ')):     # pause (on next frame)
                pause_playback = not(pause_playback)
            if pause_playback and key != ord('n'):
                continue
            break
        if quit:
            break
    else:
        print("-- files skipped (perhaps one is missing or not PNG)");
        print();

# close all windows

cv2.destroyAllWindows()

#####################################################################
