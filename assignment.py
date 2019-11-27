import cv2
import os
import numpy as np
from yolo2 import yolov3
from utils import annotate_image, mode, tiled_histogram_eq, compute_luma, is_valid_match

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

# skip_forward_file_pattern = "1506943191.487683"; # set to timestamp to skip forward to
skip_forward_file_pattern = ""
# skip_forward_file_pattern = "1506943221.487363";
# skip_forward_file_pattern = "1506943231.485679"
# skip_forward_file_pattern = "1506942484.480963"
# skip_forward_file_pattern = "1506942530.47524"
# skip_forward_file_pattern = "1506942630.475890"
# skip_forward_file_pattern = "1506942718.476805"

pause_playback = False; # pause until key press after each image

#####################################################################

# resolve full directory location of data set for left / right images

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);

# get a list of the left image files and sort them (by timestamp in filename)

left_file_list = sorted(os.listdir(full_path_directory_left));

#####################################################################


def get_distance_otsu(disparities, bounding_box):
    # Uses Otsu thresholding
    # Get distance from disparities and bounding_box
    f = camera_focal_length_px
    B = stereo_camera_baseline_m

    x0, x1, y0, y1 = bounding_box
    # Get an array of non-zero depths
    depths = disparities[y0:y1, x0:x1].ravel()
    depths = depths[depths > 0]
    if len(depths) == 0:
        return np.nan

    ret, _ = cv2.threshold(depths, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Hypothesis: the mode of a 'good' disparity map contains mostly
    # the correct values.
    depth_mode = mode(depths)
    depths = depths[depths > ret]
    # Be conservative here, we take the maximum disparity =>
    # minimum depth between the mode and the median.
    return (f * B) / max(depth_mode, np.percentile(depths, 50, interpolation='higher'))


def preprocess(imgL, imgR):
    # Does preprocessing of the colour images.
    # The output is a pair of corresponding images in grayscale.
    grayL = compute_luma(imgL)
    grayR = compute_luma(imgR)

    grayL = tiled_histogram_eq(grayL)
    grayR = tiled_histogram_eq(grayR)

    grayL = np.power(grayL, 0.75).astype('uint8')
    grayR = np.power(grayR, 0.75).astype('uint8')
    return grayL, grayR

#####################################################################


max_disparity = 128
left_matcher = cv2.StereoSGBM_create(0, max_disparity, 21)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(8000)
wls_filter.setSigmaColor(1.2)


for filename_left in left_file_list:

    # skip forward to start a file we specify by timestamp (if this is set)

    if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_left)):
        continue;
    elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left)):
        skip_forward_file_pattern = "";

    # from the left image filename get the correspondoning right image

    filename_right = filename_left.replace("_L", "_R")
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left)
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right)

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

        # compute disparity image from undistorted and rectified stereo images
        # that we have loaded
        # (which for reasons best known to the OpenCV developers is returned scaled by 16)
        displ = left_matcher.compute(grayL, grayR)
        dispr = right_matcher.compute(grayR, grayL)

        disparity = wls_filter.filter(displ, imgL, None, dispr)

        # divide by 16 and convert to 8-bit image (then range of values should
        # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
        # so we fix this also using a initial threshold between 0 and max_disparity
        # as disparity=-1 means no disparity available
        _, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)
        disparity_scaled = (disparity / 16.0).astype(np.uint8)
        disparity_scaled = disparity_scaled[0:390,:]

        tags = []
        for class_name, confidence, left, top, right, bottom in yolov3(imgL):
            left = max(left, 0)
            top = max(top, 0)
            if not is_valid_match(class_name, left, top, right, bottom):
                continue
            depth = get_distance_otsu(disparity_scaled, (left, right, top, bottom))
            # Ignore if we have a nan depth
            if np.isnan(depth):
                continue

            tags.append((depth, class_name, confidence, left, top, right, bottom))

        # Foreground first
        tags.sort(reverse=True)

        # display image (scaling it to the full 0->255 range based on the number
        # of disparities in use for the stereo part)
        disparity_display = (disparity_scaled * (256. / max_disparity)).astype(np.uint8)
        # cv2.imshow("disparity", disparity_display)

        annotate_image(tags, imgL)
        # cv2.imshow('result', imgL)

        # Find nearest object and print
        nearest = "No detected objects (0.0m)" if len(tags) == 0 else "%s (%.1fm)" % (tags[-1][1], tags[-1][0])
        print(filename_left)
        print(filename_right, ":", nearest)

        # cv2.imwrite("res/left_dense_%s" % (filename_left.replace("_L", "")), imgL)

        # keyboard input for exit (as standard), save disparity and cropping
        # exit - x
        # save - s
        # pause - space
        # next - n (in pause mode)

        quit = False
        while not quit:
            key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
            if (key == ord('x')):       # exit
                quit = True
                break; # exit
            elif (key == ord('s')):     # save
                cv2.imwrite("left_dense_%s" % (filename_left.replace("_L", "")), imgL)
                # cv2.imwrite("grayL.png", grayL)
                # cv2.imwrite("disparity.png", disparity_display)
            elif (key == ord(' ')):     # pause (on next frame)
                pause_playback = not(pause_playback)
            if pause_playback and key != ord('n'):
                continue
            break
        if quit:
            break
    else:
        print("-- files skipped (perhaps one is missing or not PNG)")
        print()

# close all windows

cv2.destroyAllWindows()

#####################################################################
