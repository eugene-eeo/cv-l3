import cv2
import os
import numpy as np
from yolo2 import yolov3
from utils import annotate_image, USEFUL_NAMES, mode, equalise_hist
import matplotlib.pyplot as plt

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
skip_forward_file_pattern = "1506943221.487363";
skip_forward_file_pattern = "1506943231.485679"
skip_forward_file_pattern = "1506942484.480963"
skip_forward_file_pattern = ""

pause_playback = True; # pause until key press after each image

#####################################################################

# resolve full directory location of data set for left / right images

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);

# get a list of the left image files and sort them (by timestamp in filename)

left_file_list = sorted(os.listdir(full_path_directory_left));

#####################################################################


def z_buffer_update(disparities, bounding_box, threshold):
    x0, x1, y0, y1 = bounding_box
    disp = disparities[y0:y1, x0:x1]
    disp[disp > threshold] = 0


def get_distance_otsu(disparities, bounding_box):
    # Uses Otsu thresholding
    # Get distance from disparities and bounding_box
    f = camera_focal_length_px
    B = stereo_camera_baseline_m

    x0, x1, y0, y1 = bounding_box
    # Get an array of non-zero depths
    depths = disparities[y0:y1, x0:x1].ravel()
    useful = depths > 0

    # ~80% not useful information, just discard...
    if useful.sum() / len(depths) < 0.20:
        return np.nan, None

    depths = depths[useful]
    ret, thresholds = cv2.threshold(depths, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Hypothesis: the mode of a 'good' disparity map contains mostly
    # the correct values.
    depth_mode = mode(depths)
    depths = depths[depths > ret]
    # Be conservative here, we take the maximum disparity =>
    # minimum depth between the mode and the median.
    return (f * B) / max(np.median(depths), depth_mode), ret


def preprocess(imgL, imgR):
    # Does preprocessing of the colour images.
    # The output is a pair of corresponding images in grayscale.
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2HSV)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2HSV)

    imgL[:,:,2] = np.power(imgL[:,:,2], 0.75).astype('uint8')
    imgR[:,:,2] = np.power(imgR[:,:,2], 0.75).astype('uint8')

    imgL = equalise_hist(imgL)
    imgR = equalise_hist(imgR)

    # imgL = cv2.bilateralFilter(imgL_hsv, 5, 50, 50)
    # imgR = cv2.bilateralFilter(imgR_hsv, 5, 50, 50)

    grayL = imgL[:, :, 2]
    grayR = imgR[:, :, 2]

    # grayL = sharpen(grayL, grayL)
    # grayR = sharpen(grayR, grayR)

    # grayL = cv2.equalizeHist(grayL)
    # grayR = cv2.equalizeHist(grayR)

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

    # for sanity print out these filenames

    print(full_path_filename_left)
    print(full_path_filename_right)
    print()

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

        print("-- files loaded successfully")
        print()

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

        # Z-Buffering algorithm:
        #  - We compute first a list of predicted depths and bounding boxes (tags).
        #  - We then sort by the depth in ascending order.
        #  - Go through the tags again, and then recompute the distances. This time,
        #    we remove any depth information we use during computation (set it to 0).
        #  - This should mean that we have better depth information...
        tags = []
        for class_name, confidence, left, top, right, bottom in yolov3(imgL):
            if class_name in USEFUL_NAMES:
                left = max(left, 0)
                depth, _ = get_distance_otsu(disparity_scaled, (left, right, top, bottom))

                # Ignore if we have a nan depth
                if np.isnan(depth):
                    continue

                tags.append((depth, class_name, confidence, left, top, right, bottom))

        # Foreground first
        tags.sort()
        disparity_z_buffer = np.copy(disparity_scaled)
        new_tags = []
        for i, (_, class_name, confidence, left, top, right, bottom) in enumerate(tags):
            depth, threshold = get_distance_otsu(disparity_z_buffer, (left, right, top, bottom))
            if np.isnan(depth):
                continue

            z_buffer_update(disparity_z_buffer, (left, right, top, bottom), threshold)
            new_tags.append((depth, class_name, confidence, left, top, right, bottom))

        # Sort by z (depth) so we draw the back labels first
        new_tags.sort(reverse=True)

        # display image (scaling it to the full 0->255 range based on the number
        # of disparities in use for the stereo part)
        disparity_display = (disparity_scaled * (256. / max_disparity)).astype(np.uint8)
        cv2.imshow("disparity", disparity_display)

        annotate_image(new_tags, imgL)
        cv2.imshow('result', imgL)
        # cv2.imshow('grayL', grayL)
        # cv2.imshow('grayR', grayR)

        # plt.figure(1)
        # plt.axis("off")
        # plt.imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
        # plt.tight_layout()

        # fig, axes = plt.subplots(len(new_tags), 2, gridspec_kw={'hspace': 0, 'wspace': 0}, sharex=True, squeeze=False)
        # max_frame_disp = max([np.nanmax(disparity_scaled[top:bottom, left:right]) for (_, _, _, left, top, right, bottom) in tags] + [0])
        # for i, (depth, class_name, _, left, top, right, bottom) in enumerate(new_tags):
        #     # Plotting histogram of these disparities
        #     disps = buffers[i]
        #     disps = disps[disps > 0].ravel()
        #     lax = axes[i, 0]
        #     ret, _ = cv2.threshold(disps, 0, max_disparity, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #     depth_mode = mode(disps)

        #     less, more = ('white', '#85BB65') if depth_mode >= ret else ('#85BB65', 'white')

        #     (n, bins, patches) = lax.hist(disps, bins=range(0, max_frame_disp + 1), density=True, stacked=True, linewidth=1, edgecolor='black', color=less)
        #     lax.set_ylim([0, 1])
        #     lax.text(0.5,0.85,"%s (%.2fm)" % (class_name, depth),
        #              horizontalalignment='center',
        #              transform=lax.transAxes)
        #     lax.label_outer()
        #     lax.grid(True)
        #     for j, patch in zip(range(max_disparity + 1), patches):
        #         if j > ret:
        #             patch.set_color(more)
        #             patch.set_linewidth(1)
        #             patch.set_edgecolor('black')

        #     rax = axes[i, 1]
        #     disps = disparity_scaled[top:bottom, left:right].ravel()
        #     disps = disps[disps > 0]
        #     rax.hist(disps, bins=range(0, max_frame_disp + 1), density=True, stacked=True, linewidth=1, edgecolor='black', color='white')
        #     rax.set_ylim([0, 1])
        #     rax.text(0.5,0.85,"%s (%.2fm)" % (class_name, depth),
        #              horizontalalignment='center',
        #              transform=lax.transAxes)
        #     rax.label_outer()
        #     rax.grid(True)

        # fig.tight_layout()
        # plt.show()

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
                cv2.imwrite("left.png", imgL)
                cv2.imwrite("right.png", imgR)
                cv2.imwrite("disparity.png", disparity_display)
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
