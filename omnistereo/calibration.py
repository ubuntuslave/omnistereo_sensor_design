# -*- coding: utf-8 -*-
# calibration.py

# Copyright (c) 20012-2016, Carlos Jaramillo
# Produced at the Laboratory for Robotics and Intelligent Systems of the City College of New York
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# from sympy_test import GUMsymbolic
from __future__ import division
from __future__ import print_function

import warnings
import cv2
import numpy as np
import omnistereo.transformations as tr
from omnistereo import common_cv
import sys

def draw_detected_points_manually(img_input, points, num_valid_points=0, show=False):
    '''
    Creates an omni_image with the selected points drawn based on the num_valid_points parameter
    @param img_input: The input omni_image
    @param points: The set of points to be used in the drawing
    @param num_valid_points: If 0 (default), all points are drawn.
                             If positive, num_valid_points points are drawn from the first point in points.
                             If negative, num_valid_points points are drawn in reverse order, starting from the last in points.
    @param show: Indicates whether to show the omni_image in a window.
    @return: the omni_image with drawn points.

    @todo: combine into common_cv draw_points function
    '''
    from time import time
    time_now_secs = str(time())
    thickness = 2
    color = (0, 255, 0)  # Green because BGR(B,G,R)

    points_listed = points.reshape(-1, 2)  # Convert to a 2D numpy ndarray

    if num_valid_points == 0:
        points_range = range(len(points_listed))
    elif num_valid_points < 0:
        points_range = range(len(points_listed) + num_valid_points, len(points_listed))
    else:
        points_range = range(num_valid_points)


    channels = img_input.ndim
    if channels < 3:
        vis = cv2.cvtColor(img_input, cv2.COLOR_GRAY2BGR)
    else:
        vis = img_input.copy()

    for i in points_range:
        pt = points_listed[i]
        pt_as_tuple = tuple(pt)  # (pt[0],pt[1]) # (x or col,y or row)
        cv2.circle(vis, pt_as_tuple, 2, color, thickness, 8, 0)

    if show:
        manual_drawing_window_name = 'Manual Points Drawing ' + time_now_secs
        cv2.namedWindow(manual_drawing_window_name)
        cv2.imshow(manual_drawing_window_name, vis)
        cv2.waitKey(1)

    return vis


def get_masked_omni_image_manual(img_input, crop_result=False):
    '''
    This mask is rectangular and usually used for masking the checkerboard pattern during corner extraction
    '''
    manual_mask_window_name = 'Manual Mask (Click At least 2 Points to mask about)'
    cv2.namedWindow(manual_mask_window_name, cv2.WINDOW_NORMAL)
    pt_clicker = common_cv.PointClicker(manual_mask_window_name, max_clicks=4, draw_polygon_clicks=True)
    mask_points = pt_clicker.get_clicks_uv_coords(img_input)
    valid_manual_mask = pt_clicker.click_counter > 1
    roi_position = np.array([0., 0.])
    cv2.destroyWindow(manual_mask_window_name)

    if valid_manual_mask:
        masked_img = common_cv.create_arbitrary_mask(img_input, mask_points, preview=False)

        if crop_result:
            min_point_position = np.min(mask_points, 0)
            max_point_position = np.max(mask_points, 0)
            roi_position = min_point_position
            # Crop from row (or y):y+height_offset, col (or x), x+width_offset
            masked_img = masked_img[min_point_position[1]:max_point_position[1], min_point_position[0]:max_point_position[0]]
        return masked_img, valid_manual_mask, roi_position
    else:
        return img_input, valid_manual_mask, roi_position

def find_corner_points(img, pattern_size, corner_finder_flags, reduce_pattern_size=True):
    '''
    @param pattern_size: Tuple containing (number_of_rows, number_of_columns) for the pattern
    @return: a ndarray of shape (n, 2) for the (u,v) coordinates of the n 2D points (corners) found
    '''
    corners_result = None
    custom_pattern_size = pattern_size  # Start with the base pattern size

    # Note: this OpenCV function takes the pattern size as (points_per_row or width as number of points, points_per_colum or height as number of points)
    found, corners = cv2.findChessboardCorners(img, (custom_pattern_size[1], custom_pattern_size[0]), flags=corner_finder_flags)
    if reduce_pattern_size:
        resizing_counter = 0
        while not found and custom_pattern_size[0] > 3 and custom_pattern_size[1] > 3:
            # Reduce size of pattern and try again
            if resizing_counter % 2 == 0:
                saved_custom_pattern_size = custom_pattern_size
                custom_pattern_size = (saved_custom_pattern_size[0], saved_custom_pattern_size[1] - 1)  # Decrease the number of columns
            else:
                custom_pattern_size = (saved_custom_pattern_size[0] - 1, saved_custom_pattern_size[1])  # Decrease the number of rows
            print("Trying with", custom_pattern_size, "pattern size...")
            found, corners = cv2.findChessboardCorners(img, (custom_pattern_size[1], custom_pattern_size[0]), flags=corner_finder_flags)
            if resizing_counter % 2 != 0 and not found:
                custom_pattern_size = (saved_custom_pattern_size[0], saved_custom_pattern_size[1] - 1)
            resizing_counter += 1

        if not found:  # Now, attempt to reduce only the columns
            custom_pattern_size = pattern_size  # Start with the base pattern size
            while not found and custom_pattern_size[1] > 3:
                custom_pattern_size = (custom_pattern_size[0], custom_pattern_size[1] - 1)  # Decrease the number of columns
                print("Trying with", custom_pattern_size, "pattern size...")
                found, corners = cv2.findChessboardCorners(img, (custom_pattern_size[1], custom_pattern_size[0]), flags=corner_finder_flags)

        if not found:  # Attempt to reduce only the rows
            custom_pattern_size = pattern_size  # Start with the base pattern size
            while not found and custom_pattern_size[0] > 3:
                custom_pattern_size = (custom_pattern_size[0] - 1, custom_pattern_size[1])  # Decrease the number of rows
                print("Trying with", custom_pattern_size, "pattern size...")
                found, corners = cv2.findChessboardCorners(img, (custom_pattern_size[1], custom_pattern_size[0]), flags=corner_finder_flags)

    if found:  # Refine found corners with subpixel accuracy
        print("Refining...", end="")
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        # NOTE: Be careful! A (5x5) window may be too big or too small?
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        corners_result = corners.reshape(-1, 2)

    return found, corners_result, custom_pattern_size

def extract_corners_single(image, img_number=None, reduce_pattern_size=False, **kwargs):
    '''
    @brief Corners are extracted from the omni_image of a chessboard pattern using OpenCV methods
    @param omni_image: A single omni_image (numpy array) where corner points will be extracted from
    @param img_number: (Optional) The number of omni_image from the list to work with.
    @param kwargs: Possible keyworded arguments are:
            [--save] [--show <'Boolean'>] [--show_cropped <'Boolean'>][--show_omni_mask <'Boolean'>][--save_dir <output path>] [--output_prefix_name <prefix name>] [--pattern_size <"(rows, cols)">] [--square_size <lenght in meters>]
    @return: The resulting calibration omni_image structure (instance of OmniMono class)
    @todo:  For now, it is saving with the same filenames as a BMP.
    '''

    if kwargs.get("sys_args"):
        args_dict = kwargs.get("sys_args")
    else:
        args_dict = kwargs

    save_dir = args_dict.get('--save_dir')
    save_images = '--save' in args_dict.keys()
    # I don't like the inconsistency!:
    # Should I use:
    #     column-major order (rows by cols) as for the size of a matrix?
    # or it's better
    #     row-major order (u by v) == (width by height) == (x by y) as OpenCV does
    pattern_size = eval(args_dict.get('--pattern_size', (6, 9)))  # (rows, cols)
    square_size = float(args_dict.get('--square_size', 1.0))
    show_patterns = eval(args_dict.get('--show', 'False'))
    show_cropped = eval(args_dict.get('--show_cropped', 'False'))
    show_omni_mask = eval(args_dict.get('--show_omni_mask', 'False'))
    output_prefix_name = args_dict.get('--output_prefix_name', 'omni_image')

    # CALIB_CB_FAST_CHECK saves a lot of time on omni_image but it misses points
    # that do not contain any chessboard corners
    # Decent compromise between speed and corners found
    corner_finder_flags_fast = cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE  # Default by OpenCV
    corner_finder_flags_slow = cv2.CALIB_CB_ADAPTIVE_THRESH

    # Generate the 3D points on the flat chessboard pattern
    # Initialize a full general pattern that relates to all the possible points. These will be truncated for the respective corners detected for each omni_image.
    pattern_3D_indices = np.zeros((np.prod(pattern_size), 3), np.float32)
    # ATTENTION: Pattern is on its XZ-plane!!!!
    # Recall that their Z value will be 0 w.r.t. the pattern's frame, and origin is the first valid corner (not the actual pattern board corner)
    pattern_indices = np.indices((pattern_size[1], pattern_size[0])).T.reshape(-1, 2)
    pattern_3D_indices[:, [0, 2]] = pattern_indices
    # TODO: make it work with something that is not always square (such as using the widthx, and widthy of the pattern's cells)
    pattern_all_points = pattern_3D_indices
    pattern_all_points *= square_size
    # FIXME: Do we WANT this offset? or should we assume the
    pattern_all_points[:, [0, 2]] += square_size  # Because the corners at the border aren't intersections, so an offset to the first intersection corner is required.

    # Reshape the np array to represent the 2D spatial order of the points in M rows and N columns
    # The row-wise order for the 3D Object Point coordinates look like this:
    # [P1(row1,col1), P2(row1,col2), P3(row1,col3), ..., PN(row1,colN)]
    # [PN+1(row2,col1), PN+2(row2,col2), PN+3(row2,col3), ...,P2N(row2,colN)]
    # :  .....
    # [(rowM,col1), (rowM,col2),(rowM,col3), ... , PMN(rowM, colN)]
    pattern_all_points = pattern_all_points.reshape(pattern_size[0], pattern_size[1], -1)  # CHECKME:

    mono = OmniMono(image)

    if img_number is not None: print('processing omni_image %s...' % (img_number))
    if show_omni_mask:
        omni_img_masked_name = "Omni %d (MASKED)" % (img_number)
        cv2.namedWindow(omni_img_masked_name, cv2.WINDOW_NORMAL)
        cv2.imshow(omni_img_masked_name, image)
        cv2.waitKey(0)
        cv2.destroyWindow(omni_img_masked_name)

    if image.ndim == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image.copy()

    # Sharpen up image a little bit
    img_blurred = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=3)
    img = cv2.addWeighted(img, 1.5, img_blurred, -0.5, gamma=0)

    custom_pattern_size = pattern_size  # Start with the base pattern size

    found, corners, custom_pattern_size = find_corner_points(img, custom_pattern_size, corner_finder_flags_fast, reduce_pattern_size=reduce_pattern_size)
    # proceed with manual masking:
    valid_manual_mask = True
    while (not found) and valid_manual_mask:  # Pressing the Escape key breaks the loop
        if img_number is not None: print('Re-processing omni_image %s using manual mask...' % (img_number))
        custom_pattern_size = pattern_size  # Start with the base pattern size
        manually_masked_img, valid_manual_mask, roi_position = get_masked_omni_image_manual(img, crop_result=True)
        if valid_manual_mask:
            # It will be more meticulous and use slower detection flags
            found, corners, custom_pattern_size = find_corner_points(manually_masked_img, custom_pattern_size, corner_finder_flags_slow, reduce_pattern_size=reduce_pattern_size)
            if found:  # And because we are cropping
                roi_position = roi_position.astype(corners.dtype)  # Make sure data types are compatible
                corners = corners + roi_position

    if found:
        print("found!")
        # Reshape the np array of corners to match the 2D spatial order of the object points in M rows and N columns
        corners = corners.reshape(custom_pattern_size[0], custom_pattern_size[1], -1)
        mono.set_points_info(found, image_points=corners, obj_points=pattern_all_points, pattern_size_applied=custom_pattern_size)

        if save_images or show_patterns:
            if img_number == None:
                img_name = '_%d-with_detected_corners.bmp'
            else:
                img_name = '_%d-with_detected_corners.bmp' % (img_number)
            result_img_name = output_prefix_name.strip() + img_name
            if show_patterns:
                vis = mono.visualize_points(result_img_name, show_cropped)
            if save_images:
                if save_dir:
                    complete_save_name = save_dir + "/" + result_img_name
                else:
                    complete_save_name = result_img_name

                print('Saving', complete_save_name)
                cv2.imwrite(complete_save_name, vis)
    else:
        print('chessboard not found')

    return mono

def extract_corners_batch(images, img_indices=[], reduce_pattern_size=False, **kwargs):
    '''
    @brief Corners are extracted from the omni_image of a chessboard pattern using OpenCV methods
    @param images: A list of images (numpy arrays) where corner points will be extracted from
    @param img_indices: A list of indices of the images from the list to work with. If the list is empty, then all of the images will be considered.
    @param kwargs: Possible keyworded arguments are:
            [--save] [--show <'Boolean'>] [--show_cropped <'Boolean'>][--save_dir <output path>] [--output_prefix_name <prefix name>] [--pattern_size <"(rows, cols)">] [--square_size <lenght in meters>]
    @todo:  For now, it is saving with the same filenames as a BMP.

    @return: A list of the resulting calibration images (instances of OmniMono class)
    '''

    l = len(images)

    if l < 1:
        print("No images passed")
        print("Exiting from", __name__)
        sys.exit(1)

    omni_monos = l * [None]
    if img_indices == None or len(img_indices) == 0:
        img_indices = range(l)  # Use all images

    for i in img_indices:
        try:
            if images[i] is not None:
                omni_monos[i] = extract_corners_single(images[i], i, reduce_pattern_size, **kwargs)
        except:
            warnings.warn("Warning...corner extraction of omni_image index %d failed at %s" % (i, __name__))

    return omni_monos

def extract_corners_from_files(*args, **kwargs):
    '''
    @brief Corners are extracted from a batch of file images of a chessboard pattern using OpenCV methods
    @param args: List of possible parameters: [<omni_image filename mask]: The omni_image filename to be used as a template, for example some_path/img*.png
    @param kwargs: Possible keyworded arguments are: [--save] [--show <'Boolean'>] [--save_dir <output path>] [--pattern_size <"(rows, cols)">] [--square_size <lenght in meters>]
    @todo: For now, it is saving with the same filenames as a BMP

    @retval image_points: An "appended" list of numpy ndarrays filled with the [u, v] coordinates for the extracted corner points, where each row of the ndarray belongs to a single point per row
    @retval obj_points: An "appended" list of numpy arrays (in table format of rows by columns) corresponding to the 3D points on the flat chessboard pattern that were extracted from the omni_image.
                        Recall that their Z value will be 0 in reference to a pattern's frame, and the origin is the first corner point (instead of the beginning of the pattern board)
    '''
    from glob import glob

    if kwargs.get("sys_args"):
        args_dict = kwargs.get("sys_args")
    else:
        args_dict = kwargs

    try: filename_mask = args[0]
    except:
        print("No file names passed")
        print("Exiting from", __name__)
        sys.exit(1)

    print("Using filename mask:", filename_mask)

    # WISH: fix ugly names for params with trailing -- due to getopt adding that to the key name!
    img_names = glob(filename_mask)
    save_dir = args_dict.get('--save_dir')
    save_images = '--save' in args_dict.keys()
    pattern_size = eval(args_dict.get('--pattern_size', "(6, 9)"))
    square_size = float(args_dict.get('--square_size', 1.0))
    show_patterns = eval(args_dict.get('--show', 'False'))

    # Generate the 3D points on the flat chessboard pattern
    # Recall that their Z value will be 0 w.r.t. the pattern's frame, and origin is the first valid corner (not the actual pattern board corner)
    pattern_all_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_all_points[:, :2] = np.indices((pattern_size[1], pattern_size[0])).T.reshape(-1, 2)
    pattern_all_points *= square_size
    # Reshape the np array to represent the 2D spatial order of the points in M rows and N columns
    # The row-wise order for the 3D Object Point coordinates look like this:
    # [P1(row1,col1), P2(row1,col2), P3(row1,col3), ..., PN(row1,colN)]
    # [PN+1(row2,col1), PN+2(row2,col2), PN+3(row2,col3), ...,P2N(row2,colN)]
    # :  .....
    # [(rowM,col1), (rowM,col2),(rowM,col3), ... , PMN(rowM, colN)]
#     pattern_all_points = pattern_all_points.reshape(pattern_size[0], pattern_size[1], -1)  # CHECKME: why am I inverting as cols, rows


    obj_points = []
    img_points = []
    for fn in img_names:
        print('processing %s...' % fn)
        img = cv2.imread(fn, 0)
        h, w = img.shape[:2]
        # NOTE: this OpenCV function takes the pattern size as (points_per_row,points_per_column) or (width_as_points,height_as_points)
        # as opposed to the more logical (number_of_rows, number_of_columns)
        found, corners = cv2.findChessboardCorners(img, (pattern_size[1], pattern_size[0]))

        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        if save_images or show_patterns:
            channels = img.ndim
            if channels < 3:
                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            path, name, ext = common_cv.splitfn(fn)
            result_img_name = '%s-with_detected_corners.bmp' % (name)
            if show_patterns:
                cv2.imshow(result_img_name, vis)
                cv2.waitKey(0)
                cv2.destroyWindow(result_img_name)
            if save_images:
                if save_dir:
                    complete_save_name = save_dir + "/" + result_img_name
                else:
                    complete_save_name = path + result_img_name

                print('Saving', complete_save_name)
                cv2.imwrite(complete_save_name, vis)
        if not found:
            print('chessboard not found')
            continue

        img_points.append(corners.reshape(-1, 2))  # This produces an 2D table of [u, v] coordinates with a single point per row.
        obj_points.append(pattern_all_points)
        print('ok')

    image_size = (w, h)
    return img_points, obj_points, image_size

class OmniStereoPair(object):

    def __init__(self, omni_image, mono_top, mono_bottom, resolve_point_ordering=True, idx=-1):
        '''
        The appropriate mirror-style reordering of correspondences is achieved with this initialization

        @param stereo_calibrator: The corresponding stereo calibrator instance for this pair
        @param omni_image: The omnistereo omni_image related to the pair.
        @param mono_top: The OmniMono instance for the top mirror
        @param mono_bottom: The OmniMono instance for the bottom mirror
        @param idx: Reference index number as calibration pattern
        '''

        self.omni_image = omni_image
        # Manual reordering method:
        # ------------------------
        # a) Allow user to click on each omni_image around to the origin of the pattern in order to establish point correspondences (detected corners)
        #=======================================================================
        # self.mono_top = self.resolve_points_order_manually(mono_top, "TOP", close_window=True)
        # self.mono_bottom = self.resolve_points_order_manually(mono_bottom, "BOTTOM", close_window=True)
        #=======================================================================

        if resolve_point_ordering:
            self.mono_top = self.resolve_points_order_automatically(mono_top, "TOP", debug=False)
            self.mono_bottom = self.resolve_points_order_automatically(mono_bottom, "BOTTOM", debug=False)
        else:
            self.mono_top = mono_top
            self.mono_bottom = mono_bottom

        self.found_points = self.mono_top.found_points and self.mono_bottom.found_points
        self.index = idx

    def resolve_points_order_automatically(self, mono, name_str, debug=False):
        '''
        Automatic resolution of origin corners correspondence

        Because the calibration grid poses of is NOT known in 3D (before calibration), we follow these steps:

        Step 1) Out of the 4 possible corners,
        we look for the top-two with the largest angular distance from the U-axis on the frame originating the center point.

        NOTE: Since grids crossing the U-axis can produce the wrong points, we must check if this is the case, so we resolve only points on the 1st quadrant.

        Step 2) Among the two candidates, we select the point with the smallest Euclidean pixel distance (with respect to the center)


        FUTURE: singletons (for example when only one of the top/bottom has found_points), this pair will be discarded through this process
        '''
        from omnistereo import common_tools

        all_points_wrt_Icenter = mono.image_points - mono.center_point
        num_rows, num_cols = mono.pattern_size_applied
        corners_4_pattern_indices = np.array([[0, 0], [num_rows - 1, 0], [num_rows - 1, num_cols - 1], [0, num_cols - 1]])
        corners_4 = all_points_wrt_Icenter[corners_4_pattern_indices[:, 0], corners_4_pattern_indices[:, 1]]

        corners_4_angles = np.arctan2(corners_4[:, 1], corners_4[:, 0])  # Takes (y_coords, x_coords)

        if np.all(np.logical_and(corners_4_angles < np.pi / 2.0, corners_4_angles > -np.pi / 2.0)):
            # The case when found on the 1st or 4th quadrant, resolving for the largest angular distance may be wrong using only positive-angle representation
            corners_4_angles_filtered = corners_4_angles
        else:
            # It's safe to convert to positive angle-representation
            corners_4_angles_filtered = np.where(corners_4_angles < 0, 2.*np.pi + corners_4_angles, corners_4_angles)

        sorted_angles_indices = np.argsort(corners_4_angles_filtered)  # Sorts non-decreasingly

        # Find the indices for the 2 points with the largest angular distances
        two_out_of_4_corner_list_indices = [sorted_angles_indices[-1], sorted_angles_indices[-2]]
        two_out_of_4_corners_values = corners_4[two_out_of_4_corner_list_indices]

        two_norms_from_center = np.linalg.norm(two_out_of_4_corners_values, axis=-1)
        two_norms_from_center_sorted_indices = np.argsort(two_norms_from_center)  # Sorts non-decreasingly

        # Extract final candidate based on radial (Euclidean) distances to center
        if name_str == "TOP":  # Select closest corners
            best_corner_list_index = two_out_of_4_corner_list_indices[two_norms_from_center_sorted_indices[0]]
        else:  # Select farthest corners
            best_corner_list_index = two_out_of_4_corner_list_indices[two_norms_from_center_sorted_indices[-1]]

        selected_corner_index = corners_4_pattern_indices[best_corner_list_index]

        # Compute 2D index of flatten index
        row_idx = selected_corner_index[0]
        col_idx = selected_corner_index[1]
        # Unused:
        # selected_corner_coords = mono.image_points[row_idx, col_idx]

        # There are 3 possible situations to consider for the new images corner
        new_image_pts = mono.image_points

        # 1) When it's already at the desired position:
        # Do nothing

        # 2) If the row disagrees, reverse the rows:
        if row_idx != 0:
            axis_reversal = 0
            new_image_pts = common_tools.reverse_axis_elems(arr=new_image_pts, k=axis_reversal)

        # 3) If the column disagrees, reverse the columns:
        if col_idx != 0:
            axis_reversal = 1
            new_image_pts = common_tools.reverse_axis_elems(arr=new_image_pts, k=axis_reversal)

        mono.set_points_info(found=mono.found_points, image_points=new_image_pts)

        # Visualization of resolved origin for debugging purposes
        if debug:
            img_final_origin_vis = mono.omni_image.copy()
            common_cv.draw_points(img_final_origin_vis, mono.image_points.reshape(-1, 2)[0, np.newaxis], num_valid_points=1, thickness=15)
            resolving_win = name_str + " mirror (Origin Resolving %d)" % (self.index)
            cv2.namedWindow(resolving_win, cv2.WINDOW_NORMAL)
            cv2.imshow(resolving_win, img_final_origin_vis)
            cv2.waitKey(0)
            cv2.destroyWindow(resolving_win)

        return mono

    def resolve_points_order_manually(self, mono, name_str, close_window=False):
        from omnistereo import common_tools

        resolving_win = name_str + " mirror (Origin Resolving)"
        cv2.namedWindow(resolving_win, cv2.WINDOW_NORMAL)
        origin_clicker = common_cv.PointClicker(win_name=resolving_win, max_clicks=1)
        target_click_near_origin = origin_clicker.get_clicks_uv_coords(mono.omni_image)
        norms_from_target = np.linalg.norm(target_click_near_origin - mono.image_points, axis=-1)
        closest_pt_flat_index = np.argmin(norms_from_target)
        num_rows, num_cols = mono.pattern_size_applied

        # Compute 2D index of flatten index
        row_idx = int(closest_pt_flat_index // num_cols)
        col_idx = int(closest_pt_flat_index % num_cols)

        # There are 3 possible situations to consider for the new images corner
        new_image_pts = mono.image_points

        # 1) When it's already at the desired position:
        # Do nothing

        # 2) If the row disagrees, reverse the rows:
        if row_idx != 0:
            axis_reversal = 0
            new_image_pts = common_tools.reverse_axis_elems(arr=new_image_pts, k=axis_reversal)

        # 3) If the column disagrees, reverse the columns:
        if col_idx != 0:
            axis_reversal = 1
            new_image_pts = common_tools.reverse_axis_elems(arr=new_image_pts, k=axis_reversal)

        mono.set_points_info(found=mono.found_points, image_points=new_image_pts)

        img_final_origin_vis = mono.omni_image.copy()
        common_cv.draw_points(img_final_origin_vis, mono.image_points.reshape(-1, 2)[0, np.newaxis], num_valid_points=1)
        cv2.imshow(resolving_win, img_final_origin_vis)
        cv2.waitKey(1)

        if close_window:
            cv2.destroyWindow(resolving_win)

        return mono

    def visualize_points(self, window_name="Correspondences on Pattern"):
        '''
        Visualize correspondences in a single omnidirectional omni_image (instead of separate images as it is done with the OmniMono visualize_points function)

        @return: The visualization omni_image of chessboard correspondence points (detected corners) if any.
                 Otherwise, the result is simply the gray scale omni_image without drawn points.
        '''
        if self.omni_image.ndim < 3:
            vis = cv2.cvtColor(self.omni_image, cv2.COLOR_GRAY2BGR)
        else:
            vis = self.omni_image.copy()

        if self.mono_top.found_points and self.mono_bottom.found_points and (self.mono_top.pattern_size_applied == self.mono_bottom.pattern_size_applied):
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.drawChessboardCorners(vis, self.mono_top.pattern_size_applied, self.mono_top.image_points.reshape(-1, 2), self.mono_top.found_points)
            cv2.drawChessboardCorners(vis, self.mono_bottom.pattern_size_applied, self.mono_bottom.image_points.reshape(-1, 2), self.mono_bottom.found_points)

            cv2.imshow(window_name, vis)
            cv2.waitKey(1)
        return vis


class OmniMono(object):

    def __init__(self, image, image_points=[], obj_points=[], img_center_point=None, pattern_size_applied=None):
        '''
        A collection of properties for a single omni_image used for calibration

        @param omni_image:  A numpy ndarray filled with the extracted corner points for each pattern
        @param image_points: A numpy ndarray filled with the extracted corner points for each pattern
        @param obj_points: A numpy ndarray corresponding to the 3D points on the flat chessboard pattern that were extracted from the omni_image.
                            Recall that their Z value will be 0 in reference to a pattern's frame, and the origin is the first corner point (instead of the beginning of the pattern board)
        '''

        # The omnidirectional image
        self.omni_image = image
        # The number of pixels referring to the size of the image as a tuple (width, height)
        self.image_size = np.array([image.shape[1], image.shape[0]])  # width by height
        self.color_channels = self.omni_image.ndim
        self.found_points = False
        self.image_points = image_points
        self.image_points_homo = []  # list of pixels in homogeneous coordinates. Thus, each pixel coord is of size 3x1
        self.obj_points = obj_points
        self.obj_points_homo = []  # list of points in pattern in homogeneous coordinates. Thus, each point is of size 4x1
        self.pattern_size_applied = (0, 0)

        if img_center_point is None:
            self.center_point = (self.image_size / 2.0) - 1  # image size should be given as (width, height) by our convention
        else:
            self.center_point = img_center_point  # FIXME: Who is passing this?

        self.boundary_radii = 0  # TODO: Using an ellipsoid rather than circle

        # I think these methods don't belong to this class!
#         self.T_wrt_cam_frame = None  # Transform matrix (4x4) with respect to camera model frame
#         self.set_transform_wrt_cam()

    def estimate_center_and_radius(self):
        # TODO: implement like this MATLAB code:
        '''
            imshow(images{1,i}.Imean);

            # Click on center and on boundary:
            d = ginput(2);

            info.xc{1,i} = d(1,1);
            info.yc{1,i} = d(1,2);

            info.radius{i} = sqrt( (d(2,1)-d(1,1))^2 + (d(2,2)-d(1,2))^2 );

            # Plot (visualize) initial circle
            line( [d(1,1)-50,d(1,1)+50], [d(1,2)-50,d(1,2)+50],'Color','y');
            line( [d(1,1)-50,d(1,1)+50], [d(1,2)+50,d(1,2)-50],'Color','y');

            t = 0 : pi/100 : 2*pi;
            plot(d(1,1)+info.radius{i} *cos(t),d(1,2)+info.radius{i} *sin(t),'r-','LineWidth',3)
        '''
        pass

    def set_points_info(self, found, image_points=None, obj_points=None, pattern_size_applied=None):
        self.found_points = found
        if found:
            # A numpy ndarray of shape (rows, cols, 2) filled with the extracted corner point coords as (u,v) for each pattern
            if image_points is not None:
                self.image_points = image_points
                # Homogeneous coordinates
                self.image_points_homo = np.dstack((image_points, np.ones(image_points.shape[:-1])))
            else:
                self.found_points = False

            # A numpy ndarray of shape (rows, cols, 3) corresponding to the 3D point coords as (X,Y,Z)  on the flat chessboard pattern that were extracted from the omni_image.
            if obj_points is not None:
                self.obj_points = obj_points
                # Homogeneous coordinates
                self.obj_points_homo = np.dstack((obj_points, np.ones(obj_points.shape[:-1])))

            # The size of the object/pattern as (rows, cols) corresponding to the points (corners) found
            if pattern_size_applied is None:
                pattern_size_applied = image_points.shape[:2]
            self.pattern_size_applied = pattern_size_applied

#     def set_transform_wrt_cam(self, transform=None):
#         '''
#         @param transform: the 4x4 transformation matrix (homogeneous coordinates) encoding the Rotation and translation transformation of a calibration pattern wrt the camera frame
#         '''
#         if transform == None:
#             self.T_wrt_cam_frame = np.identity(4)  # Just the homogeneous identity matrix if no transform matrix has been passed
#         else:
#             self.T_wrt_cam_frame = transform

    def visualize_points(self, window_name="Corners", show_cropped=False):
        '''
        @return: The visualization omni_image with drawn points (detected corners) if any.
                 Otherwise, the result is simply the gray scale omni_image without drawn points.
        '''
        if self.color_channels < 3:
            vis = cv2.cvtColor(self.omni_image, cv2.COLOR_GRAY2BGR)
        else:
            vis = self.omni_image.copy()

        if self.found_points:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.drawChessboardCorners(vis, self.pattern_size_applied, self.image_points.reshape(-1, 2), self.found_points)

            if show_cropped:
                min_point_position = np.min(self.image_points, 0)
                max_point_position = np.max(self.image_points, 0)
                # Crop from row (or y):y+height_offset, col (or x), x+width_offset
                crop_img = vis[min_point_position[1]:max_point_position[1], min_point_position[0]:max_point_position[0]]
                cv2.imshow(window_name, crop_img)
            else:
                cv2.imshow(window_name, vis)
            cv2.waitKey(1)
        return vis

class CalibratorMono(object):
    '''
    Acts an interface to perform a single omnidirectional omni_image calibration by detecting points from raw images
    (automatically, and semi-automatically if not found),
    '''

    def __init__(self, working_units="mm"):
        '''
        Constructor
        @TODO: Implement (repeat) intrinsic calibration with Xiang's GUM model (See MATLAB toolbox).
        '''
        self.working_units = working_units
        self.corner_extraction_args = {}
        self.mirror_images = []
        self.masked_images = []

        # list of OmniMono objects
        self.omni_monos = []

        self.T_G_wrt_C_list = []  # List of spatial Transformations of calibration chessboards (usually) with respect to the omni_model (camera, [C]) frame
        self.T_G_wrt_F_list = []  # List of spatial Transformations of calibration chessboards (usually) with respect to the model frame (mirror focus)
        self.chessboard_3D_points_wrt_C = []  # List of patterns as ground truth pose for its detected points wrt Camera pinhole
        self.chessboard_3D_points_wrt_F = []  # List of patterns as ground truth pose for its detected points wrt mirror focus (model frame)

        self.display_size = (800, 600)  # Default. It will be updated for the first time during visualization
        self.opt_vis_wins_exist = False
        self.opt_vis_wins_list = []
        self.has_chesboard_pose_info = False


    def initialize_parameters(self, app=None, finish_drawing=True, visualize=False, only_extrinsics=False, only_grids_extrinsics=False, only_C_wrt_M_tz=False, normalize=False, only_translation_params=False):
        '''
        Assuming, the theoretical model has been provided
        '''
        # grid_poses = list_len * [None]
        T_G_wrt_C_initial = []
        grid_poses = []
        means = []
        initial_params_scaled = []
        param_limits = []  # So far it should have nothing!
        scale_factors = []  # for feature scaling while optimizing

        bdry_padding = 1000  # +/- [mm] error
        # Set lower and upper bounds as (min, max) tuples
        # As quaternion components
        # Quaternions w+ix+jy+kz are represented as [w, x, y, z].
        if self.opt_method == "brute":
            quat_bounds = slice(-1., 1., 0.5)
            rot_q0_bounds = quat_bounds
            rot_q1_bounds = quat_bounds
            rot_q2_bounds = quat_bounds
            rot_q3_bounds = quat_bounds
        else:
            rot_q0_bounds = (-1., 1.)
            rot_q1_bounds = (-1., 1.)
            rot_q2_bounds = (-1., 1.)
            rot_q3_bounds = (-1., 1.)

        # start = common_cv.clock()
        if not only_C_wrt_M_tz:
            # for img_pts, obj_pts, T_G_wrt_C_true in zip(self.detected_points_for_calibration, self.points_wrt_pattern_for_calibration, self.T_G_wrt_C_list_for_calibration):
            for img_pts, obj_pts in zip(self.detected_points_for_calibration, self.points_wrt_pattern_for_calibration):
                # Approximate pose of calibration grid using planar homography or PnP (Both seem to be as fast)
                # NOTE: This kind of pose approximation can only be done if we have some idea of the THEORETICAL model
                T_G_wrt_C_approx, app = self.estimated_omni_model.approximate_transformation(img_pts, obj_pts, use_PnP=True, visualize=visualize, T_G_wrt_C=None, app=app, finish_drawing=finish_drawing)
                T_G_wrt_C_initial += [T_G_wrt_C_approx]  # Appending to the entire list!

                # Get rotation quaternion and translation vectors
                q = tr.quaternion_from_matrix(T_G_wrt_C_approx, isprecise=False)  # WARNING: a precise matrix can suffer of singularity issues!
                # TEST:
                # >>>> angles_from_q = tr.euler_from_quaternion(q)
                # >>>> scale, shear, angles, trans, persp = tr.decompose_matrix(T_G_wrt_C_approx)
                # >>>> np.allclose(angles_from_q, angles)
                t = T_G_wrt_C_approx[:3, 3]  # Translation vector

                # grid_poses[idx] = list(q) + list(t)  # appending the q and t lists
                if only_translation_params:
                    grid_poses += list(t)  # Appending to the entire list!
                else:
                    grid_poses += list(q) + list(t)  # Appending to the entire list!

                # Set new boundaries using heuristics of up to a %50 error on the translation
                if self.opt_method == "brute":
                    tx_bounds = slice(t[0] - bdry_padding, t[0] + bdry_padding, 1000)
                    ty_bounds = slice(t[1] - bdry_padding, t[1] + bdry_padding, 1000)
                    tz_bounds = slice(t[2] - bdry_padding, t[2] + bdry_padding, 1000)
                else:
                    tx_bounds = (t[0] - bdry_padding, t[0] + bdry_padding)
                    ty_bounds = (t[1] - bdry_padding, t[1] + bdry_padding)
                    tz_bounds = (t[2] - bdry_padding, t[2] + bdry_padding)
        #             tx_bounds = (None, None)
        #             ty_bounds = (None, None)
        #             tz_bounds = (None, None)
                if normalize:
                    factors = [1, 1, 1, 1, abs(tx_bounds[1] - tx_bounds[0]), abs(ty_bounds[1] - ty_bounds[0]), abs(tz_bounds[1] - tz_bounds[0])]
                    param_limits += [rot_q0_bounds, rot_q1_bounds, rot_q2_bounds, rot_q3_bounds, ((tx_bounds[0] - t[0]) / factors[4], (tx_bounds[1] - t[0]) / factors[4]), ((ty_bounds[0] - t[1]) / factors[5], (ty_bounds[1] - t[1]) / factors[5]), ((tz_bounds[0] - t[2]) / factors[6], (tz_bounds[1] - t[2]) / factors[6])]
                    initial_params_scaled += 7 * [0.]
                else:
                    if only_translation_params:
                        factors = [1, 1, 1]
                        param_limits += [tx_bounds, ty_bounds, tz_bounds]
                        initial_params_scaled += list(t)
                    else:
                        factors = [1, 1, 1, 1, 1, 1, 1]
                        grid_pose_limits = [rot_q0_bounds, rot_q1_bounds, rot_q2_bounds, rot_q3_bounds, tx_bounds, ty_bounds, tz_bounds]
                        print("Grid pose param limits:", grid_pose_limits)
                        param_limits += grid_pose_limits
                        initial_params_scaled += list(q) + list(t)

                scale_factors += factors
                # Also scale down boundaries:

#              param_limits += [rot_q0_bounds, rot_q1_bounds, rot_q2_bounds, rot_q3_bounds, (tx_bounds[0] / factors[4], tx_bounds[1] / factors[4]), (ty_bounds[0] / factors[5], ty_bounds[1] / factors[5]), (tz_bounds[0] / factors[6], tz_bounds[1] / factors[6])]
                # Only scale translation components!
        # print("%.2f ms" % ((common_cv.clock() - start) * 1000))

        # Recall: the s parameter is the pose of [C] wrt to [M], don't get confused)
        if only_grids_extrinsics == False or only_C_wrt_M_tz:
            # The pose limits for the focus of the mirror (NOTE: the s parameter is the pose of [C] wrt to [M], don't get confused)
            # CHECKME: The GUM compensates for rotations of the mirror (Perhaps also translations on the XY-plane) by finding the optimal position of the projection point Cp=[xi_x, xi_y, xi_z]??
            z_error = 0.  # For testing
            C_wrt_M_pose_z = -self.estimated_omni_model.F[2, 0] + z_error

            # NOTE: only 5% for the tz bounds
            bdry_padding_on_tz = 20  # +/- [mm] error
            tz_lb = C_wrt_M_pose_z - bdry_padding_on_tz
            tz_ub = C_wrt_M_pose_z + bdry_padding_on_tz
            if self.opt_method == "brute":
                C_wrt_M_z_bounds = slice(tz_lb, tz_ub, 10)
            else:
                C_wrt_M_z_bounds = (tz_lb, tz_ub)

            #===================================================================
            # Using ground-truth poses of the grids
            # self.plot_tz_vs_projection_cost(self.estimated_omni_model, self.points_wrt_pattern_for_calibration, self.detected_points_for_calibration, self.T_G_wrt_C_list_for_calibration, tz_min=tz_lb, tz_max=tz_ub)
            # Using initialization poses of the grids:
            # self.plot_tz_vs_projection_cost(self.estimated_omni_model, self.points_wrt_pattern_for_calibration, self.detected_points_for_calibration, T_G_wrt_C_initial, tz_min=tz_lb, tz_max=tz_ub)
            #===================================================================

            if normalize:
                factors = [abs(C_wrt_M_z_bounds[1] - C_wrt_M_z_bounds[0])]
                initial_params_scaled += [0.]
                param_limits += [((C_wrt_M_z_bounds[0] - C_wrt_M_pose_z) / factors[0], (C_wrt_M_z_bounds[1] - C_wrt_M_pose_z) / factors[0])]
            else:
                factors = [1]
                initial_params_scaled += [C_wrt_M_pose_z]
                param_limits += [C_wrt_M_z_bounds]

            scale_factors += factors
            # Also scale down boundaries:
#             param_limits += [(C_wrt_M_z_bounds[0] / factors[0], C_wrt_M_z_bounds[1] / factors[0])]

        total_extrinsic_params = len(param_limits)

        if only_extrinsics == False:
            # Initialize INTRINSIC parameters from theoretical model
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # xi_vector = [xi_x, xi_y, xi_z]
            xi_error = 0.  # Set error for testing convergence to known theoretical value
            xi_x_init = self.estimated_omni_model.precalib_params.xi1 + xi_error
            xi_y_init = self.estimated_omni_model.precalib_params.xi2 + xi_error
            xi_z_init = self.estimated_omni_model.precalib_params.xi3 + xi_error

            initial_params_scaled += [xi_x_init, xi_y_init, xi_z_init]
            if self.opt_method == "brute":
                xi_x_limits = slice(-1., 1., 0.5)
                xi_y_limits = slice(-1., 1., 0.5)
                xi_z_limits = slice(-1., 1., 0.5)
            else:
                xi_x_limits = (-1., 1.)
                xi_y_limits = (-1., 1.)
                xi_z_limits = (-1, 1.)
            param_limits += [xi_x_limits, xi_y_limits, xi_z_limits]

            # Radial distortion parameters: k_dist_1, k_dist_2
            k_error = 0.0  # Set error for testing convergence to no distortion at all
            k1_init = 0.0 + k_error
            k2_init = 0.0 + k_error

            initial_params_scaled += [k1_init, k2_init]
            if self.opt_method == "brute":
                k1_limits = slice(-1., 1., 0.5)
                k2_limits = slice(-1., 1., 0.5)
            else:
                k1_limits = (-1., 1.)  # CHECKME: find the true limits
                k2_limits = (-1., 1.)  # CHECKME: find the true limits

            param_limits += [k1_limits, k2_limits]

            # Camera intrinsic parameters: alpha, gamma1, gamma2, u_center, v_center
            alpha_c = self.estimated_omni_model.precalib_params.alpha_c
            gamma1 = self.estimated_omni_model.precalib_params.gamma1
            gamma2 = self.estimated_omni_model.precalib_params.gamma2
            u_center = self.estimated_omni_model.precalib_params.u_center
            v_center = self.estimated_omni_model.precalib_params.v_center
            initial_params_scaled += [alpha_c, gamma1, gamma2, u_center, v_center]

            img_width, img_height = self.estimated_omni_model.precalib_params.image_size
            if self.opt_method == "brute":
                alpha_limits = slice(-1., 1., 0.5)
                gamma1_limits = slice(1, 1000., 500)  # FIXME: what is the real range for gamma?
                gamma2_limits = slice(1, 1000., 500)
                u_center_limits = slice(img_width / 4, img_width - img_width / 4, 500)  # Because it would be nearly impossible to have the center in the 1/4 of the omni_image
                v_center_limits = slice(img_height / 4, img_height - img_height / 4, 500)
            else:
                alpha_limits = (-1., 1.)
                gamma1_limits = (1, 2000)
                gamma2_limits = (1, 2000)
                u_center_limits = (img_width / 4, img_width - img_width / 4)  # Because it would be nearly impossible to have the center in the 1/4 of the omni_image
                v_center_limits = (img_height / 4, img_height - img_height / 4)
            param_limits += [alpha_limits, gamma1_limits, gamma2_limits, u_center_limits, v_center_limits]
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # Rough feature scaling down:
        if normalize:
            means = np.array(initial_params_scaled)
#             initial_params_scaled = np.array(initial_params_scaled)  # With zero mean!
        else:
            means = np.zeros_like(initial_params_scaled)
#             initial_params_scaled = np.array(params_init_values)  # With zero mean!

        return means, initial_params_scaled, param_limits, np.array(scale_factors), total_extrinsic_params, app
#         initial_params_scaled = np.array(params_init_values) / scale_factors
#         return initial_params_scaled, param_limits, np.array(scale_factors)

    def run_corner_detection(self, cam_model, img_filename_template, chessboard_params_filename, input_units="cm", chessboard_indices=None, reduce_pattern_size=False):
        '''
        @param chessboard_params_filename:  the comma separated file for chessboard sizing (first row) and pose information for each pattern (used by POV-Ray)
        @param chessboard_indices: the indices for the working images
        '''

        info_file = open(chessboard_params_filename, 'r')
        info_content_lines = info_file.readlines()  # Read contents is just a long string
        info_file.close()

        chessboard_size_info_list = info_content_lines[0].split(",")  # The rows, cols, width_row, width_col, margin are saved on the first line
        rows = int(chessboard_size_info_list[0])
        cols = int(chessboard_size_info_list[1])
        width_row = float(chessboard_size_info_list[2])
        width_col = float(chessboard_size_info_list[3])
        margin = float(chessboard_size_info_list[4])
        # TODO: passing strings as arguments to Calibrator (for now)
        # IMPORTANT: the number of corners with intersection is always 1 less than the number of row and cols of the pattern!
        pattern_size_str = '(' + str(rows - 1) + ', ' + str(cols - 1) + ')'
        if input_units == "cm":
            if self.working_units == "mm":
                unit_conversion_factor = 10.0
        else:
            unit_conversion_factor = 1.0

        square_size_str = str(width_row * unit_conversion_factor)

        # Do corner detection on omnistereo based on above size information
        # Detect corners of all (or just indicated) patterns to be use as projected pixel points

        corner_extraction_args = {'--square_size': square_size_str, '--pattern_size': pattern_size_str, '--show': 'False', '--show_cropped': 'False', '--show_omni_mask': 'False', '--output_prefix_name': 'Single', }
        self.detect_corners(cam_model, img_filename_template, chessboard_indices, reduce_pattern_size, corner_extraction_args)

    #===========================================================================
    # def detect_corners_and_calibrate(self, omni_model, img_filename_pattern, img_indices, reduce_pattern_size, corner_extraction_args):
    #     '''
    #     Calibrates single GUM (monocular)
    #     '''
    #     self.corner_extraction_args = corner_extraction_args
    #     self.detect_corners(omni_model, img_filename_pattern, img_indices, reduce_pattern_size, corner_extraction_args)
    #     # TODO: put calibration routine here
    #===========================================================================

    def calibrate_mono(self, omni_model, chessboard_indices=[], normalize=False, visualize=False, only_extrinsics=False, only_grids=False, only_C_wrt_M_tz=False, only_translation_params=False, return_jacobian=True, do_single_trial=False):
        '''
        @param chessboard_indices: To work with the selected indices of the patterns pre-loaded in the Omnistereo calibrator's list
        '''
        from scipy import optimize

        list_len = len(self.omni_monos)

        if len(chessboard_indices) == 0 or chessboard_indices == None:
            self.chessboard_indices = range(list_len)
        else:
            self.chessboard_indices = chessboard_indices

        from copy import deepcopy
        self.original_model = deepcopy(omni_model)  # Make a copy of this object instance!
        self.estimated_omni_model = deepcopy(omni_model)  # Make a copy of this object instance!

        # self.detected_points_for_calibration = np.array([])
        # Appending desire calibration points to a list
        self.detected_points_for_calibration = []
        self.points_wrt_pattern_for_calibration = []
        self.T_G_wrt_C_list_for_calibration = []  # Just for testing agains ground truth poses

        for idx in self.chessboard_indices:
            if self.omni_monos[idx].found_points:
                # The detected (observed) pixels for chessboard points
                # No need to flatten because partial derivates must be computed wrt u and v independently
                # self.detected_points_for_calibration = np.append(self.detected_points_for_calibration, self.omni_monos[idx].image_points.flatten())  # serialized coordinates
                # Appending set of point to calibration list
                self.detected_points_for_calibration.append(self.omni_monos[idx].image_points)
#===============================================================================
#                 # vvvvvvv Using True projection points (as opposed to detected points) for testing vvvvvvvvvv:
#                 _, _, true_corners = self.estimated_omni_model.get_pixel_from_3D_point_homo(self.chessboard_3D_points_wrt_F[idx])
# #                 _, _, true_corners_theo = self.estimated_omni_model.theoretical_model.get_pixel_from_3D_point_homo(self.chessboard_3D_points_wrt_C[idx])
#                 self.detected_points_for_calibration.append(true_corners[..., :-1])
#                 # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#===============================================================================
                self.points_wrt_pattern_for_calibration.append(self.omni_monos[idx].obj_points_homo)
                try:
                    self.T_G_wrt_C_list_for_calibration.append(self.T_G_wrt_C_list[idx])
                except:
                    print("Warning: No  T_G_wrt_C_list[%d] found in %s" % (idx, __name__))

        #=======================================================================
        # Multivariate Constrained methods:
        #=======================================================================
        # All of these work better if initial values are given!
#         self.opt_method = 'SLSQP'  # is super fast, but it MUST have a good initialization
        self.opt_method = 'TNC'  # Not as fast as SLSQP but it can also succeed (also works better with good initializations)
#         self.opt_method = 'COBYLA'  # Cannot use the Jacobian, but it survives local minima points
#         self.opt_method = 'L-BFGS-B'  # (slower) but it succeeds even when initializations are far.
#         self.opt_method = 'Newton-CG'

        #=======================================================================
        # Multivariate Unconstrained methods:
        #=======================================================================
#         self.opt_method = 'Nelder-Mead'; return_jacobian = False
#         self.opt_method = 'CG'; return_jacobian = False
        #=======================================================================
#         self.opt_method = 'Powell'; return_jacobian = False  # MOST ACCURATE, but slow!
        #=======================================================================
#         self.opt_method = 'BFGS'  # It can be assisted by the Jacobian (first derivatives)
#         self.opt_method = 'leastsq'
#         self.opt_method = 'brute' # Brute force requires a LOT of memory to run and it's too slow (Never finished)

        # Visualization
        app = None
        if visualize:
            import visvis as vv
            try:
                from PySide import QtGui, QtCore
                backend = 'pyside'
            except ImportError:
                from PyQt4 import QtGui, QtCore
                backend = 'pyqt4'

            app = vv.use(backend)
        # NOTE: that app is passed to the initialize_parameters parameters function

        # Parameters Initialization:
        means, self.initial_params, param_limits, scale_factors, self.total_extrinsic_params, app = self.initialize_parameters(app=app, visualize=visualize, finish_drawing=False, only_extrinsics=only_extrinsics, only_grids_extrinsics=only_grids, only_C_wrt_M_tz=only_C_wrt_M_tz, normalize=normalize, only_translation_params=only_translation_params)

        # Equation Constraints (Only with ^^^^ SLSQP ^^^^ method)
        # Constraint 1: for the normalization of the quaternion
        # Equality constraint means that the constraint function result is to be zero
        # WISH: Use this contraint for the quaternion if you want to get rid of the normalization part of the Rotation transformation
        #       so that the Jacobian can be computed using the generalized form for R(q) and there may be an improvement in speed!
        cons_quat = {'type': 'eq', 'fun': lambda q:  1 - np.linalg.norm(np.array(q[0:4]))}

        # inequality means that it is to be non-negative
        # cons_inequality = {'type': 'ineq', 'fun': lambda q:  1 - np.linalg.norm(np.array(q[0:4]))}

        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        pts_scale_factor = 1.
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        live_visualization = False
        func_args = (live_visualization, only_extrinsics, only_grids, only_C_wrt_M_tz, only_translation_params, return_jacobian, pts_scale_factor)

        if do_single_trial:
#             if self.opt_method == 'leastsq':
#                 x, others = optimize.leastsq(func=self.pixel_projection_error_cost, x0=self.initial_params, args=func_args, Dfun=self.pixel_projection_error_jacobian)
#             elif self.opt_method == 'brute':
#                 x, others = optimize.brute(func=self.pixel_projection_error_cost, ranges=param_limits, args=func_args, full_output=True, finish=optimize.fmin, disp=True)
#             else:
    #             params = optimize.fmin_l_bfgs_b(func=self.pixel_projection_error, x0=self.initial_params, fprime=None, args=func_args, approx_grad=(not return_jacobian), bounds=param_limits, disp=1)
            params = optimize.minimize(fun=self.pixel_projection_error, x0=self.initial_params, args=func_args, method=self.opt_method, jac=return_jacobian, hess=None, hessp=None, bounds=param_limits, constraints=(), tol=None, callback=None, options={'maxiter' : 20000, 'disp' : True, })
            # FIXME: Always True for now, since non-convergence (failure of linear search) sometimes is better than no calibration at all!
            # self.estimated_omni_model.is_calibrated = params.status
            self.estimated_omni_model.is_calibrated = True
        else:
            minimizer_kwargs = dict(args=func_args, method=self.opt_method, jac=return_jacobian, hess=None, hessp=None, bounds=param_limits, constraints=(), tol=None, callback=None, options={'maxiter' : 20000, 'disp' : True, })
            params = optimize.basinhopping(func=self.pixel_projection_error, x0=self.initial_params, niter=20, minimizer_kwargs=minimizer_kwargs)
            self.estimated_omni_model.is_calibrated = True  # FIXME: always True? Use something like a ratio of the minimization_failures

        print("MONO:\n", params)
        self.estimated_params = params.x
        self.optimization_results = params

        self.estimated_grid_poses = self.estimated_params[:self.total_extrinsic_params - 1]
        # Update intrinsic params with results from final optimization
        self.estimated_omni_model.update_optimized_params(self.estimated_params[self.total_extrinsic_params - 1:], only_extrinsics=only_extrinsics, suppress_tz_optimization=only_grids, final_update=True)

#         only_grids = False
#         initial_params = self.initialize_parameters(visualize, only_grids_extrinsics=only_grids)
#         initial_params[:-7] = params.x  # Use previous results
#         param_limits += [rot_q0_bounds, rot_q1_bounds, rot_q2_bounds, rot_q3_bounds, C_wrt_M_x_bounds, C_wrt_M_y_bounds, C_wrt_M_z_bounds]
#         self.total_extrinsic_params = len(param_limits)
#         # TEST: Recall the angle is the transformation that took place between the original coordinates of the grid and the camera frame
# #         params = optimize.minimize(fun=self.pixel_projection_error, x0=initial_params, args=([visualize, only_grids, only_C_wrt_M_tz, return_jacobian]), method=opt_method, jac=return_jacobian, hess=None, hessp=None, bounds=param_limits, constraints=(), tol=None, callback=None, options={'maxiter' : 2000, 'disp' : True})
#         err = optimize.check_grad(self.pixel_projection_error_cost, self.pixel_projection_error_jacobian, initial_params, only_grids, only_C_wrt_M_tz)

        # ATTENTION: The following visualization is "NOT ACCURATE" because it projects the chessboard points from the "OPTIMIZED" poses
        #            rather than from the "ground truth" poses used to ray-trace the images.
        #            That is the reason for the nice overlap of projected points against the detected points.
        # OUR HOPE: with triangulation implied from stereo triangulation, we can have a more reliable metric to converge to the truth poses of the chessboard patterns (calibration grids)
#         err = optimize.check_grad(self.pixel_projection_error_cost, self.pixel_projection_error_jacobian, initial_params, only_grids, only_C_wrt_M_tz)

        return app


    def detect_corners(self, cam_model, img_filename_pattern, img_indices, reduce_pattern_size, corner_extraction_args):
        self.mirror_images = common_cv.get_images(img_filename_pattern, indices_list=img_indices, show_images=False)
        self.masked_images = common_cv.get_masked_images_mono(self.mirror_images, cam_model, img_indices, show_images=False, color_RGB=(0, 180, 0))

        self.corner_extraction_args = corner_extraction_args
        # Get the list of OmniMono objects
        self.omni_monos = extract_corners_batch(self.masked_images, img_indices, reduce_pattern_size, sys_args=self.corner_extraction_args)

        if hasattr(cam_model, "mirror_name"):
            print("%s mirror has %d sets for calibration" % (cam_model.mirror_name.capitalize(), sum(x is not None for x in self.omni_monos)))
        else:
            print("There are %d grids for calibration" % (sum(x is not None for x in self.omni_monos)))

    def draw_all_detected_corners(self):

        for i, grid in enumerate(self.omni_monos):
            grid.visualize_points(window_name="Corners - Grid %d" % (i))

    def set_true_chessboard_pose(self, omni_model, chessboard_params, input_units="cm", chessboard_indices=None, show_corners=False):
        '''
        @param omni_model: The desired projection model (i.e. a Generalized Unified Model (GUM) or any other model adhering the OmniCamModel template)
        @param chessboard_params:  Either a  comma separated file for chessboard sizing (first row) and pose information for each pattern (used by POV-Ray)
                                    or an array of homogeneous transform matrix for the grid pose information (usually given in [mm])
        @param chessboard_indices: the indices for the working images
        '''

        unit_conversion_factor = 1.0
        if input_units == "cm":
            if self.working_units == "mm":
                unit_conversion_factor = 10.0
        elif input_units == "m":
            if self.working_units == "mm":
                unit_conversion_factor = 1000.0

        if isinstance(chessboard_params, str):  # It's a filename
            info_file = open(chessboard_params, 'r')
            info_content_lines = info_file.readlines()  # Read contents is just a long string
            info_file.close()
            data_from_file = True
            if len(info_content_lines) < 2:
                self.has_chesboard_pose_info = False
                return
        elif isinstance(chessboard_params, np.ndarray):
            data_from_file = False
            T_Ggt_wrt_C_array = chessboard_params.copy()
            T_Ggt_wrt_C_array[..., :3, -1] = T_Ggt_wrt_C_array[..., :3, -1] * unit_conversion_factor
            if len(T_Ggt_wrt_C_array) < 1:
                self.has_chesboard_pose_info = False
                return

        # Get each patterns pose:
        list_len = len(self.omni_monos)
        self.T_G_wrt_C_list = [None] * list_len
        self.T_G_wrt_F_list = [None] * list_len
        self.chessboard_3D_points_wrt_C = [None] * list_len
        self.chessboard_3D_points_wrt_F = [None] * list_len

        if len(chessboard_indices) == 0 or chessboard_indices == None:
            chessboard_indices = range(list_len)

        for chessboard_number in chessboard_indices:
            if data_from_file:
                pose_info = 7 * [0.0]
                try:
                    pose_info_list = info_content_lines[chessboard_number + 1].split(",")  # Offset from the header line
                    # pose info will be given as a list (7-vector) of the rotation quaternion components [w, x, y, z] followed by translation components [tx, ty, tz].
                    # We grab the values (given as RHS)
                    pose_info[4] = unit_conversion_factor * float(pose_info_list[0])
                    pose_info[5] = unit_conversion_factor * float(pose_info_list[1])
                    pose_info[6] = unit_conversion_factor * float(pose_info_list[2])
                    GwrtC_angle_rot_x = np.deg2rad(float(pose_info_list[3]))  # because rotations in POV-Ray are given in degrees
                    GwrtC_angle_rot_y = np.deg2rad(float(pose_info_list[4]))  # because rotations in POV-Ray are given in degrees
                    GwrtC_angle_rot_z = np.deg2rad(float(pose_info_list[5]))  # because rotations in POV-Ray are given in degrees
                    #  In our RHS, the order of rotations are rotX --> rotY --> rotZ
                    #  tr.concatenate_matrices(GwrtC_rot_z, GwrtC_rot_y, GwrtC_rot_x)
                    # GwrtC_rot_matrix = tr.euler_matrix(GwrtC_angle_rot_x, GwrtC_angle_rot_y, GwrtC_angle_rot_z, 'sxyz')  # CHECKME: use various rotations to test if my understanding is correct
                    # q_test = tr.quaternion_from_matrix(GwrtC_rot_matrix, isprecise=True)
                    GwrtC_rot_q = tr.quaternion_from_euler(GwrtC_angle_rot_x, GwrtC_angle_rot_y, GwrtC_angle_rot_z, 'sxyz')
                    [pose_info[0], pose_info[1], pose_info[2], pose_info[3]] = GwrtC_rot_q
                    # print(tr.is_same_transform(tr.quaternion_matrix(GwrtC_rot_q), GwrtC_rot_matrix))
                    current_T_G_wrt_C = self.get_transformation_of_flat_pattern_wrt_camera(pose_info)
                except:
                    print("No file names passed")
                    print("Exiting from", __name__)
                    sys.exit(1)
                    # Add transformation to list
                    # TODO: WARNING: pay attention at indices if maintained throughout
            else:
                current_T_G_wrt_C = T_Ggt_wrt_C_array[chessboard_number]
            try:
                self.T_G_wrt_C_list[chessboard_number] = current_T_G_wrt_C
                current_T_G_wrt_F = self.get_transformation_of_flat_pattern_wrt_mirror_focus(current_T_G_wrt_C, omni_model)
                self.T_G_wrt_F_list[chessboard_number] = current_T_G_wrt_F
                # Test with first pattern
                # chessboard_3D_points_wrt_pattern_homo = np.array([[0, 0, 0, 1], [2, 0, 0, 1], [0, 0, 2, 1], [2, 0, 2, 1], [0, 0, 4, 1], [2, 0, 4, 1]])
                # Note that we only need to choose any set of object points from the calibration pair because the object pattern should be the same.
                corner_points_wrt_pattern = self.omni_monos[chessboard_number].obj_points_homo
                # chess_coords_wrt_C_homo = np.dot(T_G_wrt_C_list, chessboard_points.T).T
                # This above ^^^^^^^^^ is CLEARER, if we have T_G2C as (4x4), and the points matrix P_in_G as (rows x cols x 4), we have to do
                # P_in_C = (T_G2C x P_in_G^T)^T in order to get back the points matrix P_in_C as (rows x cols x 4)
                # as well as:
                # CHECKME: check the transformation again!
                chess_coords_wrt_C_homo = corner_points_wrt_pattern.dot(self.T_G_wrt_C_list[chessboard_number].T)
                self.chessboard_3D_points_wrt_C[chessboard_number] = chess_coords_wrt_C_homo
                chess_coords_wrt_F_homo = corner_points_wrt_pattern.dot(self.T_G_wrt_F_list[chessboard_number].T)
                self.chessboard_3D_points_wrt_F[chessboard_number] = chess_coords_wrt_F_homo
            except:  # catch *all* exceptions
                err_msg = sys.exc_info()[1]
                warnings.warn("Warning...%s" % (err_msg))

            if show_corners:
                try:
                    # Draw detected points in pattern (observed or measured pixel positions)
                    self.omni_monos[chessboard_number].visualize_points(window_name="Omni Corners - " + str(chessboard_number))
                except:  # catch *all* exceptions
                    err_msg = sys.exc_info()[1]
                    warnings.warn("Warning...%s" % (err_msg))

        self.has_chesboard_pose_info = True

    def get_transformation_of_flat_pattern_wrt_camera(self, pose_wrt_C_info_list):
        '''
        Finds tranformation of the chessboard pattern (calibration object frame) with respect to the  omnistereo camera frame.
        Coordinates are using the usual RHS as expected (the chessboard lays on the XZ-plane):

        @param pose_wrt_C_info_list: A list (7-vector) of the rotation quaternion components [w, x, y, z] that imply w+ix+jy+kz and  translation components [tx, ty, tz].

        '''
        from omnistereo import common_tools
        Tr_G_wrt_C = common_tools.get_transformation_matrix(pose_wrt_C_info_list)
        return Tr_G_wrt_C

    def get_transformation_of_flat_pattern_wrt_mirror_focus(self, Tr_G_wrt_C, omni_model):
        '''
        Finds tranformation of the chessboard pattern (calibration object frame) with respect to the  omnistereo camera frame.
        Coordinates are using the usual RHS as expected (the chessboard lays on the XZ-plane):
        @param Tr_G_wrt_C: the homogeneous transform matrix for the pose of the grid frame [G] wrt [C]
        '''
        # Recall that the pose information is with respect to the omnistereo camera frame described in POV-Ray, which uses a left-handed system.
        # G: the grid pattern frame
        # pose_wrt_C_info_list: A list (7-vector) of the rotation quaternion components [w, x, y, z] that imply w+ix+jy+kz and  translation components [tx, ty, tz].
        # Tr_G_wrt_C = self.get_transformation_of_flat_pattern_wrt_camera(pose_wrt_C_info_list)

        Tr_F_wrt_C = tr.translation_matrix(omni_model.F[:3].reshape(3))  # Creates a 4x4 homogeneous matrix
        Tr_C_wrt_F = tr.inverse_matrix(Tr_F_wrt_C)  # Invert

        # Transform between G and mirror
        Tr_G_wrt_F = tr.concatenate_matrices(Tr_C_wrt_F, Tr_G_wrt_C)  # chain of transformations in order:  GwrtF <- CwrtF GwrtC
        return Tr_G_wrt_F

    def pixel_projection_error(self, x, visualize=False, only_extrinsics=False, do_only_grids_extrinsics=False, only_C_wrt_M_tz=False, only_translation_params=False, return_jacobian=True, pts_scale_factor=1.):
        '''
        @brief: Objective function that computes the sum of the squared differences of projected pixel coordinates
        @param x: is the vector of parameters to minimize for

        @return: The sum of the squared differences of projected pixels, in addition to its jacobian matrix.
        '''
        proj_points, projections_jacobians = self.project_points_with_hypothesis(x, visualize, only_extrinsics, do_only_grids_extrinsics, only_C_wrt_M_tz, only_translation_params, self.T_G_wrt_C_list_for_calibration, return_jacobian, pts_scale_factor)

        # First sum only the differences between u_est, and between v_est
        # (BUT don't mix yet because it's needed to compute the jacobian of this objective function)
        total_cost = 0
        jacobian_matrix_of_obj_func = np.zeros_like(x)

        for m_est, m_true, proj_jac in zip(proj_points, self.detected_points_for_calibration, projections_jacobians):
            m_est_s = m_est * pts_scale_factor
            m_true_s = m_true * pts_scale_factor
            points_diff_norm = np.linalg.norm((m_est_s - m_true_s), axis=-1)
            sum_of_points_diff_norms = np.sum(points_diff_norm)
            total_cost += sum_of_points_diff_norms

            if return_jacobian:
                # Finish Jacobian composition with the pixel difference partial derivative
                # multiply each difference by its projection jacobian (see equation)

                u_est = m_est_s[..., 0]
                u_true = m_true_s[..., 0]
                v_est = m_est_s[..., 1]
                v_true = m_true_s[..., 1]
                d_of_fdiff_wrt_u_est = (u_est - u_true) / points_diff_norm  # * (point_norms_diffs) / np.abs(point_norms_diffs)
                d_of_fdiff_wrt_v_est = (v_est - v_true) / points_diff_norm  # * (point_norms_diffs) / np.abs(point_norms_diffs)
                d_of_fdiff_wrt_m_est = np.append(d_of_fdiff_wrt_u_est[..., np.newaxis], d_of_fdiff_wrt_v_est[..., np.newaxis], axis=-1)

                d_of_f_diffs_wrt_params = np.einsum("ijt, ijtk->ijk", d_of_fdiff_wrt_m_est, proj_jac)
                # Tested among all elements (using my HACK):
                # >>>> np.all([np.allclose(np.dot(d_of_fdiff_wrt_m_est[i,j], proj_jac[i,j]),d_of_f_diffs_wrt_params[i,j]) for (i,j), x in np.ndenumerate(d_of_f_diffs_wrt_params[...,0])])

                # CHECKME: should if be summed to each one? or as a lump sum?
                # The difference partial derivative
                # Simply, multiply each difference by the rotation jacobian
                jacobian_matrix_of_obj_func += np.einsum("ijk -> k", d_of_f_diffs_wrt_params)
            # -------------- END loop

#         total_cost_normalized = total_cost / self.estimated_omni_model.h_radial_image ** 2
#         print("Cost =", total_cost, "Normalized cost =", total_cost_normalized, end=" ")
        if return_jacobian:
            # Normalize Jacobian
#             if normalize:
#                 jac_scale_factors = x / total_cost
#                 cost_func_jac = jac_scale_factors * jacobian_matrix_of_obj_func  # / count
#             else:
            cost_func_jac = jacobian_matrix_of_obj_func  # / count
#             print("Jacobians=", cost_func_jac)
            return total_cost, cost_func_jac
        else:
#             print("\n")
            return total_cost

    def pixel_projection_error_jacobian(self, x, visualize=False, only_extrinsics=False, do_only_grids_extrinsics=False, only_C_wrt_M_tz=False, only_translation_params=False, return_jacobian=True, pts_scale_factor=1.):
        '''
        @brief: Computes the jacobian of the objective function that computes the sum of the squared differences of projected pixel coordinates
        @note: This function is rudundant and should be used only for checking the Jacobian, as it's actaully calling pixel_projection_error_function, which does both.

        @param x: is the vector of parameters to minimize for

        @return: The jacobian matrix for the objective function of the projected points using parameters x.
        '''
        _, jac = self.pixel_projection_error(x, visualize=visualize, only_extrinsics=only_extrinsics, do_only_grids_extrinsics=do_only_grids_extrinsics, only_C_wrt_M_tz=only_C_wrt_M_tz, only_translation_params=only_translation_params, return_jacobian=return_jacobian, pts_scale_factor=pts_scale_factor)
        return jac

    def pixel_projection_error_cost(self, x, visualize=False, only_extrinsics=False, do_only_grids_extrinsics=False, only_C_wrt_M_tz=False, only_translation_params=False, return_jacobian=True, pts_scale_factor=1.):
        '''
        @brief: Computes the sum of the squared differences of projected pixel coordinates
        @note: This function is rudundant and should be used only for checking the Jacobian, as it's actaully calling pixel_projection_error_function, which does both.

        @param x: is the vector of parameters to minimize for

        @return: The cost of the objective function for the projected points using parameters x.
        '''
        cost, _ = self.pixel_projection_error(x, visualize=visualize, only_extrinsics=only_extrinsics, do_only_grids_extrinsics=do_only_grids_extrinsics, only_C_wrt_M_tz=only_C_wrt_M_tz, only_translation_params=only_translation_params, return_jacobian=return_jacobian, pts_scale_factor=pts_scale_factor)
        return cost

    def project_points_with_hypothesis(self, x, visualize=False, only_extrinsics=False, do_only_grids_extrinsics=False, only_C_wrt_M_tz=False, only_translation_params=False, T_G_wrt_C_list_for_testing=[], return_jacobian=True, pts_scale_factor=1.):

        '''
        @param x: list of parameters to use in the optimization

        @return: a tuple of the list of omni_image coordinates of the resulting projected points using the x parameters,
                and the list of Jacobians of the projection function for each point as matrices of size (2x(17+7L))
        '''
        # Initialize empty list of sets of pattern imaged (projected) points
        list_len = len(self.detected_points_for_calibration)
        # Using simple lists
        projected_points_res = list_len * [None]
        projections_jacobians = list_len * [None]

        if visualize and not self.opt_vis_wins_exist:
            from omnistereo.common_tools import get_screen_resolution
            self.display_size = get_screen_resolution(measurement="px")
            win_width, win_height = 400, 300
            max_idx_width = int(self.display_size[0] / win_width)

        # IMPORTANT: Set new parameters!
        extrinsic_parameters = x[:self.total_extrinsic_params]
        self.estimated_omni_model.update_optimized_params(x[self.total_extrinsic_params - 1:], only_extrinsics=only_extrinsics, suppress_tz_optimization=do_only_grids_extrinsics)

        for idx in range(list_len):
            try:
                T_G_wrt_C = T_G_wrt_C_list_for_testing[idx]
            except:
                T_G_wrt_C = None

            # Get JACOBIAN for each point in the grid
            jacobians_of_proj_func, projected_points = self.estimated_omni_model.compute_jacobian_projection(extrinsic_parameters, self.points_wrt_pattern_for_calibration[idx], idx, only_extrinsics, do_only_grids_extrinsics, only_C_wrt_M_tz, only_translation_params, T_G_wrt_C, return_jacobian, pts_scale_factor)

            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#                 # TEST Jacobian implementation d_of_fH_wrt_fw:
#                 # Ground truth is given as g:
#                 # Use any arbitrary g to test
#                 from scipy.optimize import check_grad
#                 q_estimated = tr.quaternion_from_euler(np.deg2rad(3.005), np.deg2rad(2.005), np.deg2rad(100.1), axes="sxyz")  # [1, 0, 0, 0]
#                 t_estimated = [900, 10, 10]
#                 params_est = list(q_estimated) + t_estimated
#
#                 def func_fp_wrt_g_est(p_est, pts_wrt_G):
#                     _, pixels_est = self.estimated_omni_model.compute_jacobian_projection(p_est, pts_wrt_G)
#                     points_diffs = (pixels_est - projected_points)[..., :2]
#                     sum_sq_diffs = 0.5 * np.sum(points_diffs ** 2)
#                     return sum_sq_diffs
#
#                 def jac_fp_wrt_g_est(p_est, pts_wrt_G):
#                     d_of_fp_wrt_e_est, pixels_est = self.estimated_omni_model.compute_jacobian_projection(p_est, pts_wrt_G)
#                     points_diffs = (pixels_est - projected_points)[..., :2]
#
#                     # The difference partial derivative
#                     d_fp_diffs = np.einsum("ijk, ijkl->ijl", points_diffs, d_of_fp_wrt_e_est)
#                     gradient_of_all_diffs_sum = np.einsum("ijk -> k", d_fp_diffs)
#                     return gradient_of_all_diffs_sum
#
#                 err_fp_jacs = check_grad(func_fp_wrt_g_est, jac_fp_wrt_g_est, params_est, (corner_points_wrt_pattern))
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#                 # Get PROJECTION
#                 pose_G_wrt_C = [rot_q0, rot_q1, rot_q2, rot_q3, tx1, ty1, tz1]
#                 # T_obj_frame_wrt_C = self.get_transformation_of_flat_pattern_wrt_camera(pose_info_list, self.estimated_omni_model, input_units="mm")
#                 T_G_wrt_F = self.get_transformation_of_flat_pattern_wrt_mirror_focus(**OLDpose_G_wrt_COLD**, self.estimated_omni_model, input_units="mm")
#
#                 # chess_coords_wrt_C_homo = corner_points_wrt_pattern.dot(T_obj_frame_wrt_C[idx].T)
#                 chess_coords_wrt_F_homo = corner_points_wrt_pattern.dot(T_G_wrt_F.T)
#                 # Measurements:
#                 wrt_mirror_focus = True  # TODO: using transforms wrt mirror focus in order to test individual calibration
#                 if wrt_mirror_focus:
#                     # Initialize a model with xi1
#                     _, _, projected_points = self.estimated_omn            i_model.get_pixel_from_3D_point_homo(chess_coords_wrt_F_homo)
#                 else:
#                     # The true (forward) projected pixels due to the analytical equation
#                     pass
#                 # DON'T flatten!
# #                 projected_points_res = np.append(projected_points_res, projected_points[..., :-1].flatten())

            projections_jacobians[idx] = jacobians_of_proj_func
            projected_points_res[idx] = projected_points[..., :-1]

            if visualize:
                if not self.opt_vis_wins_exist:
                    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
                    win = "OPTIMIZATION (%s) Point Projection - Pattern [%d]" % (self.estimated_omni_model.mirror_name, idx)
                    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(win, win_width, win_height)
                    pos_x = (idx % max_idx_width) * win_width
                    pos_y = 50 + int(idx / max_idx_width) * win_height
                    cv2.moveWindow(win, pos_x, pos_y)
                    self.opt_vis_wins_list.append(win)
                    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                else:
                    win = self.opt_vis_wins_list[idx]

                omni_img = self.omni_monos[self.chessboard_indices[idx]].omni_image.copy()  # copy omni_image
                top_detected_color = (0, 0, 255)  # blue because (R,G,B)
                common_cv.draw_points(omni_img, self.detected_points_for_calibration[idx][..., :2].reshape(-1, 2), color=top_detected_color, thickness=1)
                top_projected_color = (255, 0, 0)  # red because (R,G,B)
                common_cv.draw_points(omni_img, projected_points[..., :2].reshape(-1, 2), color=top_projected_color, thickness=3)
#                     bot_detected_color = (0, 255, 0)  # green
#                     common_cv.draw_points(omni_img, observed_corners_bottom[..., :2].reshape(-1, 2), color=bot_detected_color)
#                     bot_projected_color = (255, 0, 255)  # magenta
#                     common_cv.draw_points(omni_img, true_corners_bottom[..., :2].reshape(-1, 2), color=bot_projected_color)
                cv2.imshow(win, omni_img)


        if visualize:
            cv2.waitKey(1)
            # On first loop, set flag that indicates creation of visualization highgui windows:
            if not self.opt_vis_wins_exist:
                self.opt_vis_wins_exist = True

        return projected_points_res, projections_jacobians

    def plot_tz_vs_projection_cost(self, omni_model, points_3D_wrt_G, pixels_true, T_G_wrt_C_true, tz_min=2.0, tz_max=20, ax=None):
        '''

        @return the drawn axis corresponding to the parameter ax of the figure
        '''
        import matplotlib.pyplot as plt
        from mpldatacursor import datacursor

        tz_s = np.linspace(tz_min, tz_max, 1000, endpoint=True)
        costs = np.zeros_like(tz_s)

        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        # Plot
        font = {'family' : 'serif',
                'color'  : 'darkblue',
                'weight' : 'normal',
                'size'   : 16,
                }
        font_big = {'family' : 'serif',
                'color'  : 'black',
                'weight' : 'normal',
                'size'   : 24,
                }

        ax.grid(True)
        ax.set_title(r'Effect of $t_{z}$ on Projection Error cost $J$', fontdict=font)
        ax.set_xlabel(r'$t_{z} \, [\mathrm{%s}]$' % (omni_model.units), fontdict=font)
        ax.set_ylabel(r'$J \, [\mathrm{pixels}]$', fontdict=font)

        # Tweak spacing to prevent clipping of ylabel
#         ax.set_xlim(0, tz_max)
#         official_yticks = ax.get_yticks()
#         x_lims = ax.get_xlim()

        for i, tz in enumerate(tz_s):
            cost = 0
            T_C_wrt_M = tr.translation_matrix([0, 0, tz])
            for pts_3D_grid, px_true, T_G_wrt_C in zip(points_3D_wrt_G, pixels_true, T_G_wrt_C_true):
                # Compute pts_3d
                pts_3D_wrt_C = np.einsum("ij, klj->kli", T_G_wrt_C, pts_3D_grid)
                #===============================================================
                # pts_3D_wrt_M = np.einsum("ij, klj->kli", T_C_wrt_M, pts_3D_wrt_C)
                # # Get projected points
                # _, _, px_proj = omni_model.get_pixel_from_3D_point_homo(pts_3D_wrt_M)
                #===============================================================
                _, _, px_proj = omni_model.get_pixel_from_3D_point_wrt_C(pts_3D_wrt_C)
                # Compute and accumulate cost
                points_diff_norm = np.linalg.norm((px_proj[..., :2] - px_true), axis=-1)
                sum_of_points_diff_norms = np.sum(points_diff_norm)
                cost += sum_of_points_diff_norms

            costs[i] = cost

        line, = ax.plot(tz_s, costs, label=r"Cost due to Projection Error")
        y_lims = ax.get_ylim()

        tz_at_min = tz_s[np.argmin(costs)]
        min_cost = np.min(costs)
        min_msg = "Minimum projection cost = %f [pixels] located at tz = %f [%s]" % (min_cost, tz_at_min, omni_model.units)
        print(min_msg)
        min_label = r"$\mathrm{min} \, J = %.2f [\mathrm{pixels}] \,|\, t_z = %.4f [\mathrm{%s}]$" % (min_cost, tz_at_min, omni_model.units)
        ax.axvline(tz_at_min, color='red', linestyle='--')  # , label=min_msg)
        ax.text(tz_at_min - 1., max(y_lims) * (8 / 10), min_label, rotation=90, size="12", color="r")

        datacursor(display='single', formatter='$(t_z,J)$=({y:.2f},{x:.2f})'.format, draggable=True, bbox=dict(alpha=1))

        ax.legend().draggable()
#         plt.tight_layout()
        plt.show()

class CalibratorStereo(object):
    '''
    Acts an interface to perform the omnistereo calibration by detecting points from raw images
    (automatically, and semi-automatically if not found).
    It can establish correspondence pairs (semi-manual step),
    and optimizing the error of the essential matrix in order to find the extrinsic transformation parameters between the stereo GUM
    '''

    def __init__(self, working_units="mm"):
        '''
        Constructor
        '''

#         self.omni_model = stereo_model  # A stereo GUM instance
#         self.calib_top = stereo_model.top_model.calibrator
#         self.calib_bottom = stereo_model.bot_model.calibrator
        self.working_units = working_units
        self.calib_top = CalibratorMono(working_units)
        self.calib_bottom = CalibratorMono(working_units)
        self.calibration_pairs = []  # The list of OmniStereoPair objects
        self.has_chesboard_pose_info = False

#===============================================================================
#     def calibrate(self, omnistereo_model, img_filename_pattern, img_indices, reduce_pattern_size, corner_extraction_args_top, corner_extraction_args_bottom):
#         # TODO: calibrate for the Stereo model
#         print("Stereo omnidirectional system calibration (for extrinsic and intrinsic parameters?)")
#         # Call calibrate on top GUM and bottom GUM
#         self.detect_corners(omnistereo_model, img_filename_pattern, img_indices, reduce_pattern_size, corner_extraction_args_top, corner_extraction_args_bottom)
#===============================================================================

    def calibrate_omnistereo(self, omnistereo_model, chessboard_indices=[], visualize=False, return_jacobian=True, only_extrinsics=False, init_grid_poses_from_both_views=False, do_single_trial=True):
        '''
        @param chessboard_indices: To work with the selected indices of the patterns pre-loaded in the Omnistereo calibrator's list
        '''
        list_len = len(self.calibration_pairs)

        if len(chessboard_indices) == 0 or chessboard_indices == None:
            self.chessboard_indices = range(list_len)
        else:
            self.chessboard_indices = chessboard_indices

        from copy import deepcopy
        self.original_omnistereo_model = deepcopy(omnistereo_model)  # Make a copy of this object instance!
        self.estimated_omnistereo_model = deepcopy(omnistereo_model)  # Make a copy of this object instance!

        # Appending desire calibration points to a list
        self.detected_points_for_calibration = []  # Combine top and bottom omni_image points into a single list
        self.calib_top.detected_points_for_calibration = []
        self.calib_bottom.detected_points_for_calibration = []
        self.points_wrt_pattern_for_calibration = []
        self.T_G_wrt_C_list_for_calibration = []  # Just for testing agains ground truth poses

        for idx in self.chessboard_indices:
            if self.calibration_pairs[idx].found_points:
                # The detected (observed) pixels for chessboard points
                # No need to flatten because partial derivates must be computed wrt u and v independently
                # self.detected_points_for_calibration = np.append(self.detected_points_for_calibration, self.omni_monos[idx].image_points.flatten())  # serialized coordinates
                # Appending set of point to calibration list
                self.points_wrt_pattern_for_calibration.append(self.calibration_pairs[idx].mono_top.obj_points_homo)
                try:
                    if self.calib_top.has_chesboard_pose_info:
                        self.T_G_wrt_C_list_for_calibration.append(self.calib_top.T_G_wrt_C_list[idx])
                except:
                    print("Warning: No  calib_top.T_G_wrt_C_list[%d] found in %s" % (idx, __name__))
                self.detected_points_for_calibration.append(self.calibration_pairs[idx].mono_top.image_points)
                self.detected_points_for_calibration.append(self.calibration_pairs[idx].mono_bottom.image_points)
                # It's useful to propagate the respective data to the corresponding child
                self.calib_top.detected_points_for_calibration.append(self.calibration_pairs[idx].mono_top.image_points)
                self.calib_bottom.detected_points_for_calibration.append(self.calibration_pairs[idx].mono_bottom.image_points)

        # It's useful to propagate this data to the children
        self.calib_top.points_wrt_pattern_for_calibration = self.points_wrt_pattern_for_calibration
        self.calib_bottom.points_wrt_pattern_for_calibration = self.points_wrt_pattern_for_calibration
        self.calib_top.T_G_wrt_C_list_for_calibration = self.T_G_wrt_C_list_for_calibration
        self.calib_bottom.T_G_wrt_C_list_for_calibration = self.T_G_wrt_C_list_for_calibration

        #=======================================================================
        # Multivariate Constrained methods:
        #=======================================================================
        # All of these work better if initial values are given!
#         self.opt_method = 'SLSQP'  # is super fast, but it MUST have a good initialization
        self.opt_method = 'TNC'  # Not as fast as SLSQP but it can also succeed (also works better with good initializations)
#         self.opt_method = 'COBYLA'  # Cannot use the Jacobian, but it survives local minima points
#         self.opt_method = 'L-BFGS-B'  # (slower) but it succeeds even when initializations are far.
#         self.opt_method = 'Newton-CG'

        #=======================================================================
        # Multivariate Unconstrained methods:
        #=======================================================================
#         self.opt_method = 'Nelder-Mead'; return_jacobian = False
#         self.opt_method = 'CG'; return_jacobian = False
        #=======================================================================
#         self.opt_method = 'Powell'; return_jacobian = False  # MOST ACCURATE, but slow!
        #=======================================================================
#         self.opt_method = 'BFGS'  # It can be assisted by the Jacobian (first derivatives)

        # Visualization
        app = None
        if visualize:
            import visvis as vv
            try:
                from PySide import QtGui, QtCore
                backend = 'pyside'
            except ImportError:
                from PyQt4 import QtGui, QtCore
                backend = 'pyqt4'

            app = vv.use(backend)
        # NOTE: that app is passed to the initialize_parameters parameters function

        # Parameters Initialization:
        self.initial_params, param_limits, self.total_extrinsic_params, app = self.initialize_omnistereo_parameters(app=app, finish_drawing=True, visualize=visualize, only_extrinsics=only_extrinsics, init_grid_poses_from_all_views=init_grid_poses_from_both_views)

        live_visualization = False
        func_args = (live_visualization, return_jacobian, only_extrinsics)

        if do_single_trial:
            from scipy.optimize import minimize
            params = minimize(fun=self.pixel_projection_error_omnistereo, x0=self.initial_params, args=func_args, method=self.opt_method, jac=return_jacobian, hess=None, hessp=None, bounds=param_limits, constraints=(), tol=None, callback=None, options={'maxiter' : 20000, 'disp' : True, })
            # Always True for now, since non-convergence (failure of linear search) sometimes is better than no calibration at all!
            #===================================================================
            # self.estimated_omnistereo_model.top_model.is_calibrated = params.status
            # self.estimated_omnistereo_model.bot_model.is_calibrated = params.status
            #===================================================================
        else:
            from scipy.optimize import basinhopping
            minimizer_kwargs = dict(args=func_args, method=self.opt_method, jac=return_jacobian, hess=None, hessp=None, bounds=param_limits, constraints=(), tol=None, callback=None, options={'maxiter' : 20000, 'disp' : True, })
            params = basinhopping(func=self.pixel_projection_error_omnistereo, x0=self.initial_params, niter=100, minimizer_kwargs=minimizer_kwargs)
            # FIXME: Find another way to determine if the status of the optimization was successfull. Maybe based on the ratio of minimization_failures.
            print("Minimization Failures", params.minimization_failures)
        self.estimated_omnistereo_model.top_model.is_calibrated = True
        self.estimated_omnistereo_model.bot_model.is_calibrated = True

        self.optimization_results = params
        print("STEREO\n", self.optimization_results)
        self.estimated_params = params.x

        # Update intrinsic params with results from final optimization
        self.estimated_grid_poses = self.estimated_params[:self.total_extrinsic_params - 2]
        # It's useful to propagate the knowledge of the estimated grid poses to the children calib objects
        self.calib_top.estimated_grid_poses = self.estimated_grid_poses
        self.calib_bottom.estimated_grid_poses = self.estimated_grid_poses
        self.estimated_omnistereo_model.update_optimized_params(self.estimated_params[self.total_extrinsic_params - 2:], only_extrinsics=only_extrinsics, final_update=True)

        #=======================================================================
        # sensor_evaluation.forward_projection_eval_mono(camera_model=self.estimated_omnistereo_model.top_model, calibrator=self.calib_top, chessboard_indices=self.chessboard_indices, wrt_mirror_focus=True, visualize=True, visualize_panorama=True, show_detected=True, verbose=True)
        # sensor_evaluation.forward_projection_eval_mono(camera_model=self.estimated_omnistereo_model.bot_model, calibrator=self.calib_bottom, chessboard_indices=self.chessboard_indices, wrt_mirror_focus=True, visualize=True, visualize_panorama=True, show_detected=True, verbose=True)
        #=======================================================================

        # ATTENTION: The following visualization is "NOT ACCURATE" because it projects the chessboard points from the "OPTIMIZED" poses
        #            rather than from the "ground truth" poses used to ray-trace the images.
        #            That is the reason for the nice overlap of projected points against the detected points.
        # CHECKME TODAY: perhaps the "compute_jacobian_projection" is not updating the estimated poses accordingly during the optimization steps!
        # HOPE: with triangulation implied from stereo triangulation, we can have a more reliable metric to converge to the truth poses of the chessboard patterns (calibration grids)
#===============================================================================
#         extrinsic_parameters = x_opt_scaled_up[:self.total_extrinsic_params]
#         for idx in range(list_len):
#             # Get JACOBIAN for each point in the grid
#             jacobians_of_proj_func, projected_points = self.estimated_omni_model.compute_jacobian_projection(extrinsic_parameters, self.points_wrt_pattern_for_calibration[idx], idx, return_jacobian=False)
#
#             if visualize:
#                 win = "OPTIMIZATION (%s) FINAL Point Projection - Pattern [%d]" % (self.estimated_omni_model.mirror_name, idx)
#                 cv2.namedWindow(win, cv2.WINDOW_NORMAL)
#
#                 omni_img = self.omni_monos[self.chessboard_indices[idx]].omni_image.copy()  # copy omni_image
#                 top_detected_color = (0, 0, 255)  # blue because (R,G,B)
#                 common_cv.draw_points(omni_img, self.detected_points_for_calibration[idx][..., :2].reshape(-1, 2), color=top_detected_color, thickness=1)
#                 top_projected_color = (255, 0, 0)  # red because (R,G,B)
#                 common_cv.draw_points(omni_img, projected_points[..., :2].reshape(-1, 2), color=top_projected_color, thickness=3)
#                 cv2.imshow(win, omni_img)
#                 cv2.waitKey(1)
#===============================================================================

        return app

    def pixel_projection_error_omnistereo(self, x, visualize=False, return_jacobian=True, only_extrinsics=False):
        '''
        @brief: Objective function that computes the sum of the squared differences of projected pixel coordinates
        @param x: is the vector of parameters to minimize for

        @return: The sum of the squared differences of projected pixels, in addition to its jacobian matrix.
        '''
        proj_points, projections_jacobians = self.project_points_with_hypothesis_omnistereo(x, visualize, self.T_G_wrt_C_list_for_calibration, return_jacobian, only_extrinsics)

        total_cost = 0
        jacobian_matrix_of_obj_func = np.zeros_like(x)

#         grid_counter = 0
        # RECALL: detected_points_for_calibration is a list appended alternating top THEN bottom, THEN top THEN bottom
        for m_est, m_true, proj_jac in zip(proj_points, self.detected_points_for_calibration, projections_jacobians):
            # First sum only the norms of u_est  and  v_est
            points_diff_norm = np.linalg.norm((m_est - m_true), axis=-1)
            #===================================================================
            # # IDEA: add a  weight to the error due to points from the top mirror since their resolution is higher.
            # if grid_counter % 2 == 0:  # For top
            #     error_weight = 0.5
            # else:  # For bottom
            #     error_weight = 1.0
            #===================================================================
            error_weight = 1.0

            sum_of_points_diff_norms = error_weight * np.sum(points_diff_norm)
            total_cost += sum_of_points_diff_norms

            if return_jacobian:
                # Finish Jacobian composition with the pixel difference partial derivative
                # multiply each difference by its projection jacobian (see equation)

                u_est = m_est[..., 0]
                u_true = m_true[..., 0]
                v_est = m_est[..., 1]
                v_true = m_true[..., 1]
                d_of_fdiff_wrt_u_est = (u_est - u_true) / points_diff_norm  # * (point_norms_diffs) / np.abs(point_norms_diffs)
                d_of_fdiff_wrt_v_est = (v_est - v_true) / points_diff_norm  # * (point_norms_diffs) / np.abs(point_norms_diffs)
                d_of_fdiff_wrt_m_est = error_weight * np.append(d_of_fdiff_wrt_u_est[..., np.newaxis], d_of_fdiff_wrt_v_est[..., np.newaxis], axis=-1)

                d_of_f_diffs_wrt_params = np.einsum("ijt, ijtk->ijk", d_of_fdiff_wrt_m_est, proj_jac)
                # Tested among all elements (using my HACK):
                # >>>> np.all([np.allclose(np.dot(d_of_fdiff_wrt_m_est[i,j], proj_jac[i,j]),d_of_f_diffs_wrt_params[i,j]) for (i,j), x in np.ndenumerate(d_of_f_diffs_wrt_params[...,0])])

                # CHECKME: should if be summed to each one? or as a lump sum?
                # The difference partial derivative
                # Simply, multiply each difference by the rotation jacobian
                jacobian_matrix_of_obj_func += np.einsum("ijk -> k", d_of_f_diffs_wrt_params)

#             grid_counter += 1
            # -------------- END loop

#         total_cost_normalized = total_cost / self.estimated_omni_model.h_radial_image ** 2
#         print("Cost =", total_cost, "Normalized cost =", total_cost_normalized, end=" ")
        if return_jacobian:
            # Normalize Jacobian
#             if normalize:
#                 jac_scale_factors = x / total_cost
#                 cost_func_jac = jac_scale_factors * jacobian_matrix_of_obj_func  # / count
#             else:
            cost_func_jac = jacobian_matrix_of_obj_func  # / count
#             print("Jacobians=", cost_func_jac)
            return total_cost, cost_func_jac
        else:
#             print("\n")
            return total_cost

    def project_points_with_hypothesis_omnistereo(self, x, visualize=False, T_G_wrt_C_list_for_testing=[], return_jacobian=True, only_extrinsics=False):

        '''
        @param x: list of parameters to use in the optimization

        @return: a tuple of the list of omni_image coordinates of the resulting projected points using the x parameters,
                and the list of Jacobians of the projection function for each point as matrices of size (2x(17+7L))
        '''
        # vvvvvvvvvvvvvvvvvvvvvvvvvvv
        # Default params for mono projection functions:
        only_C_wrt_M_tz = False
        do_only_grid_extrinsics = False
        only_translation_params = False
        pts_scale_factor = 1.
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # Initialize empty list of sets of pattern imaged (projected) points
        # RECALL: for the stereo case, the points are duplicated (appended)
        # See function calibrate_omnistereo() to understand how the appending of point duplicates is done such as:
        # self.detected_points_for_calibration.append(self.calibration_pairs[idx].mono_top.image_points)
        # self.detected_points_for_calibration.append(self.calibration_pairs[idx].mono_bottom.image_points)
        list_len = len(self.detected_points_for_calibration)
        # Using simple lists
        projected_points_res = list_len * [None]
        projections_jacobians = list_len * [None]

        # IMPORTANT: Set new parameters!
        self.estimated_omnistereo_model.update_optimized_params(x[self.total_extrinsic_params - 2:], only_extrinsics)
        total_extrinsic_params_mono = self.total_extrinsic_params - 1
        for idx in range(list_len):
            idx_mono = int(idx / 2)
            # Get JACOBIAN for each point in the grid
            try:
                T_G_wrt_C = T_G_wrt_C_list_for_testing[idx_mono]
            except:
                T_G_wrt_C = None

            if idx % 2 == 0:  # Even indices refer to TOP mirror omni_image points
                extrinsic_parameters_top = x[:self.total_extrinsic_params - 1]
                jacobians_of_proj_func, projected_points = self.estimated_omnistereo_model.top_model.compute_jacobian_projection(extrinsic_parameters_top, self.points_wrt_pattern_for_calibration[idx_mono], idx_mono, only_extrinsics, do_only_grid_extrinsics, only_C_wrt_M_tz, only_translation_params, T_G_wrt_C, return_jacobian, pts_scale_factor)
                if return_jacobian:
                    data_shape = jacobians_of_proj_func[..., 0, 0].shape
                    tz_zero_padding = np.zeros((data_shape + (2, 1)))
                    partials_of_g = jacobians_of_proj_func[..., :total_extrinsic_params_mono - 1]
                    partials_of_tz = jacobians_of_proj_func[..., total_extrinsic_params_mono - 1, np.newaxis]
                    if only_extrinsics:
                        # Append zero sub-matrices and partial derivatives accordingly
                        jacobians_of_proj_func_omnistereo = np.concatenate((partials_of_g, partials_of_tz, tz_zero_padding), axis=-1)
                    else:
                        # Reformat jacobian matrix
                        intrinsics_zero_padding = np.zeros((data_shape + (2, 10)))
                        partials_of_intrinsics = jacobians_of_proj_func[..., total_extrinsic_params_mono:]
                        # Append zero sub-matrices and partial derivatives accordingly
                        jacobians_of_proj_func_omnistereo = np.concatenate((partials_of_g, partials_of_tz, tz_zero_padding, partials_of_intrinsics, intrinsics_zero_padding), axis=-1)
            else:  # For BOTTOM mirror points on the omni_image
                extrinsic_parameters_bottom = np.concatenate((x[:self.total_extrinsic_params - 2], x[self.total_extrinsic_params - 1, np.newaxis]))
                jacobians_of_proj_func, projected_points = self.estimated_omnistereo_model.bot_model.compute_jacobian_projection(extrinsic_parameters_bottom, self.points_wrt_pattern_for_calibration[idx_mono], idx_mono, only_extrinsics, do_only_grid_extrinsics, only_C_wrt_M_tz, only_translation_params, T_G_wrt_C, return_jacobian, pts_scale_factor)
                if return_jacobian:
                    # Reformat jacobian matrix
                    data_shape = jacobians_of_proj_func[..., 0, 0].shape
                    tz_zero_padding = np.zeros((data_shape + (2, 1)))
                    partials_of_g = jacobians_of_proj_func[..., :total_extrinsic_params_mono - 1]
                    partials_of_tz = jacobians_of_proj_func[..., total_extrinsic_params_mono - 1, np.newaxis]
                    if only_extrinsics:
                        # Append zero sub-matrices and partial derivatives accordingly
                        jacobians_of_proj_func_omnistereo = np.concatenate((partials_of_g, tz_zero_padding, partials_of_tz), axis=-1)
                    else:
                        intrinsics_zero_padding = np.zeros((data_shape + (2, 10)))
                        partials_of_intrinsics = jacobians_of_proj_func[..., total_extrinsic_params_mono:]
                        # Append zero sub-matrices and partial derivatives accordingly
                        jacobians_of_proj_func_omnistereo = np.concatenate((partials_of_g, tz_zero_padding, partials_of_tz, intrinsics_zero_padding, partials_of_intrinsics), axis=-1)

            projected_points_res[idx] = projected_points[..., :-1]
            if return_jacobian:
                projections_jacobians[idx] = jacobians_of_proj_func_omnistereo

        # NOTE: projected_points_res has to be a list of alternating results for top THEN bottom for grid [0], THEN top THEN bottom for grid [1], etc.
        return projected_points_res, projections_jacobians

    def initialize_omnistereo_parameters(self, app=None, finish_drawing=True, visualize=False, only_extrinsics=False, init_grid_poses_from_all_views=True):
        '''
        @param init_grid_poses_from_all_views: When True, an average is computed from the PnP solution for the grid poses by averiging the respective views. Otherwise, only the main view (such as top view) is considered.
        Assuming, the theoretical model has been provided
        '''
        if visualize:
            import visvis as vv

        T_G_wrt_C_initial = []
        grid_poses = []
        initial_params = []
        param_limits = []  # So far it should have nothing!

        bdry_padding = 1000  # +/- [mm] error
        # Set lower and upper bounds as (min, max) tuples
        # As quaternion components
        # Quaternions w+ix+jy+kz are represented as [w, x, y, z].
        rot_q0_bounds = (-1., 1.)
        rot_q1_bounds = (-1., 1.)
        rot_q2_bounds = (-1., 1.)
        rot_q3_bounds = (-1., 1.)

        # start = common_cv.clock()
        # Using TOP model only for grid initial approximation since it's omni_image resolution is usually higher:
        # TODO: incoroporate both models for initialization
        for idx, obj_pts in enumerate(self.points_wrt_pattern_for_calibration):
            # Approximate pose of calibration grid using planar homography or PnP (Both seem to be as fast)
            T_G_wrt_C = None
            try:
                T_G_wrt_C = self.T_G_wrt_C_list_for_calibration[idx]
            except:
                print("Warning: there is no self.T_G_wrt_C_list_for_calibration[%d] at %s" % (idx, __name__))
            # NOTE: This kind of pose approximation can only be done if we have some idea of the THEORETICAL model
            if init_grid_poses_from_all_views:
                img_points_top = self.detected_points_for_calibration[idx * 2]
                img_points_bottom = self.detected_points_for_calibration[(idx * 2) + 1]
                T_G_wrt_C_approx_top, app = self.estimated_omnistereo_model.top_model.approximate_transformation(img_points=img_points_top, obj_pts_homo=obj_pts, use_PnP=True, visualize=visualize, T_G_wrt_C=T_G_wrt_C, app=app, finish_drawing=False)
                T_G_wrt_C_approx_bottom, app = self.estimated_omnistereo_model.bot_model.approximate_transformation(img_points=img_points_bottom, obj_pts_homo=obj_pts, use_PnP=True, visualize=visualize, T_G_wrt_C=T_G_wrt_C, app=app, finish_drawing=False)
                #===============================================================
                # Use the SLERP on quaternions at 50% (midway) to compute interpolated average of rotation in order to avoid "Gimbal Lock" situations from averaging Euler angles (Verified that they CAN occur in this situation)
                # quat_top = tr.quaternion_from_matrix(T_G_wrt_C_approx_top, isprecise=False)
                # quat_bottom = tr.quaternion_from_matrix(T_G_wrt_C_approx_bottom, isprecise=False)
                # rot_quat_avg = tr.quaternion_slerp(quat_top, quat_bottom, fraction=0.5)  # To find average
                # rot_matrix_avg = tr.quaternion_matrix(rot_quat_avg)
                # # translation avg:
                # trans_top = tr.translation_from_matrix(T_G_wrt_C_approx_top)
                # trans_bottom = tr.translation_from_matrix(T_G_wrt_C_approx_bottom)
                # translation_avg = np.mean(np.dstack((trans_top, trans_bottom)), axis=-1)
                # T_G_wrt_C_approx = tr.concatenate_matrices(tr.translation_matrix(translation_avg), rot_matrix_avg)
                #===============================================================

                # Instead, using a weighted SLERP and translation based on the pixel reprojection error from each view
                weight_top = self.estimated_omnistereo_model.get_confidence_weight_from_pixel_RMSE_stereo(img_points_top=img_points_top, img_points_bot=img_points_bottom, obj_pts_homo=obj_pts, T_G_wrt_C=T_G_wrt_C_approx_top)
                weight_bottom = self.estimated_omnistereo_model.get_confidence_weight_from_pixel_RMSE_stereo(img_points_top=img_points_top, img_points_bot=img_points_bottom, obj_pts_homo=obj_pts, T_G_wrt_C=T_G_wrt_C_approx_bottom)
                T_G_wrt_C_approx_new = tr.pose_average(poses_list=[T_G_wrt_C_approx_top, T_G_wrt_C_approx_bottom], weights=[weight_top, weight_bottom], use_birdal_method=True)
                #===============================================================
                # if visualize: # In order to visualize the differences between the initialization style (using 1/2 VS. weighted SLERP fractions)
                #     from common_plot import draw_frame_poses
                #     draw_frame_poses(pose_list=[T_G_wrt_C_approx] + [T_G_wrt_C_approx_new])
                #===============================================================
                T_G_wrt_C_approx = T_G_wrt_C_approx_new
            else:
                T_G_wrt_C_approx, app = self.estimated_omnistereo_model.top_model.approximate_transformation(img_points=self.detected_points_for_calibration[idx * 2], obj_pts_homo=obj_pts, use_PnP=True, visualize=visualize, T_G_wrt_C=T_G_wrt_C, app=app, finish_drawing=False)
#                 T_G_wrt_C_approx, app = self.estimated_omnistereo_model.bot_model.approximate_transformation(img_points=self.detected_points_for_calibration[(idx * 2) + 1], obj_pts_homo=obj_pts, use_PnP=True, visualize=visualize, T_G_wrt_C=T_G_wrt_C, app=app, finish_drawing=finish_drawing)

            if visualize and init_grid_poses_from_all_views:
                # Paint the resulting average pose
                grid_color_init = 'b'
                # ----------------- Initial values ------------------
                obj_pts_init = np.einsum("ij, mnj->mni", T_G_wrt_C_approx, obj_pts)
                xx_obj_pts_init = obj_pts_init[..., 0]
                yy_obj_pts_init = obj_pts_init[..., 1]
                zz_obj_pts_init = obj_pts_init[..., 2]
                a = vv.gca()
                obj_pts_init_grid = vv.grid(xx_obj_pts_init, yy_obj_pts_init, zz_obj_pts_init, axesAdjust=True, axes=a)
                obj_pts_init_grid.edgeColor = grid_color_init
                obj_pts_init_grid.edgeShading = "plain"  # possible shaders: None, plain, flat, gouraud, smooth
                obj_pts_init_grid.diffuse = 0.0
                Og_pt_init = vv.Point(obj_pts_init[0, 0, 0], obj_pts_init[0, 0, 1], obj_pts_init[0, 0, 2])
                vv.plot(Og_pt_init, ms='.', mc=grid_color_init, mw=5, ls='', mew=0, axesAdjust=False)

            T_G_wrt_C_initial += [T_G_wrt_C_approx]  # Appending to the entire list! NOT USED anywhere else really!

            # Get rotation quaternion and translation vectors
            q = tr.quaternion_from_matrix(T_G_wrt_C_approx, isprecise=False)  # WARNING: a precise matrix can suffer of singularity issues!
            # TEST:
            # >>>> angles_from_q = tr.euler_from_quaternion(q)
            # >>>> scale, shear, angles, trans, persp = tr.decompose_matrix(T_G_wrt_C_approx)
            # >>>> np.allclose(angles_from_q, angles)
            t = T_G_wrt_C_approx[:3, 3]  # Translation vector
            grid_poses += list(q) + list(t)  # Appending to the entire list!

            # Set new boundaries using heuristics of up to a %50 error on the translation
            tx_bounds = (t[0] - bdry_padding, t[0] + bdry_padding)
            ty_bounds = (t[1] - bdry_padding, t[1] + bdry_padding)
            tz_bounds = (t[2] - bdry_padding, t[2] + bdry_padding)
    #             tx_bounds = (None, None)
    #             ty_bounds = (None, None)
    #             tz_bounds = (None, None)
            grid_pose_limits = [rot_q0_bounds, rot_q1_bounds, rot_q2_bounds, rot_q3_bounds, tx_bounds, ty_bounds, tz_bounds]
            print("Grid pose param limits:", grid_pose_limits)
            param_limits += grid_pose_limits
            initial_params += list(q) + list(t)

        if finish_drawing:
            from visvis import title
            # Start app
            title('Initialization of Extrinsic Poses')
            app.Run()

        # Recall: the s parameter is the pose of [C] wrt to [M], don't get confused)
        # The pose limits for the focus of the mirror (NOTE: the s parameter is the pose of [C] wrt to [M], don't get confused)
        # CHECKME: The GUM compensates for rotations of the mirror (Perhaps also translations on the XY-plane) by finding the optimal position of the projection point Cp=[xi_x, xi_y, xi_z]??
        z_error = 0.  # For testing
        C_wrt_M_pose_z_top = -self.estimated_omnistereo_model.top_model.F[2, 0] + z_error
        C_wrt_M_pose_z_bottom = -self.estimated_omnistereo_model.bot_model.F[2, 0] + z_error
        initial_params += [C_wrt_M_pose_z_top, C_wrt_M_pose_z_bottom]

        # NOTE: only 5% for the tz bounds
        bdry_padding_on_tz = 20  # +/- [mm] error
        tz_lb_top = C_wrt_M_pose_z_top - bdry_padding_on_tz
        tz_ub_top = C_wrt_M_pose_z_top + bdry_padding_on_tz
        C_wrt_M_z_bounds_top = (tz_lb_top, tz_ub_top)
        tz_lb_bottom = C_wrt_M_pose_z_bottom - bdry_padding_on_tz
        tz_ub_bottom = C_wrt_M_pose_z_bottom + bdry_padding_on_tz
        C_wrt_M_z_bounds_bottom = (tz_lb_bottom, tz_ub_bottom)

        param_limits += [C_wrt_M_z_bounds_top, C_wrt_M_z_bounds_bottom]
        total_extrinsic_params = len(param_limits)

        if only_extrinsics == False:
            # Intrinsic limits (Same for TOP and BOTTOM):
            xi_x_limits = (-1., 1.)
            xi_y_limits = (-1., 1.)
            xi_z_limits = (-1, 1.)
            # CHECKME: find the true limits
            k1_limits = (-1., 1.)
            k2_limits = (-1., 1.)
            alpha_limits = (-1., 1.)
            gamma1_limits = (1, 2000)
            gamma2_limits = (1, 2000)
            img_width, img_height = self.estimated_omnistereo_model.top_model.precalib_params.image_size
            u_center_limits = (img_width / 4, img_width - img_width / 4)  # Because it would be nearly impossible to have the center in the 1/4 of the omni_image
            v_center_limits = (img_height / 4, img_height - img_height / 4)

            # Initialize TOP INTRINSIC parameters from theoretical model
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            xi_error = 0.  # Set error for testing convergence to known theoretical value
            xi_x_init_top = self.estimated_omnistereo_model.top_model.precalib_params.xi1 + xi_error
            xi_y_init_top = self.estimated_omnistereo_model.top_model.precalib_params.xi2 + xi_error
            xi_z_init_top = self.estimated_omnistereo_model.top_model.precalib_params.xi3 + xi_error
            initial_params += [xi_x_init_top, xi_y_init_top, xi_z_init_top]
            param_limits += [xi_x_limits, xi_y_limits, xi_z_limits]

            # Radial distortion parameters: k_dist_1, k_dist_2
            k_error = 0.0  # Set error for testing convergence to no distortion at all
            k1_init = 0.0 + k_error
            k2_init = 0.0 + k_error
            initial_params += [k1_init, k2_init]
            param_limits += [k1_limits, k2_limits]

            # Camera intrinsic parameters: alpha, gamma1, gamma2, u_center, v_center
            alpha_c_top = self.estimated_omnistereo_model.top_model.precalib_params.alpha_c
            gamma1_top = self.estimated_omnistereo_model.top_model.precalib_params.gamma1
            gamma2_top = self.estimated_omnistereo_model.top_model.precalib_params.gamma2
            u_center_top = self.estimated_omnistereo_model.top_model.precalib_params.u_center
            v_center_top = self.estimated_omnistereo_model.top_model.precalib_params.v_center
            initial_params += [alpha_c_top, gamma1_top, gamma2_top, u_center_top, v_center_top]
            param_limits += [alpha_limits, gamma1_limits, gamma2_limits, u_center_limits, v_center_limits]
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            # Initialize BOTTOM INTRINSIC parameters from theoretical model
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            xi_error = 0.  # Set error for testing convergence to known theoretical value
            xi_x_init_bot = self.estimated_omnistereo_model.bot_model.precalib_params.xi1 + xi_error
            xi_y_init_bot = self.estimated_omnistereo_model.bot_model.precalib_params.xi2 + xi_error
            xi_z_init_bot = self.estimated_omnistereo_model.bot_model.precalib_params.xi3 + xi_error
            initial_params += [xi_x_init_bot, xi_y_init_bot, xi_z_init_bot]
            param_limits += [xi_x_limits, xi_y_limits, xi_z_limits]

            # Radial distortion parameters: k_dist_1, k_dist_2
            k_error = 0.0  # Set error for testing convergence to no distortion at all
            k1_init = 0.0 + k_error
            k2_init = 0.0 + k_error
            initial_params += [k1_init, k2_init]
            param_limits += [k1_limits, k2_limits]

            # Camera intrinsic parameters: alpha, gamma1, gamma2, u_center, v_center
            alpha_c_bot = self.estimated_omnistereo_model.bot_model.precalib_params.alpha_c
            gamma1_bot = self.estimated_omnistereo_model.bot_model.precalib_params.gamma1
            gamma2_bot = self.estimated_omnistereo_model.bot_model.precalib_params.gamma2
            u_center_bot = self.estimated_omnistereo_model.bot_model.precalib_params.u_center
            v_center_bot = self.estimated_omnistereo_model.bot_model.precalib_params.v_center
            initial_params += [alpha_c_bot, gamma1_bot, gamma2_bot, u_center_bot, v_center_bot]
            param_limits += [alpha_limits, gamma1_limits, gamma2_limits, u_center_limits, v_center_limits]
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        return initial_params, param_limits, total_extrinsic_params, app


    def detect_corners(self, omnistereo_model, img_filename_pattern, img_indices, reduce_pattern_size, corner_extraction_args_top, corner_extraction_args_bottom, visualize=False):

        # Original code:
        # mirror_images = common_cv.get_images(img_filename_pattern, show_images=False)
        # masked_images = common_cv.get_masked_images_as_pairs(mirror_images, self.omni_model, show_images=False)
        # masked_images_transposed = zip(*masked_images)
        # masked_images_top = masked_images_transposed[0]
        # masked_images_bottom = masked_images_transposed[1]
        # masked_images_top, masked_images_bottom = common_cv.get_masked_images_as_pairs(mirror_images, self.omni_model, show_images=False)

        # Generates the list of OmniMono objects for the top images
        self.calib_top.detect_corners(omnistereo_model.top_model, img_filename_pattern, img_indices, reduce_pattern_size, corner_extraction_args_top)

        # Generates the list of OmniMono objects for the bottom images
        self.calib_bottom.detect_corners(omnistereo_model.bot_model, img_filename_pattern, img_indices, reduce_pattern_size, corner_extraction_args_bottom)

        l = len(self.calib_top.mirror_images)
        self.calibration_pairs = l * [None]  # clear list

        if img_indices == None or len(img_indices) == 0:
            img_indices = range(l)  # Use all images


        for i in img_indices:
            try:
                if visualize:
                    # Show before point reordering
                    calib_pair_unresolved = OmniStereoPair(self.calib_top.mirror_images[i], self.calib_top.omni_monos[i], self.calib_bottom.omni_monos[i], resolve_point_ordering=False, idx=i)
                    calib_pair_unresolved.visualize_points(window_name="BEFORE Reordering: OmniStereoPair Corners - %d" % (i))

                self.calibration_pairs[i] = OmniStereoPair(self.calib_top.mirror_images[i], self.calib_top.omni_monos[i], self.calib_bottom.omni_monos[i], resolve_point_ordering=True, idx=i)

                if visualize:
                    self.calibration_pairs[i].visualize_points(window_name="AFTER Reordering: OmniStereoPair Corners - %d" % (i))
            except:  # catch *all* exceptions
                err_msg = sys.exc_info()[1]
                warnings.warn("Warning...%s" % (err_msg))

    def run_corner_detection(self, omnistereo_model, omni_img_filename_template, chessboard_params_filename, input_units="cm", chessboard_indices=None, reduce_pattern_size=False, visualize=False):
        '''
        @param chessboard_params_filename:  the comma separated file for chessboard sizing (first row) and pose information for each pattern (used by POV-Ray)
        @param chessboard_indices: the indices for the working images
        '''

        info_file = open(chessboard_params_filename, 'r')
        info_content_lines = info_file.readlines()  # Read contents is just a long string
        info_file.close()

        chessboard_size_info_list = info_content_lines[0].split(",")  # The rows, cols, width_row, width_col, margin are saved on the first line
        rows = int(chessboard_size_info_list[0])
        cols = int(chessboard_size_info_list[1])
        width_row = float(chessboard_size_info_list[2])
        width_col = float(chessboard_size_info_list[3])
        margin = float(chessboard_size_info_list[4])
        # TODO: passing strings as arguments to Calibrator (for now)
        # IMPORTANT: the number of corners with intersection is always 1 less than the number of row and cols of the pattern!
        pattern_size_str = '(' + str(rows - 1) + ', ' + str(cols - 1) + ')'
        if input_units == "cm":
            if self.working_units == "mm":
                unit_conversion_factor = 10.0
        elif input_units == "m":
            if self.working_units == "mm":
                unit_conversion_factor = 1000.0

        square_size_str = str(width_row * unit_conversion_factor)

        # Do corner detection on omnistereo based on above size information
        # Detect corners of all (or just indicated) patterns to be use as projected pixel points
        corner_extraction_args_top = {'--square_size': square_size_str, '--pattern_size': pattern_size_str, '--show': str(visualize), '--show_cropped': 'False', '--show_omni_mask': 'False', '--output_prefix_name': 'top', }
        corner_extraction_args_bottom = {'--square_size': square_size_str, '--pattern_size': pattern_size_str, '--show': str(visualize), '--show_cropped': 'False', '--show_omni_mask': 'False', '--output_prefix_name': 'bottom'}
        self.detect_corners(omnistereo_model, omni_img_filename_template, chessboard_indices, reduce_pattern_size, corner_extraction_args_top, corner_extraction_args_bottom, visualize=visualize)


    def set_true_chessboard_pose(self, omnistereo_model, chessboard_params, input_units="cm", chessboard_indices=None, show_corners=False):
        '''
        @param omnistereo_model: The desired projection model (i.e. a Generalized Unified Model (GUM) or any other model adhering the CamModel template)
        @param chessboard_params:  Either a  comma separated file for chessboard sizing (first row) and pose information for each pattern (used by POV-Ray)
                                    or an array of homogeneous transform matrix for the grid pose information in [mm]
        @param chessboard_indices: the indices for the working images
        '''
        # Get each patterns pose:
        if len(chessboard_indices) == 0 or chessboard_indices == None:
            list_len = len(self.calibration_pairs)
            chessboard_indices = range(list_len)

        self.calib_top.set_true_chessboard_pose(omnistereo_model.top_model, chessboard_params, input_units, chessboard_indices, show_corners=False)
        self.calib_bottom.set_true_chessboard_pose(omnistereo_model.bot_model, chessboard_params, input_units, chessboard_indices, show_corners=False)
        self.has_chesboard_pose_info = self.calib_top.has_chesboard_pose_info and self.calib_bottom.has_chesboard_pose_info

        if show_corners:
            for chessboard_number in chessboard_indices:
                try:
                    # Draw detected points in pattern (observed or measured pixel positions)
                    self.calibration_pairs[chessboard_number].visualize_points(window_name="OmniStereoPair Corners - " + str(chessboard_number))
        #             calibration.draw_detected_points_manually(camera_model.top_model.calibrator.omni_monos[idx].omni_image, camera_model.top_model.calibrator.omni_monos[idx].image_points, 5, show=True)
        #             calibration.draw_detected_points_manually(camera_model.bot_model.calibrator.omni_monos[idx].omni_image, camera_model.bot_model.calibrator.omni_monos[idx].image_points, 5, show=True)
                except:  # catch *all* exceptions
                    err_msg = sys.exc_info()[1]
                    warnings.warn("Warning...%s" % (err_msg))

    def visualize_all_calibration_pairs(self, window_name="Correspondences of ALL Omnistereo Calibration Pairs"):
        '''
        Visualize correspondences  of all the omnistereo instances

        @return: The visualization omni_image of chessboard correspondence points (detected corners) if any.
                 Otherwise, the result is simply the gray scale omni_image without drawn points.
        '''
        for os in self.calibration_pairs:
            os.visualize_points(window_name="%s - %d" % (window_name, os.index))
