# -*- coding: utf-8 -*-
# common_cv.py

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

'''
This module contains some common routines based on OpenCV
'''
from __future__ import print_function
from __future__ import division
import warnings
import numpy as np
import cv2
import os
from contextlib import contextmanager
import itertools as it

image_extensions = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.pbm', '.pgm', '.ppm']

def has_opencv():
    try:
        import cv2 as lib
        return True
    except ImportError:
        return False

def is_cv2():
    # if we are using OpenCV 2, then our cv2.__version__ will start
    # with '2.'
    return check_opencv_version("2.")

def is_cv3():
    # if we are using OpenCV 3.X, then our cv2.__version__ will start
    # with '3.'
    return check_opencv_version("3.")

def check_opencv_version(major, lib=None):
    # if the supplied library is None, import OpenCV
    if lib is None:
        try:
            import cv2 as lib
        except ImportError:
            return False
    # return whether or not the current OpenCV version matches the
    # major version number
    return lib.__version__.startswith(major)

def get_cv_img_as_RGB(image):
    cv_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv_rgb

def clean_up(wait_key_time=0):
    cv2.waitKey(wait_key_time)
    cv2.destroyAllWindows()

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def draw_str(dst, coords, s):
    xc, yc = coords
    cv2.putText(dst, s, (xc + 1, yc + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (xc, yc), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

def getsize(img):
    h, w = img.shape[:2]
    return w, h

def mdot(*args):
    from functools import reduce
    return reduce(np.dot, args)

def draw_keypoints(vis, keypoints, color=(0, 255, 255)):
    for kp in keypoints:
            xc, yc = kp.pt
            cv2.circle(vis, (int(xc), int(yc)), 2, color)

def filter_correspondences_manually(train_img, query_img, train_kpts, query_kpts, colors_RGB, first_row_to_crop_bottom=0, do_filtering=False):
    '''
    @param do_filtering: if True, the manual filtering is excuted, other wise only the matches are drawn and shown.
    '''
    if do_filtering:
        valid_indices = np.ones(shape=(query_kpts.shape), dtype="bool")
    else:
        valid_indices = np.ones(shape=(1, len(query_kpts)), dtype="bool")

    # Visualize inlier matches
    if train_img is not None:
        if train_img.ndim == 3:
            top_pano_gray = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
        else:
            top_pano_gray = train_img.copy()
    top_pano_gray_vis = cv2.cvtColor(top_pano_gray, cv2.COLOR_GRAY2BGR)

    if query_img is not None:
        if query_img.ndim == 3:
            bot_pano_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        else:
            bot_pano_gray = query_img.copy()
    bot_pano_gray_vis = cv2.cvtColor(bot_pano_gray, cv2.COLOR_GRAY2BGR)

    row_offset = top_pano_gray.shape[0]  # Rows

    for top_kpt, bot_kpt, random_RGB_color in zip(train_kpts, query_kpts, colors_RGB):
        top_pano_gray_vis = cv2.drawKeypoints(top_pano_gray_vis, [top_kpt], outImage=top_pano_gray_vis, color=rgb2bgr_color(random_RGB_color))
        bot_pano_gray_vis = cv2.drawKeypoints(bot_pano_gray_vis, [bot_kpt], outImage=bot_pano_gray_vis, color=rgb2bgr_color(random_RGB_color))
    matches_img = np.vstack((top_pano_gray_vis, bot_pano_gray_vis[first_row_to_crop_bottom:, ...]))  # ATTENTION: Bottom panorama may be cropped.
    num_of_points = len(query_kpts)
    index_counter = 0
    while index_counter < num_of_points:
        top_kpt = train_kpts[index_counter]
        bot_kpt = query_kpts[index_counter]
        random_RGB_color = colors_RGB[index_counter]
        top_pt = (int(top_kpt.pt[0]), int(top_kpt.pt[1]))  # Recall, pt is given as (u,v)
        bot_pt = (int(bot_kpt.pt[0]), int(bot_kpt.pt[1] + row_offset - first_row_to_crop_bottom))
        if do_filtering:
            _, matches_img = filter_correspondences_manually(train_img=train_img, query_img=query_img, query_kpts=query_kpts[valid_indices[:index_counter + 1]], train_kpts=train_kpts[valid_indices[:index_counter + 1]], colors_RGB=colors_RGB[valid_indices[:index_counter + 1]], first_row_to_crop_bottom=first_row_to_crop_bottom, do_filtering=False)
            ch_pressed_waitkey = cv2.waitKey(0)
            if (ch_pressed_waitkey & 255) == ord('v'):  # To save as VALID mactch
                valid_indices[index_counter] = True
            elif (ch_pressed_waitkey & 255) == ord('r'):  # Rewind
                if index_counter > 0:
                    index_counter = index_counter - 1
                    valid_indices[index_counter] = True  # because it needs to be drawn again
                index_counter = index_counter - 1  # To ask again
            elif ch_pressed_waitkey == 27:  # Stop filtering at this point
                break
            else:
                valid_indices[index_counter] = False
        else:
            matches_img = cv2.line(matches_img, top_pt, bot_pt, color=rgb2bgr_color(random_RGB_color), thickness=1, lineType=cv2.LINE_8)
            win_name = 'Current Matches'
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, matches_img)
        index_counter += 1

    return valid_indices, matches_img

class StereoMatchTuner(object):
    def __init__(self, left_img, right_img, rotate_images=False, method="sgbm", win_name="Disparity Map", disp_first_valid_row=0, disp_last_valid_row=-1):
        self.reset_images(left_img, right_img, rotate_images)
        self.window_name = win_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.disp_first_valid_row = disp_first_valid_row
        self.disp_last_valid_row = disp_last_valid_row
        self.matching_method = method
        self.disparity_map = None
        self.disparity_img = None
        self.needs_update = True
        self.pano_mask = None
        # TODO: try several presets available:
        # //   CV_STEREO_BM_NORMALIZED_RESPONSE,
        # //   CV_STEREO_BM_BASIC,
        # //   CV_STEREO_BM_FISH_EYE,
        # //   CV_STEREO_BM_NARROW
        # // the preset is one of ..._PRESET above.
        # // the disparity search range. For each pixel algorithm will find the best disparity from 0 (default minimum disparity) to n disparities. The search range can then be shifted by changing the minimum disparity.

# Parameters for REAL experiments:
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

        if self.matching_method == "bm":
            self.median_kernel_size = 5
            self.SAD_window_size = 21  # Matched block size. It must be an odd number >=1", SAD_window_size_value, 1, 128)
            self.num_of_disparities = 128  # The size of the disparity search window. Together with min_disparity, this defines the horopter (the 3D volume that is visible to the stereo algorithm). This parameter must be divisible by 16", Range: 16, 256) # MAX disp shouldn't exceed the image's height
#             panoramic_stereo = cv2.StereoBM()
#             panoramic_stereo = cv2.StereoBM(preset=cv2.STEREO_BM_BASIC_PRESET, ndisparities=self.num_of_disparities, SADWindowSize=self.SAD_window_size)
#             panoramic_stereo = cv2.StereoBM(preset=cv2.STEREO_BM_FISH_EYE_PRESET, ndisparities=self.num_of_disparities, SADWindowSize=self.SAD_window_size)
        if self.matching_method == "sgbm":
            # Parameters for SYNTHETIC experiments:
            self.median_kernel_size = 0
            self.min_disparity = 1  # Minimum disparity (controls the offset from the x-position of the left pixel at which to begin searching)" (Range: -128, 128)
            self.disp_12_max_diff = -1  # left.shape[1] // 2  # TODO: compute with the nearest point: Maximum allowed difference (in integer pixel units) in the left-right disparity check (How many pixels to slide the window over). The larger it is, the larger the range of visible depths, but more computation is required. Set it to a non-positive value to disable the check.", Range: (-400, 400)
            self.num_of_disparities = 64  # The size of the disparity search window. Together with min_disparity, this defines the horopter (the 3D volume that is visible to the stereo algorithm). This parameter must be divisible by 16", Range: 16, 256) # MAX disp shouldn't exceed the image's height
            self.SAD_window_size = 3  # Matched block size. It must be an odd number >=1", SAD_window_size_value, 1, 128)
            self.smooth_P1 = 50  # The first parameter controlling the disparity smoothness", P1, 0, 4096
            self.smooth_P2 = 1000  # The second parameter controlling the disparity smoothness", P2, 0, 32768
            self.pre_filter_cap = 7  # Truncation value for the prefiltered image pixels. The algorithm first computes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval.", (1, 256)
            self.uniqueness_ratio = 20  # Margin in percentage by which the best (minimum) computed cost function value should win the second best value to consider the found match correct", (1, 100). Normally, a value within the 5-15 range is good enough
            self.speckle_window_size = 100  # Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.", (0, 600)
            self.speckle_range = 2  # Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
            self.full_dyn_prog = False  # not used by VAR stereo) Use Dynamic Programming? (DP uses more memory)", True)

            #===================================================================
            # # Parameters For REAL panoramas
            # self.median_kernel_size = 5
            # self.SAD_window_size = 10  # Matched block size. It must be an odd number >=1", SAD_window_size_value, 1, 128)
            # self.min_disparity = 16 - 129  # Minimum disparity (controls the offset from the x-position of the left pixel at which to begin searching)" (Range: -128, 128)
            # # self.num_of_disparities = ((self.rows/8) + 15) & -16
            # self.num_of_disparities = 240  # The size of the disparity search window. Together with min_disparity, this defines the horopter (the 3D volume that is visible to the stereo algorithm). This parameter must be divisible by 16", Range: 16, 256) # MAX disp shouldn't exceed the image's height
            # self.disp_12_max_diff = 0  # self.num_of_disparities - self.min_disparity  # left.shape[1] // 2  # TODO: compute with the nearest point: Maximum allowed difference (in integer pixel units) in the left-right disparity check (How many pixels to slide the window over). The larger it is, the larger the range of visible depths, but more computation is required. Set it to a non-positive value to disable the check.", Range: (-400, 400)
            # self.smooth_P1 = 4096  # 50 The first parameter controlling the disparity smoothness", P1, 0, 4096
            # self.smooth_P2 = 32768  # 100 The second parameter controlling the disparity smoothness", P2, 0, 32768
            # self.pre_filter_cap = 0  # Truncation value for the prefiltered image pixels. The algorithm first computes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval.", (1, 256)
            # self.uniqueness_ratio = 8  # Margin in percentage by which the best (minimum) computed cost function value should win the second best value to consider the found match correct", (1, 100)
            # self.speckle_window_size = 425  # Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.", (0, 600)
            # self.speckle_range = 16  # Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value.", (0, 32)
            # self.full_dyn_prog = True  # not used by VAR stereo) Use Dynamic Programming? (DP uses more memory)", True)
            #===================================================================
        if self.matching_method == "var":
            self.median_kernel_size = 5
            self.min_disparity = 128 - 128
            self.max_disparity = 192  # Default is 16
            self.var_cycles_dict = {"CYCLE_0":0, "CYCLE_V":1}
            self.var_cycle = "CYCLE_V"  #  "CYCLE_O", O-cycle or null-cycle: performs significantly more iterations on the coarse grids (faster convergence)
                                # "CYCLE_V" : The V-cycles makes one recursive call of a two-grid cycle per level.")
            self.var_penalizations_dict = {"P_TICHONOV":0, "P_CHARBONNIER":1, "P_PERONA_MALIK":2}
            self.var_penalization = "P_CHARBONNIER"  # Penalization regulizer method: "P_TICHONOV",  "P_CHARBONNIER", "P_PERONA_MALIK"
            self.var_levels = 8  # number of multigrid levels" (1, 10)
            self.var_pyrScale = 0.80  # VAR stereo: pyramid scale", (0.4, 1.0) Default: 0.5
            self.var_poly_n = 5  # degree of polynomial (see paper)", (1, 7) Default: 3
            self.var_poly_sigma = 0.64  # sigma value in polynomial (see paper) (0, 1.0)  # TODO: Find proper bounds
            self.var_fi = 80  # fi value (see paper)", (1.0, 100.0) Default: 25.
            self.var_lambda = 1.1  # lambda value (see paper): (0, 2.0) Default: 0.03 # TODO: Find proper bounds
            self.var_nIt = 15  # The number of iterations the algorithm does at each pyramid level.  (If the flag USE_SMART_ID is set, the number of iterations will be redistributed in such a way, that more iterations will be done on more coarser levels.)

            # TODO:
#             VAR_flag_auto_params = True
#             VAR_flag_init_disp = False  # USE_INITIAL_DISPARITY?", False
#             VAR_flag_eq_hist = False  # USE_EQUALIZE_HIST?", False
#             VAR_flag_smart_id = False  # USE_SMART_ID?", True
#             VAR_flag_median_filter = False  # USE_MEDIAN_FILTERING?", True
#             var_auto = 0;
#             var_init_params = 0;
#             var_eq_hist = 0;
#             var_smart_id = 0;
#             var_median_filter = 0;
#             if VAR_flag_auto_params:
#                 var_auto = cv2.StereoVar.USE_AUTO_PARAMS
#             if VAR_flag_init_disp:
#                 var_init_params = cv2.StereoVar.USE_INITIAL_DISPARITY
#             if VAR_flag_eq_hist:
#                 var_eq_hist = cv2.StereoVar.USE_EQUALIZE_HIST
#             if VAR_flag_smart_id:
#                 var_smart_id = cv2.StereoVar.USE_SMART_ID
#             if VAR_flag_median_filter:
#                 var_median_filter = cv2.StereoVar.USE_MEDIAN_FILTERING

#             var_flags = var_auto | var_init_params | var_eq_hist | var_smart_id | var_median_filter;
#             panoramic_stereo.flags = var_flags
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        self._setup_gui()

    def reset_images(self, left_img, right_img, rotate_images=True, disp_first_valid_row=0, disp_last_valid_row=-1):
        self.left_raw = left_img
        self.right_raw = right_img
        self.rotate_images = rotate_images
        self.disp_first_valid_row = disp_first_valid_row
        self.disp_last_valid_row = disp_last_valid_row
        self.rows, self.cols = left_img.shape[0:2]
        self.needs_update = True

    def start_tuning(self, win_name="", save_file_name="data/StereoMatchTuner.pkl", tune_live=False, pano_mask=None):
        self.pano_mask = pano_mask
        self.save_file_name = save_file_name
        if win_name:
            self.window_name = win_name
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._setup_gui()

#         if self.matching_method == "sgbm":
        self.disparity_map = np.zeros((self.rows, self.cols), dtype="float64")
#         else:
#             self.disparity_map = np.zeros((self.rows, self.cols), dtype="float32")

        self.disparity_img = np.zeros((self.rows, self.cols), dtype="uint8")

        if self.matching_method == "bm":
#             panoramic_stereo = cv2.StereoBM()
            while not(cv2.waitKey(10) == 27):
                if self.needs_update:
                    panoramic_stereo = cv2.StereoBM_create(ndisparities=self.num_of_disparities, SADWindowSize=self.SAD_window_size)
                    disparity = panoramic_stereo.compute(self.left, self.right)
                    disparity_rotated = cv2.flip(cv2.transpose(disparity), flipCode=1)
                    # It contains disparity values scaled by 16. So, to get the floating-point disparity map, you need to divide each disp element by 16.
#                     self.disparity_map = disparity_rotated / 16. + 1
#                     self.disparity_img = np.uint8((self.disparity_map - self.disparity_map.min()) * 255. / (self.disparity_map.max() - self.disparity_map.min()))
#                     self.disparity_map[self.disp_first_valid_row:] = (disparity_rotated / 16.)[self.disp_first_valid_row:]
#                     self.disparity_img[self.disp_first_valid_row:] = np.uint8((self.disparity_map[self.disp_first_valid_row:] - self.disparity_map.min()) * 255. / (self.disparity_map.max() - self.disparity_map.min()))
                    self.disparity_map = (disparity_rotated / 16.)
                    self.disparity_img = np.uint8((self.disparity_map - self.disparity_map.min()) * 255. / (self.disparity_map.max() - self.disparity_map.min()))

                    disp_img_color = cv2.cvtColor(self.disparity_img, cv2.COLOR_GRAY2BGR)
                    # Draw min/max bounds (valid region of disparities)
                    line_thickness = 2
                    line_color = (0, 255, 0)  # Green in BGR
#                     cv2.line(img=disp_img_color, pt1=(0, self.disp_first_valid_row), pt2=(self.cols - 1, self.disp_first_valid_row), color=line_color, thickness=line_thickness , lineType=cv2.LINE_AA)
#                     stereo_match_view = np.vstack((disp_img_color[self.disp_first_valid_row:self.disp_last_valid_row + 1], self.divider_line_img, self.stereo_view_img))
                    stereo_match_view = np.vstack((disp_img_color, self.divider_line_img, self.stereo_view_img))
                    cv2.imshow(self.window_name, stereo_match_view)
                    self.needs_update = False
                    if tune_live:
                        break

        if self.matching_method == "sgbm":
            number_of_image_channels = self.left.ndim
                    # while cv2.waitKey(1) == -1:  # While not any key has been pressed
            ch_pressed_waitkey = cv2.waitKey(10)
            while not(ch_pressed_waitkey == 27):  # Pressing the Escape key breaks the loop
                if self.needs_update:
                    if self.full_dyn_prog:
                        mode = cv2.STEREO_SGBM_MODE_HH  # to run the full-scale two-pass dynamic programming algorithm
                    else:
                        mode = cv2.STEREO_SGBM_MODE_SGBM
                    # self.smooth_P1 = 2 * number_of_image_channels * self.SAD_window_size * self.SAD_window_size
                    # self.smooth_P2 = 8 * number_of_image_channels * self.SAD_window_size * self.SAD_window_size
                    # panoramic_stereo.P1 = self.smooth_P1
                    # panoramic_stereo.P2 = self.smooth_P2
                    panoramic_stereo = cv2.StereoSGBM_create(minDisparity=self.min_disparity, numDisparities=self.num_of_disparities, blockSize=self.SAD_window_size, \
                                                             P1=0, P2=0, disp12MaxDiff=self.disp_12_max_diff, preFilterCap=self.pre_filter_cap, uniquenessRatio=self.uniqueness_ratio, \
                                                             speckleWindowSize=self.speckle_window_size, speckleRange=self.speckle_range, mode=mode\
                                                             )
#                     panoramic_stereo.minDisparity = self.min_disparity
#                     panoramic_stereo.numDisparities = self.num_of_disparities
#                     panoramic_stereo.SADWindowSize = self.SAD_window_size
                    #===========================================================
                    # panoramic_stereo.disp12MaxDiff = self.disp_12_max_diff  # TODO: value for nearest triangulated point
                    # panoramic_stereo.preFilterCap = self.pre_filter_cap
                    # panoramic_stereo.speckleRange = self.speckle_range
                    # panoramic_stereo.speckleWindowSize = self.speckle_window_size
                    # panoramic_stereo.uniquenessRatio = self.uniqueness_ratio
                    # panoramic_stereo.fullDP = self.full_dyn_prog
                    #===========================================================

                    disparity = panoramic_stereo.compute(self.left, self.right)
                    disparity_rotated = cv2.flip(cv2.transpose(disparity), flipCode=1)
                    # It contains disparity values scaled by 16. So, to get the floating-point disparity map, you need to divide each disp element by 16.
#                     self.disparity_map[self.disp_first_valid_row:self.disp_last_valid_row + 1] = (disparity_rotated / 16.)[self.disp_first_valid_row:self.disp_last_valid_row + 1]
#                     self.disparity_img[self.disp_first_valid_row:self.disp_last_valid_row + 1] = np.uint8((self.disparity_map[self.disp_first_valid_row:self.disp_last_valid_row + 1] - self.disparity_map.min()) * 255. / (self.disparity_map.max() - self.disparity_map.min()))
#                     self.disparity_map[self.disp_first_valid_row:] = (disparity_rotated / 16.)[self.disp_first_valid_row:]
#                     self.disparity_img[self.disp_first_valid_row:] = np.uint8((self.disparity_map[self.disp_first_valid_row:] - self.disparity_map.min()) * 255. / (self.disparity_map.max() - self.disparity_map.min()))
                    disparity_map_normalized = (disparity_rotated / 16.)
                    disparity_img_normalized = np.uint8((disparity_map_normalized - disparity_map_normalized.min()) * 255. / (disparity_map_normalized.max() - disparity_map_normalized.min()))
                    # (my TRICK) Filter out of bound values by applying panoramic mask to the disparity image and depth map using radial bounds:
                    self.disparity_map = np.zeros_like(disparity_map_normalized)
                    self.disparity_map = cv2.bitwise_and(src1=disparity_map_normalized, src2=disparity_map_normalized, dst=self.disparity_map, mask=self.pano_mask)
                    self.disparity_img = np.zeros_like(disparity_img_normalized)
                    self.disparity_img = cv2.bitwise_and(src1=disparity_img_normalized, src2=disparity_img_normalized, dst=self.disparity_img, mask=self.pano_mask)

                    disp_img_color = cv2.cvtColor(self.disparity_img, cv2.COLOR_GRAY2BGR)
                    # Draw min/max bounds (valid region of disparities)
                    line_thickness = 2
                    line_color = (0, 255, 0)  # Green in BGR
#                     cv2.line(img=disp_img_color, pt1=(0, self.disp_first_valid_row), pt2=(self.cols - 1, self.disp_first_valid_row), color=line_color, thickness=line_thickness , lineType=cv2.LINE_AA)

#                     stereo_match_view = np.vstack((disp_img_color[self.disp_first_valid_row:self.disp_last_valid_row + 1], self.divider_line_img, self.stereo_view_img))
#                     stereo_match_view = np.vstack((disp_img_color[self.disp_first_valid_row:], self.divider_line_img, self.stereo_view_img))
                    stereo_match_view = np.vstack((disp_img_color, self.divider_line_img, self.stereo_view_img))

                    cv2.imshow(self.window_name, stereo_match_view)
                    self.needs_update = False
                    if tune_live:
                        break

                ch_pressed_waitkey = cv2.waitKey(10)
                if (ch_pressed_waitkey & 255) == ord('s'):  # Save Tuner to pickle
                    from omnistereo.common_tools import save_obj_in_pickle
                    save_obj_in_pickle(self, self.save_file_name, locals())

        # NOTE: not longer supported by OpenCV 3
        # WARNING: it seems that one needs to run it twice (at first) for the algorithm to work better (as expected)
#===============================================================================
#         if self.matching_method == "var":
#             panoramic_stereo = cv2.StereoVar()
# #             while count < 2:
#             while not(cv2.waitKey(10) == 27):
#                 if self.needs_update:
#                     panoramic_stereo.levels = self.var_levels
#                     panoramic_stereo.pyrScale = self.var_pyrScale
#                     panoramic_stereo.nIt = self.var_nIt
#                     panoramic_stereo.minDisp = self.min_disparity
#                     panoramic_stereo.maxDisp = self.max_disparity
#                     panoramic_stereo.poly_n = self.var_poly_n
#                     panoramic_stereo.poly_sigma = self.var_poly_sigma
#                     panoramic_stereo.fi = self.var_fi
#                     # panoramic_stereo.lambda  = self.var_lambda # TODO: not allowed to call this in Python, but for some reason the class has it
#                     panoramic_stereo.penalization = self.var_penalizations_dict[self.var_penalization]
#                     panoramic_stereo.cycle = self.var_cycles_dict[self.var_cycle]
#
#                     # panoramic_stereo.flags = var_flags # TODO: How?
#
#                     disparity_rotated = panoramic_stereo.compute(self.left, self.right)
#                     num_of_disparities = float(panoramic_stereo.maxDisp - panoramic_stereo.minDisp)
#     #                 self.disparity_img = np.zeros_like(self.disparity_map, dtype=np.uint8)
#     #                 self.disparity_img = np.uint8((self.disparity_map + 1) * 255. / num_of_disparities)
#                     self.disparity_map[self.disp_first_valid_row:] = cv2.flip(cv2.transpose(disparity_rotated), flipCode=1)[self.disp_first_valid_row:]
#                     # Recall that ranges don't include the last index (so we want to include it with +1)
#                     # FIXME: it seems that the some rows are exceeding the rows of the image while resolving the match coordinates on the other panorama
#                     self.disparity_img[self.disp_first_valid_row:] = np.uint8((self.disparity_map[self.disp_first_valid_row:] - self.disparity_map.min()) * 255. / (self.disparity_map.max() - self.disparity_map.min()))
#
#                     disp_img_color = cv2.cvtColor(self.disparity_img, cv2.COLOR_GRAY2BGR)
#                     stereo_match_view = np.vstack((disp_img_color[self.disp_first_valid_row:], self.divider_line_img, self.stereo_view_img))
#                     cv2.imshow(self.window_name, stereo_match_view)
#                     self.needs_update = False
#                     if tune_live:
#                         break
#                     # FIXME: it seems that the image buffer (or the np.array is being reused while displaying)
#                     # Also, related to above problem forcing to run it twice (at first)
#                     # TODO: crop the noise bottom results from the depth map (since anything beyond the black stripe is incorrect)
#===============================================================================

        return self.disparity_map, self.disparity_img

    def _setup_gui(self):
        # General Trackbars (sliders)
        self.tb_name_median_filter = "Median Filter"
        cv2.createTrackbar(self.tb_name_median_filter, self.window_name, self.median_kernel_size, 32, self.on_median_filter_callback)

        if self.matching_method == "bm" or self.matching_method == "sgbm":
            self.tb_name_SAD_window_size = "SAD window size"
            cv2.createTrackbar(self.tb_name_SAD_window_size, self.window_name, self.SAD_window_size, int(self.rows / 2), self.on_SAD_win_size_callback)
            self.tb_name_num_of_disparities = "N Disparities"
            cv2.createTrackbar(self.tb_name_num_of_disparities, self.window_name, self.num_of_disparities, (self.rows // 16) * 16, self.on_num_of_disps_callback)

        if self.matching_method == "sgbm":
            self.tb_name_min_disp = "Min Disp [-128,128]"
            cv2.createTrackbar(self.tb_name_min_disp, self.window_name, 128 + self.min_disparity, 256, self.on_min_disp_callback)
            self.tb_name_disp12MaxDiff = "disp12MaxDiff"
            cv2.createTrackbar(self.tb_name_disp12MaxDiff, self.window_name, self.disp_12_max_diff, self.rows, self.on_disp_12_max_diff_callback)
            self.tb_name_pre_filter_cap = "PreFilter Cap"
            cv2.createTrackbar(self.tb_name_pre_filter_cap, self.window_name, self.pre_filter_cap, 256, self.on_pre_filter_cap_callback)
            self.tb_name_uniqueness_ratio = "Uniqueness Ratio"
            cv2.createTrackbar(self.tb_name_uniqueness_ratio, self.window_name, self.uniqueness_ratio, 100, self.on_uniqueness_ratio_callback)
            self.tb_name_speckle_range = "Speckle Range"
            cv2.createTrackbar(self.tb_name_speckle_range, self.window_name, self.speckle_range, 32, self.on_speckle_range_callback)
            self.tb_name_speckle_win_size = "Speckle Window Size"
            cv2.createTrackbar(self.tb_name_speckle_win_size, self.window_name, self.speckle_window_size, 1000, self.on_speckle_win_size_callback)
            self.button_name_full_dyn_prog = "Full DP"
            cv2.createTrackbar(self.button_name_full_dyn_prog, self.window_name, self.full_dyn_prog, 1, self.full_DP_callback)
        if self.matching_method == "var":
            self.tb_name_min_disp = "Min Disp [-128,128]"
            cv2.createTrackbar(self.tb_name_min_disp, self.window_name, 128 + self.min_disparity, 256, self.on_min_disp_callback)
            self.tb_name_max_disp = "Max Disp"
            cv2.createTrackbar(self.tb_name_max_disp, self.window_name, self.max_disparity, 256, self.on_max_disp_callback)
            self.tb_name_cycle = "CYCLE"
            cv2.createTrackbar(self.tb_name_cycle, self.window_name, self.var_cycles_dict[self.var_cycle], 1, self.on_cycle_callback)
            self.tb_name_penalization = "PENALIZATION"
            cv2.createTrackbar(self.tb_name_penalization, self.window_name, self.var_penalizations_dict[self.var_penalization], 2, self.on_penalization_callback)
            self.tb_name_levels = "Levels"
            cv2.createTrackbar(self.tb_name_levels, self.window_name, self.var_levels, 10, self.on_levels_callback)
            self.tb_name_pyr_scale = "Pyr.Sc x100"
            cv2.createTrackbar(self.tb_name_pyr_scale, self.window_name, int(self.var_pyrScale * 100), 100, self.on_pyr_scale_callback)
            self.tb_name_poly_n = "Poly Num"
            cv2.createTrackbar(self.tb_name_poly_n, self.window_name, self.var_poly_n, 7, self.on_poly_n_callback)
            self.tb_name_poly_sigma = "Poly Sigma"
            cv2.createTrackbar(self.tb_name_poly_sigma, self.window_name, int(self.var_poly_sigma * 100), 100, self.on_poly_sigma_callback)
            self.tb_name_fi = "Fi"
            cv2.createTrackbar(self.tb_name_fi, self.window_name, self.var_fi, 100, self.on_fi_callback)
#             self.tb_name_lambda = "Lambda"
#             cv2.createTrackbar(self.tb_name_lambda, self.window_name, int(self.var_lambda * 100), 200, self.on_lambda_callback)
            self.tb_name_num_iters = "N Iters"
            cv2.createTrackbar(self.tb_name_num_iters, self.window_name, self.var_nIt, 50, self.on_num_iters_callback)

        self.preprocess_images()  # Applies median filter and rotation if any

    def preprocess_images(self):
        if self.median_kernel_size > 0:
            left_img = cv2.medianBlur(self.left_raw, self.median_kernel_size)
            right_img = cv2.medianBlur(self.right_raw, self.median_kernel_size)
        else:
            left_img = self.left_raw
            right_img = self.right_raw

        if self.rotate_images:
            # Produce vertical panoramas
            self.left = cv2.flip(cv2.transpose(left_img), flipCode=0)
            self.right = cv2.flip(cv2.transpose(right_img), flipCode=0)

            # Composite horizontal visualization top above bottom
            if left_img.ndim < 3:
                left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
            if right_img.ndim < 3:
                right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)

            line_thickness = 4
            line_color = (0, 255, 255)  # yellow in BGR
            self.divider_line_img = np.zeros_like(left_img)[:line_thickness]
            cv2.line(img=self.divider_line_img, pt1=(0, 0), pt2=(self.cols - 1, 0), color=line_color, thickness=line_thickness , lineType=cv2.LINE_AA)

#             self.stereo_view_img = np.vstack((left_img[self.disp_first_valid_row:], self.divider_line_img, right_img))
            self.stereo_view_img = np.vstack((left_img, self.divider_line_img, right_img))
        else:
            # WISH: side to side stereo view
            self.left = left_img
            self.right = right_img

    def on_median_filter_callback(self, pos):
        if pos > 0 and pos % 2 == 0:  # Even values are not allowed
            self.median_kernel_size = pos + 1  # Jump ahead to an odd value
            cv2.setTrackbarPos(self.tb_name_median_filter, self.window_name, self.median_kernel_size)
        else:
            self.median_kernel_size = pos

        self.preprocess_images()
        self.needs_update = True

    def on_SAD_win_size_callback(self, pos):
        if pos > 0:
            if self.matching_method == "bm":
                if pos < 5:
                    pos = 5
                if pos % 2 == 0:  # ADWindowSize must be odd
                    pos = pos + 1  # Jump ahead to an odd value
                cv2.setTrackbarPos(self.tb_name_SAD_window_size, self.window_name, pos)

            self.SAD_window_size = pos
            self.needs_update = True

    def on_min_disp_callback(self, pos):
        self.min_disparity = pos - 128  # because value range goes from -128 to 128
        self.needs_update = True

    def on_max_disp_callback(self, pos):
        self.max_disparity = pos
        self.needs_update = True

    def on_num_of_disps_callback(self, pos):
        if pos == 0:
            self.num_of_disparities = 16
        else:
            self.num_of_disparities = (pos // 16) * 16
        cv2.setTrackbarPos(self.tb_name_num_of_disparities, self.window_name, self.num_of_disparities)
        self.needs_update = True

    def on_disp_12_max_diff_callback(self, pos):
        if pos == 0:
            self.disp_12_max_diff = -1
        else:
            self.disp_12_max_diff = pos

        self.needs_update = True

    def on_pre_filter_cap_callback(self, pos):
        self.pre_filter_cap = pos
        self.needs_update = True

    def on_speckle_range_callback(self, pos):
        self.speckle_range = pos
        self.needs_update = True

    def on_speckle_win_size_callback(self, pos):
        self.speckle_window_size = pos
        self.needs_update = True

    def on_uniqueness_ratio_callback(self, pos):
        self.uniqueness_ratio = pos
        self.needs_update = True

    def full_DP_callback(self, pos):
        if pos == 0:
            self.full_dyn_prog = False
        else:
            self.full_dyn_prog = True
        self.needs_update = True

    def on_cycle_callback(self, pos):
        self.var_cycle = self.var_cycles_dict.keys()[self.var_cycles_dict.values().index(pos)]
        self.needs_update = True

    def on_penalization_callback(self, pos):
        self.var_penalization = self.var_penalizations_dict.keys()[self.var_penalizations_dict.values().index(pos)]
        self.needs_update = True

    def on_levels_callback(self, pos):
        if pos == 0:
            self.var_levels = 1
            cv2.setTrackbarPos(self.tb_name_levels, self.window_name, self.var_levels)
        else:
            self.var_levels = pos
        self.needs_update = True

    def on_pyr_scale_callback(self, pos):
        min_perc_scale = 10
        if pos < min_perc_scale:
            self.var_pyrScale = min_perc_scale / 100.
            cv2.setTrackbarPos(self.tb_name_pyr_scale, self.window_name, int(self.var_pyrScale * 100))
        else:
            self.var_pyrScale = pos / 100.
        self.needs_update = True

    def on_poly_n_callback(self, pos):
        if pos % 2 == 0:  # Even number of polynomial terms are not allowed
            self.var_poly_n = pos + 1  # Make it odd
            cv2.setTrackbarPos(self.tb_name_poly_n, self.window_name, self.var_poly_n)
        else:
            self.var_poly_n = pos
        self.needs_update = True

    def on_poly_sigma_callback(self, pos):
        self.var_poly_sigma = pos / 100.

    def on_fi_callback(self, pos):
        if pos == 0:
            self.var_fi = 1
            cv2.setTrackbarPos(self.tb_name_fi, self.window_name, self.var_fi)
        else:
            self.var_fi = pos
        self.needs_update = True

    def on_lambda_callback(self, pos):
        self.var_lambda = pos / 100.
        self.needs_update = True

    def on_num_iters_callback(self, pos):
        self.var_nIt = pos
        self.needs_update = True

class PointClicker(object):
    def __init__(self, win_name, max_clicks=1, save_path="", draw_polygon_clicks=False):
        self.window_name = win_name
        self.save_path = save_path
        self.click_counter = 0
        self.img_save_number = 0
        self.is_new_mouse_click = False
        self.max_number_of_clicks = max_clicks
        self.clicked_points = np.ndarray((self.max_number_of_clicks, 2), dtype=int)
        self.shift_mouse_pos = None
        self.verbose = True
        self.draw_lines = draw_polygon_clicks
        self.lines = self.max_number_of_clicks * [None]  # To Contain list of line pairs for example: [[(x0,y0),(x1,y1)], [(x1,y1),(x2,y2)],[(x2,y2),(x_curr,y_curr)]]
        cv2.setMouseCallback(self.window_name, self.on_mouse_callback)

    def on_mouse_callback(self, event, xc, yc, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            if flags == cv2.EVENT_FLAG_SHIFTKEY:
#         if flags == (cv2.EVENT_LBUTTONDOWN + cv2.EVENT_FLAG_SHIFTKEY):
                self.shift_mouse_pos = (xc, yc)
            if self.draw_lines:
                if self.click_counter > 0 and self.click_counter != self.max_number_of_clicks:
                    self.lines[self.click_counter - 1] = [tuple(self.clicked_points[self.click_counter - 1]), (xc, yc)]
                    self.is_new_mouse_click = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.click_counter += 1
            if self.click_counter > self.max_number_of_clicks:
                self.click_counter = 1  # Reset counter
                self.lines = self.max_number_of_clicks * [None]  # Reset all lines
            self.clicked_points[self.click_counter - 1] = (xc, yc)
            if self.draw_lines:
                if self.click_counter > 1:
                    self.lines[self.click_counter - 2] = [tuple(self.clicked_points[self.click_counter - 2]), tuple(self.clicked_points[self.click_counter - 1])]
                    if self.click_counter == self.max_number_of_clicks:  # Close the loop
                        self.lines[self.click_counter - 1] = [tuple(self.clicked_points[self.click_counter - 1]), tuple(self.clicked_points[0])]
            if self.verbose:
                print("Clicked on (u,v) = ", self.clicked_points[self.click_counter - 1])
            self.is_new_mouse_click = True

    def get_clicks_uv_coords(self, img, verbose=True):
        '''
        @return: the np array of valid points clicked. NOTE: the arrangement is in the (u,v) coordinates
        '''
        self.verbose = verbose
        cv2.imshow(self.window_name, img)

        # while cv2.waitKey(1) == -1:  # While not any key has been pressed
        ch_pressed_waitkey = cv2.waitKey(1)
        while not(ch_pressed_waitkey == 27):  # Pressing the Escape key breaks the loop
            if (ch_pressed_waitkey & 255) == ord('r'):
                self.click_counter = 0  # reset count
                self.is_new_mouse_click = True
                self.lines = self.max_number_of_clicks * [None]

            # Grab a point
            if self.is_new_mouse_click:
                channels = img.ndim
                    # img_copy = img.copy()  # Keep the original image
                if channels < 3:
                    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    vis = img.copy()

                draw_points(vis, self.clicked_points, num_valid_points=self.click_counter)
                draw_lines(vis, self.lines, thickness=5)
                draw_str(vis, (10, 20), "Keyboard Commands:")
                draw_str(vis, (20, 40), "'R': to restart")
                draw_str(vis, (20, 60), "'Esc': to finish")
                cv2.imshow(self.window_name, vis)

            self.is_new_mouse_click = False  # Reset indicator
            ch_pressed_waitkey = cv2.waitKey(1)

#         cv2.destroyWindow(self.window_name)

        return self.clicked_points[:self.click_counter]

    def get_clicks_uv_coords_for_stereo(self, stereo_model, show_correspondence_on_circular_img=False, min_disparity=1, max_disparity=0, verbose=False):
        '''
        @return: the two np arrays of valid points clicked and its correspondences. NOTE: the arrangement is in the (u,v) coordinates
        '''
        self.verbose = verbose
        target_window_name = 'Target Point Correspondences'
        cv2.namedWindow(target_window_name, cv2.WINDOW_NORMAL)

        target_coords = None
        reference_coords = None
        img_reference = stereo_model.top_model.panorama.panoramic_img  # Acting as the right image
        img_target = stereo_model.bot_model.panorama.panoramic_img
        if show_correspondence_on_circular_img:
            omni_top_coords = None
            omni_bot_coords = None
            img_omni = stereo_model.current_omni_img
            omni_window_name = 'Correspondences on Omni Image'
            cv2.namedWindow(omni_window_name, cv2.WINDOW_NORMAL)

        cv2.imshow(self.window_name, img_reference)
        # while cv2.waitKey(1) == -1:  # While not any key has been pressed
        ch_pressed_waitkey = cv2.waitKey(1)
        while not(ch_pressed_waitkey == 27):  # Pressing the Escape key breaks the loop
            if (ch_pressed_waitkey & 255) == ord('r'):
                self.click_counter = 0  # reset count
                self.is_new_mouse_click = False
                cv2.imshow(self.window_name, img_reference)
                cv2.imshow(target_window_name, img_target)
                if show_correspondence_on_circular_img:
                    cv2.imshow(omni_window_name, img_omni)

            # Grab a point
            if self.is_new_mouse_click:
                channels = img_reference.ndim
                    # img_copy = img_reference.copy()  # Keep the original image
                if channels < 3:
                    vis_ref = cv2.cvtColor(img_reference, cv2.COLOR_GRAY2BGR)
                    vis_target = cv2.cvtColor(img_target, cv2.COLOR_GRAY2BGR)
                    if show_correspondence_on_circular_img:
                        vis_omni = cv2.cvtColor(img_omni, cv2.COLOR_GRAY2BGR)
                else:
                    vis_ref = img_reference.copy()
                    vis_target = img_target.copy()
                    if show_correspondence_on_circular_img:
                        vis_omni = img_omni.copy()

                # Find correspondence
                reference_coords, target_coords, disparities = stereo_model.resolve_pano_correspondences_from_disparity_map(self.clicked_points[:self.click_counter], min_disparity=min_disparity, max_disparity=max_disparity, verbose=verbose)
                # Update clicks
                self.click_counter = np.count_nonzero(reference_coords) / 2
                self.clicked_points[:self.click_counter] = reference_coords

                # Write instructions on image
                draw_str(vis_ref, (10, 20), "Keyboard Commands:")
                draw_str(vis_ref, (20, 40), "'R': to restart")
                draw_str(vis_ref, (20, 60), "'Esc': to finish")
                # Draw points on panoramas
                ref_pts_color = (255, 0, 0)  # RGB
                tgt_pts_color = (0, 0, 255)  # RGB
                pt_thickness = 5
                draw_points(vis_ref, reference_coords.reshape(-1, 2), color=ref_pts_color, thickness=pt_thickness)
                cv2.imshow(self.window_name, vis_ref)
                draw_points(vis_target, reference_coords.reshape(-1, 2), color=ref_pts_color, thickness=pt_thickness)
                draw_points(vis_target, target_coords.reshape(-1, 2), color=tgt_pts_color, thickness=pt_thickness)
                cv2.imshow(target_window_name, vis_target)
                if show_correspondence_on_circular_img and self.click_counter > 0 and self.verbose:
                    _, _, omni_top_coords = stereo_model.top_model.panorama.get_omni_pixel_coords_from_panoramic_pixel(reference_coords)
                    _, _, omni_bot_coords = stereo_model.bot_model.panorama.get_omni_pixel_coords_from_panoramic_pixel(target_coords)
                    print("Omni pixel coords: TOP %s, BOT %s" % (omni_top_coords[0, self.click_counter - 1], omni_bot_coords[0, self.click_counter - 1]))
                    draw_points(vis_omni, omni_top_coords[..., :2].reshape(-1, 2), color=ref_pts_color, thickness=pt_thickness)
                    draw_points(vis_omni, omni_bot_coords[..., :2].reshape(-1, 2), color=tgt_pts_color, thickness=pt_thickness)
                    cv2.imshow(omni_window_name, vis_omni)

            self.is_new_mouse_click = False  # Reset indicator
            ch_pressed_waitkey = cv2.waitKey(1)

        return reference_coords, target_coords, disparities

    def save_image(self, img, img_name=None):
        if img_name:
            name_prefix = img_name
        else:
            name_prefix = "img"

        img_name = '%s-%d.png' % (name_prefix, self.img_save_number)

        if self.save_path:
            complete_save_name = self.save_path + img_name
        else:
            complete_save_name = img_name

        print('Saving', complete_save_name)
        cv2.imwrite(complete_save_name, img)

        self.img_save_number += 1  # Increment save counter

def rgb2bgr_color(rgb_color):
    return (rgb_color[2], rgb_color[1], rgb_color[0])

def draw_points(img_input, points_uv_coords, num_valid_points=None, color=None, thickness=1):
    '''
    @param img_input: The image on which points will be drawn to (NOTE: it doesn't preserve the image)
    @param points_uv_coords: FIXME: the uv coordinates list or ndarray must be of shape (n, 2) for n points.
    Note that the coordinates will be expressed as integers while visualizing
    @param color: a 3-tuple of the RGB color for these points
    '''
    if color == None:
        color = (0, 0, 255)  # Red because BGR(B,G,R)
    else:  # Swap the passed color from RGB into BGR
        color = rgb2bgr_color(color)

    if num_valid_points == None:
        num_valid_points = len(points_uv_coords)

    for i in range(num_valid_points):
        pt = points_uv_coords[i]
        if np.isnan(pt[0]) or np.isnan(pt[1]):
            print("nan cannot be drawn!")
        else:
            pt_as_tuple = (int(pt[0]), int(pt[1]))  # Recall: (pt[0],pt[1]) # (x, u or col and y, v or row)
            cv2.circle(img_input, pt_as_tuple, 2, color, thickness, 8, 0)

def draw_lines(img_input, lines_list, color=None, thickness=2):
    '''
    @param img_input: The image on which points will be drawn to (NOTE: it doesn't preserve the image)
    @param lines_list: A list of point pairs such as [[(x0,y0),(x1,y1)], [(x1,y1),(x2,y2)],[(x2,y2),(x_last,y_Last)], None, None]
    @param color: a 3-tuple of the RGB color for these points
    '''
    if color == None:
        color = (0, 0, 255)  # Red because BGR(B,G,R)
    else:  # Swap the passed color from RGB into BGR
        color = rgb2bgr_color(color)

    for pts in lines_list:
        if pts is not None:
            [pt_beg, pt_end] = pts
            cv2.line(img_input, pt_beg, pt_end, color, thickness=thickness, lineType=cv2.LINE_AA)

def get_masked_omni_image(img_input, center_point, outer_radius, inner_radius=0.0, color_RGB=None):
    '''
    @param color_RGB: A tuple specifying the desired background as (Red,Green,Blue). If None, the background is black
    '''
    mask = np.zeros(img_input.shape[0:2], dtype=np.uint8)  # Black, single channel mask

    # Paint outer perimeter:
    cv2.circle(mask, center_point, int(outer_radius), (255, 255, 255), -1, 8, 0)

    # Paint inner perimeter:
    if inner_radius > 0:
        cv2.circle(mask, center_point, int(inner_radius), (0, 0, 0), -1, 8, 0)

    # Apply mask
    masked_img = np.zeros(img_input.shape)
    masked_img = cv2.bitwise_and(src1=img_input, src2=img_input, dst=masked_img, mask=mask)

    if color_RGB is not None:  # Paint the masked area other than black
        background_img = np.zeros_like(masked_img)
        color_BGR = np.array([color_RGB[2], color_RGB[1], color_RGB[0]], dtype="uint8")
        background_img[..., :] += color_BGR  # Paint the B-G-R channels for OpenCV
        mask_inv = cv2.bitwise_not(src=mask)
        # Apply the background using the inverted mask
        masked_img = cv2.bitwise_and(src1=background_img, src2=background_img, dst=masked_img, mask=mask_inv)
        #=======================================================================
        # mask2 = np.zeros(img_input.shape[0:2], dtype=np.uint8) + 255  # Now, we start on a white canvas
        # # Paint outer perimeter (a black filled circle):
        # cv2.circle(mask2, center_point, int(outer_radius), (0, 0, 0), -1, 8, 0)
        # # Paint a white inner perimeter:
        # if inner_radius > 0:
        #     cv2.circle(mask2, center_point, int(inner_radius), (255, 255, 255), -1, 8, 0)
        # # Apply mask
        # masked_img = cv2.bitwise_and(src1=background_img, src2=background_img, dst=masked_img, mask=mask2)
        #=======================================================================

    return masked_img

def get_images(filename_template, indices_list=[], show_images=False):
    '''
    @param indices_list: Returns only those images indices from the entire list. If this list is empty (default), all images read are returned
    @note: all images files acquired by glob will be read and shown (however), but only those indexed in the list (if any) will be returned
    @return: A list of the retrieved images (based on an index list, if any) from the filenames template
    '''
    from glob import glob
    img_names = glob(filename_template)

    l = len(img_names)
    images = l * [None]
    if indices_list == None or len(indices_list) == 0:
        indices_list = range(l)

    for i in indices_list:
        try:
            fn = img_names[i]
            print('Reading %s...' % fn, end="")
            img = cv2.imread(fn)
            if img is not None:
                print("success")
                images[i] = img
                if show_images:
                    path, name, ext = splitfn(fn)
                    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                    cv2.imshow(name, img)
        except:
            warnings.warn("Warning...image index %d not found at %s" % (i, __name__))

    if show_images:
        cv2.waitKey(1)

    # We want the whole list!, so this is not good any more
#     if len(indices_list) == 0:
#         return images  # all
#     else:
#         return list(np.take(images, indices_list, 0))

    return images  # all even if None

def get_feature_matches_data_from_files(filename_template, indices_list=[]):
    '''
    @param indices_list: Returns only those frame indices from the entire list. If this list is empty (default), all existing pickles are read are returned
    @note: all feature_data files acquired by glob will be read and shown (however), but only those indexed in the list (if any) will be returned
    @return: A list of the retrieved feature_data (based on an index list, if any) from the filenames template. NOTE that cv2.KeyPoints have been serialized as tuples.
    Each data entry in the returned list is organized as: (matched_m_top, matched_kpts_top_serial, matched_desc_top), (matched_m_bot, matched_kpts_bot_serial, matched_desc_bot), random_colors_RGB
    '''
    from omnistereo.common_tools import load_obj_from_pickle

    if indices_list == None or len(indices_list) == 0:
        from glob import glob
        pickle_names = glob(filename_template)
        l = len(pickle_names)
        indices_list = range(l)
    else:
        l = indices_list[-1] + 1  # Assuming indices are ordered increasingly

    feature_data = l * [None]

    for i in indices_list:
        features_data_filename = filename_template.replace("*", str(i), 1)
        try:
            print('Reading %s...' % features_data_filename, end="")
            data = load_obj_from_pickle(filename=features_data_filename)
            feature_data[i] = data
        except:
            warnings.warn("Warning...file index %d not found at %s" % (i, __name__))

    return feature_data  # all even if None

def get_masked_images_mono(unmasked_images, camera_model, img_indices=[], show_images=False, color_RGB=None):
    '''
    @param color_RGB: A tuple specifying the desired background as (Red,Green,Blue). If None, the background is black
    '''

    l = len(unmasked_images)
    masked_images = l * [None]
    if img_indices is None or len(img_indices) == 0:
        img_indices = range(l)  # Use all images

    if hasattr(camera_model, "outer_img_radius"):
        u0, v0 = camera_model.get_center()
        use_circular_mask = True
    else:
        use_circular_mask = False

    for i in img_indices:
        try:
            img = unmasked_images[i]
            if use_circular_mask:
                masked_img = get_masked_omni_image(img, (int(u0), int(v0)), camera_model.outer_img_radius, camera_model.inner_img_radius, color_RGB=color_RGB)
            else:
                masked_img = img.copy()

            if masked_img is not None:
                masked_images[i] = masked_img

                if show_images:
                    win_name = "%s (masked) - [%d]" % (camera_model.mirror_name , i)
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(win_name, masked_img)
        except:
            print("Problems with masking image_%d" % (i))

    if show_images:
        cv2.waitKey(1)

    return masked_images

def get_masked_images_as_pairs(unmasked_images, omnistereo_model, img_indices=[], show_images=False, color_RGB=None):
    '''
    @param color_RGB: A tuple specifying the desired background as (Red,Green,Blue). If None, the background is black
    '''

    l = len(unmasked_images)
    masked_images = l * [None]
    if img_indices is None or len(img_indices) == 0:
        img_indices = range(l)  # Use all images

    for i in img_indices:
        try:
            img = unmasked_images[i]
            masked_img_top, masked_img_bottom = omnistereo_model.get_fully_masked_images(omni_img=img, view=False, color_RGB=color_RGB)
            masked_images[i] = (masked_img_top, masked_img_bottom)

            if show_images:
                win_name_top = "Top masked - [%d]" % (i)
                win_name_bot = "Bottom masked - [%d]" % (i)
                cv2.namedWindow(win_name_top, cv2.WINDOW_NORMAL)
                cv2.namedWindow(win_name_bot, cv2.WINDOW_NORMAL)
                cv2.imshow(win_name_top, masked_img_top)
                cv2.imshow(win_name_bot, masked_img_bottom)
        except:
            print("Problems with masking image [%d]" % (i))

    #===========================================================================
    # masked_images_top = get_masked_images_mono(unmasked_images, omnistereo_model.top_model, img_indices=[], show_images=show_images, color_RGB=color_RGB)
    # masked_images_bottom = get_masked_images_mono(unmasked_images, omnistereo_model.bot_model, img_indices=[], show_images=show_images, color_RGB=color_RGB)
    #===========================================================================

    if show_images:
        cv2.waitKey(1)

    return masked_images  # zip(masked_images_top, masked_images_bottom)

def create_arbitrary_mask(img_input, points, preview=False):
    mask = np.zeros(img_input.shape[0:2], dtype=np.uint8)  # Black, single channel mask
    masked_img = np.zeros(img_input.shape)

    cv2.fillConvexPoly(mask, points, color=(255, 255, 255), lineType=8, shift=0)
    masked_img = cv2.bitwise_and(img_input, img_input, masked_img, mask=mask)

    if preview:
        resulting_mask_window_name = "Resulting Mask"
        cv2.namedWindow(resulting_mask_window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(resulting_mask_window_name, masked_img)
        cv2.waitKey(0)
        cv2.destroyWindow(resulting_mask_window_name)

    return masked_img

def refine_radial_bounds_mono(omni_img, initial_values=[]):
    '''
    @param initial_values: A list of initial values: [center_pixel, outer_radius, inner_radius]
    '''
    if len(initial_values) == 3:
        center_pixel, outer_radius, inner_radius = initial_values
    else:
        center_pixel, outer_radius, inner_radius = None, None, None

    # Find circular boundaries
    win_name_outter = "Outter boundary"
    center_pixel, outer_radius = extract_circular_bound(omni_img, win_name_outter, center_coords=center_pixel, radius=outer_radius)
    win_name_inner = "Inner boundary"
    center_pixel, inner_radius = extract_circular_bound(omni_img, win_name_inner, center_coords=center_pixel, radius=inner_radius)

    from cv2 import destroyWindow
    destroyWindow(win_name_outter)
    destroyWindow(win_name_inner)

    return (center_pixel, outer_radius, inner_radius)

def refine_radial_bounds(omni_img, top_values, bottom_values):
    [[center_pixel_top, outer_radius_top, inner_radius_top], [center_pixel_bottom, outer_radius_bottom, inner_radius_bottom]] = find_center_and_radial_bounds(omni_img, initial_values=[top_values, bottom_values], save_to_file=False)
    return (outer_radius_top, inner_radius_top), (outer_radius_bottom, inner_radius_bottom)


def find_center_and_radial_bounds(omni_img, initial_values=[], radial_bounds_filename="", save_to_file=True, fiducial_rings_radii_top=[], fiducial_rings_radii_bottom=[], is_stereo=True):
    from cv2 import destroyWindow

    # TODO: Load existing filename if any:
    # if exists, use it to initialize the data
    import os.path
    file_exists = os.path.isfile(radial_bounds_filename)
    if file_exists:
        from omnistereo.common_tools import load_obj_from_pickle
    if save_to_file:
        from omnistereo.common_tools import save_obj_in_pickle

    if is_stereo:
        if file_exists:
            [[center_pixel_top, outer_radius_top, inner_radius_top], [center_pixel_bottom, outer_radius_bottom, inner_radius_bottom]] = load_obj_from_pickle(radial_bounds_filename)
        else:
            if len(initial_values) > 0:
                # use initial values and do testing
                [[center_pixel_top, outer_radius_top, inner_radius_top], [center_pixel_bottom, outer_radius_bottom, inner_radius_bottom]] = initial_values
            else:
                [[center_pixel_top, outer_radius_top, inner_radius_top], [center_pixel_bottom, outer_radius_bottom, inner_radius_bottom]] = [[None, None, None], [None, None, None]]

        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Find center and radial boundaries
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        #=======================================================================
        # Find circular boundaries
        win_name_top_outter = "Top mirror - outter boundary"
        center_pixel_top, outer_radius_top = extract_circular_bound(omni_img, win_name_top_outter, center_coords=center_pixel_top, radius=outer_radius_top, ring_fiducials_radii=fiducial_rings_radii_top)
        win_name_top_inner = "Top mirror - inner boundary"
        center_pixel_top, inner_radius_top = extract_circular_bound(omni_img, win_name_top_inner, center_coords=center_pixel_top, radius=inner_radius_top, ring_fiducials_radii=fiducial_rings_radii_top)
        # NOTE: trusting only the centers extracted from the outer radius
        win_name_bottom_outter = "Bottom mirror - outter boundary"
        center_pixel_bottom, outer_radius_bottom = extract_circular_bound(omni_img, win_name_bottom_outter, center_coords=center_pixel_bottom, radius=outer_radius_bottom, ring_fiducials_radii=fiducial_rings_radii_bottom)
        win_name_bottom_inner = "Bottom mirror - inner boundary"
        center_pixel_bottom, inner_radius_bottom = extract_circular_bound(omni_img, win_name_bottom_inner, center_coords=center_pixel_bottom, radius=inner_radius_bottom, ring_fiducials_radii=fiducial_rings_radii_bottom)
        # NOTE: trusting only the center extracted from the outer radius
        if save_to_file:
            save_obj_in_pickle([[center_pixel_top, outer_radius_top, inner_radius_top], [center_pixel_bottom, outer_radius_bottom, inner_radius_bottom]], radial_bounds_filename, locals())

        destroyWindow(win_name_top_outter)
        destroyWindow(win_name_top_inner)
        destroyWindow(win_name_bottom_outter)
        destroyWindow(win_name_bottom_inner)

        return [[center_pixel_top, outer_radius_top, inner_radius_top], [center_pixel_bottom, outer_radius_bottom, inner_radius_bottom]]
    else:
        if file_exists:
            [center_pixel, outer_radius, inner_radius] = load_obj_from_pickle(radial_bounds_filename)
        else:
            if len(initial_values) > 0:
                # use initial values and do testing
                [center_pixel, outer_radius, inner_radius] = initial_values
            else:
                [center_pixel, outer_radius, inner_radius] = [None, None, None]

        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Find center and radial boundaries
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        #=======================================================================
        # Find circular boundaries
        win_name_outter = "Outter boundary"
        center_pixel, outer_radius = extract_circular_bound(omni_img, win_name_outter, center_coords=center_pixel, radius=outer_radius)
        win_name_inner = "Inner boundary"
        center_pixel, inner_radius = extract_circular_bound(omni_img, win_name_inner, center_coords=center_pixel, radius=inner_radius)
        if save_to_file:
            save_obj_in_pickle([center_pixel, outer_radius, inner_radius], radial_bounds_filename, locals())

        destroyWindow(win_name_outter)
        destroyWindow(win_name_inner)

        return [center_pixel, outer_radius, inner_radius]


def create_rectangular_mask(img_input, points, preview=False):
    mask = np.zeros(img_input.shape[0:2], dtype=np.uint8)  # Black, single channel mask
    masked_img = np.zeros(img_input.shape)

    cv2.rectangle(img=mask, pt1=points[0], pt2=points[1], color=(255, 255, 255), thickness=cv2.FILLED)
    masked_img = cv2.bitwise_and(img_input, img_input, masked_img, mask=mask)

    if preview:
        resulting_mask_window_name = "Resulting Rectangular Mask"
        cv2.namedWindow(resulting_mask_window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(resulting_mask_window_name, masked_img)
        cv2.waitKey(1)

    return masked_img

def extract_circular_bound(omni_img, win_name="Circle Extraction", center_coords=None, radius=None, ring_fiducials_radii=[]):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    win_handler = PointClicker(win_name)
    if omni_img.ndim == 3:
        omni_img_gray = cv2.cvtColor(omni_img, cv2.COLOR_BGR2GRAY)
    else:
        omni_img_gray = omni_img.copy()


    try_again = True
    if center_coords is not None:
        current_center = center_coords
    else:
        current_center = ((omni_img_gray.shape[1] / 2) - 1, (omni_img_gray.shape[0] / 2) - 1)

    if radius is None:
        current_circle_radius = 500  # @ISH: this default value is arbitrary
    else:
        current_circle_radius = radius

    while try_again:
        omni_img_drawn = omni_img.copy()

        if win_handler.click_counter > 0:
            current_center = win_handler.clicked_points[0]
        if win_handler.shift_mouse_pos:  # Basically a SHIFT and move the mouse
            rp = win_handler.shift_mouse_pos
            current_circle_radius = int(np.sqrt((current_center[0] - rp[0]) ** 2 + (current_center[1] - rp[1]) ** 2))
            win_handler.shift_mouse_pos = None  # Clear

        # draw the circle center
        draw_points(omni_img_drawn, [current_center], color=(255, 0, 0), thickness=3)

        # draw the circle outline
        circle_outline_thickness = 2
        current_center_as_int = (int(current_center[0]), int(current_center[1]))
        cv2.circle(omni_img_drawn, current_center_as_int, current_circle_radius, (0, 0, 255), circle_outline_thickness, 8, 0)

        # Draw ring fiducials:
        fiducials_line_color = (0, 255, 255)  # yellow in BGR
        for fid_radius in ring_fiducials_radii:
            cv2.circle(omni_img_drawn, current_center_as_int, fid_radius, fiducials_line_color, circle_outline_thickness, 8, 0)

        cv2.imshow(win_name, omni_img_drawn)

        ch_pressed_waitkey = cv2.waitKey(10)
        # Vim style motion commands for center adjustment
        if (ch_pressed_waitkey & 255) == ord('i'):  # move up
            current_center = (current_center[0], current_center[1] - 1)
        if (ch_pressed_waitkey & 255) == ord('k'):  # move down
            current_center = (current_center[0], current_center[1] + 1)
        if (ch_pressed_waitkey & 255) == ord('j'):  # move left
            current_center = (current_center[0] - 1, current_center[1])
        if (ch_pressed_waitkey & 255) == ord('l'):  # move right
            current_center = (current_center[0] + 1, current_center[1])
        # Update manual adjustement of center
        win_handler.clicked_points[0] = current_center

        # Resize circle radius
        if (ch_pressed_waitkey & 255) == ord('+') or (ch_pressed_waitkey & 255) == ord('='):
            current_circle_radius += 1
        if (ch_pressed_waitkey & 255) == ord('-'):
            current_circle_radius -= 1

        # Save image
        if (ch_pressed_waitkey & 255) == ord('s'):
            win_handler.save_image(omni_img_drawn, "test_center")

        # Quit
        if (ch_pressed_waitkey == 27) or (ch_pressed_waitkey & 255) == ord('q'):  # Pressing the Escape key breaks the loop
            break

    # TODO: not trusting this yet:
#===============================================================================
#     try:
#         # FIXME: This method for circle detection is too naive and prone to initial estimation error due to my logic of finding the closest point!
#         circles = cv2.HoughCircles(omni_img_gray, method=cv2.cv.CV_HOUGH_GRADIENT, dp=1, minDist=5, minRadius=current_circle_radius - 200, maxRadius=current_circle_radius + 200)
#         # pick closest center point to initial estimate:
#         center_diffs = np.linalg.norm(circles[..., :2] - current_center, axis=-1)
#         closest_circle_match = circles[0, np.argmin(center_diffs)]
#         # Draw detected circle:
#         center = (int(closest_circle_match[0]), int(closest_circle_match[1]))
#         radius = int(closest_circle_match[2])
#         # circle center
#         cv2.circle(omni_img_drawn, center, 3, (0, 255, 0), -1, 8, 0)
#         # circle outline
#         cv2.circle(omni_img_drawn, center, radius, (0, 255, 0), 3, 8, 0)
#     except:
#         print("Not circles detected!")
#
#     cv2.imshow(win_name, omni_img_drawn)
#     cv2.waitKey(0)
#===============================================================================

    return current_center, current_circle_radius

def mask_rect_min_area(base_img, input_image, points, use_all_points=True):
    padding = 2 * np.linalg.norm(points[0, 0] - points[0, 1])  # Use the double of distance between 2 points
    if use_all_points:
        min_area_rect = cv2.minAreaRect(points[..., :2].reshape(-1, 2).astype("int32"))
    else:  # Only use the four courners
        num_rows, num_cols = points.shape[:2]
        corners_4_pattern_indices = np.array([[0, 0], [num_rows - 1, 0], [num_rows - 1, num_cols - 1], [0, num_cols - 1]])
        corners_4_top = points[corners_4_pattern_indices[:, 0], corners_4_pattern_indices[:, 1]].astype("int32")
        min_area_rect = cv2.minAreaRect(corners_4_top)

    # Add padding to box size
    min_area_rect = (min_area_rect[0], (min_area_rect[1][0] + padding, min_area_rect[1][1] + padding), min_area_rect[2])
    box = cv2.boxPoints(min_area_rect).astype("int32")
    # Masking the on the top view:
    mask = np.zeros(input_image.shape[0:2], dtype=np.uint8)  # Black, single channel mask
    # draw rotated rectangle (as filled contours)
    cv2.drawContours(mask, [box], 0, (255, 255, 255), lineType=8, thickness=-1)
    mask_inverted = np.zeros(input_image.shape[0:2], dtype=np.uint8) + 255  # White, single channel mask
    cv2.drawContours(mask_inverted, [box], 0, (0, 0, 0), lineType=8, thickness=-1)
    masked_img = np.zeros_like(input_image)
    masked_img = cv2.bitwise_and(input_image, input_image, dst=masked_img, mask=mask)
    masked_img_base = np.zeros_like(input_image)
    masked_img_base = cv2.bitwise_and(base_img, base_img, masked_img_base, mask=mask_inverted)
    base_img = cv2.bitwise_or(masked_img_base, masked_img)  # Update resulting image as the new base_img

    return base_img

def overlay_all_chessboards(omni_model, calibrator, indices=[], draw_detection=False, visualize=False):

    from omnistereo.camera_models import OmniStereoModel
    if visualize:
        win_name = "Overlayed Chessboard Images"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    base_img = None
    base_img_idx = None
    if isinstance(omni_model, OmniStereoModel):
        omni_calib_list = calibrator.calibration_pairs
        is_stereo = True
    else:
        omni_calib_list = calibrator.omni_monos
        is_stereo = False

    if len(indices) == 0:
        for idx, oc in enumerate(omni_calib_list):
            if oc is not None:
                if oc.found_points:
                    base_img = oc.omni_image.copy()
                    base_img_idx = idx
                    break
        indices = list(range(len(omni_calib_list)))
    else:
        base_img = omni_calib_list[indices[0]].omni_image.copy()
        base_img_idx = indices[0]

    if base_img_idx is not None:
        for idx in indices[base_img_idx:]:  # Start from the next one on
            oc = omni_calib_list[idx]
            if hasattr(oc, "found_points") and oc.found_points:
                if is_stereo:
                    pts_top = oc.mono_top.image_points
                    base_img = mask_rect_min_area(base_img, oc.omni_image, pts_top)
                    pts_bottom = oc.mono_bottom.image_points
                    base_img = mask_rect_min_area(base_img, oc.omni_image, pts_bottom)

                    if draw_detection:
                        det_corner_pixels_top = oc.mono_top.image_points  # Must be a float32 for OpenCV to work!
                        # SIMPLER: draw_points(base_img, det_corner_pixels_top[..., :2].reshape(-1, 2), color=(255, 0, 0), thickness=2)
                        cv2.drawChessboardCorners(base_img, oc.mono_top.pattern_size_applied, det_corner_pixels_top.reshape(-1, 2), oc.found_points)

                        det_corner_pixels_bottom = oc.mono_bottom.image_points  # Must be a float32 for OpenCV to work!
                        # SIMPLER: draw_points(base_img, det_corner_pixels_bottom[..., :2].reshape(-1, 2), color=(0, 0, 255), thickness=2)
                        cv2.drawChessboardCorners(base_img, oc.mono_bottom.pattern_size_applied, det_corner_pixels_bottom.reshape(-1, 2), oc.found_points)
                else:  # mono
                    pts_top = oc.image_points
                    base_img = mask_rect_min_area(base_img, oc.omni_image, pts_top)
                    if draw_detection:
                        det_corner_pixels = oc.image_points  # Must be a float32 for OpenCV to work!
                        # SIMPLER: draw_points(base_img, det_corner_pixels[..., :2].reshape(-1, 2), color=(255, 0, 0), thickness=2)
                        cv2.drawChessboardCorners(base_img, oc.pattern_size_applied, det_corner_pixels.reshape(-1, 2), oc.found_points)

    if visualize:
        cv2.imshow(win_name, base_img)
        cv2.waitKey(1)

    return base_img

# Older way of overlayig by projection from 3D:
#===============================================================================
# def overlay_all_chessboards(omni_model, calibrator, indices=[], draw_detection=False, visualize=False):
#
#     from camera_models import OmniStereoModel
#     if visualize:
#         win_name = "Overlayed Chessboard Images"
#         cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
#
#     base_img = None
#     base_img_idx = None
#     if isinstance(omni_model, OmniStereoModel):
#         omni_calib_list = calibrator.calibration_pairs
#         is_stereo = True
#     else:
#         omni_calib_list = calibrator.omni_monos
#         is_stereo = False
#
#     if len(indices) == 0:
#         for idx, oc in enumerate(omni_calib_list):
#             if oc is not None:
#                 if oc.found_points:
#                     base_img = oc.omni_image.copy()
#                     base_img_idx = idx
#                     break
#         indices = list(range(len(omni_calib_list)))
#     else:
#         base_img = omni_calib_list[indices[0]].omni_image.copy()
#         base_img_idx = indices[0]
#
#     if base_img_idx is not None and calibrator.has_chesboard_pose_info:
#         for idx in indices[base_img_idx:]:  # Start from the next one on
#             oc = omni_calib_list[idx]
#             if hasattr(oc, "found_points") and oc.found_points:
#                 if is_stereo:
#                     T_G_wrt_F_top = calibrator.calib_top.T_G_wrt_F_list[idx]
#                     T_G_wrt_F_bottom = calibrator.calib_bottom.T_G_wrt_F_list[idx]
#                     all_points_wrt_G = oc.mono_top.obj_points_homo
#                     # Recall on [G] points are on the XZ plane (Find margin from the difference of coordinates between consecutive points)
#                     margin_x = 3 * (all_points_wrt_G[0, 1, 0] - all_points_wrt_G[0, 0, 0])
#                     margin_z = 3 * (all_points_wrt_G[1, 0, 2] - all_points_wrt_G[0, 0, 2])
#                     # Create a mask_top offset from the outer corners of the grid
#                     points_wrt_G = np.ndarray((1, 4, 4))
#                     points_wrt_G[0, 0] = all_points_wrt_G[0, 0] + np.array([-margin_x, 0, -margin_z, 0])  # ORIGIN: Lower left corner
#                     points_wrt_G[0, 1] = all_points_wrt_G[-1, 0] + np.array([-margin_x, 0, +margin_z, 0])  # Upper left corner
#                     points_wrt_G[0, 2] = all_points_wrt_G[-1, -1] + np.array([+margin_x, 0, +margin_z, 0])  # Upper right corner
#                     points_wrt_G[0, 3] = all_points_wrt_G[0, -1] + np.array([+margin_x, 0, -margin_z, 0])  # lower right corner
#                     obj_pts_wrt_M_top = np.einsum("ij, mnj->mni", T_G_wrt_F_top, points_wrt_G)
#                     obj_pts_wrt_M_bottom = np.einsum("ij, mnj->mni", T_G_wrt_F_bottom, points_wrt_G)
#                     # Project the 4 margin chessboard corners as mask_top
#                     _, _, projected_corners_top = omni_model.top_model.get_pixel_from_3D_point_homo(obj_pts_wrt_M_top)
#                     _, _, projected_corners_bottom = omni_model.bot_model.get_pixel_from_3D_point_homo(obj_pts_wrt_M_bottom)
#
#                     # Masking the on the top view:
#                     mask_top = np.zeros(oc.omni_image.shape[0:2], dtype=np.uint8)  # Black, single channel mask
#                     cv2.fillConvexPoly(mask_top, projected_corners_top[..., :2].reshape(-1, 2).astype("int32"), color=(255, 255, 255), lineType=8, shift=0)
#                     mask_inverted_top = np.zeros(oc.omni_image.shape[0:2], dtype=np.uint8) + 255  # White, single channel mask
#                     cv2.fillConvexPoly(mask_inverted_top, projected_corners_top[..., :2].reshape(-1, 2).astype("int32"), color=(0, 0, 0), lineType=8, shift=0)
#                     masked_img_top = np.zeros_like(oc.omni_image)
#                     masked_img_top = cv2.bitwise_and(oc.omni_image, oc.omni_image, dst=masked_img_top, mask=mask_top)
#                     masked_img_base_top = np.zeros_like(oc.omni_image)
#                     masked_img_base_top = cv2.bitwise_and(base_img, base_img, masked_img_base_top, mask=mask_inverted_top)
#                     base_img = cv2.bitwise_or(masked_img_base_top, masked_img_top)  # Update resulting image as the new base_img
#
#                     # Masking the on the bottom view:
#                     mask_bottom = np.zeros(oc.omni_image.shape[0:2], dtype=np.uint8)  # Black, single channel mask
#                     cv2.fillConvexPoly(mask_bottom, projected_corners_bottom[..., :2].reshape(-1, 2).astype("int32"), color=(255, 255, 255), lineType=8, shift=0)
#                     mask_inverted_bottom = np.zeros(oc.omni_image.shape[0:2], dtype=np.uint8) + 255  # White, single channel mask
#                     cv2.fillConvexPoly(mask_inverted_bottom, projected_corners_bottom[..., :2].reshape(-1, 2).astype("int32"), color=(0, 0, 0), lineType=8, shift=0)
#                     masked_img_bottom = np.zeros_like(oc.omni_image)
#                     masked_img_bottom = cv2.bitwise_and(oc.omni_image, oc.omni_image, dst=masked_img_bottom, mask=mask_bottom)
#                     masked_img_base_bottom = np.zeros_like(oc.omni_image)
#                     masked_img_base_bottom = cv2.bitwise_and(base_img, base_img, masked_img_base_bottom, mask=mask_inverted_bottom)
#                     base_img = cv2.bitwise_or(masked_img_base_bottom, masked_img_bottom)  # Update resulting image as the new base_img
#                     if draw_detection:
#                         det_corner_pixels_top = oc.mono_top.image_points  # Must be a float32 for OpenCV to work!
#                         # SIMPLE: draw_points(base_img, det_corner_pixels_top[..., :2].reshape(-1, 2), color=(255, 0, 0), thickness=2)
#                         cv2.drawChessboardCorners(base_img, oc.mono_top.pattern_size_applied, det_corner_pixels_top.reshape(-1, 2), oc.found_points)
#
#                         det_corner_pixels_bottom = oc.mono_bottom.image_points  # Must be a float32 for OpenCV to work!
#                         # SIMPLE: draw_points(base_img, det_corner_pixels_bottom[..., :2].reshape(-1, 2), color=(0, 0, 255), thickness=2)
#                         cv2.drawChessboardCorners(base_img, oc.mono_bottom.pattern_size_applied, det_corner_pixels_bottom.reshape(-1, 2), oc.found_points)
#                 else:
#                     T_G_wrt_F = calibrator.T_G_wrt_F_list[idx]
#                     all_points_wrt_G = oc.obj_points_homo
#                     # Recall on [G] points are on the XZ plane (Find margin from the difference of coordinates between consecutive points)
#                     margin_x = 3 * (all_points_wrt_G[0, 1, 0] - all_points_wrt_G[0, 0, 0])
#                     margin_z = 3 * (all_points_wrt_G[1, 0, 2] - all_points_wrt_G[0, 0, 2])
#                     # Create a mask offset from the outer corners of the grid
#                     points_wrt_G = np.ndarray((1, 4, 4))
#                     points_wrt_G[0, 0] = all_points_wrt_G[0, 0] + np.array([-margin_x, 0, -margin_z, 0])  # ORIGIN: Lower left corner
#                     points_wrt_G[0, 1] = all_points_wrt_G[-1, 0] + np.array([-margin_x, 0, +margin_z, 0])  # Upper left corner
#                     points_wrt_G[0, 2] = all_points_wrt_G[-1, -1] + np.array([+margin_x, 0, +margin_z, 0])  # Upper right corner
#                     points_wrt_G[0, 3] = all_points_wrt_G[0, -1] + np.array([+margin_x, 0, -margin_z, 0])  # lower right corner
#                     obj_pts_wrt_M = np.einsum("ij, mnj->mni", T_G_wrt_F, points_wrt_G)
#                     # Project the 4 margin chessboard corners as mask
#                     _, _, projected_corners = omni_model.get_pixel_from_3D_point_homo(obj_pts_wrt_M)
#
#                     # Masking the single view:
#                     mask = np.zeros(oc.omni_image.shape[0:2], dtype=np.uint8)  # Black, single channel mask
#                     cv2.fillConvexPoly(mask, projected_corners[..., :2].reshape(-1, 2).astype("int32"), color=(255, 255, 255), lineType=8, shift=0)
#                     mask_inverted = np.zeros(oc.omni_image.shape[0:2], dtype=np.uint8) + 255  # White, single channel mask
#                     cv2.fillConvexPoly(mask_inverted, projected_corners[..., :2].reshape(-1, 2).astype("int32"), color=(0, 0, 0), lineType=8, shift=0)
#                     masked_img = np.zeros_like(oc.omni_image)
#                     masked_img = cv2.bitwise_and(oc.omni_image, oc.omni_image, dst=masked_img, mask=mask)
#                     masked_img_base = np.zeros_like(oc.omni_image)
#                     masked_img_base = cv2.bitwise_and(base_img, base_img, masked_img_base, mask=mask_inverted)
#                     base_img = cv2.bitwise_or(masked_img_base, masked_img)  # Update resulting image as the new base_img
#                     if draw_detection:
#                         det_corner_pixels = oc.image_points  # Must be a float32 for OpenCV to work!
#                         # SIMPLE: draw_points(base_img, det_corner_pixels[..., :2].reshape(-1, 2), color=(255, 0, 0), thickness=2)
#                         cv2.drawChessboardCorners(base_img, oc.pattern_size_applied, det_corner_pixels.reshape(-1, 2), oc.found_points)
#
#     if visualize:
#         cv2.imshow(win_name, base_img)
#         cv2.waitKey(1)
#
#     return base_img
#===============================================================================
