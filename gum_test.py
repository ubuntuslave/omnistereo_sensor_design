# -*- coding: utf-8 -*-
# gum_test.py

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
Testing the calibration of omnidirectional cameras

* Notes:

* Features:
- The panorama pairs (object instances of Panorama) can be saved as a pickle file
  WISH: Eventually, only the panoramas' LUTs should be saved in order to reduce the size of the pickles.

TODO: Fix LUTs for remapping invalid pixels!!!!
TODO: Extrinsic calibration


@author: carlos
@contact: cjaramillo@gc.cuny.edu
'''

from __future__ import division
from __future__ import print_function

import warnings
import configparser
import numpy as np
import cv2
import os.path as osp
from omnistereo.common_tools import load_obj_from_pickle, save_obj_in_pickle, make_sure_path_exists

def test_direction_angles(pano_top, pano_bot):
    # Testing row from elevation computation:
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    #     elev_test_degrees = -10
    #     elev_test_radians = np.deg2rad(elev_test_degrees)
    #     is_valid_top, row_top, row_top_test = pano_top.get_panoramic_row_from_elevation(elev_test_radians, True)
    #     print("CHECK TOP: find elevation %f degrees: at Row %d  VS Row %d" % (elev_test_degrees, row_top, row_top_test))
    #     is_valid_bot, row_bot, row_bot_test = pano_bot.get_panoramic_row_from_elevation(elev_test_radians, True)
    #     print("CHECK BOTTOM: find elevation %f degrees: at Row %d  VS Row %d" % (elev_test_degrees, row_bot, row_bot_test))
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # Testing column from elevation computation:
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    azimuth_test_degrees = 359
    azimuth_test_radians = np.deg2rad(azimuth_test_degrees)
    col_top, col_top_test = pano_top.get_panorama_col_from_azimuth(azimuth_test_radians, True)
    print("CHECK TOP: find azimuth %f degrees: at Col %d  VS Col %d" % (azimuth_test_degrees, col_top, col_top_test))
    col_bot, col_bot_test = pano_bot.get_panorama_col_from_azimuth(azimuth_test_radians, True)
    print("CHECK BOTTOM: find azimuth %f degrees: at Col %d  VS Col %d" % (azimuth_test_degrees, col_bot, col_bot_test))

def test_triangulation(omnistereo_model):
    # Arbitrary column to test on
    pano_top = omnistereo_model.panorama_top
    pano_bot = omnistereo_model.panorama_bot
    rows_top, cols_top = pano_top.rows, 1  # only 1 column for now (instead of pano_top.cols)
    u_coords_top = np.zeros((rows_top, cols_top, 1))  # + cols_top
    v_coords_top = np.arange(0, rows_top).reshape(rows_top, cols_top, 1)
    m_pixels_top = np.dstack((u_coords_top, v_coords_top)).astype('int32')
    azimuths_top, elevations_top = pano_top.get_direction_angles_from_pixel(m_pixels_top)
    rows_bot, cols_bot = pano_bot.rows, 1  # only 1 column for now (instead of pano_bot.cols)
    u_coords_bot = np.zeros((rows_bot, cols_bot, 1))  # + cols_bot
    v_coords_bot = np.arange(0, rows_bot).reshape(rows_bot, cols_bot, 1)
    m_pixels_bot = np.dstack((u_coords_bot, v_coords_bot)).astype('int32')
    azimuths_bot, elevations_bot = pano_bot.get_direction_angles_from_pixel(m_pixels_bot)
    # Triangulate
    triangulated_points = omnistereo_model.get_triangulated_point_wrt_Oc(elevations_top, elevations_bot, azimuths_top)

    # Near-stereo region bounding vertices
    # CHECKME: something is wrong with the math? Values don't match
    mirror1 = omnistereo_model.top_model
    mirror2 = omnistereo_model.bot_model
    _, _, zero_degree_pixel1 = mirror1.get_pixel_from_direction_angles(0, 0)
    _, _, zero_degree_pixel2 = mirror2.get_pixel_from_direction_angles(0, 0)
    _, _, zero_degree_pixel1_on_panorama = pano_top.get_panorama_pixel_coords_from_direction_angles(0, 0)
    _, _, zero_degree_pixel2_on_panorama = pano_bot.get_panorama_pixel_coords_from_direction_angles(0, 0)

    az1, el1 = pano_top.get_direction_angles_from_pixel_pano(np.array([[[0, 340]]]))
    P_triang = omnistereo_model.get_triangulated_point_wrt_Oc(el1, np.deg2rad(0), 0)

    Pns_high = omnistereo_model.get_triangulated_point_wrt_Oc(mirror1.highest_elevation_angle, mirror2.highest_elevation_angle, 0)
    Pns_mid = omnistereo_model.get_triangulated_point_wrt_Oc(mirror1.lowest_elevation_angle, mirror2.highest_elevation_angle, 0)
    Pns_low = omnistereo_model.get_triangulated_point_wrt_Oc(mirror1.lowest_elevation_angle, mirror2.lowest_elevation_angle, 0)
    return triangulated_points

def run_omnistereo_tests(omnistereo_model):

#     test_direction_angles(omnistereo_model.panorama_top, omnistereo_model.panorama_bot)
    test_triangulation(omnistereo_model)

def run_single_calibration(omni_model, calibrator, is_synthetic, chessboard_params_filename, img_indices=[], eval_indices=[], load_calibrated_model_from_file=False, calibrator_filename_prefix="single_calibrator", visualize=True, evaluate_against_truth=False, do_radial_bounds_refinement=True):
    calibrator_filename = calibrator_filename_prefix + "_" + omni_model.mirror_name + ".pkl"
    if visualize:
        app = None

    from omnistereo import sensor_evaluation

    if load_calibrated_model_from_file:
        calibrator = load_obj_from_pickle(calibrator_filename)
    else:
        app = calibrator.calibrate_mono(omni_model, chessboard_indices=img_indices, visualize=visualize, only_extrinsics=False, only_grids=False, only_C_wrt_M_tz=False, only_translation_params=False, normalize=False, return_jacobian=True, do_radial_bounds_refinement=do_radial_bounds_refinement, do_single_trial=True)

    if do_radial_bounds_refinement:
        # Refine Radial Bounds
        from omnistereo.common_cv import refine_radial_bounds_mono
        center_pixel, outer_radius, inner_radius = refine_radial_bounds_mono(omni_img=calibrator.estimated_omni_model.current_omni_img, initial_values=[calibrator.estimated_omni_model.precalib_params.center_point, calibrator.estimated_omni_model.outer_img_radius, calibrator.estimated_omni_model.inner_img_radius])
        # Set new radii values
        calibrator.estimated_omni_model.set_radial_limits_in_pixels_mono(inner_img_radius=inner_radius, outer_img_radius=outer_radius, center_point=center_pixel)

    if evaluate_against_truth:  # and not is_synthetic:
        # TODO: (ONLY for REAL experiments!!!!) Reset previously known params to account for the new estimated extrinsic parameters tz of the models and the new relative pose of the ground-truth grids
        calibrator.set_true_chessboard_pose(calibrator.estimated_omni_model, chessboard_params_filename, input_units="cm", chessboard_indices=img_indices, show_corners=False)  # RESET information of calibrator data

    if calibrator.estimated_omni_model.mirror_name == "top":
        radial_height = np.linalg.norm(calibrator.estimated_omni_model.lowest_img_point - calibrator.estimated_omni_model.precalib_params.center_point)
    else:
        radial_height = np.linalg.norm(calibrator.estimated_omni_model.highest_img_point - calibrator.estimated_omni_model.precalib_params.center_point)

    pano_width = np.pi * radial_height
    omni_img = calibrator.estimated_omni_model.current_omni_img
    calibrator.estimated_omni_model.set_omni_image(omni_img, pano_width_in_pixels=pano_width, idx=-1, generate_panorama=True, view=True)
    save_obj_in_pickle(calibrator, calibrator_filename, locals())

    # CHECKME URGENT: Panoramas on evaluation seem way off even for detected points. I think the tz offsets (adjustments are ignored) when doing the eval wrt mirror focus
    print("Pose of [C] wrt [M]: Translation on z-axis: %f [%s]   VS  theoretical: %f [%s]" % (calibrator.estimated_omni_model.F[2, 0], calibrator.estimated_omni_model.units, calibrator.original_model.F[2, 0], calibrator.original_model.units))
    print("Position of Point Cp (center of projection) wrt [M]: <xi_x, xi_y, xi_z> = ", calibrator.estimated_omni_model.Cp_wrt_M, " VS  theoretical =", calibrator.original_model.Cp_wrt_M)
    print("Image center (pixel coords):", calibrator.estimated_omni_model.precalib_params.center_point, " VS  original =", calibrator.original_model.precalib_params.center_point)
    print("Radial distortion params: <k1, k2> = <%f, %f>" % (calibrator.estimated_omni_model.precalib_params.k1, calibrator.estimated_omni_model.precalib_params.k2))
    calibrator.estimated_omni_model.precalib_params.print_params(header_message="All top GUM Parameters:")
    print(30 * "-+")


    if evaluate_against_truth:
        # Evaluate Pixel Projection
        sensor_evaluation.forward_projection_eval_mono(camera_model=calibrator.estimated_omni_model, calibrator=calibrator, chessboard_indices=eval_indices, wrt_mirror_focus=True, visualize=True, visualize_panorama=True, show_detected=True, overlay_all_omni=True, base_img=None, proj_pt_RGB_color=(0, 255, 0), verbose=True)

    sensor_evaluation.evaluate_3D_error_from_grid_poses(calibrator, chessboard_indices=img_indices, eval_indices=eval_indices, app=app, visualize=visualize)

    return calibrator.estimated_omni_model, calibrator

def run_omnistereo_calibration(gums_uncalibrated, omnistereo_calibrator, is_synthetic, model_version, chessboard_params_filename, img_indices=[], eval_indices=[], use_gums_coupled_calibration_method=True, load_calibrated_model_from_file=False, calibrator_filename_prefix="gums_calibrator", visualize=True, evaluate_against_truth=False, do_radial_bounds_refinement=True, do_single_trial=False):
    # TODO: Refine initialization using HISTOGRAMS!
    test_with_jacobian = True  # SET!!!!

    from omnistereo import sensor_evaluation
    omni_img = gums_uncalibrated.current_omni_img


    if use_gums_coupled_calibration_method:
        calibrator_filename = calibrator_filename_prefix + ".pkl"
    else:
        calibrator_filename = calibrator_filename_prefix + "-decoupled.pkl"

    if visualize:
        app = None

    if load_calibrated_model_from_file:
        omnistereo_calibrator = load_obj_from_pickle(calibrator_filename)
        gums_calibrated = omnistereo_calibrator.estimated_omnistereo_model
    else:
        if use_gums_coupled_calibration_method:
            # THE HOLY GRAIL: main GUMS contribution
            # Do the stereo calibration as STEREO (taking advantage of double views and same transform of grid frame [G] wrt [C])
            app = omnistereo_calibrator.calibrate_omnistereo(gums_uncalibrated, chessboard_indices=img_indices, visualize=visualize, return_jacobian=test_with_jacobian, only_extrinsics=False, init_grid_poses_from_both_views=True, do_single_trial=do_single_trial)
        else:
            from omnistereo import panorama
            from omnistereo import gum

            # Compare to the "De-coupled" approach where the intrinsic model parameters are calibrated separately:
            app = omnistereo_calibrator.calib_top.calibrate_mono(gums_uncalibrated.top_model, chessboard_indices=img_indices, visualize=True, only_extrinsics=False, only_grids=False, only_C_wrt_M_tz=False, only_translation_params=False, normalize=False, return_jacobian=True, do_single_trial=do_single_trial)
            gum_top_calibrated = omnistereo_calibrator.calib_top.estimated_omni_model
            omnistereo_calibrator.calib_top.set_true_chessboard_pose(gum_top_calibrated, chessboard_params_filename, input_units="cm", chessboard_indices=img_indices)  # RESET information of calibrator data
            pano_width_calib_coupled = np.pi * np.linalg.norm(gum_top_calibrated.lowest_img_point - gum_top_calibrated.precalib_params.center_point)
            gum_top_calibrated.panorama = panorama.Panorama(gum_top_calibrated, width=pano_width_calib_coupled)
            if evaluate_against_truth:
                sensor_evaluation.forward_projection_eval_mono(camera_model=gum_top_calibrated, calibrator=omnistereo_calibrator.calib_top, chessboard_indices=img_indices, wrt_mirror_focus=True, visualize=True, visualize_panorama=True, show_detected=True, overlay_all_omni=True, base_img=None)

            app = omnistereo_calibrator.calib_bottom.calibrate_mono(gums_uncalibrated.bot_model, chessboard_indices=img_indices, visualize=False, only_extrinsics=False, only_grids=False, only_C_wrt_M_tz=False, only_translation_params=False, normalize=False, return_jacobian=True, do_single_trial=do_single_trial)
            gum_bot_calibrated = omnistereo_calibrator.calib_bottom.estimated_omni_model
            omnistereo_calibrator.calib_bottom.set_true_chessboard_pose(gum_bot_calibrated, chessboard_params_filename, input_units="cm", chessboard_indices=img_indices)  # RESET information of calibrator data
            gum_bot_calibrated.panorama = panorama.Panorama(gum_bot_calibrated, width=pano_width_calib_coupled)
            if evaluate_against_truth:
                sensor_evaluation.forward_projection_eval_mono(camera_model=gum_bot_calibrated, calibrator=omnistereo_calibrator.calib_bottom, chessboard_indices=img_indices, wrt_mirror_focus=True, visualize=False, visualize_panorama=True, show_detected=True, overlay_all_omni=True, base_img=None)

            gums_decoupled = gum.GUMStereo(gum_top_calibrated, gum_bot_calibrated)  # set top and bottom models new parameters separately
            pano_width_decoupled = np.pi * np.linalg.norm(gums_decoupled.bot_model.lowest_img_point - gums_decoupled.bot_model.precalib_params.center_point)
            gums_decoupled.set_current_omni_image(omni_img, pano_width_in_pixels=pano_width_decoupled, generate_panoramas=True, idx=-1, view=True)

            gums_decoupled.init_theoretical_model(gums_uncalibrated.theoretical_model)  # ONLY done for 3D visualization purposes of model
            # Perform de-coupled Omnistereo Calibration:
            app = omnistereo_calibrator.calibrate_omnistereo(gums_decoupled, chessboard_indices=img_indices, visualize=visualize, return_jacobian=True, only_extrinsics=True, init_grid_poses_from_both_views=True, do_single_trial=do_single_trial)

        gums_calibrated = omnistereo_calibrator.estimated_omnistereo_model
        pano_width = np.pi * np.linalg.norm(gums_calibrated.bot_model.lowest_img_point - gums_calibrated.bot_model.precalib_params.center_point)
        omnistereo_calibrator.estimated_omnistereo_model.set_current_omni_image(omni_img, pano_width_in_pixels=pano_width, generate_panoramas=True, idx=-1, view=True)
        save_obj_in_pickle(omnistereo_calibrator, calibrator_filename, locals())

    # EVALUATE forward projection error of theoretical model ONLY when given Ground Truth poses
    theor_proj_img = None
    if evaluate_against_truth and evaluate_against_truth:
        print("THEORETICAL EVALUATION:")
        theor_proj_img = sensor_evaluation.forward_projection_eval_stereo(camera_model=gums_uncalibrated.theoretical_model, calibrator=omnistereo_calibrator, chessboard_indices=img_indices, wrt_mirror_focus=False, use_ground_truth_pose=omnistereo_calibrator.has_chesboard_pose_info, show_detected=True, visualize=True)
        sensor_evaluation.sparse_triangulation_eval(camera_model=gums_uncalibrated.theoretical_model, calibrator=omnistereo_calibrator, chessboard_indices=img_indices, use_midpoint_triangulation=True, manual_pixel_bias=0.0, debug_mode=False)
        # Show Initial Radial Bounds, as well
        theor_proj_img = gums_uncalibrated.draw_radial_bounds_stereo(omni_img=theor_proj_img, is_reference=True, view=True)

    # CHECKME URGENT: Panoramas on evaluation seem way off even for detected points.
    # I think the tz offsets (adjustments are ignored) when doing the eval wrt mirror focus
    if do_radial_bounds_refinement:
        # Refine Radial Bounds
        from omnistereo.common_cv import refine_radial_bounds
        (outer_radius_top, inner_radius_top), (outer_radius_bottom, inner_radius_bottom) = refine_radial_bounds(omni_img=omnistereo_calibrator.estimated_omnistereo_model.current_omni_img, top_values=[omnistereo_calibrator.estimated_omnistereo_model.top_model.precalib_params.center_point, omnistereo_calibrator.estimated_omnistereo_model.top_model.outer_img_radius, omnistereo_calibrator.estimated_omnistereo_model.top_model.inner_img_radius], bottom_values=[omnistereo_calibrator.estimated_omnistereo_model.bot_model.precalib_params.center_point, omnistereo_calibrator.estimated_omnistereo_model.bot_model.outer_img_radius, omnistereo_calibrator.estimated_omnistereo_model.bot_model.inner_img_radius])
        # Set new radii values
        omnistereo_calibrator.estimated_omnistereo_model.set_params(inner_radius_bottom=inner_radius_bottom, outer_radius_bottom=outer_radius_bottom, inner_radius_top=inner_radius_top, outer_radius_top=outer_radius_top)
        gums_calibrated = omnistereo_calibrator.estimated_omnistereo_model
        # Set the flag for mask regeneration after this bounds change.
        omnistereo_calibrator.estimated_omnistereo_model.construct_new_mask = True
        # Again, compute new panoramas
        pano_width = np.pi * np.linalg.norm(gums_calibrated.bot_model.lowest_img_point - gums_calibrated.bot_model.precalib_params.center_point)
        omnistereo_calibrator.estimated_omnistereo_model.set_current_omni_image(omni_img, pano_width_in_pixels=pano_width, generate_panoramas=True, apply_pano_mask=True, idx=-1, view=True)
        # And save in pickle
        save_obj_in_pickle(omnistereo_calibrator, calibrator_filename, locals())

    gums_calibrated = omnistereo_calibrator.estimated_omnistereo_model
    print("TOP:")
    print("Pose of [C] wrt [M]: Translation on z-axis: %f [%s]   VS  theoretical: %f [%s]" % (gums_calibrated.top_model.F[2, 0], gums_calibrated.top_model.units, omnistereo_calibrator.original_omnistereo_model.top_model.F[2, 0], gums_calibrated.top_model.units))
    print("Position of Point Cp (center of projection) wrt [M]: <xi_x, xi_y, xi_z> = ", gums_calibrated.top_model.Cp_wrt_M, " VS  theoretical =", omnistereo_calibrator.original_omnistereo_model.top_model.Cp_wrt_M)
    print("Image center (pixel coords):", gums_calibrated.top_model.precalib_params.center_point, " VS  original =", omnistereo_calibrator.original_omnistereo_model.top_model.precalib_params.center_point)
    print("Radial distortion params: <k1, k2> = <%f, %f>" % (gums_calibrated.top_model.precalib_params.k1, gums_calibrated.top_model.precalib_params.k2))
    gums_calibrated.top_model.precalib_params.print_params(header_message="All top GUM Parameters:")
    print(30 * "-+")
    print("BOTTOM:")
    print("Pose of [C] wrt [M]: Translation on z-axis: %f [%s]   VS  theoretical: %f [%s]" % (gums_calibrated.bot_model.F[2, 0], gums_calibrated.bot_model.units, omnistereo_calibrator.original_omnistereo_model.bot_model.F[2, 0], gums_calibrated.bot_model.units))
    print("Position of Point Cp (center of projection) wrt [M]: <xi_x, xi_y, xi_z> = ", gums_calibrated.bot_model.Cp_wrt_M, " VS  theoretical =", omnistereo_calibrator.original_omnistereo_model.bot_model.Cp_wrt_M)
    print("Image center (pixel coords):", gums_calibrated.bot_model.precalib_params.center_point, " VS  original =", omnistereo_calibrator.original_omnistereo_model.bot_model.precalib_params.center_point)
    print("Radial distortion params: <k1, k2> = <%f, %f>" % (gums_calibrated.bot_model.precalib_params.k1, gums_calibrated.bot_model.precalib_params.k2))
    gums_calibrated.bot_model.precalib_params.print_params(header_message="All bottom GUM Parameters:")

    if evaluate_against_truth and not is_synthetic:
        # (ONLY for REAL experiments!!!!) Reset previously known params to account for the new estimated extrinsic parameters tz of the models and the new relative pose of the ground-truth grids
        from omnistereo.vicon_utils import optimize_cam_pos_wrt_C_theoretical
        x_axis_points = None  # We must extract them for the first time
        # NOTE: passing the theoretical model:
        T_Ggt_wrt_C_array, T_C_wrt_W_refinement, x_axis_points = optimize_cam_pos_wrt_C_theoretical(omnistereo_model=gums_calibrated.theoretical_model, calibrator=omnistereo_calibrator, eval_indices=eval_indices, model_version=model_version, chessboard_params_filename=chessboard_params_filename, x_axis_points=x_axis_points)
        print(20 * "*", "EVALUATION after Calibration (Against GT Rig Poses resolved from THEORETICAL MODEL):", 20 * "*")
        #=======================================================================
        # from vicon_utils import resolve_poses_wrt_C_theoretical
        # # NOTE: here we enforce the use of the theoretical OmniStereo model, so we can compare if our rig pose ground truth is ok (or it has some critial error)
        # T_Ggt_wrt_C_array, x_axis_points = resolve_poses_wrt_C_theoretical(gums_calibrated.theoretical_model, model_version, chessboard_params_filename, x_axis_points=x_axis_points)
        #=======================================================================
        omnistereo_calibrator.T_G_wrt_C_list_for_calibration = T_Ggt_wrt_C_array[img_indices]
#         omnistereo_calibrator.set_true_chessboard_pose(gums_calibrated, chessboard_params_filename, input_units="cm", chessboard_indices=img_indices, show_corners=False)  # RESET information of calibrator data
        omnistereo_calibrator.set_true_chessboard_pose(gums_calibrated, T_Ggt_wrt_C_array, input_units="mm", chessboard_indices=img_indices, show_corners=False)  # RESET information of calibrator data
        # EVALUATE forward projection error (By projecting the Ground Truth data of the chessboard poses)
        sensor_evaluation.forward_projection_eval_stereo(camera_model=gums_calibrated, calibrator=omnistereo_calibrator, chessboard_indices=eval_indices, wrt_mirror_focus=True, use_ground_truth_pose=True, visualize=True, visualize_panorama=True, show_detected=True, overlay_all_omni=True, proj_pt_RGB_color=(0, 255, 0), verbose=True)
        sensor_evaluation.sparse_triangulation_eval(camera_model=gums_calibrated, calibrator=omnistereo_calibrator, chessboard_indices=eval_indices, use_midpoint_triangulation=True, manual_pixel_bias=0.0, debug_mode=False)
        sensor_evaluation.evaluate_3D_error_from_grid_poses(omnistereo_calibrator, chessboard_indices=img_indices, eval_indices=eval_indices, visualize=visualize, vis_only_evaluated_grids=True)
        # Resolve the adjusted ground-truth captured by the VICON system
        # Set the new estimated extrinsic parameters tz of the models and the new relative pose of the ground-truth grids
        # NOTE: passing the CALIBRATED GUMS:
        T_Ggt_wrt_C_array, T_C_wrt_W_refinement, x_axis_points = optimize_cam_pos_wrt_C_theoretical(omnistereo_model=gums_calibrated, calibrator=omnistereo_calibrator, eval_indices=eval_indices, model_version=model_version, chessboard_params_filename=chessboard_params_filename, x_axis_points=x_axis_points)
#         T_Ggt_wrt_C_array, x_axis_points = resolve_poses_wrt_C_theoretical(gums_calibrated, model_version, chessboard_params_filename, x_axis_points=x_axis_points)
        omnistereo_calibrator.T_G_wrt_C_list_for_calibration = T_Ggt_wrt_C_array[img_indices]
#         omnistereo_calibrator.set_true_chessboard_pose(gums_calibrated, chessboard_params_filename, input_units="cm", chessboard_indices=img_indices, show_corners=False)  # RESET information of calibrator data
        omnistereo_calibrator.set_true_chessboard_pose(gums_calibrated, T_Ggt_wrt_C_array, input_units="mm", chessboard_indices=img_indices, show_corners=False)  # RESET information of calibrator data
        save_obj_in_pickle(omnistereo_calibrator, calibrator_filename, locals())
    if evaluate_against_truth:
        # Perform pixel-level evaluation
        print(20 * "*", "EVALUATION after GT Pose Adjustment with Calibrated GUMS:", 20 * "*")
        sensor_evaluation.forward_projection_eval_stereo(camera_model=gums_calibrated, calibrator=omnistereo_calibrator, chessboard_indices=eval_indices, wrt_mirror_focus=True, use_ground_truth_pose=True, visualize=True, visualize_panorama=True, show_detected=True, overlay_all_omni=True, proj_pt_RGB_color=(0, 255, 0), verbose=True)
        sensor_evaluation.sparse_triangulation_eval(camera_model=gums_calibrated, calibrator=omnistereo_calibrator, chessboard_indices=eval_indices, use_midpoint_triangulation=True, manual_pixel_bias=0.0, debug_mode=False)

    print(20 * "*", "EVALUATION without GT for the Calibrated GUMS:", 20 * "*")
    # Finally, perform pixel-level evaluation (without Ground Truth pose info)
    # NOTE: here we use all pixels (not just those chosen for evaluation)

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # TEST: weights computation from reprojection errors:
    # weights = sensor_evaluation.get_confidence_weights_from_pixel_error_stereo(camera_model=gums_calibrated, calibrator=omnistereo_calibrator, chessboard_indices=img_indices)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    proj_pixels_img = sensor_evaluation.forward_projection_eval_stereo(camera_model=gums_calibrated, calibrator=omnistereo_calibrator, chessboard_indices=img_indices, wrt_mirror_focus=True, use_ground_truth_pose=False, visualize=True, visualize_panorama=True, show_detected=False, omni_img=theor_proj_img, overlay_all_omni=True, proj_pt_RGB_color=(255, 0, 0), verbose=True)
    # Show RESULTING Radial Bounds
    gums_calibrated.draw_radial_bounds_stereo(omni_img=proj_pixels_img, is_reference=False, view=True)

    print(20 * "*", "3D EVALUATION for the Calibrated GUMS:", 20 * "*")
    sensor_evaluation.evaluate_3D_error_from_grid_poses(omnistereo_calibrator, chessboard_indices=img_indices, eval_indices=eval_indices, visualize=visualize, vis_only_evaluated_grids=False)

    return omnistereo_calibrator


def init_OmniStereo(omni_img, top_gum_filename, bottom_gum_filename, radial_bounds_filename, use_perfect_model, theoretical_params_filename, model_version, is_synthetic):
    from omnistereo import gum
    #===========================================================================
    # config = configparser.ConfigParser()
    # config.read("options.ini")
    # top_width = config.getint("TopMirror", "width")
    # print("Test width", top_width)
    # prefix_dir = config.get("General", "input_dir")
    # print(prefix_dir)
    # top_gum_filename = prefix_dir + config.get("TopMirror", "gum_filename")
    # print(top_gum_filename)
    # bottom_gum_filename = prefix_dir + config.get("BottomMirror", "gum_filename")
    # print(bottom_gum_filename)
    #===========================================================================

    # NOTE: The convention of digital images sizes (width, height)
    image_size = np.array([omni_img.shape[1], omni_img.shape[0]])
    # sensor_size = np.array([4, 3])  # (width, height) in [mm]
#     image_size = np.array([752, 480])  # BlueFox-MLC = (752x480) | PointGrey Chameleon = (1280x960)

    # Radial Pixel boundaries
    # Refine manually
    radial_initial_values = []
    from omnistereo.common_cv import find_center_and_radial_bounds
    file_exists = osp.isfile(radial_bounds_filename)
    if not file_exists:
        radial_initial_values = [[(image_size / 2.0) - 1, None, None], [(image_size / 2.0) - 1, None, None]]  # At least initialize the center pixel from MATLAB's calibration file

    [[center_pixel_top, outer_radius_top, inner_radius_top], [center_pixel_bottom, outer_radius_bottom, inner_radius_bottom]] = find_center_and_radial_bounds(omni_img, initial_values=radial_initial_values, radial_bounds_filename=radial_bounds_filename, save_to_file=True)


    # Changing boundaries manually
    # gums.top_model.set_occlussion_boundaries(321, 580)
    # gums.bot_model.set_occlussion_boundaries(75, 320)

    # THEORETICAL VALUES:
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    if use_perfect_model:
        from omnistereo.cata_hyper_model import PinholeCamera, HyperCata, HyperCataStereo
        from omnistereo.common_tools import get_theoretical_params_from_file
        c1, c2, k1, k2, d, r_sys, r_reflex, r_cam = get_theoretical_params_from_file(theoretical_params_filename, file_units="cm")
        # Points as homogeneous column vectors:
        Oc = np.array([0, 0, 0, 1]).reshape(4, 1)  # also F1'
        F1 = np.array([0, 0, c1, 1]).reshape(4, 1)  # F1
        F2 = np.array([0, 0, d - c2, 1]).reshape(4, 1)  # F2
        F2v = np.array([0, 0, d, 1]).reshape(4, 1)  # F2' (virtual camera, also)
        mirror1 = HyperCata(1, F1, Oc, c1, k1, d)
        mirror2 = HyperCata(2, F2, F2v, c2, k2, d)

        if is_synthetic:
            focal_length = 1  # Camera Focal length: 1 mm (for synthetic images)

            if model_version == "new":
                cam_hor_FOV = 38  # Horizontal FOV of "synthetic" perspective camera
            elif model_version == "old":
                cam_hor_FOV = 45  # Horizontal FOV of "synthetic" perspective camera
                                    # With our 4:3 aspect ratio, the vertical FOV of the camera is about 34.5 degrees
            # pixel_size = np.array([6, 6]) * (10 ** -3)  # in [mm]: BlueFox-MLC = 6x6 um
            # Only for synthetic images vvvvvvvvvvvv
            img_cols = image_size[0]  # the width
            synthetic_pixel_size_horizontal = 2 * focal_length * np.tan(np.deg2rad(cam_hor_FOV) / 2.0) / img_cols
            # square pixels: we get [ 0.00064721  0.00064721] mm ~ 6x6 um for the 1280x960 POV-Ray image with 45deg FOV camera
            pixel_size = np.array([synthetic_pixel_size_horizontal, synthetic_pixel_size_horizontal])  # in [mm]: Simulated parameters for camera (in POV-Ray)
        else:  # For real cameras
            # For Logitech HD Pro Webcam C910:  Sensor size: 1/2.5" or  5.270  [mm] x 3.960[mm] -> diagonal = 6.592 [mm]
            # TODO: theoretical values for Pt. Grey cameras
            aperture_width = 5.270  # [mm]
            aperture_height = 3.960  # [mm]
            sensor_size = np.array([aperture_width, aperture_height])  # (width, height) in [mm]
            pixel_size = sensor_size / image_size
            z_at_r_sys_top = mirror1.get_z_hyperbola(x=r_sys, y=0)
            f_u = outer_radius_top * (z_at_r_sys_top / r_sys)  # Camera Focal length in pixels (NOT [mm])
            # Infer focal length and pixel size from image for REAL camera!
            focal_length = f_u * pixel_size[0]

        cam_mirror1 = PinholeCamera(mirror1, image_size_pixels=image_size, focal_length=focal_length, pixel_size=pixel_size, custom_center=center_pixel_top)  # Sets mirror1 as parent for this cam_mirror1
        mirror1.precalib_params = cam_mirror1
        mirror1.set_radial_limits(r_reflex, r_sys)
        mirror1.set_radial_limits_in_pixels_mono(inner_img_radius=inner_radius_top, outer_img_radius=outer_radius_top)

        cam_mirror2 = PinholeCamera(mirror2, image_size_pixels=image_size, focal_length=focal_length, pixel_size=pixel_size, custom_center=center_pixel_bottom)  # Sets mirror2 as parent for this cam_mirror2
        mirror2.precalib_params = cam_mirror2
        mirror2.set_radial_limits(r_cam, r_sys)
        mirror2.set_radial_limits_in_pixels_mono(inner_img_radius=inner_radius_bottom, outer_img_radius=outer_radius_bottom)
        theoretical_omni_stereo = HyperCataStereo(mirror1, mirror2)

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # ATTENTION: to the axis hack for (I don't like it, because it seems inconsistent!)
    top_GUM = gum.GUM(top_gum_filename, z_axis=-1.0, use_theoretical_xi_and_gamma=use_perfect_model, center_uv_point=center_pixel_top)
    bot_GUM = gum.GUM(bottom_gum_filename, z_axis=1.0, use_theoretical_xi_and_gamma=use_perfect_model, center_uv_point=center_pixel_bottom)

    gums = gum.GUMStereo(top_GUM, bot_GUM)

    if use_perfect_model:
        # Initialization with theoretical values
        gums.init_theoretical_model(theoretical_omni_stereo)

    # Finally, set (reset) params for the camera model with the radial bounds e (Must be done after theoretical parameters setup).
    # Reset Params of model:
    gums.set_params(inner_radius_bottom=inner_radius_bottom, outer_radius_bottom=outer_radius_bottom, inner_radius_top=inner_radius_top, outer_radius_top=outer_radius_top, center_point_top=center_pixel_top, center_point_bottom=center_pixel_bottom)
    gums.baseline = gums.get_baseline()
    gums.print_omnistereo_info()

    return gums

def overlay_pair(src1, src2, alpha):
    beta = (1.0 - alpha)
    img_dst = cv2.addWeighted(src1, alpha, src2, beta, gamma=0)

    win_name = "Linear Blend"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img_dst)
    cv2.waitKey(1)
    return img_dst


def main_test():
    is_synthetic = False
    model_version = "new"  # "old"  # <<<< SETME: Can be "old" for older (Laura's) model, "new" for the new model parameters (mine)
    experiment_name = "CVPR"  # "simple", "VICON", "CVPR", "with_misalignment-4", etc.  # <<<<<- SET For example, "VICON" uses ground truth data, otherwise use "simple"

    # vvvvvvvvvvvvvvvvvvvvvvv OPTIONS vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    load_omnistereo_from_file = True  # BECAREFUL: it will disregard whatever is set next for "use_perfect_model", so don't load from file if not sure what was set as
    use_perfect_model = True  # SET to True for using perfect "theoretical xi and gammas" model
#     resolve_pose_data_wrt_C = False  # <<< We must resolve the grid poses gathered from a VICON mo-cap system and set it with respect to the camera frame of the GUMS model
    view_panos = True

    do_calibration = True  # <<< SET to True to do Corner Extraction and calibration
    do_single_trial = True
    load_calibrator_from_file = True
    load_calibrated_model_from_file = True  # NOTE: converged successfully for "new" and "old" mirrors. Not sure why convergence fails sometimes, but it's not my code!
    refine_radial_bounds = not load_calibrated_model_from_file  # Allows to perform a manual circle fitting for the radial bounds of the calibrated model.
    use_gums_coupled_calibration_method = True  # <<< SET: False for Decoupled calibration. True: for proposed method (New/Coupled).
    evaluate = False

    get_pointclouds = True
    compute_new_3D_points = True
    dense_cloud = False
    dense_manual_3D_point_selection = False
    show_3D_reference_cyl = False
    tune_live = False  # <<< SET: True for contiguous frames (movies). When False: the tuning is attempted (until Esc is pressed)
    load_stereo_tuner_from_pickle = True
    save_3D_points = True
    save_pcl = True
    save_sparse_features = False
    load_sparse_features_from_file = False

    data_root = "data"  # The root folder for all data

    if is_synthetic:
        model_type = "synthetic"
    else:
        model_type = "real"

    data_path = osp.join(data_root, model_type, model_version, experiment_name)
    calibration_data_path = osp.join(data_path, "calibration")
    make_sure_path_exists(calibration_data_path)  # This creates the path if necessary

    calib_img_file_name_prefix = osp.join(calibration_data_path, "chessboard-")
    calib_img_filename_template = calib_img_file_name_prefix + "*.png"
    radial_bounds_filename = osp.join(data_path, "radial_bounds.pkl")
    omnistereo_filename = osp.join(calibration_data_path, "omnistereo-gums.pkl")
    stereo_calibrator_filename = osp.join(calibration_data_path, "gums_calibrator.pkl")
    calibrated_model_filename_prefix = osp.join(calibration_data_path, "gums_calibrator-calibrated")

    # Scene:
    scene_name = "scene"
    scene_path = osp.join(data_path, scene_name)  # Pose estimation experiment: Translation on x only by 0, 25 cm and 75 cm (wrt init)
    scene_img_filename_template = osp.join(scene_path, "office-*.png")  # With PUBLISHED parameters


    top_gum_filename = ""  # calibration_data_path + "/model_parameters_new-top-SYNT_HYPERBOLIC-1600x1200.bin"
    bottom_gum_filename = ""  # calibration_data_path + "/model_parameters_new-bottom-SYNT_HYPERBOLIC-1600x1200.bin"
    theoretical_params_filename = osp.join(data_root, "parameters-%s.txt" % (model_version))

    if do_calibration:
        chessboard_params_filename = osp.join(calibration_data_path, "calib_pattern_poses.csv")
        if experiment_name == "VICON" or experiment_name == "CVPR" or is_synthetic:
            do_evaluation = evaluate and False  # FIXME: just because we are not using this GT data from 2016
        else:
            do_evaluation = False

    if get_pointclouds:
        points_3D_filename_template = "3d_points-*.pkl"
        if dense_cloud:
            points_3D_path = osp.join(scene_path, "cloud_dense")
        else:
            points_3D_path = osp.join(scene_path, "cloud_sparse")
        make_sure_path_exists(points_3D_path)
        stereo_tuner_filename = osp.join(scene_path, "stereo_tuner.pkl")
        features_detected_filename_template = "sparse_correspondences-*.pkl"

    img_indices = []
    eval_indices = None  # Use None to evaluate all
    img_index = 0  # <<<<<------ Choosing an arbitrary image to work with out of the set
    if not is_synthetic:
        if model_version == "old":
            if experiment_name == "VICON":
                img_indices = [0, 1, 2, 4, 5]  # No data in [0,6]
    #             img_indices = [3]  # Used for single grid visualizatio (for CVPR paper)
                eval_indices = [2, 4]  # Omitting: [1 , 4, 5] bad VICON pose
            else:
                img_indices = [0, 2, 3, 4, 5, 6, 7]
                eval_indices = [0, 2, 3, 4, 5, 6, 7]  # Use None to evaluate all
            img_index = 4  # <<<<<------ Choosing an arbitrary image to work with out of the set

    omni_img_filename = calib_img_filename_template.replace("*", str(img_index), 1)
    omni_img = cv2.imread(omni_img_filename, 1)
    if load_omnistereo_from_file:
        gums = load_obj_from_pickle(omnistereo_filename)
    else:
        gums = init_OmniStereo(omni_img, top_gum_filename, bottom_gum_filename, radial_bounds_filename, use_perfect_model, theoretical_params_filename, model_version, is_synthetic=is_synthetic)

        pano_width = np.pi * np.linalg.norm(gums.bot_model.lowest_img_point - gums.bot_model.precalib_params.center_point)
        #=======================================================================
        # from common_cv import get_images, get_masked_images_mono
        # mirror_images = get_images(calib_img_filename_template, indices_list=img_indices, show_images=False)
        # masked_images = get_masked_images_mono(mirror_images, gums.top_model, img_indices, show_images=False, color_RGB=(0, 180, 0))
        # cv2.imshow("????", masked_images[0])
        #=======================================================================

        gums.set_current_omni_image(omni_img, pano_width_in_pixels=pano_width, generate_panoramas=True, idx=img_index, view=True)
#         gums.set_current_omni_image(masked_images[img_index], pano_width_in_pixels=pano_width, generate_panoramas=True, idx=img_index, view=False)

        #=======================================================================
        # pano_masked_top = gums.top_model.panorama.get_panoramic_image(masked_images[img_index])
        # pano_masked_bot = gums.bot_model.panorama.get_panoramic_image(masked_images[img_index])
        # cv2.imshow("Pano Top????", pano_masked_top)
        # cv2.imshow("Pano Bottom????", pano_masked_bot)
        #=======================================================================

        key = gums.draw_elevations_on_panoramas(draw_own_limits=True)
        save_obj_in_pickle(gums, omnistereo_filename, locals())

#     from omnistereo.common_plot import draw_fwd_projection_GUMS
#     gums.bot_model.draw_fwd_projection()
#     gums.top_model.draw_fwd_projection()

#     draw_fwd_projection_GUMS(gums)  # Draws both figures as subplots

#     model_view = gums.top_model.theoretical_model.draw_model_vispy(finish_drawing=False, view=None)
#     model_view = gums.bot_model.theoretical_model.draw_model_vispy(finish_drawing=True, view=model_view)
#     import common_plot
#     common_plot.draw_model_mono_visvis(gums.top_model.theoretical_model)
#     common_plot.draw_omnistereo_model_visvis(gums.theoretical_model)
#     gums.theoretical_model.draw_model_vispy()
    # FIXME: Translate the initial camera view

#     test_space2plane(gums)
#     test_pixel_lifting(gums)
    if view_panos:
        pano_win_name_not_calib = "UNCALIBRATED - "
        gums.view_all_panoramas(calib_img_filename_template, img_indices, win_name_modifier=pano_win_name_not_calib, use_mask=True, mask_color_RGB=(0, 255, 0))

    if do_calibration:
        if load_calibrator_from_file:
            # Use when loading CALIBRATOR from file
            omnistereo_calibrator = load_obj_from_pickle(stereo_calibrator_filename)
        else:
            from omnistereo.calibration import CalibratorStereo
            # Use when saving CALIBRATOR to file
            omnistereo_calibrator = CalibratorStereo(working_units=gums.units)
            omnistereo_calibrator.run_corner_detection(gums, calib_img_filename_template, chessboard_params_filename, input_units="cm", chessboard_indices=img_indices, reduce_pattern_size=False, visualize=True)
            # SAVE calibrator:
            omnistereo_calibrator.set_true_chessboard_pose(gums, chessboard_params_filename, input_units="cm", chessboard_indices=img_indices, show_corners=False)  # RESET information of calibrator data
            save_obj_in_pickle(omnistereo_calibrator, stereo_calibrator_filename, locals())

        # Testing automatic point resolution
        #=======================================================================
        # from calibration import OmniStereoPair
        # import sys
        # for i in img_indices:
        #     try:
        #         omnistereo_calibrator.calibration_pairs[i] = OmniStereoPair(omnistereo_calibrator.calib_top.mirror_images[i], omnistereo_calibrator.calib_top.omni_monos[i], omnistereo_calibrator.calib_bottom.omni_monos[i])
        #     except:  # catch *all* exceptions
        #         err_msg = sys.exc_info()[1]
        #         warnings.warn("Warning...%s" % (err_msg))
        #=======================================================================

        #=======================================================================
        from omnistereo.common_cv import overlay_all_chessboards
        overlay_all_chessboards(gums, omnistereo_calibrator, draw_detection=True, visualize=True)  # <<<< Overlaying images
        #=======================================================================
        # OR
        # omnistereo_calibrator.visualize_all_calibration_pairs()

        # omnistereo_calibrator.calibration_pairs[0].visualize_points(window_name="OmniStereoPair Corners - 0")
        # omnistereo_calibrator.calibration_pairs[1].visualize_points(window_name="OmniStereoPair Corners - 1")
        # omnistereo_calibrator.calibration_pairs[2].visualize_points(window_name="OmniStereoPair Corners - 2")
        # omnistereo_calibrator.calibration_pairs[3].visualize_points(window_name="OmniStereoPair Corners - 3")
        # omnistereo_calibrator.calibration_pairs[4].visualize_points(window_name="OmniStereoPair Corners - 4")


        # top_calibrated, top_calibrator_calibrated = run_single_calibration(omni_model=gums.top_model, calibrator=omnistereo_calibrator.calib_top, is_synthetic=is_synthetic, chessboard_params_filename=chessboard_params_filename, img_indices=img_indices, eval_indices=eval_indices, load_calibrated_model_from_file=load_calibrated_model_from_file, calibrator_filename_prefix=calibration_data_path + "/single_calibrator", visualize=True, evaluate_against_truth=do_evaluation, do_radial_bounds_refinement=refine_radial_bounds)
        omnistereo_calibrator_calibrated = run_omnistereo_calibration(gums_uncalibrated=gums, omnistereo_calibrator=omnistereo_calibrator, is_synthetic=is_synthetic, model_version=model_version, chessboard_params_filename=chessboard_params_filename, img_indices=img_indices, eval_indices=eval_indices, use_gums_coupled_calibration_method=use_gums_coupled_calibration_method, load_calibrated_model_from_file=load_calibrated_model_from_file, calibrator_filename_prefix=calibrated_model_filename_prefix, evaluate_against_truth=do_evaluation, do_radial_bounds_refinement=refine_radial_bounds, do_single_trial=do_single_trial)
        gums_calibrated = omnistereo_calibrator_calibrated.estimated_omnistereo_model
    else:  # Attempting to just load the calibrated model
        if load_calibrated_model_from_file:
            if use_gums_coupled_calibration_method:
                calibrator_filename = calibrated_model_filename_prefix + ".pkl"
            else:
                calibrator_filename = calibrated_model_filename_prefix + "-decoupled.pkl"

            omnistereo_calibrator_calibrated = load_obj_from_pickle(calibrator_filename)
            gums = omnistereo_calibrator_calibrated
            gums_calibrated = omnistereo_calibrator_calibrated.estimated_omnistereo_model

    if view_panos:
        gums.view_all_panoramas(calib_img_filename_template, img_indices, win_name_modifier=pano_win_name_not_calib, use_mask=True, mask_color_RGB=(255, 0, 255))
        pano_win_name_calibrated = "CALIBRATED - "
        gums_calibrated.view_all_panoramas(calib_img_filename_template, img_indices, win_name_modifier=pano_win_name_calibrated, use_mask=True, mask_color_RGB=(0, 255, 0))

    if get_pointclouds:
#         if len(img_indices) > 0:
#             indices_for_3D_pointclouds = [img_indices[0]]
#         else:
            # HACK: Want only a-single element list?
        if experiment_name == "VICON" and model_version == "new" and not is_synthetic:
            indices_for_3D_pointclouds = [1, 2]
        else:
            indices_for_3D_pointclouds = [0]
        # Generate Panoramas for comparison
        if view_panos:
            gums.view_all_panoramas(omni_images_filename_pattern=scene_img_filename_template, img_indices=indices_for_3D_pointclouds, win_name_modifier="Scene - UNCALIBRATED - ", use_mask=True, mask_color_RGB=(0, 255, 0))
            gums_calibrated.view_all_panoramas(omni_images_filename_pattern=scene_img_filename_template, img_indices=indices_for_3D_pointclouds, win_name_modifier="Scene - CALIBRATED - ", use_mask=True, mask_color_RGB=(0, 255, 0))

        from omnistereo.common_plot import compute_pointclouds
        compute_pointclouds(omnistereo_model=gums_calibrated, poses_filename=None, omni_img_filename_template=scene_img_filename_template, features_detected_filename_template=features_detected_filename_template, img_indices=indices_for_3D_pointclouds, compute_new_3D_points=compute_new_3D_points, save_3D_points=save_3D_points, points_3D_path=points_3D_path, points_3D_filename_template=points_3D_filename_template, dense_cloud=dense_cloud, manual_point_selection=dense_manual_3D_point_selection, show_3D_reference_cyl=show_3D_reference_cyl, load_stereo_tuner_from_pickle=load_stereo_tuner_from_pickle, save_pcl=save_pcl, stereo_tuner_filename=stereo_tuner_filename, tune_live=tune_live, save_sparse_features=save_sparse_features, load_sparse_features_from_file=load_sparse_features_from_file)

    from omnistereo.common_cv import clean_up
    clean_up(wait_key_time=0)
    print("GOODBYE!")

def main_test_live():
    view_panos = False
    data_root = "data"  # The root folder for all data
    model_version = "new"  # "old"  # <<<< SETME: Can be "old" for older (Laura's) model, "new" for the new model parameters (mine)
#     exp_base = osp.join("chessboards", "2592x1944" + model_version)  # <<<<<-----  For new model_parameters. Otherwise "1m" uses only Laura's
    exp_base = osp.join("real", model_version)  # <<<<<-----  For new model_parameters. Otherwise "1m" uses only Laura's

    calibration_ground_truth = "VICON"  # <<<<<- SET to "VICON" if using ground truth data, otherwise use ""
    experiment_data_path = osp.join(data_root, exp_base, calibration_ground_truth)
    calibrated_model_filename_prefix = osp.join(experiment_data_path, "gums_calibrator")
    from omnistereo.webcam_live import WebcamLive
    cam = WebcamLive(cam_index=0, mirror_image=False, file_name="", cam_model="BLACKFLY", show_img=False)
    calibrator_filename = calibrated_model_filename_prefix + ".pkl"
    omnistereo_calibrator_calibrated = load_obj_from_pickle(calibrator_filename)
    gums_calibrated = omnistereo_calibrator_calibrated.estimated_omnistereo_model
    pano_width = np.pi * np.linalg.norm(gums_calibrated.bot_model.lowest_img_point - gums_calibrated.bot_model.precalib_params.center_point)
    stereo_tuner_filename = osp.join(experiment_data_path, "stereo_tuner.pkl")
    tune_live = True  # <<< SET: True for contiguous frames (movies). When False: the tuning is attempted (until Esc is pressed)
    #===========================================================================
    # win_pano_top = "Pano TOP"
    # win_pano_bot = "Pano BOTTOM"
    # cv2.namedWindow(win_pano_top, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(win_pano_bot, cv2.WINDOW_NORMAL)
    #===========================================================================

    success, omni_frame = cam.get_single_frame(show_img=False)
    gums_calibrated.set_current_omni_image(omni_frame, pano_width_in_pixels=pano_width, generate_panoramas=True, view=False, apply_pano_mask=True, mask_RGB=(0, 0, 0))  # Using Black pano mask

    # 3D point cloud visualization
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # 3D Visualization (Setup)
    import vispy.scene
    from vispy.scene import visuals
    from vispy import use, app
    use(app="glfw", gl="gl2")

    #
    # Make a canvas and add simple view
    #
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, title="Point Cloud")
#     # Implement key presses
#     @canvas.events.key_press.connect
#     def on_key_press(event):
#         global do_next_frame  # NOTE: Ugly need of global variable since "I think" these events cannot take arguments
#         # TODO: instantiate own Canvas class (see examples) so class attributes can accessed without using global variables
#         if event.text.lower() == 'n':
#             do_next_frame = True

    view = canvas.central_widget.add_view()
    view.camera = 'arcball'  # 'turntable'  # or try 'arcball'
    # add a colored 3D axis for orientation
    axis = visuals.XYZAxis(parent=view.scene)
    view.add(axis)
    # Add grid
    grid_3D = vispy.scene.visuals.GridLines()
    view.add(grid_3D)

    import sys
    if sys.flags.interactive != 1:
        app.create()
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # 3D Visualization (Setup)
    units_scale_factor = 1 / 1000.  # In Meters (so XYZ axes can be shown up to scale)
    min_disp = 2
    max_disp = 0
    # create scatter object and fill in the data
    scatter = visuals.Markers()
    view.add(scatter)

    # MAIN LOOP
    while success and cam.capture.isOpened():
        success, omni_frame = cam.get_single_frame(show_img=False)
        view.children[0].children.remove(scatter)  # Clear point makers
        # Clean old points
        # TODO: test this approach (just fixed, hopefully) VS the commented code just below:
        gums_calibrated.set_current_omni_image(omni_frame, pano_width_in_pixels=pano_width, generate_panoramas=False, view=False, apply_pano_mask=True, mask_RGB=(0, 0, 0))  # Using Black pano mask
        #=======================================================================
        # pano_img_top = gums_calibrated.top_model.panorama.set_panoramic_image(omni_frame)
        # cv2.imshow(win_pano_top, pano_img_top)
        # pano_img_bot = gums_calibrated.bot_model.panorama.set_panoramic_image(omni_frame)
        # cv2.imshow(win_pano_bot, pano_img_bot)
        #=======================================================================
        gums_calibrated.get_depth_map_from_panoramas(method="sgbm", use_cropped_panoramas=False, show=False, load_stereo_tuner_from_pickle=True, stereo_tuner_filename=stereo_tuner_filename, tune_live=tune_live)
        #===========================================================================
        # Generate 3D point cloud
        xyz_points, rgb_points = gums_calibrated.triangulate_from_depth_map(min_disparity=min_disp, max_disparity=max_disp, use_PCL=False, export_to_pcd=False, cloud_path="live_cloud", use_LUTs=True)

        #===============================================================================
        # 3D Visualization (Continued)
        # Points data
        # Transform point positions wrt Scene (reference frame)
        xyz_points_nonhomo = xyz_points[0, ...]
        points_wrt_C = np.hstack((xyz_points_nonhomo, np.ones(shape=(xyz_points_nonhomo.shape[0], 1))))  # Make homogeneous point coordinates
        pts_pos = points_wrt_C[:, :3] * units_scale_factor
        # points_wrt_S = np.einsum("ij, nj->ni", transform_matrices_list[idx], points_wrt_C)
        # pts_pos = points_wrt_S[:, :3] * units_scale_factor
        pts_colors = np.hstack((rgb_points / 255., np.ones_like(rgb_points[..., 0, np.newaxis])))  # Adding alpha=1 channel
        scatter.set_data(pts_pos, edge_color=None, face_color=pts_colors, size=5)
        view.add(scatter)

        app.process_events()
        canvas.update()

#         cv2.waitKey(10)
    from omnistereo.common_cv import clean_up
    clean_up(wait_key_time=1)

def main_test_remotereality_mono():
    from omnistereo import gum
    data_root = "data"  # The root folder for all data
    model_version = "rr"
    exp_base = osp.join("chessboards", model_version)  # <<<<<-----  For new model_parameters. Otherwise "1m" uses only Laura's

    calibration_ground_truth = ""  # "VICON"  # <<<<<- SET to "VICON" if using ground truth data, otherwise use ""
    experiment_data_path = osp.join(data_root, exp_base, calibration_ground_truth)
#     omni_img_file_name_prefix = osp.join(experiment_data_path, "cam_0-duo-") # BOTTOM
    omni_img_file_name_prefix = osp.join(experiment_data_path, "cam_1-duo-")  # TOP
    omni_img_filename_template = omni_img_file_name_prefix + "*.png"
    radial_bounds_filename = osp.join(experiment_data_path, "radial_bounds.pkl")
    omnistereo_filename = osp.join(experiment_data_path, "omnistereo-hyperbolic.pkl")
    stereo_calibrator_filename = osp.join(experiment_data_path, "cata_hyper_gums_calibrator.pkl")
    calibrated_model_filename_prefix = osp.join(experiment_data_path, "gums_calibrator")
    img_indices = []
    eval_indices = None  # Use None to evaluate all
    img_index = 0  # <<<<<------ Choosing an arbitrary image to work with out of the set
    gum_filename = ""

    omni_img_filename = omni_img_filename_template.replace("*", str(img_index), 1)
    omni_img = cv2.imread(omni_img_filename, 1)
    # NOTE: The convention of digital images sizes (width, height)
    image_size = np.array([omni_img.shape[1], omni_img.shape[0]])
    # Radial Pixel boundaries
    # Refine manually
    radial_initial_values = []
    from omnistereo.common_cv import find_center_and_radial_bounds
    import os
    file_exists = os.path.isfile(radial_bounds_filename)
    if not file_exists:
        radial_initial_values = [(image_size / 2.0) - 1, None, None]  # At least initialize the center pixel from MATLAB's calibration file

    [center_pixel, outer_radius, inner_radius] = find_center_and_radial_bounds(omni_img, initial_values=radial_initial_values, radial_bounds_filename=radial_bounds_filename, save_to_file=True, is_stereo=False)
    rr_GUM = gum.GUM(gum_filename, z_axis=1.0, use_theoretical_xi_and_gamma=True, center_uv_point=center_pixel)

    from omnistereo.common_cv import clean_up
    clean_up(wait_key_time=1)

if __name__ == '__main__':
    main_test()
#     main_test_live()
#     main_test_remotereality_mono()

