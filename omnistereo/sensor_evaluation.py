'''
Created on Sep 15, 2014

@author: carlos
'''
from __future__ import division
from __future__ import print_function

import numpy as np
import omnistereo.transformations as tr
from omnistereo import common_tools

def forward_projection_eval_mono(camera_model, calibrator, chessboard_indices=[], wrt_mirror_focus=False, use_ground_truth_pose=False, visualize=False, show_detected=False, visualize_panorama=False, overlay_all_omni=True, base_img=None, proj_pt_RGB_color=None, verbose=False):
    '''
    @param chessboard_indices: To work with the selected indices of the patterns pre-loaded in the Omnistereo calibrator's list
    '''
    if use_ground_truth_pose:
        # Check if it's possible to do so
        use_ground_truth_pose = calibrator.has_chesboard_pose_info

    # Initialize empty list of sets of pattern triangulated points
    list_len = len(calibrator.omni_monos)
    # triangulated_points = [None] * list_len
    # chessboard_object_points_wrt_C = [None] * list_len
    # error_norms = [None] * list_len
    all_pixel_errors = []

    if hasattr(camera_model, "mirror_name"):
        mirror_name = camera_model.mirror_name.upper()
    else:
        if camera_model.mirror_number == 1:
            mirror_name = "TOP"
        else:
            mirror_name = "BOTTOM"

    if mirror_name == "TOP":
        pt_thickness = 9
    else:
        pt_thickness = 6
    pt_thickness_pano = 6

    if chessboard_indices is None or len(chessboard_indices) == 0:
        chessboard_indices = range(list_len)

    if not use_ground_truth_pose:
        eval_type = "No GT"
    else:
        eval_type = "With GT"

    if visualize:
        from omnistereo import common_cv
        import cv2
        # The forward-projected (true) pixels due to the projection model parameters or analytical equation
        if proj_pt_RGB_color is None:
            proj_pt_RGB_color = (0, 255, 0)  # Green because it's the projection from the GROUND TRUTH data
        if show_detected:
            detected_pt_RGB_color = (0, 0, 255)  # blue because (R,G,B)


    if overlay_all_omni:
        if visualize:
            win = mirror_name.capitalize() + " Point Projection Error Test - ALL Patterns" + "(%s)" % (eval_type)
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        if base_img is None:
            # Overlaying images
            from omnistereo.common_cv import overlay_all_chessboards
            omni_img = overlay_all_chessboards(camera_model, calibrator, indices=chessboard_indices, draw_detection=False, visualize=False)
        else:
            omni_img = base_img

    if not use_ground_truth_pose:
        s = -camera_model.F[2, 0]
        T_C_wrt_M = tr.translation_matrix([0, 0, s])
        # For the projection loop, if not using GT data, we need to find new indices that match the calibration's list
        points_wrt_pattern = calibrator.points_wrt_pattern_for_calibration
        optimized_extrinsics = calibrator.estimated_grid_poses
        original_chessboard_indices = chessboard_indices
        chessboard_indices = range(int(len(optimized_extrinsics) / 7))

    for idx in chessboard_indices:
        if use_ground_truth_pose:
            if not calibrator.omni_monos[idx].found_points:
                continue  # to next index
            # The detected (observed) pixels for chessboard points
            detected_corners = calibrator.omni_monos[idx].image_points_homo
            if wrt_mirror_focus:
                chessboard_points_in_ref_frame = calibrator.chessboard_3D_points_wrt_F[idx]
                get_pixel_from_3D_func = camera_model.get_pixel_from_3D_point_wrt_M
            else:
                chessboard_points_in_ref_frame = calibrator.chessboard_3D_points_wrt_C[idx]
                get_pixel_from_3D_func = camera_model.get_pixel_from_3D_point_wrt_C
        else:
            # The detected (observed) pixels for chessboard points
            detected_corners = calibrator.detected_points_for_calibration[idx]
            # Resolve 3D points on grid from extrinsic poses estimated after calibration
            q = optimized_extrinsics[idx * 7:idx * 7 + 4]  # rotation quaternion
            t = optimized_extrinsics[idx * 7 + 4:idx * 7 + 7]  # translation vector
            T_G_wrt_C = tr.quaternion_matrix(q)
            T_G_wrt_C[:3, 3] = t
            chessboard_points_wrt_C = np.einsum("ij, mnj->mni", T_G_wrt_C, points_wrt_pattern[idx])
                    # Transform between [G] and mirror [M]
            # T_G_wrt_M = tr.concatenate_matrices(T_C_wrt_M, T_G_wrt_C)
            # points_wrt_M = np.einsum("ij, klj->kli", T_G_wrt_M, points_wrt_G)
            # Or just using the intermediate results:
            points_wrt_M = np.einsum("ij, klj->kli", T_C_wrt_M, chessboard_points_wrt_C)
            chessboard_points_in_ref_frame = points_wrt_M
            # Resolve idx to monos (for the upcoming visualization, if so):
            idx = original_chessboard_indices[idx]
            get_pixel_from_3D_func = camera_model.get_pixel_from_3D_point_wrt_M

        # The projected corners
        _, _, projected_corners = get_pixel_from_3D_func(chessboard_points_in_ref_frame)

        if visualize:
            current_omni_img = calibrator.omni_monos[idx].omni_image.copy()  # copy omni_image
            if not overlay_all_omni:
                win = "%s - Point Projection Error Test - Pattern [%d]" % (mirror_name, idx)
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                omni_img = current_omni_img.copy()
            if visualize_panorama:
                pano_img = camera_model.panorama.get_panoramic_image(current_omni_img)
                win_pano = "%s - Point Projection Error Test (Onto Panorama)- Pattern [%d]" % (mirror_name, idx)
                cv2.namedWindow(win_pano, cv2.WINDOW_NORMAL)
            if show_detected:
                common_cv.draw_points(omni_img, detected_corners[..., :2].reshape(-1, 2), color=detected_pt_RGB_color, thickness=pt_thickness)
                if visualize_panorama:
                    is_valid, detected_corners_pano = camera_model.panorama.get_panoramic_pixel_coords_from_omni_pixel(detected_corners)
                    common_cv.draw_points(pano_img, detected_corners_pano[..., :2].reshape(-1, 2), color=detected_pt_RGB_color, thickness=pt_thickness_pano)

            common_cv.draw_points(omni_img, projected_corners[..., :2].reshape(-1, 2), color=proj_pt_RGB_color, thickness=int(pt_thickness / 3))
            if visualize_panorama:
                is_valid, projected_corners_pano = camera_model.panorama.get_panoramic_pixel_coords_from_omni_pixel(projected_corners)
                common_cv.draw_points(pano_img, projected_corners_pano[..., :2].reshape(-1, 2), color=proj_pt_RGB_color, thickness=int(pt_thickness_pano / 3))

            if not overlay_all_omni:
                cv2.imshow(win, omni_img)

            if visualize_panorama:
                cv2.imshow(win_pano, pano_img)


        if overlay_all_omni and visualize:
            cv2.imshow(win, omni_img)

        if visualize:
            cv2.waitKey(1)

        error_corners = projected_corners[..., :2] - detected_corners[..., :2]
        error_norms = np.linalg.norm(error_corners, axis=-1).flatten()
        if verbose:
            ans_rmse = common_tools.rms(error_norms)
            mean, std_dev = common_tools.mean_and_std(error_norms)
            print("%s RMSE of Calibration Pattern %d: %f [%s]" % (camera_model.mirror_name, idx, ans_rmse, "pixels"))
            print("Mean= %f [%s] Std.dev = %f [%s]" % (mean, "pixels", std_dev, "pixels"))

        all_pixel_errors.append(error_norms)


    # Finally, compute the entire RMSE for this set
    all_rmse = common_tools.rms(all_pixel_errors)
    print("%s RMSE of projected points in this set of Calibration Patterns: %f [%s]" % (camera_model.mirror_name, all_rmse, "pixels"))
    all_mean, all_std_dev = common_tools.mean_and_std(all_pixel_errors)
    print("Mean= %f [%s] Std.dev = %f [%s]" % (all_mean, "pixels", all_std_dev, "pixels"))
    print(60 * "^", "\n")

    return all_pixel_errors, omni_img

def forward_projection_eval_stereo(camera_model, calibrator, chessboard_indices=[], wrt_mirror_focus=False, use_ground_truth_pose=False, visualize=False, show_detected=False, visualize_panorama=False, omni_img=None, overlay_all_omni=True, proj_pt_RGB_color=None, verbose=False):
    '''
    @param chessboard_indices: To work with the selected indices of the patterns pre-loaded in the Omnistereo calibrator's list
    '''
    if omni_img is None:
        if overlay_all_omni:
            from omnistereo.common_cv import overlay_all_chessboards
            omni_img = overlay_all_chessboards(camera_model, calibrator, draw_detection=False, visualize=False)
        else:
            omni_img = None

    all_pixel_errors_top, omni_img = forward_projection_eval_mono(camera_model=camera_model.top_model, calibrator=calibrator.calib_top, chessboard_indices=chessboard_indices, wrt_mirror_focus=wrt_mirror_focus, use_ground_truth_pose=use_ground_truth_pose, visualize=visualize, show_detected=show_detected, visualize_panorama=visualize_panorama, overlay_all_omni=overlay_all_omni, base_img=omni_img, proj_pt_RGB_color=proj_pt_RGB_color, verbose=verbose)
    all_pixel_errors_bottom, omni_img = forward_projection_eval_mono(camera_model=camera_model.bot_model, calibrator=calibrator.calib_bottom, chessboard_indices=chessboard_indices, wrt_mirror_focus=wrt_mirror_focus, use_ground_truth_pose=use_ground_truth_pose, visualize=visualize, show_detected=show_detected, visualize_panorama=visualize_panorama, overlay_all_omni=overlay_all_omni, base_img=omni_img, proj_pt_RGB_color=proj_pt_RGB_color, verbose=verbose)

    # Finally, compute the entire RMSE for this set
    all_pixel_errors = all_pixel_errors_top + all_pixel_errors_bottom  # Join the lists
    total_rmse = common_tools.rms(all_pixel_errors)
    print("TOTAL RMSE of projected points in this set of Calibration Patterns: %f [%s]" % (total_rmse, "pixels"))
    total_mean, total_std_dev = common_tools.mean_and_std(all_pixel_errors)
    print("Mean= %f [%s] Std.dev = %f [%s]" % (total_mean, "pixels", total_std_dev, "pixels"))
    print(60 * "&", "\n")
    return omni_img


# TODO: put this in camera_models.py OmnistereoModel
def get_confidence_weights_from_pixel_error_stereo(camera_model, calibrator, chessboard_indices=[]):
    '''
    We define a confidence weight as the inverse of accumulated pixel projection error (reprojection cost)
    '''
    list_len = len(calibrator.points_wrt_pattern_for_calibration)
    if chessboard_indices is None or len(chessboard_indices) == 0:
        chessboard_indices = range(list_len)
    # Weights are base on each grid's pose:
    weights = list_len * [0.0]
    optimized_extrinsics = calibrator.estimated_grid_poses
    for i in range(list_len):
        img_points_top = calibrator.detected_points_for_calibration[i * 2]
        img_points_bottom = calibrator.detected_points_for_calibration[(i * 2) + 1]

        q = optimized_extrinsics[i * 7:i * 7 + 4]  # rotation quaternion
        t = optimized_extrinsics[i * 7 + 4:i * 7 + 7]  # translation vector
        T_G_wrt_C = tr.pose_matrix_from_quaternion_and_translation(q, t)

        weights[i] = camera_model.get_confidence_weight_from_pixel_RMSE_stereo(img_points_top=img_points_top, img_points_bot=img_points_bottom, obj_pts_homo=calibrator.points_wrt_pattern_for_calibration[i], T_G_wrt_C=T_G_wrt_C)

    return weights

def sparse_triangulation_eval(camera_model, calibrator, chessboard_indices=[], use_midpoint_triangulation=True, manual_pixel_bias=0.0, debug_mode=False):
    '''
    @param chessboard_indices: List of selected indices of the patterns pre-loaded in the Omnistereo calibrator's list to be used.
    '''
    # Initialize empty list of sets of pattern triangulated points
    list_len = len(calibrator.calibration_pairs)
    # triangulated_points = [None] * list_len
    # chessboard_object_points_wrt_C = [None] * list_len
    # error_norms = [None] * list_len
    all_error_norms = []

    if chessboard_indices is None or len(chessboard_indices) == 0:
        chessboard_indices = range(list_len)

    if use_midpoint_triangulation:
        method_str = "(Using Midpoint Method)"
    else:
        method_str = "(Using Rectified Method)"

    print(10 * "v", "Sparse Triangulation Evaluation", method_str, 10 * "v")
    for idx in chessboard_indices:
        if calibrator.calibration_pairs[idx].found_points:
            # The detected (observed) pixels for chessboard points
            chessboard_pixels_top = calibrator.calibration_pairs[idx].mono_top.image_points + manual_pixel_bias
            chessboard_pixels_bottom = calibrator.calibration_pairs[idx].mono_bottom.image_points + manual_pixel_bias

            triangulated_points = camera_model.get_triangulated_point_from_pixels(chessboard_pixels_top, chessboard_pixels_bottom, use_midpoint_triangulation=use_midpoint_triangulation)[..., :3]

            # The true (computed from the analytical transformation of pattern given as a priori)
            # Note, that taking either the top or bottom is equivalent since the pattern is the same (and at the same position wrt C)
            chessboard_object_points_wrt_C = calibrator.calib_top.chessboard_3D_points_wrt_C[idx][..., :3]

            # Use the Root Mean Square of the Euclidean vector difference norms (magnitude)
            # between observed (detected) and true (analytical) pixel points.
            error_vectors = triangulated_points - chessboard_object_points_wrt_C
            error_norms = np.linalg.norm(error_vectors, axis=-1).flatten()
            ans_rmse = common_tools.rms(error_norms)
            if debug_mode:
                print("RMSE of Calibration Pattern %d: %f [%s] \t(Added pixel bias = %f)" % (idx, ans_rmse, camera_model.units, manual_pixel_bias))

                mean, std_dev = common_tools.mean_and_std(error_norms)
                print("Pattern %d: Mean= %f [%s] Std.dev = %f [%s]" % (idx, mean, camera_model.units, std_dev, camera_model.units))

            all_error_norms.append(error_norms)

    # Finally, compute the entire RMSE for this set
    all_rmse = common_tools.rms(all_error_norms)
    print("RMSE of All Points in this set of Calibration Patterns: %f [%s]" % (all_rmse, camera_model.units), end="")
    if manual_pixel_bias > 0:
        print("\t(Added pixel bias = %f)" % (manual_pixel_bias))
    else:
        print("")
    all_mean, all_std_dev = common_tools.mean_and_std(all_error_norms)
    print("Mean= %f [%s] Std.dev = %f [%s]" % (all_mean, camera_model.units, all_std_dev, camera_model.units))
    print(40 * "^", "\n")


def unit_test_for_triangulation(camera_model, calibrator, chessboard_indices=[], use_midpoint_triangulation=True, manual_pixel_bias=0.0, debug_mode=False):
    '''
    @param chessboard_indices: List of selected indices of the patterns pre-loaded in the Omnistereo calibrator's list to be used.
    '''
    # Initialize empty list of sets of pattern triangulated points
    list_len = len(calibrator.calibration_pairs)
    # triangulated_3D_points = [None] * list_len
    # chessboard_object_points_wrt_C = [None] * list_len
    # error_norms = [None] * list_len
    all_error_norms = []

    if chessboard_indices is None or len(chessboard_indices) == 0:
        chessboard_indices = range(list_len)

    for idx in chessboard_indices:
        if calibrator.calibration_pairs[idx].found_points:
            # The detected (observed) pixels for chessboard points
            # The true (computed from the analytical transformation of pattern given as a priori)
            # Note, that taking either the top or bottom is equivalent since the pattern is the same (and at the same position wrt C)
            true_3D_points_wrt_C_top = calibrator.calib_top.chessboard_3D_points_wrt_C[idx]
            _, _, true_corners_top = camera_model.top_model.get_pixel_from_3D_point_wrt_C(true_3D_points_wrt_C_top)
            true_3D_points_wrt_C_bottom = calibrator.calib_bottom.chessboard_3D_points_wrt_C[idx]
            _, _, true_corners_bottom = camera_model.bot_model.get_pixel_from_3D_point_wrt_C(true_3D_points_wrt_C_bottom)

            true_corners_top = true_corners_top + manual_pixel_bias
            true_corners_bottom = true_corners_bottom + manual_pixel_bias

            if use_midpoint_triangulation:
                direction_vectors_top = camera_model.top_model.get_direction_vector_from_focus(true_corners_top)
                direction_vectors_bottom = camera_model.bot_model.get_direction_vector_from_focus(true_corners_bottom)

                rows = direction_vectors_top.shape[0]
                cols = direction_vectors_top.shape[1]
                triangulated_3D_points = np.ndarray((rows, cols, 3))
                # TODO: this function should be implemented using the common perpendicular midpoint method (vectorized)
                for row in range(rows):
                    for col in range(cols):
                        mid_Pw, _, _, _ = camera_model.get_triangulated_midpoint(direction_vectors_top[row, col], direction_vectors_bottom[row, col])
                        triangulated_3D_points[row, col] = mid_Pw
            else:
                triangulated_3D_points = camera_model.get_triangulated_point_from_pixels(true_corners_top, true_corners_bottom)[..., :3]


            # Use the Root Mean Square of the Euclidean vector difference norms (magnitude)
            # between observed (detected) and true (analytical) pixel points.
            error_vectors = triangulated_3D_points - true_3D_points_wrt_C_top[..., :3]
            error_norms = np.linalg.norm(error_vectors, axis=-1).flatten()
            ans_rmse = common_tools.rms(error_norms)
            if debug_mode:
                print("Unit Test RMSE of pattern %d: %f [%s] \t(Added pixel bias = %f)" % (idx, ans_rmse, camera_model.units, manual_pixel_bias))

            all_error_norms.append(error_norms)

    # Finally, compute the entire RMSE for this set
    all_rmse = common_tools.rms(all_error_norms)
    print("Unit Test RMSE of All Points in this set of Calibration Patterns: %f [%s] \t(Added pixel bias = %f)" % (all_rmse, camera_model.units, manual_pixel_bias))


def evaluate_3D_error_from_grid_poses(calibrator, chessboard_indices=[], eval_indices=[], app=None, visualize=False, vis_only_evaluated_grids=False, verbose=True):
    '''
    @param omni_model: A monocular GUM
    @param chessboard_indices: The data indices that should be considered for visualization. This is not the same as for the evaluation.
    @param eval_indices: List of indices to perform evaluation with. Use None to evaluate all
    '''
    from omnistereo.calibration import CalibratorStereo
    if isinstance(calibrator, CalibratorStereo):
        omni_model = calibrator.estimated_omnistereo_model
    else:
        omni_model = calibrator.estimated_omni_model

    if chessboard_indices is None or len(chessboard_indices) == 0:
        chessboard_indices = calibrator.chessboard_indices  # range(int(len(optimized_extrinsics) / 7))

    if eval_indices is None:
        eval_indices = chessboard_indices

    if visualize:
        from omnistereo.gum import GUMStereo
        if isinstance(omni_model, GUMStereo):
            grid_color_init = 'b'
            if visualize:
                from omnistereo.common_plot import draw_omnistereo_model_visvis
                draw_model_func = draw_omnistereo_model_visvis
        else:
            if visualize:
                from omnistereo.common_plot import draw_model_mono_visvis
                draw_model_func = draw_model_mono_visvis
            if omni_model.mirror_number == 1:
                grid_color_init = 'c'
                if verbose:
                    print("TOP Mirror - Optimized extrinsic parameters:")
            else:
                grid_color_init = 'm'
                if verbose:
                    print("BOTTOM Mirror - Optimized extrinsic parameters:")

        import visvis as vv
        if app is None:
            try:
                from PySide import QtGui, QtCore
                backend = 'pyside'
            except ImportError:
                from PyQt4 import QtGui, QtCore
                backend = 'pyqt4'

            app = vv.use(backend)


        if hasattr(omni_model, "theoretical_model"):
            z_offset, app = draw_model_func(omni_model.theoretical_model, app=app, finish_drawing=False, mirror_transparency=0.5, show_labels=False, show_only_real_focii=True, show_reference_frame=True, show_grid_box=True, busy_grid=False)

        a = vv.gca()

    # Expand to all elements (not just those selected for calibration)
    from omnistereo.camera_models import OmniStereoModel
    if isinstance(omni_model, OmniStereoModel):
        len_all_monos = len(calibrator.calibration_pairs)
    else:
        len_all_monos = len(calibrator.omni_monos)
    T_W_wrt_C_list_true = len_all_monos * [None]
    points_wrt_pattern = len_all_monos * [None]
    obj_pts_init = len_all_monos * [None]
    obj_pts_opt = len_all_monos * [None]
    obj_pts_true = len_all_monos * [None]
    initialization_extrinsics = calibrator.initial_params[:calibrator.total_extrinsic_params]
    optimized_extrinsics = calibrator.estimated_grid_poses
    all_grid_pose_errors = []
    all_grid_pose_errors_init = []

    g = 0
    for idx in chessboard_indices:  # Draw the desired chessboard indices
        try:
            points_wrt_pattern[idx] = calibrator.points_wrt_pattern_for_calibration[g]
            if calibrator.has_chesboard_pose_info:
                T_W_wrt_C_list_true[idx] = calibrator.T_G_wrt_C_list_for_calibration[g]
            # ----------------- Initial values ------------------
            q_init = initialization_extrinsics[g * 7:g * 7 + 4]  # rotation quaternion
            t_init = initialization_extrinsics[g * 7 + 4:g * 7 + 7]  # translation vector
            T_init = tr.quaternion_matrix(q_init)
            T_init[:3, 3] = t_init
            obj_pts_init[idx] = np.einsum("ij, mnj->mni", T_init, points_wrt_pattern[idx])
            # ----------------- Optimized values ------------------
            q = optimized_extrinsics[g * 7:g * 7 + 4]  # rotation quaternion
            t = optimized_extrinsics[g * 7 + 4:g * 7 + 7]  # translation vector
            angles = tr.euler_from_quaternion(q, axes="sxyz")
            if verbose:
                print("Grid [%d]: Euler angles %s [degrees]" % (idx, np.rad2deg(angles)), "Translation %s [%s]:" % (t, omni_model.units))
            T_opt = tr.quaternion_matrix(q)
            T_opt[:3, 3] = t
            obj_pts_opt[idx] = np.einsum("ij, mnj->mni", T_opt, points_wrt_pattern[idx])
            if calibrator.has_chesboard_pose_info:
                obj_pts_true[idx] = np.einsum("ij, mnj->mni", T_W_wrt_C_list_true[idx], points_wrt_pattern[idx])
        except:
            print("Warning: No chessboard pose info set for index [%d]!" % (idx))
        g = g + 1  # Update to consecutive index in the calibration list

    g_all_rmse = None
    if calibrator.has_chesboard_pose_info:
        for e in eval_indices:
            try:
                g_error_norms_init = np.linalg.norm((obj_pts_true[e] - obj_pts_init[e]), axis=-1).flatten()
                all_grid_pose_errors_init.append(g_error_norms_init)
                # Quantify error from ground truth
                g_error_norms = np.linalg.norm((obj_pts_true[e] - obj_pts_opt[e]), axis=-1).flatten()
                all_grid_pose_errors.append(g_error_norms)
                if verbose and calibrator.has_chesboard_pose_info:
                    g_rmse_init = common_tools.rms(g_error_norms_init)
                    g_rmse = common_tools.rms(g_error_norms)
                    g_mean, g_std_dev = common_tools.mean_and_std(g_error_norms)
                    print("--- Initial RMSE of Calibration Pattern %d: %f [%s]" % (e, g_rmse_init, omni_model.units))
                    print("+++ Optimal RMSE of Calibration Pattern %d: %f [%s]" % (e, g_rmse, omni_model.units))
                    print("Mean= %f [%s] Std.dev = %f [%s]" % (g_mean, omni_model.units, g_std_dev, omni_model.units))
                    print(60 * ".")
            except:
                print("PROBLEM: evaluating index [%d]" % (e))
        try:  # Compute the entire RMSE for this set
            g_all_rmse = common_tools.rms(all_grid_pose_errors)
            g_all_rmse_init = common_tools.rms(all_grid_pose_errors_init)
            g_all_mean, g_all_std_dev = common_tools.mean_and_std(all_grid_pose_errors)
            if verbose and calibrator.has_chesboard_pose_info:
                print(50 * "-")
                print("Initial RMSE using all points: %f [%s]" % (g_all_rmse_init, omni_model.units))
                print("Optimized RMSE using all points: %f [%s]" % (g_all_rmse, omni_model.units))
                print("Mean= %f [%s] Std.dev = %f [%s]" % (g_all_mean, omni_model.units, g_all_std_dev, omni_model.units))
                print(50 * "-")
        except:
            print("Problem: computing RMSE!")

    if visualize:
        if vis_only_evaluated_grids:
            vis_indices = eval_indices
        else:
            vis_indices = chessboard_indices

        for idx in vis_indices:  # Draw the desired chessboard indices
            # Draw INITIAL calibration grid in the world (in color)
            #===================================================================
            # xx_obj_pts_init = obj_pts_init[idx][..., 0]
            # yy_obj_pts_init = obj_pts_init[idx][..., 1]
            # zz_obj_pts_init = obj_pts_init[idx][..., 2]
            # obj_pts_init_grid = vv.grid(xx_obj_pts_init, yy_obj_pts_init, zz_obj_pts_init, axesAdjust=True, axes=a)
            # obj_pts_init_grid.edgeColor = grid_color_init
            # obj_pts_init_grid.edgeShading = "plain"  # possible shaders: None, plain, flat, gouraud, smooth
            # obj_pts_init_grid.diffuse = 0.0
            # Og_pt_init = vv.Point(obj_pts_init[idx][0, 0, 0], obj_pts_init[idx][0, 0, 1], obj_pts_init[idx][0, 0, 2])
            # vv.plot(Og_pt_init, ms='.', mc=grid_color_init, mw=5, ls='', mew=0, axesAdjust=False)
            #===================================================================
            # Draw ESTIMATED calibration grid in the world (in color)
            xx_obj_pts_opt = obj_pts_opt[idx][..., 0]
            yy_obj_pts_opt = obj_pts_opt[idx][..., 1]
            zz_obj_pts_opt = obj_pts_opt[idx][..., 2]
            obj_pts_opt_grid = vv.grid(xx_obj_pts_opt, yy_obj_pts_opt, zz_obj_pts_opt, axesAdjust=True, axes=a)
            obj_pts_opt_grid.edgeColor = 'r'
            obj_pts_opt_grid.edgeShading = "plain"  # possible shaders: None, plain, flat, gouraud, smooth
            obj_pts_opt_grid.diffuse = 0.0
            Og_pt_opt = vv.Point(obj_pts_opt[idx][0, 0, 0], obj_pts_opt[idx][0, 0, 1], obj_pts_opt[idx][0, 0, 2])
            vv.plot(Og_pt_opt, ms='.', mc='r', mw=5, ls='', mew=0, axesAdjust=False)

            # Draw TRUE calibration grid in the world (in green) only for those EVALUATED indices
            try:
                if calibrator.has_chesboard_pose_info and (idx in eval_indices):
                    xx_obj_pts_true = obj_pts_true[idx][..., 0]
                    yy_obj_pts_true = obj_pts_true[idx][..., 1]
                    zz_obj_pts_true = obj_pts_true[idx][..., 2]
                    obj_pts_true_grid = vv.grid(xx_obj_pts_true, yy_obj_pts_true, zz_obj_pts_true, axesAdjust=True, axes=a)
                    obj_pts_true_grid.edgeColor = "g"  # Ground Truth = green
                    obj_pts_true_grid.edgeShading = "plain"  # possible shaders: None, plain, flat, gouraud, smooth
                    obj_pts_true_grid.diffuse = 0.0
                    Og_pt_true = vv.Point(obj_pts_true[idx][0, 0, 0], obj_pts_true[idx][0, 0, 1], obj_pts_true[idx][0, 0, 2])
                    vv.plot(Og_pt_true, ms='.', mc="g", mw=5, ls='', mew=0, axesAdjust=False)
            except:
                print("Warning: True drawing of grid index [%d]!" % (idx))

        vv.title('Extrinsic Poses')
        # Start app
        app.Run()

    return g_all_rmse


if __name__ == '__main__':
    pass
