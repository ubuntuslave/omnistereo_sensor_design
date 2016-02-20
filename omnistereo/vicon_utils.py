'''
@summary: Custom functions for transforming pose data of grids and omnidirectional rig obtained with our VICON system.
'''

from __future__ import division
from __future__ import print_function

import numpy as np
import transformations as tr

working_units = "cm"  # ONLY for OLD experiments (initial-CVPR submission)
# Information for header
pattern_rows = 6  # Recall this is not the number of corners, but the number of rows depicted (printed)
pattern_cols = 9  # Recall this is not the number of corners, but the number of cols depicted (printed)
units_factor_from_mm_to_cm = 1 / 10.
marker_radius = 7. * units_factor_from_mm_to_cm  # in [cm]
pattern_width_row = 24 * units_factor_from_mm_to_cm  # 24 [mm]
pattern_width_col = 24 * units_factor_from_mm_to_cm  # 24 [mm]
pattern_margin = 16 * units_factor_from_mm_to_cm  # 16 [mm] Also used for the XY-offset from markers at origin (See illustration)

header_as_str = "%d, %d, %f, %f, %f" % (pattern_rows, pattern_cols, pattern_width_row, pattern_width_col, pattern_margin)

xaxis, yaxis, zaxis = [1, 0, 0], [0, 1, 0], [0, 0, 1]

# TODO: Read from file.
# Units ARE NOT meters, but they are in FEET:
units_factor_from_ft_to_cm = 100. / 3.28084  # Recal 1 [m] = 3.28084 [ft]
#===============================================================================
grid_translations_wrt_W = units_factor_from_ft_to_cm * np.array([
                                                                 [1.25, 1.19, 0.64],  # 0
                                                                 [1.42, 1.29, 0.75],  # 1
                                                                 [-0.26, 1.14, 0.73],  # 2
                                                                 [-0.51, 0.45, 0.74],  # 3
                                                                 [-0.23, 0.21, 0.65],  # 4
                                                                 [0.47, -0.42, 0.64],  # 5
                                                                 [1.59, -0.14, 0.74]  # 6
                                                                 ])
#===============================================================================

# Units in degrees (Given in order: [pitch, roll, yaw])
#===============================================================================
# grid_orientations_wrt_W = np.array([[178.6, -85.1, 8.],  # 0 --- NOT corners on bottom
#                                 [178.7, -86.4, -20.1],  # 1 <<< WRONG
#                                 [-54., 27., -85.7],  # 2 <<< WRONG
#                                 [-51.1, 30.2, -85.1 ],  # 3 <<< WRONG
#                                 [0.7, 99.8, -23.2],  # 4
#                                 [-0.35, 95.4, -2.85],  # 5
#                                 [-6.3, 77.5, 47.75 ],  # 6
#                                 [-26.0, 109.5, 80.0],  # 7  <<< WRONG
#                                 [-20.7, 86.4, 65.9]  # 8  --- NOT corners on bottom
#                                 ])
#===============================================================================
#===============================================================================
grid_orientations_wrt_W = np.array([
                                    [-4.4, 113.6, -63],  # 0
                                    [12.6, 110.7, -63.65],  # 1
                                    [-2.0, 83.8, 17.7],  # 2
                                    [-4.0, 87.5, 66.5],  # 3
                                    [7.8, 76.1, 82.5],  # 4
                                    [178.4, -77.6, 4.6],  # 5
                                    [-179.0, -110.9, -29.6]  # 6  --- REALLY UGLY corners on bottom
                                ])
#===============================================================================

# Same value for every instance
# Translation units were given in FEET
#===============================================================================
# rig_translation_wrt_W = units_factor_from_ft_to_cm * np.array([[0.55, 0.43, 1.14]])
#===============================================================================
#===============================================================================
rig_translation_wrt_W = units_factor_from_ft_to_cm * np.array([[0.56, 0.34, 1.14]])
#===============================================================================

# Units in degrees (Given in order: [pitch, roll, yaw])
#===============================================================================
# rig_orientations_wrt_W = np.array([[-0.1, 1.2, 5.9]])
#===============================================================================
#===============================================================================
rig_orientations_wrt_W = np.array([[0.3, 1.0, 0.1]])
#===============================================================================


def find_chessboard_pose(Gvicon_translation_wrt_W, Gvicon_rot_wrt_W_angles_degrees, axes_sequence="sxyz"):
    '''
    The [Gvicon] is the rigid object grid being tracked by the vicon system

    @note: The Pattern is on its XZ-plane!!!!

    @param Gvicon_translation_wrt_W: An numpy array of [x, y, z] position for a grid frame [Gvicon] wrt the world global frame [W]
    @param Gvicon_rot_wrt_W_angles_degrees: A numpy array of [pitch, roll, yaw] Euler angles (in degrees) of the grid frame [Gvicon] wrt [W]

    @return:  A list encoding the pose of the ground-truth Grid frame [Ggt] wrt [W], such as [x, y, z, roll_degrees, pitch_degrees, yaw_degrees]
    '''
    # First, convert angles to radians
    Gvicon_rot_angles_wrt_W = np.deg2rad(Gvicon_rot_wrt_W_angles_degrees)
    # Because Dtrack was giving angles in order: [pitch, roll, yaw]:
    #===========================================================================
    # T_Gvicon_wrt_W = tr.euler_matrix(ai=Gvicon_rot_angles_wrt_W[..., 1], aj=Gvicon_rot_angles_wrt_W[..., 0], ak=Gvicon_rot_angles_wrt_W[..., 2], axes='sxyz')
    # T_Gvicon_wrt_W[0:3, 3] = Gvicon_translation_wrt_W  # Append translation vector to homogeneous matrix
    #===========================================================================
    # OR similarly:
    T_Gvicon_wrt_W_rot = tr.euler_matrix(ai=Gvicon_rot_angles_wrt_W[..., 1], aj=Gvicon_rot_angles_wrt_W[..., 0], ak=Gvicon_rot_angles_wrt_W[..., 2], axes=axes_sequence)
    T_Gvicon_wrt_W_trans = tr.translation_matrix(Gvicon_translation_wrt_W)
    T_Gvicon_wrt_W = tr.concatenate_matrices(T_Gvicon_wrt_W_trans, T_Gvicon_wrt_W_rot)  # Order: 1st) Rotation, and then 2nd) Translation
    # Compare to input angles: np.rad2deg(tr.euler_from_matrix(T_Gvicon_wrt_W, axes='sxyz'))

    Ogt_translation_wrt_Gvicon = np.array([pattern_margin, pattern_margin, -marker_radius, 1])
    # Since, pattern is on its XZ-plane, we must rotate -90 degrees around the Gvicon's x-axis
    T_Ggt_wrt_Gvicon = tr.rotation_matrix(-np.pi / 2.0, xaxis)
    T_Ggt_wrt_Gvicon[:, 3] = Ogt_translation_wrt_Gvicon  # Append translation vector to homogeneous matrix

    T_Ggt_wrt_W = tr.concatenate_matrices(T_Gvicon_wrt_W, T_Ggt_wrt_Gvicon)  # Implies: T_Ggt_wrt_W = T_Gvicon_wrt_W * T_Ggt_wrt_Gvicon
    # or T_Ggt_wrt_W = np.einsum("ij, jk->ik", T_Gvicon_wrt_W, T_Ggt_wrt_Gvicon)
    # or T_Ggt_wrt_W = np.dot(T_Gvicon_wrt_W, T_Ggt_wrt_Gvicon)
    #===========================================================================
    # angles = tr.euler_from_matrix(T_Ggt_wrt_W, axes='sxyz')
    # trans = tr.translation_from_matrix(T_Ggt_wrt_W)
    # # scale, shear, angles, trans, persp = tr.decompose_matrix(T_Ggt_wrt_W)  # SAME
    # euler_angles_in_degrees = np.rad2deg(angles)
    # return [trans[0], trans[1], trans[2], euler_angles_in_degrees[0], euler_angles_in_degrees[1], euler_angles_in_degrees[2]]
    #===========================================================================
    return T_Ggt_wrt_W

def find_camera_pose(pos_Rvicon_wrt_C, rot_Rvicon_wrt_C_angles, Rvicon_translation_wrt_W, Rvicon_wrt_W_angles_degrees, axes_sequence="sxyz"):
    '''
    @param Gvicon_translation_wrt_W: An numpy array of [x, y, z] position for a grid frame [Rvicon] wrt the world global frame [W]
    @param Rvicon_wrt_W_angles_degrees: A numpy array of [pitch, roll, yaw] Euler angles (in degrees) of the rig frame set with markers [Rvicon] wrt [W]
    '''
    T_Rvicon_wrt_C = tr.euler_matrix(ai=rot_Rvicon_wrt_C_angles[..., 0], aj=rot_Rvicon_wrt_C_angles[..., 1], ak=rot_Rvicon_wrt_C_angles[..., 2], axes='sxyz')
    T_Rvicon_wrt_C[0:3, 3] = pos_Rvicon_wrt_C  # Append translation vector to homogeneous matrix
    T_C_wrt_Rvicon = tr.inverse_matrix(T_Rvicon_wrt_C)

    Rvicon_rot_angles_wrt_W = np.deg2rad(Rvicon_wrt_W_angles_degrees)
    # Because Dtrack was giving angles in order: [pitch, roll, yaw]:
    #===========================================================================
    # T_Rvicon_wrt_W = tr.euler_matrix(ai=Rvicon_rot_angles_wrt_W[..., 1], aj=Rvicon_rot_angles_wrt_W[..., 0], ak=Rvicon_rot_angles_wrt_W[..., 2], axes='sxyz')
    # T_Rvicon_wrt_W[0:3, 3] = Rvicon_translation_wrt_W  # Append translation vector to homogeneous matrix
    #===========================================================================
    # OR Similarly:
    T_Rvicon_wrt_W_rot = tr.euler_matrix(ai=Rvicon_rot_angles_wrt_W[..., 1], aj=Rvicon_rot_angles_wrt_W[..., 0], ak=Rvicon_rot_angles_wrt_W[..., 2], axes=axes_sequence)
    T_Rvicon_wrt_W_trans = tr.translation_matrix(Rvicon_translation_wrt_W)
    T_Rvicon_wrt_W = tr.concatenate_matrices(T_Rvicon_wrt_W_trans, T_Rvicon_wrt_W_rot)  # Order: 1st) Rotation, and then 2nd) Translation
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    T_C_wrt_W = tr.concatenate_matrices(T_Rvicon_wrt_W, T_C_wrt_Rvicon)

    return T_C_wrt_W

# TODO: IDEA optimize the pose resolution of the vicon system so it reduces the error against the calibrated results evaluation.
from sensor_evaluation import evaluate_3D_error_from_grid_poses

def optimize_cam_pos_wrt_C_theoretical(omnistereo_model, calibrator, eval_indices, model_version, chessboard_params_filename, x_axis_points=None, do_single_trial=True):
    if x_axis_points is None:
        x_axis_points = get_axis_points_manually(omnistereo_model)  # We must extract them for the first time

    from copy import deepcopy
    omni_model_for_calibration = deepcopy(omnistereo_model)  # Make a copy of this object instance!

    from scipy import optimize
    opt_method = 'TNC'
    # Bounds:
    roll_bounds = (-.3, .3)  # in [radians]
    pitch_bounds = (-.3, .3)  # in [radians]
    yaw_bounds = (-.3, .3)  # in [radians]
    tx_bounds = (-30, 30)  # in [mm]
    ty_bounds = (-30, 30)  # in [mm]
    tz_bounds = (-30, 30)  # in [mm]
    cam_pose_error_limits = [roll_bounds, pitch_bounds, yaw_bounds, tx_bounds, ty_bounds, tz_bounds]
    cam_pose_error_init = [0, 0, 0, 0, 0, 0]

    func_args = (omni_model_for_calibration, calibrator, eval_indices, x_axis_points, model_version)
    if do_single_trial:
        params = optimize.minimize(fun=measure_cam_pos_error, x0=cam_pose_error_init, args=func_args, method=opt_method, jac=False, hess=None, hessp=None, bounds=cam_pose_error_limits, constraints=(), tol=None, callback=None, options={'maxiter' : 2000, 'disp' : True, })
    else:
        minimizer_kwargs = dict(args=func_args, method=opt_method, jac=False, hess=None, hessp=None, bounds=cam_pose_error_limits, constraints=(), tol=None, callback=None, options={'maxiter' : 2000, 'disp' : True, })
        params = optimize.basinhopping(func=measure_cam_pos_error, x0=cam_pose_error_init, niter=20, minimizer_kwargs=minimizer_kwargs)

#     if params.status:
    print(params)
    #=======================================================================
    # euler_angles = params.x[:3]
    # rotation_mat = tr.euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes='sxyz')
    # translation_mat = tr.translation_matrix(params.x[3:])
    # T_C_wrt_W_refinement = tr.concatenate_matrices(translation_mat, rotation_mat)  # Order: 1st) Rotation, and then 2nd) Translation
    #=======================================================================
    angles = params.x[:3]
    translation = params.x[3:]
    T_C_wrt_W_refinement = tr.compose_matrix(angles=angles, translate=translation)
    print(50 * "~")
    print("Successfully found VICON rig pose (Refinement): %s degrees, %s [mm]" % (np.rad2deg(angles), translation))
    print(50 * "~")
    #===========================================================================
    # else:
    #     T_C_wrt_W_refinement = tr.identity_matrix()
    #===========================================================================

    T_Ggt_wrt_C_array, x_axis_points = resolve_poses_wrt_C_theoretical(omnistereo_model, model_version, grid_output_filename=chessboard_params_filename, x_axis_points=x_axis_points, T_C_wrt_W_refinement=T_C_wrt_W_refinement)

    return T_Ggt_wrt_C_array, T_C_wrt_W_refinement, x_axis_points

def measure_cam_pos_error(x, omnistereo_model, calibrator, eval_indices, x_axis_points, model_version):
    T_C_wrt_W_refinement_candidate = tr.compose_matrix(angles=x[:3], translate=x[3:])
    T_Ggt_wrt_C_array, x_axis_points = resolve_poses_wrt_C_theoretical(omnistereo_model, model_version, grid_output_filename=None, x_axis_points=x_axis_points, T_C_wrt_W_refinement=T_C_wrt_W_refinement_candidate)
    calibrator.T_G_wrt_C_list_for_calibration = T_Ggt_wrt_C_array[calibrator.chessboard_indices]
    calibrator.set_true_chessboard_pose(omnistereo_model, chessboard_params=T_Ggt_wrt_C_array, input_units="mm", chessboard_indices=eval_indices, show_corners=False)  # RESET information of calibrator data
    error = evaluate_3D_error_from_grid_poses(calibrator, eval_indices=eval_indices, visualize=False, verbose=False)
    return error

def get_axis_points_manually(omnistereo_model):
    from common_cv import PointClicker
    import cv2
    clicker_window_name = "Vicon's X-axis extraction (click Origin, then direction point)"
    cv2.namedWindow(clicker_window_name, cv2.WINDOW_NORMAL)
    pt_clicker = PointClicker(clicker_window_name, max_clicks=2, draw_polygon_clicks=True)
    axis_points = pt_clicker.get_clicks_uv_coords(omnistereo_model.current_omni_img)
    axis_points = axis_points.reshape(1, 2, 2)  # Needs to be reshaped to operate on functions that use 2D-tables of points
    cv2.destroyWindow(clicker_window_name)
    return axis_points

def resolve_poses_wrt_C_theoretical(omnistereo_model, rig_version, grid_output_filename=None, axes_sequence="sxyz", x_axis_points=None, T_C_wrt_W_refinement=None):
    """
    Each pose of the ground-truth grid [Ggt] obtained from a Vicon system is transformed with respect to [C],
    where the relative pose of the rig frame [R] and the theoretical camera model [C] are inferred from the known marker positions
    with respect to the theoretical model.

    @param T_C_wrt_W_refinement: a 4x4 correction matrix for the transformation. Values are assumed in [mm], so proper conversion must be attempted

    @retval: the array of poses of the ground-truth grids with respect to [C].
    @retval: the points describing the x-axis vector
    """
    if rig_version == "old":
        distance_rig_marker_to_Oc = 52.33 * units_factor_from_mm_to_cm  # Rotated means that the [Rvicon] is not alined with [C]. TODO: azimuthal alignment
        rig_mount_top_holder_thickness = 10 * units_factor_from_mm_to_cm  # In [cm]
    else:  # Dimensions on the new rig
        # TODO: set values by looking at them in Solidworks sketches
        distance_rig_marker_to_Oc = 35 * units_factor_from_mm_to_cm  # Rotated means that the [Rvicon] is not alined with [C]. TODO: azimuthal alignment
        rig_mount_top_holder_thickness = 22 * units_factor_from_mm_to_cm  # In [cm]

    T_Ggt_wrt_W_array = np.zeros(((grid_translations_wrt_W.shape[0],) + (4, 4)))
    T_Ggt_wrt_C_array = np.zeros(((grid_translations_wrt_W.shape[0],) + (4, 4)))
    T_Ggt_wrt_C_pos_AND_euler_degree_angles = np.zeros(((grid_translations_wrt_W.shape[0],) + (6,)))

    # Collect the points by clicking on omnidirectional image
    if x_axis_points is None:
        x_axis_points = get_axis_points_manually(omnistereo_model)

    azimuth_angles, elevation_angles_wrt_F = omnistereo_model.top_model.get_direction_angles_from_pixel(x_axis_points)
    azimuth_Rvicon_origin_wrt_C = azimuth_angles[0, 0]
    elevation_rsys_wrt_F = elevation_angles_wrt_F[0, 0]  # This point should be coplanar with the marker's centroid that acts as the [Rvicon] origin
    azimuth_Rvicon_x_axis_end_wrt_C = azimuth_angles[0, 1]

    from cata_hyper_model import HyperCataStereo
    from gum import GUMStereo
    if isinstance(omnistereo_model, HyperCataStereo):  # Using the Theoretical model
        z_rsys_wrt_C = omnistereo_model.height_above
    elif isinstance(omnistereo_model, GUMStereo):
        r_sys = omnistereo_model.theoretical_model.system_radius
        z_rsys_wrt_F = r_sys * np.tan(elevation_rsys_wrt_F)
        z_rsys_wrt_C = omnistereo_model.top_model.F[2, 0] + z_rsys_wrt_F

    z_top_marker_wrt_C = (z_rsys_wrt_C * units_factor_from_mm_to_cm) + rig_mount_top_holder_thickness + marker_radius

    rig_A_x_wrt_C = distance_rig_marker_to_Oc * np.cos(azimuth_Rvicon_origin_wrt_C)
    rig_A_y_wrt_C = distance_rig_marker_to_Oc * np.sin(azimuth_Rvicon_origin_wrt_C)
    rig_B_x_wrt_C = distance_rig_marker_to_Oc * np.cos(azimuth_Rvicon_x_axis_end_wrt_C)
    rig_B_y_wrt_C = distance_rig_marker_to_Oc * np.sin(azimuth_Rvicon_x_axis_end_wrt_C)
    pos_Orig_vicon_wrt_C = np.array([rig_A_x_wrt_C, rig_A_y_wrt_C, z_top_marker_wrt_C])
    pos_B_vicon_wrt_C = np.array([rig_B_x_wrt_C, rig_B_y_wrt_C, z_top_marker_wrt_C])
    rig_x_axis_wrt_C = pos_B_vicon_wrt_C - pos_Orig_vicon_wrt_C
    rig_x_axis_normalized = rig_x_axis_wrt_C / np.linalg.norm(rig_x_axis_wrt_C)
    angle_x_axis_C_and_R = np.arccos(np.dot(xaxis, rig_x_axis_normalized))

    rig_orientation_wrt_C = np.array([0, 0, angle_x_axis_C_and_R])  # Roll, pitch, yaw

    T_C_wrt_W_raw = find_camera_pose(pos_Orig_vicon_wrt_C, rig_orientation_wrt_C, rig_translation_wrt_W, rig_orientations_wrt_W, axes_sequence=axes_sequence)
    if T_C_wrt_W_refinement is None:
        T_C_wrt_W_refinement_in_cm = tr.identity_matrix()  # Make an identity matrix if no transformation refinement was passed
    else:
        T_C_wrt_W_refinement_in_cm = T_C_wrt_W_refinement.copy()
        T_C_wrt_W_refinement_in_cm[..., :3, -1] = T_C_wrt_W_refinement[..., :3, -1] * units_factor_from_mm_to_cm

    T_C_wrt_W = tr.concatenate_matrices(T_C_wrt_W_refinement_in_cm, T_C_wrt_W_raw)  # Order: 1st) Rotation, and then 2nd) Translation

    T_W_wrt_C = tr.inverse_matrix(T_C_wrt_W)
    for k in range(T_Ggt_wrt_W_array.shape[0]):
        T_Ggt_wrt_W_array[k] = find_chessboard_pose(grid_translations_wrt_W[[k], :], grid_orientations_wrt_W[[k], :], axes_sequence=axes_sequence)
        # Transform Ggt wrt C
        T_Ggt_wrt_C_array[k] = tr.concatenate_matrices(T_W_wrt_C, T_Ggt_wrt_W_array[k])

        # Convert to vector of position and euler angles in degrees
        angles = tr.euler_from_matrix(T_Ggt_wrt_C_array[k], axes='sxyz')
        euler_angles_in_degrees = np.rad2deg(angles)
        trans = tr.translation_from_matrix(T_Ggt_wrt_C_array[k])
        T_Ggt_wrt_C_pos_AND_euler_degree_angles[k] = np.array([trans[0], trans[1], trans[2], euler_angles_in_degrees[0], euler_angles_in_degrees[1], euler_angles_in_degrees[2]])

    # We are looking for T_Ggt_wrt_C
    if grid_output_filename is not None:
        print("Saving results of Grid pose to file", grid_output_filename, end="...")
        # NOTE: we don't want the header to start with #, so we set comments=""
        np.savetxt(grid_output_filename, T_Ggt_wrt_C_pos_AND_euler_degree_angles, delimiter=",", header=header_as_str, comments="", fmt='%10.4f')
        # RECALL: positions in the file are given in [cm] and angles in [degrees]
        print("Done!")

    # One final step: Convert the translation components to the corresponding units
    T_Ggt_wrt_C_array_in_mm = T_Ggt_wrt_C_array
    # The translation vector is encoded in the last column
    T_Ggt_wrt_C_array_in_mm[..., :3, -1] = T_Ggt_wrt_C_array[..., :3, -1] / units_factor_from_mm_to_cm
    return T_Ggt_wrt_C_array_in_mm, x_axis_points

def test_T_from_3_points(filename_markers, filename_T):
    T_from_file = tr.identity_matrix()
    T_from_file[:, 0], T_from_file[:, 1], T_from_file[:, 2], T_from_file[:, 3] = np.loadtxt(filename_T, delimiter=',', usecols=(0, 1, 2, 3), unpack=True)

    pt_markers = np.loadtxt(filename_markers, delimiter=',', usecols=(1, 2, 3), unpack=False)
    pt_orig_marker = pt_markers[0]  # np.array([10, 20, 10])
    pt_on_x_marker = pt_markers[1]  # np.array([1, 2, 33])
    pt_on_y__marker = pt_markers[2]  # np.array([-1, -20, 20])

    T = tr.get_frame_pose_from_3_planar_points(pt_orig_marker, pt_on_x_marker, pt_on_y__marker)
    print("Transformation matrix:", T)
    if np.allclose(T, T_from_file):
        print("Check is Correct!")

def get_T_from_3_points(filename_markers, idx_order, verbose=False):
    pt_markers = np.loadtxt(filename_markers, delimiter=',', usecols=(0, 1, 2), unpack=False)
    pt_orig_marker = pt_markers[idx_order[0]]
    pt_on_x_marker = pt_markers[idx_order[1]]
    pt_on_y_marker = pt_markers[idx_order[2]]

    T = tr.get_frame_pose_from_3_planar_points(pt_orig_marker, pt_on_x_marker, pt_on_y_marker)
    if verbose:
        print("Transformation matrix:", T)
    return T

def test_T_with_point_on_plane(filename_markers, T_G_wrt_Vicon, idx_test_pt, pt_test_pos_wrt_G):
    pt_markers = np.loadtxt(filename_markers, delimiter=',', usecols=(0, 1, 2), unpack=False)
    pt_test_wrt_Vicon = np.append(pt_markers[idx_test_pt], [1])

    pt_test_pos_wrt_Vicon_result = np.einsum("ij, j->i", T_G_wrt_Vicon, pt_test_pos_wrt_G)
    # print("Computed test marker position wrt [Vicon] ", pt_test_pos_wrt_Vicon_result)
    wrt_Vicon_pos_error = np.linalg.norm(pt_test_pos_wrt_Vicon_result - pt_test_wrt_Vicon)

    #===========================================================================
    # T_Vicon_wrt_G = tr.inverse_matrix(T_G_wrt_Vicon)
    # pt_test_pos_wrt_G_result = np.einsum("ij, j->i", T_Vicon_wrt_G, pt_test_wrt_Vicon)
    # print("Computed test marker position wrt [G] ", pt_test_pos_wrt_G_result)
    #===========================================================================

    return wrt_Vicon_pos_error

def experiment_rr(data_path, rig_markers_indices, grid_markers_indices):
    filename_rig_markers = data_path + "/Rig_pose_0-markers.csv"
    T_rig_wrt_Vicon = get_T_from_3_points(filename_markers=filename_rig_markers, idx_order=rig_markers_indices, verbose=False)
#     test_T_with_point_on_plane(filename_rig_markers, T_rig_wrt_Vicon, idx_test_pt=0, pt_test_pos_wrt_G=[40, 0, 0, 1])
    grid_poses_list = [0, 1, 2, 3, 4, 5]
    pos_errors = []
    for p in grid_poses_list:
        filename_grid_markers = data_path + "/grid_pose_%d-markers.csv" % (p)
        T_grid_wrt_Vicon = get_T_from_3_points(filename_markers=filename_grid_markers, idx_order=grid_markers_indices)
        pos_err = test_T_with_point_on_plane(filename_grid_markers, T_grid_wrt_Vicon, idx_test_pt=0, pt_test_pos_wrt_G=[373, 0, 0, 1])
        pos_errors += [pos_err]
        pos_err = test_T_with_point_on_plane(filename_grid_markers, T_grid_wrt_Vicon, idx_test_pt=3, pt_test_pos_wrt_G=[0, 190, 0, 1])
        pos_errors += [pos_err]
        pos_err = test_T_with_point_on_plane(filename_grid_markers, T_grid_wrt_Vicon, idx_test_pt=1, pt_test_pos_wrt_G=[373, 100, 0, 1])
        pos_errors += [pos_err]

    from common_tools import error_analysis_simple
    error_analysis_simple(pos_errors, units="mm")

def marker_error_omnistereo_rig(data_path, rig_markers_indices, grid_markers_indices):
    '''
    Marker on rig look like, where O is the origin, X and ~y are coplanar, but m3 is the extra marker closer to O
            . . .
          .   X    .
       .   /          .
     .   /              .
    .  /                 .
    . O                  .
     .                  .
       .m3             .
          .          .
             . ~y.
    '''
    grid_poses_list = [0, 1, 2, 3, 4, 5, 6, 7]

    # Analysis about known makers positions:
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    pos_errors = []
    for p in grid_poses_list:
        filename_grid_markers = data_path + "/grid_pose_%d-markers.csv" % (p)
        T_grid_wrt_Vicon = get_T_from_3_points(filename_markers=filename_grid_markers, idx_order=grid_markers_indices)
        pos_err = test_T_with_point_on_plane(filename_grid_markers, T_grid_wrt_Vicon, idx_test_pt=1, pt_test_pos_wrt_G=[920, 0, 0, 1])
        pos_errors += [pos_err]
        pos_err = test_T_with_point_on_plane(filename_grid_markers, T_grid_wrt_Vicon, idx_test_pt=2, pt_test_pos_wrt_G=[0, 300, 0, 1])
        pos_errors += [pos_err]
        pos_err = test_T_with_point_on_plane(filename_grid_markers, T_grid_wrt_Vicon, idx_test_pt=3, pt_test_pos_wrt_G=[920, 200, 0, 1])
        pos_errors += [pos_err]
    from common_tools import error_analysis_simple
    error_analysis_simple(pos_errors, units="mm")
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # This is what we reallly need to compute:
    # NOTE: units are in [mm]
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    x_Og_to_Gvicon = 60  # mm
    y_Og_to_Gvicon = -100  # mm
    z_Og_to_Gvicon = 2  # mm CHECKME: measure the thickness of the board and the marker holder (3D printed)
    T_Ggt_wrt_Gvicon_trans = tr.translation_matrix(np.array([x_Og_to_Gvicon, y_Og_to_Gvicon, z_Og_to_Gvicon, 1]))
    # Since, pattern is on its XZ-plane, we must rotate -90 degrees around the Gvicon's x-axis
    T_Ggt_wrt_Gvicon_rot = tr.rotation_matrix(-np.pi / 2.0, xaxis)
    T_Ggt_wrt_Gvicon_static = tr.concatenate_matrices(T_Ggt_wrt_Gvicon_trans, T_Ggt_wrt_Gvicon_rot)  # Order: 1st) Rotation, and then 2nd) Translation

    T_Ggt_wrt_R_list = get_T_Ggt_wrt_R(data_path, grid_poses_list, rig_markers_indices, grid_markers_indices, T_Ggt_wrt_Gvicon_static)
    from common_plot import draw_frame_poses
    draw_frame_poses(T_Ggt_wrt_R_list)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def get_T_Ggt_wrt_R(data_path, grid_poses_indices, rig_markers_indices, grid_markers_indices, T_Ggt_wrt_Gvicon_static):
    filename_rig_markers = data_path + "/Rig_pose_0-markers.csv"
    T_rig_wrt_Vicon = get_T_from_3_points(filename_markers=filename_rig_markers, idx_order=rig_markers_indices, verbose=False)
    T_Ggt_wrt_R_list = []
    for p in grid_poses_indices:
        filename_grid_markers = data_path + "/grid_pose_%d-markers.csv" % (p)
        T_grid_wrt_Vicon = get_T_from_3_points(filename_markers=filename_grid_markers, idx_order=grid_markers_indices)
        T_Gvicon_wrt_R = get_T_G_wrt_R_vicon(T_R_wrt_V=T_rig_wrt_Vicon, T_G_wrt_V=T_grid_wrt_Vicon)
        T_Ggt_wrt_R = tr.concatenate_matrices(T_Gvicon_wrt_R, T_Ggt_wrt_Gvicon_static)  # Implies: T_Ggt_wrt_W = T_Gvicon_wrt_W * T_Ggt_wrt_Gvicon
        T_Ggt_wrt_R_list.append(T_Ggt_wrt_R)
        print("T_Ggt[{idx}]_wrt_R = {T_G_wrt_R}".format(idx=p, T_G_wrt_R=T_Ggt_wrt_R))

    return T_Ggt_wrt_R_list

def get_T_G_wrt_R_vicon(T_R_wrt_V, T_G_wrt_V):
    T_G_wrt_R = tr.concatenate_matrices(tr.inverse_matrix(T_R_wrt_V), T_G_wrt_V)
    return T_G_wrt_R

if __name__ == '__main__':
    # NEW CVPR experiments:
    marker_radius = 7.  # mm
    #===========================================================================
    # data_path_rr="data/chessboards/rr/VICON"
    # rig_markers_indices_rr = [0, 1, 2]
    # grid_markers_indices_rr =[2, 0, 3]
    # experiment_rr(data_path=data_path_rr, rig_markers_indices=rig_markers_indices_rr, grid_markers_indices=grid_markers_indices_rr)
    #===========================================================================
    data_path_omnistereo = "data/chessboards/1920x1200/new/VICON"
    rig_markers_indices_omnistereo = [3, 1, 2]
    grid_markers_indices_omnistereo = [0, 1, 2]
    marker_error_omnistereo_rig(data_path=data_path_omnistereo, rig_markers_indices=rig_markers_indices_omnistereo, grid_markers_indices=grid_markers_indices_omnistereo)


