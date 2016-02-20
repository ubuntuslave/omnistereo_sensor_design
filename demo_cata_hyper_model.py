# -*- coding: utf-8 -*-
# demo_cata_hyper_model.py

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
Demonstration of both synthetic (simulation) and real omnidirectional stereo model
for the related MDPI Sensors article titled:
'Design and Analysis of a Single−Camera Omnistereo Sensor for Quadrotor Micro Aerial Vehicles (MAVs)'.
'''
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpldatacursor import datacursor, HighlightingDataCursor
from sympy import Matrix, ImmutableMatrix
import os.path as osp
from omnistereo import common_tools

# pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
# mpl.rcParams.rotate_to(pgf_with_rc_fonts)

def init_omnistereo_theoretical(omni_img, radial_bounds_filename, theoretical_params_filename, model_version, is_synthetic):
    # NOTE: The convention of digital images sizes (width, height)
    image_size = np.array([omni_img.shape[1], omni_img.shape[0]])

    # Radial Pixel boundaries
    # Refine manually
    radial_initial_values = []
    from omnistereo.common_cv import find_center_and_radial_bounds
    file_exists = osp.isfile(radial_bounds_filename)
    if not file_exists:
        radial_initial_values = [[(image_size / 2.0) - 1, None, None], [(image_size / 2.0) - 1, None, None]]  # At least initialize the center pixel from MATLAB's calibration file

    [[center_pixel_top, outer_radius_top, inner_radius_top], [center_pixel_bottom, outer_radius_bottom, inner_radius_bottom]] = find_center_and_radial_bounds(omni_img, initial_values=radial_initial_values, radial_bounds_filename=radial_bounds_filename, save_to_file=True)

    # THEORETICAL VALUES:
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    from omnistereo.cata_hyper_model import PinholeCamera, HyperCata, HyperCataStereo
    c1, c2, k1, k2, d, r_sys, r_reflex, r_cam = common_tools.get_theoretical_params_from_file(theoretical_params_filename, file_units="cm")
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
    return theoretical_omni_stereo


def main_demo(is_synthetic=True):
    # HYPERBOLIC Parameters (Used in Publication):
    #===========================================================================
    # k1 = 5.7319  # Unitless
    # k2 = 9.7443  # Unitless
    # Using millimeters
    # r_sys = 37.0
    # r_reflex = 17.226
    # r_cam = 7.25
    # c1 = 123.488
    # c2 = 241.803
    # d = 233.684
    #===========================================================================
    # vvvvvvvvvvvvvvvvvvvvvvv OPTIONS vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    load_model_from_file = True  # <<< SETME: to load omnistereo model from a pickle or start anew
    show_panoramic_img = True
    show_3D_model = False
    get_pointclouds = False
    compute_new_3D_points = True
    dense_cloud = True
    manual_point_selection = False

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # vvvvvvvvvvvvvvvvvvvvvvv SETUP vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    data_root = "data"  # The root folder for all data
    model_version = "old"  # Set to "old" for the PUBLISHED params, or "new" for the new one
    if is_synthetic:
        model_type = "synthetic"
    else:
        model_type = "real"
    data_path_prefix = osp.join(data_root, model_type, model_version)
    experiment_name = "office"
    experiment_path = osp.join(data_path_prefix, experiment_name)  # Pose estimation experiment: Translation on x only by 0, 25 cm and 75 cm (wrt init)

    # For SHOWING OFF: virtual office
    omni_img_filename_template = osp.join(experiment_path, "office-%s-*.png" % (model_version))  # With PUBLISHED parameters
    # omni_img_filename_template = osp.join(data_path_prefix, experiment_path, "office" + model_version + "-*.png")  # NEW design
    img_indices = []  # Choosing a predefined set of images to work with out of the set
    img_index = 0  # <<<<<------ Choosing an arbitrary image to work with out of the set

    omnistereo_model_filename = osp.join(data_path_prefix, "omnistereo-hyperbolic.pkl")
    # ------------------------------------------------
    radial_bounds_filename = osp.join(data_path_prefix, "radial_bounds.pkl")
    # ------------------------------------------------
    points_3D_filename_template = osp.join(experiment_path, "3d_points-" + model_version + "-*.pkl")

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    from omnistereo.common_cv import has_opencv, get_images
    opencv_exists = has_opencv()

    omni_images_list = get_images(omni_img_filename_template, indices_list=img_indices, show_images=True)
    # Read params from file and scale to [mm] units since using [cm] (only those params with dimensions)
    theoretical_params_filename = osp.join(data_root, "parameters-%s.txt" % (model_version))

    if load_model_from_file:
        omnistereo_model = common_tools.load_obj_from_pickle(omnistereo_model_filename)
    else:
        # omni_img_filename = omni_img_filename_template.replace("*", str(img_index), 1)
        # omni_img = cv2.imread(omni_img_filename, 1)
        omni_img = omni_images_list[img_index]
        omnistereo_model = init_omnistereo_theoretical(omni_img, radial_bounds_filename, theoretical_params_filename, model_version, is_synthetic=is_synthetic)
        pano_width = np.pi * np.linalg.norm(omnistereo_model.bot_model.lowest_img_point - omnistereo_model.bot_model.precalib_params.center_point)
        omnistereo_model.set_current_omni_image(omni_img, pano_width_in_pixels=pano_width, generate_panoramas=True, idx=img_index, view=True)
        common_tools.save_obj_in_pickle(omnistereo_model, omnistereo_model_filename, locals())

    #===========================================================================
    # sanity_check(omnistereo_model)
    #===========================================================================

    # Get pixel from pano test
    u, v, m_homo = omnistereo_model.top_model.panorama.get_panorama_pixel_coords_from_direction_angles(theta=np.deg2rad([10., 11, 80, -10]), psi=np.deg2rad([1, 12., 360, 60]))

    if show_panoramic_img and opencv_exists:
        pano_win_name_prefix = "DEMO - "
        omnistereo_model.view_all_panoramas(omni_img_filename_template, img_indices, win_name_modifier=pano_win_name_prefix, use_mask=True, mask_color_RGB=(0, 255, 0))

    if show_3D_model:  # Figure 4 (MDPI Sensors journal article)
        try:
            # Drawing forward projection from 3D points:
            xw, yw, zw = 80, 10 , 100
            # Pw = [(xw, yw, zw), (-xw, yw, zw), (xw, -yw, zw), (-xw, -yw, zw)]
            Pw = [(xw, yw, zw)]
            from omnistereo.common_plot import draw_fwd_projection_omnistereo
            draw_fwd_projection_omnistereo(omnistereo_model, Pw, verbose=True, fig_size=None)
            plt.show()  # Show both figures in separate windows
        except ImportError:
            print("MPLOT3D could not be imported for 3D visualization!")

            try:
                # NOTE: drawing with visvis and PyQt4 is troublesome when OpenCV is displaying windows that are using Qt5!!!
                # Drawing just the model:
                from omnistereo.common_plot import draw_omnistereo_model_visvis
                draw_omnistereo_model_visvis(omnistereo_model, finish_drawing=True, show_grid_box=False, mirror_transparency=0.5, show_reference_frame=True)
                # common_plot.draw_model_mono_visvis(omnistereo_model.top_model.theoretical_model, finish_drawing=True, show_grid_box=False, show_reference_frame=True)

            except ImportError:
                print("VISVIS could not be imported for 3D visualization!")
                try:
                    # USING Vispy:
                    from omnistereo.common_plot import draw_omnistereo_model_vispy
                    draw_omnistereo_model_vispy(omnistereo_model, show_grid=True, backend='pyqt4')
                except ImportError:
                    print("VISPY could not be imported for 3D visualization!")

    # UNCOMMENT THE FOLLOWING CODE BLOCKS AS DESIRED:
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    #===========================================================================
    # Draw a single mirror's profile in 2D
    # from omnistereo.common_plot import draw_fwd_projection
    # draw_fwd_projection(omnistereo_model.top_model)
    #===========================================================================

    # Drawing backprojected pixels AND covariance ellipsoids
    #===========================================================================
    # # Warning: Don't use anything less than 10 because it will be hard to visualize
    # pixels_to_skew = 0
    # #     delta_pixel = np.array([[65, 40, 20, 10]])
    # delta_pixel = np.array([[75., 47.4, 23.7]])
    # # For example, 1 pixel disparity produces a convergence about 50 m away
    # # whereas, with a disparity of 10 pixels, the diraction rays converge around 2.5 m away (horizontal range).
    # #     m1 = np.array([[[920, cam_mirror1.v_center, 1], [940, cam_mirror1.v_center, 1], [950, cam_mirror1.v_center, 1], [960, cam_mirror1.v_center, 1]]])
    # m1 = np.array([[[920, omnistereo_model.top_model.precalib_params.v_center, 1], [930, omnistereo_model.top_model.precalib_params.v_center, 1], [950, omnistereo_model.top_model.precalib_params.v_center, 1]]])
    # az1, el1 = omnistereo_model.top_model.get_direction_angles_from_pixel(m1)
    # u2, v2, m2_same_el1 = omnistereo_model.bot_model.get_pixel_from_direction_angles(az1, el1)
    # m2 = np.dstack((u2 - delta_pixel, v2 - pixels_to_skew))  # Needs to decrease delta_pixel pixels (disparity) on u2 (only for this example on the u-axis) so the elevation on mirror 2 increases for convergence
    # from omnistereo.common_plot import draw_bak_projection_visvis
    # draw_bak_projection_visvis(omnistereo_model, m1, m2, number_of_std_deviations=1, draw_covariance=True, line_thickness=2, show_grid_box=True, show_labels=False, plot_density_function=False)
    #===========================================================================

    # Drawing a single point backprojected AND its covariance ellipsoids
    #===========================================================================
    #===========================================================================
    # pixels_to_skew = 0
    # delta_pixel = np.array([[150]])
    # m1 = np.array([[[920, omnistereo_model.top_model.precalib_params.v_center, 1]]])
    # az1, el1 = omnistereo_model.top_model.get_direction_angles_from_pixel(m1)
    # m2 = np.dstack((m1[..., 0] - delta_pixel * np.cos(az1), m1[..., 1] - delta_pixel * np.sin(az1) - pixels_to_skew))  # Needs to decrease delta_pixel pixels (disparity) on u2 (only for this example on the u-axis) so the elevation on mirror 2 increases for convergence
    #===========================================================================
    #===========================================================================
    # Using visvis:
    # from omnistereo.common_plot import draw_bak_projection_visvis
    # draw_bak_projection_visvis(omnistereo_model, m1, m2, number_of_std_deviations=1, draw_covariance=True, plot_density_function=True)
    #===========================================================================
    #===========================================================================
    # Using matplotlib only:
    # from omnistereo.common_plot import draw_bak_projection
    # draw_bak_projection(omnistereo_model, m1, m2)
    #===========================================================================

    #===========================================================================
    # # Figure 9 (Sensors Journal article)
    # from omnistereo.common_plot import plot_k_vs_rsys_for_vFOV
    # plot_k_vs_rsys_for_vFOV(omnistereo_model.top_model, fig_size=None)
    #===========================================================================

    #===========================================================================
    # # Figure 10 (Sensors Journal article)
    # from omnistereo.common_plot import plot_k_vs_baseline_for_vFOV
    # plot_k_vs_baseline_for_vFOV(omnistereo_model, fig_size=None)
    #===========================================================================

    #===========================================================================
    # from omnistereo.common_plot import plot_mirror_profiles
    # plot_mirror_profiles(omnistereo_model)
    #===========================================================================

    #===========================================================================
    # # Figure 11 (Sensors Journal article)
    # from omnistereo.common_plot import plot_catadioptric_spatial_resolution_vs_k
    # plot_catadioptric_spatial_resolution_vs_k(omnistereo_model, fig_size=None, legend_location=None)
    #===========================================================================

    # Plotting spatial resolution

    #===========================================================================
    # from omnistereo.common_plot import plot_perspective_camera_spatial_resolution
    # plot_perspective_camera_spatial_resolution(omnistereo_model.top_model.precalib_params, in_2D=True)
    #===========================================================================

    #===========================================================================
    # from omnistereo.common_plot import plot_catadioptric_spatial_resolution_by_BakerNayar
    # plot_catadioptric_spatial_resolution_by_BakerNayar(omnistereo_model)
    #===========================================================================

    #===========================================================================
    # # Figure 12 (Sensors Journal article)
    # from omnistereo.common_plot import plot_catadioptric_spatial_resolution
    # plot_catadioptric_spatial_resolution(omnistereo_model, in_2D=True, eta_max=18, fig_size=None)
    #===========================================================================

    # Range variation:
    #===========================================================================
    # Pns_high = omnistereo_model.get_triangulated_point_wrt_Oc(omnistereo_model.top_model.highest_elevation_angle, omnistereo_model.bot_model.highest_elevation_angle, 0)
    # Pns_mid = omnistereo_model.get_triangulated_point_wrt_Oc(omnistereo_model.top_model.lowest_elevation_angle, omnistereo_model.bot_model.highest_elevation_angle, 0)
    # Pns_low = omnistereo_model.get_triangulated_point_wrt_Oc(omnistereo_model.top_model.lowest_elevation_angle, omnistereo_model.bot_model.lowest_elevation_angle, 0)
    # print(Pns_high, Pns_mid, Pns_low)
    # hor_range_min_for_plot = min(Pns_low[0, 0, 0], Pns_high[0, 0, 0])
    # vert_range_min_for_plot = min(Pns_low[0, 0, 2], Pns_high[0, 0, 2])
    # vert_range_max_for_plot = max(Pns_low[0, 0, 2], Pns_high[0, 0, 2])
    # delta_z_mirror1, z_level_1 = omnistereo_model.top_model.get_vertical_range_variation(hor_range_min_for_plot)
    # delta_rho_mirror1, rho_level_1 = omnistereo_model.top_model.get_horizontal_range_variation(vert_range_min_for_plot)
    # delta_phi_mirror1, phi_level_1 = omnistereo_model.top_model.get_angular_range_variation(150)
    # delta_z_mirror2, z_level_2 = omnistereo_model.bot_model.get_vertical_range_variation(hor_range_min_for_plot)
    # delta_rho_mirror2, rho_level_2 = omnistereo_model.bot_model.get_horizontal_range_variation(vert_range_min_for_plot)
    # delta_phi_mirror2, phi_level_2 = omnistereo_model.bot_model.get_angular_range_variation(150)
    #===========================================================================

    #===========================================================================
    # # Figure 17 (Sensors Journal article)
    # from omnistereo.common_plot import plot_range_variation_due_to_pixel_disparity
    # plot_range_variation_due_to_pixel_disparity(omnistereo_model, disp_min=1, disp_max=100, fig_size=None)
    #===========================================================================

    #===========================================================================
    # from omnistereo.common_plot import plot_effect_of_pixel_disparity_on_range
    # plot_effect_of_pixel_disparity_on_range(omnistereo_model, disp_min=1, disp_max=100, disp_nums=5, use_log=True, plot_zoom=True, fig_size=None)
    #===========================================================================

    #===========================================================================
    # from omnistereo.common_plot import plot_vertical_range_variation
    # plot_vertical_range_variation(omnistereo_model, hor_range_max=30, depth_nums=5, use_meters=True, fig_size=None)
    #===========================================================================

    #===========================================================================
    # from omnistereo.common_plot import plot_horizontal_range_variation
    # plot_horizontal_range_variation(omnistereo_model, vertical_range_min=-500, vertical_range_max=500, depth_nums=5, use_meters=False, fig_size=None)
    #===========================================================================


    plt.show()  # Show both figures in separate windows


    if get_pointclouds:
        stereo_tuner_filename = osp.join(experiment_path, "stereo_tuner.pkl")
        from omnistereo.common_plot import compute_pointclouds_simple
        compute_pointclouds_simple(omnistereo_model, omni_img_filename_template=None, img_indices=[img_index], compute_new_3D_points=compute_new_3D_points, dense_cloud=dense_cloud, manual_point_selection=manual_point_selection, load_stereo_tuner_from_pickle=True, save_pcl=False, pcd_cloud_path=experiment_path, stereo_tuner_filename=stereo_tuner_filename, tune_live=False, save_sparse_features=False, load_sparse_features_from_file=False)


    from omnistereo.common_cv import clean_up
    clean_up(wait_key_time=0)

def sanity_check(omnistereo_model):
    print("\nSANITY CHECK:")
    x = omnistereo_model.reflex_radius
    y = 0
    z = omnistereo_model.d / 2.0
    p1_fp = np.array([[[x, y, z, 1]]])
    _, _, pixel_fp = omnistereo_model.top_model.get_pixel_from_3D_point_wrt_C(p1_fp)
    pixel_bp = np.array([[[868, 480, 1]]])
    print("I got pixel_fp = %s" % (pixel_fp))
    print("but it SHOULD BE close to: %s" % (pixel_bp))
    common_tools.unit_test(pixel_fp, pixel_bp, decimals=0)

    print("BP test:")
    p1_bp = omnistereo_model.top_model.lift_pixel_to_mirror_surface(pixel_fp)
    common_tools.unit_test(p1_fp, p1_bp, decimals=3)
    print("I got p1_bp = %s" % (p1_bp))
    print("but it SHOULD BE: %s" % (p1_fp))

    Q_fp = omnistereo_model.top_model.project_3D_point_to_normalized_plane(p1_fp)
    # or
    # Q_fp = omnistereo_model.top_model.project_mirror_point_to_normalized_plane(p1_fp)
    Q_bp = omnistereo_model.top_model.lift_pixel_to_projection_plane(pixel_fp)
    common_tools.unit_test(Q_fp, Q_bp, decimals=3)


if __name__ == '__main__':
    main_demo(is_synthetic=True)

    # NOT AVAILABLE in omnistereo_sensor_design repo:
    # common_tools.demo_optimize_FOV()

