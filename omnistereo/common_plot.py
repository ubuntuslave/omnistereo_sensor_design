# -*- coding: utf-8 -*-
# common_plot.py

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
This module contains some common routines for plotting
'''

import matplotlib.pyplot as plt
import numpy as np

def add_subplot_axes(ax, rect, axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x, y, width, height], axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2] ** 0.5
    y_labelsize *= rect[3] ** 0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

# Visvis related:
shot_counter = 0
filename_prefix = "screenshot_"
file_format = "png"

def OnKey(event):
    """ Called when a key is pressed down in the axes.
    """
    import visvis as vv
    global shot_counter, filename_prefix, file_format

    if event.text and event.text.lower() in 's':
        current_filename = "%s-%02d.%s" % (filename_prefix, shot_counter, file_format)
        print("Saving screenshot as %s" % (current_filename))
        vv.screenshot(filename=current_filename, bg="w", sf=3, format=file_format)
        shot_counter += 1

def draw_axes_visvis(length, line_thickness=1, line_style="-", z_origin_offset=0):
    '''
    @note: We are using Line directly, but it doesn't work with transparency as opposed to using the "solidLine" function generator (They both have their pros and cons)
    @param line_style: Possible line styles (ls) are:
          * Solid line: '-'
          * Dotted line: ':'
          * Dashed line: '--'
          * Dash-dot line: '-.' or '.-'
          * A line that is drawn between each pair of points: '+'
          * No line: '' or None.
    '''
    # make simple, bare axis lines through space:
    import visvis as vv

    a = vv.gca()
    pp_x = vv.Pointset(3)
    pp_x.append(0, 0, z_origin_offset); pp_x.append(length, 0, z_origin_offset);
#     line_x = vv.solidLine(pp_x, radius=line_thickness)
#     line_x.faceColor = "r"
    line_x = vv.Line(a, pp_x)
    line_x.ls = line_style
    line_x.lw = line_thickness
    line_x.lc = "r"

    pp_y = vv.Pointset(3)
    pp_y.append(0, 0, z_origin_offset); pp_y.append(0, length, z_origin_offset);
#     line_y = vv.solidLine(pp_y, radius=line_thickness)
#     line_y.faceColor = "g"
    line_y = vv.Line(a, pp_y)
    line_y.ls = line_style
    line_y.lw = line_thickness
    line_y.lc = "g"

    pp_z = vv.Pointset(3)
    pp_z.append(0, 0, z_origin_offset); pp_z.append(0, 0, z_origin_offset + length);
#     line_z = vv.solidLine(pp_z, radius=line_thickness)
#     line_z.faceColor = "b"
    line_z = vv.Line(a, pp_z)
    line_z.ls = line_style
    line_z.lw = line_thickness
    line_z.lc = "b"


def get_plane_surface(width, height, z_offset, img_face=None):
    import visvis as vv

    xx, yy = np.meshgrid((-width / 2.0, width / 2.0), (-height / 2.0, height / 2.0))
    zz = z_offset + np.zeros_like(xx)

    if img_face:
        plane_surf = vv.surf(xx, yy, zz, img_face)
    else:
        plane_surf = vv.surf(xx, yy, zz)

    return plane_surf

def plot_marginalized_pdf(mean, sigma, n_samples, how_many_sigmas, ax_pdf, x_axis_symbol="x", units="mm"):
    from omnistereo.common_tools import pdf
    data_X = np.linspace(-how_many_sigmas * sigma, how_many_sigmas * sigma, n_samples) + mean
    cons_X = 1. / (np.sqrt(2 * np.pi) * sigma)
    pdf_X = pdf(point=data_X, cons=cons_X, mean=mean, det_sigma=sigma)
    ax_pdf.set_xlabel(r'$%s\,[\mathrm{%s}]$' % (x_axis_symbol, units))
    ax_pdf.set_ylabel(r'$\mathrm{f}_{\mu,\sigma^2}(%s)$' % (x_axis_symbol))
    ax_pdf.plot(data_X, pdf_X,)
    ax_pdf.axvline(mean, color='black', linestyle='--', label="$\mu_{\mathrm{f}_{%s}}=%0.2f \, \mathrm{%s}$" % (x_axis_symbol, mean, units))
    for s in range(how_many_sigmas):
        ax_pdf.axvline(mean + s * sigma, color='red', linestyle=':')  # , label="$%d\sigma_{\mathrm{f}_{%s}}=%0.2f$" % (s, x_axis_symbol))
        ax_pdf.axvline(mean - s * sigma, color='red', linestyle=':')  # , label="$-%d\sigma_{\mathrm{f}_{%s}}=%0.2f$" % (s, x_axis_symbol))

    ax_pdf.legend().draggable()

def vis_omnistereo_and_grids(omni_model, T_W_wrt_C_list, points_wrt_pattern, chessboard_indices=[], app=None):
    '''
    @param omni_model: A monocular GUM
    '''
    if len(chessboard_indices) == 0 or chessboard_indices == None:
        chessboard_indices = range(len(T_W_wrt_C_list))

    import visvis as vv
    if app == None:
        try:
            from PySide import QtGui, QtCore
            backend = 'pyside'
        except ImportError:
            from PyQt4 import QtGui, QtCore
            backend = 'pyqt4'

        app = vv.use(backend)

    if hasattr(omni_model, "theoretical_model"):
        z_offset, app = draw_omnistereo_model_visvis(omni_model.theoretical_model, app=app, finish_drawing=False, pt_size=5, line_thickness=1, pt_font_size=14, show_labels=True, show_only_real_focii=True, show_reference_frame=True, show_grid_box=True, busy_grid=True)

    a = vv.gca()

    for g in chessboard_indices:
        obj_pts_true = np.einsum("ij, mnj->mni", T_W_wrt_C_list[g], points_wrt_pattern[g])
        # Draw TRUE calibration grid in the world (in black)
        xx_obj_pts_true = obj_pts_true[..., 0]
        yy_obj_pts_true = obj_pts_true[..., 1]
        zz_obj_pts_true = obj_pts_true[..., 2]
        obj_pts_true_grid = vv.grid(xx_obj_pts_true, yy_obj_pts_true, zz_obj_pts_true, axesAdjust=True, axes=a)
        obj_pts_true_grid.edgeColor = "k"  # color
        obj_pts_true_grid.edgeShading = "plain"  # possible shaders: None, plain, flat, gouraud, smooth
        obj_pts_true_grid.diffuse = 0.0
        Og_pt_true = vv.Point(obj_pts_true[0, 0, 0], obj_pts_true[0, 0, 1], obj_pts_true[0, 0, 2])
        vv.plot(Og_pt_true, ms='.', mc="k", mw=5, ls='', mew=0, axesAdjust=False)

    return app

do_next_frame = False  # Global variable used during 3D Point Cloud visualization

def compute_pointclouds_simple(omnistereo_model, omni_img_filename_template=None, img_indices=[], compute_new_3D_points=True, points_3D_filename_template="3d_points-*.pkl", features_detected_filename_template="feature_correspondences-*.pkl", dense_cloud=True, manual_point_selection=False, load_stereo_tuner_from_pickle=False, save_pcl=False, pcd_cloud_path="data", stereo_tuner_filename="stereo_tuner.pkl", tune_live=False, save_sparse_features=False, load_sparse_features_from_file=False):
    '''
    This simple function doesn't transform the pose of the cloud frame with respect to the scene
    '''
    global do_next_frame
    from cv2 import waitKey  # WISHME: use something more generic in case OpenCV doesn't exist

    from omnistereo.common_tools import save_obj_in_pickle, load_obj_from_pickle
    from omnistereo.camera_models import FeatureMatcher
    from omnistereo.common_cv import get_images, get_feature_matches_data_from_files
    # 3D Visualization (Setup)
    import vispy.scene
    from vispy.scene import visuals
    from vispy import use, app
    use(app="glfw", gl="gl2")

    #
    # Make a canvas and add simple view
    #
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, title="Point Cloud")

    # Implement key presses
    @canvas.events.key_press.connect
    def on_key_press(event):
        global do_next_frame  # NOTE: Ugly need of global variable since "I think" these events cannot take arguments
        # TODO: instantiate own Canvas class (see examples) so class attributes can accessed without using global variables
        if event.text.lower() == 'n':
            do_next_frame = True

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

    # create scatter object for the first time
    scatter = visuals.Markers()
    view.add(scatter)

    if omni_img_filename_template is None:
        # Assuming the list is sorted increasingly:
        # Use only the current image in the list by align to index? Not feasible because we don't know what index the image is from
        omni_images_list = (img_indices[-1] + 1) * [omnistereo_model.current_omni_img]
    else:
        omni_images_list = get_images(omni_img_filename_template, indices_list=img_indices, show_images=True)

    units_scale_factor = 1 / 1000.  # In Meters (so XYZ axes can be shown up to scale)
    axis_length = 0.25

    if dense_cloud:
        min_disp = 3
        max_disp = 0
        if manual_point_selection:
            vis_pt_size = 10
        else:
            vis_pt_size = 5
    else:
        # Visualize inlier matches
        from omnistereo.common_cv import filter_correspondences_manually
        vis_pt_size = 10
        first_row_to_crop_bottom = 20  # <<<< SETME (TEMP) For Journal Paper (Sensors)

        detection_method = "AGAST"  # Best so far with "BRISK" as its descriptor
        matching_type = "BF"
        features_detected_filename_template = features_detected_filename_template.replace("*", detection_method + "_" + matching_type + "-*", 1)
        if load_sparse_features_from_file:
            from cv2 import KeyPoint
            feature_matches_data_list = get_feature_matches_data_from_files(features_detected_filename_template, indices_list=img_indices)
        else:
            min_disp = 1  # for v-axis
            max_u_dist = 1.10  # for u-axis in order to filter vertically aligned features
            k_best_matches = 1  # To set the number of k nearest matches in the matcher
            omnistereo_model.feature_matcher = FeatureMatcher(method=detection_method, matcher_type=matching_type, k_best=k_best_matches)  # NOTE: BRISK and ORB fail to match with FLANN
            # pixel_error_threshold = 1.2  # WRONG! there is not point since the ray should back-project to the same spot.
            # Use this for the frame to frame comparisson only
            min_range = 100  # in [mm]  <<<<<<<< SETME: ?? for new rig
            max_range = 4000  # in [mm] = 10 meters <<<<<<<< SETME: 10000 for new rig
            manual_filtering = False

        # Generate azimuthal masks on panoramas
        omnistereo_model.top_model.panorama.generate_azimuthal_masks(azimuth_mask_degrees=5, overlap_degrees=0, show=False)
        omnistereo_model.bot_model.panorama.generate_azimuthal_masks(azimuth_mask_degrees=5, overlap_degrees=0, show=False)

    # Running multiple views (as visualizing all point clouds)
    for idx in img_indices:
        omnistereo_model.set_current_omni_image(omni_images_list[idx], generate_panoramas=False, view=False, apply_pano_mask=True, mask_RGB=(0, 0, 0))  # Using Black pano mask
        points_3D_filename = points_3D_filename_template.replace("*", str(idx), 1)
        if compute_new_3D_points:
            if dense_cloud:
                omnistereo_model.get_depth_map_from_panoramas(method="sgbm", use_cropped_panoramas=False, show=True, load_stereo_tuner_from_pickle=load_stereo_tuner_from_pickle, stereo_tuner_filename=stereo_tuner_filename, tune_live=tune_live)
                # Generate 3D point cloud
                if manual_point_selection:
                    xyz_points, rgb_points = omnistereo_model.triangulate_from_clicked_points(min_disparity=min_disp, max_disparity=max_disp, use_PCL=save_pcl, export_to_pcd=False, cloud_path=pcd_cloud_path, use_LUTs=False)
                else:
                    xyz_points, rgb_points = omnistereo_model.triangulate_from_depth_map(min_disparity=min_disp, max_disparity=max_disp, use_PCL=save_pcl, export_to_pcd=True, cloud_path=pcd_cloud_path, use_LUTs=False, use_midpoint_triangulation=True)
                    # NOTE: From the following TIME analysis, we observe using the midpoint triangulation method is slightly slower than the naive approach!
                    #===========================================================
                    # from time import process_time
                    # start_time = process_time()
                    # xyz_points, rgb_points = omnistereo_model.triangulate_from_depth_map(min_disparity=min_disp, max_disparity=max_disp, use_PCL=save_pcl, export_to_pcd=True, cloud_path=pcd_cloud_path, use_LUTs=False, use_midpoint_triangulation=True)
                    # end_time = process_time()
                    # time_ellapsed_1 = end_time - start_time
                    # print("Time elapsed: {time:.8f} seconds".format(time=time_ellapsed_1))
                    # start_time = process_time()
                    # xyz_points, rgb_points = omnistereo_model.triangulate_from_depth_map(min_disparity=min_disp, max_disparity=max_disp, use_PCL=save_pcl, export_to_pcd=True, cloud_path=pcd_cloud_path, use_LUTs=False, use_midpoint_triangulation=False)
                    # end_time = process_time()
                    # time_ellapsed_2 = end_time - start_time
                    # print("Time elapsed: {time:.8f} seconds".format(time=time_ellapsed_2))
                    # print("Time DIFF: {time:.8f} seconds".format(time=(time_ellapsed_1 - time_ellapsed_2)))
                    #===========================================================
            else:
                view.children[0].children.remove(scatter)  # TEMP: Clear point makers (to visualize each frame's features only) NOT cummulative
                if load_sparse_features_from_file:
                    # Load feature data set from list
                    (matched_m_top, matched_kpts_top_serial, matched_desc_top), (matched_m_bot, matched_kpts_bot_serial, matched_desc_bot), random_colors_RGB = feature_matches_data_list[idx]
                                        # WISHME: Needs to serialize the cv2.KeyPoint before dumping them with Pickle.
                    # So represent every keypoint with a tuple:
                    matched_kpts_top = np.empty_like(matched_kpts_top_serial)
                    matched_kpts_bot = np.empty_like(matched_kpts_bot_serial)
                    num_of_point_correspondences = len(matched_kpts_top_serial)
                    print(num_of_point_correspondences, "Point Correspondences loaded from pickle.")

                    for i in range(num_of_point_correspondences):
                        k_top = matched_kpts_top_serial[i]
                        k_bot = matched_kpts_bot_serial[i]
                        matched_kpts_top[i] = KeyPoint(x=k_top[0][0], y=k_top[0][1], _size=k_top[1], _angle=k_top[2], _response=k_top[3], _octave=k_top[4], _class_id=k_top[5])
                        matched_kpts_bot[i] = KeyPoint(x=k_bot[0][0], y=k_bot[0][1], _size=k_bot[1], _angle=k_bot[2], _response=k_bot[3], _octave=k_bot[4], _class_id=k_bot[5])

                    az1, el1 = omnistereo_model.top_model.panorama.get_direction_angles_from_pixel_pano(matched_m_top, use_LUTs=False)  # FIXME: change use_LUTs to True
                    az2, el2 = omnistereo_model.bot_model.panorama.get_direction_angles_from_pixel_pano(matched_m_bot, use_LUTs=False)
                    # Get XYZ from triangulation and put into some cloud
                    xyz_points = omnistereo_model.get_triangulated_point_from_direction_angles(dir_angs_top=(az1, el1), dir_angs_bot=(az2, el2), use_midpoint_triangulation=True)
                else:  # Compute and save
                    # omni_stereo.top_model.detect_sparse_features_on_panorama()
                    (matched_m_top, matched_kpts_top, matched_desc_top), (matched_m_bot, matched_kpts_bot, matched_desc_bot), random_colors_RGB = omnistereo_model.match_features_panoramic_top_bottom(min_rectified_disparity=min_disp, max_horizontal_diff=max_u_dist, show_matches=False)
                    # Get xyz and rgb sparse points
                    # WISH: Get the floating point coordinates instead of int, so we can use precision elevation without LUTs
                    # NOTE: at the moment, angles are being resolved discretely, which can add error to the triangulation
                    az1, el1 = omnistereo_model.top_model.panorama.get_direction_angles_from_pixel_pano(matched_m_top, use_LUTs=False)  # FIXME: change use_LUTs to True
                    az2, el2 = omnistereo_model.bot_model.panorama.get_direction_angles_from_pixel_pano(matched_m_bot, use_LUTs=False)
                    # Get XYZ from triangulation and put into some cloud
                    xyz_points_initial = omnistereo_model.get_triangulated_point_from_direction_angles(dir_angs_top=(az1, el1), dir_angs_bot=(az2, el2), use_midpoint_triangulation=True)
                    # Filter outlier feature correspondences by projecting 3D points and measuring pixel norm to matched_m_top and matched_m_bot, so only pixels under a certain distance threshold remain.
                    # good_points_indices = omnistereo_model.filter_panoramic_points_due_to_reprojection_error(matched_m_top, matched_m_bot, xyz_points_initial, pixel_error_threshold=pixel_error_threshold)
                    good_points_indices = omnistereo_model.filter_panoramic_points_due_to_range(xyz_points_initial, min_3D_range=min_range, max_3D_range=max_range)
                    num_of_inliers = np.count_nonzero(good_points_indices)
                    print(num_of_inliers, "inliers.")
                    xyz_points = xyz_points_initial[good_points_indices][np.newaxis, ...]
                    matched_m_top = matched_m_top[good_points_indices][np.newaxis, ...]
                    matched_m_bot = matched_m_bot[good_points_indices][np.newaxis, ...]
                    random_colors_RGB = np.array(random_colors_RGB, dtype=tuple)[good_points_indices[0, :]]
                    # NOTE: it's safe to convert these lists to numpy arrays, but just remember they have changed!
                    matched_kpts_top = np.array(matched_kpts_top)[good_points_indices[0, :]]
                    matched_desc_top = np.array(matched_desc_top)[good_points_indices[0, :]]
                    matched_kpts_bot = np.array(matched_kpts_bot)[good_points_indices[0, :]]
                    matched_desc_bot = np.array(matched_desc_bot)[good_points_indices[0, :]]

                    if manual_filtering:  # Filter lists according to only good points
                        _, _ = filter_correspondences_manually(train_img=omnistereo_model.top_model.panorama.panoramic_img, query_img=omnistereo_model.bot_model.panorama.panoramic_img, train_kpts=matched_kpts_top, query_kpts=matched_kpts_bot, colors_RGB=random_colors_RGB, first_row_to_crop_bottom=first_row_to_crop_bottom, do_filtering=False)
                        waitKey(0)  # TEMP: just to visualize initial point filtering
                        # Filter again:
                        valid_match_indices, matches_img = filter_correspondences_manually(train_img=omnistereo_model.top_model.panorama.panoramic_img, query_img=omnistereo_model.bot_model.panorama.panoramic_img, train_kpts=matched_kpts_top, query_kpts=matched_kpts_bot, colors_RGB=random_colors_RGB, first_row_to_crop_bottom=first_row_to_crop_bottom, do_filtering=True)
                        random_colors_RGB = random_colors_RGB[valid_match_indices]
                        matched_kpts_top = matched_kpts_top[valid_match_indices]
                        matched_desc_top = matched_desc_top[valid_match_indices]
                        matched_kpts_bot = matched_kpts_bot[valid_match_indices]
                        matched_desc_bot = matched_desc_bot[valid_match_indices]
                        xyz_points = xyz_points[0, valid_match_indices][np.newaxis, ...]
                        matched_m_top = matched_m_top[0, valid_match_indices][np.newaxis, ...]
                        matched_m_bot = matched_m_bot[0, valid_match_indices][np.newaxis, ...]
                        num_of_inliers = np.count_nonzero(valid_match_indices)
                        print(num_of_inliers, "inliers after manual filtering.")

                        # and filter again (just in case):
                        valid_match_indices_refined, matches_img = filter_correspondences_manually(train_img=omnistereo_model.top_model.panorama.panoramic_img, query_img=omnistereo_model.bot_model.panorama.panoramic_img, train_kpts=matched_kpts_top, query_kpts=matched_kpts_bot, colors_RGB=random_colors_RGB, first_row_to_crop_bottom=first_row_to_crop_bottom, do_filtering=True)
                        random_colors_RGB = random_colors_RGB[valid_match_indices_refined]
                        matched_kpts_top = matched_kpts_top[valid_match_indices_refined]
                        matched_desc_top = matched_desc_top[valid_match_indices_refined]
                        matched_kpts_bot = matched_kpts_bot[valid_match_indices_refined]
                        matched_desc_bot = matched_desc_bot[valid_match_indices_refined]
                        xyz_points = xyz_points[0, valid_match_indices_refined][np.newaxis, ...]
                        matched_m_top = matched_m_top[0, valid_match_indices_refined][np.newaxis, ...]
                        matched_m_bot = matched_m_bot[0, valid_match_indices_refined][np.newaxis, ...]

                        num_of_inliers = np.count_nonzero(valid_match_indices_refined)
                        print(num_of_inliers, "inliers (after second manual refinement).")

                    # Save feature data set to pickle
                    features_data_filename = features_detected_filename_template.replace("*", str(idx), 1)
                    # WISHME: Needs to serialize the cv2.KeyPoint before dumping them with Pickle.
                    # So represent every keypoint with a tuple:
                    matched_kpts_top_serial = np.empty_like(matched_kpts_top)
                    matched_kpts_bot_serial = np.empty_like(matched_kpts_bot)
                    for i in range(num_of_inliers):
                        k_top = matched_kpts_top[i]
                        k_bot = matched_kpts_bot[i]
                        matched_kpts_top_serial[i] = (k_top.pt, k_top.size, k_top.angle, k_top.response, k_top.octave, k_top.class_id)
                        matched_kpts_bot_serial[i] = (k_bot.pt, k_bot.size, k_bot.angle, k_bot.response, k_bot.octave, k_bot.class_id)

                    save_obj_in_pickle([(matched_m_top, matched_kpts_top_serial, matched_desc_top), (matched_m_bot, matched_kpts_bot_serial, matched_desc_bot), random_colors_RGB], features_data_filename, locals())

                # Just show the resulting matches:
                _, _ = filter_correspondences_manually(train_img=omnistereo_model.top_model.panorama.panoramic_img, query_img=omnistereo_model.bot_model.panorama.panoramic_img, train_kpts=matched_kpts_top, query_kpts=matched_kpts_bot, colors_RGB=random_colors_RGB, first_row_to_crop_bottom=first_row_to_crop_bottom, do_filtering=False)
                points_3D, rgb_points = omnistereo_model.generate_point_clouds(xyz_points, matched_m_top, rgb_colors=random_colors_RGB, use_PCL=save_pcl, export_to_pcd=save_pcl, cloud_path=pcd_cloud_path + "/sparse")
                xyz_points = points_3D[..., :3]  # In case they are homogeneous
                # TODO: Track inliers on second frame

            if save_sparse_features:
                save_obj_in_pickle([xyz_points, rgb_points], points_3D_filename, locals())

        else:
            [xyz_points, rgb_points] = load_obj_from_pickle(points_3D_filename)


        #===============================================================================
        # 3D Visualization (Continued)

        # Points data
        # Transform point positions wrt Scene (reference frame)
        xyz_points_nonhomo = xyz_points[0, ...]
        points_wrt_C = np.hstack((xyz_points_nonhomo, np.ones(shape=(xyz_points_nonhomo.shape[0], 1))))  # Make homogeneous point coordinates

        pts_pos = points_wrt_C[:, :3] * units_scale_factor
        # fill in the point-cloud data
        pts_colors = np.hstack((rgb_points / 255., np.ones_like(rgb_points[..., 0, np.newaxis])))  # Adding alpha=1 channel
        scatter.set_data(pts_pos, edge_color=None, face_color=pts_colors, size=vis_pt_size)
        view.add(scatter)

        while not do_next_frame:
            app.process_events()
            canvas.update()
            waitKey(10)  # From OpenCV (cv2 module)

        print("DONE with", idx)
        do_next_frame = False  # Update global variable


def compute_pointclouds(omnistereo_model, poses_filename=None, omni_img_filename_template=None, img_indices=[], compute_new_3D_points=True, points_3D_filename_template="3d_points-*.pkl", features_detected_filename_template="feature_correspondences-*.pkl", dense_cloud=True, manual_point_selection=False, show_3D_reference_cyl=False, load_stereo_tuner_from_pickle=False, save_pcl=False, pcd_cloud_path="data", stereo_tuner_filename="stereo_tuner.pkl", tune_live=False, save_sparse_features=False, load_sparse_features_from_file=False):
    global do_next_frame
    from omnistereo import common_tools
    from omnistereo.camera_models import FeatureMatcher
    from cv2 import waitKey  # WISHME: use something more generic in case OpenCV doesn't exist
    from omnistereo.common_cv import get_images, get_feature_matches_data_from_files
    # 3D Visualization (Setup)
    import vispy.scene
    from vispy.scene import visuals
    from vispy import use, app
    use(app="glfw", gl="gl2")

    #
    # Make a canvas and add simple view
    #
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, title="Point Cloud")

    # Implement key presses
    @canvas.events.key_press.connect
    def on_key_press(event):
        global do_next_frame  # NOTE: Ugly need of global variable since "I think" these events cannot take arguments
        # TODO: instantiate own Canvas class (see examples) so class attributes can accessed without using global variables
        if event.text.lower() == 'n':
            do_next_frame = True

    view = canvas.central_widget.add_view()
    view.camera = 'arcball'  # 'turntable'  # or try 'arcball'
    # add a colored 3D axis for orientation
    axis = visuals.XYZAxis(parent=view.scene)
    view.add(axis)
    # Add grid
    grid_3D = vispy.scene.visuals.GridLines()  # color="yellow")
    view.add(grid_3D)

    import sys
    if sys.flags.interactive != 1:
        app.create()

    # create scatter object for the first time
    scatter = visuals.Markers()
    view.add(scatter)

    if omni_img_filename_template is None:
        # Assuming the list is sorted increasingly:
        # Use only the current image in the list by align to index? Not feasible because we don't know what index the image is from
        omni_images_list = (img_indices[-1] + 1) * [omnistereo_model.current_omni_img]
    else:
        omni_images_list = get_images(omni_img_filename_template, indices_list=img_indices, show_images=True)

    # Visual Odometry (Pose Estimation) test
    pano_width = np.pi * np.linalg.norm(omnistereo_model.bot_model.lowest_img_point - omnistereo_model.bot_model.precalib_params.center_point)
    # ------------------------------------------------------
    if poses_filename is None:
        pose_info = 7 * [0.0]
        # Zero translation
        pose_info[4] = 0.0
        pose_info[5] = 0.0
        pose_info[6] = 0.0
        # No rotation for Camera frame [C] pose wrt to Scene frame [S]
        [pose_info[0], pose_info[1], pose_info[2], pose_info[3]] = [1., 0., 0., 0.]
        # Fill up the result lists:
        # Use only the current image in the list by align to index? Not feasible because we don't know what index the image is from
        grid_poses_list = (img_indices[-1] + 1) * [pose_info]
        transform_matrices_list = (img_indices[-1] + 1) * [common_tools.get_transformation_matrix(pose_info)]
    else:
        grid_poses_list, transform_matrices_list = common_tools.get_poses_from_file(poses_filename=poses_filename, input_units="cm", model_working_units=omnistereo_model.units, indices=img_indices)
    # ------------------------------------------------------
    units_scale_factor = 1 / 1000.  # In Meters (so XYZ axes can be shown up to scale)
    axis_length = 0.25

    if dense_cloud:
        min_disp = 2
        max_disp = 0
        vis_pt_size = 5
    else:
        # Visualize inlier matches
        from  omnistereo_model.common_cv import filter_correspondences_manually
        vis_pt_size = 10
        first_row_to_crop_bottom = 20  # <<<< SETME (TEMP) For Journal Paper (Sensors)

        detection_method = "AGAST"  # Best so far with "BRISK" as its descriptor
        matching_type = "BF"
        features_detected_filename_template = features_detected_filename_template.replace("*", detection_method + "_" + matching_type + "-*", 1)
        if load_sparse_features_from_file:
            from cv2 import KeyPoint
            feature_matches_data_list = get_feature_matches_data_from_files(features_detected_filename_template, indices_list=img_indices)
        else:
            min_disp = 1  # for v-axis
            max_u_dist = 1.10  # for u-axis in order to filter vertically aligned features
            k_best_matches = 1  # To set the number of k nearest matches in the matcher
            omnistereo_model.feature_matcher = FeatureMatcher(method=detection_method, matcher_type=matching_type, k_best=k_best_matches)  # NOTE: BRISK and ORB fail to match with FLANN
            # pixel_error_threshold = 1.2  # WRONG! there is not point since the ray should back-project to the same spot.
            # Use this for the frame to frame comparisson only
            min_range = 100  # in [mm]  <<<<<<<< SETME: ?? for new rig
            max_range = 4000  # in [mm] = 10 meters <<<<<<<< SETME: 10000 for new rig
            manual_filtering = False

        # Generate azimuthal masks on panoramas
        omnistereo_model.top_model.panorama.generate_azimuthal_masks(azimuth_mask_degrees=5, overlap_degrees=0, show=False)
        omnistereo_model.bot_model.panorama.generate_azimuthal_masks(azimuth_mask_degrees=5, overlap_degrees=0, show=False)

    if dense_cloud == False:
        last_key_pts_top = None
        last_desc_top = None
        last_m_top = None

    # Running multiple views (as visualizing all point clouds)
    for idx in img_indices:
        # TODO: speed up this setting?
        omnistereo_model.set_current_omni_image(omni_images_list[idx], pano_width_in_pixels=pano_width, generate_panoramas=True, view=False, apply_pano_mask=True, mask_RGB=(0, 0, 0))  # Using Black pano mask

        points_3D_filename = points_3D_filename_template.replace("*", str(idx), 1)

        if compute_new_3D_points:
            if dense_cloud:
                omnistereo_model.get_depth_map_from_panoramas(method="sgbm", use_cropped_panoramas=False, show=True, load_stereo_tuner_from_pickle=load_stereo_tuner_from_pickle, stereo_tuner_filename=stereo_tuner_filename, tune_live=tune_live)
            #     omni_stereo.get_correspondences_from_clicked_points()  # For testing disparity matches purposes
                #===========================================================================
                # Generate 3D point cloud
                if manual_point_selection:
                    xyz_points, rgb_points = omnistereo_model.triangulate_from_clicked_points(min_disparity=min_disp, max_disparity=max_disp, use_PCL=save_pcl, export_to_pcd=False, cloud_path=pcd_cloud_path, use_LUTs=False)
                else:
                    xyz_points, rgb_points = omnistereo_model.triangulate_from_depth_map(min_disparity=min_disp, max_disparity=max_disp, use_PCL=save_pcl, export_to_pcd=True, cloud_path=pcd_cloud_path, use_LUTs=False)
            else:
                view.children[0].children.remove(scatter)  # TEMP: Clear point makers (to visualize each frame's features only) NOT cummulative
                if load_sparse_features_from_file:
                    # Load feature data set from list
                    (matched_m_top, matched_kpts_top_serial, matched_desc_top), (matched_m_bot, matched_kpts_bot_serial, matched_desc_bot), random_colors_RGB = feature_matches_data_list[idx]
                                        # WISHME: Needs to serialize the cv2.KeyPoint before dumping them with Pickle.
                    # So represent every keypoint with a tuple:
                    matched_kpts_top = np.empty_like(matched_kpts_top_serial)
                    matched_kpts_bot = np.empty_like(matched_kpts_bot_serial)
                    num_of_point_correspondences = len(matched_kpts_top_serial)
                    print(num_of_point_correspondences, "Point Correspondences loaded from pickle.")

                    for i in range(num_of_point_correspondences):
                        k_top = matched_kpts_top_serial[i]
                        k_bot = matched_kpts_bot_serial[i]
                        matched_kpts_top[i] = KeyPoint(x=k_top[0][0], y=k_top[0][1], _size=k_top[1], _angle=k_top[2], _response=k_top[3], _octave=k_top[4], _class_id=k_top[5])
                        matched_kpts_bot[i] = KeyPoint(x=k_bot[0][0], y=k_bot[0][1], _size=k_bot[1], _angle=k_bot[2], _response=k_bot[3], _octave=k_bot[4], _class_id=k_bot[5])

                    az1, el1 = omnistereo_model.top_model.panorama.get_direction_angles_from_pixel_pano(matched_m_top, use_LUTs=False)  # FIXME: change use_LUTs to True
                    az2, el2 = omnistereo_model.bot_model.panorama.get_direction_angles_from_pixel_pano(matched_m_bot, use_LUTs=False)
                    # Get XYZ from triangulation and put into some cloud
                    xyz_points = omnistereo_model.get_triangulated_point_from_direction_angles(dir_angs_top=(az1, el1), dir_angs_bot=(az2, el2), use_midpoint_triangulation=True)
                else:  # Compute and save
                    # omni_stereo.top_model.detect_sparse_features_on_panorama()
                    (matched_m_top, matched_kpts_top, matched_desc_top), (matched_m_bot, matched_kpts_bot, matched_desc_bot), random_colors_RGB = omnistereo_model.match_features_panoramic_top_bottom(min_rectified_disparity=min_disp, max_horizontal_diff=max_u_dist, show_matches=False)
                    # Get xyz and rgb sparse points
                    # WISH: Get the floating point coordinates instead of int, so we can use precision elevation without LUTs
                    # NOTE: at the moment, angles are being resolved discretely, which can add error to the triangulation
                    az1, el1 = omnistereo_model.top_model.panorama.get_direction_angles_from_pixel_pano(matched_m_top, use_LUTs=False)  # FIXME: change use_LUTs to True
                    az2, el2 = omnistereo_model.bot_model.panorama.get_direction_angles_from_pixel_pano(matched_m_bot, use_LUTs=False)
                    # Get XYZ from triangulation and put into some cloud
                    xyz_points_initial = omnistereo_model.get_triangulated_point_from_direction_angles(dir_angs_top=(az1, el1), dir_angs_bot=(az2, el2), use_midpoint_triangulation=True)
                    # Filter outlier feature correspondences by projecting 3D points and measuring pixel norm to matched_m_top and matched_m_bot, so only pixels under a certain distance threshold remain.
                    # good_points_indices = omnistereo_model.filter_panoramic_points_due_to_reprojection_error(matched_m_top, matched_m_bot, xyz_points_initial, pixel_error_threshold=pixel_error_threshold)
                    good_points_indices = omnistereo_model.filter_panoramic_points_due_to_range(xyz_points_initial, min_3D_range=min_range, max_3D_range=max_range)
                    num_of_inliers = np.count_nonzero(good_points_indices)
                    print(num_of_inliers, "inliers.")
                    xyz_points = xyz_points_initial[good_points_indices][np.newaxis, ...]
                    matched_m_top = matched_m_top[good_points_indices][np.newaxis, ...]
                    matched_m_bot = matched_m_bot[good_points_indices][np.newaxis, ...]
                    random_colors_RGB = np.array(random_colors_RGB, dtype=tuple)[good_points_indices[0, :]]
                    # NOTE: it's safe to convert these lists to numpy arrays, but just remember they have changed!
                    matched_kpts_top = np.array(matched_kpts_top)[good_points_indices[0, :]]
                    matched_desc_top = np.array(matched_desc_top)[good_points_indices[0, :]]
                    matched_kpts_bot = np.array(matched_kpts_bot)[good_points_indices[0, :]]
                    matched_desc_bot = np.array(matched_desc_bot)[good_points_indices[0, :]]

                    if manual_filtering:  # Filter lists according to only good points
                        _, _ = filter_correspondences_manually(train_img=omnistereo_model.top_model.panorama.panoramic_img, query_img=omnistereo_model.bot_model.panorama.panoramic_img, train_kpts=matched_kpts_top, query_kpts=matched_kpts_bot, colors_RGB=random_colors_RGB, first_row_to_crop_bottom=first_row_to_crop_bottom, do_filtering=False)
                        waitKey(0)  # TEMP: just to visualize initial point filtering
                        # Filter again:
                        valid_match_indices, matches_img = filter_correspondences_manually(train_img=omnistereo_model.top_model.panorama.panoramic_img, query_img=omnistereo_model.bot_model.panorama.panoramic_img, train_kpts=matched_kpts_top, query_kpts=matched_kpts_bot, colors_RGB=random_colors_RGB, first_row_to_crop_bottom=first_row_to_crop_bottom, do_filtering=True)
                        random_colors_RGB = random_colors_RGB[valid_match_indices]
                        matched_kpts_top = matched_kpts_top[valid_match_indices]
                        matched_desc_top = matched_desc_top[valid_match_indices]
                        matched_kpts_bot = matched_kpts_bot[valid_match_indices]
                        matched_desc_bot = matched_desc_bot[valid_match_indices]
                        xyz_points = xyz_points[0, valid_match_indices][np.newaxis, ...]
                        matched_m_top = matched_m_top[0, valid_match_indices][np.newaxis, ...]
                        matched_m_bot = matched_m_bot[0, valid_match_indices][np.newaxis, ...]
                        num_of_inliers = np.count_nonzero(valid_match_indices)
                        print(num_of_inliers, "inliers after manual filtering.")

                        # and filter again (just in case):
                        valid_match_indices_refined, matches_img = filter_correspondences_manually(train_img=omnistereo_model.top_model.panorama.panoramic_img, query_img=omnistereo_model.bot_model.panorama.panoramic_img, train_kpts=matched_kpts_top, query_kpts=matched_kpts_bot, colors_RGB=random_colors_RGB, first_row_to_crop_bottom=first_row_to_crop_bottom, do_filtering=True)
                        random_colors_RGB = random_colors_RGB[valid_match_indices_refined]
                        matched_kpts_top = matched_kpts_top[valid_match_indices_refined]
                        matched_desc_top = matched_desc_top[valid_match_indices_refined]
                        matched_kpts_bot = matched_kpts_bot[valid_match_indices_refined]
                        matched_desc_bot = matched_desc_bot[valid_match_indices_refined]
                        xyz_points = xyz_points[0, valid_match_indices_refined][np.newaxis, ...]
                        matched_m_top = matched_m_top[0, valid_match_indices_refined][np.newaxis, ...]
                        matched_m_bot = matched_m_bot[0, valid_match_indices_refined][np.newaxis, ...]

                        num_of_inliers = np.count_nonzero(valid_match_indices_refined)
                        print(num_of_inliers, "inliers (after second manual refinement).")

                    # Save feature data set to pickle
                    features_data_filename = features_detected_filename_template.replace("*", str(idx), 1)
                    # WISHME: Needs to serialize the cv2.KeyPoint before dumping them with Pickle.
                    # So represent every keypoint with a tuple:
                    matched_kpts_top_serial = np.empty_like(matched_kpts_top)
                    matched_kpts_bot_serial = np.empty_like(matched_kpts_bot)
                    for i in range(num_of_inliers):
                        k_top = matched_kpts_top[i]
                        k_bot = matched_kpts_bot[i]
                        matched_kpts_top_serial[i] = (k_top.pt, k_top.size, k_top.angle, k_top.response, k_top.octave, k_top.class_id)
                        matched_kpts_bot_serial[i] = (k_bot.pt, k_bot.size, k_bot.angle, k_bot.response, k_bot.octave, k_bot.class_id)

                    common_tools.save_obj_in_pickle([(matched_m_top, matched_kpts_top_serial, matched_desc_top), (matched_m_bot, matched_kpts_bot_serial, matched_desc_bot), random_colors_RGB], features_data_filename, locals())

                # Just show the resulting matches:
                _, _ = filter_correspondences_manually(train_img=omnistereo_model.top_model.panorama.panoramic_img, query_img=omnistereo_model.bot_model.panorama.panoramic_img, train_kpts=matched_kpts_top, query_kpts=matched_kpts_bot, colors_RGB=random_colors_RGB, first_row_to_crop_bottom=first_row_to_crop_bottom, do_filtering=False)
                points_3D, rgb_points = omnistereo_model.generate_point_clouds(xyz_points, matched_m_top, rgb_colors=random_colors_RGB, use_PCL=save_pcl, export_to_pcd=save_pcl, cloud_path=pcd_cloud_path + "/sparse")
                xyz_points = points_3D  # In case they are homogeneous
                # TODO: Track inliers on second frame

                if save_sparse_features:
                    common_tools.save_obj_in_pickle([xyz_points, rgb_points], points_3D_filename, locals())

                if last_desc_top is not None:
                    # Perform match between top panoramic images from the current time frame and previous frame.
                    pass

                last_key_pts_top = matched_kpts_top
                last_desc_top = matched_desc_top
                last_m_top = matched_m_top

        else:
            [xyz_points, rgb_points] = common_tools.load_obj_from_pickle(points_3D_filename)


        #===============================================================================
        # 3D Visualization (Continued)

        # Points data
        # Transform point positions wrt Scene (reference frame)
        # xyz_points_nonhomo = xyz_points[0, ...]
        # points_wrt_C = np.hstack((xyz_points_nonhomo, np.ones(shape=(xyz_points_nonhomo.shape[0], 1))))  # Make homogeneous point coordinates
        points_wrt_C = xyz_points[0, ...]
        points_wrt_S = np.einsum("ij, nj->ni", transform_matrices_list[idx], points_wrt_C)

        pts_pos = points_wrt_S[:, :3] * units_scale_factor
        # fill in the point-cloud data
        pts_colors = np.hstack((rgb_points / 255., np.ones_like(rgb_points[..., 0, np.newaxis])))  # Adding alpha=1 channel
        scatter.set_data(pts_pos, edge_color=None, face_color=pts_colors, size=vis_pt_size)
        view.add(scatter)

        if show_3D_reference_cyl:
            import cv2
#             cyl_tube = visuals.Tube()
#             view.add(cyl_tube)
            cyl_scatter = visuals.Markers()
            cyl_pts_pos = omnistereo_model.top_model.panorama.get_points_on_cylinder(radius_cyl_pan=1).astype('float32').reshape(-1, 3)  # * units_scale_factor
            cyl_RGB_colors = cv2.cvtColor(omnistereo_model.top_model.panorama.panoramic_img, cv2.COLOR_BGR2RGB).astype('float32').reshape(-1, 3)
            # create scatter object and fill in the data
            cyl_pts_colors = np.hstack((cyl_RGB_colors / 255., np.ones_like(cyl_RGB_colors[..., 0, np.newaxis])))  # Adding alpha=1 channel
            cyl_scatter.set_data(cyl_pts_pos, edge_color=None, face_color=cyl_pts_colors, size=5)
            view.add(cyl_scatter)

        # Add axis of camera to visualize pose changes
        pose_vector = grid_poses_list[idx]
        orig_coords = np.array(pose_vector[4:]) * units_scale_factor
        axis_end_pts_homo = np.array([[axis_length / units_scale_factor, 0, 0], [0, axis_length / units_scale_factor, 0], [0, 0, axis_length / units_scale_factor], [1, 1, 1]])  # Column vectors
        axis_end_pts_transformed = np.dot(transform_matrices_list[idx], axis_end_pts_homo)[:-1] * units_scale_factor
        # TODO: Apply axis rotation
        pose_frame_axis = np.array([
                                    orig_coords,  # Orig
                                    axis_end_pts_transformed[:, 0],  # X
                                    orig_coords,  # Orig
                                    axis_end_pts_transformed[:, 1],  # Y
                                    orig_coords,  # Orig
                                    axis_end_pts_transformed[:, 2]  # Z
                                    ])
        rig_axis = visuals.XYZAxis()
        rig_axis.set_data(pos=pose_frame_axis, width=5)
        view.add(rig_axis)

        while not do_next_frame:
            app.process_events()
            canvas.update()
            waitKey(10)  # From OpenCV (cv2 module)

        print("DONE with", idx)
        do_next_frame = False  # Update global variable



def draw_model_mono_visvis(theoretical_omni_model, app=None, finish_drawing=True, pt_size=5, line_thickness=1, pt_font_size=14, mirror_transparency=0.9, show_labels=True, show_focii=True, show_only_real_focii=False, show_reference_frame=True, show_grid_box=True, busy_grid=True):
    # FIXME: there is coloring problem in visvis for custom surfaces throught the XY-plane (I'm avoiding so with a Z-offset)
    '''
    @param app: The object representing the GUI backen application (if any has been instantiated already)
    @param finish_drawing: To indicate when the main loop of vispy's App can run (scene is updated through this App)
    '''
    # params were:
    # NOTE: I perform a trick by offsetting the z-values since there is a coloring problem for any custom "surf" object traversing the XY-plane at zero
    import visvis as vv
    if app == None:
        try:
            from PySide import QtGui, QtCore
            backend = 'pyside'
        except ImportError:
            from PyQt4 import QtGui, QtCore
            backend = 'pyqt4'

        app = vv.use(backend)
            # Create time and enter main loop
    a = vv.gca()
    # a.eventKeyDown.Bind(OnKey) # FIXME: This is redundant when writing over the current figure

    # Prepare
    # set labels
    if show_grid_box:
        vv.xlabel('X_C [mm]')
        vv.ylabel('Y_C [mm]')
        vv.zlabel('Z_C [mm]')

    scale = 1.0
    diffuse_level = .9
    specular_level = 0.9
    z_offset = 0

    # Computer np arrays of coordinates for the hyperbolic mirrors:
    hyper_xx, hyper_yy, hyper_zz = theoretical_omni_model.get_surf_points(scale)
    if theoretical_omni_model.mirror_number == 1:
        x_reflex, y_reflex, z_reflex = theoretical_omni_model.get_reflex_surf_points(scale)
#     else:
#         # Trick for bottom color problem for a "surf" object traversing the XY-plane at zero
#         if show_grid_box == False:
#             z_offset = np.abs(hyper_zz.min()) + 1

    # Draw Points
    if show_focii:
        # Simpler variable names:
        F_pt = vv.Point(theoretical_omni_model.F[0], theoretical_omni_model.F[1], theoretical_omni_model.F[2] + z_offset)
        pt_F = vv.plot(F_pt, ms='.', mc="b", mw=pt_size, ls='', mew=0, axesAdjust=False)
        if not show_only_real_focii:
            Oc_pt = vv.Point(0, 0, 0 + z_offset)
            pt_Oc = vv.plot(Oc_pt, ms='.', mc="k", mw=pt_size, ls='', mew=0, axesAdjust=False)

        if show_labels:
            txt_F = vv.Text(a, text='F', x=theoretical_omni_model.F[0] + 5, y=theoretical_omni_model.F[1], z=theoretical_omni_model.F[2] + z_offset, fontName="mono", fontSize=pt_font_size, color="b")
            if not show_only_real_focii:
                txt_Fv = vv.Text(a, text="F_v'", x=theoretical_omni_model.Fv[0] + 5, y=theoretical_omni_model.Fv[1], z=theoretical_omni_model.Fv[2] + z_offset, fontName="mono", fontSize=pt_font_size, color="m")

    if show_reference_frame:
        draw_axes_visvis(2 * theoretical_omni_model.r_max, line_thickness, '--', 0 + z_offset)

    # Draw mirrors:
    # Set Color for mirrors
    # cmaps = [vv.CM_GRAY, vv.CM_JET, vv.CM_SUMMER, vv.CM_HOT]
    color_rgb = [0.8, 0.8, 0.8]  # Grey
    color_alpha = [mirror_transparency]
    color_mirror = tuple(color_rgb + color_alpha)

    top_surf = vv.surf(hyper_xx, hyper_yy, hyper_zz + z_offset)
    top_surf.faceShading = "smooth"
    top_surf.faceColor = color_mirror
    top_surf.diffuse = diffuse_level
    top_surf.specular = specular_level

    if theoretical_omni_model.mirror_number == 1:
        reflex_surf = vv.surf(x_reflex, y_reflex, z_reflex + z_offset)
        reflex_surf.faceShading = "smooth"
        reflex_surf.faceColor = color_mirror
        reflex_surf.diffuse = diffuse_level
        reflex_surf.specular = specular_level / 10


    # Modifying lights
    # The other lights are off by default and are positioned at the origin
#         light0 = a.lights[0]
#         light0.On()
#         light0.ambient = 1.0  # 0.0 is default for other lights
#         light0.position = (0, 0, -10, 1)
#         light1 = a.lights[1]
#         light1.On()
#         light1.ambient = 0.0  # 0.0 is default for other lights
# #         light1.color = (1, 0, 0)  # this light is red
#         light1.position = (50, 50, 100, 0)
#         # If the fourth element is a 1, the light
#         # has a position, if it is a 0, it represents a direction (i.o.w. the
#         # light is a directional light, like the sun).

    a.axis.visible = show_grid_box
    a.axis.showGrid = busy_grid and show_grid_box
    a.axis.showGridX = show_grid_box
    a.axis.showGridY = busy_grid and show_grid_box
    a.axis.showGridZ = busy_grid and show_grid_box
    a.axis.ShowBorder = show_grid_box


    if finish_drawing:
        # Start app
        app.Run()

    return z_offset, app

def draw_frame_poses(pose_list):
    '''
    @param pose_list: The list of relative poses with respect to the common (global or world) frame
    '''
    global do_next_frame
    from cv2 import waitKey  # WISH: use something more generic in case OpenCV doesn't exist
    import omnistereo.transformations as tr
    # 3D Visualization (Setup)
    import vispy.scene
    from vispy.scene import visuals
    from vispy import use, app
    use(app="glfw", gl="gl2")

    #
    # Make a canvas and add simple view
    #
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, title="Frames", bgcolor="white")

    # Implement key presses
    @canvas.events.key_press.connect
    def on_key_press(event):
        global do_next_frame  # NOTE: Ugly need of global variable since "I think" these events cannot take arguments
        # TODO: instantiate own Canvas class (see examples) so class attributes can accessed without using global variables
        if event.text.lower() == 'n':
            do_next_frame = True

    view = canvas.central_widget.add_view()
    view.camera = 'arcball'  # 'turntable'  # or try 'arcball'
    # add a colored 3D axis for orientation
    axis = visuals.XYZAxis(parent=view.scene)
    view.add(axis)
    # Add grid
    grid_3D = vispy.scene.visuals.GridLines()  # color="blue", scale=(1, 1))  # NOT WORKING in any other bgcolor but "black"
    view.add(grid_3D)

    import sys
    if sys.flags.interactive != 1:
        app.create()

    # ------------------------------------------------------
    units_scale_factor = 1 / 1000.  # From mm to meters (so XYZ axes can be shown up to scale)
    axis_length = 100  # mm

    # Running multiple views (as visualizing all point clouds)
    for T in pose_list:
        #===============================================================================
        # 3D Visualization (Continued)
        scale, shear, angles, trans, persp = tr.decompose_matrix(T)
        orig_coords = np.array(trans) * units_scale_factor
        axis_end_pts_homo = np.array([[axis_length , 0, 0], [0, axis_length, 0], [0, 0, axis_length], [1, 1, 1]])  # Column vectors
        axis_end_pts_transformed = np.dot(T, axis_end_pts_homo)[:-1] * units_scale_factor
        # TODO: Apply axis rotation???
        pose_frame_axis = np.array([
                                    orig_coords,  # Orig
                                    axis_end_pts_transformed[:, 0],  # X
                                    orig_coords,  # Orig
                                    axis_end_pts_transformed[:, 1],  # Y
                                    orig_coords,  # Orig
                                    axis_end_pts_transformed[:, 2]  # Z
                                    ])
        rig_axis = visuals.XYZAxis()
        rig_axis.set_data(pos=pose_frame_axis, width=2)
        view.add(rig_axis)

        while not do_next_frame:
            app.process_events()
            canvas.update()
            waitKey(10)  # From OpenCV (cv2 module)

        do_next_frame = False  # Update global variable


def draw_omnistereo_model_visvis(theoretical_omnistereo_model, app=None, finish_drawing=True, pt_size=5, line_thickness=1, pt_font_size=14, mirror_transparency=0.6, show_labels=True, show_only_real_focii=False, show_reference_frame=True, show_grid_box=True, busy_grid=True, backend=None):
    '''
    @note: I perform a trick by offsetting the z-values since there is a coloring problem for any custom "surf" object traversing the XY-plane at zero

    @param app: The object representing the GUI backen application (if any has been instantiated already)
    @param finish_drawing: To indicate when the main loop of vispy's App can run (scene is updated through this App)
    '''
    # NOTE: there is coloring problem in visvis for custom surfaces throught the XY-plane (I'm avoiding so with a Z-offset)
    import visvis as vv

    if app == None:
        if backend is None:
            try:
                from PySide import QtGui, QtCore
                backend = 'pyside'
            except ImportError:
                from PyQt4 import QtGui, QtCore
                backend = 'pyqt4'
        app = vv.use(backend)

    z_offset, app = draw_model_mono_visvis(theoretical_omni_model=theoretical_omnistereo_model.top_model, app=app, finish_drawing=False, pt_size=pt_size, line_thickness=line_thickness, pt_font_size=pt_font_size, mirror_transparency=mirror_transparency, show_labels=show_labels, show_focii=False, show_only_real_focii=show_only_real_focii, show_reference_frame=show_reference_frame, show_grid_box=show_grid_box, busy_grid=busy_grid)
    z_offset, app = draw_model_mono_visvis(theoretical_omni_model=theoretical_omnistereo_model.bot_model, app=app, finish_drawing=False, pt_size=pt_size, line_thickness=line_thickness, pt_font_size=pt_font_size, mirror_transparency=mirror_transparency, show_labels=show_labels, show_focii=False, show_only_real_focii=show_only_real_focii, show_reference_frame=show_reference_frame, show_grid_box=show_grid_box, busy_grid=busy_grid)
    a = vv.gca()
    # a.eventKeyDown.Bind(OnKey) # FIXME: This is redundant when writing over the current figure

#     # Trick for bottom color problem for a "surf" object traversing the XY-plane at zero
#     if show_grid_box == False:
#         z_offset = np.abs(hyper_z2.min()) + 1
#
    # Draw Points
    # Simpler variable names:
    Oc = theoretical_omnistereo_model.Oc
    F1 = theoretical_omnistereo_model.F1
    F2 = theoretical_omnistereo_model.F2
    F2v = theoretical_omnistereo_model.F2v

    F1_pt = vv.Point(F1[0], F1[1], F1[2] + z_offset)
    pt_F1 = vv.plot(F1_pt, ms='.', mc="b", mw=pt_size, ls='', mew=0, axesAdjust=False)
    F2_pt = vv.Point(F2[0], F2[1], F2[2] + z_offset)
    pt_F2 = vv.plot(F2_pt, ms='.', mc="r", mw=pt_size, ls='', mew=0, axesAdjust=False)
    if not show_only_real_focii:
        Oc_pt = vv.Point(Oc[0], Oc[1], Oc[2] + z_offset)
        pt_Oc = vv.plot(Oc_pt, ms='.', mc="k", mw=pt_size, ls='', mew=0, axesAdjust=False)
        F2v_pt = vv.Point(F2v[0], F2v[1], F2v[2] + z_offset)
        pt_F2v = vv.plot(F2v_pt, ms='.', mc="m", mw=pt_size, ls='', mew=0, axesAdjust=False)

    if show_labels:
        txt_F1 = vv.Text(a, text='F_1', x=F1[0] + 5, y=F1[1], z=F1[2] + z_offset, fontName="mono", fontSize=pt_font_size, color="b")
        txt_F2 = vv.Text(a, text='F_2', x=F2[0] + 5, y=F2[1], z=F2[2] + z_offset, fontName="mono", fontSize=pt_font_size, color="r")
        if not show_only_real_focii:
            txt_Oc = vv.Text(a, text='O_c', x=Oc[0] + 5, y=Oc[1], z=Oc[2] + z_offset, fontName="mono", fontSize=pt_font_size, color="k")
            txt_F2v = vv.Text(a, text="F_2'", x=F2v[0] + 5, y=F2v[1], z=F2v[2] + z_offset, fontName="mono", fontSize=pt_font_size, color="m")

    if finish_drawing:
        # Start app
        app.Run()

    return z_offset, app

def draw_bak_projection_visvis(omnistereo_model, m1, m2, number_of_std_deviations=1, draw_covariance=False, line_thickness=1, show_grid_box=False, show_labels=False, plot_density_function=False):
    '''
    Draws back projecting rays and triangulation (as midpoint in the common perpendicular of the lifted direction rays)
    @param m1: pixel position from top mirror image (in homogenous coordinates)
    @param m2: pixel position from top mirror image (in homogeneous coordinates)
    @param number_of_std_deviations: the number of sigmas from the covariance matrix to be drawn as an uncertainty ellipsoid

    @note: For transparecy to work as expected, the order of drawing is very important! Otherwise, the alpha won't work for anything that's drawn later.
    For example, lines should be drawn "before" the transparent solids so they appear through the solid
    '''
    from omnistereo import covariance_ellipsoid
    from omnistereo.cata_hyper_symbolic import load_marshalled_func_from_file
    import visvis as vv

    if draw_covariance:
        filename_jacobian_func = "data/omnistereo_jacobian_marshalled_func.pkl"
        jac_func = load_marshalled_func_from_file(filename_jacobian_func)
        omnistereo_model.set_pixel_coordinates_covariance(1)
#         mid_Pw_func_from_dirs = omnistereo_model.lambdify_mid_Pw(omnistereo_sym)
#         mid_Pw_func_from_dirs_expanded = omnistereo_model.lambdify_mid_Pw_expanded(omnistereo_sym)

    corner_detection_std_dev = 1.  # pixel error on feature detection (RMSE from experimets)
    proj_plane_scale = 0
    pt_size = 5
    pt_font_size = 14

    try:
        from PySide import QtGui, QtCore
        backend = 'pyside'
    except ImportError:
        from PyQt4 import QtGui, QtCore
        backend = 'pyqt4'

    app = vv.use(backend)

    a = vv.gca()
    # Axis commands
    vv.axis("equal")  # make a circle be displayed circular
    vv.axis("tight")  #  show all data


    z_offset = 0
    if show_grid_box == False:
        # Trick for bottom color problem for a "surf" object traversing the XY-plane at zero
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        hyper_xx, hyper_yy, hyper_z2 = omnistereo_model.bot_model.get_surf_points()
        # Computer np arrays of coordinates for the hyperbolic mirrors:
        z_offset = np.abs(hyper_z2.min()) + 1
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    pt_F1 = vv.Point(omnistereo_model.F1[0, 0], omnistereo_model.F1[1, 0], omnistereo_model.F1[2, 0] + z_offset)
    pt_F2 = vv.Point(omnistereo_model.F2[0, 0], omnistereo_model.F2[1, 0], omnistereo_model.F2[2, 0] + z_offset)
    pt_Oc = vv.Point(omnistereo_model.Oc[0, 0], omnistereo_model.Oc[1, 0], omnistereo_model.Oc[2, 0] + z_offset)
    mirror1 = omnistereo_model.top_model
    mirror2 = omnistereo_model.bot_model

    # Make sure v1 and v2 are in 3-vector form (not in homogeneous form!)
    v1 = mirror1.lift_pixel_to_unit_sphere_wrt_focus(m1)[..., :3]
    v2 = mirror2.lift_pixel_to_unit_sphere_wrt_focus(m2)[..., :3]

    # Set uncertainty ellipsoid color
    color_rgb = [0., 1., 0.]
    color_alpha = [0.5]
    color_ellipsoid = tuple(color_rgb + color_alpha)

    if plot_density_function:
        fig, ax_pdf = plt.subplots(nrows=m1.shape[0], ncols=m1.shape[1])
        fig.tight_layout()
        if isinstance(ax_pdf, np.ndarray):
            if ax_pdf.ndim == 1:
                ax_pdf = ax_pdf.reshape(m1.shape[0], m1.shape[1])
        else:
            ax_pdf = np.array([[ax_pdf]])

    for row in range(m1.shape[0]):
        pt_Pwz_offset = 0.25
        for col in range(m1.shape[1]):
            mid_Pw, (lambda1, lambda2, lambda_perp), (G1, G2), (normal_unit, perp_mag) = omnistereo_model.get_triangulated_midpoint(v1[row, col][np.newaxis, np.newaxis, ...], v2[row, col][np.newaxis, np.newaxis, ...])
            # Draw real projected point on the camera plane (Not using Q, but something different for visualization)
            # Create geometric 3D planes with Euclid:
            g1x, g1y, g1z = G1[..., 0], G1[..., 1], G1[..., 2] + z_offset
            pt_G1 = vv.Point(g1x, g1y, g1z)
            plotted_G1 = vv.plot(pt_G1, ms='.', mc="k", mw=pt_size, ls='', mew=0, axesAdjust=True)
            pp_F1 = vv.Pointset(3)
            pp_F1.append(pt_G1); pp_F1.append(pt_F1);
            line_G1 = vv.solidLine(pp_F1, radius=line_thickness / 4)
            line_G1.faceColor = "b"
            line_G1.faceShading = "plain"

            g2x, g2y, g2z = G2[..., 0], G2[..., 1], G2[..., 2] + z_offset
            pt_G2 = vv.Point(g2x, g2y, g2z)
            plotted_G2 = vv.plot(pt_G2, ms='.', mc="k", mw=pt_size, ls='', mew=0, axesAdjust=True)
            pp_F2 = vv.Pointset(3)
            pp_F2.append(pt_G2); pp_F2.append(pt_F2);
            line_G2 = vv.solidLine(pp_F2, radius=line_thickness / 4)
            line_G2.faceColor = "r"
            line_G2.faceShading = "plain"

            Pwx, Pwy, Pwz = mid_Pw[..., 0], mid_Pw[..., 1], mid_Pw[..., 2] + z_offset
            pt_Pw = vv.Point(Pwx, Pwy, Pwz)
            plotted_Pw = vv.plot(pt_Pw, ms='.', mc="k", mw=pt_size, ls='', mew=0, axesAdjust=True)
            if show_labels:
                txt_Pw = vv.Text(a, text='P_{w_G}', x=Pwx - 1 / pt_Pwz_offset, y=Pwy, z=Pwz + pt_Pwz_offset * pt_font_size, fontName="mono", fontSize=pt_font_size, color="k")
                pt_Pwz_offset += pt_Pwz_offset / 2

            # Draw common perpendicular vector
            pp_G1G2 = vv.Pointset(3)
            pp_G1G2.append(pt_G1); pp_G1G2.append(pt_G2);
            line_G1G2 = vv.solidLine(pp_G1G2, radius=line_thickness / 8)
            line_G1G2.faceColor = "m"
            line_G1G2.faceShading = "plain"

            if draw_covariance or plot_density_function:
                # Compute convariance matrix
                # Using the expanded Jacobian function obtained via lambdification:
                jac_matrix = np.array(jac_func(m1[row, col][0], m1[row, col][1], m2[row, col][0], m2[row, col][1], \
                                    omnistereo_model.top_model.precalib_params.f_u, omnistereo_model.top_model.precalib_params.f_v, omnistereo_model.top_model.precalib_params.skew_factor, omnistereo_model.top_model.precalib_params.u_center, omnistereo_model.top_model.precalib_params.v_center, \
                                    omnistereo_model.k1, omnistereo_model.k2, omnistereo_model.c1, omnistereo_model.c2, omnistereo_model.d))
                cov_matrix = omnistereo_model.get_covariance_matrix(jac_matrix, pixel_std_dev=corner_detection_std_dev)

            if plot_density_function:
                n_samples = 1000
                # The determinant of sigma
                mean = mid_Pw
                sigma_XYZ = np.diag(cov_matrix)
                mean_X = mid_Pw[..., 0][0]
                sigma_X = sigma_XYZ[0] ** 0.5
                plot_marginalized_pdf(mean_X, sigma_X, n_samples, 4, ax_pdf[row, col], x_axis_symbol=r"x_w", units=omnistereo_model.units)
# #                         det_sigma_XYZ = np.linalg.det(cov_matrix)
#                         cov_mat_XY = cov_matrix[0:2, 0:2]
#                         mean_XY = mean[0:2]
#                         det_sigma_XY = np.linalg.det(cov_mat_XY)
#
#                         # Plot a distribution of x, y points (Example)
#                         # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# #                         x, y, z = np.random.multivariate_normal(mean, cov_matrix, size=n_samples).T
# #                         plt.plot(x, y, 'x'); plt.axis('equal'); plt.show()
#                         # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# #                         data = np.random.multivariate_normal(mean, cov_matrix, size=n_samples)
#                         data_XY = np.random.multivariate_normal(mean_XY, cov_mat_XY, size=n_samples)
# #                         pdfXYZ = np.zeros(data.shape[0])
#                         pdfXY = np.zeros(data_XY.shape[0])
# #                         cons_XYZ = 1. / ((2 * np.pi) ** (data.shape[1] / 2.) * det_sigma ** (-0.5))
#                         cons_XY = 1. / ((2 * np.pi) ** (data_XY.shape[1] / 2.) * det_sigma_XY ** (-0.5))
# #                         X, Y, Z = np.meshgrid(data.T[0], data.T[1], data.T[2])
#                         X, Y = np.meshgrid(data_XY.T[0], data_XY.T[1])
#
# #                         zs = np.array([pdf(np.array(ponit), cons_XYZ, mean, det_sigma_XYZ) for ponit in zip(np.ravel(X), np.ravel(Y), np.ravel(Z))])
#                         zs_XY = np.array([common_tools.pdf(np.array(ponit), cons_XY, mean_XY, det_sigma_XY) for ponit in zip(np.ravel(X), np.ravel(Y))])
#                         z_XY = zs_XY.reshape(X.shape)
#
#                         from mpl_toolkits.mplot3d import Axes3D  # To be run from the command line
#                         fig = plt.figure()
#                         ax3D = fig.add_subplot(111, projection='3d', aspect="equal")
#                         ax3D.set_xlabel('X')
#                         ax3D.set_ylabel('Y')
#
#                         surf = ax3D.plot_surface(X, Y, z_XY, rstride=1, cstride=1, cmap="YlGnBu", linewidth=0, antialiased=False)

                covariance_ellipsoid.draw_error_ellipsoid_visvis(mu=mid_Pw[0, 0], covariance_matrix=cov_matrix, stdev=number_of_std_deviations, z_offset=z_offset, color_ellipsoid=color_ellipsoid, pt_size=pt_size)
#
    # Finally, draw (overlay) the barebones model
    z_offset, app = draw_omnistereo_model_visvis(omnistereo_model, app, False, proj_plane_scale, pt_size, line_thickness, pt_font_size, show_labels, show_only_real_focii=True, show_reference_frame=True, show_grid_box=show_grid_box, busy_grid=False)

    # Start app
    app.Run()

def draw_bak_projection(omnistereo_model, m1, m2, pixel_std_dev=1.0, ax=None):
    '''
    Draws back projecting rays and triangulation (as midpoint in the common perpendicular of the lifted direction rays)
    @param m1: pixel position from top mirror image (in homogenous coordinates)
    @param m2: pixel position from top mirror image (in homogeneous coordinates)
    @param pixel_std_dev: pixel error on feature detection (usually determined from experiments)

    @return the drawn axis corresponding to the parameter ax of the figure
    '''
    from omnistereo import covariance_ellipsoid
    from omnistereo.cata_hyper_symbolic import load_marshalled_func_from_file

    filename_jacobian_func = "data/omnistereo_jacobian_marshalled_func.pkl"
    jac_func = load_marshalled_func_from_file(filename_jacobian_func)
    omnistereo_model.set_pixel_coordinates_covariance(1)

    omnistereo_model.set_pixel_coordinates_covariance(pixel_std_dev)
#         mid_Pw_func_from_dirs = omnistereo_model.lambdify_mid_Pw(omnistereo_sym)
#         mid_Pw_func_from_dirs_expanded = omnistereo_model.lambdify_mid_Pw_expanded(omnistereo_sym)

    proj_plane_scale = 0
    show_labels = True
    ax = draw_omnistereo_model(omnistereo_model, ax, proj_plane_scale, show_labels)  # Draw the barebones model
    if ax == None:
        from mpl_toolkits.mplot3d import Axes3D
        # from mplot3d import axes3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', aspect="equal")
        # ax.set_aspect("equal")
        fig.tight_layout()

        # draw unit sphere
        Mtop = [0, 0, 0]  # Center position
        u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
        x = np.cos(u) * np.sin(v) + Mtop[0]
        y = np.sin(u) * np.sin(v) + Mtop[1]
        z = np.cos(v) + Mtop[2]
        ax.plot_wireframe(x, y, z, color="yellow")

    F1 = (omnistereo_model.F1[:-1].T)[np.newaxis, ...]
    Oc = (omnistereo_model.Oc[:-1].T)[np.newaxis, ...]
    F2 = (omnistereo_model.F2[:-1].T)[np.newaxis, ...]
    mirror1 = omnistereo_model.top_model
    mirror2 = omnistereo_model.bot_model

    # Make sure v1 and v2 are in 3-vector form (not in homogeneous form!)
    v1 = mirror1.lift_pixel_to_unit_sphere_wrt_focus(m1)[..., :3]
    v2 = mirror2.lift_pixel_to_unit_sphere_wrt_focus(m2)[..., :3]

    v_scale = 250
    v1_wrt_Oc = mirror1.get_point_wrt_origin(v_scale * v1)  # Put vector in the common frame
    v2_wrt_Oc = mirror2.get_point_wrt_origin(v_scale * v2)  # Put vector in the common frame

    is_Pw_mean = True

    for row in range(m1.shape[0]):
        for col in range(m1.shape[1]):
            mid_Pw, (lambda1, lambda2, lambda_perp), (G1, G2), (normal_unit, perp_mag) = omnistereo_model.get_triangulated_midpoint(v1[row, col][np.newaxis, np.newaxis, ...], v2[row, col][np.newaxis, np.newaxis, ...])

            # Test symbolic results with lambdification: (Passed!)
#                 mid_Pw_sym = mid_Pw_func_from_dirs(v_1_x=v1[row, col][0], v_1_y=v1[row, col][1], v_1_z=v1[row, col][2], v_2_x=v2[row, col][0], v_2_y=v2[row, col][1], v_2_z=v2[row, col][2], \
#                                                    c_1=omnistereo_model.c1, c_2=omnistereo_model.c2, d=omnistereo_model.d)
#                 mid_Pw_sym_expanded = mid_Pw_func_from_dirs_expanded(u_1=m1[row, col][0], v_1=m1[row, col][1], u_2=m2[row, col][0], v_2=m2[row, col][1], \
#                                                                      f_u=omnistereo_model.top_model.precalib_params.f_u, f_v=omnistereo_model.top_model.precalib_params.f_v, s=omnistereo_model.top_model.precalib_params.skew_factor, u_c=omnistereo_model.top_model.precalib_params.u_center, v_c=omnistereo_model.top_model.precalib_params.v_center, \
#                                                                      k_1=omnistereo_model.k1, k_2=omnistereo_model.k2, c_1=omnistereo_model.c1, c_2=omnistereo_model.c2, d=omnistereo_model.d)

            perp_wrt_Oc = mirror2.get_point_wrt_origin(v_scale / 10 * normal_unit)  # Put vector in the common frame

            # Exagerated
            v1x, v1y, v1z = float(v1_wrt_Oc[row, col, 0]), float(v1_wrt_Oc[row, col, 1]), float(v1_wrt_Oc[row, col, 2])
            ax.plot3D([v1x, F1[..., 0]], [v1y, F1[..., 1]], [v1z, F1[..., 2]], color="blue")  # , linestyle='-', linewidth=1.5, alpha=1.0)
            v2x, v2y, v2z = float(v2_wrt_Oc[row, col, 0]), float(v2_wrt_Oc[row, col, 1]), float(v2_wrt_Oc[row, col, 2])
            ax.plot3D([v2x, F2[..., 0]], [v2y, F2[..., 1]], [v2z, F2[..., 2]], color="red")  # , linestyle='-', linewidth=1.5, alpha=1.0)


            # Draw real projected point on the camera plane (Not using Q, but something different for visualization)
            # Create geometric 3D planes with Euclid:
#                 g1x, g1y, g1z = float(G1[row, col][0]), float(G1[row, col][1]), float(G1[row, col][2])
                        # Draw real projected point on the camera plane (Not using Q, but something different for visualization)
            # Create geometric 3D planes with Euclid:
            g1x, g1y, g1z = float(G1[..., 0]), float(G1[..., 1]), float(G1[..., 2])
            ax.scatter(g1x, g1y, g1z, color="k", s=20)
            g1_xs = [g1x, F1[..., 0]]
            g1_ys = [g1y, F1[..., 1]]
            g1_zs = [g1z, F1[..., 2]]
            ax.plot3D(g1_xs, g1_ys, g1_zs, color="blue")  # , linestyle='-', linewidth=1.5, alpha=1.0)

#                 g2x, g2y, g2z = float(G2[row, col][0]), float(G2[row, col][1]), float(G2[row, col][2])
            g2x, g2y, g2z = float(G2[..., 0]), float(G2[..., 1]), float(G2[..., 2])
            ax.scatter(g2x, g2y, g2z, color="k", s=20)
            g2_xs = [g2x, F2[..., 0]]
            g2_ys = [g2y, F2[..., 1]]
            g2_zs = [g2z, F2[..., 2]]
            ax.plot3D(g2_xs, g2_ys, g2_zs, color="red")  # , linestyle='-', linewidth=1.5, alpha=1.0)

#                 Pwx, Pwy, Pwz = float(mid_Pw[row, col][0]), float(mid_Pw[row, col][1]), float(mid_Pw[row, col][2])
            Pwx, Pwy, Pwz = float(mid_Pw[..., 0]), float(mid_Pw[..., 1]), float(mid_Pw[..., 2])
            ax.scatter(Pwx, Pwy, Pwz, color="m", s=20)
            Pw_xs = [Pwx, Oc[..., 0]]
            Pw_ys = [Pwy, Oc[..., 1]]
            Pw_zs = [Pwz, Oc[..., 2]]
            # ax.plot3D(Pw_xs, Pw_ys, Pw_zs, color="black", linestyle='-', linewidth=1.5, alpha=1.0)

            if is_Pw_mean:
                if show_labels:
                    ax.text(g1x - 5, g1y, g1z + 0.4, "G$_1$", color="blue")
                    ax.text(g2x - 5, g2y, g2z + 0.4, "G$_2$", color="red")
                    ax.text(Pwx - 5, Pwy, Pwz + 0.4, "P$_{w_G}$", color="m")
                # Draw common perpendicular vector
                G1G2_xs = [g1x, g2x]
                G1G2_ys = [g1y, g2y]
                G1G2_zs = [g1z, g2z]
                ax.plot3D(G1G2_xs, G1G2_ys, G1G2_zs, color="black", linestyle=':', linewidth=1.5, alpha=1.0)

                # Compute convariance matrix
#                     jac_matrix = np.array(jac_func(m1[row, col][0], m1[row, col][1], m2[row, col][0], m2[row, col][1], \
#                                           v1[row, col][0], v1[row, col][1], v1[row, col][2], v2[row, col][0], v2[row, col][1], v2[row, col][2], \
#                                           omnistereo_model.top_model.precalib_params.f_u, omnistereo_model.top_model.precalib_params.f_v, omnistereo_model.top_model.precalib_params.skew_factor, omnistereo_model.top_model.precalib_params.u_center, omnistereo_model.top_model.precalib_params.v_center, \
#                                           omnistereo_model.k1, omnistereo_model.k2, omnistereo_model.c1, omnistereo_model.c2, omnistereo_model.d))
                # Using the expanded Jacobian function obtained via lambdification:
                jac_matrix = np.array(jac_func(m1[row, col][0], m1[row, col][1], m2[row, col][0], m2[row, col][1], \
                                    omnistereo_model.top_model.precalib_params.f_u, omnistereo_model.top_model.precalib_params.f_v, omnistereo_model.top_model.precalib_params.skew_factor, omnistereo_model.top_model.precalib_params.u_center, omnistereo_model.top_model.precalib_params.v_center, \
                                    omnistereo_model.k1, omnistereo_model.k2, omnistereo_model.c1, omnistereo_model.c2, omnistereo_model.d))
                cov_matrix = omnistereo_model.get_covariance_matrix(jac_matrix)
                covariance_ellipsoid.draw_error_ellipsoid(mu=mid_Pw, covariance_matrix=cov_matrix, stdev=1, color="blue", ax=ax)
                is_Pw_mean = False  # Set flag

    return ax

def draw_omnistereo_model(omnistereo_model, ax=None, proj_plane_scale=20, show_labels=True):
    '''
    @param proj_plane_scale: The scale for drawin the projection plane. If 0 is passed, then planes are not drawn.
    @return the drawn axis corresponding to the parameter ax of the figure
    '''
    import matplotlib.pyplot as plt
#             import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    if ax == None:
#             from mplot3d import axes3d
#             from mplot3d import axes3d  # To be run from eclipse
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', aspect="equal")
        # ax.set_aspect("equal")
        fig.tight_layout()
    # Simpler variable names:
    F1 = omnistereo_model.F1.reshape(4)
    Oc = omnistereo_model.Oc.reshape(4)
    F2 = omnistereo_model.F2.reshape(4)
    F2v = omnistereo_model.F2v.reshape(4)
    r_sys = omnistereo_model.system_radius
    r_reflex = omnistereo_model.reflex_radius
    r_cam = omnistereo_model.camera_hole_radius
    mirror1 = omnistereo_model.top_model
    mirror2 = omnistereo_model.bot_model

    from omnistereo.common_plot import draw_axes_3D_mplot3D
    draw_axes_3D_mplot3D(2 * r_sys, ax)
    ax.set_xlabel('X')  # Mat example: "$\mu{g}\/ (h_1)^{-1}$"
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(False)  # Turn grid on/off
    ax.set_axis_off()  # Turn xy, yz, xz, planes off

    # plot_title = "HyperCata" + " Mirror"
    # ax.set_title(plot_title, loc='center')
    # or
    # Show name of mirror as text
    # ax.text2D(0.05, 0.95, plot_title, transform=ax.transAxes)

    # Draw hyperbolic mirrors:
    scale = 1.0
    stride_value = 2

    hyper_x1, hyper_y1, hyper_z1 = omnistereo_model.top_model.get_surf_points(scale)
    # ax.plot_wireframe(x, y, hyper_z1, color="g")
    ax.plot_surface(hyper_x1, hyper_y1, hyper_z1, rstride=stride_value, cstride=stride_value, alpha=0.5, linewidth=0.05, shade=True, cmap="YlGnBu")

    x_reflex, y_reflex, z_reflex = omnistereo_model.top_model.get_reflex_surf_points(scale)
    ax.plot_surface(x_reflex, y_reflex, z_reflex, rstride=stride_value, cstride=stride_value, alpha=0.4, linewidth=0.05, shade=True, color="yellow")

    hyper_x2, hyper_y2, hyper_z2 = omnistereo_model.bot_model.get_surf_points(scale)
    ax.plot_surface(hyper_x2, hyper_y2, hyper_z2, rstride=stride_value, cstride=stride_value, alpha=0.5, linewidth=0.05, shade=True, cmap="YlGnBu_r")

    # Draw foci of model
    ax.scatter(F1[0], F1[1], F1[2], color="blue", s=20)
    ax.scatter(Oc[0], Oc[1], Oc[2], color="black", s=20)
    ax.scatter(F2[0], F2[1], F2[2], color="red", s=20)
    ax.scatter(F2v[0], F2v[1], F2v[2], color="red", s=20)
    if show_labels:
        ax.text(F1[0] + 0.1, F1[1] + 0.1, F1[2], "F$_1$", color="blue")
        ax.annotate('arrowstyle', xy=(F1[0], F1[2]), xycoords='data',
                       xytext=(-10, 10), textcoords='offset points',
                       arrowprops=dict(arrowstyle="->")
                       )

        ax.text(Oc[0] + 1, Oc[1], Oc[2] - 2, "O$_c$", color="black")
        ax.text(F2[0] + 2, F2[1], F2[2] - 2, "F$_2$", color="red")
        ax.text(F2v[0] + 0.5, F2v[1], F2v[2] - 4, "O$_c$'", color="red")

    # Draw a fake projection plane
    if proj_plane_scale != 0:
        # Extra large (just for better visualization)
        plane_width, plane_height = 2 * proj_plane_scale * mirror1.precalib_params.sensor_size
        focal_length = proj_plane_scale * mirror1.precalib_params.focal_length
        # create grid of x,y values
        # Recall that stop in range is non-inclusive.
        # This would be the real grid:
        # h_x, h_y = proj_plane_scale * mirror1.precalib_params.pixel_size  # Pixel size in [mm]
        h_x, h_y = plane_width / 10, plane_height / 10  # Some sparse grid
        xx, yy = np.meshgrid(np.arange(-plane_width / 2, plane_width / 2 + 1, h_x),
                         np.arange(-plane_height / 2, plane_height / 2 + 1, h_y))
        zz = focal_length + np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, color='cyan', alpha=0.5)
        if show_labels:
            ax.text(-1.25 * plane_width, -plane_height / 2, focal_length + 5, "Camera Plane", color='blue')  # , zdir='y')

        # Draw virtual projection plane
        # reuse grid of x,y values
        zz_virt = F2v[2] - focal_length
        ax.plot_surface(xx, yy, zz_virt, color="pink", alpha=0.5)
        if show_labels:
            ax.text(plane_width / 2, -plane_height, zz_virt + 20, "Virtual Camera Plane", color='purple')  # , zdir='y')

    return ax

def draw_fwd_projection(omni_model, ax=None):
    '''
    @return the drawn axis corresponding to the parameter ax of the figure
    '''
    import matplotlib.pyplot as plt

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
    # Simpler variable names:

    ax.grid(True)  # Turn grid on/off

    # plot_title = "HyperCata" + " Mirror"
    # ax.set_title(plot_title, loc='center')
    # or
    # Show name of mirror as text
    # ax.text2D(0.05, 0.95, plot_title, transform=ax.transAxes)

    # Draw hyperbolic mirrors:
    profile_r, profile_z = omni_model.get_2D_profile_wrt_itself(100)
    # ax.plot_wireframe(x, y, profile_z, color="g")
    ax.plot(profile_r, profile_z)
    ax.scatter(0, 0, color="blue", s=20)
    ax.text(0, 0 - 2, "F$_%d$" % omni_model.mirror_number, color="blue")

#         plt.legend()
#         plt.tight_layout(0.0)  # Use it when saving figures, as it breaks interactivity
#         plt.savefig('figure.png')
#         plt.savefig('figure.pgf')
    fig.tight_layout()
    plt.show()  # Show both figures in separate windows
    return ax

def draw_fwd_projection_omnistereo(omnistereo_model, Pw, ax=None, verbose=False, fig_size=None):
    '''
    @return the drawn axis corresponding to the parameter ax of the figure
    '''
    from omnistereo.common_tools import unit_test

    proj_plane_scale = 20


    if ax == None:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection='3d', aspect="equal")

    ax = draw_omnistereo_model(omnistereo_model, ax, proj_plane_scale)  # Draw the barebones model

    # Simpler variable names:
    F1 = omnistereo_model.F1.reshape(4)
    Oc = omnistereo_model.Oc.reshape(4)
    F2 = omnistereo_model.F2.reshape(4)
    F2v = omnistereo_model.F2v.reshape(4)
    mirror1 = omnistereo_model.top_model
    mirror2 = omnistereo_model.bot_model
    focal_length = proj_plane_scale * mirror1.precalib_params.focal_length

    # Create geometric 3D planes with Euclid:
    from omnistereo import euclid
    vector_of_camera_plane = euclid.Vector3(0, 0, 1.0)  # A perpendicular plane to the Z-axis
    point_on_camera_plane = euclid.Point3(0, 0, focal_length)
    camera_plane = euclid.Plane(point_on_camera_plane, vector_of_camera_plane)
    vector_of_virtual_plane = euclid.Vector3(0, 0, 1.0)  # A perpendicular plane to the Z-axis
    point_on_virtual_plane = euclid.Point3(0, 0, F2v[2] - focal_length)
    virtual_plane = euclid.Plane(point_on_virtual_plane, vector_of_virtual_plane)
    point_Oc = euclid.Point3(Oc[0], Oc[1], Oc[2])
    point_Ocv = euclid.Point3(F2v[0], F2v[1], F2v[2])

    # Draw test points
    for P in Pw:
        (xw, yw, zw) = P
        ax.scatter(xw, yw, zw, color="black", s=20)
        ax.text(xw + 0.1, yw + 0.1, zw, "P$_w$", color="black")
        P_homo = np.array([[[xw, yw, zw, 1.0]]])
        # Reflection via mirror 1:
        if verbose: print("MIRROR 1:")

        P1_fp = mirror1.get_reflection_point_on_mirror(P_homo)
        x1, y1, z1 = float(P1_fp[..., 0]), float(P1_fp[..., 1]), float(P1_fp[..., 2])
        if verbose: print("World Point (%f,%f,%f) reflects at: (%f,%f,%f)" % (xw, yw, zw, x1, y1, z1))

        ax.scatter(x1, y1, z1, color="blue", s=20)
        ax.text(x1 + 0.2, y1 + 0.2, z1 + 0.2, "P$_1$", color="blue")

        # Plot the Pw-P1 line segment
        line_PwP1_xs = [xw, x1]
        line_PwP1_ys = [yw, y1]
        line_PwP1_zs = [zw, z1]
        ax.plot3D(line_PwP1_xs, line_PwP1_ys, line_PwP1_zs, color="blue", linestyle='-', linewidth=1.5, alpha=1.0)
        # Plot the P1-F1 line segment
        line_P1F1_xs = [x1, F1[0]]
        line_P1F1_ys = [y1, F1[1]]
        line_P1F1_zs = [z1, F1[2]]
        ax.plot3D(line_P1F1_xs, line_P1F1_ys, line_P1F1_zs, color="blue", linestyle=':', linewidth=1.5, alpha=1.0)

        # Project to normalized camera plane
        Q1 = mirror1.project_3D_point_to_normalized_plane(P_homo)
        if verbose: print("Q_1 = ", Q1)
        P1_bp = mirror1.back_project_Q_to_mirror(Q1)
        if verbose:
            print("P1 back projected:", P1_bp)
            unit_test(P1_fp[..., :3], P1_bp[..., :3], 6)

        # Draw projected point on camera plane (Not using Q, but something different for visualization)
        line_P1Oc = euclid.Line3(euclid.Point3(x1, y1, z1), point_Oc)
        point_proj_to_cam_plane = line_P1Oc.intersect(camera_plane)
        ax.scatter(point_proj_to_cam_plane.x, point_proj_to_cam_plane.y, point_proj_to_cam_plane.z, color="blue", s=20)
        ax.text(point_proj_to_cam_plane.x + 1, point_proj_to_cam_plane.y, point_proj_to_cam_plane.z + 2, "m$_1$", color="blue")

        # Draw projection of Pr1 through Camera (Oc) Pinhole
        line_P1C_xs = [x1, Oc[0]]
        line_P1C_ys = [y1, Oc[1]]
        line_P1C_zs = [z1, Oc[2]]
        ax.plot3D(line_P1C_xs, line_P1C_ys, line_P1C_zs, color="blue", linestyle='-' , linewidth=1.5, alpha=1.0, label=r"Projection via mirror $1$")

        # Reflection via mirror 2:
        if verbose: print("MIRROR 2:")
        P2_fp = mirror2.get_reflection_point_on_mirror(np.array([[[xw, yw, zw, 1.0]]]))
        x2, y2, z2 = float(P2_fp[..., 0]), float(P2_fp[..., 1]), float(P2_fp[..., 2])

        if verbose: print("World Point (%f,%f,%f) reflects at: (%f,%f,%f)" % (xw, yw, zw, x2, y2, z2))

        ax.scatter(x2, y2, z2, color="red", s=20)
        ax.text(x2 + 1, y2, z2 - 5, "P$_2$", color="red")

        # Draw the Pw-P2 line segment
        line_PwP2_xs = [xw, x2]
        line_PwP2_ys = [yw, y2]
        line_PwP2_zs = [zw, z2]
        ax.plot3D(line_PwP2_xs, line_PwP2_ys, line_PwP2_zs, color="red", linestyle='-', linewidth=1.5, alpha=1.0, label=r"Projection via mirror $2$")
        # Draw the P2-F2 line segment
        line_P2F2_xs = [x2, F2[0]]
        line_P2F2_ys = [y2, F2[1]]
        line_P2F2_zs = [z2, F2[2]]
        ax.plot3D(line_P2F2_xs, line_P2F2_ys, line_P2F2_zs, color="red", linestyle=':', linewidth=1.5, alpha=1.0)
        # Draw line from P2 to  vitual camera at F2v
        line_P2F2v_xs = [x2, F2v[0]]
        line_P2F2v_ys = [y2, F2v[1]]
        line_P2F2v_zs = [z2, F2v[2]]
        ax.plot3D(line_P2F2v_xs, line_P2F2v_ys, line_P2F2v_zs, color="pink", linestyle='--', linewidth=1.5, alpha=1.0)


        # Project to normalized camera plane
        Q2 = mirror2.project_3D_point_to_normalized_plane(P_homo)
        if verbose: print("Q_2 = ", Q2)
        P2_bp = mirror2.back_project_Q_to_mirror(Q2)
        if verbose:
            print("P2 back projected:", P2_bp)
            unit_test(P2_fp[..., :3], P2_bp[..., :3], 6)

        # Project to reflex mirror
        P2_wrt_F2v = mirror2.transform_due_to_reflex(np.array([x2, y2, z2, 1.0]))
        if verbose: print("P2 w.r.t. [F2v] via reflex transform:", P2_wrt_F2v)
        Pr = mirror2.get_reflection_point_reflex_vector(P2_fp)
        xr, yr, zr = float(Pr[0]), float(Pr[1]), float(Pr[2])
        ax.scatter(xr, yr, zr, color="red", s=20)
        ax.text(xr + 1, yr, zr - 5, "P$_r$", color="red")
        line_P2Pr_xs = [x2, xr]
        line_P2Pr_ys = [y2, yr]
        line_P2Pr_zs = [z2, zr]
        ax.plot3D(line_P2Pr_xs, line_P2Pr_ys, line_P2Pr_zs, color="red", linestyle='-', linewidth=1.0, alpha=0.8)
        # Draw projection from reflex to real camera Pinhole
        line_PrC_xs = [xr, Oc[0]]
        line_PrC_ys = [yr, Oc[1]]
        line_PrC_zs = [zr, Oc[2]]
        ax.plot3D(line_PrC_xs, line_PrC_ys, line_PrC_zs, color="red", linestyle='-', linewidth=1.0, alpha=0.8)

        # Draw real projected point on the camera plane (Not using Q, but something different for visualization)
        line_P2Oc = euclid.Line3(euclid.Point3(xr, yr, zr), point_Oc)
        point2_proj_to_cam_plane = line_P2Oc.intersect(camera_plane)
        ax.scatter(point2_proj_to_cam_plane.x, point2_proj_to_cam_plane.y, point2_proj_to_cam_plane.z, color="red", s=20)
        ax.text(point2_proj_to_cam_plane.x - 5, point2_proj_to_cam_plane.y, point2_proj_to_cam_plane.z + 0.4, "m$_2$", color="red")
        # Draw projected point on the virtual camera plane (Not using Q, but something different for visualization)
        line_P2Ocv = euclid.Line3(euclid.Point3(x2, y2, z2), point_Ocv)
        point2_proj_to_virtual_plane = line_P2Ocv.intersect(virtual_plane)
        ax.scatter(point2_proj_to_virtual_plane.x, point2_proj_to_virtual_plane.y, point2_proj_to_virtual_plane.z, color="red", s=20)
        ax.text(point2_proj_to_virtual_plane.x + 0.2, point2_proj_to_virtual_plane.y, point2_proj_to_virtual_plane.z + 0.4, "m$_2$", color="red")


#         plt.legend()
#         plt.tight_layout(0.0)  # Use it when saving figures, as it breaks interactivity
#         plt.savefig('figure.png')
#         plt.savefig('figure.pgf')
#     plt.tight_layout()

    ax.dist = 70  # 10 is default
    plt.tight_layout()
    ax.set_aspect("equal")

    r = 10
    ax.set_xlim3d([-r, r])
    ax.set_ylim3d([-r, r])
    ax.set_zlim3d([10 * r, 12 * r])

    ax.set_axis_off()

    plt.draw()
    plt.show()  # Show both figures in separate windows
    return ax

def draw_fwd_projection_GUM(model, ax=None):
    '''
    @return the drawn axis correspoding to the parameter ax of the figure
    '''

    is_new_figure = False

    if ax == None:
        is_new_figure = True
        from mpl_toolkits.mplot3d import Axes3D  # To be run from the command line
#             from mplot3d import axes3d
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # make simple, bare axis lines through space:
    ax.set_aspect("equal")
    max_axis = 2
    ax.set_xlim3d(-max_axis, max_axis)
    ax.set_ylim3d(-max_axis, max_axis)
    ax.set_zlim3d(-max_axis, max_axis)
    axis_range = (0, max_axis)
    xAxisLine = (axis_range, (0, 0), (0, 0))  # 2 points make the x-axis line at the data extrema along x-axis
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')  # make a red line for the x-axis.
    yAxisLine = ((0, 0), axis_range, (0, 0))  # 2 points make the y-axis line at the data extrema along y-axis
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'g')  # make a green line for the y-axis.
    zAxisLine = ((0, 0), (0, 0), axis_range)  # 2 points make the z-axis line at the data extrema along z-axis
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'b')  # make a blue line
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plot_title = model.mirror_name + " GUM"
    ax.set_title(plot_title, loc='center')
    # Show name of mirror as text
    # ax.text2D(0.05, 0.95, plot_title, transform=ax.transAxes)

    # Draw unit sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    scale = 1.0
    x = scale * np.outer(np.cos(u), np.sin(v))
    y = scale * np.outer(np.sin(u), np.sin(v))
    z = scale * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, rstride=6, cstride=6, color='g', alpha=0.25, linewidth=0.05, shade=False)

    # Draw center of model
    ax.scatter(0. , 0., 0. , color="black", s=20)
    ax.text(0.1, 0., 0., "M", color="black")
    # Draw center of projection (Cp)
    ax.scatter(model.Cp_wrt_M[0], model.Cp_wrt_M[1], model.Cp_wrt_M[2], color="r", marker='o', s=50)
    ax.text(model.Cp_wrt_M[0] + 0.1, model.Cp_wrt_M[1], model.Cp_wrt_M[2], "Cp", color='red')

    # Draw normalized projection plane
    # Plot a 3D plane:
#         normal = model.normalized_projection_plane.n
    normal = model.plane_n
    # a plane is a*x+b*y+c*z=d
    # [a,b,c] is the normal.
    # To calculate d = -point.dot(normal)
#         d = model.normalized_projection_plane.k
    d = model.plane_k
    # create grid of x,y values
    plane_width = 6.
    plane_height = 4.
    # Recall that stop in range is non-inclusive.
    xx, yy = np.meshgrid(np.arange(-plane_width / 2, plane_width / 2 + 1, 0.5),
                         np.arange(-plane_height / 2, plane_height / 2 + 1, 0.5))
    # calculate corresponding z = -(a*x+b*y-d)/c
#         zz = -(normal.x * xx + normal.y * yy - d) / normal.z
    zz = -(normal[0] * xx + normal[1] * yy - d) / normal[2]
    # draw "solid" planar surface (but it occludes others sometimes on view)
    # ax.plot_surface(xx, yy, zz, color="yellow", shade=False, alpha=.75, linewidth=0, zorder=-1)
    ax.plot_wireframe(xx, yy, zz, color="yellow")
    ax.text(plane_width / 4, 0, -d + 0.1, "Proj. Norm. Plane", color='brown', zdir='y')

    # WISH: Draw image plane with pixels

    if is_new_figure:
        plt.show()  # Show both figures in separate windows

    return ax

def draw_fwd_projection_GUMS(omnistereo_model):
    # TODO: Draw an integrated system as for the HyperCataStereo case
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
#         from mplot3d import axes3d
    fig = plt.figure()
    ax_top = fig.add_subplot(211, projection='3d')
    ax_bot = fig.add_subplot(212, projection='3d')

    draw_fwd_projection_GUM(omnistereo_model.top_model, ax_top)
    draw_fwd_projection_GUM(omnistereo_model.bot_model, ax_bot)

    plt.show()  # Show both figures as subplots (same window)

def draw_model_mono_vispy(omni_model_mono, finish_drawing=True, view=None, show_grid=False, backend=None):
    '''
    Draws the monocular (theoretical) omnidirectional model in 3D using vispy

    @param omni_model_mono: The theoretical omnidirectional model (monocular/single mirror)
    @param finish_drawing: To indicate when the main loop of vispy's App can run (scene is updated through this App)
    @param view: The scene's canvas view object instance, if any.
    '''
    from vispy import scene
    from vispy.scene import visuals

    scale = 1. / 1000  # scale = 1. in [mm] | scale = 1 / 1000 in [m]

    # WISH: Get the the mirror surface drawn better:
    # TODO2: Do the omnistereo model
    if view is None:
        from vispy import use
        if backend is None:
            use(app="glfw", gl="gl2")
        else:
            # For some reason it fails if  we did on the Notebook this
            # %load_ext vispy.ipython
            if backend != "ipynb_webgl":
                use(app=backend, gl="gl2")

        # Make a canvas and add simple view
        canvas = scene.SceneCanvas(keys='interactive', show=True, title="Hyperstereo Model", bgcolor="black")
        canvas.show()
        view = canvas.central_widget.add_view()
        view.camera = 'arcball'  # 'turntable'  # or try 'arcball'

        # Add grid
        if show_grid:
            grid_3D = scene.visuals.GridLines()
            view.add(grid_3D)

        # add a colored 3D axis for orientation
        axis = visuals.XYZAxis(parent=view.scene)
        view.add(axis)


    lin_data_size = 200
    # Points data
    (hyper_xx, hyper_x), (hyper_yy, hyper_y), hyper_zz = omni_model_mono.get_surf_points(scale, steps=lin_data_size, dense=True)
    hyperboloidal_surface = scene.visuals.SurfacePlot(x=hyper_x, y=hyper_y, z=hyper_zz, color=(0.5, 0.5, 0.5, 1))
    view.add(hyperboloidal_surface)

    lip_x, lip_y, lip_zz = omni_model_mono.get_mounting_lip_points(scale, steps=lin_data_size, lip_radius=5, seam_size=1)
    if omni_model_mono.mirror_number == 1:  # Top mirror has blue lip
        lip_color = (0, 0, 1., 1.)
    else:  # Bottom mirror lip is red
        lip_color = (1., 0, 0, 1.)
    lip_surface = scene.visuals.SurfacePlot(x=lip_x, y=lip_y, z=lip_zz, color=lip_color, shading='smooth')
    view.add(lip_surface)


    #=======================================================================
    # from vispy import plot as vp
    # fig = vp.Fig(bgcolor='w', size=(800, 800), show=False)
    # clim = [32, 192]
    # vol_pw = fig[0, 0]
    # vol_pw.mesh(vertices=verts, face_colors="blue")
    # #         vol_pw.view.camera.elevation = 30
    # #         vol_pw.view.camera.azimuth = 30
    # #         vol_pw.view.camera.scale_factor /= 1.5
    # fig.show(run=True)
    #=======================================================================

    #=======================================================================
    # xax = scene.Axis(pos=[[-0.5, -0.5], [0.5, -0.5]], tick_direction=(0, -1),
    #                  font_size=16, axis_color='r', tick_color='r', text_color='r',
    #                  parent=view.scene)
    # #         xax.transform = scene.STTransform(translate=(0, 0, -0.2))
    # yax = scene.Axis(pos=[[-0.5, -0.5], [-0.5, 0.5]], tick_direction=(-1, 0),
    #                  font_size=16, axis_color='g', tick_color='g', text_color='g',
    #                  parent=view.scene)
    # #         yax.transform = scene.STTransform(translate=(0, 0, -0.2))
    #=======================================================================

#         import sys
#         if sys.flags.interactive != 1:
    if finish_drawing:
        from vispy import app
        app.run()


    return view

def draw_omnistereo_model_vispy(omnistereo_model, show_grid=False, backend=None):
    model_view = draw_model_mono_vispy(omnistereo_model.top_model, finish_drawing=False, view=None, show_grid=show_grid, backend=backend)
    model_view = draw_model_mono_vispy(omnistereo_model.bot_model, finish_drawing=False, view=model_view, show_grid=show_grid, backend=backend)
    from vispy import app
    app.run()

def draw_axes_3D_mplot3D(r_sys, ax):
    # make simple, bare axis lines through space:
    max_axis = r_sys
    ax.set_xlim3d(-max_axis, max_axis)
    ax.set_ylim3d(-max_axis, max_axis)  # ax.set_zlim3d(-max_axis, max_axis)
    axis_range = 0, max_axis
    # TODO: Label end of axes with "X", "Y", "Z"
    xAxisLine = axis_range, (0, 0), (0, 0)  # 2 points make the x-axis line at the data extrema along x-axis
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], linestyle=':', linewidth=0.5, color='r')  # make a red line for the x-axis.
    yAxisLine = (0, 0), axis_range, (0, 0)  # 2 points make the y-axis line at the data extrema along y-axis
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], linestyle=':', linewidth=0.5, color='g')  # make a green line for the y-axis.
    zAxisLine = (0, 0), (0, 0), axis_range  # 2 points make the z-axis line at the data extrema along z-axis
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], linestyle=':', linewidth=0.5, color='b')  # make a blue line

# Draw a cylinder mesh
def draw_cyl_with_mplot3d():
    from mpl_toolkits.mplot3d import Axes3D
    # from mplot3d import axes3d
    import matplotlib.image as mpimg
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import colorConverter
    import numpy as np

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', aspect="equal")
    ax.grid(False)  # Turn grid on/off
    ax.set_axis_off()  # Turn xy, yz, xz, planes off
#     ax.set_xlabel('X')  # Mat example: "$\mu{g}\/ (h_1)^{-1}$"
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

    MIRROR_IMAGE = True
    img = mpimg.imread('figures/panorama_bottom.jpg')
    print("Img size:", img.shape)
    if MIRROR_IMAGE:
        img = img[-1::-1, -1::-1]
#     print(img[:rows, :unwrapped_cols, :])
    my_colors = np.array([[0, 0.7, 1], [0, 0.7, 0], [0, 0.7, 0], [0, 0.7, 0], [0, 0.7, 0]])
#     my_colors = [0.1, 0.1, 0.1, 0.1, 0.1]
#     print(my_colors)

    radius = 1
    height = 1  # in meters, as the height value (along z) is obtained from the vertical field of view
    wrapping_ratio = 3  # Decides the proportion between wrapped and unwrapped regions of the cylindrical panorama
    cols = img.shape[1]  # one extra column will be added for visualization
    unwrapped_cols = img.shape[1] / wrapping_ratio  # number of columns not showing as a cylinder because have been unwrapped to the panoramic plane
    wrapped_cols = cols + 1 - unwrapped_cols
    rows = img.shape[0]  # one extra column will be added for visualization
    unwrapped_angle = unwrapped_cols * 2 * np.pi / (cols + 1)
    unwrapped_circumference = unwrapped_angle * radius

    draw_axes_3D_mplot3D(max(radius, unwrapped_circumference / 2), ax)
    # TODO: Add Axis labels
    # TODO: Shift cylinder around a fake focus

    delta_psi = np.linspace(0, 2 * np.pi - unwrapped_angle, wrapped_cols) + np.pi / 2  # Azimuth intervals sifted by 90 degrees for nicer visualization
    delta_h = np.linspace(0, height, rows + 1)  # Height intervals
    delta_circ_unwrapped = np.linspace(0, unwrapped_circumference, unwrapped_cols + 1, endpoint=True)
#     print("delta_psi", delta_psi)
#     print("delta_h", delta_h)
    print("Unwrapped: angle %f degrees,  circumference=%f m" % (np.rad2deg(unwrapped_angle), unwrapped_circumference))
    x_cyl = radius * np.outer(np.ones(rows + 1), np.cos(delta_psi))
    print("x_cyl", np.shape(x_cyl))
    y_cyl = radius * np.outer(np.ones(rows + 1), np.sin(delta_psi))
    print("y_cyl", np.shape(y_cyl))
    z_cyl = np.outer(delta_h, np.ones(np.size(delta_psi)))
    print("z_cyl", np.shape(z_cyl))

    #     my_colors = np.array([[0, 0.7, 1], [0, 0.7, 0], [0, 0.7, 0], [0, 0.7, 0], [0, 0.7, 0]])
    cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)
    wrapped_img_colors = img[:rows, wrapped_cols:0:-1, :] / 255.0

    cyl_as_poly = ax.plot_surface(x_cyl, y_cyl, z_cyl, rstride=1, cstride=1, facecolors=wrapped_img_colors, linewidth=0, alpha=1.0)
    # Draw partially unwrapped panorama
    x_pano = np.outer(np.ones(rows + 1), delta_circ_unwrapped)
    print("x_pano", np.shape(x_pano))
    y_pano = np.zeros((rows + 1, unwrapped_cols + 1)) + radius
    print("y_pano", np.shape(y_pano))
    z_pano = np.outer(delta_h, np.ones(np.size(delta_circ_unwrapped)))
    print("z_pano", np.shape(z_pano))

    unwrapped_img_colors = img[:rows, wrapped_cols - 1:, :] / 255.0
    pano_as_poly = ax.plot_surface(x_pano, y_pano, z_pano, rstride=1, cstride=1, facecolors=unwrapped_img_colors, linewidth=0, alpha=1.0)
    # pano_as_poly.set_facecolors(img_colors)
    ax.add_collection3d(cyl_as_poly)  # It helps to make it less transparent!
    ax.add_collection3d(pano_as_poly)  # It helps to make it less transparent!

    # Draw a pretend pixel grid
    wire_color = "yellow"
    wire_row_stride = 10
    wire_col_stride = 10
    wire_thickness = 0.5
    wire_alpha = 0.6
    ax.plot_wireframe(x_cyl, y_cyl, z_cyl, rstride=wire_row_stride, cstride=wire_col_stride, color=wire_color, linewidth=wire_thickness, alpha=wire_alpha)
    ax.plot_wireframe(x_pano, y_pano, z_pano, rstride=wire_row_stride, cstride=wire_col_stride, color=wire_color, linewidth=wire_thickness, alpha=wire_alpha)
    plt.show()

# draw_cyl_with_mplot3d()

# Example of PolyCollection
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.collections import PolyCollection
# from matplotlib.colors import colorConverter
# import matplotlib.pyplot as plt
# import numpy as np
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)
#
# xs = np.arange(0, 10, 0.4)
# verts = []
# zs = [0.0, 1.0, 2.0, 3.0]
# for z in zs:
#     ys = np.random.rand(len(xs))
#     ys[0], ys[-1] = 0, 0
#     verts.append(list(zip(xs, ys)))
#
# poly = PolyCollection(verts, facecolors=[cc('r'), cc('g'), cc('b'),
#                                            cc('y')])
# poly.set_alpha(0.7)
# ax.add_collection3d(poly, zs=zs, zdir='y')
#
# ax.set_xlabel('X')
# ax.set_xlim3d(0, 10)
# ax.set_ylabel('Y')
# ax.set_ylim3d(-1, 4)
# ax.set_zlabel('Z')
# ax.set_zlim3d(0, 1)
#
# plt.show()

def draw_pano_as_point_cloud_visvis(pano, radius, z_origin_offset=0):
    import visvis as vv

    # img_pano = pano.get_panoramic_image()  # img_pano[-1::-1, ...]  # Flip rows because pixel origin is on top (not bottom like for Cartesian coordinates)
    img_pano = pano.panoramic_img  # img_pano[-1::-1, ...]  # Flip rows because pixel origin is on top (not bottom like for Cartesian coordinates)

    rows = img_pano.shape[0]  # one extra column will be added for visualization
    cols = img_pano.shape[1]  # one extra column will be added for visualization

    # set labels
    vv.xlabel('X')
    vv.ylabel('Y')
    vv.zlabel('Z')

    # Prepare
    a = vv.gca()

    # Draw Origin
    pt_color = 'w'
    # t_focus = vv.Text(a, text='F', x=0.1, y=0., z=0, fontName="mono", fontSize=30, color=pt_color)
    orig_pt = vv.Point(0, 0, z_origin_offset)  # TODO: For now, just hacking the focus position to make it coincide with proper elevation angles
    l = vv.plot(orig_pt, ms='.', mc=pt_color, mw='12', ls='', mew=0, axesAdjust=False)
    draw_axes_visvis(1, 1, '-', z_origin_offset)

    #     x_full_surf, y_full_surf, z_full_surf = get_cylinder_surface_matrices(rows, cols, radius, height, wrapping_ratio=0, debug=False)
    for row in range(rows):
        for col in range(cols):
            point_color = tuple(img_pano[row, col] / 255)
            X, Y, Z = pano.get_point_on_cylinder_from_pano_pixel(row, col, radius)
            cyl_pt = vv.Point(X, Y, Z + z_origin_offset)  # TODO: For now, just hacking the focus position to make it coincide with proper elevation angles
            vv.plot(cyl_pt, ms='.', mc=point_color, mw='0.1', ls='', mew=0, axesAdjust=False)

    app = vv.use()
    app.Run()


#===============================================================================
# 2D Plots for Theoretical Model
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

#===============================================================================
# Using PinholeCamera
#===============================================================================

def plot_spatial_resln_cam(cam_model, r, eta, eta_max, in_pixels, in_radians, use_spatial_resolution, in_2D, ax):
    model_label = r"$\eta_{cam}$"

    is_parent_plot = False
    if ax == None:
        is_parent_plot = True
        fig = plt.figure()
        ax = fig.add_subplot(111)
# Plot
    font = {'family':'serif', 'color':'darkblue',
        'weight':'normal',
        'size':16}
    font_big = {'family':'serif',
        'color':'black',
        'weight':'normal',
        'size':20}
    ax.plot(r, eta[-1], label=model_label)
#         ax.plot(r, eta, label=model_label)
    if is_parent_plot:
        # Tweak spacing to prevent clipping of ylabel
        ax.grid(True)
        # ax.set_xlim(0)  # TODO: expand limits after debuging close up
        ax.set_ylim(0, eta_max)

        if in_2D:
            res_dim = r"2D "
            res_symbol = r"\eta_{cam_{\mathrm{%s}}}" % res_dim
        else:
            res_dim = r"3D "
            res_symbol = r"\eta_{cam}"

        if use_spatial_resolution == False:
            res_symbol = r"\frac{1}{%s}" % res_symbol

        if use_spatial_resolution:
            type_res = "Spatial"
        else:
            type_res = "Angular"

        if in_2D:
            power_str = r""
        else:
            power_str = r"^2"

        ax.set_title(res_dim + type_res + ' Resolution for an Ideal Perspective Camera', fontdict=font)

        if in_pixels:
            x_units = r"$^{[\mathrm{I_C}]} u-\mathrm{coordinate} \, [\mathrm{pixel}%s]$" % power_str
            res_unit_denom = r"\mathrm{pixel}%s" % power_str
            area_factor = r"s_{\mathrm{pix}}"  # Pixel area
        else:
            x_units = r"$r\, [\mathrm{mm}%s]$" % power_str
            res_unit_denom = r"\mathrm{mm}%s" % power_str
            area_factor = ""
        if in_radians:
            if in_2D:
                res_unit_num = r"\mathrm{rad}"
            else:
                res_unit_num = r"\mathrm{sr}"
            solid_angle_factor = ""
        else:
            res_unit_num = r"\mathrm{degree}%s" % power_str
        # To convert from steradian to square degrees multiply by (180/pi)^2
#                 solid_angle_factor = "\left( \frac{180}{\pi} \right)^2"
            solid_angle_factor = r"\left(\frac{180}{\pi}\right)%s" % power_str
        x_lims = ax.get_xlim()
        y_lims = ax.get_ylim()

        if use_spatial_resolution:
            y_units = r'$%s\,\left[\frac{\mathrm{%s}}{%s}\right]$' % (res_symbol, res_unit_denom, res_unit_num)
        else:
            y_units = r'$%s\,\left[\frac{\mathrm{%s}}{%s}\right]$' % (res_symbol, res_unit_num, res_unit_denom)

        ax.set_xlabel(x_units, fontdict=font)
        ax.set_ylabel(y_units, fontdict=font)

    return ax

def plot_perspective_camera_spatial_resolution(cam_model, in_2D=False):
    # Plot Camera Resolution
    # NOTE: A complete sphere solid angle = 4 pi^2
    r_min = 0  # mirror1.r_min
    r_max = 100  # mirror1.r_max
    eta_max = None  # 0.0015  # 1.25
    use_pixels = True
    use_radians = False
    use_spatial_resolution = True

    if in_2D:
        ax = plot_spatial_resolution_2D_cam(cam_model, r_min, r_max, eta_max, use_pixels, use_radians, use_spatial_resolution)
    else:
        ax = plot_spatial_resolution_cam(cam_model, r_min, r_max, eta_max, use_pixels, use_radians, use_spatial_resolution)

    ax = add_camera_info_box_on_plot(cam_model, in_2D, ax)

    extraticks = []  # [r_sys]
    officialticks = ax.get_xticks()
    ax.set_xticks(np.append(officialticks, extraticks))

    # ax.legend().draggable()
    plt.tight_layout()
    plt.show()

def plot_spatial_resolution_2D_cam(cam_model, r_min=0, r_max=1000, eta_max=None, in_pixels=False, in_radians=True, use_spatial_resolution=True, ax=None):
    '''
    @param use_spatial_resolution: If True, it indicates to calculate the spatial resolution (as a ratio) in [area per st] units instead of [st per area].
    @note: Checking integral under curve, for an average eta of 0.0184 [mm/degree] x FOV 45 [degrees]/2 x 2 = 0.828 mm, which is close the sensor width
            In pixels, pick eta around 28.45 [pixel/degree] x FOV 45 [degrees]/2 x 2
    @return the drawn axis corresponding to the parameter ax of the figure
    '''

    r_size = 100
#         r = np.linspace(r_min, r_max, r_size)  # Also x coordinates
    r_pixels = np.arange(-cam_model.image_size[0] / 2, cam_model.image_size[0] / 2)
    r = r_pixels * cam_model.pixel_size[0]

    y = np.zeros_like(r)  # The y coordinates (all zeros)
    z = np.zeros_like(r) + cam_model.focal_length

    p = np.dstack((r, y, z))
    eta = cam_model.get_spatial_resolution_in_2D(p, in_pixels=in_pixels, in_radians=in_radians, use_spatial_resolution=use_spatial_resolution)
    in_2D = True
    if in_pixels:
        r = r_pixels
    ax = plot_spatial_resln_cam(cam_model, r, eta, eta_max, in_pixels, in_radians, use_spatial_resolution, in_2D, ax)

    # datacursor(display='multiple', draggable=True)

    return ax

def plot_spatial_resolution_2D_as_BakerNayar(cam_model, space_model, r_min=0, r_max=1000, eta_max=None, in_pixels=False, in_radians=True, use_spatial_resolution=True, ax=None):
    '''
    @param space_model: It could be just a z value (a number) for plotting the spatial resolution on a plane at such distance
                        or a HyperCata instance (a mirror model) for the resolution of the camera at those points on the surface of the reflector.
    @param use_spatial_resolution: If True, it indicates to calculate the spatial resolution (as a ratio) in [area per st] units instead of [st per area].

    @return the drawn axis corresponding to the parameter ax of the figure
    '''
    from omnistereo.cata_hyper_model import HyperCata
    num_of_points = 100
    if isinstance(space_model, HyperCata):
        r, z = space_model.get_2D_profile_wrt_itself(num_of_points, r_max)
    else:  # Assume it's just the z value for points on a plane parallel to the XY-plane
        r = np.linspace(r_min, r_max, num_of_points)  # Also x coordinates
        z = np.zeros_like(r) + space_model
        # Correct z for our model where origin is the pinhole instead
        # CHECKME: this is not needed??? or is it?
        # z = np.zeros_like(r) + (cam_model.parent_model.c - space_model)

    y = np.zeros_like(r)  # The y coordinates (all zeros)
    p = np.dstack((r, y, z))

    eta_2D = cam_model.get_spatial_resolution_in_2D_as_BakerNayar(p, in_pixels=in_pixels, in_radians=in_radians, use_spatial_resolution=use_spatial_resolution)
    in_2D = True
    ax = plot_spatial_resln_cam(cam_model, space_model, r, eta_2D, eta_max, in_pixels, in_radians, use_spatial_resolution, in_2D, ax)

    return ax

def plot_spatial_resolution_cam(cam_model, r_min=0, r_max=1000, eta_max=None, in_square_pixels=False, in_steradians=True, use_spatial_resolution=True, ax=None):
    '''
    @param use_spatial_resolution: If True, it indicates to calculate the spatial resolution (as a ratio) in [area per st] units instead of [st per area].

    @return the drawn axis corresponding to the parameter ax of the figure
    '''
    r_size = 100
#         r = np.linspace(r_min, r_max, r_size)  # Also x coordinates
    r = np.arange(-cam_model.image_size[0] / 2, cam_model.image_size[0] / 2)
    if in_square_pixels == False:
        r = r * cam_model.pixel_size[0]

    y = np.zeros_like(r)  # The y coordinates (all zeros)
    z = np.zeros_like(r) + cam_model.focal_length
    p = np.dstack((r, y, z))
    eta = cam_model.get_spatial_resolution(p, in_pixels=in_square_pixels, in_steradians=in_steradians, use_spatial_resolution=use_spatial_resolution)
    in_2D = False
    ax = plot_spatial_resln_cam(cam_model, r, eta, eta_max, in_square_pixels, in_steradians, use_spatial_resolution, in_2D, ax)

    return ax

#===============================================================================
# def plot_spatial_resolution_as_BakerNayar(cam_model, space_model, r_min=0, r_max=1000, eta_max=None, in_square_pixels=False, in_steradians=True, use_spatial_resolution=True, ax=None):
#     '''
#     @param space_model: It could be just a z value (a number) for plotting the spatial resolution on a plane at such distance
#                         or a HyperCata instance (a mirror model) for the resolution of the camera at those points on the surface of the reflector.
#     @param use_spatial_resolution: If True, it indicates to calculate the spatial resolution (as a ratio) in [area per st] units instead of [st per area].
#
#     @return the drawn axis corresponding to the parameter ax of the figure
#     '''
#     from omnistereo.cata_hyper_model import HyperCata
#     num_of_points = 100
#     if isinstance(space_model, HyperCata):
#         r, z = space_model.get_2D_profile_wrt_itself(num_of_points, r_max)
#     else:  # Assume it's just the z value for points on a plane parallel to the XY-plane
#         r = np.linspace(r_min, r_max, num_of_points)  # Also x coordinates
#         z = np.zeros_like(r) + space_model
#         # Correct z for our model where origin is the pinhole instead
#         # CHECKME: this is not needed??? or is it?
#         # z = np.zeros_like(r) + (cam_model.parent_model.c - space_model)
#
#     y = np.zeros_like(r)  # The y coordinates (all zeros)
#     p = np.dstack((r, y, z))
#     eta = cam_model.get_spatial_resolution_as_BakerNayar(p, in_pixels=in_square_pixels, in_steradians=in_steradians, use_spatial_resolution=use_spatial_resolution)
#     in_2D = False
#     ax = plot_spatial_resln_cam(cam_model, r, eta, eta_max, in_square_pixels, in_steradians, use_spatial_resolution, in_2D, ax)
#
#     return ax
#===============================================================================

def add_camera_info_box_on_plot(cam_model, in_2D, ax):
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    str_focal_length = r"Focal Length: $f=%0.2f \, [\mathrm{mm}]$" % (cam_model.focal_length)

    if in_2D:
        str_pixel_dims = r"Pixel size $\approx %0.2e \, [\mathrm{mm}]$" % (cam_model.pixel_size[0])
    else:
        str_pixel_dims = r"Pixel area $\approx %0.2e \, [\mathrm{mm}^2]$" % (cam_model.pixel_area)

    str_img_size_in_pixels = r"Image size: $w_I = %d \times h_I = %d \,[\mathrm{pixels}]$" % (cam_model.image_size[0], cam_model.image_size[1])
    str_cam_FOVs = r"FOVs: $hor \approx %0.2f^\mathrm{o}$, $vert \approx %0.2f ^\mathrm{o}$" % (np.rad2deg(cam_model.FOV[0]), np.rad2deg(cam_model.FOV[1]))


    camera_info = r'''Camera information:
%s,
%s,
%s,
%s''' % (str_focal_length, str_pixel_dims, str_img_size_in_pixels, str_cam_FOVs)

    ax.text(max(x_lims) * (-4 / 10), max(y_lims) * (5 / 10), camera_info, size=12, bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':10})

    return ax

#===============================================================================
# Using HyperCata
#===============================================================================
def plot_profile(omni_model, k_values, r_max=100, ax=None, font=None, fig_size=None):
    is_parent_plot = False
    if ax == None:
        is_parent_plot = True
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, aspect="equal")

    ax.set_aspect("equal")
    model_number = omni_model.mirror_number

    r = np.linspace(-r_max, r_max, 2 * r_max)
    y = np.zeros_like(r)  # The y coordinates (all zeros)

    if font == None:
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

    lines = []
    # First, plot the optimal k in the current model
    if model_number == 1:
        z = omni_model.get_z_hyperbola(r, y, is_virtual=False)
    elif model_number == 2:
        z = omni_model.get_z_hyperbola(r, y, is_virtual=False)

    model_label = r"$k_%d = %0.1f ^{(\mathrm{Opt.})}$" % (model_number, omni_model.k)
    line, = ax.plot(r, z, linestyle="--", linewidth=2, label=model_label)
    lines.append(line)

    for current_k in k_values:
        if model_number == 1:
            z = omni_model.get_z_hyperbola(r, y, is_virtual=False, k=current_k)
        elif model_number == 2:
            # Here, the point will be given as the virtual point seen by the second focus
            z = omni_model.get_z_hyperbola(r, y, is_virtual=False, k=current_k)

        model_label = r"$k_%d = %0.1f$" % (model_number, current_k)
        line, = ax.plot(r, z, label=model_label)
        lines.append(line)

    ax.set_title(r'Mirror Profiles', fontdict=font)
    ax.set_xlabel(r'$r=\sqrt{x_i^2+y_i^2}\, [\mathrm{%s}]$' % (omni_model.units), fontdict=font)
    ax.set_ylabel(r'$z\, [\mathrm{%s}]$' % (omni_model.units), fontdict=font)

    return ax, lines

def plot_spatial_resolution_vs_k(omni_model, min_elevation, max_elevation, k_values, in_square_pixels=False, in_steradians=True, use_spatial_resolution=True, ax=None, font=None):
    '''
    @param k_values: List of k values to plot about
    @param use_spatial_resolution: If True, it indicates to calculate the spatial resolution (as a ratio) in [area per st] units instead of [st per area].

    @retval the drawn axis corresponding to the parameter ax of the figure
    @retval lines: the set of plotted lines to be used with HighlightingDataCursor by the callee (if needed)
    '''
    from omnistereo.cata_hyper_model import get_r
    model_number = omni_model.mirror_number
    elevations = np.linspace(min_elevation, max_elevation, num=10)
    azimuths = np.zeros_like(elevations)  # The azimuths are all zero (w.l.o.g)

    world_points = omni_model.get_3D_point_from_angles_wrt_focus(azimuth=azimuths, elevation=elevations)
    reflection_point = omni_model.get_reflection_point_on_mirror(world_points)

    lines = []
    # First, plot the optimal k in the current model
    eta = omni_model.get_spatial_resolution(reflection_point, in_pixels=in_square_pixels, in_steradians=in_steradians, use_spatial_resolution=use_spatial_resolution)
    model_label = r"$k_%d = %0.1f ^{(\mathrm{Opt.})} \wedge r_{sys} = %0.1f \, [\mathrm{%s}]$" % (model_number, omni_model.k, omni_model.r_max, omni_model.units)
    line, = ax.plot(elevations, eta[-1], linestyle="--", linewidth=2, label=model_label)
    lines.append(line)

    for current_k in k_values:
        reflection_point = omni_model.get_reflection_point_on_mirror(world_points, k=current_k)
        eta = omni_model.get_spatial_resolution(reflection_point, in_pixels=in_square_pixels, in_steradians=in_steradians, use_spatial_resolution=use_spatial_resolution)

        if omni_model.mirror_number == 1:  # Top mirror
            p_w_for_rsys = omni_model.get_3D_point_from_angles_wrt_focus(azimuth=0, elevation=omni_model.highest_elevation_angle)
        else:
            p_w_for_rsys = omni_model.get_3D_point_from_angles_wrt_focus(azimuth=0, elevation=omni_model.lowest_elevation_angle)
        p_reflection_at_rsys = omni_model.get_reflection_point_on_mirror(p_w_for_rsys, k=current_k)
        current_rsys = get_r(x=p_reflection_at_rsys[0, 0, 0], y=p_reflection_at_rsys[0, 0, 1])
        model_label = r"$k_%d = %0.1f\wedge r_{sys} = %0.1f \, [\mathrm{%s}]$" % (model_number, current_k, current_rsys, omni_model.units)
        line, = ax.plot(elevations, eta[-1], label=model_label)
        lines.append(line)

    if in_square_pixels:
        res_unit_denom = r"\mathrm{pixel}^2"
    else:
        res_unit_denom = r"\mathrm{mm}^2"

    if in_steradians:
        res_unit_num = r"\mathrm{sr}"
        theta_angle_unit = r"rad"
    else:
        res_unit_num = r"\mathrm{degree}^2"
        theta_angle_unit = r"^{o}"

    if use_spatial_resolution:
        y_units = r'$\eta_{i}\,\left[\frac{\mathrm{%s}}{%s}\right]$' % (res_unit_denom, res_unit_num)
    else:
        y_units = r'$\eta_{i}\,\left[\frac{\mathrm{%s}}{%s}\right]$' % (res_unit_num, res_unit_denom)

    ax.set_title(r'Common vFOV Elevations', fontdict=font)
    ax.set_xlabel(r'$\mathrm{Elevation\, angle},\, \theta_{i}\, [\mathrm{%s}]$' % (theta_angle_unit), fontdict=font)
    ax.set_ylabel(r"$\mathrm{Spatial\, resolution},\,$" + y_units, fontdict=font)

    # Tweak spacing to prevent clipping of ylabel
    ax.grid(True)
#             ax.set_xlim(0)
#             ax.set_ylim(0, eta_max)
    return ax, lines

def plot_spatial_resln(omni_model, r, eta, eta_max, in_pixels, in_radians, use_spatial_resolution, in_2D, ax, fig_size=None):
    model_number = omni_model.mirror_number
    model_label = r"$\eta_{%d} \, \mathrm{for \, Mirror \, %d} $" % (model_number, model_number)

    is_parent_plot = False
    if ax == None:
        is_parent_plot = True
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
# Plot
    font = {'family':'serif', 'color':'darkblue',
        'weight':'normal',
        'size':16}
    font_big = {'family':'serif',
        'color':'black',
        'weight':'normal',
        'size':20}


    ax.plot(r, eta[-1], label=model_label)

    if is_parent_plot:
        # Tweak spacing to prevent clipping of ylabel
        ax.grid(True)
#             ax.set_xlim(0)
#             ax.set_ylim(0, eta_max)

        if in_2D:
            res_dim = r"2D "
            res_symbol = r"\eta_{i_{\mathrm{%s}}}" % res_dim
        else:
            res_dim = r"3D "
            res_symbol = r"\eta_{i}"

        if use_spatial_resolution == False:
            res_symbol = r"\frac{1}{%s}" % res_symbol

        if use_spatial_resolution:
            type_res = "Spatial"
        else:
            type_res = "Angular"

        if in_2D:
            power_str = r""
        else:
            power_str = r"^2"

        ax.set_title(res_dim + type_res + ' Resolution for the Omnistereo Catadioptric Sensor', fontdict=font)

        if in_pixels:
            res_unit_denom = r"\mathrm{pixel}%s" % power_str
        else:
            res_unit_denom = r"\mathrm{mm}%s" % power_str
        if in_radians:
            if in_2D:
                res_unit_num = r"\mathrm{rad}"
            else:
                res_unit_num = r"\mathrm{sr}"
        else:
            res_unit_num = r"\mathrm{degree}%s" % power_str

        if use_spatial_resolution:
            y_units = r'$%s\,\left[\frac{\mathrm{%s}}{%s}\right]$' % (res_symbol, res_unit_denom, res_unit_num)
        else:
            y_units = r'$%s\,\left[\frac{\mathrm{%s}}{%s}\right]$' % (res_symbol, res_unit_num, res_unit_denom)

        ax.set_ylabel(y_units, fontdict=font)

        x_units = r"[\mathrm{mm}%s]" % power_str
        ax.set_xlabel(r"$r\, %s$" % x_units, fontdict=font)

    return ax

def plot_spatial_resolution_2D(omni_model, r_min=0, r_max=1000, eta_max=None, in_pixels=False, in_radians=True, use_spatial_resolution=True, ax=None, fig_size=None):
    '''
    @param use_spatial_resolution: If True, it indicates to calculate the spatial resolution (as a ratio) in [area per st] units instead of [st per area].

    @return the drawn axis corresponding to the parameter ax of the figure
    '''
    model_number = omni_model.mirror_number
    r_size = 100
    r = np.linspace(r_min, r_max, r_size)  # Also x coordinates
    # TODO: get as projection to pixels
#         r_pixels = np.arange(-omni_model.image_size[0] / 2, omni_model.image_size[0] / 2)
#         r = r_pixels * omni_model.pixel_size[0]

    y = np.zeros_like(r)  # The y coordinates (all zeros)
    if model_number == 1:
        z = omni_model.get_z_hyperbola(r, y, is_virtual=False)
    elif model_number == 2:
        z = omni_model.get_z_hyperbola(r, y, is_virtual=False)

    p = np.dstack((r, y, z))
    eta = omni_model.get_spatial_resolution_in_2D(p, in_pixels=in_pixels, in_radians=in_radians, use_spatial_resolution=use_spatial_resolution)
    in_2D = True
#         if in_pixels:
#             r = r_pixels
    ax = plot_spatial_resln(omni_model, r, eta, eta_max, in_pixels, in_radians, use_spatial_resolution, in_2D, ax, fig_size=fig_size)

    return ax

def plot_spatial_resolution(omni_model, r_min=0, r_max=1000, eta_max=10.0, in_square_pixels=False, in_steradians=True, use_spatial_resolution=True, ax=None):
    '''
    @param use_spatial_resolution: If True, it indicates to calculate the spatial resolution (as a ratio) in [area per st] units instead of [st per area].

    @return the drawn axis corresponding to the parameter ax of the figure
    '''
    model_number = omni_model.mirror_number

    r_size = 100
    r = np.linspace(r_min, r_max, r_size)  # Also x coordinates
    y = np.zeros_like(r)  # The y coordinates (all zeros)
    if model_number == 1:
        z = omni_model.get_z_hyperbola(r, y, is_virtual=False)
    elif model_number == 2:  # Pay attention, the point will be given as the virtual point seen by the second focus
        z = omni_model.get_z_hyperbola(r, y, is_virtual=True)

    p = np.dstack((r, y, z))
    eta = omni_model.get_spatial_resolution(p, in_pixels=in_square_pixels, in_steradians=in_steradians, use_spatial_resolution=use_spatial_resolution)

    in_2D = False
    ax = plot_spatial_resln(omni_model, r, eta, eta_max, in_square_pixels, in_steradians, use_spatial_resolution, in_2D, ax)

    return ax

def plot_spatial_resolution_as_BakerNayar(omni_model, r_min=0, r_max=1000, eta_max=10.0, in_square_pixels=False, in_steradians=True, use_spatial_resolution=True, ax=None):
    '''
    @param use_spatial_resolution: If True, it indicates to calculate the spatial resolution (as a ratio) in [area per st] units instead of [st per area].

    @return the drawn axis corresponding to the parameter ax of the figure
    '''
    r_size = 100
    r, z = omni_model.get_2D_profile_wrt_itself(r_size)
    y = np.zeros_like(r)  # The y coordinates (all zeros)
    p = np.dstack((r, y, z))
    eta = omni_model.get_spatial_resolution_as_BakerNayar(p, in_pixels=in_square_pixels, in_steradians=in_steradians, use_spatial_resolution=use_spatial_resolution)

    in_2D = False
    ax = plot_spatial_resln(omni_model, r, eta, eta_max, in_square_pixels, in_steradians, use_spatial_resolution, in_2D, ax)
    return ax


def plot_k_vs_rsys_for_vFOV(omni_model, k_min=2.0, k_max=50, ax=None, fig_size=None):
    '''

    @return the drawn axis corresponding to the parameter ax of the figure
    '''
    from mpldatacursor import datacursor

    model_number = omni_model.mirror_number
    ks = np.linspace(k_min, k_max, 100, endpoint=True)

    is_parent_plot = False
    if ax == None:
        is_parent_plot = True
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        # fig, ax = plt.subplots()


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

    if is_parent_plot:
        # Tweak spacing to prevent clipping of ylabel
        ax.grid(True)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, k_max)

        ax.set_title(r'Effect of $k_{%d}$ on System Radius $r_{sys}$ for Several FOV Values $\alpha_%d$' % (model_number, model_number), fontdict=font)
        ax.set_xlabel('$r_{sys}\, [\mathrm{%s}]$' % (omni_model.units), fontdict=font)

        ax.set_ylabel(r'$k_{%d}$' % (model_number), fontdict=font)

        extra_yticks = [k_min]
        official_yticks = ax.get_yticks()
        ax.set_yticks(np.append(official_yticks, extra_yticks))
        ax.set_ylim(0, k_max)

        y_lims = ax.get_ylim()
        x_lims = ax.get_xlim()

        ax.axhline(k_min, color='red', linestyle='--', label="$k_{min}=%0.1f$" % (k_min))
        ax.text(max(x_lims) * (1 / 10), k_min + 1, r'$k_{min}$', rotation=0, size="16", color="red")


        ax.axvline(omni_model.r_max, color='black', linestyle='--')  # , label="$r_{sys}^{(\mathrm{Opt.})}=%0.2f \, \mathrm{mm}$" % (omni_model.r_max))
        ax.axvline(omni_model.r_min, color='red', linestyle=':')  # , label="$r_{ref}=%0.2f \, \mathrm{mm}$" % (omni_model.r_min))
        ax.text(omni_model.r_max - 6, max(y_lims) * (5 / 10), r'$r_{sys}^{(\mathrm{Opt.})}$', rotation=90, size="16", color="k")
        ax.text(omni_model.r_min - 3, max(y_lims) * (5 / 10), r'$r_{ref}$', rotation=90, size="16", color="r")

        # Info box
        str_c = r"$c_{%d} = %0.2f [\mathrm{%s}]$" % (model_number , omni_model.c, omni_model.units)
        str_reflex_radius = r"$r_{ref} = %0.2f [\mathrm{%s}]$" % (omni_model.r_min, omni_model. units)
        str_reflex_pos = r"$d/2 = %0.2f [\mathrm{%s}]$" % (omni_model.d / 2, omni_model. units)

        const_params_info = r'''Constant Parameters:
%s,
%s, %s''' % (str_c, str_reflex_radius, str_reflex_pos)

        ax.text(max(x_lims) * (6 / 10), max(y_lims) * (2 / 10), const_params_info, size=12, bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':10})

        datacursor(display='single', formatter='$k_1$={y:.1f}'.format, draggable=True, bbox=dict(alpha=1))
        # return ax, lines

    lines = []
    # First, plot the optimal k in the current model
    opt_vFOV = omni_model.get_vFOV_analytically()
    r_sys, k = omni_model.get_rsys_analytically(opt_vFOV, ki=ks, get_all=False)
    model_label = r"$\mathrm{for \, } \alpha_%d^{(\mathrm{Opt.})} = %0.1f^{\mathrm{o}}$" % (model_number, np.rad2deg(opt_vFOV))
    line, = ax.plot(r_sys, k, linestyle="--", linewidth=2, label=model_label)
    lines.append(line)

    vFOVs = np.linspace(np.deg2rad(10), np.deg2rad(70), 5)
    for current_vFOV in vFOVs:
        r_sys, k = omni_model.get_rsys_analytically(current_vFOV, ki=ks, get_all=False)
        model_label = r"$\mathrm{for \, } \alpha_%d = %0.1f^{\mathrm{o}}$" % (model_number, np.rad2deg(current_vFOV))
        line, = ax.plot(r_sys, k, label=model_label)
        lines.append(line)

    ax.legend().draggable()
    plt.tight_layout()
    plt.show()

#===============================================================================
# Using HyperCataStereo
#===============================================================================
def plot_k_vs_baseline_for_vFOV(omnistereo_model, k_min=2.0, k_max=20, ax=None, fig_size=None):
    '''

    @return the drawn axis corresponding to the parameter ax of the figure
    '''
    from mpldatacursor import datacursor  # , HighlightingDataCursor

    k1s = np.linspace(k_min, k_max, 100, endpoint=True)

    is_parent_plot = False
    if ax == None:
        is_parent_plot = True
        fig = plt.figure(figsize=fig_size)
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

    if is_parent_plot:
        # Tweak spacing to prevent clipping of ylabel
        ax.grid(True)
        ax.set_xlim(0, k_max)
        extra_cticks = [omnistereo_model.get_baseline()]
        official_yticks = ax.get_yticks()
        ax.set_yticks(np.append(official_yticks, extra_cticks))

        ax.set_ylim(1, 10e4)
        ax.set_yscale("log")
        ax.set_title(r'Effect of $k_1$ on Baseline $b$ for Various Values of $\alpha_{SROI}$', fontdict=font)
#             ax.set_title('Effect of $k_1$ on Baseline $b$', fontdict=font)

        ax.set_xlabel(r'$k_{1}$', fontdict=font)
        ax.set_ylabel(r'$b \, [\mathrm{%s}] \,\, \mathrm{(in\, log-scale)}$' % (omnistereo_model.units), fontdict=font)

        y_lims = ax.get_ylim()
        x_lims = ax.get_xlim()


#             ax.axvline(omnistereo_model.r_max, color='black', linestyle='--')  # , label="$r_{sys}^{(\mathrm{Opt.})}=%0.2f \, \mathrm{mm}$" % (omnistereo_model.r_max))
#             ax.axvline(omnistereo_model.r_min, color='red', linestyle=':')  # , label="$r_{ref}=%0.2f \, \mathrm{mm}$" % (omnistereo_model.r_min))
#             ax.text(omnistereo_model.r_max - 6, max(y_lims) * (5 / 10), r'$r_{sys}^{(\mathrm{Opt.})}$', rotation=90, size="16", color="k1")
#             ax.text(omnistereo_model.r_min - 3, max(y_lims) * (5 / 10), r'$r_{ref}$', rotation=90, size="16", color="r")

    lines = []
    cam = omnistereo_model.top_model.precalib_params

    # First, plot the optimal k1 in the current model
    opt_vFOV1 = omnistereo_model.top_model.get_vFOV_analytically()
    alpha_cam_min = np.min(cam.FOV)

#         b, (k1, k2), (c1, c2, d) = omnistereo_model.get_baseline_for_modified_FOV(alpha_cam_min, opt_vFOV1, k1=k1s, get_all=False)
    b, (k1, k2) = omnistereo_model.get_baseline_for_modified_FOV(alpha_cam_min, opt_vFOV1, k1=k1s, get_all=False)
    current_vFOV_com = omnistereo_model.get_FOVcom_from_FOV1(opt_vFOV1)
    model_label = r"$\mathrm{for \, } \alpha_{SROI}^{(\mathrm{Opt.})} = %0.1f^{\mathrm{o}}$" % (np.rad2deg(current_vFOV_com))
    line, = ax.plot(k1, b, linestyle="--", linewidth=2, label=model_label)
    lines.append(line)

    vFOVs = np.linspace(np.deg2rad(10), np.deg2rad(90 - np.rad2deg(omnistereo_model.get_max_elevations_theoretical()[0])), 5)
    for current_vFOV in vFOVs:
        current_vFOV_com = omnistereo_model.get_FOVcom_from_FOV1(current_vFOV)
#             b, (k1, k2), (c1, c2, d) = omnistereo_model.get_baseline_for_modified_FOV(alpha_cam_min, current_vFOV, k1=k1s, get_all=False)
        b, (k1, k2) = omnistereo_model.get_baseline_for_modified_FOV(alpha_cam_min, current_vFOV, k1=k1s, get_all=False)
        model_label = r"$\mathrm{for \, } \alpha_{SROI} = %0.1f^{\mathrm{o}}$" % (np.rad2deg(current_vFOV_com))
        line, = ax.plot(k1, b, linewidth=2, label=model_label)
        lines.append(line)


    extra_xticks = [k_min]
    official_xticks = ax.get_xticks()
    ax.set_xticks(np.append(official_xticks, extra_xticks))


    ax.axvline(k_min, color='red', linestyle=':')  # , label="$k_{min}=%0.1f$" % (k_min))
    ax.text(k_min - 0.8, 50, r'$k_{min}$', rotation=90, size="16", color="red")

    opt_baseline = omnistereo_model.get_baseline()
#         extra_yticks = [opt_baseline]
#         official_yticks = ax.get_yticks()
#         ax.set_yticks(np.append(official_yticks, extra_yticks))
    ax.axhline(opt_baseline, color='blue', linestyle=':', lw=2)
    ax.text(max(x_lims) * (7 / 10), opt_baseline + 10, r"$b^{(\mathrm{Opt.})}=%0.1f\, [\mathrm{%s}]$" % (opt_baseline, omnistereo_model.units), rotation=0, size="16", color="blue")

    # Info box
    str_cam_FOV = r"$\mathrm{Camera:} \, \alpha_{cam}(hor,vert)=(%0.1f^\mathrm{o}, %0.1f^\mathrm{o})$" % (np.rad2deg(cam.FOV[0]), np.rad2deg(cam.FOV[1]))
    str_focal_length = r"$f=%0.2f \, [\mathrm{mm}]$" % (cam.focal_length)
    str_reflex_radius = r"$r_{ref} = %0.2f [\mathrm{%s}]$" % (omnistereo_model.reflex_radius, omnistereo_model.units)
    str_k2 = r"$k_2 = %0.2f$" % (omnistereo_model.bot_model.k)

    const_params_info = r'''Constant Parameters:
%s, %s
%s, and %s''' % (str_cam_FOV, str_focal_length, str_reflex_radius, str_k2)

    ax.text(1, 2.5, const_params_info, size=12, bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':10})

#         ax.text(max(x_lims) * (2 / 10), 5e3, r"$b$-axis in logarithmic scale", size=14, bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

    # return ax, lines

    datacursor(display='single', formatter='$k_1$={x:.1f}'.format, draggable=True, bbox=dict(alpha=1))

    ax.legend().draggable()
    plt.tight_layout()
    plt.show()

def plot_catadioptric_spatial_resolution(omnistereo_model, in_2D=False, eta_max=None, fig_size=None):
    # Plot Camera Resolution
    r_min = 0  # mirror1.r_min
    r_max = 100  # mirror1.r_max
    in_pixels = True
    in_radians = False
    use_spatial_resolution = True
    mirror1 = omnistereo_model.top_model
    mirror2 = omnistereo_model.bot_model
    if in_2D:
        ax = plot_spatial_resolution_2D(mirror1, r_min, r_max, eta_max, in_pixels, in_radians, use_spatial_resolution, fig_size=fig_size)
        ax = plot_spatial_resolution_2D(mirror2, r_min, r_max, eta_max, in_pixels, in_radians, use_spatial_resolution, ax)
        # Note: test looks successful as for resolving the height of the sensor based on plotted spatial resolution as follows:
        # Mirror 1 covered sensor area: 0.004 [mm/degree] x vFOV of 33.69 [degrees] = 0.134 mm
        # Mirror 2 covered sensor area: 0.002 [mm/degree] x vFOV of 70.5 [ degrees] = 0.15 mm
        # area_1 + area_2 * 2 because is symmetric, we get around 0.55 mm + missing area outside mirror = 0.62

    else:
        ax = plot_spatial_resolution(mirror1, r_min, r_max, eta_max, in_pixels, in_radians, use_spatial_resolution, fig_size=fig_size)
        ax = plot_spatial_resolution(mirror2, r_min, r_max, eta_max, in_pixels, in_radians, use_spatial_resolution, ax)
    r_sys = omnistereo_model.system_radius
    r_reflex = omnistereo_model.reflex_radius
    r_cam = omnistereo_model.camera_hole_radius

    y_lims = ax.get_ylim()
    ax.axvline(r_sys, color='black', linestyle='--', label="$r_{sys}=%0.2f \, \mathrm{mm}$" % (r_sys))
    ax.axvline(r_reflex, color='red', linestyle=':', label="$r_{ref}=%0.2f \, \mathrm{mm}$" % (r_reflex))
    ax.axvline(r_cam, color='purple', linestyle='--', label="$r_{cam}=%0.2f \, \mathrm{mm}$" % (r_cam))
    ax.text(r_sys - 3, max(y_lims) * (7 / 10), r'$r_{sys}$', rotation=90, size="16", color="k")
    ax.text(r_reflex - 3, max(y_lims) * (7 / 10), r'$r_{ref}$', rotation=90, size="16", color="r")
    ax.text(r_cam - 3, max(y_lims) * (7 / 10), r'$r_{cam}$', rotation=90, size="16", color="purple")

    ax = add_info_box_on_plot(omnistereo_model, in_2D, ax)

    extraticks = []  # [int(r_reflex)]
    officialticks = ax.get_xticks()
    ax.set_xticks(np.append(officialticks, extraticks))

    ax.legend().draggable()
    plt.tight_layout()
    plt.show()


def plot_catadioptric_spatial_resolution_by_BakerNayar(omnistereo_model):
    # Plot Camera Resolution
    r_min = 0  # mirror1.r_min
    r_max = 100  # mirror1.r_max
    eta_max = None
    mirror1 = omnistereo_model.top_model
#     mirror2 = omnistereo_model.bot_model
    in_pixels = False
    in_radians = True
    use_spatial_resolution = True
    ax = plot_spatial_resolution_as_BakerNayar(mirror1, r_min, r_max, eta_max, in_pixels, in_radians, use_spatial_resolution)
#     ax = mirror2.plot_spatial_resolution(r_min, r_max, eta_max,  in_pixels, in_radians, use_spatial_resolution, ax)
    r_sys = omnistereo_model.system_radius
    r_reflex = omnistereo_model.reflex_radius
    r_cam = omnistereo_model.camera_hole_radius

    y_lims = ax.get_ylim()
    ax.axvline(r_sys, color='black', linestyle='--', label="$r_{sys}=%0.2f \, \mathrm{mm}$" % (r_sys))
    ax.axvline(r_reflex, color='red', linestyle=':', label="$r_{ref}=%0.2f \, \mathrm{mm}$" % (r_reflex))
    ax.axvline(r_cam, color='purple', linestyle='--', label="$r_{cam}=%0.2f \, \mathrm{mm}$" % (r_cam))
    ax.text(r_sys - 3, max(y_lims) * (7 / 10), r'$r_{sys}$', rotation=90, size="16", color="k")
    ax.text(r_reflex - 3, max(y_lims) * (7 / 10), r'$r_{ref}$', rotation=90, size="16", color="r")
    ax.text(r_cam - 3, max(y_lims) * (7 / 10), r'$r_{cam}$', rotation=90, size="16", color="purple")
    extraticks = []  # [int(r_reflex)]
    officialticks = ax.get_xticks()
    ax.set_xticks(np.append(officialticks, extraticks))

    ax.legend().draggable()
    plt.tight_layout()
    plt.show()


def plot_catadioptric_spatial_resolution_vs_k(omnistereo_model, fig_size=None, legend_location=None):
    # Plot Camera Resolution
    k_values = [4, 15, 30]
    r_max = 40
    eta_max = 1.6
    mirror1 = omnistereo_model.top_model
    mirror2 = omnistereo_model.bot_model
    in_square_pixels = False
    in_steradians = True
    use_spatial_resolution = True

    # Plot
    font = {'family' : 'serif',
            'color'  : 'darkblue',
            'weight' : 'normal',
            'size'   : 16,
            }
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=fig_size)
    title = r'Effect of $k_{i}$ on Resolution $\eta_{i}$ for the Common vFOV, $\alpha_{SROI}$'
    fig.canvas.set_window_title(title)
    plt.suptitle(title, fontdict=font)
#         fig.tight_layout()
    if isinstance(ax, np.ndarray):
        if ax.ndim == 1:
            ax = ax.reshape(1, 2)
    else:
        ax = np.array([[ax]])

    plot_mirror_profiles(omnistereo_model, k_values, ax[0, 0], font, is_standalone_plot=False, fig_size=None)
    ax_eta, lines = plot_spatial_resolution_vs_k(mirror1, omnistereo_model.common_lowest_elevation_angle, omnistereo_model.common_highest_elevation_angle, k_values, in_square_pixels, in_steradians, use_spatial_resolution, ax[0, 1], font)
    ax_eta, lines = plot_spatial_resolution_vs_k(mirror2, omnistereo_model.common_lowest_elevation_angle, omnistereo_model.common_highest_elevation_angle, k_values, in_square_pixels, in_steradians, use_spatial_resolution, ax_eta, font)
    # Set blue as the text color for the optimal value in the legend (the first line)
    leg = ax_eta.legend(loc=legend_location)
    texts_in_legend = leg.get_texts()
    # this part can be turned into a loop depends on the number of text objects
    texts_in_legend[0].set_color('b')
    leg.draggable()


    plt.show()

def plot_mirror_profiles(omnistereo_model, k_values=None, ax=None, font=None, is_standalone_plot=True, fig_size=None):
    if k_values == None:
        is_standalone_plot = True
        k_values = [2, 3, 4, 20, 40]
    r_max = omnistereo_model.system_radius
    mirror1 = omnistereo_model.top_model
    mirror2 = omnistereo_model.bot_model
    ax, lines = plot_profile(mirror1, k_values, r_max, ax, font, fig_size=fig_size)
    ax, lines = plot_profile(mirror2, k_values, r_max, ax, font, fig_size=fig_size)
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    r_reflex = omnistereo_model.reflex_radius
    r_cam = omnistereo_model.camera_hole_radius
    r_sys = omnistereo_model.system_radius

    ax.axvline(-r_sys, color='black', linestyle='--')  # , label="$r_{sys}=%0.2f \, \mathrm{mm}$" % (r_sys))
    ax.axvline(r_sys, color='black', linestyle='--')  # , label="$r_{sys}=%0.2f \, \mathrm{mm}$" % (r_sys))

    ax.axvline(-r_reflex, color='red', linestyle=':')  # , label="$r_{ref}=%0.2f \, \mathrm{mm}$" % (r_reflex))
    ax.axvline(r_reflex, color='red', linestyle=':')  # , label="$r_{ref}=%0.2f \, \mathrm{mm}$" % (r_reflex))

    ax.axvline(-r_cam, color='blue', linestyle=':')  # , label="$r_{cam}=%0.2f \, \mathrm{mm}$" % (r_reflex))
    ax.axvline(r_cam, color='blue', linestyle=':')  # , label="$r_{cam}=%0.2f \, \mathrm{mm}$" % (r_reflex))

    text_pos_y = np.sum(y_lims) / 2
    ax.text(r_sys - 8.5, text_pos_y, r'$r_{sys}$', rotation=90, size="16")
    ax.text(-r_sys + 0.5, text_pos_y, r'$r_{sys}$', rotation=90, size="16")
    ax.text(r_reflex - 0.5, text_pos_y, r'$r_{ref}$', rotation=90, size="16", color="r")
    ax.text(-r_reflex + 0.5, text_pos_y, r'$r_{ref}$', rotation=90, size="16", color="r")
    ax.text(r_cam - 0.5, text_pos_y, r'$r_{cam}$', rotation=90, size="16", color="blue")
    ax.text(-r_cam + 0.5, text_pos_y, r'$r_{cam}$', rotation=90, size="16", color="blue")


#     datacursor(display='single', formatter="{y:.2f}{label}".format, draggable=True)
#         datacursor(display='single', formatter='{label}'.format, draggable=True, bbox=dict(alpha=1))
#     HighlightingDataCursor(lines)

    # Completely Hide tick labels
    # [label.set_visible(False) for label in ax.get_xticklabels()]
    # or another approach
    # plt.setp(ax.get_xticklabels(), visible=False)  # ALL tick labels
    # plt.setp(ax.get_xmajorticklabels(), visible=False) # MAJOR tick labels
    # plt.setp(ax.get_xminorticklabels(), visible=False)  # MINOR tick labels

    # Hide only odd index labels:
    tick_labels = ax.get_xticklabels()
    [tick_labels[i].set_visible(False) for i in range(len(tick_labels)) if i % 2 != 0]

    if is_standalone_plot:
        # Set blue as the text color for the optimal value in the legend (the first line)
        leg = ax.legend()  # .draggable()
        texts_in_legend = leg.get_texts()
        # this part can be turned into a loop depends on the number of text objects
        texts_in_legend[0].set_color('b')
        leg.draggable()
        # Tweak spacing to prevent clipping of ylabel
        ax.grid(False)
        plt.show()

def add_info_box_on_plot(omnistereo_model, in_2D, ax):
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    mirror1 = omnistereo_model.top_model
    mirror2 = omnistereo_model.bot_model
    cam = mirror1.precalib_params
    str_focal_length = r"$\mathrm{focal\, length}: f=%0.2f \, [\mathrm{mm}]$" % (cam.focal_length)
    str_params1 = r"$c_1 = %0.2f [%s]$, $k_1 = %0.2f$ " % (mirror1.c, mirror1.units, mirror1.k)
    str_params2 = r"$c_2 = %0.2f [%s]$, $k_2 = %0.2f$ " % (mirror2.c, mirror2.units, mirror2.k)
    str_vFOV1 = r"$vFOV_1 \approx %0.2f ^\mathrm{o}$" % np.rad2deg(mirror1.vFOV)
    str_radial_img_height1 = r"$h_{I_1} = %d [\mathrm{px}]$" % (mirror1.h_radial_image)
    str_vFOV2 = r"$vFOV_2 \approx %0.2f ^\mathrm{o}$" % np.rad2deg(mirror2.vFOV)
    str_radial_img_height2 = r"$h_{I_2} = %d [\mathrm{px}]$" % (mirror2.h_radial_image)

    if in_2D:
        str_pixel_dims = r"$\mathrm{pixel\, size} \approx %0.2e \, [\mathrm{mm}]$" % (cam.pixel_size[0])
    else:
        str_pixel_dims = r"$\mathrm{pixel\, area} \approx %0.2e \, [\mathrm{mm}^2]$" % (cam.pixel_area)

    camera_info = r'''Misc. Information:
%s,
%s,
%s,
%s,
%s, %s
%s, %s''' % (str_focal_length, str_pixel_dims, str_params1, str_params2, str_vFOV1, str_radial_img_height1, str_vFOV2, str_radial_img_height2)

    ax.text(max(x_lims) * (6.45 / 10), max(y_lims) * (1 / 30), camera_info, size=12, bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':10})

    return ax

def plot_effect_of_pixel_disparity_on_range(omnistereo_model, disp_min=1, disp_max=100, disp_nums=4, use_log=True, plot_zoom=False, ax=None, fig_size=None):
    '''

    @return the drawn axis corresponding to the parameter ax of the figure
    '''
    if plot_zoom:
        cut_disp = np.sqrt(disp_max / 2)
    else:
        cut_disp = disp_min

    if use_log:
        disparities = np.around(np.logspace(np.log10(cut_disp + 1), np.log10(disp_max), disp_nums, endpoint=True))
    else:
        disparities = np.around(np.linspace(cut_disp + 1, disp_max, disp_nums, endpoint=True))

    is_parent_plot = False
    if ax == None:
        is_parent_plot = True
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)

    # Plot
    font = {'family' : 'serif',
            'color'  : 'darkblue',
            'weight' : 'normal',
            'size'   : 16,
            }
    font_small = {'family' : 'serif',
            'color'  : 'darkblue',
            'weight' : 'normal',
            'size'   : 10,
            }

    if is_parent_plot:
        # Tweak spacing to prevent clipping of ylabel
        ax.grid(True)
        ax.set_xlim(0, 8)
        ax.set_ylim(-1.0, 1.2)
#             extra_cticks = [omnistereo_model.get_baseline()]
#             official_yticks = ax.get_yticks()
#             ax.set_yticks(np.append(official_yticks, extra_cticks))

        ax.set_title(r'Effect of Omnistereo Pixel Disparity $(\Delta m_{12})$ on Range $(\rho_w, z_w)$', fontdict=font)
        ax.set_xlabel(r'$\mathrm{Horizontal\, range},\,\rho_w \, [\mathrm{%s}]$' % ("m"), fontdict=font)
        ax.set_ylabel(r'$\mathrm{Vertical\, range},\,z_w \, [\mathrm{%s}]$' % ("m"), fontdict=font)

#             y_lims = ax.get_ylim()
#             x_lims = ax.get_xlim()
#             ax.axvline(omnistereo_model.r_max, color='black', linestyle='--')  # , label="$r_{sys}^{(\mathrm{Opt.})}=%0.2f \, \mathrm{mm}$" % (omnistereo_model.r_max))
#             ax.axvline(omnistereo_model.r_min, color='red', linestyle=':')  # , label="$r_{ref}=%0.2f \, \mathrm{mm}$" % (omnistereo_model.r_min))
#             ax.text(omnistereo_model.r_max - 6, max(y_lims) * (5 / 10), r'$r_{sys}^{(\mathrm{Opt.})}$', rotation=90, size="16", color="k")
#             ax.text(omnistereo_model.r_min - 3, max(y_lims) * (5 / 10), r'$r_{ref}$', rotation=90, size="16", color="r")


    lines = []
    for disp in disparities:
        triangulated_points = omnistereo_model.get_triangulated_points_from_pixel_disp(disparity=disp)
        model_label = r"$\Delta m_{12} = %0.f \, [\mathrm{px}]$" % (disp)
        line, = ax.plot(triangulated_points[0, :, 0] / 1000., triangulated_points[0, :, 2] / 1000., linewidth=2, label=model_label)
        lines.append(line)

    if plot_zoom:
        disparities_zoom = np.arange(disp_min, cut_disp)
        subpos = [0.5, 0.1, 0.45, 0.4]  # Positions are given as percentages, so that it reads [x, y, width, height]
        subax1 = add_subplot_axes(ax, subpos)
        subax1.grid(True)
        subax1.set_title(r'Effect on Range for $%0.f \leq \Delta m_{12} \leq %0.f \, [\mathrm{pixels}]$' % (disparities_zoom.min(), disparities_zoom.max()), fontdict=font_small)
        subax1.set_xlabel(r'$\rho_w \, [\mathrm{%s}]$' % ("m"), fontdict=font_small)
        subax1.set_ylabel(r'$\,z_w \, [\mathrm{%s}]$' % ("m"), fontdict=font_small)

        for disp in disparities_zoom:
            triangulated_points = omnistereo_model.get_triangulated_points_from_pixel_disp(disparity=disp)
            if disp == disparities_zoom.min() or disp == disparities_zoom.max():
                model_label = r"$\Delta m_{12} = %0.f \, [\mathrm{px}]$" % (disp)
            else:
                model_label = None
            line, = subax1.plot(triangulated_points[0, :, 0] / 1000, triangulated_points[0, :, 2] / 1000, linewidth=2, label=model_label)
            lines.append(line)

            subax1.legend(prop={'size':10}).draggable()

        # Info box
#         str_cam_FOV = r"$\mathrm{Camera:} \, \alpha_{cam}(hor,vert)=(%0.1f^\mathrm{o}, %0.1f^\mathrm{o})$" % (np.rad2deg(cam.FOV[0]), np.rad2deg(cam.FOV[1]))
#         str_focal_length = r"$f=%0.2f \, [\mathrm{mm}]$" % (cam.focal_length)
#         str_reflex_radius = r"$r_{ref} = %0.2f [\mathrm{%s}]$" % (omnistereo_model.reflex_radius, omnistereo_model.units)
#         str_k2 = r"$k_2 = %0.2f$" % (omnistereo_model.bot_model.k)
#
#         const_params_info = r'''Constant Parameters:
# %s, %s
# %s, and %s''' % (str_cam_FOV, str_focal_length, str_reflex_radius, str_k2)
#
#         ax.text(1, 2.5, const_params_info, size=12, bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':10})

        # return ax, lines

        ax.legend().draggable()
        plt.show()


def plot_vertical_range_variation(omnistereo_model, hor_range_max=1000, depth_nums=2, use_meters=False, fig_size=None):
    '''

    @return the drawn axis corresponding to the parameter ax1 of the figure
    '''
    if use_meters:
        scale = 1.e-3
        str_units = "m"
    else:
        scale = 1.0
        str_units = omnistereo_model.units

    Pns_low, Pns_mid, Pns_high = omnistereo_model.get_stereo_ROI_nearest_vertices()
    hor_depth_min = max(Pns_low[0, 0, 0], Pns_high[0, 0, 0])
    hor_const_depths = np.around(np.logspace(np.log10(hor_depth_min * scale), np.log10(hor_range_max), depth_nums, endpoint=True))

    is_parent_plot = True
    fig = plt.figure(figsize=fig_size)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

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

    if is_parent_plot:
        # Tweak spacing to prevent clipping of ylabel
        ax1.grid(True)
        ax2.grid(True)
#             ax1.set_xlim(0, 36)
#             extra_cticks = [omnistereo_model.get_baseline()]
#             official_yticks = ax1.get_yticks()
#             ax1.set_yticks(np.append(official_yticks, extra_cticks))

        plt.suptitle(r'Vertical Range Variation $\Delta z$ Along $^{{}_{[\mathrm{C}]}}z_w$ Levels', fontdict=font)

        ax1.set_title('Mirror 1')
        ax2.set_title('Mirror 2')

        ax1.set_xlabel(r'$^{{}_{[\mathrm{C}]}}z_w \, [\mathrm{%s}]$' % (str_units), fontdict=font)
        ax2.set_xlabel(r'$^{{}_{[\mathrm{C}]}}z_w \, [\mathrm{%s}]$' % (str_units), fontdict=font)
        ax1.set_ylabel(r'$\mathrm{Vertical\, Range \, Variation},\,\Delta z \, [\mathrm{%s}]$' % (str_units), fontdict=font)
#             ax2.set_ylabel(r'$\mathrm{Vertical\, Range \, Resolution},\,\kappa_z \, \left[\frac{\mathrm{%s}}{\mathrm{pixel}}\right]$' % (omnistereo_model.units), fontdict=font)

    lines = []
    # Mirror 1
    for hor_const in hor_const_depths:
        delta_z1, level_z1 = omnistereo_model.top_model.get_vertical_range_variation(hor_const / scale)
        model_label = r"$\rho_w = %0.2f \, [\mathrm{%s}]$" % (hor_const, str_units)
        line, = ax1.plot(level_z1[0, :] * scale, delta_z1[0, :] * scale, linewidth=2, label=model_label)
        lines.append(line)
        delta_z2, level_z2 = omnistereo_model.bot_model.get_vertical_range_variation(hor_const / scale)
        model_label = r"$\rho_w = %0.2f \, [\mathrm{%s}]$" % (hor_const, str_units)
        line, = ax2.plot(level_z2[0, :] * scale, delta_z2[0, :] * scale, linewidth=2, label=model_label)
        lines.append(line)

#         ax1.legend().draggable()
    ax2.legend().draggable()
#         plt.tight_layout()
    plt.show()


def plot_horizontal_range_variation(omnistereo_model, vertical_range_min=-1000, vertical_range_max=1000, depth_nums=2, use_meters=False, fig_size=None):
    '''

    @return the drawn axis corresponding to the parameter ax1 of the figure
    '''
    if use_meters:
        scale = 1.e-3
        str_units = "m"
    else:
        scale = 1.0
        str_units = omnistereo_model.units

    vert_const_depths = np.linspace(vertical_range_min, vertical_range_max, depth_nums, endpoint=True)

    is_parent_plot = True
    fig = plt.figure(figsize=fig_size)
    ax1 = fig.add_subplot(111)
#         ax1 = fig.add_subplot(121)
#         ax2 = fig.add_subplot(122)

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

    if is_parent_plot:
        # Tweak spacing to prevent clipping of ylabel
        ax1.grid(True)
#             ax2.grid(True)
#             ax1.set_xlim(0, 36)
#             extra_cticks = [omnistereo_model.get_baseline()]
#             official_yticks = ax1.get_yticks()
#             ax1.set_yticks(np.append(official_yticks, extra_cticks))

        plt.suptitle(r'Horizontal Range Variation $\Delta \rho$ Along $^{{}_{[\mathrm{C}]}}\rho_w$ Levels', fontdict=font)

        ax1.set_title('Mirror 1')

        ax1.set_xlabel(r'$^{{}_{[\mathrm{C}]}}\rho_w \, [\mathrm{%s}]$' % (str_units), fontdict=font)
        ax1.set_ylabel(r'$\mathrm{Horizontal\, Range \, Variation},\,\Delta \rho \, [\mathrm{%s}]$' % (str_units), fontdict=font)
#             ax2.set_title('Mirror 2')
#             ax2.set_xlabel(r'$^{{}_{[\mathrm{C}]}}\rho_w \, [\mathrm{%s}]$' % (str_units), fontdict=font)
#             ax2.set_ylabel(r'$\mathrm{Vertical\, Range \, Resolution},\,\kappa_z \, \left[\frac{\mathrm{%s}}{\mathrm{pixel}}\right]$' % (omnistereo_model.units), fontdict=font)

    lines = []
    # Mirror 1
    for vert_const in vert_const_depths:
        delta_rho1, level_rho1 = omnistereo_model.top_model.get_horizontal_range_variation(vert_const / scale)
        model_label = r"$z_w = %0.2f \, [\mathrm{%s}]$" % (vert_const, str_units)
        line, = ax1.plot(level_rho1[0, :] * scale, delta_rho1[0, :] * scale, linewidth=2, label=model_label)
        lines.append(line)
#             delta_rho2, level_rho2 = omnistereo_model.bot_model.get_horizontal_range_variation(vert_const / scale)
#             model_label = r"$z_w = %0.2f \, [\mathrm{%s}]$" % (vert_const, str_units)
#             line, = ax2.plot(level_rho2[0, :] * scale, delta_rho2[0, :] * scale, linewidth=2, label=model_label)
#             lines.append(line)

    ax1.legend().draggable()
#         ax2.legend().draggable()
#         plt.tight_layout()
    plt.show()

#===============================================================================
# 2D Plots for OmniStereoModel (more general than HyperCataStereo)
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
def plot_range_variation_due_to_pixel_disparity(omnistereo_model, disp_min=1, disp_max=100, ax=None, fig_size=None):
    '''

    @return the drawn axis corresponding to the parameter ax of the figure
    '''
    disparities = np.arange(disp_min, disp_max + 1)

    is_parent_plot = False
    if ax == None:
        is_parent_plot = True
        fig = plt.figure(figsize=fig_size)
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

    if is_parent_plot:
        # Tweak spacing to prevent clipping of ylabel
        ax.grid(False)

        ax.set_title(r'Horizontal Range Variation due to Pixel Disparity', fontdict=font)
        ax.set_xlabel(r'$\mathrm{Pixel\, Disparity},\, \Delta m_{12} \, [\mathrm{px}]$', fontdict=font)
        ax.set_ylabel(r'$\mathrm{Horizontal\, Range\, Variation},\, \Delta \rho_w \, [\mathrm{%s}]$' % ("m"), fontdict=font)

        subpos = [0.3, 0.3, 0.7, 0.7]  # Positions are given as percentages, so that it reads [x, y, width, height]
        subax1 = add_subplot_axes(ax, subpos)
        subax1.grid(True)
        subax1.set_xlabel(r'$\Delta m_{12} \, [\mathrm{px}]$', fontdict=font)
        subax1.set_ylabel(r'$\Delta \rho_w \, [\mathrm{%s}]$' % ("m"), fontdict=font)

    _, _, m1 = omnistereo_model.top_model.get_pixel_from_direction_angles(azimuth=np.array([[[0]]]), elevation=np.array([[[0]]]))
    triangulated_points = omnistereo_model.get_triangulated_points_from_pixel_disp(disparity=disparities, m1=m1)
    range_variation = (triangulated_points[0, 0:-1, 0] - triangulated_points[0, 1:, 0]) / 1000
    disp_cutoff = 20
    ax.plot(disparities[:disp_cutoff], range_variation[:disp_cutoff], linestyle=":", marker="o", color="red")
    subax1.plot(disparities[disp_cutoff:-1], range_variation[disp_cutoff:], linestyle=":", marker="o", color="red")
    # return ax, lines

#         ax.legend().draggable()
#         plt.tight_layout()
    plt.show()

