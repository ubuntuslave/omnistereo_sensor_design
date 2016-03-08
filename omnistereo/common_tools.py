# -*- coding: utf-8 -*-
# common_tools.py

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

from __future__ import division
from __future__ import print_function

import warnings
import numpy as np
# import dill as pickle
# import cPickle as pickle
import pickle

import os
import errno

def get_current_os_name():
    from sys import platform as _platform
    if _platform == "linux" or _platform == "linux2":
        return "linux"  # linux
    elif _platform == "darwin":
        return "mac"  # OS X
    elif _platform == "win32" or _platform == "cygwin":
        return "windows"  # Windows...

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def get_namestr(obj, namespace):
    '''
    Finds and returns the variable name for the object instance in question in the desired namespace.
    For example

        >>> get_namestr(my_wow, globals())
        Out: 'my_wow'
    '''
    if namespace:
        results = [name for name in namespace if namespace[name] is obj]
        if len(results) == 1:
            return results[0]
        else:
            return results
    else:
        return ""

def get_theoretical_params_from_file(filename, file_units="cm"):
    # NOTE: This only works assuming the model in this project's implementation uses milimeters as the units
    if file_units == "cm":
        scale = 10.
    if file_units == "mm":
        scale = 1.

    scales = [scale, scale, 1., 1., scale, scale, scale, scale]  # Recall, k1 and k2 are dimensionless
    params_file = open(filename, 'r')
    params_line = params_file.readline().rstrip('\n')  # Read first and only line of this params file
    params_file.close()
    c1, c2, k1, k2, d, r_sys, r_reflex, r_cam = scales * np.array(params_line.split(","), dtype=np.float64)  # The rows, cols, width_row, width_col, margin are saved on the first line
    return c1, c2, k1, k2, d, r_sys, r_reflex, r_cam

def save_obj_in_pickle(obj_instance, filename, namespace=None):
    print("Saving %s instance to pickle file %s ..." % (get_namestr(obj_instance, namespace), filename), end="")
    f = open(filename, 'wb')  # Create external f
    pickle.dump(obj_instance, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    print("done!")

def load_obj_from_pickle(filename, namespace=None):
    f = open(filename, 'rb')  # Create external f
    obj_instance = pickle.load(f)
    print("Loading %s instance from pickle file %s ... " % (get_namestr(obj_instance, locals()), filename), end="")
    f.close()
    print("done!")
    return obj_instance

def pdf(point, cons, mean, det_sigma):
    if isinstance(mean, np.ndarray):
        return cons * np.exp(-(np.dot((point - mean), det_sigma) * (point - mean)) / 2.)
    else:
        return cons * np.exp(-((point - mean) / det_sigma) ** 2 / 2.)

def reverse_axis_elems(arr, k=0):
    '''
    @param arr: the numpy ndarray to be reversed
    @param k: The axis to be reversed
        Reverse the order of rows: set axis k=0
        Reverse the order of columns: set axis k=1

    @return: the reversed numpy array
    '''
    reversed_arr = np.swapaxes(np.swapaxes(arr, 0, k)[::-1], 0, k)
    return reversed_arr

def flatten(x):
    '''
    Iteratively flattens the elements of a list of lists (or any other iterable such as tuples)
    '''
    import collections
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def rms(x, axis=None):
    return np.sqrt(np.mean(np.square(x), axis=axis))


def nanrms(x, axis=None):
    '''
    If you have nans in your data, you can do
    '''
    return np.sqrt(np.nanmean(np.square(x), axis=axis))

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)  # Fast and numerically precise
    return average, np.sqrt(variance)

def mean_and_std(values):
    """
    Return the arithmetic mean or unweighted average and the standard deviation.

    @param values: Numpy ndarrays of values
    """
#     average = np.mean(values)
    average = np.mean(values)
    std_dev = np.std(values)

    return average, std_dev

def error_analysis_simple(errors, units):
    mean, std_dev = mean_and_std(np.array(errors))
    max_error = errors[np.argmax(errors) % 3]
    print("ERROR ANALYSIS: mean = {mean:.4f} [{units}], std. dev = {std:.4f} [{units}], MAX = {max:.4f} [{units}]".format(mean=mean, std=std_dev, max=max_error, units=units))

def get_transformation_matrix(pose_vector):
    '''
    Concatenates a transformation matrix using the rotation (quaternion) and translation components of the pose_vector

    @param pose_vector: A list (7-vector) of the rotation quaternion components [w, x, y, z] that imply w+ix+jy+kz and  translation components [tx, ty, tz].

    @return: the homogeneous transformation matrix encoding the rotation followed by a translation given as the pose_vector
    '''
    import omnistereo.transformations as tr
    # Normalize to unit quaternion:
    q = pose_vector[0:4]
    q_normalized = q / np.linalg.norm(q)
    rot_matrix = tr.quaternion_matrix(q_normalized)
    trans_matrix = tr.translation_matrix(pose_vector[4:])
    T = tr.concatenate_matrices(trans_matrix, rot_matrix)
    return T

def get_inversed_transformation(T):
    '''
    Let's call T = [R|t] transformation matrix, applies the rule T_inv = [R^transpose | -R^transpose * t]
    @param T: 4x4 matrix [R|t] as the mixture of 3x3 rotation matrix R and translation 3D vector t.

    @note: This gives the same results as the inverse_matrix from the transformations.py module or simply numpy.linalg.inv(matrix)
    '''
    dims = len(T)
    T_inv = np.identity(dims)
    R_transposed = T[:dims - 1, :dims - 1].T
    translation = T[:dims - 1, dims - 1]
    T_inv[:dims - 1, :dims - 1] = R_transposed
    T_inv[:dims - 1, dims - 1] = -np.dot(R_transposed, translation)

    return T_inv


def get_2D_points_normalized(points_3D):
    '''
    @param points_3D: Numpy array of points 3D coordinates
    @return: new_pts_2D -  The array of transformed 2D homogeneous coordinates where the scaling parameter is normalised to 1.
             T      -  The 3x3 transformation matrix, such that new_pts_2D <-- T * points_normalized
    '''
    # Ensure homogeneous coords have scale of 1 because of normalization type INF
    # Similar to calling the OpenCV normalize function, such as
    # >>>> dst = normalize(points_3D[...,:3], norm_type=cv2.NORM_INF)
    pts_normalized = np.ones_like(points_3D[..., :3])
    pts_normalized[..., :2] = points_3D[..., :2] / np.abs(points_3D[..., 2, np.newaxis])

    centroid = np.mean(pts_normalized[..., :2].reshape(-1, 2), axis=(0))
    # Shift origin to centroid.
    pts_offset = pts_normalized[..., :2] - centroid
    meandist = np.mean(np.linalg.norm(pts_offset))
    scale = np.sqrt(2) / meandist  # because sqrt(1^2 + 1^2)

    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])

    new_pts_2D = np.einsum("ij, mnj->mni", T, pts_normalized)
    # TEST:
    # np.all([np.allclose(np.dot(T, pts_normalized[i,j]),new_pts_2D[i,j]) for (i,j), x in np.ndenumerate(new_pts_2D[...,0])])

    return new_pts_2D, T

import sys
PYTHON_V3 = sys.version_info >= (3, 0, 0) and sys.version_info < (4, 0, 0)
def get_screen_resolution(measurement="px"):
    """
    Tries to detect the screen resolution from the system.
    @param measurement: The measurement to describe the screen resolution in. Can be either 'px', 'inch' or 'mm'.
    @return: (screen_width,screen_height) where screen_width and screen_height are int types according to measurement.
    """
    mm_per_inch = 25.4
    px_per_inch = 72.0  # most common
    try:  # Platforms supported by GTK3, Fx Linux/BSD
        from gi.repository import Gdk
        screen = Gdk.Screen.get_default()
        if measurement == "px":
            width = screen.get_width()
            height = screen.get_height()
        elif measurement == "inch":
            width = screen.get_width_mm() / mm_per_inch
            height = screen.get_height_mm() / mm_per_inch
        elif measurement == "mm":
            width = screen.get_width_mm()
            height = screen.get_height_mm()
        else:
            raise NotImplementedError("Handling %s is not implemented." % measurement)
        return (width, height)
    except:
        try:
            from PyQt4 import QtGui
            import sys
            app = QtGui.QApplication(sys.argv)
            screen_rect = app.desktop().screenGeometry()
            width, height = screen_rect.width(), screen_rect.height()
            return (width, height)
        except:
            try:  # Probably the most OS independent way
                import tkinter
                root = tkinter.Tk()
                if measurement == "px":
                    width = root.winfo_screenwidth()
                    height = root.winfo_screenheight()
                elif measurement == "inch":
                    width = root.winfo_screenmmwidth() / mm_per_inch
                    height = root.winfo_screenmmheight() / mm_per_inch
                elif measurement == "mm":
                    width = root.winfo_screenmmwidth()
                    height = root.winfo_screenmmheight()
                else:
                    raise NotImplementedError("Handling %s is not implemented." % measurement)
                return (width, height)
            except:
                try:  # Windows only
                    from win32api import GetSystemMetrics
                    width_px = GetSystemMetrics (0)
                    height_px = GetSystemMetrics (1)
                    if measurement == "px":
                        return (width_px, height_px)
                    elif measurement == "inch":
                        return (width_px / px_per_inch, height_px / px_per_inch)
                    elif measurement == "mm":
                        return (width_px / mm_per_inch, height_px / mm_per_inch)
                    else:
                        raise NotImplementedError("Handling %s is not implemented." % measurement)
                except:
                    try:  # Windows only
                        import ctypes
                        user32 = ctypes.windll.user32
                        width_px = user32.GetSystemMetrics(0)
                        height_px = user32.GetSystemMetrics(1)
                        if measurement == "px":
                            return (width_px, height_px)
                        elif measurement == "inch":
                            return (width_px / px_per_inch, height_px / px_per_inch)
                        elif measurement == "mm":
                            return (width_px / mm_per_inch, height_px / mm_per_inch)
                        else:
                            raise NotImplementedError("Handling %s is not implemented." % measurement)
                    except:
                        try:  # Mac OS X only
                            import AppKit
                            for screen in AppKit.NSScreen.screens():
                                width_px = screen.frame().size.width
                                height_px = screen.frame().size.height
                                if measurement == "px":
                                    return (width_px, height_px)
                                elif measurement == "inch":
                                    return (width_px / px_per_inch, height_px / px_per_inch)
                                elif measurement == "mm":
                                    return (width_px / mm_per_inch, height_px / mm_per_inch)
                                else:
                                    raise NotImplementedError("Handling %s is not implemented." % measurement)
                        except:
                            try:  # Linux/Unix
                                import Xlib.display
                                resolution = Xlib.display.Display().screen().root.get_geometry()
                                width_px = resolution.width
                                height_px = resolution.height
                                if measurement == "px":
                                    return (width_px, height_px)
                                elif measurement == "inch":
                                    return (width_px / px_per_inch, height_px / px_per_inch)
                                elif measurement == "mm":
                                    return (width_px / mm_per_inch, height_px / mm_per_inch)
                                else:
                                    raise NotImplementedError("Handling %s is not implemented." % measurement)
                            except:
                                try:  # Linux/Unix
                                    if not self.is_in_path("xrandr"):  # FIXME: implement is_in_path function
                                        raise ImportError("Cannot read the output of xrandr, if any.")
                                    else:
                                        args = ["xrandr", "-q", "-d", ":0"]
                                        proc = subprocess.Popen(args, stdout=subprocess.PIPE)
                                        for line in iter(proc.stdout.readline, ''):
                                            if isinstance(line, bytes):
                                                line = line.decode("utf-8")
                                            if "Screen" in line:
                                                width_px = int(line.split()[7])
                                                height_px = int(line.split()[9][:-1])
                                                if measurement == "px":
                                                    return (width_px, height_px)
                                                elif measurement == "inch":
                                                    return (width_px / px_per_inch, height_px / px_per_inch)
                                                elif measurement == "mm":
                                                    return (width_px / mm_per_inch, height_px / mm_per_inch)
                                                else:
                                                    raise NotImplementedError("Handling %s is not implemented." % measurement)
                                except:
                                    # Failover
                                    screensize = 1366, 768
                                    sys.stderr.write("WARNING: Failed to detect screen size. Falling back to %sx%s" % screensize)
                                    if measurement == "px":
                                        return screensize
                                    elif measurement == "inch":
                                        return (screensize[0] / px_per_inch, screensize[1] / px_per_inch)
                                    elif measurement == "mm":
                                        return (screensize[0] / mm_per_inch, screensize[1] / mm_per_inch)
                                    else:
                                        raise NotImplementedError("Handling %s is not implemented." % measurement)

# TODO: We don't really need this function anymore since data is loaded via drivers (see vicon_utils.py).
def get_poses_from_file(grid_poses_filename, input_units="cm", model_working_units="mm", indices=None):
    '''
    @param poses_filename:  the comma separated file for the rig's pose information at each instance (used by POV-Ray)
    @param indices: the indices for the working images

    @return A list of pose lists (7-vector) indicating the rotation quaternion components [w, x, y, z] that imply w+ix+jy+kz and translation components [tx, ty, tz].
    @return A list of transform matrices (homogeneous matrix) encoding the rotation (first) followed by a translation for the given frames wrt to the scene (reference frame)
    '''
    import omnistereo.transformations as tr

    if isinstance(grid_poses_filename, str):  # It's a filename
        grid_poses_from_file = np.loadtxt(grid_poses_filename, delimiter=',', usecols=(0, 1, 2, 3, 4, 5), comments="#", unpack=False)
        if len(grid_poses_from_file) == 0:
            print("Couldn't find grid poses in file")
            print("Exiting from", __name__)
            exit(1)

    list_len = len(indices)
    grid_poses_list = list_len * [None]
    transform_matrices_list = list_len * [None]
    for pose_number in indices:
        try:
            pose_info_list = grid_poses_from_file[pose_number]
            # pose info will be given as a list (7-vector) of the rotation quaternion components [w, x, y, z] followed by translation components [tx, ty, tz].

            pose_info = 7 * [0.0]
            unit_conversion_factor = 1.0
            if input_units == "cm":
                if model_working_units == "mm":
                    unit_conversion_factor = 10.0

            # We grab the values (given as RHS)
            pose_info[4] = unit_conversion_factor * float(pose_info_list[0])
            pose_info[5] = unit_conversion_factor * float(pose_info_list[1])
            pose_info[6] = unit_conversion_factor * float(pose_info_list[2])
            # Camera frame [C] pose wrt to Scene frame [S]
            CwrtS_angle_rot_x = np.deg2rad(float(pose_info_list[3]))  # because rotations in POV-Ray are given in degrees
            CwrtS_angle_rot_y = np.deg2rad(float(pose_info_list[4]))  # because rotations in POV-Ray are given in degrees
            CwrtS_angle_rot_z = np.deg2rad(float(pose_info_list[5]))  # because rotations in POV-Ray are given in degrees
            #  In our RHS, the order of rotations are rotX --> rotY --> rotZ

            CwrtS_rot_q = tr.quaternion_from_euler(CwrtS_angle_rot_x, CwrtS_angle_rot_y, CwrtS_angle_rot_z, 'sxyz')
            [pose_info[0], pose_info[1], pose_info[2], pose_info[3]] = CwrtS_rot_q
            # Fill up the result lists:
            grid_poses_list[pose_number] = pose_info
            transform_matrices_list[pose_number] = get_transformation_matrix(pose_info)
        except:  # catch *all* exceptions
            err_msg = sys.exc_info()[1]
            warnings.warn("Warning...%s" % (err_msg))
            print("Exiting from", __name__)
            sys.exit(1)

    return grid_poses_list, transform_matrices_list

def test_corner_detection_individually(model, calibrator, images_path_as_template, corner_extraction_args, img_indices):
    from omnistereo.calibration import draw_detected_points_manually
    calibrator.calibrate(model, images_path_as_template, img_indices, corner_extraction_args)
    draw_detected_points_manually(model.calibrator.omni_monos[0].omni_image, model.calibrator.omni_monos[0].image_points, -3, show=True)

#     print("DEBUG: %s image_points" % (model.mirror_name), model.calibrator.omni_monos[0].image_points)
#     print("DEBUG: %s Top obj_points" % (model.mirror_name), model.calibrator.omni_monos[0].obj_points)


def test_corner_detection_stereo(omnistereo_model, calibrator, images_path_as_template, corner_extraction_args_top, corner_extraction_args_bottom, img_indices):
    # Corner selection test:
    calibrator.calibrate(omnistereo_model, images_path_as_template, img_indices, corner_extraction_args_top, corner_extraction_args_bottom)

#     print("DEBUG: image_points", calibrator.calib_top.calibration_pairs[0].calib_img_top.image_points)
#     print("DEBUG: obj_points", calibrator.calib_bottom.calibration_pairs[0].calib_img_top.obj_points)

    if len(omnistereo_model.top_model.calibrator.omni_monos) > 0 and len(omnistereo_model.bot_model.calibrator.omni_monos) > 0:
        from omnistereo.calibration import draw_detected_points_manually
        calibrator.calib_top.omni_monos[0].visualize_points(window_name=omnistereo_model.panorama_top.name + " corners " + str(0))
        calibrator.calib_bottom.omni_monos[0].visualize_points(window_name=omnistereo_model.panorama_bot.name + " corners " + str(0))
        draw_detected_points_manually(calibrator.calib_top.omni_monos[0].omni_image, calibrator.calib_top.omni_monos[0].image_points, 5, show=True)
        draw_detected_points_manually(calibrator.calib_bottom.omni_monos[0].omni_image, calibrator.calib_bottom.omni_monos[0].image_points, 5, show=True)


def test_space2plane(omnistereo_model, point_3D_wrt_C, visualize=False, draw_fiducial_rings=False, z_offsets=[0]):
    # Arbitrary point in space

    offset_points = np.repeat(point_3D_wrt_C, len(z_offsets), axis=0)
    offset_points[..., 2] = (offset_points[..., 2].T + z_offsets).T
    print("Projecting:",)
    u_top, v_top, m_top = omnistereo_model.top_model.get_pixel_from_3D_point_wrt_C(offset_points)
    print("TOP: Space to Plane test: to (u,v)", m_top)

    u_bottom, v_bottom, m_bottom = omnistereo_model.bot_model.get_pixel_from_3D_point_wrt_C(offset_points)
    print("BOTTOM: Space to Plane test: to (u,v)", m_bottom)

    max_num_of_rings_top = 4
    if visualize:
        import cv2
        from omnistereo.common_cv import draw_points
        # Make copy of omni_image
        img = omnistereo_model.current_omni_img.copy()
        ring_count = 0
        fiducial_rings_radii_top = []
        fiducial_rings_radii_bottom = []
        for m_t, m_b in zip(m_top, m_bottom):
            draw_points(img, points_uv_coords=m_b[..., :2], color=(255, 0, 0), thickness=10)
            if ring_count < max_num_of_rings_top:
                draw_points(img, points_uv_coords=m_t[..., :2], color=(0, 0, 255), thickness=10)

            if draw_fiducial_rings:
                # circle centers
                img_center_point_top = omnistereo_model.top_model.precalib_params.center_point
                center_top = (int(img_center_point_top[0]), int(img_center_point_top[1]))
                cv2.circle(img, center_top, 3, (255, 0, 0), -1, 8, 0)
                img_center_point_bottom = omnistereo_model.bot_model.precalib_params.center_point
                center_bottom = (int(img_center_point_bottom[0]), int(img_center_point_bottom[1]))
                cv2.circle(img, center_bottom, 3, (0, 0, 255), -1, 8, 0)

                # Draw bottom ring:
                r_bottom = np.linalg.norm(m_b[..., :2] - img_center_point_bottom, axis=-1)
                cv2.circle(img, center_bottom, int(r_bottom), (0, 0, 255), 3, 8, 0)
                fiducial_rings_radii_bottom.append(r_bottom)

            if ring_count < max_num_of_rings_top:
                # Draw top ring:
                r_top = np.linalg.norm(m_t[..., :2] - img_center_point_top, axis=-1)
                cv2.circle(img, center_top, int(r_top), (255, 0, 0), 3, 8, 0)
                fiducial_rings_radii_top.append(r_top)
            ring_count += 1

            win_name = "TEST: 3D Space to omni_image plane (Projection)"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, img)
            cv2.waitKey(1)

        return fiducial_rings_radii_top, fiducial_rings_radii_bottom

def test_pixel_lifting(gums):
    import omnistereo.euclid as euclid
    u_offset = -300
    v_offset = -300

    u0_top, v0_top = gums.top_model.get_center()
    (u, v) = (1200, 600)
#     u = u0_top + u_offset
#     v = v0_top + v_offset
    print("\nTOP Lifting test with pixel(%f,%f):" % (u, v))
    # FIXME: use Ps vector instead of X,Y,Z
    is_valid, X, Y, Z = gums.top_model.lift_pixel_to_unit_sphere_wrt_focus(u, v, visualize=True, debug=True)
    if is_valid:
        lifted_point_on_sphere_top = euclid.Point3(X, Y, Z)
        print("as point in sphere = %s with elevation of %f degrees" % (lifted_point_on_sphere_top, np.rad2deg(np.arcsin(lifted_point_on_sphere_top.z))))

    u0_bottom, v0_bottom = gums.bot_model.get_center()
    (u, v) = (1000, 600)
#     u = u0_bottom + u_offset
#     v = v0_bottom + v_offset
    print("\nBOTTOM Lifting test with pixel(%f,%f):" % (u, v))
    # FIXME: use Ps vector instead of X,Y,Z
    # FIXME: use m instead u, v and remove is_valid
    is_valid, X, Y, Z = gums.bot_model.lift_pixel_to_unit_sphere_wrt_focus(u, v, visualize=True, debug=True)
    if is_valid:
        lifted_point_on_sphere_bottom = euclid.Point3(X, Y, Z)
        print("as point in sphere = %s with elevation of %f degrees" % (lifted_point_on_sphere_bottom, np.rad2deg(np.arcsin(lifted_point_on_sphere_bottom.z))))

def convert_steradian_to_radian(steredian):
    # FIXME: choose the correct conversion
    radian = 2 * np.arccos(1 - (steredian / (2 * np.pi)))  # According to Wikipedia, which seems more correct!
#     radian = 2 * np.arcsin(np.sqrt(steredian / np.pi))  # According to the Self Study Manual on Optial Radiation Measurements
    return radian

def convert_resolution_units(pixel_length, in_pixels, in_radians, use_spatial_resolution, eta, in_2D=False):
    if in_2D:
        pow_factor = 1.0
    else:
        pow_factor = 2.0

    if use_spatial_resolution:
        if in_pixels:
            eta = eta / (pixel_length ** pow_factor)  # [mm/rad]/[mm/px] ==> [px/rad]
        if in_radians == False:
            eta = eta / ((180 / np.pi) ** pow_factor)  # [len / rad] / [deg/rad]  ==> [len / deg]
    else:
        if in_pixels:
            eta = eta * (pixel_length ** pow_factor)  # [rad/mm]*[mm/px] ==> [rad/px]
        if in_radians == False:
            eta = eta * ((180 / np.pi) ** pow_factor)  # [deg/rad] * [rad/len] ==> [deg/len]
    return eta

def unit_test(val1, val2, decimals):
    val1_approx = np.zeros_like(val1)
    val2_approx = np.zeros_like(val2)
    np.round(val1, decimals, val1_approx)
    np.round(val2, decimals, val2_approx)
    print("UNIT TEST:")
    print("val1:", val1_approx)
    print("val2:", val2_approx)
    print("val1 != val2? -->", "Pass!" if np.count_nonzero(val1_approx != val2_approx) == 0 else "Fail")
