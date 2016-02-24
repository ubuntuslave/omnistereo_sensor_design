# -*- coding: utf-8 -*-
# camera_models.py

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
@package omnistereo_sensor_design
Tools for omnidirectional stereo vision using catadioptrics

@author: Carlos Jaramillo
'''

from __future__ import division
import omnistereo.euclid as euclid
import numpy as np
import cv2
from omnistereo.common_tools import flatten

def intersect_line3_with_plane_vectorized(line, plane):
    '''
    @brief it can compute the intersection of an 3D line and Plane using numpy arrays in a vectorized approach
    @param line: Either an Euclid Line3 object or a Numpy ndarray of points in row-wise order,
            such as [p.x,p.y,p.z,v.x,v.y,v.z], where p is a point in the line, and v is the direction vector of the line.
    @param plane: Either a Euclid Plane object or a Numpy ndarray such as [a,b,c,d] where the plane is defined by a*x+b*y+c*z=d
    @return The ndarray of the intersection coordinates with one ndarray per intersection.
            When a line is parallel to the plane, there's no intersection, so resulting coordinates are indicated by a Not a Number (nan) values.
            For example, it could return [[i1.x, i1.y, i1.z], [nan, nan, nan], [i3.x, i3.y, i3.z]] for 3 lines, where the second line doesn't intersect the plane.
    '''
    if isinstance(line, euclid.Line3):
        line_dir_vec_np = np.array(line.v)
        line_point_np = np.array(line.p)
    elif isinstance(line, np.ndarray):
        line_point_np = line[..., 0:3]
        line_dir_vec_np = line[..., 3:]
    else:
        raise TypeError ("Line must be a Euclid Line3 or defined as a ndarray \n" +
                         "such as [p.x,p.y,p.z,v.x,v.y,v.z], not %s" % type(line))

    if isinstance(plane, euclid.Plane):
        plane_normal = np.array(plane.n)
        k = plane.k
    elif isinstance(plane, np.ndarray):
        plane_normal = plane[0:3]
        k = plane[3]
    else:
        raise TypeError ("Plane must be a Euclid Plane or defined as a ndarray \n" +
                         "such as [a,b,c,d] where the plane is a*x+b*y+c*z=d, not %s" % type(plane))

    # BECAREFUL: inner product is not the same of "dot" product for matrices
    d = np.inner(plane_normal, line_dir_vec_np)

    # d == 0 when vectors are perpendicular because the line is parallel to the plane,
    # so we indicate that with Not a Number (nan) values.
    # Note: It's preferred to use nan because None changes the dtype of the ndarray to "object"
    d = np.where(d != 0.0, d, np.nan)

    u = np.where(d is not np.nan, (k - np.inner(plane_normal, line_point_np)) / d, np.nan)
    # Give u another length-1 axis on the end, and broadcasting will handle this without needing to actually construct a tripled array.
    intersection = np.where(u is not np.nan, line_point_np + u[..., np.newaxis] * line_dir_vec_np, np.nan)
    return intersection

def ortho_project_point3_onto_plane(point, plane):
    '''
    @brief Projects 3D points onto a Plane using numpy arrays in a vectorized approach
    @param point: Either an Euclid Point3 object or a Numpy ndarray of points in row-wise order,
            such as [p.x,p.y,p.z]
    @param plane: Either a Euclid Plane object or a Numpy ndarray such as [a,b,c,d] where the plane is defined by a*x+b*y+c*z=d
    @return The ndarray of the intersection coordinates with one ndarray per intersection.
            When a line is parallel to the plane, there's no intersection, so resulting coordinates are indicated by a Not a Number (nan) values.
            For example, it could return [[i1.x, i1.y, i1.z], [nan, nan, nan], [i3.x, i3.y, i3.z]] for 3 lines, where the second line doesn't intersect the plane.
    '''
    if isinstance(point, euclid.Point3):
        p = np.array(point)
    elif isinstance(point, np.ndarray):
        p = point[..., :3]
    else:
        raise TypeError ("Line must be a Euclid Point3 or defined as a ndarray \n" +
                         "such as [p.x,p.y,p.z], not %s" % type(point))

    if isinstance(plane, euclid.Plane):
        n = np.array(plane.n)
        k = plane.k
    elif isinstance(plane, np.ndarray):
        n = plane[0:3]
        k = plane[3]
    else:
        raise TypeError ("Plane must be a Euclid Plane or defined as a ndarray \n" +
                         "such as [a,b,c,d] where the plane is a*x+b*y+c*z=d, not %s" % type(plane))

    d = p.dot(n) - k
    p_x, p_y, p_z = p[..., 0], p[..., 1], p[..., 2]
    n_x, n_y, n_z = n[..., 0], n[..., 1], n[..., 2]

#     if res_coords_wrt_plane:
#         point_on_plane_x = p_x - n_x * (d + k)
#         point_on_plane_y = p_y - n_y * (d + k)
#         point_on_plane_z = p_z - n_z * (d + k)
#     else:
    point_on_plane_x = p_x - n_x * d
    point_on_plane_y = p_y - n_y * d
    point_on_plane_z = p_z - n_z * d
    point_on_plane = np.dstack((point_on_plane_x, point_on_plane_y, point_on_plane_z))

    return point_on_plane

def intersect_line3_with_sphere_vectorized(line, sphere=None):
    '''
    @brief it can compute the intersection of an 3D line and Plane using numpy arrays in a vectorized approach
    @param line: Either an Euclid Line3 object or a Numpy ndarray of points in row-wise order,
            such as [p.x,p.y,p.z,v.x,v.y,v.z], where p is a point in the line, and v is the direction vector of the line.
    @param sphere: A Euclid Sphere object
    @return The ndarray of the intersection coordinates with one ndarray per intersection, such that you get a matrix shape as [m,n,6],
            where the first 3 values [row,col, :3] for each entry [row,col] are the coordinates of the first point intersection, and [row,col, 3:] is for the second point.
    '''

    if isinstance(line, euclid.Line3):
        line_point_np = np.array(line.p)
        line_dir_vec_np = np.array(line.v)
    elif isinstance(line, np.ndarray):
        line_point_np = line[..., 0:3]
        line_dir_vec_np = line[..., 3:]
    else:
        raise TypeError ("Line must be a Euclid Line3 or defined as a ndarray \n" +
                         "such as [p.x,p.y,p.z,v.x,v.y,v.z], not %s" % type(line))


    if sphere == None:
        sphere_center_np = np.array([0, 0, 0])
        sphere_radius = 1.
    else:
        sphere_center_np = np.array(sphere.c)
        sphere_radius = sphere.r

    # Method based on Euclid's implementation of _intersect_line3_sphere
    # Magnitude squared
    a = np.sum(line_dir_vec_np ** 2, axis=-1)
    sph_mag_2 = np.sum(sphere_center_np ** 2, axis=-1)
    line_p_mag_2 = np.sum(line_point_np ** 2, axis=-1)

    p_wrt_sph = line_point_np - sphere_center_np
    b = 2 * np.sum(line_dir_vec_np * p_wrt_sph, axis=-1)

    c = sph_mag_2 + line_p_mag_2 - \
        2 * np.dot(line_point_np, sphere_center_np) - \
        sphere_radius ** 2

    det = b ** 2 - 4 * a * c
    sq = np.sqrt(det)

    u1 = (-b + sq) / (2 * a)
    u2 = (-b - sq) / (2 * a)
#     if not L._u_in(u1):
#         u1 = max(min(u1, 1.0), 0.0)
#     if not L._u_in(u2):
#         u2 = max(min(u2, 1.0), 0.0)

    line_segment_np = np.dstack((line_point_np + u1[..., np.newaxis] * line_dir_vec_np, line_point_np + u2[..., np.newaxis] * line_dir_vec_np))
    return line_segment_np

def get_normalized_points_XYZ(x, y, z):
    # vector magnitude
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    x_norm = x / d
    y_norm = y / d
    z_norm = z / d

    point_as_np = np.zeros(x.shape + (3,))  # Create a multidimensional array based on the shape of x, y, or z
    point_as_np[:, :, 0] = x_norm
    point_as_np[:, :, 1] = y_norm
    point_as_np[:, :, 2] = z_norm

    return point_as_np

def get_normalized_points(points_wrt_M):
#     x = points_wrt_M[..., 0]
#     y = points_wrt_M[..., 1]
#     z = points_wrt_M[..., 2]
#     # Compute the normalization (projection to the sphere) of points
#     points_norms = np.sqrt(x ** 2 + y ** 2 + z ** 2)
#     points_on_sphere = points_wrt_M[..., :3] / points_norms[..., np.newaxis]
    # More efficiently:
    points_on_sphere = points_wrt_M[..., :3] / (np.linalg.norm(points_wrt_M[..., :3], axis=-1)[..., np.newaxis])
    return points_on_sphere

# FIXME: change the data to use column vectors
def get_lines_through_single_point3(point1, point2):
    '''
    Lines composed from a single or set of x, y, z coordinates (point1) and the single point (point2).
    @param point1: An Euclid Point3 or a ndarray of point1 written as row vectors, for example [[p1.x,p1.y,p1.z],[[p2.x,p2.y,p2.z]], ...]
    @param point2: An Euclid Point3 or a row-vector point2 defined as [p.x,p.y,p.z], which is crossed by the lines passing by point1 with coordinates x, y, z.
    @return Line(s) from (a single or several) point1 directed toward the point2.
    '''
    if isinstance(point1, euclid.Point3):
        point1_as_np = np.array(point1)
    elif isinstance(point1, np.ndarray):
        point1_as_np = point1
    else:
        raise TypeError ("point2 must be a Euclid Point3 or defined as a row-vector in a ndarray \n" +
             "such as [p.x,p.y,p.z], not %s" % type(point2))
    if isinstance(point2, euclid.Point3):
        point2_as_np = np.array(point2)
    elif isinstance(point2, np.ndarray):
        point2_as_np = point2
    else:
        raise TypeError ("point2 must be a Euclid Point3 or defined as a row-vector in a ndarray \n" +
             "such as [p.x,p.y,p.z], not %s" % type(point2))
    lines = np.zeros(point1.shape[:-1] + (6,))  # Create a multidimensional array based on the point1's shape
    p = point1_as_np
    v = point1_as_np - point2_as_np
    lines[..., 0] = p[..., 0]
    lines[..., 1] = p[..., 1]
    lines[..., 2] = p[..., 2]
    lines[..., 3] = v[..., 0]
    lines[..., 4] = v[..., 1]
    lines[..., 5] = v[..., 2]

    return lines

class FeatureMatcher(object):
    def __init__(self, method, matcher_type, k_best, *args, **kwargs):
        '''
        @param method: Currently immplemented methods are: "ORB", "SIFT"
        @param matcher_type: Either "BF" for brute-force, or "FLANN" for a FLANN-based matcher
        @param k_best: The number of matches to be considered among each point feature
        '''
        self.feature_detection_method = method
        self.matcher_type = matcher_type
        self.k_best = k_best

        self.FLANN_INDEX_KDTREE = 1
        self.FLANN_INDEX_LSH = 6
        self.MIN_MATCH_COUNT = 10
        self.percentage_good_matches = 1.0
        self.num_of_features = 100  # Not useful for detector that use non-maximal suppression, such as FAST and AGAST
        self.flann_params = dict(algorithm=self.FLANN_INDEX_LSH,
                           table_number=6,  # 12
                           key_size=12,  # 20
                           multi_probe_level=1)  # 2

        if self.matcher_type == "FLANN":  # or self.feature_detection_method == "SIFT":
            index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
#             self.matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self.matcher = cv2.BFMatcher()
            # self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match(self, query_descriptors, train_descriptors):
        self.matcher.clear()  # Clear matcher
        # self..matcher.add([top_descriptors])
        if self.k_best > 1:
            # Save the best k matches:
            knn_matches = self.matcher.knnMatch(queryDescriptors=query_descriptors, trainDescriptors=train_descriptors, k=self.k_best)
            # store all the good matches as per Lowe's ratio test.
            if self.k_best == 2:  # For exactly 2 best matches
                matches = [m[0] for m in knn_matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
            # Flatten matches in list of k-lists
            matches = flatten(knn_matches)
        else:
            # returns the best match only
            matches = self.matcher.match(queryDescriptors=query_descriptors, trainDescriptors=train_descriptors)
        # Sort good matches (filter by above criteria) in the order of their distance.
        matches = sorted(matches, key=lambda x:x.distance)

        return matches

class CamParams(object):
    '''
    Super class to stores camera pre-calibration parameters from a pre-calibration step (e.g. done in some MATLAB toolbox)
    '''

    def __init__(self, *args, **kwargs):
        self.parent_model = None
        self.focal_length = None
        self.pixel_size = None
        self.sensor_size = None
        self.pixel_area = None
        self.image_size = None
        self.center_point = None

        # Distortion
        self.k1 = None
        self.k2 = None
        self.p1 = None
        self.p2 = None
        self.k5 = None

        self.gamma1 = None  # Focal length on u [px/mm or px/m] based on distance units of model
        self.gamma2 = None  # Focal length on v [px/mm or px/m] based on distance units of model
        self.f_u = self.gamma1
        self.f_v = self.gamma2
        self.u_center = None
        self.v_center = None
        self.alpha_c = None
        self.skew_factor = None

        # ROI:
        self.roi_min_x = None
        self.roi_min_y = None
        self.roi_max_x = None
        self.roi_max_y = None

        # TODO: add distortion parameters
        # Inverse camera projection matrix parameters
        self.K_inv = None

    def print_params(self):
        try :
            distortion_params = "k1={0:.6f}, k2={1:.6f}, p1={2:.6f}, p2={3:.6f}, k5={4:.6f}".format(self.k1, self.k2, self.p1, self.p2, self.k5)
            print(distortion_params)
        except :
            pass

        try :
            aberr_params = "gamma1={0:.6f}, gamma2={1:.6f}, alpha_c={2:.6f}".format(self.gamma1, self.gamma2, self.alpha_c)
            print(aberr_params)
        except :
            pass

        try :
            misc_params = "u_center={0:.6f}, v_center={1:.6f}".format(self.u_center, self.v_center)
            print(misc_params)
        except :
            pass

        try :
            roi_params = "roi_min_x={0:.6f}, roi_min_y={1:.6f}, roi_max_x={2:.6f}, roi_max_y={3:.6f}".format(self.roi_min_x, self.roi_min_y, self.roi_max_x, self.roi_max_y)
            print(roi_params)
        except :
            pass

        try :
            min_hor = np.rad2deg(self.min_useful_FOV_hor)
            max_hor = np.rad2deg(self.max_useful_FOV_hor)
            min_ver = np.rad2deg(self.min_useful_FOV_ver)
            max_ver = np.rad2deg(self.max_useful_FOV_ver)

            cam_useful_FOVs = "Useful Camera FOVs (in degrees): min_hor={0:.3f}, min_ver={1:.3f}, max_hor={2:.3f}, max_ver={3:.3f}".format(min_hor, min_ver, max_hor, max_ver)
            print(cam_useful_FOVs)
        except :
            pass

class PerspectiveCamModel(object):
    def __init__(self, **kwargs):
        self.units = kwargs.get("units", "mm")
        self.intrinsic_params_matrix = kwargs.get("intrinsic_params_matrix", None)
        self.distortion_coeffs = kwargs.get("distortion_coeffs", None)
        self.image_size = kwargs.get("image_size", None)
#         self.cam_params = CamParams(kwargs)

class OmniCamModel(object):
    '''
    A single omnidirectional camera model
    Mostly, a superclass template template to be implemented by a custom camera model
    '''

    def _init_default_values(self, **kwargs):
        self.is_calibrated = False
        self.mirror_number = kwargs.get("mirror_number", 0)
        self.F = kwargs.get("F", np.array([0, 0, 0, 1]).reshape(4, 1))  # Focus (inner, real)
        self.Fv = kwargs.get("F_virt", np.array([0, 0, 0, 1]).reshape(4, 1))  # Focus (outer, virtual)

        self.units = kwargs.get("units", "mm")

        # Physical radii
        self.r_reflex = kwargs.get("r_reflex", None)
        self.r_min = kwargs.get("r_min", None)
        self.r_max = kwargs.get("r_max", None)

        if self.r_min != None:
            # Compute elevations
            xw = self.r_min
            yw = 0
            zw = self.get_z_hyperbola(xw, yw)
            theta = np.arctan2(zw - self.F[2], xw)
            if self.mirror_number == 1:
                self.lowest_elevation_angle = theta  # The elevation determined by the reflex mirror
            else:
                self.highest_elevation_angle = theta  # The elevation determined by the camera hole
        else:
            if self.mirror_number == 1:
                self.lowest_elevation_angle = -np.pi / 2. + np.deg2rad(1)
            else:
                self.highest_elevation_angle = np.pi / 2. - np.deg2rad(1)

        if self.r_max != None:
            # Compute elevations
            xw = self.r_max
            yw = 0
            zw = self.get_z_hyperbola(xw, yw)
            theta = np.arctan2(zw - self.F[2], xw)
            if self.mirror_number == 1:  # Highest elevation determined by the the mirror's radius
                self.highest_elevation_angle = theta
            else:  # Lowest elevation is determined by the minimum elevation from mirror radius or camera hole radius
                theta_due_to_r_sys = theta
                if self.r_reflex != None:
                    c2 = self.c
                    k2 = self.k
                    d = self.d
                    r_ref = self.r_reflex
                    # NOTE: We solved the system of equations for the reflection point (as depicted in pink line in Geometry Expressions)
                    # Solution of the outward projection instead from F2v to the surface mirror 2
                    lambda2_sln = (c2 * d * k2 + c2 * np.sqrt(k2 * (k2 - 2) * (d ** 2 + 4 * r_ref ** 2))) / (k2 * (d ** 2 - 2 * k2 * r_ref ** 2 + 4 * r_ref ** 2))
                    p2_of_reflex_edge = np.array([[[lambda2_sln * r_ref , 0, d - lambda2_sln * d / 2.0, 1.0]]])
                    x_p2 = p2_of_reflex_edge[0, 0, 0]
                    z_p2 = p2_of_reflex_edge[0, 0, 2]
                    theta_due_to_r_reflex = np.arctan2(z_p2 - self.F[2], x_p2)
                else:
                    theta_due_to_r_reflex = -np.inf

                self.lowest_elevation_angle = max(theta_due_to_r_sys, theta_due_to_r_reflex)
        else:
            if self.mirror_number == 1:
                self.highest_elevation_angle = np.pi / 2. - np.deg2rad(1)
            else:
                self.lowest_elevation_angle = -np.pi / 2. + np.deg2rad(1)

        self.vFOV = self.highest_elevation_angle - self.lowest_elevation_angle
        self.globally_highest_elevation_angle = self.highest_elevation_angle
        self.globally_lowest_elevation_angle = self.lowest_elevation_angle
        # WISH: add a method to set the calibration parameters from Calibration as a single GUM

        self.outer_img_radius = 0  # outer radius encircling the ROI for occlusion boundary
        self.inner_img_radius = 0  # inner radius encircling the ROI for occlusion boundary

        self.precalib_params = CamParams()
        self.current_omni_img = None
        self.panorama = None
        self.mask = None

    def print_params(self, **kwargs):
        self.print_precalib_params()
        print(self.mirror_name + " camera parameters:")
        self.precalib_params.print_params()
        print("Radial image height: %f pixels" % self.h_radial_image)

        vFOV = kwargs.get("vFOV", self.vFOV)
        max_elevation = kwargs.get("max_elevation", self.highest_elevation_angle)
        min_elevation = kwargs.get("min_elevation", self.lowest_elevation_angle)

        print("vFOV: %f degrees" % np.rad2deg(vFOV))
        print("Highest elevation angle: %f degrees" % np.rad2deg(max_elevation))
        print("Lowest elevation angle: %f degrees" % np.rad2deg(min_elevation))

    def print_precalib_params(self):
        pass


    def map_angles_to_unit_sphere(self, theta, psi):
        '''
        Resolves the point (normalized on the unit sphere) from the given direction angles

        @param theta: Elevation angle to a world point from some origin (Usually, the mirror's focus)
        @param psi: Azimuth angle to a world point from some origin.
        @return: The homogeneous coordinates (row-vector) of the 3D point coordinates (as an ndarray) corresponding to the passed direction angles
        '''
        if isinstance(theta, np.ndarray):
            theta_valid_mask = np.logical_not(np.isnan(theta))  # Filters index that are NaNs
            theta_validation = theta_valid_mask.copy()
            # Checking within valid elevation:
            theta_validation[theta_valid_mask] = np.logical_and(self.lowest_elevation_angle <= theta[theta_valid_mask], theta[theta_valid_mask] <= self.highest_elevation_angle)
            b = np.where(theta_validation, np.cos(theta), np.nan)
            z = np.where(theta_validation, np.sin(theta), np.nan)
        else:
            b = np.cos(theta)
            z = np.sin(theta)

        x = b * np.cos(psi)
        y = b * np.sin(psi)

        #=======================================================================
        # z = np.where(np.logical_and(self.lowest_elevation_angle <= theta, theta <= self.highest_elevation_angle),
        #              np.sin(theta), np.nan)
        #=======================================================================

        w = np.ones_like(x)
        P_on_sphere = np.dstack((x, y, z, w))
        return P_on_sphere

    def get_pixel_from_direction_angles(self, azimuth, elevation, visualize=False):
        '''
        @brief Given the elevation and azimuth angles (w.r.t. mirror focus), the projected pixel on the warped omnidirectional image is found.

        @param azimuth: The azimuth angles as an ndarray
        @param elevation: The elevation angles as an ndarray
        @retval u: The u pixel coordinate (or ndarray of u coordinates)
        @retval v: The v pixel coordinate (or ndarray of v coordinates)
        @retval m_homo: the pixel as a homogeneous ndarray (in case is needed)
        '''
        Pw_wrt_F = self.get_3D_point_from_angles_wrt_focus(azimuth=azimuth, elevation=elevation)
        return self.get_pixel_from_3D_point_wrt_M(Pw_wrt_F, visualize=False)

    def get_3D_point_from_angles_wrt_focus(self, azimuth, elevation):
        '''
        Finds a world point using the given projection angles towards the focus of the mirror

        @return: The numpy ndarray of 3D points (in homogeneous coordinates) w.r.t. origin of coordinates (\f$O_C$\f)
        '''
        raise NotImplementedError


    def get_pixel_from_3D_point_wrt_C(self, Pw_wrt_C, visualize=False):
        '''
        @brief Project a three-dimensional numpy array (rows x cols x 4) of 3D homogeneous points (eg. [x, y, z, 1]) as row-vectors to the image plane in (\a u,\a v).
        This function is already vectorized for Numpy performance.

        @param Pw_wrt_C: the multidimensional array of homogeneous coordinates of the points (wrt the origin of the common frame [C], e.g. camera pinhole)
        @param visualize: To indicate if a 3D visualization will be shown

        @retval u: the resulting ndarray of u coordinates on the image plane
        @retval v: the resulting ndarray of v coordinates on the image plane
        @retval m_homo: The pixel point(s) as numpy array in homogeneous coordinates
        '''
        raise NotImplementedError

    def get_pixel_from_3D_point_wrt_M(self, Pw_homo, visualize=False):
        '''
        @brief Project a three-dimensional numpy array (rows x cols x 4) of 3D homogeneous points (eg. [x, y, z, 1]) as row-vectors to the image plane in (\a u,\a v).
        This function is already vectorized for Numpy performance.

        @param Pw_homo: the multidimensional array of homogeneous coordinates of the points (wrt the MIRROR focus frame)
        @param visualize: To indicate if a 3D visualization will be shown

        @retval u: the resulting ndarray of u coordinates on the image plane
        @retval v: the resulting ndarray of v coordinates on the image plane
        @retval m_homo: The pixel point(s) as numpy array in homogeneous coordinates
        '''
        raise NotImplementedError


    def get_pixel_from_XYZ(self, x, y, z, visualize=False):
        '''
        @brief Project a 3D points (\a x,\a y,\a z) to the image plane in (\a u,\a v)
               NOTE: This function is not vectorized (not using Numpy explicitly).

        @param x: 3D point x coordinate (wrt the center of the unit sphere)
        @param y: 3D point y coordinate (wrt the center of the unit sphere)
        @param z: 3D point z coordinate (wrt the center of the unit sphere)

        @retval u: contains the image point u coordinate
        @retval v: contains the image point v coordinate
        @retval m: the undistorted 3D point (of type euclid.Point3) in the normalized projection plane
        '''
        raise NotImplementedError

    def get_obj_pts_proj_error(self, img_points, obj_pts_homo, T_G_wrt_C):
        '''
        Compute the pixel errors between the points on the image and the projected 3D points on an object frame [G] with respect to the fixed frame [C]

        @param img_points: The corresponding points on the image
        @param obj_pts_homo: The coordinates of the corresponding points with respect to the object's own frame [G].
        @param T_G_wrt_C: The transform matrix of [G] wrt to [C].

        @return: The array of pixels errors (euclidedian distances a.k.a L2 norms)
        '''
        obj_pts_wrt_C = np.einsum("ij, mnj->mni", T_G_wrt_C, obj_pts_homo)

        # The detected (observed) pixels for chessboard points
        _, _, projected_pts = self.get_pixel_from_3D_point_wrt_C(obj_pts_wrt_C)
        error_vectors = projected_pts[..., :2] - img_points[..., :2]
        error_norms = np.linalg.norm(error_vectors, axis=-1).flatten()
        return error_norms

    def get_confidence_weight_from_pixel_RMSE(self, img_points, obj_pts_homo, T_G_wrt_C):
        '''
        We define a confidence weight as the inverse of the pixel projection RMSE

        @param img_points: The corresponding points on the image
        @param obj_pts_homo: The coordinates of the corresponding points with respect to the object's own frame [G].
        @param T_G_wrt_C: The transform matrix of [G] wrt to [C].
        '''
        from omnistereo.common_tools import rms
        all_pixel_errors = self.get_obj_pts_proj_error(img_points, obj_pts_homo, T_G_wrt_C)
        rmse = rms(all_pixel_errors)
        weight = 1.0 / rmse
        return weight

    def lift_pixel_to_unit_sphere_wrt_focus(self, m, visualize=False, debug=False):
        '''
        @brief Lifts a pixel point from the image plane to the unit sphere
        @param m: A ndarray of k image point coordinates [u, v] per cell (e.g. shape may be rows, cols, 2)
        @param visualize: Indicates if visualization will take place
        @param debug: Indicates to print debugging statements
        @retval Ps: The Euclidean coordinates (as a rows x cols x 3 ndarray) of the point(s) on the sphere.
        '''
        raise NotImplementedError


    def get_direction_vector_from_focus(self, m):
        '''
        @param m: A ndarray of image point coordinates [u, v] per cell (e.g. shape may be rows, cols, 2)

        @return the array of direction vectors in homogenous coordinates (4-vectors)
        '''
        v = self.lift_pixel_to_unit_sphere_wrt_focus(m)

        return v

    def get_direction_angles_from_pixel(self, m_omni):
        '''
        In order to determine the respective azimuth and elevation angles, it lifts the pixel to the unit sphere using lift_pixel_to_unit_sphere_wrt_focus.

        @param m_pano: A numpy array of k image point coordinates [u, v] as row vector. Thus, shape is (rows, cols, 2)
        @retval azimuth, elevation: angles in radians for the image pixel with coordinates (u,v) in the distorted image. Angles are w.r.t. the mirror's focus.
        '''
        Ps = self.lift_pixel_to_unit_sphere_wrt_focus(m_omni)  # Find point on unit sphere
        azimuth = np.arctan2(Ps[..., 1], Ps[..., 0])
        elevation = np.arcsin(Ps[..., 2])
        return azimuth, elevation

    def get_all_direction_angles_per_pixel_radially(self):
        px_all_u = np.arange(self.inner_img_radius, self.outer_img_radius + 1) + self.precalib_params.u_center
        px_all_v = np.zeros_like(px_all_u) + self.precalib_params.v_center
        px_all = np.dstack((px_all_u, px_all_v))
        azimuth_all, elev_all = self.get_direction_angles_from_pixel(px_all)
        return azimuth_all, elev_all

    def distortion(self, mx_u, my_u):
        '''
        @brief Apply distortion to input point (from the normalised plane)

        @param mx_u: undistorted x coordinate of point on the normalised projection plane
        @param my_u: undistorted y coordinate of point on the normalised projection plane

        @retval dx_u: distortion value that was added to the undistorted point \f$mx_u\f$ such that the distorted point is produced \f$ mx_d = mx_u+dx_u \f$
        @retval dy_u: distortion value that was added to the undistorted point \f$my_u\f$ such that the distorted point is produced \f$ my_d = my_u+dy_u \f$
        '''
        raise NotImplementedError


    def get_center(self):
        u = self.precalib_params.u_center
        v = self.precalib_params.v_center
        return u, v

    def _compute_boundary_elevations(self):
        '''
        Private method to compute the highest and lowest elevation angles related to occlusion boundaries
        '''
        # Approximating the pixel resolution of azimuths to be 1 degree
        # NOTE: These phi angles on the image advance on a counterclockwise direction around the center.
        # However, their order is reversed ("clockwise") around the z-axis of the model in the world.
        # This doesn't really matter because we are lifting the pixel to its corresponding 3D point, and then get chosen.
        phi_on_img_array = np.linspace(0, 2 * np.pi, num=360, endpoint=False)

        u_low = self.precalib_params.u_center + self.r_lowest_elevation * np.cos(phi_on_img_array)
        v_low = self.precalib_params.v_center + self.r_lowest_elevation * np.sin(phi_on_img_array)
#         self.low_img_points = np.array([u_low, v_low]).transpose()
        self.low_img_points = np.dstack((u_low, v_low))
        # Longer way:
        # low_img_points[:,0]=u_low
        # low_img_points[:,1]=v_low
#         valid_low, X_low, Y_low, Z_low = plane2sphere_vectorized(u_low, v_low, visualize=False, debug=False)
        self.low_3D_points = self.lift_pixel_to_unit_sphere_wrt_focus(self.low_img_points, visualize=False, debug=False)
#         self.low_3D_points = np.array([X_low, Y_low, Z_low]).transpose()
        # Find the 3D point with the minimum Z value
        self.lowest_lifted_point_index = np.argmin(self.low_3D_points[..., 2])
        self.lowest_lifted_point_on_sphere_wrt_F = self.low_3D_points[:, self.lowest_lifted_point_index]
        self.lowest_img_point = self.low_img_points[:, self.lowest_lifted_point_index]
        _, lowest_elev = self.get_direction_angles_from_pixel(self.lowest_img_point[np.newaxis, :])
        self.lowest_elevation_angle = float(lowest_elev)
        print("LOWEST pixel %s for point %s with elevation: %f degrees" % (self.lowest_img_point, self.lowest_lifted_point_on_sphere_wrt_F, np.rad2deg(self.lowest_elevation_angle)))

        u_high = self.precalib_params.u_center + self.r_highest_elevation * np.cos(phi_on_img_array)
        v_high = self.precalib_params.v_center + self.r_highest_elevation * np.sin(phi_on_img_array)
#         self.high_img_points = np.array([u_high, v_high]).transpose()
        self.high_img_points = np.dstack((u_high, v_high))
#         valid_high, X_high, Y_high, Z_high = plane2sphere_vectorized(u_high, v_high, debug=False)
        self.high_3D_points = self.lift_pixel_to_unit_sphere_wrt_focus(self.high_img_points, debug=False)
#         self.high_3D_points = np.array([X_high, Y_high, Z_high]).transpose()
        # Find the 3D point with the maximum Z value
        self.highest_lifted_point_index = np.argmax(self.high_3D_points[..., 2])
        self.highest_lifted_point_on_sphere_wrt_F = self.high_3D_points[:, self.highest_lifted_point_index]
        self.highest_img_point = self.high_img_points[:, self.highest_lifted_point_index]

        _, highest_elev = self.get_direction_angles_from_pixel(self.highest_img_point[np.newaxis, :])
        self.highest_elevation_angle = float(highest_elev)
        print("HIGHEST pixel %s for point %s with elevation: %f degrees" % (self.highest_img_point, self.highest_lifted_point_on_sphere_wrt_F, np.rad2deg(self.highest_elevation_angle)))
        print(80 * "-")
        self.vFOV = self.get_vFOV()

    def set_radial_limits(self, r_min, r_max):
        '''
        Uses the bounding radii to compute elevation angles according to the mirror's position

        @param r_min: physical radius of the small (inner) bounding circle on the image (the radius of the reflex planar mirror or the camera hole)
        @param r_max: physical radius of the large (outer) bounding circle on the image (Usually the system radius)
        '''
        self.r_min = r_min
        self.r_max = r_max

    def set_radial_limits_in_pixels_mono(self, inner_img_radius, outer_img_radius, center_point=None, preview=False, **kwargs):
        '''
        Determines which image radial boundary corresponds to the highest and lowest elevantion angles.
        '''
        if center_point is not None:
            self.precalib_params.center_point = center_point
            self.precalib_params.u_center = center_point[0]
            self.precalib_params.v_center = center_point[1]

        self.inner_img_radius = inner_img_radius
        self.outer_img_radius = outer_img_radius
        self.h_radial_image = self.outer_img_radius - self.inner_img_radius

        if self.mirror_name == "bottom" or self.mirror_number == 2:
            # Objects in image appear inverted (upside-down)
            self.r_lowest_elevation = outer_img_radius;
            self.r_highest_elevation = inner_img_radius;
        else:  # top
            self.r_lowest_elevation = inner_img_radius;
            self.r_highest_elevation = outer_img_radius;

        # Needs to set GUM variables so the Cp is set accordingly with new params if any.
        self.set_model_params(**kwargs)

        self._compute_boundary_elevations()
        self._set_camera_FOVs()

        if preview and self.current_omni_img != None:
            vis_img = self.current_omni_img.copy()
            if self.mirror_number == 1:
                circle_color = (0, 255, 255)  # yellow in BGR
            else:
                circle_color = (0, 255, 0)  # green in BGR
            center_as_int = (int(self.precalib_params.u_center), int(self.precalib_params.v_center))
            cv2.circle(img=vis_img, center=center_as_int, radius=inner_img_radius, color=circle_color, thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(img=vis_img, center=center_as_int, radius=outer_img_radius, color=circle_color, thickness=1, lineType=cv2.LINE_AA)
            radial_limits_win = 'radial limits %s mirror' % self.mirror_name
            cv2.imshow(radial_limits_win, vis_img)
            cv2.waitKey(1)


    def set_omni_image(self, img, pano_width_in_pixels=1200, generate_panorama=False, idx=-1, view=True):
        self.current_omni_img = img
        if hasattr(self, "theoretical_model"):
            if self.theoretical_model is not None:
                self.theoretical_model.set_omni_image(img=img, pano_width_in_pixels=pano_width_in_pixels, generate_panorama=generate_panorama, idx=idx, view=view)

        if generate_panorama:
            import omnistereo.panorama as panorama
            self.panorama = panorama.Panorama(self, width=pano_width_in_pixels)
            self.panorama.set_panoramic_image(img, idx, view)
            try:
                mirror_name = self.mirror_name
            except:
                mirror_name = ""

            print(mirror_name.upper() + " Panorama's Settings:")
            self.panorama.print_settings()
        else:
            if self.panorama is not None:
                self.panorama.set_panoramic_image(img, idx, view)

    def _set_camera_FOVs(self):
        r_min_img = self.inner_img_radius
        r_max_img = self.outer_img_radius
        # Useless so far:
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        self.precalib_params.roi_max_x = self.precalib_params.u_center + r_max_img
        self.precalib_params.roi_max_y = self.precalib_params.v_center + r_max_img
        self.precalib_params.roi_min_x = self.precalib_params.u_center - r_max_img
        self.precalib_params.roi_min_y = self.precalib_params.v_center - r_max_img
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        h_x, h_y = self.precalib_params.pixel_size
        f = self.precalib_params.focal_length
        self.precalib_params.max_useful_FOV_hor = 2 * np.arctan((h_x * r_max_img) / f)
        self.precalib_params.min_useful_FOV_hor = 2 * np.arctan((h_x * r_min_img) / f)
        self.precalib_params.max_useful_FOV_ver = 2 * np.arctan((h_y * r_max_img) / f)
        self.precalib_params.min_useful_FOV_ver = 2 * np.arctan((h_y * r_min_img) / f)

    def get_vFOV(self, **kwargs):
        '''
        Computes the vertical Field of View based on the physical radial limits
        @return: the vertical field of view computed from their elevation angle limits
        '''
        theta_max = kwargs.get("theta_max", self.highest_elevation_angle)
        theta_min = kwargs.get("theta_min", self.lowest_elevation_angle)

        self.vFOV = theta_max - theta_min
        return self.vFOV

    def get_a_hyperbola(self, c, k):
        a = c / 2 * np.sqrt((k - 2.0) / k)
        return a

    def get_b_hyperbola(self, c, k):
        b = c / 2 * np.sqrt(2.0 / k)
        return b

    def detect_sparse_features_on_panorama(self, feature_detection_method="ORB", num_of_features=50, median_win_size=0, show=True):
        '''
        @param median_win_size: Window size for median blur filter. If value is 0, there is no filtering.
        @return: List of list of keypoints and ndarray of descriptors. The reason we use a list of each result is for separating mask-wise results for matchin.
        '''
        # Possible Feature Detectors are:
        # SIFT, SURF, DAISY
        # ORB:  oriented BRIEF
        # BRISK
        # AKAZE
        # KAZE
        # AGAST: Uses non-maximal suppression (NMS) by default
        # FAST: Uses non-maximal suppression (NMS) by default
        # GFT: Good Feasture to track
        # MSER: http://docs.opencv.org/master/d3/d28/classcv_1_1MSER.html#gsc.tab=0

        # Possible Feature Descriptors are:
        # SIFT, SURF: work to describe FAST and AGAST keypoints
        # ORB:  oriented BRIEF
        # BRISK
        # AKAZE
        # KAZE
        # DAISY: works to describe FAST and AGAST keypoints
        # FREAK: is essentially a descriptor.

        if feature_detection_method.upper() == "ORB":
            detector = cv2.ORB_create(nfeatures=num_of_features)
            descriptor = detector
#             descriptor = cv2.xfeatures2d.FREAK_create()
            # detector.setPatchSize(10)  # TODO: set according to mask's width
        if feature_detection_method.upper() == "AKAZE":
            detector = cv2.AKAZE_create()
            descriptor = detector
        if feature_detection_method.upper() == "KAZE":
            detector = cv2.KAZE_create()
            descriptor = detector
        if feature_detection_method.upper() == "BRISK":
            detector = cv2.BRISK_create()
            descriptor = detector
        if feature_detection_method.upper() == "SIFT":
            detector = cv2.xfeatures2d.SIFT_create()
            descriptor = detector
        if feature_detection_method.upper() == "SURF":
            detector = cv2.xfeatures2d.SURF_create()
            descriptor = detector
        if feature_detection_method.upper() == "FAST":
            detector = cv2.FastFeatureDetector_create()  # Uses non-maximal suppression (NMS) by default
            detector.setNonmaxSuppression(True)
            descriptor = cv2.xfeatures2d.DAISY_create()
#             descriptor = cv2.xfeatures2d.FREAK_create()
#             descriptor = cv2.xfeatures2d.SIFT_create() # FIXME: ORB and BRISK don't seem to work
        if feature_detection_method.upper() == "AGAST":
            detector = cv2.AgastFeatureDetector_create()  # Uses non-maximal suppression (NMS) by default
            detector.setNonmaxSuppression(True)
            descriptor = cv2.BRISK_create()
#             descriptor = cv2.xfeatures2d.DAISY_create()
#             descriptor = cv2.xfeatures2d.FREAK_create()
#             descriptor = cv2.xfeatures2d.SIFT_create()


        # TODO: implement other detectors, such as:
# >>> (kps, descs) = sift.detectAndCompute(gray, None)
# >>> print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
# # kps: 274, descriptors: (274, 128)
# >>> surf = cv2.xfeatures2d.SURF_create()
# >>> (kps, descs) = surf.detectAndCompute(gray, None)
#
# >>> kaze = cv2.KAZE_create()
# >>> (kps, descs) = kaze.detectAndCompute(gray, None)
# >>> print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
# # kps: 359, descriptors: (359, 64)
# >>> akaze = cv2.AKAZE_create()
# >>> (kps, descs) = akaze.detectAndCompute(gray, None)
# >>> print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
# # kps: 192, descriptors: (192, 61)
# >>> brisk = cv2.BRISK_create()
# >>> (kps, descs) = brisk.detectAndCompute(gray, None)

        pano_img = self.panorama.panoramic_img.copy()
        if self.panorama.panoramic_img is not None:
            if feature_detection_method.upper() == "GFT":
                if pano_img.ndim == 3:
                    pano_img = cv2.cvtColor(pano_img, cv2.COLOR_BGR2GRAY)
#                 else:
#                     pano_img = self.panorama.panoramic_img.copy()
            if median_win_size > 0:
                pano_img = cv2.medianBlur(pano_img, median_win_size)

        #=======================================================================
        # TODO: implement tracking of GoodFeaturesToTrack: NOTE: doesn't seem to be consistent for matching against the query image
        if feature_detection_method.upper() == "GFT":
            pts_GFT = cv2.goodFeaturesToTrack(image=pano_img, maxCorners=num_of_features * 10, qualityLevel=0.005, minDistance=50)
            pts = pts_GFT.reshape((pts_GFT.shape[0], pts_GFT.shape[2]))
            if show:
                from omnistereo.common_cv import draw_points
                pano_with_keypts = self.panorama.panoramic_img.copy()
                draw_points(pano_with_keypts, pts)
                cv2.imshow("Detected Keypoints - Panorama of Mirror %d" % (self.mirror_number), pano_with_keypts)
                cv2.waitKey(1)
            return pts, None

        #=======================================================================

        keypts_detected_list = []
        descriptors_list = []
        if len(self.panorama.azimuthal_masks) > 0:
            # When spliting panorama in masks for spreading info around the 360 degree view.
            if show:
                pano_with_keypts = pano_img.copy()
            for m in self.panorama.azimuthal_masks:
#                 keypts_detected_on_mask, descriptors_on_mask = detector.detectAndCompute(image=pano_img, mask=m)
                keypts_detected_on_mask = detector.detect(image=pano_img, mask=m)
                keypts_detected_on_mask, descriptors_on_mask = descriptor.compute(image=pano_img, keypoints=keypts_detected_on_mask)
                keypts_detected_list.append(keypts_detected_on_mask)
                descriptors_list.append(descriptors_on_mask)
                if show:
                    pano_with_keypts = cv2.drawKeypoints(pano_with_keypts, keypts_detected_on_mask, outImage=pano_with_keypts)
                    cv2.imshow("Detected Keypoints - Panorama of Mirror %d" % (self.mirror_number), pano_with_keypts)
                    cv2.waitKey(1)
        else:
            keypts_detected = detector.detect(image=pano_img, mask=None)
            keypts_detected, descriptors = descriptor.compute(image=pano_img, keypoints=keypts_detected_on_mask)
            keypts_detected_list.append(keypts_detected)
            descriptors_list.append(descriptors)
            # keypoints are put in a list of lenght n
            # descriptor are ndarrays of n x desc_size

            if show:
                pano_with_keypts = pano_img.copy()
                pano_with_keypts = cv2.drawKeypoints(pano_img, keypts_detected, outImage=pano_with_keypts)
                cv2.imshow("Detected Keypoints - Panorama of Mirror %d" % (self.mirror_number), pano_with_keypts)
                cv2.waitKey(1)

        return keypts_detected_list, descriptors_list

    def approximate_with_PnP(self, img_points, obj_pts_homo):
        '''
        Approximate the transformation (pose) between a set of points on the image plane and their corresponding points on another plane
        '''

        raise NotImplementedError

    def draw_radial_bounds(self, omni_img=None, is_reference=False, view=True):
        '''
        @param is_reference: When set to True, it will use dotted lines and cross hairs for the center
        '''
        if omni_img is None:
            omni_img = self.current_omni_img

        center_point = self.precalib_params.center_point
        r_min = self.inner_img_radius
        r_max = self.outer_img_radius


        if hasattr(self, "mirror_name"):
            mirror_name = self.mirror_name.upper()
        else:
            if self.mirror_number == 1:
                mirror_name = "TOP"
            else:
                mirror_name = "BOTTOM"

        if mirror_name == "TOP":
            color = (255, 255, 0)  # GBR for cyan (top)
            delta_theta = 4  # degrees
        else:
            color = (255, 0, 255)  # GBR for magenta (bottom)
            delta_theta = 6  # degrees

        img = omni_img.copy()
        # Draw:
        # circle center
        u_c, v_c = int(center_point[0]), int(center_point[1])
        center = (u_c, v_c)
        if is_reference:
            dot_radius = 5
            cv2.circle(img, center, dot_radius, color, -1, cv2.LINE_AA, 0)
            # circle outline dotted
            # Because dots of the smaller radius circular bound should be more sparse
            # WISH: Factorize this drawing of dotted circle as a general function for common_cv
            delta_theta_min = np.deg2rad(delta_theta)  # written in degrees
            delta_theta_max = np.deg2rad(delta_theta / 2)  # written in degrees
            thetas_min = np.arange(start=0, stop=2 * np.pi, step=delta_theta_min)
            thetas_max = np.arange(start=0, stop=2 * np.pi, step=delta_theta_max)
            u_coords_min = u_c + r_min * np.cos(thetas_min)
            v_coords_min = v_c + r_min * np.sin(thetas_min)
            u_coords_max = u_c + r_max * np.cos(thetas_max)
            v_coords_max = v_c + r_max * np.sin(thetas_max)
            for u_min, v_min in zip(u_coords_min, v_coords_min):
                cv2.circle(img, (int(u_min), int(v_min)), dot_radius, color, -1, cv2.LINE_AA, 0)
            for u_max, v_max in zip(u_coords_max, v_coords_max):
                cv2.circle(img, (int(u_max), int(v_max)), dot_radius, color, -1, cv2.LINE_AA, 0)
        else:
            # Use crosshair as center point
            thickness = 3
            cross_hair_length = 15
            cv2.line(img=img, pt1=(u_c - cross_hair_length, v_c), pt2=(u_c + cross_hair_length, v_c), color=color, thickness=thickness, lineType=cv2.LINE_AA)  # crosshair horizontal
            cv2.line(img=img, pt1=(u_c, v_c - cross_hair_length), pt2=(u_c, v_c + cross_hair_length), color=color, thickness=thickness, lineType=cv2.LINE_AA)  # crosshair vertical
            # circle outline
            cv2.circle(img, center, int(r_min), color, thickness, cv2.LINE_AA, 0)
            cv2.circle(img, center, int(r_max), color, thickness, cv2.LINE_AA, 0)

        # Show radial boundaries
        if view:
            win_name = mirror_name + " - Radial Bounds"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, img)
            cv2.waitKey(1)

        return img


class OmniStereoModel(object):
    '''
    The vertically-folded omnistereo model using GUM
    '''

    def __init__(self, top_model, bottom_model, **kwargs):
        '''
        Constructor
        '''
        self.top_model = top_model
        self.bot_model = bottom_model
        self.units = top_model.units
        self.set_params(**kwargs)
        self.infer_additional_parameters_from_models()
#         self.calibrator = calibration.CalibratorStereo(self)
        self.baseline = self.get_baseline()
        self.current_omni_img = None
        self.construct_new_mask = True
        self.mask_RGB_color = None
        self.mask_background_img = None

    def set_params(self, **kwargs):

        if "center_point_top" in kwargs:
            center_point_top = kwargs.get("center_point_top", self.top_model.precalib_params.center_point)
            self.top_model.precalib_params.center_point = center_point_top
            self.top_model.precalib_params.u_center = center_point_top[0]
            self.top_model.precalib_params.v_center = center_point_top[1]
        if "center_point_bottom" in kwargs:
            center_point_bottom = kwargs.get("center_point_bottom", self.bot_model.precalib_params.center_point)
            self.bot_model.precalib_params.center_point = center_point_bottom
            self.bot_model.precalib_params.u_center = center_point_bottom[0]
            self.bot_model.precalib_params.v_center = center_point_bottom[1]

        # In case radii haven't been passed,
        # infer occlusion boundaries automatically from pre-calibration ROI parameters:
        _, v_center_top = self.top_model.get_center()
        _, v_center_bottom = self.bot_model.get_center()

        outer_radius_top = kwargs.get("outer_radius_top", self.top_model.outer_img_radius)
        if outer_radius_top == 0:
            try:
                outer_radius_top = v_center_top - self.top_model.precalib_params.roi_max_y
            except:
                print("outer_radius_top not set")
                pass

        inner_radius_top = kwargs.get("inner_radius_top", self.top_model.inner_img_radius)
        if inner_radius_top == 0:
            try:
                inner_radius_top = v_center_top - self.bot_model.precalib_params.roi_max_y
            except:
                print("inner_radius_top not set")
                pass

        outer_radius_bottom = kwargs.get("outer_radius_bottom", self.bot_model.outer_img_radius)
        if outer_radius_bottom == 0:
            try:
                outer_radius_bottom = v_center_bottom - self.bot_model.precalib_params.roi_max_y
            except:
                print("outer_radius_bottom not set")
                pass

        inner_radius_bottom = kwargs.get("inner_radius_bottom", self.bot_model.inner_img_radius)
        if inner_radius_bottom == 0:
            try:
                inner_radius_bottom = 0  # It's zero for now (not masking anything). Also, the pre-calibration roi_min seems
            except:
                print("inner_radius_bottom not set")
                pass

        if outer_radius_top > 0 and outer_radius_bottom > 0:
            self.set_radial_limits_in_pixels(outer_radius_top, inner_radius_top, outer_radius_bottom, inner_radius_bottom)

        self.common_vFOV = self.set_common_vFOV()

    def infer_additional_parameters_from_models(self):
        raise NotImplementedError

    def get_baseline(self, **kwargs):
        raise NotImplementedError

    def get_triangulated_point_wrt_Oc(self, elev1, elev2, azimuth):
        '''
        @brief Compute triangulated point's cartesian coordinates w.r.t. origin of camera frame using the given direction angles (in radians)

        @param alpha1: The elevation angle (or ndarray of angles) to the point with respect to mirror 1's focus \f$F_1$\f
        @param alpha2: The elevation angle (or ndarray of angles) to the point with respect to mirror 2's focus \f$F_2$\f
        @param phi: The common azimuth angle (or ndarray of angles) of the triangulated point

        @return: If parameters are scalars (for one point), the returned point is given as a simple (x,y,z) tuple. Otherwise, the np.ndarray (rows, cols, 3) encoding the Cartessian (x,y,z) coordinates of the triangulated point
        '''
        # Horizontal range:
        # FIXME: handle division by 0.
        rho = (self.baseline * np.cos(elev1) * np.cos(elev2)) / (np.sin(elev1 - elev2))
        # Compute triangulated point's cartesian coordinates w.r.t. origin of camera frame
        x = -rho * np.cos(azimuth)
        y = -rho * np.sin(azimuth)
        z = self.top_model.F[2, 0] - rho * np.tan(elev1)

        Pw_wrt_C = np.dstack((x, y, z))

        return Pw_wrt_C

    # TODO: Test multiple points (matrix of points as rows, cols)
    # TODO: pass omnidirectional correspondence points directly instead of rays
    def get_triangulated_midpoint(self, dir_ray1, dir_ray2):
        '''
        Approximates the triangulated 3D point (for both cases: intersecting rays or skew rays)
        using the common perpendicular to both back-projection rays \f$(\vect{v}_1,\vect{v}_2)$\f

        @param dir_ray1: Ray leaving top focus point. It must be of shape [rows,cols,3]
        @param dir_ray2: Ray leaving bottom focus point. It must be of shape [rows,cols,3]
        @note: back-projection vectors \f$(\vect{v}_1,\vect{v}_2)$\f must be given wrt focus

        @reval: mid_Pw: The midpoint (in Eucledian coordinates, a.k.a 3-vector) of the common perpendicular between the 2 direction rays.
        @retval (lambda1, lambda2, lambda_perp): a tuple of the relevant line parameters
        @reval: (G1, G1): The coordinates (wrt to reference/camera frame) for the end points on the common perpendicular line segment
        @reval: (perp_vect_unit, perp_mag): A tuple containing the direction (unit vector) of the common perpendicular to both rays and its magnitude (closes distance among both rays)
        '''
        v1 = dir_ray1[..., :3]
        v2 = dir_ray2[..., :3]

        perp_vect = np.cross(v1, v2)
        perp_vect_mag = np.linalg.norm(perp_vect, axis=-1)[..., np.newaxis]

#         if perp_vect_mag > 0:
        perp_vect_unit = perp_vect / perp_vect_mag
        # Solve the system of linear equations
        # Given as M * t = b. Then, t = M^-1 * b
        v1 = v1[..., np.newaxis]  # >>> Gives shape (...,3,1)
        v2 = v2[..., np.newaxis]  # >>> Gives shape (...,3,1)
        perp_vect_unit = perp_vect_unit[..., np.newaxis]  # >>> Gives shape (...,3,1)
        M = np.concatenate((v1, -v2, perp_vect_unit), axis=-1).reshape(v1.shape[0:-2] + (3, 3))  # >>> Gives shape (...,3,3)
        # NOTE: Working with column vectors, so we reshape along the way... (watch out for misshapes)
        f1 = self.top_model.F[:-1]
        f2 = self.bot_model.F[:-1]

        b = np.zeros_like(v1) + (f2 - f1)  # >>> Gives shape (...,3,1)
        # Use equation for lenght of common perpendicular:
#         perp_mag = np.abs(np.inner(b[..., 0], perp_vect_unit[..., 0]))  # >>> Gives shape ()
        perp_mag = np.abs(np.einsum("mnt, mnt->mn", b[..., 0], perp_vect_unit[..., 0]))
#===============================================================================
#         perp_mag_approx = np.zeros_like(perp_mag)
#         np.around(perp_mag, 10, perp_mag_approx)
#===============================================================================
#             ans = np.where(perp_mag_approx > 0, self.triangulate_for_skew_rays(f1, v1, perp_vect_unit, M, b), some_other_func)
#             mid_Pw, lambda1, lambda2, lambda_perp = ans[..., 0, 0], ans[..., 0, 1], ans[..., 0, 2], ans[..., 0, 3]
        # Method works for all cases (intersecting rays or skew rays)
        try:
            ans = self.triangulate_for_skew_rays(f1, v1, perp_vect_unit, M, b)  # >>> Gives shape (...,4)
            # mid_Pw, lambda1, lambda2, lambda_perp = ans[..., 0], ans[..., 1], ans[..., 2], ans[..., 3]
            mid_Pw, lambda1, lambda2, lambda_perp, G1 = ans[0], ans[1], ans[2], ans[3], ans[4]
            G2 = f2 + lambda2[..., np.newaxis] * v2
        except:
            print("Problem")
#===============================================================================
#         else:
#             # FIXME: handle diverging rays so
#
#             # Backup method using triangulation via trigonometry and intersection assumption
#             pass
#             # TODO: add function to produce results using the regular triangulation function
#===============================================================================

        return mid_Pw[..., 0], (lambda1, lambda2, lambda_perp), (G1[..., 0], G2[..., 0]), (perp_vect_unit[..., 0], perp_mag)  # perp_mag[...][0, 0])

    def triangulate_for_skew_rays(self, f1, v1, perp_vect_unit, M, b):
        # TODO: Vectorize or loop across rows and cols, eg.  f1[row,col]
        lambda_solns = np.linalg.solve(M, b)  # Implying solution vector [lambda1, lambda2, lambda_perp]

#         lambda1, lambda2, lambda_perp = lambda_solns[..., 0, 0], lambda_solns[..., 1, 0], lambda_solns[..., 2, 0]
        lambda1, lambda2, lambda_perp = lambda_solns[..., 0, :], lambda_solns[..., 1, :], lambda_solns[..., 2, :]
        # Point G1 (end point on common perpendicular)
        G1 = f1 + lambda1[..., np.newaxis] * v1
        mid_Pw = G1 + lambda_perp[..., np.newaxis] / 2.0 * perp_vect_unit
        return mid_Pw, lambda1, lambda2, lambda_perp, G1


    def resolve_pano_correspondences_from_disparity_map(self, ref_points_uv_coords, min_disparity=1, max_disparity=0, verbose=False, roi_cols=None):
        '''
        @return: 2 correspondence lists of pixel coordinates (as separate ndarrays of n rows by 2 cols) regarding the (u,v) coordinates on the top and the bottom
        '''
        # NOTE: the disparity map is linked to the left/top image so it is the reference for the correspondences
        #       Thus, the coordinates of the correspondence on the right/bottom image is found such that m_right = (u_left, v_left - disp[u,v])
        # NOTE: Throughout, the order of coordinates as (u,v) need to be swapped to (row, col)


        if roi_cols is not None:  # Fill up zeroes to those disparities outside of the ROI columns
            disparity_map = np.zeros_like(self.disparity_map)
            disparity_map[:, roi_cols[0]:roi_cols[1]] = self.disparity_map[:, roi_cols[0]:roi_cols[1]]
        else:
            disparity_map = self.disparity_map

        if max_disparity == 0:
            max_disparity = disparity_map.max()

        # Non-zero disparities:
        nonzero_disp_indices = disparity_map[ref_points_uv_coords[..., 1], ref_points_uv_coords[..., 0]] != 0
        ref_pano_points_coords_nonzero = ref_points_uv_coords[nonzero_disp_indices]

        # Select only those coordinates with valid disparities (based on minimum threshold)
        is_greater_than_min = min_disparity <= disparity_map[ref_pano_points_coords_nonzero[..., 1], ref_pano_points_coords_nonzero[..., 0]]
        is_less_than_max = disparity_map[ref_pano_points_coords_nonzero[..., 1], ref_pano_points_coords_nonzero[..., 0]] <= max_disparity
        valid_disp_indices = np.logical_and(is_greater_than_min, is_less_than_max)

        ref_pano_points_coords_valid = ref_pano_points_coords_nonzero[valid_disp_indices]
        disparities = disparity_map[ref_pano_points_coords_valid[..., 1], ref_pano_points_coords_valid[..., 0]]

        # Filter within target bounds:
        # FIXME: Dont' do this in the case of bad misaligments
        lowest_reference_row = self.bot_model.panorama.get_panorama_row_from_elevation(self.bot_model.lowest_elevation_angle)
        disp_indices_within_bounds = ref_pano_points_coords_valid[..., 1] - disparities <= lowest_reference_row
        ref_pano_points_coords = ref_pano_points_coords_valid[disp_indices_within_bounds]

        disparities = disparity_map[ref_pano_points_coords[..., 1], ref_pano_points_coords[..., 0]]
        target_pano_points_coords = np.transpose((ref_pano_points_coords[..., 0], ref_pano_points_coords[..., 1] - disparities))
        if verbose and np.count_nonzero(ref_pano_points_coords) > 0:
            print("CORRESPONDENCE: %s <-> %s using DISP: %f" % (ref_pano_points_coords[-1], target_pano_points_coords[-1], disparities[-1]))

        # Add an extra axis to form a 3 dimensional table.
        if ref_pano_points_coords.ndim < 3:
            ref_pano_points_coords = ref_pano_points_coords[np.newaxis, ...]
        if target_pano_points_coords.ndim < 3:
            target_pano_points_coords = target_pano_points_coords[np.newaxis, ...]

        return ref_pano_points_coords, target_pano_points_coords, disparities

    def get_correspondences_from_clicked_points(self, min_disparity=1, max_disparity=0):
        from omnistereo.common_cv import PointClicker
        # Get point clicked on panoramic image and mark it (visualize it)
        click_window_name = 'Reference Points (To Click)'
        cv2.namedWindow(click_window_name, cv2.WINDOW_NORMAL)
        pt_clicker = PointClicker(click_window_name, max_clicks=1000)
        top_pano_coords, bot_pano_coords, disparities = pt_clicker.get_clicks_uv_coords_for_stereo(stereo_model=self, show_correspondence_on_circular_img=True, min_disparity=min_disparity, max_disparity=max_disparity, verbose=True)

        #=======================================================================
        # _, _, omni_top_coords = self.top_model.panorama.get_omni_pixel_coords_from_panoramic_pixel(top_pano_coords, use_LUTs=False)
        # _, _, omni_bot_coords = self.bot_model.panorama.get_omni_pixel_coords_from_panoramic_pixel(bot_pano_coords, use_LUTs=False)
        # return omni_top_coords, omni_bot_coords, disparities
        #=======================================================================

        return top_pano_coords, bot_pano_coords, disparities

    def triangulate_from_clicked_points(self, min_disparity=1, max_disparity=0, use_PCL=False, export_to_pcd=True, cloud_path="data", cloud_index=None, use_LUTs=True,):
        '''
        @param cloud_index: used for saving identifiable point clouds.
        '''
        top_pano_points_coords, bot_pano_points_coords, pano_disparities = self.get_correspondences_from_clicked_points(min_disparity=min_disparity, max_disparity=max_disparity)  # For testing disparity matches purposes
        az1, el1 = self.top_model.panorama.get_direction_angles_from_pixel_pano(top_pano_points_coords, use_LUTs=use_LUTs)
        az2, el2 = self.bot_model.panorama.get_direction_angles_from_pixel_pano(bot_pano_points_coords, use_LUTs=use_LUTs)
        # Get XYZ from triangulation and put into some cloud
        points_3D_homo = self.get_triangulated_point_from_direction_angles(dir_angs_top=(az1, el1), dir_angs_bot=(az2, el2), use_midpoint_triangulation=False)
        return self.generate_point_clouds(points_3D_homo, top_pano_points_coords, use_PCL=use_PCL, export_to_pcd=export_to_pcd, cloud_path=cloud_path, cloud_index=cloud_index)

    def triangulate_from_depth_map(self, min_disparity=1, max_disparity=0, use_PCL=False, export_to_pcd=True, cloud_path="data", use_LUTs=True, roi_cols=None, use_midpoint_triangulation=False, cloud_index=None):
        '''
        @param cloud_index: used for saving identifiable point clouds.
        '''
        # Get matching pairs
        ref_points_uv_coords = np.transpose(np.indices(self.disparity_map.shape[::-1]), (1, 2, 0))
        top_pano_points_coords, bot_pano_points_coords, pano_disparities = self.resolve_pano_correspondences_from_disparity_map(ref_points_uv_coords, min_disparity=min_disparity, max_disparity=max_disparity, roi_cols=roi_cols)

        #=======================================================================
        # _, _, omni_top_coords = self.top_model.panorama.get_omni_pixel_coords_from_panoramic_pixel(top_pano_points_coords, use_LUTs=use_LUTs)
        # _, _, omni_bot_coords = self.bot_model.panorama.get_omni_pixel_coords_from_panoramic_pixel(bot_pano_points_coords, use_LUTs=use_LUTs)
        # # Get XYZ from triangulation and put into some cloud
        # xyz_points = self.get_triangulated_point_from_pixels(m1=omni_top_coords, m2=omni_bot_coords, use_midpoint_triangulation=False)
        #=======================================================================

        az1, el1 = self.top_model.panorama.get_direction_angles_from_pixel_pano(top_pano_points_coords, use_LUTs=use_LUTs)
        az2, el2 = self.bot_model.panorama.get_direction_angles_from_pixel_pano(bot_pano_points_coords, use_LUTs=use_LUTs)
        # Get XYZ from triangulation and put into some cloud
        points_3D_homo = self.get_triangulated_point_from_direction_angles(dir_angs_top=(az1, el1), dir_angs_bot=(az2, el2), use_midpoint_triangulation=use_midpoint_triangulation)

        return self.generate_point_clouds(points_3D_homo, top_pano_points_coords, use_PCL=use_PCL, export_to_pcd=export_to_pcd, cloud_path=cloud_path, cloud_index=cloud_index)

    def generate_point_clouds(self, xyz_points, pano_ref_uv_coords, rgb_colors=None, use_PCL=False, export_to_pcd=True, cloud_path="data", cloud_index=None):
        '''
        @param cloud_index: used for saving identifiable point clouds.
        @param rgb_colors: An numpy array of RGB colors can be optionally passed. This is useful in case of experimenting with spartse features where individual points must be distinguish from each other.
        '''
        # set RGB info to complimentary cloud
        #=======================================================================
        # Using OMNI image
        # channels = self.current_omni_img.ndim
        # # RGB color is taken from either panorama (top panorama for now) # TODO: maybe color should be averaged
        # if channels == 1:
        #     # GrayScale
        #     ref_color_img = cv2.cvtColor(self.current_omni_img, cv2.COLOR_GRAY2RGB).astype('float32')
        # else:
        #     ref_color_img = cv2.cvtColor(self.current_omni_img, cv2.COLOR_BGR2RGB).astype('float32')
        # # Recall that now we are giving rows and cols as coords in the omni image
        # rgb_points = ref_color_img[list(omni_top_coords[..., 1].flatten()), list(omni_top_coords[..., 0].flatten())]
        #=======================================================================

        if rgb_colors is None:  # Using PANORAMIC image
            channels = self.top_model.panorama.panoramic_img.ndim
            # RGB color is taken from either panorama (top panorama for now) # TODO: maybe color should be averaged
            if channels == 1:
                # GrayScale
                ref_color_img = cv2.cvtColor(self.top_model.panorama.panoramic_img, cv2.COLOR_GRAY2RGB).astype('float32')
            else:
                ref_color_img = cv2.cvtColor(self.top_model.panorama.panoramic_img, cv2.COLOR_BGR2RGB).astype('float32')
            # Recall that now we are giving rows and cols as coords in the omni image
            rgb_points = ref_color_img[list(pano_ref_uv_coords[..., 1].flatten()), list(pano_ref_uv_coords[..., 0].flatten())]
        else:
            rgb_points = rgb_colors

        # Generate clouds
        if use_PCL:
            import pcl
            xyz_cloud = pcl.PointCloud(xyz_points[..., :3].reshape(-1, 3).astype('float32'))
            rgb_cloud = pcl.PointCloud(rgb_points.reshape(-1, 3).astype('float32'))

            if export_to_pcd:
                from omnistereo.common_tools import make_sure_path_exists
                make_sure_path_exists(cloud_path)  # This creates the path if necessary
                if cloud_index is None:
                    cloud_id = ""
                else:
                    cloud_id = "-%d" % (cloud_index)
                pcd_xyz_filename = "%s/XYZ%s.pcd" % (cloud_path, cloud_id)
                pcd_rgb_filename = "%s/RGB%s.pcd" % (cloud_path, cloud_id)
                # Export the 2 pointclouds to PCD files
                pcl.save(cloud=xyz_cloud, path=pcd_xyz_filename, format="pcd", binary=False)
                print("Saved XYZ cloud to %s" % pcd_xyz_filename)
                pcl.save(cloud=rgb_cloud, path=pcd_rgb_filename, format="pcd", binary=False)
                print("Saved RGB cloud to %s" % pcd_rgb_filename)
#===============================================================================
#         else:  # TODO: implemet the visualization with Mayavi once it successfully installed (pain in the ars!)
#             pass
#             # Using Matplotlib 3D (Too Slow)
#             from mpl_toolkits.mplot3d import axes3d
#             import matplotlib.pyplot as plt
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d', aspect="equal")
#
#             xyz = xyz_points[...,:3].reshape(-1, 3)
#
#             # Define coordinates and points
#             x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]  # Assign x, y, z values to match color
#             ax.scatter(x, y, z, c=rgb_points.reshape(-1, 3) / 255., s=50)
#             plt.show()
#===============================================================================


#===============================================================================
#             # Imports
# #             from mayavi.mlab import quiver3d, draw
#             from mayavi.mlab import points3d, draw
#
#             rgb = rgb_points.reshape(-1, 3).astype(np.uint8)
#             xyz = xyz_points.reshape(-1, 3)
#
#             # Primitives
#             N = rgb.shape[0]  # Number of points
#             ones = np.ones(N)
#             scalars = np.arange(N)  # Key point: set an integer for each point
#
#             # Define color table (including alpha), which must be uint8 and [0,255]
#             colors = np.zeros((N, 4) , dtype=np.uint8)
#             colors[:, :3] = rgb  # RGB color channels
#             colors[:, -1] = 255  # No transparency
#
#             # Define coordinates and points
# #             x, y, z = colors[:, 0], colors[:, 1], colors[:, 2]  # Assign x, y, z values to match color
#             x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]  # Assign x, y, z values to match color
#             pts = points3d(x, y, z)  # , ones, scale_mode='none')
# #             pts = quiver3d(x, y, z, ones, ones, ones, scalars=scalars, mode='sphere')  # Create points
# #             pts.glyph.color_mode = 'color_by_scalar'  # Color by scalar
# #             # Set look-up table and redraw
# #             pts.module_manager.scalar_lut_manager.lut.table = colors
# #             draw()
#===============================================================================


        return xyz_points, rgb_points

    def set_radial_limits_in_pixels(self, outer_radius_top, inner_radius_top, outer_radius_bottom, inner_radius_bottom, **kwargs):
        self.top_model.set_radial_limits_in_pixels_mono(inner_radius_top, outer_radius_top, **kwargs)
        self.bot_model.set_radial_limits_in_pixels_mono(inner_radius_bottom, outer_radius_bottom, **kwargs)
        # compute the highest/lowest elevation globally among both top vs bottom
        global_high_elev = max(self.top_model.highest_elevation_angle, self.bot_model.highest_elevation_angle)
        global_low_elev = min(self.top_model.lowest_elevation_angle, self.bot_model.lowest_elevation_angle)  # Reset new global highest and lowest elevation angles on each GUM
        self.top_model.globally_highest_elevation_angle = global_high_elev
        self.bot_model.globally_highest_elevation_angle = global_high_elev
        self.top_model.globally_lowest_elevation_angle = global_low_elev
        self.bot_model.globally_lowest_elevation_angle = global_low_elev

    def draw_radial_bounds_stereo(self, omni_img=None, is_reference=False, view=True):
        if omni_img is None:
            omni_img = self.current_omni_img

        img = self.top_model.draw_radial_bounds(omni_img=omni_img, is_reference=is_reference, view=False)
        img = self.bot_model.draw_radial_bounds(omni_img=img, is_reference=is_reference, view=False)
        if view:
            win_name = "OMNISTEREO Radial Bounds"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, img)
            cv2.waitKey(1)

        return img

    def generate_panorama_pair(self, omni_img, width_in_pixels=1200, idx=-1, view=True, win_name_modifier="", use_mask=True, border_RGB_color=None):
        if self.top_model.panorama is None or self.bot_model.panorama is None:
            self.set_current_omni_image(self, omni_img, pano_width_in_pixels=width_in_pixels, generate_panoramas=True, idx=idx, view=view)
        # Same omni image by default
        omni_img_top = omni_img
        omni_img_bot = omni_img

        if use_mask:
            import warnings
            try:
                from omnistereo.common_cv import get_masked_images_as_pairs
                omni_images_as_pairs = get_masked_images_as_pairs(unmasked_images=[omni_img], omnistereo_model=self, show_images=True, color_RGB=border_RGB_color)
                omni_img_top, omni_img_bot = omni_images_as_pairs[0]
                self.top_model.panorama.set_panoramic_image(omni_img=omni_img_top, idx=idx, view=True, win_name_modifier=win_name_modifier, border_RGB_color=border_RGB_color)
                self.bot_model.panorama.set_panoramic_image(omni_img=omni_img_bot, idx=idx, view=True, win_name_modifier=win_name_modifier, border_RGB_color=border_RGB_color)
            except:
                warnings.warn("Panorama (Pair) index %d problem in %s" % (idx, __name__))

        pano_img_top = self.top_model.panorama.set_panoramic_image(omni_img_top, idx, view, win_name_modifier, border_RGB_color=border_RGB_color)
        pano_img_bot = self.bot_model.panorama.set_panoramic_image(omni_img_bot, idx, view, win_name_modifier, border_RGB_color=border_RGB_color)
        return pano_img_top, pano_img_bot

    def view_all_panoramas(self, omni_images_filename_pattern, img_indices, win_name_modifier="", use_mask=False, mask_color_RGB=None):
        import warnings
        from omnistereo.common_cv import get_images

        omni_images = get_images(omni_images_filename_pattern, indices_list=img_indices, show_images=not use_mask)

        if img_indices is None or len(img_indices) == 0:  # Default value
            # use all the images in the set:
            img_indices = range(len(omni_images))

        for i in img_indices:
            try:
                #===============================================================
                # from time import process_time  # , perf_counter
                # start_time = process_time()
                #===============================================================
                self.generate_panorama_pair(omni_images[i], idx=i, view=True, win_name_modifier=win_name_modifier, use_mask=use_mask, border_RGB_color=mask_color_RGB)
                #===============================================================
                # end_time = process_time()
                # time_ellapsed_1 = end_time - start_time
                # print("Time elapsed: {time:.8f} seconds".format(time=time_ellapsed_1))
                #===============================================================
            except:
                warnings.warn("Image index %d not found at %s" % (i, __name__))


    def draw_elevations_on_panoramas(self, left8=None, right8=None, draw_own_limits=False):
        '''
        NOTE: Image should not be cropped.
        '''
        if left8 is None:
            left8_annotated = self.top_model.panorama.panoramic_img.copy()
        else:
            left8_annotated = left8.copy()

        if right8 is None:
            right8_annotated = self.bot_model.panorama.panoramic_img.copy()
        else:
            right8_annotated = right8.copy()

        pano_cols = int(self.top_model.panorama.cols)
        last_col = pano_cols - 1
        row_common_highest = self.top_model.panorama.get_panorama_row_from_elevation(self.common_highest_elevation_angle)
        row_common_lowest = self.top_model.panorama.get_panorama_row_from_elevation(self.common_lowest_elevation_angle)

        # Draw 0-degree line
        row_at_zero_degrees = self.top_model.panorama.get_panorama_row_from_elevation(0)
        if not np.isnan(row_at_zero_degrees):
            line_color = (255, 0, 255)  # magenta in BGR
            cv2.line(img=left8_annotated, pt1=(0, row_at_zero_degrees), pt2=(last_col, row_at_zero_degrees), color=line_color, thickness=1 , lineType=cv2.LINE_AA)
            cv2.line(img=right8_annotated, pt1=(0, row_at_zero_degrees), pt2=(last_col, row_at_zero_degrees), color=line_color, thickness=1 , lineType=cv2.LINE_AA)

        # Draw line at common highest elevation
        if not np.isnan(row_common_highest):
            line_color = (255, 0, 0)  # blue in BGR
            cv2.line(img=left8_annotated, pt1=(0, row_common_highest), pt2=(last_col, row_common_highest), color=line_color, thickness=1 , lineType=cv2.LINE_AA)
            cv2.line(img=right8_annotated, pt1=(0, row_common_highest), pt2=(last_col, row_common_highest), color=line_color, thickness=1 , lineType=cv2.LINE_AA)

        # Draw line at common lowest elevation
        if not np.isnan(row_common_lowest):
            line_color = (0, 255, 0)  # green in BGR
            cv2.line(img=left8_annotated, pt1=(0, row_common_lowest), pt2=(last_col, row_common_lowest), color=line_color, thickness=1 , lineType=cv2.LINE_AA)
            cv2.line(img=right8_annotated, pt1=(0, row_common_lowest), pt2=(last_col, row_common_lowest), color=line_color, thickness=1 , lineType=cv2.LINE_AA)

        if draw_own_limits:
            line_color = (0, 0, 255)  # red in BGR
            # Top:
            top_highest = self.top_model.panorama.get_panorama_row_from_elevation(self.top_model.highest_elevation_angle)
            top_lowest = self.top_model.panorama.get_panorama_row_from_elevation(self.top_model.lowest_elevation_angle)
            cv2.line(img=left8_annotated, pt1=(0, top_highest), pt2=(last_col, top_highest), color=line_color, thickness=1 , lineType=cv2.LINE_AA)
            cv2.line(img=right8_annotated, pt1=(0, top_lowest), pt2=(last_col, top_lowest), color=line_color, thickness=1 , lineType=cv2.LINE_AA)
            # Bottom:
            bot_highest = self.bot_model.panorama.get_panorama_row_from_elevation(self.bot_model.highest_elevation_angle)
            bot_lowest = self.bot_model.panorama.get_panorama_row_from_elevation(self.bot_model.lowest_elevation_angle)
            cv2.line(img=left8_annotated, pt1=(0, bot_highest), pt2=(last_col, bot_highest), color=line_color, thickness=1 , lineType=cv2.LINE_AA)
            cv2.line(img=right8_annotated, pt1=(0, bot_lowest), pt2=(last_col, bot_lowest), color=line_color, thickness=1 , lineType=cv2.LINE_AA)

        cropped_top_win = 'Annotated panorama (top or LEFT)'
        cv2.namedWindow(cropped_top_win, cv2.WINDOW_NORMAL)
        cv2.imshow(cropped_top_win, left8_annotated)

        cropped_bot_win = 'Annotated panorama (bottom or RIGHT)'
        cv2.namedWindow(cropped_bot_win, cv2.WINDOW_NORMAL)
        cv2.imshow(cropped_bot_win, right8_annotated)
        pressed_key = cv2.waitKey(10)

        return pressed_key

    def get_fully_masked_images(self, omni_img=None, view=True, color_RGB=None):
        '''
        @param color_RGB: A tuple specifying the desired background as (Red,Green,Blue). If None, the background is black

        @return: masked_img_top, masked_img_bottom
        '''
        if omni_img is None:
            omni_img = self.current_omni_img

        # Small performance improvement by only generating masks if needed. Now the masks are stored for reusability!
        if self.construct_new_mask:
            # TOP:
            center_point_top = self.top_model.precalib_params.center_point
            center_point_bottom = self.bot_model.precalib_params.center_point
            r_inner_top = self.top_model.inner_img_radius
            r_outer_top = self.top_model.outer_img_radius
            r_inner_bottom = self.bot_model.inner_img_radius
            r_outer_bottom = self.bot_model.outer_img_radius
            # circle centers
            center_top = (int(center_point_top[0]), int(center_point_top[1]))
            center_bot = (int(center_point_bottom[0]), int(center_point_bottom[1]))
            mask_top = np.zeros(omni_img.shape[0:2], dtype=np.uint8)  # Black, single channel mask
            # Paint outer perimeter:
            cv2.circle(mask_top, center_top, int(r_outer_top), (255, 255, 255), -1, 8, 0)
            # Paint inner bound for top (as the union of the two inner/outer masks)
            if r_inner_top > 0:
                cv2.circle(mask_top, center_top, int(r_inner_top), (0, 0, 0), -1, 8, 0)
                if r_outer_bottom > 0:
                    cv2.circle(mask_top, center_bot, int(r_outer_bottom), (0, 0, 0), -1, 8, 0)
            self.top_model.mask = mask_top  # Save mask as property

            # BOTTOM:
            # Paint 2 black masks:
            mask_bottom_outer = np.zeros(omni_img.shape[0:2], dtype=np.uint8)  # Black, single channel mask
            mask_top_inner = np.zeros(omni_img.shape[0:2], dtype=np.uint8)  # Black, single channel mask
            cv2.circle(mask_bottom_outer, center_bot, int(r_outer_bottom), (255, 255, 255), -1, 8, 0)
            cv2.circle(mask_top_inner, center_top, int(r_inner_top), (255, 255, 255), -1, 8, 0)
            # Paint the outer bound mask for the bottom (as the intersection of the two inner/outer masks)
            mask_bottom = np.zeros(omni_img.shape)
            mask_bottom = cv2.bitwise_and(src1=mask_bottom_outer, src2=mask_top_inner, dst=mask_bottom, mask=None)
            # Paint the black inner bound for the bottom
            cv2.circle(mask_bottom, center_bot, int(r_inner_bottom), (0, 0, 0), -1, 8, 0)
            self.bot_model.mask = mask_bottom

        # Apply TOP mask
        masked_img_top = np.zeros(omni_img.shape)
        masked_img_top = cv2.bitwise_and(src1=omni_img, src2=omni_img, dst=masked_img_top, mask=self.top_model.mask)
        # Apply BOTTOM mask
        masked_img_bottom = np.zeros(omni_img.shape)
        masked_img_bottom = cv2.bitwise_and(src1=omni_img, src2=omni_img, dst=masked_img_bottom, mask=self.bot_model.mask)

        if color_RGB is not None:  # Paint the masked area other than black
            if color_RGB != self.mask_RGB_color:
                self.mask_RGB_color = color_RGB
                color_BGR = (color_RGB[2], color_RGB[1], color_RGB[0])
                self.mask_background_img = np.zeros_like(omni_img)
                self.mask_background_img[:, :, :] += np.array(color_BGR, dtype="uint8")  # Paint the B-G-R channels for OpenCV

            mask_top_inv = cv2.bitwise_not(src=self.top_model.mask)
            # Apply the background using the inverted mask
            masked_img_top = cv2.bitwise_and(src1=self.mask_background_img, src2=self.mask_background_img, dst=masked_img_top, mask=mask_top_inv)

            mask_bottom_inv = cv2.bitwise_not(src=self.bot_model.mask)
            # Apply the background using the inverted mask
            masked_img_bottom = cv2.bitwise_and(src1=self.mask_background_img, src2=self.mask_background_img, dst=masked_img_bottom, mask=mask_bottom_inv)

        self.construct_new_mask = False  # Clear mask construction flag

        # Show radial boundaries
        if view:
            win_name_top = "TOP Masked with ALL Radial Bounds"
            cv2.namedWindow(win_name_top, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name_top, masked_img_top)
            cv2.waitKey(1)
            win_name_bot = "BOTTOM Masked with ALL Radial Bounds"
            cv2.namedWindow(win_name_bot, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name_bot, masked_img_bottom)
            cv2.waitKey(1)

        return masked_img_top, masked_img_bottom


    def match_features_panoramic_top_bottom(self, min_rectified_disparity=1, max_horizontal_diff=1, show_matches=False):
        '''
        @param min_rectified_disparity: This disparity helps to check for point correspondences with positive disparity (which should be the case for rectified stereo)
        @param max_horizontal_diff: This pixel distance on the u-axis. Ideally, this shouldn't be an issuel for rectified panoramas, but it's useful to set when panoramas are not aligened.
        @return (matched_m_top_all, matched_kpts_top, matched_desc_top), (matched_m_bot_all, matched_kpts_bot, matched_desc_bot), random_RGB_colors
        '''
        # Optical flow doesn't work too well. For example, corner points were being track not exactly on the same column where they should.
        top_keypts_list, top_descriptors_list = self.top_model.detect_sparse_features_on_panorama(feature_detection_method=self.feature_matcher.feature_detection_method, num_of_features=self.feature_matcher.num_of_features, median_win_size=0, show=True)
        bot_keypts_list, bot_descriptors_list = self.bot_model.detect_sparse_features_on_panorama(feature_detection_method=self.feature_matcher.feature_detection_method, num_of_features=self.feature_matcher.num_of_features, median_win_size=1, show=True)
        # TRACK:
#         if len(bot_keypts) < self.feature_matcher.MIN_MATCH_COUNT or len(top_keypts) < self.feature_matcher.MIN_MATCH_COUNT:
#             return []
        matched_m_top_list = []
        matched_m_bot_list = []
        matched_kpts_top = []
        matched_kpts_bot = []
        matched_desc_top = []
        matched_desc_bot = []
        random_colors = []
        row_offset = self.top_model.panorama.rows
        if show_matches:
            from omnistereo.common_cv import rgb2bgr_color
            show_timeout_key = 0  # Paused visualization to begin with

        for top_keypts, top_descriptors, bot_keypts, bot_descriptors in zip(top_keypts_list, top_descriptors_list, bot_keypts_list, bot_descriptors_list):
            if (len(top_keypts) == 0) or (len(bot_keypts) == 0):
                continue

            matches = self.feature_matcher.match(query_descriptors=bot_descriptors, train_descriptors=top_descriptors)


            if len(matches) > 0:
                if self.top_model.panorama.panoramic_img is not None:
                    if self.top_model.panorama.panoramic_img.ndim == 3:
                        top_pano_gray = cv2.cvtColor(self.top_model.panorama.panoramic_img, cv2.COLOR_BGR2GRAY)
                    else:
                        top_pano_gray = self.top_model.panorama.panoramic_img.copy()
                top_pano_gray_vis = cv2.cvtColor(top_pano_gray, cv2.COLOR_GRAY2BGR)

                if self.bot_model.panorama.panoramic_img is not None:
                    if self.bot_model.panorama.panoramic_img.ndim == 3:
                        bot_pano_gray = cv2.cvtColor(self.bot_model.panorama.panoramic_img, cv2.COLOR_BGR2GRAY)
                    else:
                        bot_pano_gray = self.bot_model.panorama.panoramic_img.copy()
                bot_pano_gray_vis = cv2.cvtColor(bot_pano_gray, cv2.COLOR_GRAY2BGR)

                # matches_img = np.vstack((top_pano_gray_vis, bot_pano_gray_vis))
                # matches_img = None
                # matches_img = cv2.drawMatches(top_pano_gray_vis, top_keypts, bot_pano_gray_vis, bot_keypts, matches[:MIN_MATCH_COUNT], matches_img, flags=2)
                from random import randint
                num_of_good_matches = int(self.feature_matcher.percentage_good_matches * len(matches))

                if num_of_good_matches > 0:
                    random_colors_current = []
                    matched_kpts_top_current = []
                    matched_kpts_bot_current = []

                    for m in matches[:num_of_good_matches]:
                        top_kpt = top_keypts[m.trainIdx]
                        bot_kpt = bot_keypts[m.queryIdx]
                        # Filter matches based on vertical disparity (Recall v-coord is pt[1])
                        if (top_kpt.pt[1] - bot_kpt.pt[1]) < min_rectified_disparity:
                            continue  # Try next match
                        # Also, check that matches are vertically aligned on the u-axis:
                        if np.abs(top_kpt.pt[0] - bot_kpt.pt[0]) > max_horizontal_diff:
                            continue  # Try next match

                        top_desc = top_descriptors[m.trainIdx]
                        bot_desc = bot_descriptors[m.queryIdx]
                        matched_kpts_top_current.append(top_kpt)
                        matched_kpts_bot_current.append(bot_kpt)
                        matched_kpts_top.append(top_kpt)
                        matched_kpts_bot.append(bot_kpt)
                        matched_desc_top.append(top_desc)
                        matched_desc_bot.append(bot_desc)
                        random_color_RGB = (randint(0, 255), randint(0, 255), randint(0, 255))
                        random_colors.append(random_color_RGB)
                        random_colors_current.append(random_color_RGB)
                        if show_matches:
                            random_color_BGR = rgb2bgr_color(random_color_RGB)
                            top_pano_gray_vis = cv2.drawKeypoints(top_pano_gray_vis, [top_kpt], outImage=top_pano_gray_vis, color=random_color_BGR)
                            bot_pano_gray_vis = cv2.drawKeypoints(bot_pano_gray_vis, [bot_kpt], outImage=bot_pano_gray_vis, color=random_color_BGR)
                            matches_img = np.vstack((top_pano_gray_vis, bot_pano_gray_vis))
                            # Enable to draw matches one by one:
                            #===================================================================
                            # top_pt = (int(top_kpt.pt[0]), int(top_kpt.pt[1]))  # Recall, pt is given as (u,v)
                            # bot_pt = (int(bot_kpt.pt[0]), int(bot_kpt.pt[1] + row_offset))
                            # matches_img = cv2.line(matches_img, top_pt, bot_pt, color=random_color_RGB, thickness=1, lineType=cv2.LINE_8)
                            # cv2.imshow('Matches', matches_img)
                            # cv2.waitKey(0)
                            #===================================================================
                    # Update number of good matches based on disparity filtering above
                    num_of_good_matches = len(matched_kpts_top_current)

                    # Draw top and bottom in a single image
                    if show_matches:
                        matches_img = np.vstack((top_pano_gray_vis, bot_pano_gray_vis))
                    # Draw connecting lines for matches and compose pixel matrices
                    matched_m_top = np.ones((1, num_of_good_matches, 3))
                    matched_m_bot = np.ones((1, num_of_good_matches, 3))
                    idx = 0
                    # TODO: Only append those points far enough from the last point inserted.
                    for top_kpt, bot_kpt, random_RGB_color in zip(matched_kpts_top_current, matched_kpts_bot_current, random_colors_current):
                        top_pt = (int(top_kpt.pt[0]), int(top_kpt.pt[1]))  # Recall, pt is given as (u,v)
                        bot_pt = (int(bot_kpt.pt[0]), int(bot_kpt.pt[1] + row_offset))
                        if show_matches:
                            matches_img = cv2.line(matches_img, top_pt, bot_pt, color=rgb2bgr_color(random_RGB_color), thickness=1, lineType=cv2.LINE_8)
                            cv2.imshow('Matches', matches_img)
                            ch_pressed_waitkey = cv2.waitKey(show_timeout_key)
                            if ch_pressed_waitkey == 27:  # Pressing the Escape key stops visualization of each match
                                show_timeout_key = 1  # Not longer wait for key presses
                        # Getting the floating point coordinates instead of int, so we can use precision elevation without LUTs
                        matched_m_top[0, idx, 0] = top_kpt.pt[0]
                        matched_m_top[0, idx, 1] = top_kpt.pt[1]
                        matched_m_bot[0, idx, 0] = bot_kpt.pt[0]
                        matched_m_bot[0, idx, 1] = bot_kpt.pt[1]
                        idx += 1  # incremente index

                    matched_m_top_list.append(matched_m_top)
                    matched_m_bot_list.append(matched_m_bot)

            # cv2.waitKey(1)
            # top_pts = cv2.goodFeaturesToTrack(image=top_pano_gray, maxCorners=5, qualityLevel=0.01, minDistance=100)
            # PARAMETERS:
            #    qualityLevel: Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue (see cornerMinEigenVal() ) or the Harris function response (see cornerHarris() ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected.
            #    minDistance: Minimum possible Euclidean distance between the returned corners.

#         from omnistereo.common_cv import draw_flow, draw_keypoints
#         top_pts_color = (255, 0, 0)  # red because (R,G,B)
#         draw_points(top_pano_gray_vis, top_pts[..., :2].reshape(-1, 2), color=top_pts_color, thickness=2)
#         draw_keypoints(top_pano_gray_vis, top_keypts, color=top_pts_color)
#         cv2.imshow('Points (TOP)', top_pano_gray_vis)


#         bot_pts = np.copy(top_pts)
#         bot_pts, status, err = cv2.calcOpticalFlowPyrLK(prevImg=top_pano_gray, nextImg=bot_pano_gray, prevPts=top_pts, nextPts=bot_pts)
#         draw_points(bot_pano_gray_vis, bot_pts[..., :2].reshape(-1, 2), color=top_pts_color, thickness=2)
#         draw_keypoints(bot_pano_gray_vis, bot_keypts, color=top_pts_color)
#         cv2.imshow('Points (BOTTOM)', bot_pano_gray_vis)
#         cv2.imshow('flow', draw_flow(bot_pano_gray, flow))
        cv2.waitKey(1)
        # FIXME: something is wrong with concatenation.
        matched_m_top_all = np.concatenate(matched_m_top_list, axis=1)
        matched_m_bot_all = np.concatenate(matched_m_bot_list, axis=1)
        return (matched_m_top_all, matched_kpts_top, matched_desc_top), (matched_m_bot_all, matched_kpts_bot, matched_desc_bot), random_colors

    def set_panoramas(self, pano_top, pano_bottom):
        self.top_model.panorama = pano_top
        self.bot_model.panorama = pano_bottom

    def set_current_omni_image(self, img, pano_width_in_pixels=1200, generate_panoramas=False, idx=-1, view=False, apply_pano_mask=True, mask_RGB=None):
        self.current_omni_img = img
        if hasattr(self, "theoretical_model"):
            if self.theoretical_model is not None:
                self.theoretical_model.set_current_omni_image(img=img, pano_width_in_pixels=pano_width_in_pixels, generate_panoramas=generate_panoramas, idx=idx, view=False, apply_pano_mask=apply_pano_mask, mask_RGB=mask_RGB)
        if apply_pano_mask:
            img_top, img_bot = self.get_fully_masked_images(omni_img=img, view=view, color_RGB=mask_RGB)
        else:
            img_top = img
            img_bot = img
        self.top_model.set_omni_image(img_top, pano_width_in_pixels=pano_width_in_pixels, generate_panorama=generate_panoramas, idx=idx, view=view)
        self.bot_model.set_omni_image(img_bot, pano_width_in_pixels=pano_width_in_pixels, generate_panorama=generate_panoramas, idx=idx, view=view)

    def get_system_vFOV(self, **kwargs):
        '''
        Computes the so-called system vFOV angle out of the total view covered by  the two mirrors' vFOVs

        @return: the total system's vertical field of view in radians
        '''
        theta1_max = kwargs.get("theta1_max", self.top_model.highest_elevation_angle)
        theta1_min = kwargs.get("theta1_min", self.top_model.lowest_elevation_angle)
        theta2_max = kwargs.get("theta2_max", self.bot_model.highest_elevation_angle)
        theta2_min = kwargs.get("theta2_min", self.bot_model.lowest_elevation_angle)

        max_elevation = max(theta1_max, theta2_max)
        min_elevation = min(theta1_min, theta2_min)

        alpha_sys = max_elevation - min_elevation

        return alpha_sys

    def get_imaging_ratio(self, print_info=False):
        _, _, m1_common_highest = self.top_model.get_pixel_from_direction_angles(0, self.common_highest_elevation_angle)
        _, _, m1_common_lowest = self.top_model.get_pixel_from_direction_angles(0, self.common_lowest_elevation_angle)
        _, _, m2_common_highest = self.bot_model.get_pixel_from_direction_angles(0, self.common_highest_elevation_angle)
        _, _, m2_common_lowest = self.bot_model.get_pixel_from_direction_angles(0, self.common_lowest_elevation_angle)
        h1 = np.linalg.norm(m1_common_highest - m1_common_lowest)  # FIXME:use the norm
        h2 = np.linalg.norm(m2_common_highest - m2_common_lowest)
        img_ratio = h1 / h2
        if print_info:
            print("Stereo ROI's imaging ratio = %f" % (img_ratio))
            print("using h1 = %f,  and h2=%f" % (h1, h2))

        return img_ratio


    def set_common_vFOV(self, **kwargs):
        '''
        Computes the so-called common vFOV angle out of the overlapping region of the two mirrors' vFOVs
        @note: We are assuming the bottom model's maximum elevation angle is always greater than or equal to the top's maximum elevation angle.

        @return: the common vertical field of view in radians
        '''
        verbose = kwargs.get("verbose", False)
        theta1_max = kwargs.get("theta1_max", self.top_model.highest_elevation_angle)
        theta1_min = kwargs.get("theta1_min", self.top_model.lowest_elevation_angle)
        theta2_max = kwargs.get("theta2_max", self.bot_model.highest_elevation_angle)
        theta2_min = kwargs.get("theta2_min", self.bot_model.lowest_elevation_angle)

        self.common_lowest_elevation_angle = max(theta1_min, theta2_min)
        self.common_highest_elevation_angle = min(theta1_max, theta2_max)

        # Generalized approach:
        self.common_vFOV = self.common_highest_elevation_angle - self.common_lowest_elevation_angle

        if verbose:
            print("Common vFOV: %.2f degrees" % (np.rad2deg(self.common_vFOV)))
            print("\tCommon highest elevation: %.2f degrees" % (np.rad2deg(self.common_highest_elevation_angle)))
            print("\tCommon highest elevation: %.2f degrees" % (np.rad2deg(self.common_lowest_elevation_angle)))
            print("\tusing (min,max) elevations: Top(%.2f,%.2f) degrees, Bottom(%.2f,%.2f) degrees" % (np.rad2deg(theta1_min), np.rad2deg(theta1_max), np.rad2deg(theta2_min), np.rad2deg(theta2_max)))

        return self.common_vFOV

#     FIXME: redundant definition?
#        def calibrate(self):
#         '''
#         Performs the omnidirectional stereo calibration parameters.
#         @note: Only doing extrinsic optimization at the moment.
#         '''
#         self.calibrator.calibrate()

    def print_omnistereo_info(self):
        self.top_model.print_params()
        self.bot_model.print_params()
        print("Baseline = %0.4f %s" % (self.baseline, self.units))
        print("Common vFOV %0.4f degrees" % (np.rad2deg(self.common_vFOV)))
        self.get_imaging_ratio(print_info=True)

    def get_triangulated_points_from_pixel_disp(self, disparity=1, m1=None):
        '''
        Use this method only for plotting because it's unrealistic to obtain disparities on the omnidirectional images

        @param disparity: The pixel disparity (on the omnidirectional image) to use while computing the depth
        @param m1: A ndarray of specific pixels to get the depth for
        @param m2: A ndarray of corresponding pixels to triangulate with
        @return: the ndarray of \f$ \rho_w$\f for all ray intersection with \f$\Delta m$\f pixel disparity
        @note: This is true way of computing max depth resolution from pixel disparity
        '''
        if m1 is None:
            azim1, elev1 = self.top_model.get_all_direction_angles_per_pixel_radially()
        else:
            azim1, elev1 = self.top_model.get_direction_angles_from_pixel(m1)

        pixels2_u, pixels2_v, _ = self.bot_model.get_pixel_from_direction_angles(azim1, elev1)
        px2_u_with_disp = pixels2_u - disparity  # np.floor(pixels2_u) - disparity

        if m1 is None:
            pixels2 = np.dstack((px2_u_with_disp, pixels2_v))
        else:
            pixels2_v_to_use = np.repeat(pixels2_v, px2_u_with_disp.size).reshape(px2_u_with_disp.shape)
            pixels2 = np.dstack((px2_u_with_disp, pixels2_v_to_use))

        azim2, elev2 = self.bot_model.get_direction_angles_from_pixel(pixels2)
        triangulated_points = self.get_triangulated_point_wrt_Oc(elev1, elev2, azim2)

        return triangulated_points

    def get_triangulated_point_from_pixels(self, m1, m2, use_midpoint_triangulation=False):
        # Using the common perpendicular midpoint method (vectorized)
        if use_midpoint_triangulation:
            direction_vectors_top = self.top_model.get_direction_vector_from_focus(m1)
            direction_vectors_bottom = self.bot_model.get_direction_vector_from_focus(m2)
            rows = direction_vectors_top.shape[0]
            cols = direction_vectors_top.shape[1]
            triangulated_points = np.ndarray((rows, cols, 3))
            for row in range(rows):
                for col in range(cols):
                    mid_Pw, _, _, _ = self.get_triangulated_midpoint(direction_vectors_top[row, col], direction_vectors_bottom[row, col])
                    triangulated_points[row, col] = mid_Pw
        else:
            az1, el1 = self.top_model.get_direction_angles_from_pixel(m1)
            az2, el2 = self.bot_model.get_direction_angles_from_pixel(m2)
#             triangulated_points = self.get_triangulated_point_wrt_Oc(el1, el2, (az1 + az2) / 2.0)
            triangulated_points = self.get_triangulated_point_wrt_Oc(el1, el2, (az1 + az2) / 2.0)
#             triangulated_points = np.where(pano_disparities[..., np.newaxis] > 0, triangulated_points_original[0], -1.0 * triangulated_points_original[0])
#             triangulated_points = triangulated_points_original[0, pano_disparities > 0]

        return triangulated_points

    def get_confidence_weight_from_pixel_RMSE_stereo(self, img_points_top, img_points_bot, obj_pts_homo, T_G_wrt_C):
        '''
        We define a confidence weight as the inverse of the pixel projection RMSE

        @param img_points_top: The corresponding points on the top image
        @param img_points_bot: The corresponding points on the bottom image
        @param obj_pts_homo: The coordinates of the corresponding points with respect to the object's own frame [G].
        @param T_G_wrt_C: The transform matrix of [G] wrt to [C].
        '''
        from omnistereo.common_tools import rms
        all_pixel_errors_top = self.top_model.get_obj_pts_proj_error(img_points_top, obj_pts_homo, T_G_wrt_C)
        all_pixel_errors_bot = self.bot_model.get_obj_pts_proj_error(img_points_bot, obj_pts_homo, T_G_wrt_C)
        rmse = rms([all_pixel_errors_top] + [all_pixel_errors_bot])
        weight = 1.0 / rmse
        return weight


    def filter_panoramic_points_due_to_reprojection_error(self, m_top, m_bot, xyz_points_wrt_C, pixel_error_threshold=1):
        '''
        Filter outlier feature correspondences by projecting 3D points and measuring pixel norm to matched_m_top and matched_m_bot, so only pixels under a certain distance threshold remain.

        @param m_top: Pixel coordinates on its panoramic image (for the top model)
        @param m_bot: Pixel coordinates on its panoramic image (for the bot model)
        @param xyz_points_wrt_C: The coordinateds of the estimated points wrt to the common frame [C]
        @param pixel_error_threshold: By default is 1 pixel of the pixel error computed of out the norm betwen the detected m pixel points and the reprojected pixels from the XYZ points
        @return: a Boolean list related to the validity of the indices of good points from set
        '''

        _, _, m_top_projected = self.top_model.get_pixel_from_3D_point_wrt_C(xyz_points_wrt_C)
        _, m_pano_top = self.top_model.panorama.get_panoramic_pixel_coords_from_omni_pixel(m_top_projected)
        p2p_distances_top = np.linalg.norm(m_pano_top - m_top, axis=-1)
        # WISH: Ignoring for now the Warning generated when comparing np.nan's and threshold
        valid_top = np.where(p2p_distances_top < pixel_error_threshold, True, False)

        _, _, m_bot_projected = self.bot_model.get_pixel_from_3D_point_wrt_C(xyz_points_wrt_C)
        _, m_pano_bot = self.bot_model.panorama.get_panoramic_pixel_coords_from_omni_pixel(m_bot_projected)
        p2p_distances_bot = np.linalg.norm(m_pano_bot - m_bot, axis=-1)
        # WISH: Ignoring for now the Warning generated when comparing np.nan's and threshold
        valid_bot = np.where(p2p_distances_bot < pixel_error_threshold, True, False)

        valid_indices = np.logical_and(valid_top, valid_bot)

        return valid_indices

    def filter_panoramic_points_due_to_range(self, xyz_points_wrt_C, min_3D_range=0, max_3D_range=0.):
        '''
        Filter outlier feature correspondences by projecting 3D points under a certain range threshold remain.

        @param xyz_points_wrt_C: The coordinateds of the estimated points wrt to the common frame [C]
        @param min_3D_range: The minimum euclidean norm to be considered a valid point. 0 by default
        @param max_3D_range: The maximum euclidean norm to be considered a valid point. If 0 (defaul), this filtering is bypassed.

        @return: a Boolean list related to the validity of the indices of good points from set
        '''
        valid_indices = np.ones(shape=(xyz_points_wrt_C.shape[:-1]), dtype="bool")
        if min_3D_range > 0 or max_3D_range > 0:
            norm_of_3D_points = np.linalg.norm(xyz_points_wrt_C, axis=-1)

            if min_3D_range > 0:
                valid_min_ranges = np.where(norm_of_3D_points >= min_3D_range, True, False)
                valid_indices = np.logical_and(valid_indices, valid_min_ranges)

            if max_3D_range > 0:
                valid_max_ranges = np.where(norm_of_3D_points <= max_3D_range, True, False)
                valid_indices = np.logical_and(valid_indices, valid_max_ranges)

        return valid_indices

    def get_triangulated_point_from_direction_angles(self, dir_angs_top, dir_angs_bot, use_midpoint_triangulation=False):
        '''
        @return: the homogeneous coordinates of the triangulated points
        '''
        az1, el1 = dir_angs_top
        az2, el2 = dir_angs_bot
        # Using the common perpendicular midpoint method (vectorized)
        if use_midpoint_triangulation:
            # We need to extract direction vectors from the given direction angles:
            # Assuming unit cilinder
            x1 = np.cos(az1)
            y1 = np.sin(az1)
            z1 = np.tan(el1)
            direction_vectors_top = np.dstack((x1, y1, z1))
            x2 = np.cos(az2)
            y2 = np.sin(az2)
            z2 = np.tan(el2)
            direction_vectors_bottom = np.dstack((x2, y2, z2))
            #===================================================================
            # rows = direction_vectors_top.shape[0]
            # cols = direction_vectors_top.shape[1]
            # triangulated_points = np.ndarray((rows, cols, 3))
            # # TODO: Don't use a loop!. It should be implemented using the common perpendicular midpoint method (vectorized)
            # for row in range(rows):
            #     for col in range(cols):
            #         mid_Pw, _, _, _ = self.get_triangulated_midpoint(direction_vectors_top[row, col], direction_vectors_bottom[row, col])
            #         triangulated_points[row, col] = mid_Pw
            #===================================================================
            #===================================================================
            triangulated_points, _, _, _ = self.get_triangulated_midpoint(direction_vectors_top, direction_vectors_bottom)
            #===================================================================
        else:
#             triangulated_points = self.get_triangulated_point_wrt_Oc(el1, el2, (az1 + az2) / 2.0)
            triangulated_points = self.get_triangulated_point_wrt_Oc(el1, el2, (az1 + az2) / 2.0)
#             triangulated_points = np.where(pano_disparities[..., np.newaxis] > 0, triangulated_points_original[0], -1.0 * triangulated_points_original[0])
#             triangulated_points = triangulated_points_original[0, pano_disparities > 0]

        # Append 1's for homogeneous coordinates
        ones_matrix = np.ones(triangulated_points.shape[:-1])
        points_3D_homo = np.dstack((triangulated_points, ones_matrix))  # Put back the ones for the homogeneous coordinates
        return points_3D_homo

    def get_stereo_ROI_nearest_vertices(self):
        '''
        @return: The 3 near bounding points (namely, \f${P}_{ns_{low}},{P}_{ns_{mid}}, {P}_{ns_{high}}$\f) for the stereo ROI
        '''
        mirror1 = self.top_model
        mirror2 = self.bot_model

        Pns_low = self.get_triangulated_point_wrt_Oc(mirror1.lowest_elevation_angle, mirror2.lowest_elevation_angle, 0)
        Pns_mid = self.get_triangulated_point_wrt_Oc(mirror1.lowest_elevation_angle, mirror2.highest_elevation_angle, 0)
        Pns_high = self.get_triangulated_point_wrt_Oc(mirror1.highest_elevation_angle, mirror2.highest_elevation_angle, 0)

        return Pns_low, Pns_mid, Pns_high

    def get_far_phony_depth_from_panoramas(self, only_valid_points=True):
        '''
        NOT GOOD: Just dealing with rays that don't converge in the front (but in the back).
        @return: the ndarray of \f$ \rho_w$\f for all ray intersection with only 1 pixel disparity (almost parallel rays, so they are far apart)
        @note: This is not an ideal way to compute the possible way for true pixel disparity (since that should only happen within the warped omnidirectional images)
        '''
        elevations_bot = self.bot_model.panorama.get_all_elevations(validate=True)[..., :-1]  # Pick from index 0 but not the last
        elevations_top = self.top_model.panorama.get_all_elevations(validate=True)[..., 1:]  # Pick from index 1 to the last
        azimuths_null = np.zeros_like(elevations_top)
        triangulated_points = self.get_triangulated_point_wrt_Oc(elevations_top, elevations_bot, azimuths_null)

        if only_valid_points:
            return triangulated_points[0, self.top_model.panorama.get_row_limits()[0]:self.bot_model.panorama.get_row_limits()[1]]
        else:
            return triangulated_points


    def get_depth_map_from_panoramas(self, method="sgbm", use_cropped_panoramas=False, rows_roi=[], cols_roi=[], show=True, load_stereo_tuner_from_pickle=False, stereo_tuner_filename="stereo_tuner.pkl", tune_live=False):
        '''
        @param tune_live: Allows continuous frames to come (as for movies) so tuning can be perform live. When False, the tuning is attempted (until Esc is pressed)
        '''

        import cv2
        from omnistereo.common_cv import StereoMatchTuner

        # Rotate images counter-clockwise to do horizontal stereo
        if method == "bm":
            # Top panorama is the reference image (or thought as left image)
            if self.top_model.panorama.panoramic_img.ndim == 3:
                left8 = cv2.cvtColor(self.top_model.panorama.panoramic_img, cv2.COLOR_BGR2GRAY)
            else:
                left8 = self.top_model.panorama.panoramic_img.copy()

            # Bottom panorama is the target image (or thought as left image)
            if self.bot_model.panorama.panoramic_img.ndim == 3:
                right8 = cv2.cvtColor(self.bot_model.panorama.panoramic_img, cv2.COLOR_BGR2GRAY)
            else:
                right8 = self.bot_model.panorama.panoramic_img.copy()
        else:
            left8 = self.top_model.panorama.panoramic_img.copy()
            right8 = self.bot_model.panorama.panoramic_img.copy()

        # Initialize zeroed disparity map
        pano_rows = int(self.top_model.panorama.rows)
        pano_cols = int(self.top_model.panorama.cols)

        # NOTE: we choose either panorama due to rectification assumption as row-to-elevation mapping used in the following function
        row_common_highest = self.top_model.panorama.get_panorama_row_from_elevation(self.common_highest_elevation_angle)
        row_common_lowest = self.top_model.panorama.get_panorama_row_from_elevation(self.common_lowest_elevation_angle)

        if len(cols_roi) == 1:
            # Ambiguous:
            left = left8[:, cols_roi[0]:]
            right = right8[:, cols_roi[0]:]
        elif len(cols_roi) == 2:
            left = left8[:, cols_roi[0]:cols_roi[1]]
            right = right8[:, cols_roi[0]:cols_roi[1]]
        else:
            left = left8
            right = right8

        if show:
            self.draw_elevations_on_panoramas(left, right)  # Uncropped

        if use_cropped_panoramas:
            # FIXME: crop it without destroying the left and right
            # mask = common_cv.create_rectangular_mask(img_input=right, points=[(0, row_common_highest), (self.top_model.panorama.width - 1, row_common_lowest)], preview=True)
            right = right[row_common_highest:row_common_lowest + 1]  # Recall that range doesn't include the last index (so we want to include it with +1)
            left = left[row_common_highest:row_common_lowest + 1]

        disp_img_win = 'Panoramic Disparity Map'
        # NOTE: the disparity map is linked to the left/top image so it is the reference for the correspondences
        #       Thus, the coordinates of the correspondence on the right/bottom image is found such that m_right = (u_left, v_left - disp[u,v])

        self.disparity_map = np.zeros(left8.shape[0:2], dtype="float64")
        self.disparity_img = np.zeros(left8.shape[0:2], dtype="uint8")


        if load_stereo_tuner_from_pickle:
            from omnistereo.common_tools import load_obj_from_pickle
            stereo_matcher_tuner = load_obj_from_pickle(stereo_tuner_filename)
            stereo_matcher_tuner.reset_images(left_img=left, right_img=right, rotate_images=True, disp_first_valid_row=row_common_highest, disp_last_valid_row=row_common_lowest)
        else:
            stereo_matcher_tuner = StereoMatchTuner(left_img=left, right_img=right, rotate_images=True, method=method, win_name=disp_img_win, disp_first_valid_row=row_common_highest, disp_last_valid_row=row_common_lowest)

        # (my TRICK) Filter out of bound values by applying panoramic mask to the disparity image and depth map using radial bounds:
        white_blank_img = np.zeros_like(self.current_omni_img) + 255
        top_circ_mask, bot_circular_mask = self.get_fully_masked_images(omni_img=white_blank_img, view=False, color_RGB=None)
        # cv2.imshow("OMNI MASK (Reference)", top_circ_mask)
        # Recall, we need a single channel mask
        pano_mask = self.top_model.panorama.get_panoramic_image(top_circ_mask[..., 0], set_own=False)  # Using the left/top image as reference
        # cv2.imshow("PANORAMA MASK (Reference)", pano_mask)
        disp_map, disp_img = stereo_matcher_tuner.start_tuning(win_name=disp_img_win, save_file_name=stereo_tuner_filename, tune_live=tune_live, pano_mask=pano_mask)

        # Merge disparity results from plausible ROI maps
        if len(rows_roi) == 0:
            if len(cols_roi) == 1:
                # Ambiguous:
                self.disparity_map[:, cols_roi[0]:] = disp_map
                self.disparity_img[:, cols_roi[0]:] = disp_img
            elif len(cols_roi) == 2:
                self.disparity_map[:, cols_roi[0]:cols_roi[1]] = disp_map
                self.disparity_img[:, cols_roi[0]:cols_roi[1]] = disp_img
            else:
                self.disparity_map = disp_map
                self.disparity_img = disp_img
        elif len(rows_roi) == 1:  # Max row case
            if len(cols_roi) == 1:
                # Ambiguous:
                self.disparity_map[:rows_roi[0], cols_roi[0]:] = disp_map[:rows_roi[0]]
                self.disparity_img[:rows_roi[0], cols_roi[0]:] = disp_img[:rows_roi[0]]
            elif len(cols_roi) == 2:
                self.disparity_map[:rows_roi[0], cols_roi[0]:cols_roi[1]] = disp_map[:rows_roi[0]]
                self.disparity_img[:rows_roi[0], cols_roi[0]:cols_roi[1]] = disp_img[:rows_roi[0]]
            else:
                self.disparity_map[:rows_roi[0]] = disp_map[:rows_roi[0]]
                self.disparity_img[:rows_roi[0]] = disp_img[:rows_roi[0]]

