# -*- coding: utf-8 -*-
# covariance_ellipsoid.py

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

import matplotlib.pyplot as plt

import numpy as np

def draw_eigen_vectors(sigma, ax):
    # Visualize eigen vectors
    # The normalized (unit "length") eigenvectors, such that the
    # column ``eig_vects[:,i]`` is the eigenvector corresponding to the eigenvalue ``eig_vals[i]``.
    eig_vals, eig_vects = np.linalg.eig(sigma)
    x_eig_vect_plus = mu + eig_vects[:, 0] * eig_vals[0]
    x_eig_vect_minus = mu - eig_vects[:, 0] * eig_vals[0]
    ax.plot3D([x_eig_vect_minus[0], x_eig_vect_plus[0]], [x_eig_vect_minus[1], x_eig_vect_plus[1]], [x_eig_vect_minus[2], x_eig_vect_plus[2]], color="red")  # , linestyle='-', linewidth=1.5, alpha=1.0)
    y_eig_vect_plus = mu + eig_vects[:, 1] * eig_vals[1]
    y_eig_vect_minus = mu - eig_vects[:, 1] * eig_vals[1]
    ax.plot3D([y_eig_vect_minus[0], y_eig_vect_plus[0]], [y_eig_vect_minus[1], y_eig_vect_plus[1]], [y_eig_vect_minus[2], y_eig_vect_plus[2]], color="green")  # , linestyle='-', linewidth=1.5, alpha=1.0)
    z_eig_vect_plus = mu + eig_vects[:, 2] * eig_vals[2]
    z_eig_vect_minus = mu - eig_vects[:, 2] * eig_vals[2]
    ax.plot3D([z_eig_vect_minus[0], z_eig_vect_plus[0]], [z_eig_vect_minus[1], z_eig_vect_plus[1]], [z_eig_vect_minus[2], z_eig_vect_plus[2]], color="blue")  # , linestyle='-', linewidth=1.5, alpha=1.0)

    return ax

def draw_error_ellipsoid(mu, covariance_matrix, stdev=1, color="blue", ax=None):
    """
    @brief Plot the error (uncertainty) ellipsoid using a 3D covariance matrix for a given standard deviation
    @param mu: The mean vector
    @param covariance_matrix: the 3x3 covariance matrix
    @param stdev: The desire standard deviation value to draw the uncertainty ellipsoid for. Default is 1.
    @param color: The color of the ellipsoid (Optional)
    @param ax: The figure axis to use (optional)

    @return: The axis in the figure where that has been drawn
    """

    if ax == None:
        from mpl_toolkits.mplot3d import Axes3D  # To be run from the command line
        # from mplot3d import axes3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', aspect="equal")
        # ax.set_aspect("equal")
        fig.tight_layout()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    # Step 1: make a unit n-sphere
    u = np.linspace(0.0, 2.0 * np.pi, 20)
    v = np.linspace(0.0, np.pi, 20)
    x_sph = np.outer(np.cos(u), np.sin(v))
    y_sph = np.outer(np.sin(u), np.sin(v))
    z_sph = np.outer(np.ones_like(u), np.cos(v))
    X_sphere = np.dstack((x_sph, y_sph, z_sph))
    # Step 2:  apply the following linear transformation to get the points of your ellipsoid (Y):
    C = np.linalg.cholesky(covariance_matrix)

    ellipsoid_pts = mu + stdev * np.dot(X_sphere, C)
    x = ellipsoid_pts[..., 0]
    y = ellipsoid_pts[..., 1]
    z = ellipsoid_pts[..., 2]

    # plot
    ax.plot_wireframe(x, y, z, color=color, alpha=0.2)
    # ax = draw_eigen_vectors(covariance_matrix, ax)

    # Adjustment of the axes, so that they all have the same span:
    max_radius = np.max(ellipsoid_pts)

    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))


    return ax

def draw_error_ellipsoid_visvis(mu, covariance_matrix, stdev=1, z_offset=0, color_ellipsoid="g", pt_size=5):
    """
    @brief Plot the error (uncertainty) ellipsoid using a 3D covariance matrix for a given standard deviation
    @param mu: The mean vector
    @param covariance_matrix: the 3x3 covariance matrix
    @param stdev: The desire standard deviation value to draw the uncertainty ellipsoid for. Default is 1.
    """
    import visvis as vv

    # Step 1: make a unit n-sphere
    u = np.linspace(0.0, 2.0 * np.pi, 30)
    v = np.linspace(0.0, np.pi, 30)
    x_sph = np.outer(np.cos(u), np.sin(v))
    y_sph = np.outer(np.sin(u), np.sin(v))
    z_sph = np.outer(np.ones_like(u), np.cos(v))
    X_sphere = np.dstack((x_sph, y_sph, z_sph))
    # Step 2:  apply the following linear transformation to get the points of your ellipsoid (Y):
    C = np.linalg.cholesky(covariance_matrix)

    ellipsoid_pts = mu + stdev * np.dot(X_sphere, C)

    x = ellipsoid_pts[..., 0]
    y = ellipsoid_pts[..., 1]
    z = ellipsoid_pts[..., 2] + z_offset

    # plot
    ellipsoid_surf = vv.surf(x, y, z)
 # Get axes


#     ellipsoid_surf = vv.grid(x, y, z)
    ellipsoid_surf.faceShading = "smooth"
    ellipsoid_surf.faceColor = color_ellipsoid
#     ellipsoid_surf.edgeShading = "plain"
#     ellipsoid_surf.edgeColor = color_ellipsoid
    ellipsoid_surf.diffuse = 0.9
    ellipsoid_surf.specular = 0.9

    mu_pt = vv.Point(mu[0], mu[1], mu[2] + z_offset)
    pt_mu = vv.plot(mu_pt, ms='.', mc="k", mw=pt_size, ls='', mew=0, axesAdjust=True)


def draw_demo_ellipsoid(mu, covariance_matrix, stdev=1, z_offset=0, color_ellipsoid="g", pt_size=5, show_grid_box=True):
    import visvis as vv
    from omnistereo import common_plot

    # Prepare
    a = vv.gca()
    common_plot.draw_axes_visvis(10, 1, "--", z_offset)

    draw_error_ellipsoid_visvis(mu, covariance_matrix, stdev=stdev, z_offset=z_offset, color_ellipsoid=color_ellipsoid, pt_size=pt_size)

#     sphere = vv.solidSphere(mu_pt, scaling=0.5)
#     sphere.faceColor = color_ellipsoid
    # Make the sphere dull
#     sphere.specular = 0
#     sphere.diffuse = 0.4

    # ax = draw_eigen_vectors(covariance_matrix, ax)
    # Modifying lights
    # The other lights are off by default and are positioned at the origin
#     light0 = a.lights[0]
#     light0.On()
#     light0.ambient = 0.0  # 0.0 is default for other lights
#     light0.position = (100, 0, 100, 1)
#     light1 = a.lights[1]
#     light1.On()
#     light1.ambient = 0.0  # 0.0 is default for other lights
# #         light1.color = (1, 0, 0)  # this light is red
#     light1.position = (50, 50, 100, 0)
    # If the fourth element is a 1, the light
    # has a position, if it is a 0, it represents a direction (i.o.w. the
    # light is a directional light, like the sun).

    a.axis.visible = show_grid_box
    a.axis.showGrid = show_grid_box
#     a.axis.showGridX = False
#     a.axis.ShowBorder = False

    # Create time and enter main loop
    a.eventKeyDown.Bind(common_plot.OnKey)

if __name__ == '__main__':
    Cov = np.array([[1, 0.5, 0.3], [0.5, 2, 0], [0.3, 0, 3]])
    # NOTE:it works for negative values, but it's not physically meaningful
    # Cov = np.array([[4, 0, 0], [0, 1, 0], [0, 0, 2]])
    # mu = np.array([[1], [2], [3]])  # The center of the ellipsoid covariance
#     mu = np.array([1, 2, 3])
    mu = np.array([0, 0, 0])

#     ax = draw_error_ellipsoid(mu, Cov, 1, "blue")
    # ax = draw_error_ellipsoid(mu, Cov, 2, "red", ax)
    # ax = draw_error_ellipsoid(mu, Cov, 3, "green", ax)
#     plt.show()
    import visvis as vv
    app = vv.use()
    # Set colors
    color_rgb = [0., 1., 0.]
    color_alpha = [0.5]
    color_ellipsoid = tuple(color_rgb + color_alpha)
    draw_demo_ellipsoid(mu, Cov, 1, z_offset=0, color_ellipsoid=color_ellipsoid, pt_size=5)
    app.Run()



