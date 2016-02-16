# -*- coding: utf-8 -*-
# cata_hyper_symbolic.py

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

from omnistereo import common_tools
import sympy as sym
from sympy import symbols, sqrt, Eq, solve, Abs, Matrix, ImmutableMatrix, eye, diag, lambdify
# import numpy as np
# import matplotlib.pyplot as plt
# from sympy import pprint as pp

class HyperCataStereoSymbolic(object):

    def __init__(self):
        self.set_camera_parameters()
        self.compute_vects_from_pixel()
        self.compute_triangulated_point()

    def set_camera_parameters(self):
        # Camera parameters:
        self.fu, self.fv, self.s = symbols("f_u, f_v, s", real=True, positve=True)
        self.uc, self.vc = symbols("u_c, v_c", real=True, nonnegative=True)
        # 3 by 3 inverse camera matrix
        # Kc_inv = MatrixSymbol('K_c', 3, 3)
        self.Kc_inv = Matrix([[1 / self.fu, -self.s / (self.fu * self.fv), (self.s * self.vc - self.fv * self.uc) / (self.fu * self.fv)], [0, 1 / self.fv, -self.vc / self.fv], [0, 0, 1]])

    def compute_vects_from_pixel(self):
        self.compute_vect1_from_pixel()
        self.compute_vect2_from_pixel()

    def compute_vect1_from_pixel(self):
        # Parameters related to Mirror 1:
        self.k1, self.c1 = symbols("k_1, c_1", real=True, positive=True)
        self.u1, self.v1 = symbols("u_1, v_1", real=True, nonnegative=True)
        # TODO: add assumption for k > 2
        # Viewpoint (focus) position vector:
        self.f1x, self.f1y, self.f1z = symbols("x_f_1, y_f_1, z_f_1", real=True)
        # (Coaxial alignment assumption)
        # where
        self.f1x = 0
        self.f1y = 0
        self.f1z = self.c1
        self.f1 = Matrix([self.f1x, self.f1y, self.f1z])
        # Pixel vector
        self.m1h = Matrix([self.u1, self.v1, 1])
        # Point in normalized projection plane
        self.q1 = self.Kc_inv * self.m1h
        # Point in mirror wrt C
        self.t1 = self.c1 / (self.k1 - self.q1.norm() * sqrt(self.k1 * (self.k1 - 2)))
        self.p1 = self.t1 * self.q1
        self.p1h = self.p1.col_join(eye(1))

        # Transform matrix from C to F1 frame
        self.T1_CtoF1 = eye(3).row_join(-self.f1)

        # Direction vector
        self.d1_vect = self.T1_CtoF1 * self.p1h

        return self.d1_vect

    def compute_vect2_from_pixel(self):
        # Parameters related to Mirror2:
        self.k2, self.c2, self.d = symbols("k_2, c_2, d", real=True, positive=True)
        self.u2, self.v2 = symbols("u_2, v_2", real=True, nonnegative=True)
        # Viewpoint (focus) position vector:
        self.f2x, self.f2y, self.f2z = symbols("x_f_2, y_f_2, z_f_2", real=True)
        # (Coaxial alignment assumption)
        # where
        self.f2x = 0
        self.f2y = 0
        self.f2z = self.d - self.c2
        self.f2 = Matrix([self.f2x, self.f2y, self.f2z])

        # Reflex mirror's normal vector wrt [C]
        self.n_ref = Matrix([0, 0, -1])
        # Virtual focus
        self.f2virt = Matrix([self.f2x, self.f2y, self.d])  # where f2virtz = d

        # Pixel vector
        self.m2h = Matrix([self.u2, self.v2, 1])

        # Point in the normalized projection plane
        self.q2 = self.Kc_inv * self.m2h
        # Planar Reflection Matrix (Change of coordinates)
        self.M_ref = (eye(3) + 2 * diag(*self.n_ref)).row_join(self.f2virt)

        # Change coordinates from C' to C (according to explanation in journal paper)
        self.q2v = self.M_ref * self.q2.col_join(eye(1))

        # Point in mirror wrt C
        self.t2 = self.c2 / (self.k2 - self.q2.norm() * sqrt(self.k2 * (self.k2 - 2)))
        self.p2 = self.f2virt + self.t2 * (self.q2v - self.f2virt)
        self.p2h = self.p2.col_join(eye(1))

        # Transform matrix from C to F1 frame
        self.T2_CtoF2 = eye(3).row_join(-self.f2)

        # Direction vector
        self.d2_vect = self.T2_CtoF2 * self.p2h

        return self.d2_vect

    def create_direction_vectors_symbols(self, as_real=None):
        self.d1x, self.d1y, self.d1z = symbols("v_1_x, v_1_y, v_1_z", real=as_real)
        self.d1_vect_as_symbol = Matrix([self.d1x, self.d1y, self.d1z])
        self.d2x, self.d2y, self.d2z = symbols("v_2_x, v_2_y, v_2_z", real=as_real)
        self.d2_vect_as_symbol = Matrix([self.d2x, self.d2y, self.d2z])
        self.direction_vectors_as_symb = Matrix([self.d1_vect_as_symbol, self.d2_vect_as_symbol])

    def get_triangulated_point_expanded(self):
        # Substitute expanded symbols
        reps_for_lambda_solns = [(self.d1x, self.d1_vect[0]), (self.d1y, self.d1_vect[1]), (self.d1z, self.d1_vect[2]), (self.d2x, self.d2_vect[0]), (self.d2y, self.d2_vect[1]), (self.d2z, self.d2_vect[2])]
        self.lambdaG1_expanded = self.lambdaG1.subs(reps_for_lambda_solns)
        self.lambda_perp_expanded = self.lambda_perp.subs(reps_for_lambda_solns)
        self.perp_vect_unit_expanded = self.perp_vect_unit_as_symbol.subs(reps_for_lambda_solns)
        self.G1_expanded = self.f1 + self.lambdaG1_expanded * self.d1_vect

        self.mid_Pw_expanded = self.G1_expanded + self.lambda_perp_expanded / 2.0 * self.perp_vect_unit_expanded
        return self.mid_Pw_expanded

    def compute_triangulated_point(self):
        # Solving using the common perpendicular method

        # Step 1. Create symbolic variables.
        self.create_direction_vectors_symbols(as_real=True)
        self.lambdaG1, self.lambdaG2, self.lambda_perp = symbols("lambda_G_1, lambda_G_2, lambda_{perp}", nonnegative=True, real=True)
        # To simplify computation, treat the direction vectors as independent symbolic variables
        # self.nx, self.ny, self.nz = symbols("n_x, n_y, n_z", real=True)
        # self.perp_vect_unit = Matrix([self.nx, self.ny, self.nz])
        d1_cross_d2 = self.d1_vect_as_symbol.cross(self.d2_vect_as_symbol)
        d1d2_norm = Abs(d1_cross_d2.norm())
        self.perp_vect_unit_as_symbol = d1_cross_d2 / d1d2_norm

        # Solving the linear system
        # self.M = Matrix([self.d1_vect_as_symbol.T, -self.d2_vect_as_symbol.T, self.perp_vect_unit_as_symbol.T]).T
        self.b_vect = self.f2 - self.f1
        # Step 1. Create a list of equations using the symbolic variables
        self.eqn_G2_is_G2 = Eq(self.lambdaG1 * self.d1_vect_as_symbol - self.lambdaG2 * self.d2_vect_as_symbol + self.lambda_perp * self.perp_vect_unit_as_symbol, self.b_vect)

        self.solve_eqn_G2_is_G2(use_solver=True)

        self.G1 = self.f1 + self.lambdaG1 * self.d1_vect_as_symbol
        self.mid_Pw = self.G1 + self.lambda_perp / 2.0 * self.perp_vect_unit_as_symbol
        return self.mid_Pw


    def solve_eqn_G2_is_G2(self, use_solver=True):
        # Step 2. Solve linear system of equations for the common perpendicular between two 3D rays
        if use_solver:
            soln = solve(self.eqn_G2_is_G2, [self.lambdaG1, self.lambdaG2, self.lambda_perp])
            self.lambdaG1 = soln[self.lambdaG1]
            self.lambdaG2 = soln[self.lambdaG2]
            self.lambda_perp = soln[self.lambda_perp]
        else:
            # This would be another way to solve this system
            # Given as M * t = b. Then, t = M^-1 * b
            M1_vect = Matrix([self.d1_vect_as_symbol, -self.d2_vect_as_symbol, self.perp_vect_unit_as_symbol])
            M1_3x3_T = Matrix(3, 3, M1_vect)
            M = M1_3x3_T.T
            lambda_vect = M.inv() * self.b_vect
            self.lambdaG1 = lambda_vect[0]
            self.lambdaG2 = lambda_vect[1]
            self.lambda_perp = lambda_vect[2]

    def compute_Jacobian_matrix_for_cov(self, use_chain=False):
        '''
        Compute Jacobian matrix (partial derivatives of Pw.x, y, z  with respect to u1, v1, u2, v2)
        '''
        self.uv_coords = Matrix([self.u1, self.v1, self.u2, self.v2])

        # Jacobians
        if use_chain:
            # Using a chain rule aproach to compute Jacobians
            self.jac_P_wrt_dirs = self.mid_Pw.jacobian(self.direction_vectors_as_symb)
            self.direction_vectors_as_funcs = Matrix([self.d1_vect, self.d2_vect])
            self.jac_dirs_wrt_coords = self.direction_vectors_as_funcs.jacobian(self.uv_coords)  # (a 6x4 Matrix)
            self.cov_jacobian_matrix = self.jac_P_wrt_dirs * self.jac_dirs_wrt_coords
        else:
            self.get_triangulated_point_expanded()
            self.cov_jacobian_matrix = self.mid_Pw_expanded.jacobian(self.uv_coords)


    def set_pixel_coordinates_covariance(self, stdev_on_pixel_coord):
        self.cov_pixel_coords_matrix = stdev_on_pixel_coord ** 2 * eye(4)

    def lambdify_jacobian_as_code_string(self):
        jac_args = (self.u1, self.v1, self.u2, self.v2, self.fu, self.fv, self.s, self.uc, self.vc, self.k1, self.k2, self.c1, self.c2, self.d)
        jac_expr = self.cov_jacobian_matrix
        jacobian_as_function = lambdify(jac_args, jac_expr, modules="numpy")
        return dump_function(jacobian_as_function)


    def get_jacobian_func_from_code_string(self):
#         code = marshal.loads(self.jacobian_function_marshaled)
#         func = types.FunctionType(code, locals(), "cov_jac_func")
        func = load_function(self.jacobian_function_marshaled)
        return func

def dump_function(function):
    """
    Returns a serialized version of the given function.
    """
    import marshal
    function_serialized = marshal.dumps((
            function.__code__,
            function.__defaults__,
            function.__dict__,
            function.__doc__,
            function.__name__))

    return function_serialized

def load_function(marshalled_function):
    """
    Returns a function loaded from data produced by the dump_function method
    of this module.
    """
    import marshal

    pairing = zip(
        ("__code__", "__defaults__", "__dict__", "__doc__", "__name__"),
        marshal.loads(marshalled_function))

    dummyfunction = lambda: None
    for attribute, value in pairing:
        setattr(dummyfunction, attribute, value)

    return dummyfunction

def load_marshalled_func_from_file(filename):
    marshalled_func = common_tools.load_obj_from_pickle(filename)
    return load_function(marshalled_func)

def get_symbolic_model(load_pickle=True):
    filename = "../data/omnistereo_symbolic.pkl"
    filename_jacobian_func = "../data/omnistereo_jacobian_marshalled_func.pkl"

    if load_pickle:
        omnistereo_symbolic = common_tools.load_obj_from_pickle(filename)
    else:
        omnistereo_symbolic = HyperCataStereoSymbolic()
        omnistereo_symbolic.compute_Jacobian_matrix_for_cov(use_chain=False)
        common_tools.save_obj_in_pickle(omnistereo_symbolic, filename, locals())

    jacobian_marshalled = omnistereo_symbolic.lambdify_jacobian_as_code_string()
    common_tools.save_obj_in_pickle(jacobian_marshalled, filename_jacobian_func, locals())

    return omnistereo_symbolic

if __name__ == '__main__':
    sym.init_printing()
    omnistereo_symbolic = get_symbolic_model(load_pickle=True)
