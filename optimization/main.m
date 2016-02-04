% -*- coding: utf-8 -*-
% main.m: Optimization of mirror parameters for wide baseline

% Copyright (c) 2016, Carlos Jaramillo and Ling Guo
% Produced at the Laboratory for Robotics and Intelligent Systems of the City College of New York
%All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% * Redistributions of source code must retain the above copyright
%   notice, this list of conditions and the following disclaimer.
% * Redistributions in binary form must reproduce the above copyright
%   notice, this list of conditions and the following disclaimer in the
%   documentation and/or other materials provided with the distribution.
% * Neither the name of the copyright holders nor the names of any
%   contributors may be used to endorse or promote products derived
%   from this software without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.

clc
clear

% Initial values:
% vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
% SMALLER rig (diameter 5.6 cm)
                         % ---> Results: [c1;      c2;       k1;     k2;        d;    r_sys |||  r_ref;  r_hole]
x0 = [10;20;3;5;20;2.8]; % ---> Results: 10.4585, 20.4335, 6.8810, 11.4684, 20.0000, 2.4567, ||| 1.1738, 0.1972 ---> baseline= 10.892, height=12cm
                         
% BIG rig: (diameter 7.4 cm)
%x0 = [12;22;3;5;20;3.6]; % ---> Results: 12.3488, 24.1803, 5.7319, 9.7443, 23.3684, 3.7000, ||| 1.7226, 0.3392

% ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

%c1 = 12.349 cm
%c2 = 24.180 cm
%k1 =  5.732 unitless
%k2 =  9.744 unitless
%d = 23.368 cm
%r_sys =  3.700 cm
%r_ref =  1.723 cm
%r_cam =  0.700 cm
%baseline = 13.161 cm
%Height of rig = 15.000 cm

% The set of LINEAR inequalities as A*x <= b, where x = [c1;c2;k1;k2;d;r_sys]
A_ineq = [ 0 -1  0   0  1   0;  % Mirror 2's focus position: d <= c2 
-1  0  0   0  0.5 0;  % Mirror 1's focus    d/2 <= c1 , where d/2 is the position of the reflex mirror
0  0 5/3 -1  0   0]; %  1.7 <= k2/k1  Arbitrary in order to give more curvature to the bottom mirror as it needs to look more forward rather than up?
                       % This is an empirical constraint to adapt the
                       % lowest common field of view to the quadrotor propellers as well as to  the view downwards  
b_ineq = [0;0;0];

%If no equalities exist, set A_eq = [] and b_eq = []
A_eq = [];
b_eq = [];

% (Set LB = [] and/or UB = [] if no bounds exist.)
% Lower and upper bounds:
% Recall optimization parameter vector: [c1;c2;k1;k2;d;r_sys]
% SMALLER rig:
v_lb = [8; 15;  3;  5; 10; 1.0];
v_ub = [15; 30; 10; 12; 20; 2.8]; 
% BIG rig:
% Values are in centimeters
%v_lb = [12; 15;  3;  5; 15; 3.6];      
%v_ub = [50; 50; 14; 14; 50; 4.0];

options = optimset('fmincon');
%options.MaxFunEvals = 6000;
%options.MaxIter = 1000; % 1000;    

%  The function 'mycon' accepts X 
%   and returns the vectors C and Ceq, representing the NONLINEAR 
%   inequalities and equalities respectively. 
% FMINCON minimizes 'myfun' such that C(X) <= 0 and Ceq(X) = 0.

% min F(X)  subject to:  A_ineq*X  <= b_ineq, A_eq*X  = b_eq (linear constraints)
%     X                     C(X) <= 0, Ceq(X) = 0   (nonlinear constraints defined in 'mycon')
%                              LB <= X <= UB        (lower and upper Bounds)
[x, fval] = fmincon('objectivefunctionforbaseline', x0, A_ineq, b_ineq, A_eq, b_eq, v_lb, v_ub, 'nonlinearconstraints', options)

params = save_params(x, 'parameter-SMALL.txt'); % params order: [c1;c2;k1;k2;d;r_sys;r_ref;r_hole]
baseline = objectivefunctionforbaseline(x)
display('Optimal baseline = ' + baseline);
draw(params);