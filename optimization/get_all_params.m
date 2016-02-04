% -*- coding: utf-8 -*-
% get_all_params.m 

% Copyright (c) 2016, Carlos Jaramillo
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

% Save design parameter to file
% c1 and c2 are the focal lengths of the upper and the lower mirror respectively
% k1 and k2 are the other parameters of the SVP mirror shape 
% d is distance between the far foci of the two mirrors, which 
% is also the distance beween pinhole and the far foci and the lower mirror

function [params] = get_all_params(x)

syms r z


c1 = x(1);
c2 = x(2);
k1 = x(3);
k2 = x(4);
d = x(5);
r_sys = x(6);

r_ref = sqrt(((d-c1)^2-c1^2*(1-2/k1))/(2*k1-4)); % Planar (reflex) mirror radius
z1 = c1/2+sqrt(r_sys^2*(k1/2-1)+c1^2/4*(1-2/k1));  %upper mirror edge (highest point on mirror1)
z2 = d-c2/2-sqrt(r_sys^2*(k2/2-1)+c2^2/4*(1-2/k2)); %lower mirror egde (lowest point on mirror2)

%the hole of the lower mirror (r_hole,z_hole), is collinear with the point(r_sys,z1), 
% but the center is F2 %FIXME: I think it should be centered on F1'
% that is, r_hole/(z_hole + c2-d) = r_sys/(z1 + c2-d)
% so the intersection point, on the surface of the lower mirror, is the solution of :
%(z_hole - (d-c2/2) )^2 - r_hole^2 * (k2/2 -1) = c2^2*(k2 - 2)/(4*k2)
% that is,a*r_hole^2-b*r_hole+c=0;
a = ((z1+c2-d)/r_sys)^2-(k2/2-1);
b = (z1+c2-d)/r_sys*c2;
c = c2^2/(2*k2);
r_hole = (b-sqrt(b^2-4*a*c))/(2*a);   %radius of camera hole on lower mirror 
% CHECKME: z-coordinate of the camera hole
z_hole = (z1+c2-d)*r_hole/r_sys-c2+d;  % Not saved as params, but included here for completeness.

% Save the relevant parameters in a file to be used in POV-Ray for the synthetic model.
params = [x' r_ref r_hole];
end