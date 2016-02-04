% -*- coding: utf-8 -*-
% nonlinearconstraints.m 

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

% Nonlinear constraints:
%  The function 'mycon' accepts X 
%   and returns the vectors C and Ceq, representing the nonlinear 
%   inequalities and equalities respectively. 
% FMINCON minimizes 'myfun' such that C(X) <= 0 and Ceq(X) = 0.
% NOTE: we use "cineq" instead of C for the inequalities here.
function [ cineq,ceq ] = nonlinearconstraints( x )

%x is [c1;c2;k1;k2;d;r_sys]
c1 = x(1);
c2 = x(2);
k1 = x(3);
k2 = x(4);
d  = x(5);
r_sys = x(6);

% Nonlinear Equality Constraint (on the edge?...maximum point using r_sys)
% Mirror radius of both expressions should be identical
% r_{sys} = r_{1,max} = r_{2,max}
ceq(1) = ((d-c1)^2 - c1^2*(k1-2)/k1) / (2*k1-4) - ...
        (d*r_sys)^2/(c2 + 2*sqrt(r_sys^2*(k2/2-1) + c2^2*(k2-2)/(4*k2)))^2; 
        
% vvvvvvvvvvvvv Nonlinear Inequalities: vvvvvvvvvvvvvvvvv
        
% LENGTH constraints        
        
% TODO: find/verify the proper equation of h_sys        
% System's height constraint        
% NOTE: set this value for SMALL or BIG rigs
% max_height = 12; 
max_height = 20; % BIG rig 
% h_{sys} <= max_height
% h_{sys} = sqrt(c1^2*(k1-2)/(4*k1) + r_sys^2*(k1/2-1)) + c1/2 + sqrt(c2^2*(k2-2)/(4*k2)+r_sys^2*(k2/2-1)) - d + c2/2
cineq(1) = sqrt(c1^2*(k1-2)/(4*k1) + r_sys^2*(k1/2-1)) + c1/2 + sqrt(c2^2*(k2-2)/(4*k2)+r_sys^2*(k2/2-1)) - d + c2/2 -max_height; 

% On the lower mirror:
% We interpreted correctly!
%c2/2*sqrt((k2-2)/k2) <= d -1/2 - c2/2   ===> z0_2 - a2 > 1/2 = 0.5 cm
% In order to place vertices on the vertical transverse axis at greater than 0.5 cm from the center (origin of coordinate or camera frame).
cineq(2) = 1/2 + c2/2 - d + c2/2*sqrt((k2-2)/k2); 

% ANGULAR Constraints
% FIXME Ensures that incident light can be reflected by lower mirror

% 14 <= \theta_{1, max}
% where \theta_{1, max} := (sqrt(c1^2*(k1-2)/(4*k1)+r_sys^2*(k1/2-1))-c1/2)/r_sys
cineq(3) = tan(14*pi/180) - (sqrt(c1^2*(k1-2)/(4*k1)+r_sys^2*(k1/2-1))-c1/2)/r_sys;

% 65 <= \beta (as known by GUO: \beta = 90 - \alpha_{1,min})
% tan(\beta) = sqrt( (d-c1)^2-c1^2*(k1-2)/k1) / (2*k1-4)/((c1-d/2)^2)))
% 65 <= \theta_{1,min}  Translates to -25 <= \theta_{1,min} USING equation \alpha_{1,min} instead!
cineq(4) = (tan(65*pi/180))^2-((d-c1)^2-c1^2*(k1-2)/k1)/(2*k1-4)/((c1-d/2)^2); 
%cineq(4) = (tan(-25*pi/180))^2 - equation \alpha_{1,min}; 

% -14 <= \theta_{2, min}
% \theta_{2, min} := (sqrt(c2^2*(k2-2)/(4*k2)+r_sys^2*(k2/2-1))-c2/2)/r_sys
cineq(5) = ((sqrt(c2^2*(k2-2)/(4*k2)+r_sys^2*(k2/2-1))-c2/2) / r_sys ) - tan(14*pi/180);

% MAV'S payload:
% rig's weight <= 650 g (NOTE: it was originally set at 500 g by GUO)
% So W_{sys} = pi*2.5*(sqrt(c1^2*(k1-2)/(4*k1)+r_sys^2*(k1/2-1))+c1/2+sqrt(c2^2*(k2-2)/(4*k2)+r_sys^2*(k2/2-1))-d+c2/2)*(0.6*r_sys+0.09)
cineq(6) = pi*2.5*(sqrt(c1^2*(k1-2)/(4*k1)+r_sys^2*(k1/2-1))+c1/2+sqrt(c2^2*(k2-2)/(4*k2)+r_sys^2*(k2/2-1))-d+c2/2)*(0.6*r_sys+0.09)-650;

% Not using this constraint:
%cineq(7) = 50 - (c1+c2-d)/((c2/2-sqrt(c2^2/4*(1-2/k2)+r_sys^2*(k2/2-1)))/r_sys + (c1-d/2)/(d/2*r_sys/(c2/2+sqrt(c2^2/4*(1-2/k2)+r_sys^2*(k2/2-1)))));

end