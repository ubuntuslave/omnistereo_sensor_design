% -*- coding: utf-8 -*-
% load_params.m 

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

% Load design parameter from file
% c1 and c2 are the focal lengths of the upper and the lower mirror respectively
% k1 and k2 are the other parameters of the SVP mirror shape 
% d is distance between the far foci of the two mirrors, which 
% is also the distance beween pinhole and the far foci and the lower mirror

function [params] = load_params(filename)

if nargin < 1
    filename='parameter-BIG-test_no_k1k2_ineq_constraint.txt';
end

params = load(filename)

c1 = params(1);
c2 = params(2);
k1 = params(3);
k2 = params(4);
d = params(5);
r_sys = params(6);
r_ref = params(7);
r_hole = params(8);

end