% -*- coding: utf-8 -*-
% save_params.m 

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

function [params] = save_params(x, filename)

% Save the relevant parameters in a file to be used in POV-Ray for the synthetic model.
params = get_all_params(x);

if nargin < 2
    filename='parameter.txt';
end

msgsaving = sprintf('Saving parameters to file %s', filename);
display(msgsaving)

fid = fopen(filename,'w');  % A local file name
% fprintf(fid,'%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n',params);
fprintf(fid,'%.4f', params(1));
for elm = params(2:end)
    fprintf(fid,', %.4f', elm);
end
fprintf(fid,'\n');

fclose(fid);

end