% -*- coding: utf-8 -*-
% draw.m 

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

% draw the section shape of a folded mirror given the following parameters
% c1 and c2 are the focal lengths of the upper and the lower mirror respectively
% k1 and k2 are the other parameters of the SVP mirror shape 
% d is distance between the far foci of the two mirrors, which 
% is also the distance beween pinhole and the far foci and the lower mirror

function draw(params)

clf

syms r z

c1 = params(1)
c2 = params(2)
k1 = params(3)
k2 = params(4)
d = params(5)
r_sys = params(6)
baseline = params(7)
r_ref = params(8)
r_hole = params(9)

z1 = c1/2+sqrt(r_sys^2*(k1/2-1)+c1^2/4*(1-2/k1));  %upper mirror edge (toppest point)
z2 = d-c2/2-sqrt(r_sys^2*(k2/2-1)+c2^2/4*(1-2/k2)); %lower mirror egde (lowest point)
height = z1-z2; %System height
zz1 = c1/2+c1/2*sqrt(1-2/k1);   %upper mirror (Equation for F1, real focal point)
zz2 = d-c2/2-c2/2*sqrt(1-2/k2);  %lower mirror (Equation for F2, real focal point)
z_hole = (z1+c2-d)*r_hole/r_sys-c2+d; % z-coordinate of the camera hole

% FOV angles: TODO: not anymore using these alpha,beta angle names
alpha = atan((z1-c1)/r_sys)*180/pi;
beta = atan(r_ref/(c1-d/2))*180/pi;
gama = atan((-z2-c2+d)/r_sys)*180/pi;

%Computing the intersection of the two lines of sight emanating from the focci, in order to determine the mounting height, H
l1 = (d/2-c1)/r_ref; % Equation of the slope of the upper line of sight from F1
l2 = (z2+c2-d)/r_sys; %Equation of the lower line of sight from F2
%The intersection of the lines of sight at point A(p, q)
p = (d-c2-c1)/(l1-l2);  % r-coordinate 
q = (l1*d-l1*c2-l2*c1)/(l1-l2); % z-coordinate
H = c1-q;  % Mounting height % CHECKME: not sure 


% Start DRAWING:
% vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
zmax = max(z1,d); % Highest margin point for drawing
zmin = -10 % q-2;    % Lowest margin point for drawing

% Horizontal and vertical axes define the image starting position
axis equal; hold on
% axis ([-20 40 zmin zmax]);hold on
axis([-15 95 zmin zmax]);hold on


% draw top and bottom bases:
plot([-r_sys, r_sys],[z1, z1]);hold on
plot([-r_sys, r_sys],[z2, z2]);hold on
% Draw central axis of symmetry
plot([0,0],[z2,d],'b-.');hold on

% upper mirror (without the flat parts)
ezplot( (z - c1/2)^2 - r^2 * (k1/2 - 1) - c1^2 * (k1 - 2)/(4*k1) ,[r_ref,10,6,z1]);hold on  %ezplot(fun,[xmin,xmax,ymin,ymax])
ezplot( (z - c1/2)^2 - r^2 * (k1/2 - 1) - c1^2 * (k1 - 2)/(4*k1) ,[-10,-r_ref,6,z1]);hold on %ezplot(fun,[xmin,xmax,ymin,ymax])

% planar mirror
plot([-r_ref, r_ref],[d/2, d/2],'r');hold on
% projective center & focus of the upper mirror
plot(0,0,'k:.');hold on
text(0,0,'(F1'')O ','FontSize',8,'HorizontalAlignment','right','VerticalAlignment','middle');
%text(0,0,' (O)','FontSize',8,'HorizontalAlignment','left','VerticalAlignment','middle');
plot(0,c1,'k:.');hold on
text(0,c1,'F1 ','FontSize',8,'HorizontalAlignment','right','VerticalAlignment','middle');
 
% lower mirror, it's two foci and the hole
ezplot( (z - (d - c2/2))^2 - r^2 * (k2/2 - 1) - c2^2 * (k2 - 2)/(4*k2) ,[-r_sys,-r_hole,z2,10]);hold on %ezplot(fun,[xmin,xmax,ymin,ymax])
ezplot( (z - (d - c2/2))^2 - r^2 * (k2/2 - 1) - c2^2 * (k2 - 2)/(4*k2) ,[r_hole,r_sys,z2,10]);hold on  %ezplot(fun,[xmin,xmax,ymin,ymax])
plot(0,(d - c2),'r:.');hold on
text(0,d-c2,'F2 ','FontSize',8,'HorizontalAlignment','right','VerticalAlignment','middle');
plot(0,d,'r:.');hold on
text(0,d,'F2'' ','FontSize',8,'HorizontalAlignment','right','VerticalAlignment','middle');
plot(r_hole,z_hole,'k:.');hold on

%draw field of view
%upper view
plot([0,r_sys,15],[c1,z1,c1+15*(z1-c1)/r_sys],'g:');hold on
plot([0,r_ref,30],[c1,d/2,c1+30*(d/2-c1)/r_ref],'g:');hold on
%lower view
plot([0,r_sys,15],[d-c2,z1,d-c2+15*(z1-d+c2)/r_sys],'r:');hold on
plot([0,r_sys,30],[d-c2,z2,d-c2+30*(z2-d+c2)/r_sys],'r:');hold on

% Draw support tube (glass cylinder)
plot([-r_sys,-r_sys],[z2,z1],'k');hold on % Left side
plot([r_sys,r_sys],[z2,z1],'k');hold on % Right side


% Draw intersection point A (for lower lines of sight)
% Point A is the lowest and closest (horizontal distance) that can be seen by both mirrors.
plot([0,r_ref,p],[c1,d/2,q],'m:');hold on
plot([0,r_sys,p],[d-c2,z2,q],'m:');hold on
text(p,q,'\color{Red}A');
% By plotting the horizontal abscissa to A from the Z-axis, it exceeds the default 50cm (design constraint)
% We have to consider this as an optimization constraint, as well.



% A summary table for the parameters values
text(82,z1-2,['c1 = ',num2str(c1)]);
text(82,z1-4,['c2 = ',num2str(c2)]);
text(82,z1-6,['k1 = ',num2str(k1)]);
text(82,z1-8,['k2 = ',num2str(k2)]);
text(82,z1-10,['d = ',num2str(d)]);
text(82,z1-12,['r_{sys} = ',num2str(r_sys)]);
text(82,z1-14,['r = ',num2str(r_ref)]);
text(82,z1-16,['height = ',num2str(height)]);
text(82,z1-18,['r_{hole} = ',num2str(r_hole)]);
text(82,z1-20,['alpha = ',num2str(alpha)]);
text(82,z1-22,['beta = ',num2str(beta)]);
text(82,z1-24,['gama = ',num2str(gama)]);
text(82,z1-26,['H=',num2str(H)]);
text(82,z1-28,['A=(',num2str(p), ', ', num2str(q), ')']);

text(82,z1+2,['z1 = ',num2str(z1)]);
text(82,z1+4,['z2 = ',num2str(z2)]);
text(82,z1+6,['zz1 = ',num2str(zz1)]);
text(82,z1+8,['zz2 = ',num2str(zz2)]);

end