#!/usr/bin/python
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# the resulting .ply file can be viewed for example with meshlab
# sudo apt-get install meshlab

"""
This script reads a registered pair of color and depth images and generates a
colored 3D point cloud in the PLY format.
"""

import argparse
import sys
import os
from PIL import Image
import numpy as np
from surfel_element import SurfelElement

def generate_pointcloud(surfels,ply_file):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.
    
    Input:
    surfels -- list of surfels
    ply_file -- filename of ply file
    
    """
    surfel_mesh = []   
    points = [] 
    for i,surfel in enumerate(surfels):
        surfel_position = np.array([surfel.px, surfel.py, surfel.pz])
        surfel_norm = np.array([surfel.nx, surfel.ny, surfel.nz])
        x_dir = np.array([-surfel.ny,surfel.nx,0])
        y_dir = np.cross(surfel_norm,x_dir)
        radius = surfel.size
        h_r = radius*0.5
        t_r = radius*0.86603
        point1 = surfel_position - x_dir * h_r - y_dir * t_r
        point2 = surfel_position + x_dir * h_r - y_dir * t_r
        point3 = surfel_position - x_dir * radius
        point4 = surfel_position + x_dir * radius
        point5 = surfel_position - x_dir * h_r + y_dir * t_r
        point6 = surfel_position + x_dir * h_r + y_dir * t_r
        surfel_mesh.append("%f %f %f %d %d %d\n"%(point1[0],point1[1],point1[2],surfel.color*255,surfel.color*255,surfel.color*255))
        surfel_mesh.append("%f %f %f %d %d %d\n"%(point2[0],point2[1],point2[2],surfel.color*255,surfel.color*255,surfel.color*255))
        surfel_mesh.append("%f %f %f %d %d %d\n"%(point3[0],point3[1],point3[2],surfel.color*255,surfel.color*255,surfel.color*255))
        surfel_mesh.append("%f %f %f %d %d %d\n"%(point4[0],point4[1],point4[2],surfel.color*255,surfel.color*255,surfel.color*255))
        surfel_mesh.append("%f %f %f %d %d %d\n"%(point5[0],point5[1],point5[2],surfel.color*255,surfel.color*255,surfel.color*255))
        surfel_mesh.append("%f %f %f %d %d %d\n"%(point6[0],point6[1],point6[2],surfel.color*255,surfel.color*255,surfel.color*255))
        #
        p1 = i*6 + 0 
        p2 = i*6 + 1 
        p3 = i*6 + 2 
        p4 = i*6 + 3 
        p5 = i*6 + 4 
        p6 = i*6 + 5 
        points.append("3 %d %d %d\n"%(p1,p2,p3)) 
        points.append("3 %d %d %d\n"%(p2,p4,p3)) 
        points.append("3 %d %d %d\n"%(p3,p4,p5)) 
        points.append("3 %d %d %d\n"%(p5,p4,p6)) 

    numSurfels = len(surfels)
    numPoints = numSurfels*6
    file = open(ply_file,"w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face %d
property list uchar int vertex_index
end_header
%s
'''%(numPoints,numSurfels*4,"".join(surfel_mesh)))
    file.write("".join(points))

    file.close()

if __name__ == '__main__':
    surf_1 = SurfelElement(15, 16, 17, 18, 19, 20, 3, 200, 2, 1, 5)
    surf_2 = SurfelElement(1, 2, 3, 4, 5, 6, 3, 210, 2, 1, 5)
    surfels = [surf_1,surf_2]
    
    generate_pointcloud(surfels,"../dataset/1.ply")
    