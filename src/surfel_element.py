"""Surfel Element."""

from typing import Tuple
from superpixel_seed import SuperpixelSeed
import numpy as np

class SurfelElement():
    def __init__(self, px, py, pz, nx, ny, nz, size, color, weight, update_times, last_update):
        """Surfel Element Data Structure.
        Arguments:
            px,py,pz: the surfel center
            nx,ny,nz:
            size: 
            color:
            weight:
            update_times: 
            last_update:
        """
        self.px, self.py, self.pz = px, py, pz
        self.nx, self.ny, self.nz = nx, ny, nz
        self.size = size
        self.color = color
        self.weight = weight
        self.update_times = update_times
        self.last_update = last_update

    def change_coordinates(self, Twc):
        """Transforms surfel from camera pose into world coordinates
        Arguments:
            Twc: camera pose in world coordinates"""
        p = np.array([self.px, self.py, self.pz, 1])
        p = Twc.dot(p)
        n = np.array([self.nx, self.ny, self.nz, 0])
        n = Twc.dot(n)
        return SurfelElement(p[0], p[1], p[2],
                             n[0], n[1], n[2],
                             self.size, self.color, self.weight, self.update_times,
                             self.last_update)

    def back_project(self, camera_parameters) -> Tuple[float, float]:
        """Back projects surfel into camera image
        Arguments:
            camera_parameters: camera parameters as dictionary with fx, fy, cx, cy
        Returns:
            (x, y): surfel center position in camera image
        """
        x = self.px / self.pz * camera_parameters['fx'] + camera_parameters['cx']
        y = self.py / self.pz * camera_parameters['fy'] + camera_parameters['cy']
        return x, y