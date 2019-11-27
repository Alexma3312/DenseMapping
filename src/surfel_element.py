"""Surfel Element."""

# from __future__ import annotations
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

    def change_coordinates(self, Twc):# -> SurfelElement:
        """Transforms surfel from camera pose into world coordinates
        Arguments:
            Twc: camera pose in world coordinates
        Returns:
            surfel: SurfelElement in new coordinates
        """
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

    def is_fuseable(self, surfel,#: SurfelElement,
                    bf=1, disparity_std=1) -> bool:
        """Test whether another surfel is a correspondence and can be fused or not
        Arguments:
            surfel: other surfel
            bf: focal length times baseline for the depth sensor.  Scaling factor for raw to real
                distance.
            disparity_std: standard deviation of disparity estimate
        Returns:
            is_fuseable: True if correspond else False
        """
        n_new =  np.array([[surfel.nx],[surfel.ny],[surfel.nz]])
        n_local=  np.array([[self.nx],[self.ny],[self.nz]])

        if ((abs(surfel.pz-self.pz)<(self.pz/2)) and float(np.dot(n_new.T,n_local))>0.8):
            return True
        return False

    def fuse_surfel(self, surfel#: SurfelElement
                    ) -> None:
        """Fuses this surfel with another Surfel.  This surfel changes to the fused version.
        Arguments:
            surfel: other surfel
        """
        return