"""Surfel Generation"""
from math import fabs
from typing import List
import numpy as np

from superpixel_seed import SuperpixelSeed
from surfel_element import SurfelElement
import numpy as np

class SurfelGeneration():
    """Surfel Generation."""
    def __init__(self, camera_parameters, MAX_ANGLE_COS=0.1, VERBOSE=False):
        self.all_surfels = []
        self.camera_parameters = camera_parameters
        self.MAX_ANGLE_COS = MAX_ANGLE_COS
        self.VERBOSE = VERBOSE

    def get_weight(self, depth):
        return min(1.0 / depth / depth, 1.0)

    def create_surfels(self, frame_idx, superpixels: List[SuperpixelSeed]) -> List[SurfelElement]:
        """Create surfels from superpixels for given frame
        Arguments:
            superpixels: list of SuperpixelSeed
        Returns:
            surfels: a list of surfel elements
        """
        surfels = []
        for superpixel in superpixels:
            if superpixel.mean_depth == 0:
                continue
            if superpixel.fused:
                continue
            if superpixel.view_cos < self.MAX_ANGLE_COS:
                continue
            px = superpixel.posi_x
            py = superpixel.posi_y
            pz = superpixel.posi_z
            nx = superpixel.norm_x
            ny = superpixel.norm_y
            nz = superpixel.norm_z
            camera_f = (fabs(self.fx) + fabs(self.fy)) / 2.0
            new_size = superpixel.size * \
                fabs(superpixel.mean_depth / (camera_f * superpixel.view_cos))
            if(new_size > 0.1 and self.VERBOSE):
                print("max eigen: ", superpixel.max_eigen_value, ", min eigen: ", superpixel.min_eigen_value, ', camera_f: ',
                      camera_f , ", distence: ", superpixel.mean_depth, ", cos: ", superpixel.view_cos, " ->size: ", new_size)
            color = superpixel.mean_intensity
            weight = self.get_weight(superpixel.mean_depth)
            update_times = 1
            last_update = frame_idx
            surfel = SurfelElement(
                px, py, pz, nx, ny, nz, new_size, color, weight, update_times, last_update)
            surfels.append(surfel)
        return surfels

    def update_surfels(self, superpixels: List[SuperpixelSeed], pose) -> None:
        """Update internal global list of surfels using the superpixels from one frame
        Arguments:
            superpixels: list of SuperpixelSeed
            pose: camera pose in world coordinates
        """
        new_surfels = self.create_surfels(superpixels)
        projected_locs = np.zeros((len(self.all_surfels), 2))
        for i in range(len(self.all_surfels)):
            projected_locs[i, :] = \
                self.all_surfels[i].change_coordinates(pose).back_project(self.camera_parameters)
        for i, sp in enumerate(superpixels):
            sp_pos = np.array([sp.x, sp.y])
            dists2 = np.sum(np.square(sp_pos - projected_locs), axis=1)
            candidate_surfels = dists2 < sp.size
            did_fuse = False
            for surfel in candidate_surfels:
                if surfel.is_fuseable(new_surfels[i]):
                    surfel.fuse_surfel(new_surfels[i])
                    did_fuse = True
            if not did_fuse:
                self.all_surfels.append(new_surfels[i]) # todo: update projected_locs
