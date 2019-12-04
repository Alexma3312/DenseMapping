"""Surfel Generation"""
from math import fabs
from typing import List
import numpy as np

from superpixel_seed import SuperpixelSeed
from surfel_element import SurfelElement
import numpy as np

class SurfelGeneration():
    """Surfel Generation."""
    def __init__(self, camera_parameters, MAX_ANGLE_COS=.1, VERBOSE=False):
        self.all_surfels = []
        self.camera_parameters = camera_parameters
        self.MAX_ANGLE_COS = MAX_ANGLE_COS
        self.VERBOSE = VERBOSE

    def get_weight(self, depth):
        return min(1.0 / depth / depth, 1.0)

    def create_surfels(self, frame_idx, superpixels: List[SuperpixelSeed], pose) -> List[SurfelElement]:
        """Create surfels from superpixels for given frame
        Arguments:
            frame_idx: index of current frame
            superpixels: list of SuperpixelSeed
            pose: pose of camera in current frame
        Returns:
            surfels: a list of surfel elements
        """

        surfels = []
        for superpixel in superpixels:
            px = superpixel.posi_x
            py = superpixel.posi_y
            pz = superpixel.posi_z
            nx = superpixel.norm_x
            ny = superpixel.norm_y
            nz = superpixel.norm_z
            camera_f = (fabs(self.camera_parameters['fx']) +
                        fabs(self.camera_parameters['fy'])) / 2.0
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
            surfels.append(surfel.change_coordinates(pose))
        return surfels

    def update_surfels(self, frame_idx, superpixels: List[SuperpixelSeed], pose) -> None:
        """Update internal global list of surfels using the superpixels from one frame
        Arguments:
            frame_idx: index of current frame, for storing when a surfel last got updated
            superpixels: list of SuperpixelSeed
            pose: camera pose in world coordinates
        """
        inverse_pose = np.zeros((4,4))
        inverse_pose[0:3, 0:3] = pose[0:3, 0:3].transpose()
        inverse_pose[0:3, 3] = -np.dot(inverse_pose[0:3, 0:3], pose[0:3, 3])
        inverse_pose[3, 3] = 1

        superpixels_copy = superpixels.copy()
        for sp in superpixels_copy:
            if (sp.mean_depth == 0) or (sp.fused) or (sp.view_cos < self.MAX_ANGLE_COS) or np.isnan(sp.view_cos):
                superpixels.remove(sp)
        new_surfels = self.create_surfels(frame_idx, superpixels, pose)
        # print(new_surfels)
        projected_locs = np.zeros((len(self.all_surfels), 2))
        for i in range(len(self.all_surfels)):
            projected_locs[i, :] = \
                self.all_surfels[i].change_coordinates(inverse_pose).back_project(self.camera_parameters)
        for i, sp in enumerate(superpixels):
            sp_pos = np.array([sp.x, sp.y])
            dists2 = np.sum(np.square(sp_pos - projected_locs), axis=1)
            candidate_surfels = dists2 < (sp.size)
            did_fuse = False
            for surfel_i in np.argwhere(candidate_surfels):
                surfel = self.all_surfels[surfel_i[0]]
                if surfel.is_fuseable(new_surfels[i]):
                    surfel.fuse_surfel(new_surfels[i])
                    did_fuse = True
            if not did_fuse:
                # print('bad surfel.')
                self.all_surfels.append(new_surfels[i]) # todo: update projected_locs
