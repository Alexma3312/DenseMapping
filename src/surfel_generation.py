"""Surfel Generation"""
"""Surfel Generation."""
from surfel_element import SurfelElement
from superpixel_seed import SuperpixelSeed
from typing import List
import numpy as np

class SurfelGeneration():

    def __init__(self, camera_parameters):
        self.all_surfels: List[SurfelElement] = []
        self.camera_parameters = camera_parameters

    def create_surfels(self, superpixels: List[SuperpixelSeed]) -> List[SurfelElement]:
        """Create surfels from superpixels for given frame
        Arguments:
            superpixels: list of SuperpixelSeed
        Returns:
            surfels: a list of surfel elements
        """
        pass

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
