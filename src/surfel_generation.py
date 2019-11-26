"""Surfel Generation"""
"""Surfel Generation."""
from surfel_element import SurfelElement
from superpixel_seed import SuperpixelSeed
from typing import List

class SurfelGeneration():
    
    def __init__(self):
        self.all_surfels = []
    
    def create_surfels(self, superpixels -> List[SuperpixelSeed]) -> List[SurfelElement]:
        """Create surfels from superpixels for given frame
        Arguments:
            superpixels: list of SuperpixelSeed
        Returns:
            surfels: a list of surfel elements
        """
        pass
    
    def update_surfels(self, superpixels -> List[SuperpixelSeed], pose) -> None:
        """Update internal global list of surfels using the superpixels from one frame
        Arguments:
            superpixels: list of SuperpixelSeed
            pose: camera pose in world coordinates
        """
        pass