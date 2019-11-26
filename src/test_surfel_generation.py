"""Test Surfel Generation."""
import unittest

from surfel_generation import SurfelGeneration
from superpixel_seed import SuperpixelSeed
import numpy as np

class TestSurfelGeneration(unittest.TestCase):
    def test_create_surfels(self):
        """Test initialize Surfels from superpixels."""
        # initialize_surfels
        pass

    def test_update_surfels(self):
        "Test update surfels"
        sg = SurfelGeneration()
        sp = SuperpixelSeed(0, 0, 1,
                            0, 0, 1,
                            0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
        pose0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        sg.update_surfels([sp], pose0)
        # identical surfel should be fused
        self.assertEqual(len(sg.all_surfels), 1)
        sg.update_surfels([sp], pose0)
        # surfel in same location and norm should be fused
        self.assertEqual(len(sg.all_surfels), 1)
        sp.posi_x = 1
        sp.posi_z = 0
        sp.norm_x = 1
        sp.norm_z = 0
        pose1 = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        sg.update_surfels([sp], pose1)
        self.assertEqual(len(sg.all_surfels), 1)
        # surfel in same location but different norm should NOT be fused
        sp.norm_x = 0
        sp.norm_z = 1
        sg.update_surfels([sp], pose1)
        self.assertEqual(len(sg.all_surfels), 2)
        # surfel in different location and different norm should NOT be fused
        sg.update_surfels([sp], pose0)
        self.assertEqual(len(sg.all_surfels), 3)

if __name__ == '__main__':
    unittest.main()
