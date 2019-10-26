"""
tests superpixel extractions
"""

import unittest

from superpixel_extraction import *
from superpixel_seed import SuperpixelSeed
from surfel_element import SurfelElement

class TestSuperpixelExtraction(unittest.TestCase):

    def test_extract_superpixels(self):
        pass

    def test_init_seeds(self):
        """Tests initializing superpixel seeds
        """
        dims = (30,20)
        sp_size = 10
        expected_centers = [
            [5, 5], [15, 5], [25, 5],
            [5, 15], [15, 15], [25, 15]
        ]

        superpixels = init_seeds(dims, sp_size=10)
        self.assertEqual(len(superpixels), len(expected_centers),
            "didn't return the correct number of superpixels")
        for superpixel in superpixels:
            try:
                expected_centers.remove(superpixel.center)
            except ValueError:
                self.fail("found unexpected superpixel center")
            self.assertEqual(superpixel.size, 0, "size should be init to 0")
            self.assertEqual(superpixel.mean_depth, 0, "depth should init to 0")
            self.assertEqual(superpixel.mean_intensity, 0, "intensity should init to 0")
        self.assertFalse(expected_centers, "not all superpixel centers found")

    def test_assign_pixels(self):
        pass

    def test_update_seeds(self):
        pass

    def test_calc_norms(self):
        pass

if __name__ == '__main__':
    unittest.main()