"""
tests superpixel extractions
"""

import unittest

import matplotlib.pyplot as plt

from superpixel_extraction import *
from superpixel_seed import SuperpixelSeed
from surfel_element import SurfelElement

class TestSuperpixelExtraction(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestSuperpixelExtraction, self).__init__(*args, **kwargs)
        self.image = plt.imread("src/test_data/test_image.png")
        self.depth = plt.imread("src/test_data/test_depth.png")

    def test_extract_superpixels(self):
        pass

    def test_init_seeds(self):
        """Tests initializing superpixel seeds
        """
        sp_size = 20
        expected_centers = [
            [10, 10], [30, 10], [50, 10], [70, 10], [90, 10],
            [10, 30], [30, 30], [50, 30], [70, 30], [90, 30],
            [10, 50], [30, 50], [50, 50], [70, 50], [90, 50],
            [10, 70], [30, 70], [50, 70], [70, 70], [90, 70],
            [10, 90], [30, 90], [50, 90], [70, 90], [90, 90]
        ]
        expected_intensities = [

        ]
        expected_depths = [

        ]
        passed = [False]*6

        superpixels = init_seeds(test_image, test_depth, sp_size=10)
        self.assertEqual(len(superpixels), len(expected_centers),
            "didn't return the correct number of superpixels")

        for superpixel in superpixels:
            center = [superpixel.x, superpixel.y]
            try:
                index = expected_centers.index(center)
            except ValueError:
                self.fail("found unexpected superpixel center")
                continue
            self.assertFalse(passed[index], "duplicate superpixel center")

            expected_centers.remove(superpixel.center)
            self.assertEqual(superpixel.mean_depth, expected_depths[index],
                "depth not initialized properly")
            self.assertEqual(superpixel.mean_intensity, expected_intensities[index],
                "intensity not initialized properly")
            self.assertEqual(superpixel.size, 0, "size should be init to 0")
            passed[index] = True

        for elem in passed:
            self.assertTrue(elem, "not all superpixel centers found")

    def test_assign_pixels(self):
        pass

    def test_update_seeds(self):
        pass

    def test_calc_norms(self):

        pass

if __name__ == '__main__':
    unittest.main()