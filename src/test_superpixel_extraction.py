"""
tests superpixel extractions
"""

import unittest

import matplotlib.pyplot as plt
import numpy as np

from superpixel_extraction import *
from superpixel_seed import SuperpixelSeed
from surfel_element import SurfelElement


expected_init_centers = [
    [10, 10], [10, 30], [10, 50], [10, 70], [10, 90],
    [30, 10], [30, 30], [30, 50], [30, 70], [30, 90],
    [50, 10], [50, 30], [50, 50], [50, 70], [50, 90],
    [70, 10], [70, 30], [70, 50], [70, 70], [70, 90],
    [90, 10], [90, 30], [90, 50], [90, 70], [90, 90]
]
expected_init_intensities = [
    [100, 0, 0], [100, 0, 0], [100, 0, 0], [100, 0, 100], [100, 0, 100],
    [0, 100, 0], [0, 100, 0], [0, 100, 0], [100, 0, 100], [100, 0, 100],
    [0, 0, 100], [0, 0, 100], [0, 0, 100], [100, 100, 100], [100, 100, 100],
    [100, 100, 0], [100, 100, 0], [100, 100, 0], [100, 100, 100], [100, 100, 100],
    [0, 100, 100], [0, 100, 100], [0, 100, 100], [100, 100, 100], [100, 100, 100]
]
expected_init_depths = [
    1, 1, 1, 200, 200,
    10, 10, 10, 200, 200,
    50, 50, 50, 250, 250,
    100, 100, 100, 250, 250,
    150, 150, 150, 250, 250,
]

class TestSuperpixelExtraction(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestSuperpixelExtraction, self).__init__(*args, **kwargs)
        self.image = plt.imread("test_data/test_image.png")
        self.depth = plt.imread("test_data/test_depth.png")
        (w, h) = self.image.shape
        camera_parameters = {'fx': 1, 'fy': 1, 'cx': w/2, 'cy': h/2}
        self.spExtractor = SuperpixelExtraction(self.image, self.depth,
            camera_parameters, sp_size=10)
    
    def calc_distance(self):
        pass

    def test_extract_superpixels(self):
        pass

    def test_init_seeds(self):
        """Tests initializing superpixel seeds
        """
        sp_size = 20
        passed = [False]*6

        superpixels = self.spExtractor.init_seeds()

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
        image = self.image
        depth = self.depth

        superpixels = []
        for i in range(len(expected_init_centers)):
            superpixels.append(SuperpixelSeed(
                expected_init_centers[i][0],
                expected_init_centers[i][1],
                0, 0, 0, 0, 0, 0, 0, 0,
                expected_init_depths[i],
                expected_init_intensities[i],
                0, 0, 0, 0
            ))
        superpixels = init_seeds(image, depth, sp_size=10)

        expected_pixels = np.zeros((100,100), dtype=np.int32)
        expected_pixels[0:20, 0:20] = 0
        expected_pixels

        pass

    def test_update_seeds(self):
        pass

    def test_calc_norms(self):
        pass

if __name__ == '__main__':
    unittest.main()