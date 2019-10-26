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
expected_init_centers = [[c[0]-1, c[1]-1] for c in expected_init_centers]
expected_init_intensities = [
    30, 30, 30,  41,  41,
    59, 59, 59,  41,  41,
    11, 11, 11, 100, 100,
    89, 89, 89, 100, 100,
    70, 70, 70, 100, 100
]
expected_init_depths = [
      1,   1,   1, 200, 200,
     10,  10,  10, 200, 200,
     50,  50,  50, 250, 250,
    100, 100, 100, 250, 250,
    150, 150, 150, 250, 250,
]

simple2_init_centers = [ [4, 2], [13, 2], [22, 2] ]
simple2_init_intensities = [100, 0, 0]
simple2_init_depths = [100, 200, 200]

class TestSuperpixelExtraction(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestSuperpixelExtraction, self).__init__(*args, **kwargs)
        self.image = plt.imread("test_data/test_image.png")
        self.depth = plt.imread("test_data/test_depth.png")
        self.image2 = plt.imread("test_data/test_image2.png")
        self.depth2 = plt.imread("test_data/test_depth2.png")
        self.image3 = plt.imread("test_data/test_image3.png")
        self.depth3 = plt.imread("test_data/test_depth3.png")
        (w, h) = self.image.shape
        camera_parameters = {'fx': 1, 'fy': 1, 'cx': w/2, 'cy': h/2}
        weights={'Ns': 4, 'Nc': 100, 'Nd': 200}
        self.spExtractor = SuperpixelExtraction(self.image, self.depth,
            camera_parameters, sp_size=10)
    
    @unittest.skip("skip test_calc_distance")
    def test_calc_distance(self):
        superpixels = [SuperpixelSeed(1, 1, 10, 0, 0, 0, 0, 0, 0, 0, 100, 100, False, False, 1, 2)]
        self.spExtractor.image = self.image3
        self.spExtractor.depth = self.depth3
        (w, h) = self.image3.shape
        expected_distances = np.zeros((w, h, 1))
        for i in range(w):
            for j in range(h):
                expected_distances[i, j, 1] = (i - 1)**2 / 4.0 + (j - 1)**2 / 100.0 + (self.image3[i, j] - 10)**2 / 200.0
        
        actual_distances = self.spExtractor.calc_distances(superpixels)
        self.assertEqual(actual_distances, expected_distances)

    @unittest.skip("skip test_extract_superpixels")
    def test_extract_superpixels(self):
        pass

    @unittest.skip("skip test_init_seeds")
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

            expected_init_centers.remove(superpixel.center)
            self.assertEqual(superpixel.mean_depth, expected_init_depths[index],
                "depth not initialized properly")
            self.assertEqual(superpixel.mean_intensity, expected_init_intensities[index],
                "intensity not initialized properly")
            self.assertEqual(superpixel.size, 0, "size should be init to 0")
            passed[index] = True

        for elem in passed:
            self.assertTrue(elem, "not all superpixel centers found")

    @unittest.skip("skip test_assign_pixels")
    def test_assign_pixels(self):
        image = self.image2
        depth = self.depth2

        superpixels = []
        for i in range(len(expected_init_centers)):
            superpixels.append(SuperpixelSeed(
                simple2_init_centers[i][0],
                simple2_init_centers[i][1],
                0, 0, 0, 0, 0, 0, 0, 0,
                simple2_init_depths[i],
                simple2_init_intensities[i],
                0, 0, 0, 0
            ))

        # recall Ns = 4, Nc = 100, Nd = 200
        expected_pixels = np.zeros(image.shape, dtype=np.uint8)
        expected_pixels[0:]

        pass

    @unittest.skip("skip test_update_seeds")
    def test_update_seeds(self):
        pass

    @unittest.skip("skip test_calc_norms")
    def test_calc_norms(self):
        pass

if __name__ == '__main__':
    unittest.main()