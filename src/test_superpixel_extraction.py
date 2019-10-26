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
expected_init_intensities = [i/255 for i in expected_init_intensities]
expected_init_depths = [
      1,   1,   1, 200, 200,
     10,  10,  10, 200, 200,
     50,  50,  50, 250, 250,
    100, 100, 100, 250, 250,
    150, 150, 150, 250, 250,
]
expected_init_depths = [d/255 for d in expected_init_depths]

simple2_init_centers = [ [4, 2], [13, 2], [22, 2] ]
simple2_init_intensities = [100/255, 0, 0]
simple2_init_depths = [100/255, 200/255, 200/255]
simple2_superpixels = []
for i in range(len(simple2_init_centers)):
    simple2_superpixels.append(SuperpixelSeed(
        simple2_init_centers[i][0],
        simple2_init_centers[i][1],
        0, 0, 0, 0, 0, 0, 0, 0,
        simple2_init_depths[i],
        simple2_init_intensities[i],
        False, False, 0, 0
    ))

# recall Ns = 4, Nc = 100, Nd = 200
# I am 99.9% sure this is correct - Gerry
simple2_expected_pixels = np.zeros((27,5), dtype=np.uint8)
simple2_expected_pixels[0:10, :] = 0
simple2_expected_pixels[10:18, :] = 1
simple2_expected_pixels[18:27, :] = 2

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
        weights = {'Ns': 4, 'Nc': 100, 'Nd': 200}
        self.spExtractor = SuperpixelExtraction(self.image, self.depth,
            camera_parameters, weights=weights, sp_size=10)
    
    @unittest.skip("skip test_calc_distance")
    def test_calc_distance(self):
        superpixels = [SuperpixelSeed(1, 1, 10, 0, 0, 0, 0, 0, 0, 0, 100, 100, False, False, 1, 2)]
        self.spExtractor.image = self.image3
        self.spExtractor.depth = self.depth3
        (self.spExtractor.im_width, self.spExtractor.im_height) = self.spExtractor.image.shape
        (w, h) = self.image3.shape
        expected_distances = np.zeros((w, h, 1))
        for i in range(w):
            for j in range(h):
                expected_distances[i, j, 1] = (i - 1)**2 / 4.0 + (j - 1)**2 / 100.0 + (self.image3[i, j] - 10)**2 / 200.0
        
        actual_distances = self.spExtractor.calc_distances(superpixels)
        self.assertEqual(actual_distances, expected_distances)

    @unittest.skip("skip test_extract_superpixels")
    def test_extract_superpixels(self):
        self.spExtractor.image = self.image2
        self.spExtractor.depth = self.depth2
        (self.spExtractor.im_width, self.spExtractor.im_height) = self.spExtractor.image.shape
        
        superpixels = self.spExtractor.extract_superpixels()
        
        self.assertEqual(superpixels[0].x, 4.5, "superpixel 0 x incorrect")
        self.assertEqual(superpixels[0].y, 2, "superpixel 0 y incorrect")
        self.assertEqual(superpixels[1].x, 13.5, "superpixel 1 x incorrect")
        self.assertEqual(superpixels[1].y, 2, "superpixel 1 y incorrect")
        self.assertEqual(superpixels[2].x, 22, "superpixel 2 x incorrect")
        self.assertEqual(superpixels[2].y, 2, "superpixel 2 y incorrect")

        self.assertEqual(superpixels[0].mean_intensity, 90/255,
            "superpixel intensity wrong")
        self.assertEqual(superpixels[1].mean_intensity, 0,
            "superpixel intensity wrong")
        self.assertEqual(superpixels[2].mean_intensity, 0,
            "superpixel intensity wrong")

        self.assertEqual(superpixels[0].mean_depth, 100/255,
            "superpixel depth wrong")
        self.assertEqual(superpixels[1].mean_depth, 100/255,
            "superpixel depth wrong")
        self.assertEqual(superpixels[2].mean_depth, 200/255,
            "superpixel depth wrong")

        self.assertAlmostEqual(superpixels[0].size, 9.8488578018, "superpixel size wrong")
        self.assertAlmostEqual(superpixels[1].size, 8.0622577483, "superpixel size wrong")
        self.assertAlmostEqual(superpixels[2].size, 8.94427191, "superpixel size wrong")

    def test_init_seeds(self):
        """Tests initializing superpixel seeds
        """
        self.spExtractor.image = self.image
        self.spExtractor.depth = self.depth
        (self.spExtractor.im_width, self.spExtractor.im_height) = self.spExtractor.image.shape
        self.spExtractor.sp_size = 20
        passed = [False]*25

        superpixels = self.spExtractor.init_seeds()

        self.assertEqual(len(superpixels), len(expected_init_centers),
            "didn't return the correct number of superpixels")

        for superpixel in superpixels:
            center = [superpixel.y, superpixel.x]
            try:
                index = expected_init_centers.index(center)
            except ValueError:
                self.fail("found unexpected superpixel center")
                continue
            self.assertFalse(passed[index], "duplicate superpixel center")
            
            print(center)
            print(index)
            print(superpixel.mean_depth)
            print(expected_init_depths[index])
            self.assertAlmostEqual(superpixel.mean_depth, expected_init_depths[index],
                msg="depth not initialized properly")
            self.assertAlmostEqual(superpixel.mean_intensity, expected_init_intensities[index],
                msg="intensity not initialized properly")
            self.assertEqual(superpixel.size, 0, "size should be init to 0")
            passed[index] = True

        for elem in passed:
            self.assertTrue(elem, "not all superpixel centers found")

    @unittest.skip("skip test_assign_pixels")
    def test_assign_pixels(self):
        self.spExtractor.image = self.image2
        self.spExtractor.depth = self.depth2
        (self.spExtractor.im_width, self.spExtractor.im_height) = self.spExtractor.image.shape

        pixels = self.spExtractor.assign_pixels(simple2_superpixels)
        np.testing.assert_array_equal(pixels, simple2_expected_pixels)

    def test_update_seeds(self):
        self.spExtractor.image = self.image2
        self.spExtractor.depth = self.depth2
        (self.spExtractor.im_width, self.spExtractor.im_height) = self.spExtractor.image.shape
        
        superpixels = self.spExtractor.update_seeds(simple2_expected_pixels,
            simple2_superpixels)
        
        self.assertEqual(superpixels[0].x, 4.5, "new superpixel 0 x incorrect")
        self.assertEqual(superpixels[0].y, 2, "new superpixel 0 y incorrect")
        self.assertEqual(superpixels[1].x, 13.5, "new superpixel 1 x incorrect")
        self.assertEqual(superpixels[1].y, 2, "new superpixel 1 y incorrect")
        self.assertEqual(superpixels[2].x, 22, "new superpixel 2 x incorrect")
        self.assertEqual(superpixels[2].y, 2, "new superpixel 2 y incorrect")

        self.assertAlmostEqual(superpixels[0].mean_intensity*255, 100,
            places=4, msg="new superpixel intensity wrong")
        self.assertAlmostEqual(superpixels[1].mean_intensity, 0,
            places=4, msg="new superpixel intensity wrong")
        self.assertAlmostEqual(superpixels[2].mean_intensity, 0,
            places=4, msg="new superpixel intensity wrong")

        self.assertAlmostEqual(superpixels[0].mean_depth*255, 100,
            places=4, msg="new superpixel depth wrong")
        self.assertAlmostEqual(superpixels[1].mean_depth*255, 112.5,
            places=4, msg="new superpixel depth wrong")
        self.assertAlmostEqual(superpixels[2].mean_depth*255, 200,
            places=4, msg="new superpixel depth wrong")

        self.assertAlmostEqual(superpixels[0].size, 9.8488578018,
            msg="new superpixel size wrong")
        self.assertAlmostEqual(superpixels[1].size, 8.0622577483,
            msg="new superpixel size wrong")
        self.assertAlmostEqual(superpixels[2].size, 8.94427191,
            msg="new superpixel size wrong")

    @unittest.skip("skip test_calc_norms")
    def test_calc_norms(self):
        pass


    def test_back_project(self):

        pass

    def test_calculate_spaces(self):
        self.spExtractor.im_width, self.spExtractor.im_height = 2,3
        self.spExtractor.depth = np.array([[2,2,2],[3,3,3]])
        expected_space_map = np.array([[[-2,-3,2],[-2,-1,2],[-2,1,2]],[[0,-4.5,3],[0,-1.5,3],[0,1.5,3]]])
        actual_space_map = self.spExtractor.calculate_spaces()
        actual_shape = actual_space_map.shape
        self.assertEqual(actual_space_map, expected_space_map, "Result is wrong ")

    def test_calculate_pixels_norms(self):
        space_map = np.array([[[0,0,0],[1,2,3]],[[-1,-2,-3],[1,1,1]]])
        expected_pixels_norm = np.array([[[-2,-3],[-2,-1],[-2,1]],[[0,-4.5],[0,-1.5],[0,1.5]]])
        space_map = np.array([[[0,0,0],[1,1,1]],[[1,2,3],[1,1,1]]])
        expected_pixels_norms = np.array([[1/6,-1/3,-1/6]])
        actual_pixels_norms = self.spExtractor.calculate_pixels_norms(space_map)
        self.assertEqual(actual_pixels_norms, expected_pixels_norms, "Result is wrong ")

    def test_get_huber_norm(self):
        pass

    def test_calculate_sp_depth_norms(self):
        pass

if __name__ == '__main__':
    unittest.main()