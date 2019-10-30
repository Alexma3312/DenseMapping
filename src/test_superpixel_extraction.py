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

simple2_init_centers = [[4, 2], [13, 2], [22, 2]]
simple2_init_intensities = [100/255, 0, 0]
simple2_init_depths = [100/255, 200/255, 200/255]
simple2_superpixels = []
for i in range(len(simple2_init_centers)):
    simple2_superpixels.append(SuperpixelSeed(
        simple2_init_centers[i][1],
        simple2_init_centers[i][0],
        0, 0, 0, 0, 0, 0, 0, 0,
        simple2_init_depths[i],
        simple2_init_intensities[i],
        False, False, 0, 0
    ))

# recall Ns = 4, Nc = 100, Nd = 200
# I am 99.9% sure this is correct - Gerry
simple2_expected_pixels = np.zeros((27, 5), dtype=np.uint8)
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

    def test_calc_distance(self):
        superpixels = [SuperpixelSeed(
            1, 1, 10, 0, 0, 0, 0, 0, 0, 0, 100, 100, False, False, 1, 2)]
        self.spExtractor.image = self.image3
        self.spExtractor.depth = self.depth3
        (self.spExtractor.im_height,
         self.spExtractor.im_width) = self.spExtractor.image.shape
        (w, h) = self.image3.shape
        expected_distances = np.zeros((w, h, 1))
        for i in range(w):
            for j in range(h):
                expected_distances[i, j, 0] = ((i - 1)**2 + (j - 1)**2) / 4.0 \
                    + (self.image3[i, j] - 100)**2 / 100.0 \
                    + (1.0 / self.depth3[i, j] - 1.0 / 100)**2 / 200.0

        actual_distances = self.spExtractor.calc_distances(superpixels)
        self.assertEqual(actual_distances.any(), expected_distances.any())

    @unittest.skip("skip test_extract_superpixels")
    def test_extract_superpixels(self):
        self.spExtractor.image = self.image2
        self.spExtractor.depth = self.depth2
        (self.spExtractor.im_height,
         self.spExtractor.im_width) = self.spExtractor.image.shape

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

        self.assertAlmostEqual(
            superpixels[0].size, 9.8488578018, "superpixel size wrong")
        self.assertAlmostEqual(
            superpixels[1].size, 8.0622577483, "superpixel size wrong")
        self.assertAlmostEqual(
            superpixels[2].size, 8.94427191, "superpixel size wrong")

    def test_init_seeds(self):
        """Tests initializing superpixel seeds
        """
        self.spExtractor.image = self.image
        self.spExtractor.depth = self.depth
        (self.spExtractor.im_height,
         self.spExtractor.im_width) = self.spExtractor.image.shape
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
        (self.spExtractor.im_height,
         self.spExtractor.im_width) = self.spExtractor.image.shape

        pixels = self.spExtractor.assign_pixels(simple2_superpixels)
        np.testing.assert_array_equal(pixels, simple2_expected_pixels)

    def test_update_seeds(self):
        self.spExtractor.image = self.image2
        self.spExtractor.depth = self.depth2
        (self.spExtractor.im_height,
         self.spExtractor.im_width) = self.spExtractor.image.shape

        superpixels = self.spExtractor.update_seeds(simple2_expected_pixels,
                                                    simple2_superpixels)

        self.assertEqual(superpixels[0].y, 4.5, "new superpixel 0 x incorrect")
        self.assertEqual(superpixels[0].x, 2, "new superpixel 0 y incorrect")
        self.assertEqual(superpixels[1].y, 13.5,
                         "new superpixel 1 x incorrect")
        self.assertEqual(superpixels[1].x, 2, "new superpixel 1 y incorrect")
        self.assertEqual(superpixels[2].y, 22, "new superpixel 2 x incorrect")
        self.assertEqual(superpixels[2].x, 2, "new superpixel 2 y incorrect")

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

        self.assertAlmostEqual(superpixels[0].size, 9.8488578018/2,
                               msg="new superpixel size wrong")
        self.assertAlmostEqual(superpixels[1].size, 8.0622577483/2,
                               msg="new superpixel size wrong")
        self.assertAlmostEqual(superpixels[2].size, 8.94427191/2,
                               msg="new superpixel size wrong")

    @unittest.skip("skip test_calc_norms")
    def test_calc_norms(self):
        pass

    @unittest.skip("skip test_calculate_spaces")
    def test_calculate_spaces(self):
        self.spExtractor.im_width, self.spExtractor.im_height = 3, 2
        self.spExtractor.depth = np.array([[2, 2, 2], [3, 3, 3]])
        self.spExtractor.cx = self.spExtractor.im_width/2
        self.spExtractor.cy = self.spExtractor.im_height/2
        expected_space_map = np.array(
            [[[-2, -3, 2], [-2, -1, 2], [-2, 1, 2]], [[0, -4.5, 3], [0, -1.5, 3], [0, 1.5, 3]]])
        actual_space_map = self.spExtractor.calculate_spaces()
        actual_shape = actual_space_map.shape
        self.assertEqual(np.array_equal(actual_space_map,
                                        expected_space_map), True, "Result is wrong ")

    @unittest.skip("skip test_calculate_pixels_norms")
    def test_calculate_pixels_norms(self):
        # Test for bad depths filter
        space_map = np.array([[[0, 0, 0], [0, 0, 1]], [[0, 0, -1], [1, 1, 1]]])
        actual_pixels_norms = self.spExtractor.calculate_pixels_norms(
            space_map)
        self.assertEqual(actual_pixels_norms, None,
                         "If no pixel is valid return None")
        # Test for bad view angles
        space_map = np.array([[[0, 0, 1], [-1, 0, 1]], [[1, 0, 1], [1, 1, 1]]])
        mask = np.array([[[True, True, True]]])
        expected_pixels_norms = np.array([[[0, 0, 0]]])
        actual_pixels_norms = self.spExtractor.calculate_pixels_norms(
            space_map)
        np.testing.assert_array_almost_equal(
            actual_pixels_norms, expected_pixels_norms, err_msg="Bad view angles test fail.")
        # Test for return result
        space_map = np.array([[[2, 2, 2], [2.1, 2.1, 2.1]], [
                             [1.9, 2.1, 1.9], [1, 1, 1]]])
        mask = np.array([[[False, False, False]]])
        expected_pixels_norms = np.array([[[-1/2**(1/2), 0, 1/2**(1/2)]]])
        expected_pixels_norms = np.ma.array(expected_pixels_norms, mask=mask)
        actual_pixels_norms = self.spExtractor.calculate_pixels_norms(
            space_map)
        np.testing.assert_array_almost_equal(
            actual_pixels_norms, expected_pixels_norms, err_msg="Return Result test fail.")

    @unittest.skip("skip test_get_huber_norm")
    def test_get_huber_norm(self):
        pass
    
    @unittest.skip("skip test_initial_superpixel_cluster")
    def test_initial_superpixel_cluster(self):
        superpixel_seed_index = 1
        pixels = np.zeros((3, 3), dtype=np.uint8)
        pixels[0:2, 0:2] = 1
        self.spExtractor.depth = np.array(
            [[[1], [2], [3]], [[0.01], [5], [6]], [[7], [8], [9]]])
        self.spExtractor.im_width = 3
        self.spExtractor.im_height = 3
        space_map = np.array([[[0, 0, 1], [0, 0, 2], [0, 0, 3]], [[0, 0, 0.1], [
                             0, 0, 5], [0, 0, 6]], [[0, 0, 7], [0, 0, 8], [0, 0, 9]]])
        norm_map = np.array([[[0, 0, 3], [0, 0, 4]], [[0, 0, 0.1], [0, 0, 5]]])

        expected_pixel_depth = np.array([[1], [2], [5]]).reshape(3,1)
        expected_pixel_norms = np.array([[0, 0, 3], [0, 0, 4], [0, 0, 5]]).reshape(3,3)
        expected_pixel_positions = np.array([[0, 0, 1], [0, 0, 2], [0, 0, 5]]).reshape(3,3)
        expected_max_dist = 2
        expected_valid_depth_num = 3

        pixel_depths, pixel_norms, pixel_positions, max_dist, valid_depth_num = self.spExtractor.initial_superpixel_cluster(
            (0, 0), 1, pixels, space_map, norm_map)
        np.testing.assert_array_almost_equal(
            pixel_depths, expected_pixel_depth, err_msg="pixel depth wrong")
        np.testing.assert_array_almost_equal(
            pixel_norms, expected_pixel_norms, err_msg="pixel norms wrong")
        np.testing.assert_array_almost_equal(
            pixel_positions, expected_pixel_positions, err_msg="pixel positions wrong")
        self.assertEqual(max_dist, expected_max_dist, "Max Distance is wrong ")
        self.assertEqual(valid_depth_num, expected_valid_depth_num,
                         "Valid depth num is wrong ")

    def test_huber_filter(self):
        mean_depth = []
        pixel_depth = []
        pixel_positions = []
        # self.huber_filter(mean_depth, pixel_depth, pixel_positions)

        expected_norm_x, expected_norm_y, expected_norm_z = 1,2,3
        expected_inlier_num = 1
        expected_pixel_inlier_positions = []

        pass

    def test_calc_view_cos(self):
        norm = np.array([2.,2.,1.])
        avg = np.array([1,0,0])
        expected_norm = np.array([2/3, 2/3, 1/3])
        expected_view_cos = 2/3
        newnorm, view_cos = self.spExtractor.calc_view_cos(norm, avg)
        np.testing.assert_almost_equal(expected_norm, newnorm, err_msg="norm doesn't match")
        np.testing.assert_almost_equal(expected_view_cos, view_cos, err_msg="view_cos doesn't match")
        
        newnorm, view_cos = self.spExtractor.calc_view_cos(norm, -avg)
        np.testing.assert_almost_equal(expected_norm, -newnorm, err_msg="norm doesn't match")
        np.testing.assert_almost_equal(expected_view_cos, view_cos, err_msg="view_cos doesn't match")

    @unittest.skip("skip test_calculate_sp_depth_norms")
    def test_calculate_sp_depth_norms(self):
        pass


if __name__ == '__main__':
    unittest.main()
