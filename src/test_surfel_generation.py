"""Test Surfel Generation."""
import unittest

from surfel_generation import SurfelGeneration
from superpixel_seed import SuperpixelSeed
import numpy as np
from mock import patch

camera_parameters = {'fx': 1, 'fy': 1, 'cx': 0, 'cy': 0}
MAX_ANGLE_COS = 0.1
VERBOSE = False

surfel_generation_instance = SurfelGeneration(camera_parameters,
                                              MAX_ANGLE_COS=MAX_ANGLE_COS,
                                              VERBOSE=VERBOSE)


class TestSurfelGeneration(unittest.TestCase):

    def test_init(self):
        sg = SurfelGeneration(camera_parameters)
        self.assertEqual(len(sg.all_surfels), 0)

    def test_create_surfels(self):
        """Test initialize Surfels from superpixels."""
        # initialize_surfels
        x, y = 1, 1
        size = 2
        norm_x, norm_y, norm_z = [8, 9, 0]
        position_x, position_y, position_z = [5, 6, 7]
        view_cos = 0.1
        mean_depth = 5
        mean_intensity = 100
        fused = False
        stable = True
        min_eigen_value = 10
        max_eigen_value = 1
        superpixel_1 = SuperpixelSeed(x, y, size, norm_x, norm_y, norm_z, position_x, position_y, position_z,
                                      view_cos, mean_depth, mean_intensity, fused, stable, min_eigen_value, max_eigen_value)
        superpixel_2 = SuperpixelSeed(x, y, size, norm_x+10, norm_y+10, norm_z+10, position_x+10, position_y+10, position_z+10,
                                      view_cos, mean_depth, mean_intensity, fused, stable, min_eigen_value, max_eigen_value)
        superpixels = [superpixel_1, superpixel_2]

        frame_index = 99
        surfels = surfel_generation_instance.create_surfels(
            frame_index, superpixels, np.eye(4))
        surfel_1 = surfels[0]
        surfel_2 = surfels[1]
        # Surfel 1 
        self.assertEqual((5, 6, 7), (surfel_1.px,
                                     surfel_1.py, surfel_1.pz))
        self.assertEqual((8, 9, 0), (surfel_1.nx,
                                     surfel_1.ny, surfel_1.nz))
        self.assertEqual((100, 100, 1/25, 1, 99), (surfel_1.size,
                                                   surfel_1.color, surfel_1.weight, surfel_1.update_times, surfel_1.last_update))
        # Surfel 2
        self.assertEqual((15, 16, 17), (surfel_2.px,
                                     surfel_2.py, surfel_2.pz))
        self.assertEqual((18, 19, 10), (surfel_2.nx,
                                     surfel_2.ny, surfel_2.nz))
        self.assertEqual((100, 100, 1/25, 1, 99), (surfel_2.size,
                                                   surfel_2.color, surfel_2.weight, surfel_2.update_times, surfel_2.last_update))

    @unittest.skip("skip test_update_surfels")
    def test_update_surfels(self):
        "Test update surfels"
        sg = surfel_generation_instance
        sp = SuperpixelSeed(2, 0, 1,
                            0, 0, 1,
                            2, 0, 1, 1, 1, 0, 0, 0, 0, 0)
        pose0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                          [0, 0, 1, 0], [0, 0, 0, 1]])
        sg.update_surfels(0, [sp], pose0)
        # identical surfel should be fused
        self.assertEqual(len(sg.all_surfels), 1)
        sg.update_surfels(1, [sp], pose0)
        # surfel in same location and norm should be fused
        self.assertEqual(len(sg.all_surfels), 1)
        sp.x = -1
        sp.posi_x = -1
        sp.posi_z = 2
        sp.norm_x = 1
        sp.norm_z = 0
        pose1 = np.array([[0, 0, -1, 0], [0, 1, 0, 0],
                          [1, 0, 0, 0], [0, 0, 0, 1]])
        sg.update_surfels(2, [sp], pose1)
        self.assertEqual(len(sg.all_surfels), 1)
        # surfel in same location but different norm should NOT be fused
        sp.norm_x = 0
        sp.norm_z = 1
        sg.update_surfels(3, [sp], pose1)
        self.assertEqual(len(sg.all_surfels), 2)
        # surfel in different location and different norm should NOT be fused
        sg.update_surfels(4, [sp], pose0)
        self.assertEqual(len(sg.all_surfels), 3)


if __name__ == '__main__':
    unittest.main()
