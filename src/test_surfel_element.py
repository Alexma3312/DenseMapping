"""Test superpixel seed."""

from surfel_element import SurfelElement
import unittest
from superpixel_seed import SuperpixelSeed
import numpy as np

class TestSurfelElement(unittest.TestCase):
    """Unit tests for Superpixel Seed."""

    def test_surfel_element(self):
        """Test superpixel seed."""
        # Parameters to initialize superpixel seed.
        px, py, pz = 0.0, 0.0, 0.0
        nx, ny, nz = 0.0, 0.0, 0.0
        size = 0.0
        color = 0.0
        weight = 0.0
        update_times = 0
        last_update = 0
        surfel_element = SurfelElement(
            px, py, pz, nx, ny, nz, size, color, weight, update_times, last_update)
        self.assertEqual((px, py, pz), (surfel_element.px,
                                        surfel_element.py, surfel_element.pz))
        self.assertEqual((nx, ny, nz), (surfel_element.nx,
                                        surfel_element.ny, surfel_element.nz))
        self.assertEqual((size, color, weight, update_times, last_update), (surfel_element.size,
                                                                            surfel_element.color, surfel_element.weight, surfel_element.update_times, surfel_element.last_update))

    def test_change_coordinates(self):
        """Test coordinate change"""
        n = np.array([1, 0, 0])
        p = np.array([0, 0, 1])
        Twc = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        surfel = SurfelElement(p[0], p[1], p[2], n[0], n[1], n[2], 1, 1, 0, 0, 0)
        surfel_c = surfel.change_coordinates(Twc)
        self.assertEqual(surfel_c.px, 1)
        self.assertEqual(surfel_c.py, 0)
        self.assertEqual(surfel_c.pz, 1)
        self.assertEqual(surfel_c.nx, 1)
        self.assertEqual(surfel_c.ny, 0)
        self.assertEqual(surfel_c.nz, 0)
        Twc = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) # CW 90 deg
        surfel_c = surfel.change_coordinates(Twc)
        self.assertEqual(surfel_c.px, 0)
        self.assertEqual(surfel_c.py, 0)
        self.assertEqual(surfel_c.pz, 1)
        self.assertEqual(surfel_c.nx, 0)
        self.assertEqual(surfel_c.ny, -1)
        self.assertEqual(surfel_c.nz, 0)

    def test_back_project(self):
        """Test back projection of surfel into superpixel"""
        p = np.array([1, 2, 1])
        camera_parameters = {'fx': 100, 'fy': 100,
                             'cx': 0, 'cy': 0}
        surf = SurfelElement(p[0], p[1], p[2], 0, 0, 0, 0, 0, 0, 0, 0)
        x, y = surf.back_project(camera_parameters)
        self.assertEqual(x, 100)
        self.assertEqual(y, 200)
        surf.pz = 2
        x, y = surf.back_project(camera_parameters)
        self.assertEqual(x, 50)
        self.assertEqual(y, 100)

if __name__ == "__main__":
    unittest.main()
