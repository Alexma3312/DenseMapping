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

    def test_is_fuseable(self):
        """Tests is_fuseable"""
        p = np.array([1, 2, 1])
        camera_parameters = {'fx': 100, 'fy': 100,
                             'cx': 0, 'cy': 0}
        surf_local = SurfelElement(p[0], p[1], p[2], 1, 1, 1, 0, 0, 0, 0, 0)
        surf_new = SurfelElement(1, 2, 1.2, 1, 1, 1, 0, 0, 0, 0, 0)
        self.assertEqual(True, surf_local.is_fuseable(surf_new))
        surf_new = SurfelElement(1, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0)
        self.assertEqual(False, surf_local.is_fuseable(surf_new))
        

    def test_fuse_surfel(self):
        """Tests fuse_surfel"""
        p = np.array([1, 2, 1])
        surf_local = SurfelElement(1, 2, 3, 4, 5, 6, 2, 255, 8, 8, 10)
        surf_new = SurfelElement(15, 16, 17, 18, 19, 20, 3, 200, 2, 1, 5)
        expect_px, expect_py, expect_pz = 3.8, 4.8, 5.8
        expect_nx, expect_ny, expect_nz = 6.8, 7.8, 8.8
        expect_size = 2
        expect_color = 200
        expect_weight = 10
        expect_update_times = 9
        expect_last_update = 5
        surf_local.fuse_surfel(surf_new)
        # S_p
        self.assertEqual(expect_px, surf_local.px)
        self.assertEqual(expect_py, surf_local.py)
        self.assertEqual(expect_pz, surf_local.pz)
        # S_n
        self.assertEqual(expect_nx, surf_local.nx)
        self.assertEqual(expect_ny, surf_local.ny)
        self.assertEqual(expect_nz, surf_local.nz)
        # others
        self.assertEqual(expect_size, surf_local.size)
        self.assertEqual(expect_color, surf_local.color)
        self.assertEqual(expect_weight, surf_local.weight)
        self.assertEqual(expect_update_times, surf_local.update_times)
        self.assertEqual(expect_last_update, surf_local.last_update)

        

if __name__ == "__main__":
    unittest.main()
