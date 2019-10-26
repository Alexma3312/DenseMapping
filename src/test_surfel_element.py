"""Test superpixel seed."""

from surfel_element import SurfelElement
import unittest


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


if __name__ == "__main__":
    unittest.main()
