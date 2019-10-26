"""Test superpixel seed."""

from surfel_element import SurfelElement
import unittest


class TestSurfelElement(unittest.TestCase):
    """Unit tests for Superpixel Seed."""

    def test_sure(self):
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
        pass


if __name__ == "__main__":
    unittest.main()
