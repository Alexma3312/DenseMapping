"""Test superpixel seed."""

from superpixel_seed import SuperpixelSeed
import unittest


class TestSuperpixelSeed(unittest.TestCase):
    """Unit tests for Superpixel Seed."""

    def test_superpixel_seed(self):
        """Test superpixel seed."""
        # Parameters to initialize superpixel seed.
        x, y = 0.0, 0.0
        size = 0.0
        norm_x, norm_y, norm_z = 0.0, 0.0, 0.0
        position_x, position_y, position_z = 0.0, 0.0, 0.0
        view_cos = 0.0
        mean_depth = 0.0
        mean_intensity = 0.0
        fused = True
        stable = True
        min_eigen_value = 0.0
        max_eigen_value = 0.0
        superpixel_seed = SuperpixelSeed(x, y, size, norm_x, norm_y, norm_z, position_x, position_y,
                                         position_z, view_cos, mean_depth, mean_intensity, fused, stable, min_eigen_value, max_eigen_value)
        self.assertEqual((x, y, size), (superpixel_seed.x,
                                        superpixel_seed.y, superpixel_seed.size))
        self.assertEqual((norm_x, norm_y, norm_z), (superpixel_seed.norm_x,
                                                    superpixel_seed.norm_y, superpixel_seed.norm_z))
        self.assertEqual((position_x, position_y, position_z),
                         (superpixel_seed.posi_x, superpixel_seed.posi_y, superpixel_seed.posi_z))
        self.assertEqual((view_cos, mean_depth, mean_intensity), (superpixel_seed.view_cos,
                                                                  superpixel_seed.mean_depth, superpixel_seed.mean_intensity))
        self.assertEqual((fused, stable, min_eigen_value, max_eigen_value), (superpixel_seed.fused,
                                                                             superpixel_seed.stable, superpixel_seed.min_eigen_value, superpixel_seed.max_eigen_value))


if __name__ == "__main__":
    unittest.main()
