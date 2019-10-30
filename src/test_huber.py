import unittest

import numpy as np

from huber import calc_huber_norm

class TestHuber(unittest.TestCase):
    
    def test_huber_norm(self):
        pixelNorms = np.array([[1,0,0],[0,1,0],[1,0,0],[0.99,0.141,0]])
        centerLoc = np.array([1,2,0])
        pixelLocs = np.array([[1,3,0], [1,1,0], [1.1,5,-2], [1.1,-1, 2], [20,1,1], [-19,3,-1]])
        expected_norm = np.array([1,0,0])

        norm = calc_huber_norm(centerLoc, pixelLocs, pixelNorms, bias=0,
            huber_radius=8)
        
        np.testing.assert_array_almost_equal(norm, expected_norm, decimal=2)

if __name__ == "__main__":
    unittest.main()