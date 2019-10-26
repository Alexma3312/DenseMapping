"""
Extract superpixels
"""

from superpixel_seed import SuperpixelSeed

import numpy as np

class SuperpixelExtraction():
    
    def __init__(self, image, depth, camera_parameters,
        weights={'Ns': 4, 'Nc': 10, 'Nd': 0.05},
        sp_size=8):
        
        self.image = image
        self.depth = depth
        (self.im_width, self.im_height) = image.shape
        self.fx = camera_parameters['fx']
        self.fy = camera_parameters['fy']
        self.cx = camera_parameters['cx']
        self.cy = camera_parameters['cy']
        self.Ns = weights['Ns']
        self.Nc = weights['Nc']
        self.Nd = weights['Nd']
        self.sp_size = sp_size
    
    def calc_distances(self, superpixels):
        """calculate distances between each pixel to all the superpixels
        Member dependencies:
            image, depth
        Arguments:
            superpixels: list of SuperpixelSeed 
        Returns:
            distances: N*M*K numpy array (N*M is image size, K: number of SuperpixelSeed)
        """
        x = np.arange(self.im_width)
        y = np.arange(self.im_height)
        xx, yy = np.meshgrid(x, y)
        distances = np.zeros((self.im_height, self.im_width, len(superpixels)))
        for idx, superpixel in enumerate(superpixels):
            distances[:, :, idx] = ((xx - superpixel.x)**2 + (yy - superpixel.y)**2) / self.Ns \
            + (self.image - superpixel.mean_intensity)**2 / self.Nc \
            + (self.depth - superpixel.mean_depth)**2 / self.Nd
        return distances

    def extract_superpixels(self):
        """Extracts superpixels from an RGB image and depth image
        Member dependencies:
            image, depth
        Arguments:
            None
        Returns:
            superpixels: list of SuperpixelSeed
        """
        return None

    def init_seeds(self):
        """Initializes the centers for the superpixels
        Member dependencies:
            image, depth:  depth image, sp_size
        Arguments:
            None
        Returns:
            superpixels:    list of SuperpixelSeed, has correct x, y.  mean_depth
                and mean_intensity are initialized to the value of the center pixel,
                and the remaining properties are initialized to 0.
        """
        return None

    def assign_pixels(self, superpixels):
        """Assigns each pixel in an image to a superpixel seed
        Member dependencies:
            image, depth
        Arguments:
            superpixels:    list of SuperpixelSeed
        Returns:
            pixels:    nxm array of indices, where nxm is the size of the image and
                each element in the array represents the index of the superpixel
                which that pixel in the image is assigned to
        """
        return None

    def update_seeds(self, pixels, superpixels):
        """Updates the locations of the superpixel seeds
        Member dependencies:
            image, depth
        Arguments:
            pixels: array of indices (nxm) which assigns each pixel to the index in 
                `superpixels` of the superpixel under which this pixel falls
            superpixels:    list of SuperpixelSeed
        Returns:
            superpixels:    list of SuperpixelSeed with updated positions
        """
        return None

    def calc_norms(self, pixels, superpixels):
        """Calculates the norms
        Member dependencies:
            image, depth
        Arguments:
            pixels: array of indices (nxm) which assigns each pixel to the index in 
                `superpixels` of the superpixel under which this pixel falls
            superpixels:    list of SuperpixelSeed
        Returns:
            superpixels:    list of SuperpixelSeed with updated norms
        """
        return None

if __name__ == "__main__":
    pass