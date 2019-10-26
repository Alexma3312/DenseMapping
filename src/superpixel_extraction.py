"""
Extract superpixels
"""

from superpixel_seed import SuperpixelSeed
import numpy as np
from typing import Iterable, List


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
        return None


    def extract_superpixels(self) -> List[SuperpixelSeed]:
        """Extracts superpixels from an RGB image and depth image
        Member dependencies:
            image, depth
        Arguments:
            None
        Returns:
            superpixels: list of SuperpixelSeed
        """
        return None

    def init_seeds(self) -> List[SuperpixelSeed]:
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
        superpixels = []

        for row in range(int(self.sp_size/2)-1, self.im_height, self.sp_size):
            for col in range(int(self.sp_size/2)-1, self.im_width, self.sp_size):
                superpixels.append( SuperpixelSeed(
                    col, row, 0, 0,0,0, 0,0,0, 0,
                    self.depth[row, col], self.image[row, col], False, False, 0,0
                ))

        return superpixels

    def assign_pixels(self, superpixels: Iterable[SuperpixelSeed]):
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

    def update_seeds(self, pixels, superpixels: Iterable[SuperpixelSeed]) -> List[SuperpixelSeed]:
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

    # ****************************************************************
    # Sub functions for Calculating the Norms
    # ****************************************************************

    def back_project(self, u, v, depth):
        """Back project a pixel to the 3D space.
        Arguments:
            u: horizontal pixel coordinate
            v: vertical pixel coordinate
            depth: pixel depth information
        Returns:
            x, y,z: 3d point coordinate
        """
        # x = (u - self.cx) / self.fx * depth
        # y = (v - self.cy) / self.fy * depth
        # z = depth
        # return x, y, z
        pass

    def calculate_spaces(self):
        """Recover the 3D point with the depth information for each pixel and store in space map.
        Returns:
            space_map: NxMx3 array of back projected 3D points
        """
        # space_map = np.zeros((self.im_height, self.im_width, 3))
        # for row_idx in range(self.im_height):
        #     for col_idx in range(self.im_width):
        #         x, y, z = self.depth[row_idx][col_idx]
        #         space_map[my_index] = np.array(x, y, z)
        pass

    def calculate_pixels_norms(self, space_map):
        """Calculate the single pixel normalized norm along x,y,z for all pixels
        Arguments:
            space_map: NxMx3 array of 3D points (x,y,z)
        Returns:
            norm_map: NxMx3 array of normalized norm along x,y,z axes
        """
        pass

    def get_huber_norm(self, gn_nx, gn_ny, gn_nz, gn_nb, pixel_inlier_positions):
        """Re-estimate norm through Gauss-Newton iterations.
        Arguments:
            gn_nx: normal along x for Gauss-Newton initialization
            gn_ny: normal along y for Gauss-Newton initialization
            gn_nz: normal along z for Gauss-Newton initialization 
            gn_nb: bias for Gauss-Newton initialization
            pixel_inlier_positions:
        Returns:
            norm_x: normal along x 
            norm_y: normal along y
            norm_z: normal along z 
            norm_b: bias
        """
        pass

    def calculate_sp_depth_norms(self, pixels, superpixels, space_map, norm_map):
        """Calculate surfel vector from superpixel seeds.
        Arguments:
            pixels: array of indices (nxm) which assigns each pixel to the index in
                `superpixels` of the superpixel under which this pixel falls
            superpixels: list of SuperpixelSeed
            space_map: NxMx3 array of 3D points (x,y,z)
            norm_map: NxMx3 array of normalized norm along x,y,z axes
        Returns:
            update_superpixels: list of update SuperpixelSeed

        """

        # Get pixel position and depth for pixel with valid depth.

        # Calculate Huber Norm

        # Back Project the superpixel cluster center to the 3d space

        # Generate View Cos

        pass


if __name__ == "__main__":
    pass
