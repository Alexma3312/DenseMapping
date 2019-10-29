"""
Extract superpixels
"""

import numpy as np

from superpixel_seed import SuperpixelSeed
import numpy as np
from typing import Iterable, List


import numpy as np


class SuperpixelExtraction():

    def __init__(self, image, depth, camera_parameters,
                 weights={'Ns': 4, 'Nc': 10, 'Nd': 0.05},
                 sp_size=8):

        self.image = image
        self.depth = depth
        (self.im_height, self.im_width) = image.shape
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
                + (1.0 / self.depth - 1.0 / superpixel.mean_depth)**2 / self.Nd
        return distances

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
                superpixels.append(SuperpixelSeed(
                    col, row, 0, 0, 0, 0, 0, 0, 0, 0,
                    self.depth[row, col], self.image[row,
                                                     col], False, False, 0, 0
                ))

        return superpixels

    def assign_pixels(self, superpixels: Iterable[SuperpixelSeed]):
        """Assigns each pixel in an image to a superpixel seed
        Member dependencies:
            image, depth
        Arguments:
            superpixels:    list of SuperpixelSeed
        Returns:
            superpixel_idx:    nxm array of indices, where nxm is the size of the image and
                each element in the array represents the index of the superpixel
                which that pixel in the image is assigned to
        """
        print("first", self.calc_distances(superpixels)[:, :, 0])
        print("second", self.calc_distances(superpixels)[:, :, 1])
        print("third", self.calc_distances(superpixels)[:, :, 2])
        superpixel_idx = np.argmin(self.calc_distances(superpixels), axis=-1)
        return superpixel_idx

    def update_seeds(self, pixels, superpixels: Iterable[SuperpixelSeed]) -> List[SuperpixelSeed]:
        """Updates the locations, intensities, depths, and sizes of the
            superpixel seeds
        Member dependencies:
            image, depth
        Arguments:
            pixels: array of indices (nxm) which assigns each pixel to the index in
                `superpixels` of the superpixel under which this pixel falls
            superpixels:    list of SuperpixelSeed
        Returns:
            superpixels:    list of SuperpixelSeed with updated positions
        """
        import time
        t = time.time()
        for i, sp in enumerate(superpixels):
            mask = pixels != i
            # x/y
            [col, row] = np.meshgrid(
                np.arange(self.im_width), np.arange(self.im_height))
            sp.y = np.ma.array(row, mask=mask).mean()
            sp.x = np.ma.array(col, mask=mask).mean()
            # intensity/depth
            sp.mean_intensity = np.ma.array(self.image, mask=mask).mean()
            sp.mean_depth = np.ma.array(self.depth, mask=mask).mean()
            # size
            maxDist = 0
            valid_rows = np.ma.array(row, mask=mask).compressed()
            valid_cols = np.ma.array(col, mask=mask).compressed()
            xs = valid_cols - sp.x
            ys = valid_rows - sp.y
            dists2 = np.square(xs) + np.square(ys)
            sp.size = np.sqrt(np.max(dists2))
        print("updated seeds in {:0.3f}s".format(time.time() - t))
        return superpixels

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
    def calculate_spaces(self):
        """Recover the 3D point with the depth information for each pixel and store in space map.
            x, y,z: 3d point coordinate
            u: horizontal pixel coordinate
            v: vertical pixel coordinate
            depth: pixel depth information
                x = (u - self.cx) / self.fx * depth
                y = (v - self.cy) / self.fy * depth
                z = depth
        Member dependencies:
            img_width, img_height: image width and image height
            fx, fy,cx,cy: camera intrinsic matrix
        Returns:
            space_map: NxMx3 array of back projected 3D points
        """
        [col, row] = np.meshgrid(
            np.arange(self.im_width), np.arange(self.im_height))
        # Generate the z value matrix
        depth = np.copy(self.depth)
        # (x-cx)/fx*depth
        col = np.multiply((col - self.cx)/self.fx, self.depth)
        # (y-cy)/fy*depth
        row = np.multiply((row - self.cy)/self.fy, self.depth)

        col = np.expand_dims(col, axis=2)
        row = np.expand_dims(row, axis=2)
        depth = np.expand_dims(depth, axis=2)
        space_map = np.concatenate((row, col, depth), axis=2)
        return space_map

    def calculate_pixels_norms(self, space_map, MAX_ANGLE_COS=0.1):
        """Calculate the single pixel normalized norm along x,y,z for all pixels
        Arguments:
            space_map: NxMx3 array of 3D points (x,y,z)
        Returns:
            norm_map: (N-1)x(M-1)x3 array of normalized norm along x,y,z axes
        """
        # filter pixel with bad depth
        space_map_mask = space_map[:, :, -1] < 0.1
        mask = np.add(np.add(
            space_map_mask[:-1, :-1], space_map_mask[:-1, :-1]), space_map_mask[1:, :-1])

        my = space_map[:-1, :-1, :]
        right = space_map[:-1, 1, :] - my
        down = space_map[1:, :-1, :] - my

        # filter array with mask
        my = np.ma.array(my, mask=mask)
        right = np.ma.array(right, mask=mask)
        down = np.ma.array(down, mask=mask)

        my_x, my_y, my_z = my[:, :, 0], my[:, :, 1], my[:, :, 2]
        right_x, right_y, right_z = right[:,
                                          :, 0], right[:, :, 1], right[:, :, 2]
        down_x, down_y, down_z = down[:, :, 0], down[:, :, 1], down[:, :, 2]

        norm_x = np.ma.multiply(right_y, down_z) - \
            np.ma.multiply(right_z, down_y)
        norm_y = np.ma.multiply(right_z, down_x) - \
            np.ma.multiply(right_x, down_z)
        norm_z = np.ma.multiply(right_x, down_y) - \
            np.ma.multiply(right_y, down_x)

        norm_length = np.ma.sqrt(np.ma.multiply(
            norm_x, norm_x) + np.ma.multiply(norm_y, norm_y)+np.ma.multiply(norm_z, norm_z))

        number_of_valid_pixels = np.ma.count(norm_length)
        if number_of_valid_pixels == 0:
            return None

        norm_x = np.ma.divide(norm_x, norm_length)
        norm_y = np.ma.divide(norm_y, norm_length)
        norm_z = np.ma.divide(norm_z, norm_length)

        view_angle = np.ma.divide(np.ma.multiply(norm_x, my_x) + np.ma.multiply(norm_y, my_y)+np.ma.multiply(
            norm_z, my_z), np.ma.sqrt(np.ma.multiply(my_x, my_x) + np.ma.multiply(my_y, my_y)+np.ma.multiply(my_z, my_z)))

        angle_mask = view_angle <= -MAX_ANGLE_COS or view_angle >= MAX_ANGLE_COS

        norm_x = np.ma.expand_dims(
            np.ma.array(norm_x, mask=angle_mask), axis=2)
        norm_y = np.ma.expand_dims(
            np.ma.array(norm_y, mask=angle_mask), axis=2)
        norm_z = np.ma.expand_dims(
            np.ma.array(norm_z, mask=angle_mask), axis=2)
        norm_map = np.ma.concatenate((norm_x, norm_y, norm_z), axis=2)
        return norm_map

    # ****************************************************************
    # Sub functions for calculate_sp_depth_norms
    # ****************************************************************

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

    def initial_superpixel_cluster(self, superpixel_center,superpixel_seed_index, pixels, space_map, norm_map):
        """ Generate superpixel pixels
        Member dependencies:
            depth: nxmx1 depth image
        Arguments:
            superpixel_center: (x,y)
            superpixel_seed_index: the index of the current SuperpixelSeed in the list
            pixels: array of indices (nxm) which assigns each pixel to the index in
                `superpixels` of the superpixel under which this pixel falls
            space_map: NxMx3 array of 3D points (x,y,z)
            norm_map: (N-1)x(M-1)x3 array of normalized norm along x,y,z axes
        Returns:
            pixel_depths: 1xNx1 array, N is the number of valid pixel within current superpixel seed    
            pixel_norms: 1xNx3 array, N is the number of valid pixel within current superpixel seed
            pixel_positions: 1xNx3 array, N is the number of valid pixel within current superpixel seed
            max_dist: This is a distance in the image coordinate. The maximum distance of the border towards the center. 
            valid_depth_num: number of pixels with valid depths within a surfel
        """
        # Reshape depth from (3,3,1) into (3,3)
        depth = self.depth.reshape(3,3) 
        mask1 = pixels!=superpixel_seed_index
        mask2 = depth<=0.05
        mask = np.add(mask1,mask2)

        [col, row] = np.meshgrid(
            np.arange(self.im_width), np.arange(self.im_height))
        col = np.ma.array(col,mask=mask) - superpixel_center[0]
        row = np.ma.array(row,mask=mask) - superpixel_center[1]

        diff = np.ma.multiply(col,col)+ np.ma.multiply(row,row)
        max_dist = np.max(diff)

        pixel_depths = depth[~mask].reshape(1,-1)
        valid_depth_num = pixel_depths.shape[1]
        pixel_positions = space_map[~mask].reshape(1,valid_depth_num,3)
        pixel_norms = norm_map[~mask[:-1,:-1]].reshape(1,valid_depth_num,3)
        return pixel_depths, pixel_norms, pixel_positions, max_dist, valid_depth_num


    def huber_filter(self, mean_depth, pixel_depth, pixel_positions, HUBER_RANGE= 0.4):
        """ Use Huber Kernel filter outliers.
        Arguments:
            mean_depth: mean depth of current superpoint seed 
            pixel_depth: 1xNx1 array, N is the number of valid pixel within current superpixel seed
            pixel_positions: 1xNx3 array, N is the number of valid pixel within current superpixel seed 
        Returns:
            norm_x, norm_y, norm_z: normal along x,y,z axes
            inlier_num: the number of valid points
            pixel_inlier_positions: 1xNx3 array, N is the number of valid pixel within current superpixel seed
        """
        pass

    def calc_view_cos(self, norm_x, norm_y, norm_z, avg_x, avg_y, avg_z):
        """
        Arguments:
            norm_x, norm_y, norm_z: normal along x,y,z axes
            avg_x, avg_y, avg_z: average point (x,y,z), the center of the surfel
        Returns:
            new_norm_x, new_norm_y, new_norm_z: surfel normal along x,y,z axes
            view_cos: cosine value between surfel normal and the surfel center vector
        """
        pass

    def update_superpixel_cluster_with_huber(self, pixel_depths, pixel_norms, pixel_positions):
        """
        Arguments:
            pixel_depths: 1xNx1 array, N is the number of valid pixel within current superpixel seed    
            pixel_norms: 1xNx3 array, N is the number of valid pixel within current superpixel seed
            pixel_positions: 1xNx3 array, N is the number of valid pixel within current superpixel seed
        Returns:
            norm_x,norm_y,norm_z: normal along x,y,z axes
            avg_x,avg_y,avg_z: average point (x,y,z), the center of the surfel
            view_cos: cosine value between surfel normal and the surfel center vector
        """

        pass



    def calculate_sp_depth_norms(self, pixels, superpixel_seeds, space_map, norm_map):
        """Calculate surfel vector from superpixel seeds.
        Arguments:
            pixels: array of indices (nxm) which assigns each pixel to the index in
                `superpixels` of the superpixel under which this pixel falls
            superpixel_seeds: list of SuperpixelSeed
            space_map: NxMx3 array of 3D points (x,y,z)
            norm_map: NxMx3 array of normalized norm along x,y,z axes
        Returns:
            new_superpixel_seeds: list of update SuperpixelSeed

        """

        def sp_update(i, superpixel_seed):
            # Initialize superpixel cluster
            # self.initial_superpixel_cluster

            # Filter superpixel seed with valid number of depth value
            # if (valid_depth_num < 16):
            #     continue

            # Huber Range Filter
            # self.huber_filter
            # if (inlier_num / pixel_depth.size() < 0.8):
            #     continue

            # Update superpixel cluster with huber
            # self.update_superpixel_cluster_with_huber

            # Create new superpixel_seed
            # superpixel_seed = 
            pass

        # }
        new_superpixel_seeds = [ sp_update(i, superpixel_seed) for i, superpixel_seed in enumerate(superpixel_seeds)]

        return new_superpixel_seeds


if __name__ == "__main__":
    pass
