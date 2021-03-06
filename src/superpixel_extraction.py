"""
Extract superpixels
"""

import time
from typing import Iterable, List

import numpy as np

from huber import calc_huber_norm
from superpixel_seed import SuperpixelSeed


import numpy as np
from scipy.optimize import minimize


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

    def extract_superpixels(self, verbose=False, iterations=5) -> List[SuperpixelSeed]:
        """Extracts superpixels from an RGB image and depth image
        Member dependencies:
            image, depth
        Arguments:
            None
        Returns:
            superpixels: list of SuperpixelSeed
        """
        superpixels = self.init_seeds()
        for _ in range(iterations):
            superpixel_idx = self.assign_pixels(superpixels)
            superpixels = self.update_seeds(superpixel_idx, superpixels)
        superpixels = self.calc_norms(superpixel_idx, superpixels)

        return superpixels

    def init_seeds(self) -> List[SuperpixelSeed]:
        """Initializes the centers for the superpixels
        Member dependencies:
            image, depth:  depth image, sp_size
        Arguments:
            None
        Returns:
            superpixels:    list of SuperpixelSeed, has correct x, y, size.  mean_depth
                and mean_intensity are initialized to the value of the center pixel,
                and the remaining properties are initialized to 0.
        """
        superpixels = []

        for row in range(int(self.sp_size/2)-1, self.im_height, self.sp_size):
            for col in range(int(self.sp_size/2)-1, self.im_width, self.sp_size):
                superpixels.append(SuperpixelSeed(
                    col, row, self.sp_size/2*1.4142, 0, 0, 0, 0, 0, 0, 0,
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
        t = time.time()
        superpixel_idx = np.argmin(self.calc_distances(superpixels), axis=-1)
        print("assigned pixels in\t{:0.3f}s".format(time.time()-t))
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
        t = time.time()
        [col, row] = np.meshgrid(
            np.arange(self.im_width), np.arange(self.im_height))
        for i, sp in enumerate(superpixels):
            mask = pixels == i
            # x/y
            xs = col[mask]
            ys = row[mask]
            sp.y = np.mean(ys)
            sp.x = np.mean(xs)
            # intensity/depth
            sp.mean_intensity = np.mean(self.image[mask])
            sp.mean_depth = np.mean(self.depth[mask])
            # size
            xs = xs - sp.x
            ys = ys - sp.y
            sp.size = np.sqrt(np.max(np.square(xs) + np.square(ys)))
        print("updated seeds in\t{:0.3f}s".format(time.time() - t))
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
        t = time.time()
        space_map = self.calculate_spaces()
        norm_map = self.calculate_pixels_norms_for_loop(space_map)
        superpixels = self.calculate_sp_depth_norms(
            pixels, superpixels, space_map, norm_map)
        print("Calculate Norms in\t{:0.3f}s".format(time.time() - t))
        return superpixels

    # ****************************************************************
    # Sub functions for assigning pixels to superpixel centers
    # ****************************************************************
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
        distances = np.ones(
            (self.im_height, self.im_width, len(superpixels))) * 1e99
        for idx, superpixel in enumerate(superpixels):
            valid = (xx > (superpixel.x - superpixel.size*1.5)) & \
                    (xx < (superpixel.x + superpixel.size*1.5)) & \
                    (yy > (superpixel.y - superpixel.size*1.5)) & \
                    (yy < (superpixel.y + superpixel.size*1.5))
            distances[valid, idx] = \
                ((xx[valid] - superpixel.x)**2 + (yy[valid] - superpixel.y)**2) / self.Ns \
                + (self.image[valid] - superpixel.mean_intensity)**2 / self.Nc \
                + (1.0 / self.depth[valid] - 1.0 /
                   superpixel.mean_depth)**2 / self.Nd
        return distances

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
        space_map = np.concatenate((col, row, depth), axis=2)
        return space_map

    def calculate_pixels_norms_for_loop(self, space_map, MAX_ANGLE_COS=0.1):
        norm_map = np.empty((space_map.shape[0]-1,space_map.shape[1]-1,3))
        row_num, col_num,_ = space_map.shape
        for row_i in range(row_num-1):
            for col_i in range(col_num-1):
                my = space_map[row_i][col_i]
                right = space_map[row_i][col_i+1]
                down = space_map[row_i+1][col_i]
                if (my[2] < 0.1 or right[2] < 0.1 or down[2] < 0.1):
                    continue
                right -= my
                down -= my 
                norm_x = right[1] * down[2] - right[2] * down[1]
                norm_y = right[2] * down[0] - right[0] * down[2]
                norm_z = right[0] * down[1] - right[1] * down[0]
                norm = np.array([[norm_x],[norm_y],[norm_z]])
                norm_length = np.linalg.norm(norm,2)
                if norm_length == 0:
                    norm_map[row_i][col_i] = None
                    continue
                norm /= norm_length
                view_angle = np.dot(norm.T, my.reshape(3,1)) / np.linalg.norm(my,2)
                if(view_angle > -MAX_ANGLE_COS and view_angle < MAX_ANGLE_COS):
                    continue
                norm_map[row_i][col_i] = norm.reshape(3,)
        
        return norm_map
        

    def calculate_pixels_norms(self, space_map, MAX_ANGLE_COS=0.1):
        """Calculate the single pixel normalized norm along x,y,z for all pixels
        Arguments:
            space_map: NxMx3 array of 3D points (x,y,z)
        Returns:
            norm_map: (N-1)x(M-1)x3 array of normalized norm along x,y,z axes
        """
        # filter pixel with bad depth
        space_map_mask = space_map[:, :, -1] < 0.1
        mask = space_map_mask[:-1, :-1] | space_map_mask[:-
                                                         1, 1:] | space_map_mask[1:, :-1]
        mask = np.tile(np.expand_dims(mask, axis=2), (1, 1, 3))

        my = space_map[:-1, :-1, :]
        right = space_map[:-1, 1:, :] - my
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

        angle_mask = (
            view_angle > -MAX_ANGLE_COS) & (view_angle < MAX_ANGLE_COS)

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

    def get_huber_norm(self, pixel_inlier_norms, pixel_inlier_positions, bias=np.array([0]), huber_radius=0.4):
        """Re-estimate norm through Gauss-Newton iterations.
        Arguments:
            initial_norm: normal along x,y,z for Gauss-Newton initialization
            pixel_inlier_positions: 3d inlier points, Nx3 array
        Returns:
            norm: normal along x,y,z and the bias norm, vector 1x4 
        This based on Huber.py calc_huber_norm.
        """

        # Get center of 3D points
        center = np.mean(pixel_inlier_positions, axis=0)
        # initialize norm
        initNorm = np.mean(pixel_inlier_norms, axis=0)
        initNorm = initNorm / np.linalg.norm(initNorm)
        initNorm = np.concatenate((initNorm, bias), axis=0)

        return initNorm
        # optimize
        points = pixel_inlier_positions-center

        def f(norm):
            toRet = np.sum(
                np.square(
                    np.maximum(
                        np.minimum(
                            norm[:3].dot(points.transpose())+norm[3], huber_radius), -huber_radius)))
            return toRet

        def c(norm):
            return np.linalg.norm(norm[:3])-1
        constr = {'type': 'eq', 'fun': c}
        sol = minimize(f, initNorm, constraints=constr, tol=0.0001)
        norm = sol.x
        return norm

    def initial_superpixel_cluster(self, superpixel_center, superpixel_seed_index, pixels, space_map, norm_map):
        """ Generate superpixel pixels
        Filter pixels that has depth value <= 0.05 and initialize superpixel cluster
        Member dependencies:
            depth: nxmx1 depth image
            im_width: image width
            im_height: image height
        Arguments:
            superpixel_center: (x,y)
            superpixel_seed_index: the index of the current SuperpixelSeed in the list
            pixels: array of indices (nxm) which assigns each pixel to the index in
                `superpixels` of the superpixel under which this pixel falls
            space_map: NxMx3 array of 3D points (x,y,z)
            norm_map: (N-1)x(M-1)x3 array of normalized norm along x,y,z axes
        Returns:
            pixel_depths: Nx1 array, N is the number of valid pixel within current superpixel seed    
            pixel_norms: Nx3 array, N is the number of valid pixel within current superpixel seed
            pixel_positions: Nx3 array, N is the number of valid pixel within current superpixel seed
            max_dist: This is a distance in the image coordinate. The maximum distance of the border towards the center. 
            valid_depth_num: number of pixels with valid depths within a surfel
        """
        # Reshape depth from (3,3,1) into (3,3)
        shape = self.depth.shape[0:2]
        depth = self.depth.reshape(shape)
        mask1 = pixels != superpixel_seed_index
        mask2 = depth <= 0.05
        mask = mask1 | mask2

        [col, row] = np.meshgrid(
            np.arange(self.im_width), np.arange(self.im_height))
        col = np.ma.array(col, mask=mask) - superpixel_center[0]
        row = np.ma.array(row, mask=mask) - superpixel_center[1]

        diff = np.ma.multiply(col, col) + np.ma.multiply(row, row)
        max_dist = np.max(diff)

        # Reshape depth from mxn into (m-1)x(n-1)
        pixel_depths = depth[:-1, :-1][~mask[:-1, :-1]].reshape(-1, 1)
        valid_depth_num = pixel_depths.shape[0]
        pixel_positions = space_map[:-1, :-1][~mask[:-1, :-1]].reshape(-1, 3)
        pixel_norms = norm_map[~mask[:-1, :-1]].reshape(-1, 3)
        return pixel_depths, pixel_norms, pixel_positions, max_dist, valid_depth_num

    def huber_filter(self, mean_depth, pixel_depths, pixel_norms, pixel_positions, HUBER_RANGE=0.4):
        """ Use Huber Kernel filter outliers.
        Arguments:
            mean_depth: mean depth of current superpoint seed 
            pixel_depths: Nx1 array, N is the number of valid pixel within current superpixel seed
            pixel_norms: Nx3 array, N is the number of valid pixel within current superpixel seed
            pixel_positions: Nx3 array, N is the number of valid pixel within current superpixel seed 
        Returns:
            norm: normal along x,y,z axes (3-vector)
            inlier_num: the number of valid points
            pixel_inlier_positions: Nx3 array, N is the number of valid pixel within current superpixel seed
        """
        # Get huber residual
        residual = mean_depth - pixel_depths
        # Generate mask
        mask1 = residual < HUBER_RANGE
        mask2 = residual > -HUBER_RANGE
        mask = mask1 & mask2
        #
        inlier_num = np.sum(mask)
        mask = np.tile(mask, (1, 3))
        pixel_inlier_norms = pixel_norms[mask].reshape(inlier_num, 3)
        pixel_inlier_positions = pixel_positions[mask].reshape(inlier_num, 3)
        return pixel_inlier_norms, inlier_num, pixel_inlier_positions

    def calc_view_cos(self, norm, avg):
        """
        Arguments:
            norm: normal along x,y,z axes (3-vector)
            avg: average point (x,y,z), the center of the surfel (3-vector)
        Returns:
            new_norm_x, new_norm_y, new_norm_z: surfel normal along x,y,z axes
            view_cos: cosine value between surfel normal and the surfel center vector
        """
        norm = norm / np.linalg.norm(norm)
        view_cos = np.dot(norm, avg) / np.linalg.norm(avg)
        if (view_cos < 0):
            view_cos = -view_cos
            norm = -norm
        return norm, view_cos

    def update_superpixel_cluster_with_huber(self, pixel_inlier_norms, pixel_inlier_positions, superpixel_center, mean_depth):
        """
        Arguments:
            sum_norm: 1x3 array
            pixel_inlier_positions: Nx3 array, N is the number of valid pixel within current superpixel seed
            superpixel_center:(x,y)
            mean_depth: mean depth of current superpoint seed 
        Returns:
            norm: normal along x,y,z axes
            avg: average point (x,y,z), the center of the surfel
            view_cos: cosine value between surfel normal and the surfel center vector
            mean_depth: mean depth of current superpoint seed 
        """
        # Calculate Huber Normal, norm should include norm_x,norm_y,norm_z,and norm_b
        norm = self.get_huber_norm(pixel_inlier_norms, pixel_inlier_positions)
        # back project
        avg_x = (superpixel_center[0] - self.cx) / self.fx * mean_depth
        avg_y = (superpixel_center[1] - self.cy) / self.fy * mean_depth
        avg_z = mean_depth
        # make sure the avg_x, avg_y, and avg_z are one the surfel
        k = -1 * (avg_x * norm[0] + avg_y *
                  norm[1] + avg_z * norm[2]) - norm[3]
        # avg_x += k * norm[0]
        # avg_y += k * norm[1]
        # avg_z += k * norm[2]
        mean_depth = avg_z
        # Calculate view cos and update norm
        norm, view_cos = self.calc_view_cos(
            norm[0:3], np.array([avg_x, avg_y, avg_z]))
        return norm, (avg_x, avg_y, avg_z), view_cos, mean_depth

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
            superpixel_center = (superpixel_seed.x, superpixel_seed.y)
            pixel_depths, pixel_norms, pixel_positions, max_dist, valid_depth_num = self.initial_superpixel_cluster(
                superpixel_center, i, pixels, space_map, norm_map)

            # Filter superpixel seed with valid number of depth value
            if (valid_depth_num < 16):
                return

            # Huber Range Filter
            mean_depth = superpixel_seed.mean_depth
            pixel_inlier_norms, inlier_num, pixel_inlier_positions = self.huber_filter(
                mean_depth, pixel_depths, pixel_norms, pixel_positions)
            if ((inlier_num / pixel_depths.shape[0]) < 0.8):
                return

            # Update superpixel cluster with huber
            norm, center, view_cos, mean_depth = self.update_superpixel_cluster_with_huber(
                pixel_inlier_norms, pixel_inlier_positions, superpixel_center, mean_depth)

            # Create new superpixel_seed
            superpixel_seed.norm_x = norm[0]
            superpixel_seed.norm_y = norm[1]
            superpixel_seed.norm_z = norm[2]
            superpixel_seed.posi_x = center[0]
            superpixel_seed.posi_y = center[1]
            superpixel_seed.posi_z = center[2]
            superpixel_seed.mean_depth = mean_depth
            superpixel_seed.view_cos = view_cos
            superpixel_seed.size = np.sqrt(max_dist)

            return superpixel_seed

        new_superpixel_seeds = [sp_update(
            i, superpixel_seed) for i, superpixel_seed in enumerate(superpixel_seeds)]
        # Filter None in the list
        new_superpixel_seeds = list(filter(None.__ne__, new_superpixel_seeds))

        return new_superpixel_seeds


if __name__ == "__main__":
    pass
