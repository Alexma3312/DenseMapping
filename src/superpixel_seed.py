"""Superpixel Seed"""


class SuperpixelSeed():
    def __init__(self, x, y, size, norm_x, norm_y, norm_z, position_x, position_y, position_z, view_cos, mean_depth, mean_intensity, fused, stable, min_eigen_value, max_eigen_value):
        """ Superpixel Seed Data Structure.
        Arguments:
            x, y: center of the superpixel cluster
            size: 
            norm_x, norm_y, norm_z: 
            posi_x, posi_y, posi_z: 
            view_cos: 
            mean_depth: the mean of all pixel depth within the cluster
            mean_intensity: the mean of all pixel intensity within the cluster
            fused: (bool) 
            stable: (bool)
            # For Debug
            min_eigen_value:
            max_eigen_value: 
        """
        self.x, self.y = x, y
        self.size = size
        self.norm_x, self.norm_y, self.norm_z = norm_x, norm_y, norm_z
        self.posi_x, self.posi_y, self.posi_z = position_x, position_y, position_z
        self.view_cos = view_cos
        self.mean_depth = mean_depth
        self.mean_intensity = mean_intensity
        self.fused = fused
        self.stable = stable

        # for debug
        self.min_eigen_value = min_eigen_value
        self.max_eigen_value = max_eigen_value
