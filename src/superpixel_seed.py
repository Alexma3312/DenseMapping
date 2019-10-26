"""Superpixel Seed"""


class SuperpixelSeed():
    def __init__(self, x, y, size, norm_x, norm_y, norm_z, position_x, position_y, position_z, view_cos, mean_depth, mean_intensity, fused, stable, min_eigen_value, max_eigen_value):
        """ Superpixel Seed Data Structure.
        Arguments:
            x, y: (float) 
            size: (float)
            norm_x, norm_y, norm_z: (float)
            posi_x, posi_y, posi_z: (float)
            view_cos: (float)
            mean_depth: (float)
            mean_intensity: (float) 
            fused: (bool) 
            stable: (bool)
            # For Debug
            min_eigen_value:(float)
            max_eigen_value: (float)
        """
        self.x, self.y = x, y
        self.size = size
        self.norm_x, norm_y, norm_z = norm_x, norm_y, norm_z
        self.posi_x, posi_y, posi_z = position_x, position_y, position_z
        self.view_cos = view_cos
        self.mean_depth = mean_depth
        self.mean_intensity = mean_intensity
        self.fused = fused
        self.stable = stable

        # for debug
        self.min_eigen_value = min_eigen_value
        self.max_eigen_value = max_eigen_value
