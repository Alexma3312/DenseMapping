"""Surfel Element."""


class SurfelElement():
    def __init__(self, px, py, pz, nx, ny, nz, size, color, weight, update_times, last_update):
        """Surfel Element Data Structure.
        Arguments:
            px,py,pz:
            nx,ny,nz:
            size:
            color:
            weight:
            update_times: 
            last_update:
        """
        self.px, self.py, self.pz = px, py, pz
        self.nx, self.ny, self.nz = nx, ny, nz
        self.size = size
        self.color = color
        self.weight = weight
        self.update_times = update_times
        self.last_update = last_update
