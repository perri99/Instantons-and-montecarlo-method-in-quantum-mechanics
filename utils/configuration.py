import numpy as np

class Configuration:
    def __init__(
            self,
            x_min: float,
            x_max: float,
            point_num: int
    ):
        '''
        ## Parameters
        x_min: float
        '''
        self.discretized_lattice = np.linspace(x_min, x_max, point_num)
        

    def get_lattice(self) -> np.ndarray:
        return self.discretized_lattice
    