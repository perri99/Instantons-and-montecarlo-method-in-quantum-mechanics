import numpy as np
from utils.configuration import Configuration
from typing import Tuple

class SchroedingerSolver:
    def __init__(
            self, 
            function_potential: function, 
            mass: float,
            configuration: Configuration,
              ):
        self.potential = function_potential
        self.mass = mass
        self.lattice = configuration
        self.step = (configuration[len(configuration)-1] - configuration[0]) / (len(configuration)-1)

    def solve(self):
        pass
