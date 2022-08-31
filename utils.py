import numpy as np
from typing import Final

_A:               Final[int]   = 20      # Ackley function A parameter  
_B:               Final[float] = 0.2     # Ackley function B parameter  
_C:               Final[float] = 2*np.pi # 2Pi: Ackley function C parameter  
_D:               Final[int]   = 30      # Number of dimensions of each individual
_START:           Final[int]   = -15     # Start of X range
_END:             Final[int]   = 15      # End of X range
_MP:              Final[float] = 0.3     # Mutation Probability
_CP:              Final[float] = 0.9     # Crossover Probability
_POP_SIZE:        Final[int]   = 1000    # Population size
_MAX_GENERATIONS: Final[int]   = 10000