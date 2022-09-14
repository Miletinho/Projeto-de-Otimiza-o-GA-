import numpy as np
from typing import Final

_A:               Final[int]   = 20                          # Ackley function A parameter  
_B:               Final[float] = 0.2                         # Ackley function B parameter  
_C:               Final[float] = 2*np.pi                     # 2Pi: Ackley function C parameter  
_D:               Final[int]   = 30                          # Number of dimensions of each individual
_START:           Final[int]   = -15                         # Start of X range
_END:             Final[int]   = 15                          # End of X range
_MP:              Final[float] = 0.005                       # Mutation Probability (GA)
_MP_ES:           Final[float] = 0.005                       # Mutation Probability (ES)
_CP:              Final[float] = 0.9                         # Crossover Probability
_POP_SIZE:        Final[int]   = 2000                        # Population size
_MAX_GENERATIONS: Final[int]   = 10000
_EPSILON:         Final[float] = 1/5                         # Epsilon (update rule)
_GLOBAL_TAU:      Final[float] = 1/np.sqrt(2*_D)            
_LOCAL_TAU:       Final[float] = 1/np.sqrt(2*np.sqrt(_D))
_ROUND:           Final[int]   = 4
_EXECUTIONS:      Final[int]   = 1
_FITNESS:         Final[int]   = 0.001


constants = {
    "1": [_D, _START, _END, _FITNESS],
    "2": [2, -100, 100, 0.0],
    "3": [20, -5.12, 5.12, _FITNESS],
}

funcs = {
    "1": "Ackley",
    "2": "Schaffer",
    "3": "Rastrigin"
}