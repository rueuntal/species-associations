"""Module with different models of dispersal"""

from __future__ import division
import numpy as np
import math
from numpy.random import *

def rand_disperse(init_loc, M1, M2):
    """Random dispersal with equal probability to any grid cell.
    
    Input: 
    init_loc - tuple (x0, y0) representing initial location
      Not functional in this model. 
    M1, M2 - length and width of the whole grid 
    
    """
    x = randint(0, M1)
    y = randint(0, M2)
    return (x, y)

def brownian(init_loc, M1, M2, u, bound = 'p'):
    """Brownian dispersal with probability u of dipersing to neighbouring cells 
    
    and (1-u) of staying in current cell.
    A disperser can move multiple steps until it settles down (or gets lost over the edge 
    if the boundary is permeable).
    Additional input: 
    u - probability of dispersing from the current cell
    bound - whether boundary is permeable ('p') or unpermeable ('np'). An individual is lost 
    once it goes beyond the permeable boundary, while it is bounced back if the boundary is unpermeable.
    
    """
    step_list = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    num_step = geometric(1 - u) - 1 # Number of steps to move
    if num_step:
        new_steps = [step_list[randint(0, 4)] for i in range(num_step)]
        if bound == 'np':
            sum_steps = sum(new_steps)
            new_loc = (init_loc[0] + sum_steps[0], init_loc[1] + sum_steps[1])
            if new_loc[0] in range(M1) and new_loc[1] in range(M2):
                return new_loc
            else: return None
        else:
            for step in new_steps:
                while ((init_loc[0] + step[0]) not in range (M1) or \
                       (init_loc[1] + step[1]) not in range (M2)):
                    step = step_list[randint(0, 4)]
                init_loc = (init_loc[0] + step[0], init_loc[1] + step[1])
            return init_loc
    else: return init_loc
            
def Gaussian(init_loc, M1, M2, sigma, bound = 'p'):
    """Dispersal following a Gaussian distribution centered at focal cell. 
    
    Additional input: 
    sigma - standard deviation of the Gaussian distribution
    bound - See 'brownian' for details.
    
    """
    radian = uniform(0, 2 * math.pi)
    dist = normal(0, sigma)
    x_new = int(init_loc[0] + round(dist * math.cos(radian)))
    y_new = int(init_loc[1] + round(dist * math.sin(radian)))
    if x_new in range(M1) and y_new in range(M2):
        return (x_new, y_new)
    else:
        if bound == 'p':
            return None
        else: return Gaussian(init_loc, M1, M2, sigma, bound)
