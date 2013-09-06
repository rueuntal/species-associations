from __future__ import division
from scipy.stats import binom
import numpy as np
from numpy.random import *

def two_to_one_d(loc, D):
    """Compute the index in a 1-d list from (x, y) coordinates 
    
    in the corresponding 2-d grid system.
    
    Input:
    loc - list [x, y], coordinates in 2-d grid system
    D - length/width of the grid system
    """
    return loc[0] + loc[1] * D

def one_to_two_d(i, D):
    """Compute the coordinates (x, y) in a 2-d grid system from
    
    the index in the corresponding 1-d list.
    
    Input:
    i - index in 1-d list
    D - length/width of the 2-d grid system
    """
    y = int(np.floor(i / D))
    x = i - D * y
    return [x, y]
            
class community:
    def __init__(self, D):
        """Initializing the local community, which 
        
        will be held in a list of dictionaries.
        Input:
        D - dimension. There will be D ** 2 local communities.
        K - carrying capacity of a local community
        S - species richness in the regional/global pool
        
        """
        self.D = D
        self.K = K
        self.margin = [] # A list to hold the position of marginal cells
        for i in range(D)[1:-1]:
            self.margin.append([0, i]) # Upper margin
            self.margin.append([D - 1, i])  # Lower margin
            self.margin.append([i, 0]) # Left margin
            self.margin.append([i, D - 1]) # Right margin
        self.margin.append([0, 0])
        self.margin.append([D - 1, D - 1])
        self.margin.append([0, D - 1])
        self.margin.append([D - 1, 0])

        self.COMS = [{} for i in range(D ** 2)]
        for COM in self.COMS:
            for i in range(S):
                COM[str(i)] = 0
        
    def dispersal(self, dispersers, disp_func, **kwargs):
        """Dispersal process of immigrants or newborns.
        
        Input:
        dispersers - list of lists holding the identity and initial 
          location of each dispersing individual
        disp_func - dispersal function
        kwargs - parameters needed for disp_func
        
        """
        for ind in dispersers:
            ind_sp = ind[0]
            ind_loc = ind[1]
            new_loc = disp_func(ind_loc, self.D, self.D, **kwargs)
            new_index = two_to_one_d(new_loc, self.D)
            COM = self.COMS[new_index]
            COM[ind_sp] += 1
    
    def immigration(self, global_rad, m):
        """Process of individuals immigrating from 
        
        metacommunity to local communities.
        Creates a list of immigrants with species identity and initial location.
        The immigrants will then enter local communities through dispersal.
        Input:
        global_rad - relative global abundance of species, a list of length S
        m - rate of immigration, expected number of immigrants per time step
        """
        self.immigrants = []
        for sp in range(S):
            immigrants_sp = binom(m, global_rad[sp])
            if immigrants_sp:
                for ind in range(immigrants_sp):
                    loc = self.margin[randint(0, len(self.margin))]
                    self.immigrants.append((str(sp), loc))

    def birth(self, b, A):
        """Birth process which is assumed to be association-dependent.
        
        Creates a list of newborns with species identity and initial location.
        The newborns will be distributed to local communities through dispersal after 
        death process.
        Input:
        b - birth rate
        A - association matrix
        
        """
        newborns = []
        for COM in self.COMS:
            
    def death(self, d):
        """Death process which is assumed to be association-independent.
        
        Input:
        d - death rate
        
        """
        for COM in self.COMS:
            for (sp, abd) in COM.items():
                if abd: # Check abundance is not zero already
                    COM[sp] = binomial(abd, 1 - d)

    def culling(self):
        """Culling process to remove excess individuals 
        
        and maintain carrying capacity, which is assumed to
        be association-independent.
        
        """
        for COM in self.COMS:
            num_remove = sum(COM.values()) - self.K
            if num_remove > 0:
                remove_list = []
                rand_list = sorted(uniform(0, sum(COM.values()), num_remove))
                rand_list.append(sum(COM.values()) + 1)
                i = 0
                upto = 0
                for (sp, abd) in COM.items():
                    upto += abd
                    while upto >= rand_list[i]:
                        i += 1
                        remove_list.append(sp)
                for ind in remove_list:
                    COM[ind] -= 1