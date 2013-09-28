from __future__ import division
from scipy.stats import binom
import numpy as np
from numpy.random import *
from random import sample

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
    def __init__(self, D, K, S):
        """Initializing the local community, which 
        
        will be held in a list of dictionaries.
        Input:
        D - dimension. There will be D ** 2 local communities.
        K - carrying capacity of a local community
        S - species richness in the regional/global pool
        
        """
        self.D = D
        self.K = K
        self.S = S
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
    
    def start_community(self):
        """Start with empty community"""
        self.COMS = [np.array([0 for i in range(self.S)]) for j in range(self.D ** 2)]
    
    def start_community_v2(self, global_rad):
        """Start with full community, where individuals are random draws from global RAD"""
        self.COMS = [np.array([binomial(self.K, global_rad[i]) for i in range(self.S)]) \
                     for j in range(self.D ** 2)]
        
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
            if new_loc: # If the disperser is not lost over boundary
                new_index = two_to_one_d(new_loc, self.D)
                self.COMS[new_index][int(ind_sp)] += 1
    
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
        for sp in range(self.S):
            immigrants_sp = binomial(m, global_rad[sp])
            if immigrants_sp:
                for ind in range(immigrants_sp):
                    loc = self.margin[randint(0, len(self.margin))]
                    self.immigrants.append([str(sp), loc])
    
    def immigration_v2(self, global_rad, m):
        """Process of individuals immigrating from
        
        metacommunity to local communities.
        In this new version, immigrants can enter any cell, instead of
        just the marginal cells.
        Input:
        global_rad - relative global abundance of species, a list of length S
        m - rate of immigration, expected number of immigrants PER CELL per time step
        
        """
        for COM in self.COMS:
            for sp in range(self.S):
                immigrants_sp = binomial(m, global_rad[sp])
                COM[int(sp)] += immigrants_sp
                
    def birth(self, b, k, A):
        """Birth process which is assumed to be association-dependent.
        
        Creates a list of newborns with species identity and initial location.
        The newborns will be distributed to local communities through dispersal after 
        death process.
        Input:
        b - intrinsic birth rate (when association is zero)
        A - S * S association matrix (list of lists). 
        k - strength of associations on birth rate
        """
        self.newborns = []
        for i, COM in enumerate(self.COMS):
            loc_COM = one_to_two_d(i, self.D)
            for sp1, abd1 in enumerate(COM):
                if abd1:
                    A_sum_sp1 = np.dot(COM, A[int(sp1)])
                    t_sp1 = k ** (A_sum_sp1 / (self.K - 1)) # Amax = N - 1
                    newborn_sp1 = binomial(abd1, b ** (1 / t_sp1))
                    if newborn_sp1:
                        self.newborns.extend([[sp1, loc_COM]] * newborn_sp1)
                
    def death(self, d):
        """Death process which is assumed to be association-independent.
        
        Input:
        d - death rate
        
        """
        for COM in self.COMS:
            for i, abd in enumerate(COM):
                if abd: # Check abundance is not zero already
                    COM[i] = binomial(abd, 1 - d)

    def culling(self):
        """Culling process to remove excess individuals 
        
        and maintain carrying capacity, which is assumed to
        be association-independent.
        
        """
        for COM in self.COMS:
            num_remove = sum(COM) - self.K
            if num_remove > 0:
                remove_list = []
                rand_list = sorted(sample(range(sum(COM)), num_remove))
                rand_list.append(sum(COM) + 1)
                i = 0
                upto = 0
                for sp, abd in enumerate(COM):
                    upto += abd
                    while upto > rand_list[i]:
                        i += 1
                        remove_list.append(sp)
                for ind in remove_list:
                    COM[int(ind)] -= 1