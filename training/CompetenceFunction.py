import math
from collections.abc import Callable

def sqrt_competence_func(t, T, c0):
    competence = (t * (1 - c0 ** 2) / T + c0 ** 2)**(1/2)
    return min(1.0, competence)


class CompetenceFunction:
    """
    Class encapsulating the competence function used for curriculum learning.
    """
    def __init__(self, competence_func: Callable, T: int,  c0: float):
        """
        Initializes a compentence function instance
        
        :param competence_func: The competence function to be used. Requires the signature competence_func(t, T, c0).
        :type competence_func: Callable
        :param T: Number of total steps until a competence of 1 is reached.
        :type T: int
        :param c0: Initial competence level.
        :type c0: float
        """
        self.competence_func = competence_func
        self.T = T
        self.c0 = c0

    
    def compute_competence(self, t: int):
        """
        Computes the competence level for the current timestep t.
        
        :param t: The current time step to evaluate the competence for.
        :type t: int
        """
        return self.competence_func(t, self.T, self.c0)