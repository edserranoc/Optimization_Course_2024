import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable 

class Opti_functions:
    @staticmethod
    def gold_section_method(f: Callable[[float], float], 
                        xl: float, 
                        xu: float, 
                        tol: float = (np.finfo(float).eps)**(1/3), 
                        max_iter: int=100) -> Tuple[float, float,float,float,int,bool]:
        
        """Gold Section Method to find the minimum of a function inside a specified interval.
        
        :param f: function to optimize (Callable[[float], float])
        :param xl: left of the interval (float)
        :param xu: right of the interval (float)
        :param tol: tolerance (float)
        :param max_iter: Maximum iterations allowed (int)
        
        :return: xk, f(xk), xu, xl, k, bres (Tuple[float, float,float,float,int,bool])
        """
        
        bres = False
        rho = (np.sqrt(5)-1)/2
        k = 1
        
        while k < max_iter+1:
            b = rho*(xu-xl)
            x1 = xu-b
            x3 = xl+b
            
            if f(x1) < f(x3):
                xu =x3
                xk=x1
            elif f(x1) >= f(x3):
                xl=x1
                xk=x3

            if np.abs(xu-xl) < tol:
                bres = True
                return xk,f(xk),xu,xl,k,bres
            k += 1
        
        return xk,f(xk),xu,xl,k,bres