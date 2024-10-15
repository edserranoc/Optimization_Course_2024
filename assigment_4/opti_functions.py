import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple

class Opti_functions:
    
    @staticmethod
    def fncHimmelblau(x):
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    @staticmethod
    def grad_fncHimmelblau(x):
        return np.array([4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7), 
                        2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)])
        
    @staticmethod
    def fncBeale(x):
        return (1.5-x[0]+x[0]*x[1])**2+(2.25-x[0]+x[0]*x[1]**2)**2+(2.625-x[0]+x[0]*x[1]**3)**2
    
    @staticmethod
    def grad_fncBeale(x):
        return np.array([2*(1.5-x[0]+x[0]*x[1])*(-1+x[1])+2*(2.25-x[0]+x[0]*x[1]**2)*(-1+x[1]**2)+2*(2.625-x[0]+x[0]*x[1]**3)*(-1+x[1]**3),
                        2*(1.5-x[0]+x[0]*x[1])*x[0]+4*(2.25-x[0]+x[0]*x[1]**2)*x[0]*x[1]+6*(2.625-x[0]+x[0]*x[1]**3)*x[0]*x[1]**2])
    
    @staticmethod    
    def fncRosenbrock(x):
        return np.sum(100*(x[1:]-x[:-1]**2)**2+(1-x[:-1])**2)

    @staticmethod
    def grad_fncRosenbrock(x):
        grad = np.zeros(len(x))
        grad[0] = -400*x[0]*(x[1]-x[0]**2)+2*x[0]-2
        grad[-1] = 200*(x[-1]-x[-2]**2)
        for i in range(1,len(x)-1):
            grad[i] = 200*(x[i]-x[i-1]**2)-400*x[i]*(x[i+1]-x[i]**2)+2*x[i]-2
        return grad
    
    @staticmethod
    def back_tracking(alpha_init:float, 
                    rho:float,
                    c:float, 
                    xk:np.ndarray,
                    f:Callable[[np.ndarray],float],
                    fk:float,
                    grad_fk:np.ndarray,
                    dir_pk:np.ndarray,
                    iter_maxb:int=100)->Tuple[float,int]:
        
        """Backtracking line search algorithm using Armijo condition
        
        :param alpha_init:  initial step length (float)
        :param rho: decay factor for step length (float)
        :param c: constant for Armijo condition (float)
        :param xk: current point (np.ndarray)
        :param f: objective function (Callable[[np.ndarray],float])
        :param fk: objective function value at xk (float)
        :param grad_fk: gradient of objective function at xk (np.ndarray)
        :param dir_pk: search direction (np.ndarray)
        :param iter_maxb: maximum number of iterations for backtracking (int)
        
        :return: step length and number of iterations (Tuple[float,int]) 
        """
        
        alpha = alpha_init
        k = 0
        while k < iter_maxb:
            if f(xk + alpha*dir_pk) <= fk + c*alpha*np.dot(grad_fk,dir_pk):
                return alpha, k
            alpha *= rho
            k += 1
        print('Backtracking line search did not converge')
        return alpha, k
    
    
    @staticmethod
    def contornosFnc2D(fncf, xleft, xright, ybottom, ytop, levels, list_xk1, list_xk2=None):
        # Crea una discretización uniforme del intervalo [xleft, xright]
        ax = np.linspace(xleft, xright, 250)
        # Crea una discretización uniforme del intervalo [ybottom, ytop]
        ay = np.linspace(ybottom, ytop, 200)
        # La matriz mX que tiene las abscisas 
        mX, mY = np.meshgrid(ax, ay)
        # Se crea el arreglo mZ con los valores de la función en cada nodo
        mZ = mX.copy()
        for i,y in enumerate(ay):
            for j,x in enumerate(ax):
                mZ[i,j] = fncf(np.array([x,y]))
        # Grafica de las curvas de nivel
        fig, ax = plt.subplots(figsize=(8,8))
        CS = ax.contour(mX, mY, mZ, levels, cmap='Wistia')
        
        ax.plot(list_xk1[:,0], list_xk1[:,1], marker='o', linestyle='-', color='blue', label='Trajectory $Xk_1$: x0={}'.format(list_xk1[0]), markersize=4)
        if list_xk2 is not None:
            ax.plot(list_xk2[:,0], list_xk2[:,1], marker='o', linestyle='-', color='red', label='Trajectory $Xk_2$: x0={}'.format(list_xk2[0]), markersize=4)
        ax.legend()
    
    