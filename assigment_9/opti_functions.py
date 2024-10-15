import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable 

class Opti_functions:
    
    A_Hartmann = np.array([[10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14]])
    P_Hartmann = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                    [2329, 4135, 8307, 3736, 1004, 9991],
                    [2348, 1451, 3522, 2883, 3047, 6650],
                    [4047, 8828, 8732, 5743, 1091, 381]])
    alpha_Hartmann = np.array([1.0, 1.2, 3.0, 3.2])
        
    
    # Himmelblau Function
    @staticmethod
    def fncHimmelblau(x):
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    @staticmethod
    def grad_fncHimmelblau(x):
        return np.array([4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7), 
                        2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)])
    
    @staticmethod
    def hess_fncHimmelblau(x):
        x1 = x[0]
        x2 = x[1]
        hess = np.array([[12*x1**2 + 4*x2 - 42, 4*x1 + 4*x2],
                        [4*x1 + 4*x2, 4*x1 + 12*x2**2 - 26]])
        return hess
    
    # Beale Function    
    @staticmethod
    def fncBeale(x):
        return (1.5-x[0]+x[0]*x[1])**2+(2.25-x[0]+x[0]*x[1]**2)**2+(2.625-x[0]+x[0]*x[1]**3)**2
    
    @staticmethod
    def grad_fncBeale(x):
        return np.array([2*(1.5-x[0]+x[0]*x[1])*(-1+x[1])+2*(2.25-x[0]+x[0]*x[1]**2)*(-1+x[1]**2)+2*(2.625-x[0]+x[0]*x[1]**3)*(-1+x[1]**3),
                        2*(1.5-x[0]+x[0]*x[1])*x[0]+4*(2.25-x[0]+x[0]*x[1]**2)*x[0]*x[1]+6*(2.625-x[0]+x[0]*x[1]**3)*x[0]*x[1]**2])
    
    def hess_fncBeale(x):
        return np.array([[6 - 4*x[1] - 2*x[1]**2 - 4*x[1]**3 + 2*x[1]**4 + 2*x[1]**6, 
             3 - 4*x[0] + 9.*x[1] - 4*x[0]*x[1] + 15.75*x[1]**2 - 12*x[0]*x[1]**2 + 8*x[0]*x[1]**3 + 12*x[0]*x[1]**5],
            [3 - 4*x[0] + 9.*x[1] - 4*x[0]*x[1] + 15.75*x[1]**2 - 12*x[0]*x[1]**2 + 8*x[0]*x[1]**3 + 12*x[0]*x[1]**5,
             9*x[0] - 2*x[0]**2 + 31.5*x[0]*x[1] - 12*x[0]**2*x[1] + 12*x[0]**2*x[1]**2 + 30*x[0]**2*x[1]**4]])
    
    
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
    def hess_fncRosenbrock(x):
        n = len(x)
        hess = np.zeros((n,n))
        hess[0,0] = 1200*x[0]**2 - 400*x[1] + 2
        for i in range(1,n-1):
            hess[i,i] = 202 + 1200*x[i]**2 - 400*x[i+1]
            hess[i,i-1] = -400*x[i-1]
            hess[i-1,i] = -400*x[i-1]
        hess[-1,-1] = 200
        hess[-2,-1] = -400*x[-2]
        hess[-1,-2] = -400*x[-2]
        return hess    
    
    @classmethod
    def fncHartmann(cls,x:np.ndarray)->float:
        sum_term = 0
        for i in range(4):
            sum_term += cls.alpha_Hartmann[i] * np.exp(-np.sum(cls.A_Hartmann[i] * (x - cls.P_Hartmann[i])**2))
        return -1 / 1.94 * (2.58 + sum_term)
    
    @classmethod
    def grad_fncHartmann(cls,x:np.ndarray)->np.ndarray:
        grad = np.zeros(6)
        for i in range(4):
            grad += 2 * cls.alpha_Hartmann[i] * (cls.P_Hartmann[i] - x) * cls.A_Hartmann[i] * np.exp(-np.sum(cls.A_Hartmann[i] * (x - cls.P_Hartmann[i])**2))
        return -1 / 1.94 * grad
    
    @classmethod
    def hess_Hartmann(cls,x:np.ndarray)->np.ndarray:
        
        hess = np.zeros((6,6))
        for k in range(4):
            phi = np.exp(-np.sum(cls.A_Hartmann[k] * (x - cls.P_Hartmann[k])**2))
            hess += phi * (4*cls.alpha_Hartmann[k]*np.outer(cls.A_Hartmann[k] * (x - cls.P_Hartmann[k]), cls.A_Hartmann[k] * (x - cls.P_Hartmann[k])))
            hess -= 2*cls.alpha_Hartmann[k]*np.diag(cls.A_Hartmann[k] * phi)
        return -1 / 1.94 * hess

        
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
        flag = False
        while k < iter_maxb:
            if f(xk + alpha*dir_pk) <= fk + c*alpha*np.dot(grad_fk,dir_pk):
                return alpha, k
            alpha *= rho
            k += 1
        print('Backtracking line search did not converge')
        return alpha, k
    
    @staticmethod
    def brack_tracking_Wolfe(alpha_init:float, 
                             rho:float,
                             c1:float, c2:float, 
                             xk:np.ndarray,
                             f:Callable[[np.ndarray],float],
                                gradf:Callable[[np.ndarray],np.ndarray],
                                fk:float,
                                grad_fk:np.ndarray,
                                dir_pk:np.ndarray,
                                iter_maxb:int=100,
                                alt:bool=False)->Tuple[float,int]:
        """Backtracking line search algorithm using Wolfe conditions
        
        :param alpha_init: float, initial step length
        :param rho: float, decay factor for step length
        :param c1: float, constant for Armijo condition
        :param c2: float, constant for Wolfe condition
        :param xk: np.ndarray, current point
        :param f: callable, objective function
        :param gradf: callable, gradient of objective function
        :param fk: float, objective function value at xk
        :param grad_fk: np.ndarray, gradient of objective function at xk
        :param dir_pk: np.ndarray, search direction
        :param iter_maxb: int, maximum number of iterations for backtracking
        
        :return : Tuple[float,int], step length and number of iterations
        """
        
        if alt:
            alpha = alpha_init
            k = 0
            flag = False
            while k < iter_maxb:
                cond = f(xk + alpha*dir_pk) <= fk + c1*alpha*np.dot(grad_fk,dir_pk)

                if (cond and ~flag)and alt:
                    flag = True
                    xu = xk + alpha*dir_pk
                    ku = k

                if cond and (np.dot(gradf(xk + alpha*dir_pk),dir_pk) >= c2*np.dot(grad_fk,dir_pk)):
                    return alpha, k
                alpha *= rho
                k += 1
            if flag:
                return xu, ku
            return alpha, k
        else:
            alpha = alpha_init
            k = 0
            while k < iter_maxb:
                if (f(xk + alpha*dir_pk) <= fk + c1*alpha*np.dot(grad_fk,dir_pk)) and (np.dot(gradf(xk + alpha*dir_pk),dir_pk) >= c2*np.dot(grad_fk,dir_pk)):
                    return alpha, k
                alpha *= rho
                k += 1
            return alpha, k
    
    
    @staticmethod
    def contornosFnc2D(fncf, xleft, xright, ybottom, ytop, levels,r_max1, list_xk1, r_max2=None, list_xk2=None,):
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

        ax.plot(list_xk1[:,0], list_xk1[:,1], marker='o', linestyle='-', color='blue', label='Trajectory $rmax: {}$'.format(r_max1), markersize=4)
        if list_xk2 is not None:
            ax.plot(list_xk2[:,0], list_xk2[:,1], marker='o', linestyle='-', color='red', label='Trajectory $rmax$: {}'.format(r_max2), markersize=4)
        ax.legend()
        
    @classmethod
    def gradient_descent_backtracking(cls,f:callable, grad_f:callable,
                                      x0:np.ndarray, tol:float, 
                                      iter_max:int, alpha_init:float, 
                                      rho:float, c:float,max_iter_bt)->Tuple[np.ndarray, int, bool]:
        """Gradient descent with backtracking line search
        :param f: objective function
        :rtype f: callable
        :param grad_f: gradient of objective function
        :rtype grad_f: callable
        :param x0: initial point
        :rtype x0: np.ndarray
        :param tol: tolerance for stopping criterion
        :rtype tol: float
        :param iter_max: maximum number of iterations
        :rtype iter_max: int
        :param alpha_init: initial step length
        :rtype alpha_init: float
        :param rho: decay factor for step length
        :rtype rho: float
        :param c: constant for Armijo condition
        :rtype c: float
        :param max_iter_bt: maximum number of iterations for backtracking line search
        :rtype max_iter_bt: int
        
        :return: optimal point, number of iterations, success
        :rtype: Tuple[np.ndarray, int, bool]
        """
        x = x0
        k = 0
        n = x0.shape[0]
        while k < iter_max:
            grad_fk = grad_f(x)
            fk = f(x)
            dir_pk = -grad_fk
            alpha, _ = cls.back_tracking(alpha_init, rho, c, x, f, fk, grad_fk, dir_pk, max_iter_bt)
            
            if alpha*np.linalg.norm(grad_fk) < tol:
                if n==2:
                    return x, k, True
                else:
                    return x, k, True
            x = x + alpha*dir_pk
            k += 1
        return x,k,False