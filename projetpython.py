import math
import autograd
from autograd import numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import itertools



def find_seed(g, c=0, eps=2**(-26)):
    if min(g(0,0), g(0,1)) <= c <= max(g(0,0), g(0,1)):
        a = 0
        b = 1
        while abs(g(0,(a+b)/2)-c) > eps:
            if min(g(0,a), g(0,(a+b)/2)) <= c <= max(g(0,(a+b)/2), g(0,a)):
                b = (a+b)/2
            else :
                a = (a+b)/2
        return (a+b)/2
    else : 
        return (None)
   

def gradient (f, x, y):
    g = autograd.grad
    return np.r_[g(f, 0)(x, y), g(f, 1)(x, y)]

def vecteur_courbe_normalisé(v):
    if v[0] != 0 or v[1] != 0 : 
        norme = math.sqrt((v[0]**2)+(v[1]**2))
        return ([-v[1]/norme, v[0]/norme])
    else : 
        return v

def distance_courbe(point, nuage): 
    x = point[0]
    y = point[1]
    distances = []
    for coord in zip(nuage[0], nuage[1]) : 
        x_c = coord[0]
        y_c = coord[1]
        d = math.sqrt((x_c-x)**2+(y_c-y)**2)
        distances.append(d)
    return (min(distances))

def simple_contour(f, c = 0.0, delta = 0.01):
    abs_c = []
    ord_c= []
    if find_seed(f, c) is None:
        return(abs_c, ord_c)
    else : 
        abs_c.append(0.0)
        ord_c.append(find_seed(f, c))
        direc = 1
        # on va essayer de savoir s'il faut aller à gauche ou à droite
        v = gradient(f, abs_c[-1], ord_c[-1])
        x = abs_c[-1] + delta*vecteur_courbe_normalisé(v)[0]
        y = ord_c[-1] + delta*vecteur_courbe_normalisé(v)[1]
        if x<=0 : 
            direc = -1
        while (0<= abs_c[-1] <= 1 and 0<= ord_c[-1] <= 1) and distance_courbe([abs_c[-1], ord_c[-1]], [abs_c, ord_c]) >= delta**2 :
            v = gradient(f, abs_c[-1], ord_c[-1])
            x = abs_c[-1] + direc*delta*vecteur_courbe_normalisé(v)[0]
            y = ord_c[-1] + direc*delta*vecteur_courbe_normalisé(v)[1]
            abs_c.append(x)
            ord_c.append(y)
        abs_c.pop()
        ord_c.pop()
    return(abs_c, ord_c)

def f(x,y):
    return(3*x**2 + y**2) 


#print(gradient(f, 1.0,2.0))

abscisses = simple_contour(f, 0.666, 0.01)[0]
ordo = simple_contour(f, 0.666, 0.01)[1]
plt.scatter(abscisses, ordo)
plt.show()

print(distance_courbe([0.5, 0.5], simple_contour(f, 0.666, 0.01)))