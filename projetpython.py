import math
import autograd
from autograd import numpy as np
from autograd import grad
import matplotlib.pyplot as plt

#from autograd import numpy as np
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

#def distcourbe()

def simple_contour(f, c = 0.0, delta = 0.01):
    courbe_abs = []
    courbe_ord = []
    if find_seed(f, c) is None:
        return(courbe_abs, courbe_ord)
    else : 
        courbe_abs.append(0.0)
        courbe_ord.append(find_seed(f, c))
        direc = 1
        # on va essayer de savoir s'il faut aller à gauche ou à droite
        v = gradient(f, courbe_abs[-1], courbe_ord[-1])
        x = courbe_abs[-1] + delta*vecteur_courbe_normalisé(v)[0]
        y = courbe_ord[-1] + delta*vecteur_courbe_normalisé(v)[1]
        if x<=0 : 
            direc = -1
        while (0<= courbe_abs[-1] <= 1 and 0<= courbe_ord[-1] <= 1) :
            v = gradient(f, courbe_abs[-1], courbe_ord[-1])
            x = courbe_abs[-1] + direc*delta*vecteur_courbe_normalisé(v)[0]
            y = courbe_ord[-1] + direc*delta*vecteur_courbe_normalisé(v)[1]
            courbe_abs.append(x)
            courbe_ord.append(y)
        courbe_abs.pop()
        courbe_ord.pop()
    return(courbe_abs, courbe_ord)

def f(x,y):
    return(3*x**2 + y**2) 


#print(gradient(f, 1.0,2.0))

abscisses = simple_contour(f, 0.666, 0.01)[0]
ordo = simple_contour(f, 0.666, 0.01)[1]
plt.scatter(abscisses, ordo)
plt.show()
