import math
import autograd
from autograd import numpy as np
import matplotlib.pyplot as plt

#cette fonction renvoie un réel t tel que f(0,t)=c si la condition est vérifiée, None sinon
def find_seed (g,c=0,eps=2**(-26),x0=0):
    if min(g(x0,0),g(x0,1))<=c<=max(g(x0,0),g(x0,1)):
        a=0
        b=1
        while abs(g(x0,(a+b)/2)-c)>eps:
            if min(g(x0,a),g(x0,(a+b)/2))<=c<=max(g(x0,a),g(x0,(a+b)/2)):
                b=(a+b)/2
            else :
                a=(a+b)/2
        return (a+b)/2

#cette fonction revoie le gradient de f en (x,y)
def gradient (f, x, y):
    return [autograd.grad(f,0)(x,y), autograd.grad(f,1)(x,y)]

#cette fonction renvoie le vecteur normal unitaire du vecteur x s'il existe, None sinon
def vecteur_niveau_normalise (x):
    if x[0]!=0 or x[1]!=0:
        n=math.sqrt(x[0]**2+x[1]**2)
        return [-x[1]/n,x[0]/n]

#cette fonction renvoie la distance d'un point z à une courbe (x,y)
def distance (x,y,z):
    if x==[] or y==[] :
        return None
    else:
        m=min(len(x),len(y))
        a=[]
        for i in range (m):
            a=a+[math.sqrt((x[i]-z[0])**2+(y[i]-z[1])**2)]
    return min(a)

#cette fonction renvoie un fragment de ligne de niveau de valeur c de f
def simple_contour(f, c=0.0, delta=0.01, delta2=0.0001):
    x=[]
    y=[]
    if find_seed (f,c) is None:
        return x,y
    else:
        x=x+[0.0]
        y=y+[find_seed (f,c)]
        a=delta*(vecteur_niveau_normalise(gradient(f,x[0],y[0]))[0])+x[0]
        b=delta*(vecteur_niveau_normalise(gradient(f,x[0],y[0]))[1])+y[0]
        if 0<=a<=1 and 0<=b<=1:
            while 0<=a<=1 and 0<=b<=1 and distance(x,y,[a,b])>(delta2):
                x=x+[a]
                y=y+[b]
                a=(delta*(vecteur_niveau_normalise(gradient(f,x[-1],y[-1]))[0]))+x[-1]
                b=(delta*(vecteur_niveau_normalise(gradient(f,x[-1],y[-1]))[1]))+y[-1]
            return x,y
        else:
            a = ((-1)*delta*(vecteur_niveau_normalise(gradient(f,x[0],y[0]))[0]))+x[0]
            b = ((-1)*delta*(vecteur_niveau_normalise(gradient(f,x[0],y[0]))[1]))+y[0]
            while 0<=a<=1 and 0<=b<=1 and distance(x,y,[a,b])>(delta2):
                x=x+[a]
                y=y+[b]
                a=((-1)*delta*(vecteur_niveau_normalise(gradient(f,x[-1],y[-1]))[0]))+x[-1]
                b=((-1)*delta*(vecteur_niveau_normalise(gradient(f,x[-1],y[-1]))[1]))+y[-1]
            return x,y

#on teste notre code pour une fonction quadratique simple
abscisses = simple_contour((lambda x,y:x**2+y**2), 0.666, 0.01)[0]
ordo = simple_contour((lambda x,y:x**2+y**2), 0.666, 0.01)[1]
plt.scatter(abscisses, ordo)
plt.show()