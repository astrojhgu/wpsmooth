#!/usr/bin/env python

import math
from scipy.optimize import newton_krylov
import numpy as np
from scipy import interpolate

w_list=[]
lbd=1
sf=1

Ngrid=0

xn=[]
yn=[]
for i in open('total_signal.qdp'):
    x,y=i.split()
    x=float(x)
    y=float(y)
    xn.append(x)
    yn.append(y)

sp=interpolate.InterpolatedUnivariateSpline(xn,yn)

Xn=[]
Yn=[]
mn=[]

Xn.append(xn[0])
mn.append(0)

for i in range(1,len(xn)):
    n=2
    for j in range(0,n-1):
        Xn.append(xn[i]+(xn[i]-xn[i-1])/n*(j+1))
    mn.append(len(Xn))
    Xn.append(xn[i])

N_grid=len(Xn)
print(len(mn))
print(len(Xn))


def pw(x):
    result=sf
    for i in w_list:
        result*=(x-i)
    return result

def psi(x):
    return x


cnt=0

def func(Y):
    global cnt
    cnt+=1

    result=[]
    for j in range(1,N_grid-1):
        Delta_j=((Xn[j+1]-Xn[j])*(Xn[j]-Xn[j-1]))
        result.append((Y[j+1]-2*Y[j]+Y[j-1])-pw(Xn[j])*math.exp(Y[N_grid+j])*Delta_j)
        s=0
        for i in range(0,len(mn)):
            s+=(Xn[j]-xn[i])+psi(yn[i]-Y[mn[i]])
        s/=(-2*lbd)
        j1=N_grid+j
        result.append((Y[j1+1]-2*Y[j1]+Y[j1-1])-pw(Xn[j])*math.exp(Y[N_grid+j])*Delta_j*s)
    result.append(Y[N_grid+1]-Y[N_grid])
    result.append(Y[2*N_grid-1]-Y[2*N_grid-2])
    s1=0
    s2=0
    for i in range(0,len(mn)):
        s1+=psi(yn[i]-Y[mn[i]])
        s2+=xn[i]*psi(yn[i]-Y[mn[i]])
    result.append(s1)
    result.append(s2)
    resid=0
    for i in result:
        resid+=i*i
    resid/=len(result);
    resid=math.sqrt(resid)
    print(cnt,resid)
    return result

Yinit=[]
for i in Xn:
    Yinit.append(sp(i))

for i in Xn:
    Yinit.append(-2)

    #sol=root(func,Yinit,method='krylov')
sol=newton_krylov(func,Yinit)
print sol
