import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from math import pi as Pi
from math import inf as Inf

global a, V0
a=1     #size of well
V0=Inf  #depth of well

def potential(x):
    if abs(x) < a:
        V = 0
    else: 
        V = V0
    return V

global Nb
Nb=4  #number of basis vectors

import sympy as sym

x = sym.Symbol("x")

start_basis = [ x**n * (x-a) for n in range(1, Nb+1) ]

def inner_product(f1, f2):
    return sym.integrate(f1*f2, (x, 0, a))

orthogonal_basis = []
for f in start_basis:
    for prev_f in orthogonal_basis:
        f = f - inner_product(prev_f, f)*prev_f / inner_product(prev_f,prev_f)
    orthogonal_basis.append(f)
    
orthonormal_basis = []
for f in orthogonal_basis:
    f = f / sym.sqrt(inner_product(f,f))
    orthonormal_basis.append(f)
    
print("original basis ", start_basis,"\n\n")

print("orthogonal basis ", orthogonal_basis,"\n\n")

print("orthonormal basis ", orthonormal_basis,"\n\n")

names = ["phi_"+str(i)+"(x)" for i in range(1,Nb+1)]

print("These are ", names)

# the 

def basis(j, xx):
    f = orthonormal_basis[j-1]
    f = sym.lambdify(x, f)
    return f(xx)

fig, ax = plt.subplots()
plt.title('Bais functions $\phi_{j}(x)$')
plt.xlabel("x")
plt.ylabel('$\phi_j(x)$')  

for j in range(1,Nb+1):
    points = range(0,101)
    x_grid = np.zeros(len(points))
    phi_grid = np.zeros(len(points))
    for x_i,point in enumerate(points):
        x_grid[x_i] = (a/100)*point
        phi_grid[x_i] = basis(j, x_grid[x_i])
    ax.plot(x_grid, phi_grid, label=str(j))
    
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
plt.show() 

def d2_basis(j, xx):
    f = sym.diff( sym.diff( orthonormal_basis[j-1] )  )
    f = sym.lambdify(x, f)
    return f(xx)

def Integrand1(x,i,j):
    return (-1/2)*basis(i,x)*d2_basis(j,x)

##Empty matrix to fill with matrix-element values
Kij = np.empty([Nb,Nb])

##Python array indices start at zero
for j in range(Nb):
    for i in range(j+1):
        Kij[i,j] = quad(Integrand1, 0, a, args=(i+1,j+1))[0]
        Kij[j,i] = Kij[i,j]
print("Kij = ",Kij)

def Integrand2(x,i,j):
    return basis(i,x)*basis(j,x)*potential(x)

##Empty matrix to fill with matrix-element values
Vij = np.empty([Nb,Nb])

for j in range(Nb):
    for i in range(j+1):
        Vij[i,j] = quad(Integrand2, 0, a, args=(i+1,j+1))[0]
        Vij[j,i] = Vij[i,j]
        
print("Vij = ",Vij)

Energies=np.empty([Nb])

Coefficients=np.empty([Nb,Nb])

Hij = Kij + Vij
E, C = np.linalg.eigh(Hij)
for n in range(0,Nb):
    Energies[n]=E[n]
    for j in range(0,Nb):
        #***Note that the eigenfunctions are the *column* vectors of C***
        Coefficients[n,j]=C[j,n]

print("*** Energy levels ***")
for n in range(0,Nb):  #principal quantum number
    print("n =", str(n+1), ", En = ", Energies[n])

print("\n")
print("*** Exact energy levels ***")
for n in range(0,Nb):  #principal quantum number
    print("n =", str(n+1), ", En = ", (n+1)**2*Pi**2/2)
    
    width = 0.1 
count=0
for n in range(0,Nb):
    C0 = np.abs(Coefficients[n])
    plt.bar(np.arange(Nb)+width*count, C0, width, label='j='+str(n+1))
    count=count+1
        
plt.title('Coefficients of basis vectors for expansion of $\psi_{n}(x)$')
plt.xlabel("n")
plt.ylabel('$c_{n}^j$')  
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show() 

#plotting colors
cmap = plt.get_cmap("tab10") 

for j in range(1,Nb+1):
    fig, ax = plt.subplots()
    plt.title(f'Energy eigenstate $\psi_{j}(x)$')
    plt.xlabel("x")
    plt.ylabel(f'$\psi_{j}(x)$')  
    points = range(0,101)
    x_grid = np.zeros(len(points))
    psi_j_grid = np.zeros(len(points))
    for x_i,point in enumerate(points):
        x_grid[x_i] = (a/100)*point
        psi_j_grid[x_i] = sum([ Coefficients[j-1,k-1] * basis(k, x_grid[x_i]) for k in range(1,Nb+1)] )
    ax.plot(x_grid, psi_j_grid, label=str(j), color=cmap(j-1))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show() 
    
   # For grid lines
from matplotlib.ticker import AutoMinorLocator

#Plot energies
fig, ax = plt.subplots()

EList = []
EList_exact = []
xList = []
for n in range(0,Nb):
    EList.append(Energies[n])
    EList_exact.append((n+1)**2*Pi**2/2)
    xList.append(n+1)
    if n == 0:
        ax.scatter(xList, EList, s=1444, marker="_", linewidth=3, zorder=3, color=cmap(0), label=" numerical")
        ax.scatter(xList, EList_exact, s=1444, marker="_", linewidth=3, zorder=3, color=cmap(1), label=" exact")
    else:
        ax.scatter(xList, EList, s=1444, marker="_", linewidth=3, zorder=3, color=cmap(0))
        ax.scatter(xList, EList_exact, s=1444, marker="_", linewidth=3, zorder=3, color=cmap(1))

minor_locator = AutoMinorLocator(2)
plt.minorticks_on()
ax.grid(axis='y',which='major')
ax.grid(axis='y',which='minor',linestyle='dotted')
ax.set_ylabel('$E\\ \, [\hbar^2 / ma^2]$',size=14)
ax.set_xlabel('$n$',size=14)
ax.set_ylim(0,1.5*max(EList_exact))
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
plt.title('Energy levels of infinite square well')
ax.margins(0.2)
plt.show()

