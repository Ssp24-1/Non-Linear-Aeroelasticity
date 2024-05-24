import numpy as np
import matplotlib.pyplot as plt
import cmath

# Define Key Parameters
rho = 1.225
c = 0.25
xf = 0.1

I_a = 0.08
K_a = 50
K_a3 = 10*K_a                                                   # Positive (K_a3 = 10*K_a) - Hardening Stiffness ; Negative (K_a3 = -10*K_a) - Softening Stiffness 
af = 2 * cmath.pi/180                                          # Pitch Offset Angle in Radians

# Set Equations
b = c/2
a = (xf/b) - 1


# Non-Linear System Equations
n = 1000
U = np.linspace(37.2, 37.6, n)                                  # Freestream Speed (Split into 10000 points)
m = I_a + (0.125 + a**2)*rho*cmath.pi*b**4                      # Mass (Inertia + Aerodynamic)
d = 2*rho*U*cmath.pi*(b**3)*(0.5-a)                             # Damping (Aerodynamic Only)
k1 = K_a - 2*cmath.pi*rho*(U**2)*(b**2)*(a+0.5)                 # Stiffness (Aerodynamic + Structural)
k3 = K_a3                                                       # Non Linear Stiffness (Dependent on AOA) - Hardening Stiffness        


# Solve for U_Fold
U_fold = cmath.sqrt((K_a + (3*K_a3*af*af/4))/(2*cmath.pi*rho*b*b*(a+0.5)))


# Obtaining Values of Fixed Points Coordinates
XF1 = []
XF2 = []
XF3 = []

for i in range(n):
    dl = -4*K_a3*(K_a + (3*K_a3*af*af/4) - 2*cmath.pi*rho*U[i]*U[i]*b*b*(a+0.5))    # Delta
    v1 = (-3*af/2) + (cmath.sqrt(dl)/(2*K_a3))                                      # Fixed Points-2 & 3 exists only after Fold Speed
    v2 = (-3*af/2) - (cmath.sqrt(dl)/(2*K_a3))                               
    XF2.append(v1)
    XF3.append(v2)
    
    v1 = 0                                                                          # Fixed Point-1 = 0
    XF1.append(v1)


FP1 = []
FP2 = []
FP3 = []
Bifurc_Point = -3*af/2

for i in range(n):
    FP1.append([U[i], XF1[i].real])

    if U[i] < U_fold:                                                   # Only Stable Fixed Points (Negative Real Part)                             
        continue

    else:
        FP2.append([U[i], XF2[i].real])
        FP3.append([U[i], XF3[i].real])

# =============================================== #
# Plotting (Fold Bifurcation)
# =============================================== #
plt.figure(1)
plt.plot(*zip(*FP1))
plt.plot(*zip(*FP2))
plt.plot(*zip(*FP3))
plt.plot(U_fold, Bifurc_Point, '-ok')                  
plt.legend([r'$U_{F_1}$', r'$U_{F_2}$', r'$U_{F_3}$'])
plt.xlabel(r'Freestream Speed [$\frac{m}{s}$]')
plt.ylabel(r'$U_{F}$')
plt.grid()
plt.title('Fixed Points vs Speed (Fold Bifurcation)')
plt.show()