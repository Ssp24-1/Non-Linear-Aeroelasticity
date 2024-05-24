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

# Set Equations
b = c/2
a = (xf/b) - 1


# Non-Linear System Equations
U = np.linspace(35, 40, 10000)                                  # Freestream Speed (Split into 10000 points)
m = I_a + (0.125 + a**2)*rho*cmath.pi*b**4                      # Mass (Inertia + Aerodynamic)
d = 2*rho*U*cmath.pi*(b**3)*(0.5-a)                             # Damping (Aerodynamic Only)
k1 = K_a - 2*cmath.pi*rho*(U**2)*(b**2)*(a+0.5)                 # Stiffness (Aerodynamic + Structural)
k3 = K_a3                                                       # Non Linear Stiffness (Dependent on AOA) - Hardening Stiffness        


# 3 Fixed Points[x1, x2] : [0, 0]; [0, -(-k1/k3)^0.5]; [0, +(-k1/k3)^0.5]

# =============================================== #
# Since K_a, K_a3 are positive, the 2 other fixed points apart from X_F1[0, 0], will exists only if k1 < 0, that is (k1/k3) < 0
# For k1 < 0, it is also the condition for Static Divergence Speed
# =============================================== #

# Static Divergence Speed Calculation
U_d = cmath.sqrt((K_a)/(2*cmath.pi*rho*b*b*(a+0.5)))

# =============================================== #
# This means 2 fixed points will appear after the divergence speed
# =============================================== #

# Obtaining Values of Fixed Points Coordinates
XF1 = []
XF2 = []
XF3 = []

for i in range(10000):
    v1 = cmath.sqrt((-k1[i]/k3))
    v2 = -cmath.sqrt((-k1[i]/k3))                               # Fixed Points-2 & 3 exists only after Divergence Speed
    XF2.append(v1)
    XF3.append(v2)
    
    v1 = 0                                                      # Fixed Point-1 = 0
    XF1.append(v1)
        
# =============================================== #
# Check Stability of the Fixed Points
# =============================================== #
# For XF1 = [0, 0]

c1 = []
c2 = []

A1 = np.zeros([2, 2])
for i in range(10000):
    A1[0][0] = -d[i]/m
    A1[0][1] = -k1[i]/m                                         # Matrix obtained from Linearized Equation
    A1[1][0] = 1
    A1[1][1] = 0
    eigval, eigvec = np.linalg.eig(A1)

    if (eigval[0].real < 0 and eigval[1].real < 0) :
        c1.append(i)                                            # Stable
    else:
        c2.append(i)                                            # Unstable
        

# For XF2, 3 = [0, -(-k1/k3)^0.5] ; [0, +(-k1/k3)^0.5]

c3 = []
c4 = []

A2 = np.zeros([2, 2])
for i in range(10000):
    A2[0][0] = -d[i]/m
    A2[0][1] = -1*(k1[i]/m) + 3*(k1[i]/m)                       # Matrix obtained from Linearized Equation
    A2[1][0] = 1
    A2[1][1] = 0
    eigval, eigvec = np.linalg.eig(A2)

    if (eigval[0].real < 0 and eigval[1].real < 0) :
        c3.append(i)                                            # Stable
    else:
        c4.append(i)                                            # Unstable (Can be ignored since it is below U_d)


fig, ax = plt.subplots()

# Stable Plots
XF1_Stable = []

for i in range(len(c1)):
    XF1_Stable.append((U[int(c1[i])], XF1[int(c1[i])]))

XF2_Stable = []
XF3_Stable = []

for i in range(len(c3)):
    XF2_Stable.append([U[int(c3[i])], XF2[int(c3[i])]])
    XF3_Stable.append([U[int(c3[i])], XF3[int(c3[i])]])

# Unstable Plot
XF1_Unstable = []

for i in range(len(c2)):
    XF1_Unstable.append((U[int(c2[i])], XF1[int(c2[i])]))            

XF2_Unstable = []
XF3_Unstable = []

for i in range(len(c4)):
    XF2_Unstable.append([U[int(c4[i])], XF2[int(c4[i])]])
    XF3_Unstable.append([U[int(c4[i])], XF3[int(c4[i])]])     


# =============================================== #
# Plotting (Supercritical Bifurcation)
# =============================================== #
plt.figure(1)
plt.plot(*zip(*XF1_Stable), 'b')
plt.plot(*zip(*XF2_Stable), 'b', label = '_nolegend_')
plt.plot(*zip(*XF3_Stable), 'b', label = '_nolegend_')                  # Unstable nodes for XF2, XF3 are below U_d and hence not plotted
plt.plot(*zip(*XF1_Unstable), '--r')
plt.plot(U_d, 0, 'ok')

plt.legend(['Stable', 'Unstable', 'Saddle'])
plt.xlabel(r'Freestream Speed [$\frac{m}{s}$]')
plt.ylabel(r'$X_{F}$')
plt.grid()
plt.title('Fixed Points vs Speed (Supercritical Pitchfork Bifurcation)')
plt.show()


# =============================================== #
# Plotting (Subcritical Bifurcation) - (Set (K_a3  = -10*K_a), Softening Stiffness)
# =============================================== #
# plt.figure(1)
# plt.plot(*zip(*XF1_Stable), 'b')
# plt.plot(*zip(*XF2_Unstable), '--r', label = '_nolegend_')
# plt.plot(*zip(*XF3_Unstable), '--r', label = '_nolegend_')
# plt.plot(*zip(*XF1_Unstable), '--r')
# plt.plot(U_d, 0, 'ok')

# plt.legend(['Stable', 'Unstable', 'Saddle'])
# plt.xlabel(r'Freestream Speed [$\frac{m}{s}$]')
# plt.ylabel(r'$X_{F}$')
# plt.grid()
# plt.title('Fixed Points vs Speed (Subcritical Pitchfork Bifurcation)')
# plt.show()



# # =============================================== # 
# # Eigen Values of Fixed Points
# # =============================================== #

# # XF1 

# xf1_real1 = []
# xf1_imag1 = []
# xf1_real2 = []
# xf1_imag2 = []

# A1 = np.zeros([2, 2])
# for i in range(10000):
#     A1[0][0] = -d[i]/m
#     A1[0][1] = -k1[i]/m
#     A1[1][0] = 1
#     A1[1][1] = 0
#     eigval, eigvec = np.linalg.eig(A1)

#     xf1_real1.append(eigval[0].real)
#     xf1_imag1.append(eigval[0].imag)

#     xf1_real2.append(eigval[1].real)
#     xf1_imag2.append(eigval[1].imag)


# plt.figure(2)
# plt.plot(U, xf1_real1, 'b')
# plt.plot(U, xf1_real2, 'r')
# plt.grid()
# plt.title('Fixed Point $(X_{F_1})$ Real Part of Eigen Values')
# plt.legend([r'$\lambda_1$', r'$\lambda_2$'])


# plt.figure(3)
# plt.plot(U, xf1_imag1, 'b')
# plt.plot(U, xf1_imag2, 'r')
# plt.grid()
# plt.title('Fixed Point $(X_{F_1})$ Imaginary Part of Eigen Values')
# plt.legend([r'$\lambda_1$', r'$\lambda_2$'])


# # XF2,3

# xf23_real1 = []
# xf23_imag1 = []
# xf23_real2 = []
# xf23_imag2 = []

# A2 = np.zeros([2, 2])
# for i in range(10000):
#     A2[0][0] = -d[i]/m
#     A2[0][1] = -1*(k1[i]/m) + 3*(k1[i]/m)
#     A2[1][0] = 1
#     A2[1][1] = 0
#     eigval, eigvec = np.linalg.eig(A2)
    
#     xf23_real1.append(eigval[0].real)
#     xf23_imag1.append(eigval[0].imag)    

#     xf23_real2.append(eigval[1].real)
#     xf23_imag2.append(eigval[1].imag)


# plt.figure(4)
# plt.plot(U, xf23_real1, 'b')
# plt.plot(U, xf23_real2, 'r')
# plt.grid()
# plt.title('Fixed Point $(X_{F_{2,3}})$ Real Part of Eigen Values')
# plt.legend([r'$\lambda_1$', r'$\lambda_2$'])
# plt.axis([37, 37.5, 0, -5])


# plt.figure(5)
# plt.plot(U, xf23_imag1, 'b')
# plt.plot(U, xf23_imag2, 'r')
# plt.grid()
# plt.title('Fixed Point $(X_{F_{2,3}})$ Imaginary Part of Eigen Values')
# plt.legend([r'$\lambda_1$', r'$\lambda_2$'])
# plt.axis([37, 37.5, 5, -5])
# plt.show()