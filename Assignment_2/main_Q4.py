######################################## Libraries ########################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

######################################## Functions ########################################

def psi0_fun(x_list, x0, sig, k0):

    """
    
    The function returns the initial state array (normalised wavefunction) for given position array

    Parameters
    ----------
    x_list : Position array

    a : Size of the box

    sig : width of the Gaussian initial state

    """
    
    psi0_list = np.exp(-(x_list-x0)**2/(2*sig**2)) * np.exp(1j*k0*(x_list))

    overlap = overlap_fun(x_list, psi0_list, psi0_list)

    psi0_list = psi0_list/np.sqrt(overlap)

    return psi0_list

def psi_eig_fun(n, x_list, a):

    """

    The function returns the (normalised) eigenstate of the particle in a box

    Parameters
    ----------
    n : quantum number of the eigenstate

    x_list : Position array

    a : Size of the box
    
    """

    psi_eig_list = np.sqrt(2/a)*np.sin(n*np.pi*x_list/a)

    return psi_eig_list

def overlap_fun(x_list, psi1_list, psi2_list):

    """

    The function returns the overlap between two wavefunctions. If the two wavefunctions are the same, 
    it should return 1 if the wavefunctions are normalised

    Parameters
    ----------
    x_list : Position array

    psi1 : First wavefunction

    psi2 : Second wavefunction
    
    """

    dx = x_list[1]-x_list[0]
    overlap = np.dot(np.conjugate(psi1_list.T),psi2_list)*dx

    return overlap

def psi_fourier_recon_fun(N, x_list, psi_list):

    """

    The function returns the reconstructed wavefunction using the Fourier coefficients with the eigenstates

    Parameters
    ----------
    N : The number of eigenstates to consider for Fourier decomposition 

    x_list : Position array

    psi_list : The wavefunction in consideraton
    
    """

    psi_fourier_recon_list = np.zeros(len(x_list),dtype=complex)
    
    for n in range(1,N+1):
        psi_eig_list = psi_eig_fun(n,x_list,a)
        coeff = overlap_fun(x_list, psi_list, psi_eig_list)
        psi_fourier_recon_list += coeff * psi_eig_list

    return psi_fourier_recon_list

def Energies_fun(N):

    """

    The function returns the energy eigenvalyes of the particle in a box 

    Parameters
    ----------
    N : The number of eigenstates to consider for Fourier decomposition 

    """

    Energy_list = []

    for n in range(1,N+1):
        Energy_list.append(n**2*hbar**2*np.pi**2/(2*m*a**2))

    return Energy_list

def psi_fourier_recon_time_fun(N, x_list, psi_list, t_list):

    """

    The function returns the reconstructed wavefunction using the Fourier coefficients with the eigenstates with time evolution

    Parameters
    ----------
    N : The number of eigenstates to consider for Fourier decomposition 

    x_list : Position array

    psi_list : The initial wavefunction
    
    t_list : Time list

    """
    psi_fourier_recon_list_time = []
    Energy_list = Energies_fun(N)
    for t in t_list:
        psi_fourier_recon_list = np.zeros(len(x_list),dtype=complex)
        for n in range(1,N+1):
            psi_eig_list = psi_eig_fun(n,x_list,a)
            coeff = overlap_fun(x_list, psi_eig_list, psi_list)
            psi_fourier_recon_list += np.exp(- 1j/hbar * Energy_list[n-1]* t) * coeff * psi_eig_list
        psi_fourier_recon_list_time.append(psi_fourier_recon_list)
        
    return psi_fourier_recon_list_time

def exp_val_x_time_fun(x_list, t_list, psi_list_time):

    """

    The function returns the expectation value of position at each time step

    Parameters
    ----------
    x_list : Position array
    
    t_list : Time list

    psi_time_list : The list of wavefunctions at the given timesteps
    
    """

    exp_val_x_time = []
    for t_ind, t in enumerate(t_list):
        exp_val_x = overlap_fun(x_list, psi_list_time[t_ind], x_list*(psi_list_time[t_ind]))
        exp_val_x_time.append(exp_val_x)

    return exp_val_x_time

def p_psi_fun(x_list, psi_list):

    """

    The function returns the function obtained by operator p acting on the wavefuinction psi

    Parameters
    ----------
    x_list : Position array

    psi_list : The wavefunction in consideration
    
    """

    dx = x_list[1] - x_list[0]
    psi_der_list = []
    for i in range(len(psi_list)-1):
        psi_der = hbar/1j*1/dx*(psi_list[i+1]-psi_list[i])
        psi_der_list.append(psi_der)

    # psi_der_list will have length 1 less from that of x_list or psi_list

    return psi_der_list

def exp_val_p_time_fun(x_list, t_list, psi_list_time):

    """

    The function returns the expectation value of moementum over the time

    Parameters
    ----------
    x_list : Position array

    t_list : Time list

    psi_list : The wavefunction in consideration
    
    """

    exp_val_p_time = []
    for t_ind, t in enumerate(t_list):
        exp_val_p = overlap_fun(x_list, psi_list_time[t_ind][0:-1], p_psi_fun(x_list,psi_list_time[t_ind]))
        exp_val_p_time.append(exp_val_p)

    return exp_val_p_time

def classical_x_time_fun(v, t_list, x0):

    """

    The function returns the position and momentum at each time step

    Parameters
    ----------
    v : Initial velocity of the particle

    t_list : Time list

    x0 : Initial position
    
    """


    dt = t_list[1] - t_list[0]
    position_time = np.zeros(len(t_list)) 
    momentum_time = np.zeros(len(t_list)) 

    # Initialize position
    position_time[0] = x0
    momentum_time[0] = m*v

    for i in range(1,len(position_time)):
        # Update position
        position_time[i] = position_time[i-1] + v * dt
        momentum_time[i] = m*v
        # Reflect at boundaries
        if position_time[i] <= 0:
            position_time[i] = -position_time[i]         # Reflect and move back
            v = -v                         # Reverse velocity
            momentum_time[i] = m*v
        elif position_time[i] >= a:
            position_time[i] = 2 * a - position_time[i]  # Reflect and move back
            v = -v                            # Reverse velocity
            momentum_time[i] = m*v

    return position_time, momentum_time

######################################## Parameters ########################################
hbar = 1
# Size of the Box
a = 10
# Width of the initial Gaussian-like wavefunction
sig = a/10
# Mass of the particle
m = 1
# Average momentum
k0 = 1000/(10*a)
p = hbar*k0
# Position array
x_list = np.arange(0, a, a/1000)
# Initial postion
x0 = a/2
# The number of eigenstates to consider for Fourier decomposition (The more this number, the accurate will be the result)
N = 50
# Time array
T = 1
t_list = np.arange(0,T,T/1000)
# Classical velocity
v = np.sqrt(p**2/m**2)

######################################## Calculation ########################################
psi0_list = psi0_fun(x_list, x0, sig, k0)
# psi0_fourier_recon_list = psi_fourier_recon_fun(N, x_list, psi0_list)
psi_fourier_recon_list_time = psi_fourier_recon_time_fun(N, x_list, psi0_list, t_list)
exp_val_x_time = exp_val_x_time_fun(x_list, t_list, psi_fourier_recon_list_time)
exp_val_p_time = exp_val_p_time_fun(x_list, t_list, psi_fourier_recon_list_time)
position_time, momentum_time = classical_x_time_fun(v, t_list, x0)

# Probability density
plt.figure(figsize=(10, 6))
plt.contourf(x_list, t_list, np.abs(psi_fourier_recon_list_time)**2, levels=200, cmap='viridis')
plt.colorbar(label=r'$|\psi(x, t)|^2$')
plt.xlabel(r'Position $x$')
plt.ylabel(r'Time $t$')
plt.title(r'Time Evolution of $|\psi(x, t)|^2$')

# Expectation value of position
plt.figure(figsize=(10, 6))
plt.plot(t_list, exp_val_x_time, label=r"$\left<x(t)\right>$")
plt.xlabel(r'Time $t$')
plt.ylabel(r'$\left<x(t)\right>$')
plt.plot(t_list, position_time, label=f"classical"+r" $x(t)$")
plt.legend()
plt.title(f"Time Evolution of the of Position")

# Expectation value of momentum
plt.figure(figsize=(10, 6))
plt.plot(t_list, exp_val_p_time, label=r" $\left<p(t)\right>$")
plt.xlabel(r'Time $t$')
plt.ylabel(r'$\left<p(t)\right>$')
plt.plot(t_list, momentum_time, label=f"classical"+r" $p(t)$")
plt.legend()
plt.title(f"Time Evolution of the of Momentum")

# plt.show()

########################################### Animation ####################################

# Function to animate the probability density evolution
def animate(psi_fourier_recon_list_time, init_func=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Initialize plot for probability density
    line, = ax.plot([], [], lw=2, label=r"$|\psi(x,t)|^2$")
    ax.set_xlim(x_list[0], x_list[-1])
    ax.set_ylim(0, 1.1 * np.max(np.abs(psi_fourier_recon_list_time)**2).real)
    ax.set_xlabel("Position $x$")
    ax.set_ylabel(r"$|\psi(x,t)|^2$")
    ax.legend()

    # Initialization function for the animation
    def init():
        line.set_data([], [])
        return line,

    # Animation function, updates data at each frame
    def update(frame):
        prob_density = np.abs(psi_fourier_recon_list_time[frame])**2
        line.set_data(x_list, prob_density.real)  # Only plot the real part
        return line,

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(psi_fourier_recon_list_time), init_func=init, interval=50, blit=True)
    plt.show()

    return anim

# Run animation
animate(psi_fourier_recon_list_time)