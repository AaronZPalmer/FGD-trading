import numpy as np
from scipy.integrate import odeint, cumulative_trapezoid
from scipy.interpolate import interp1d

# Calculations from Section 11.2 of "Trading with the Crowd" by Eyal Neuman and Moritz Voss.

# Model parameters to be used:

# T = 10

# kappa = 1
# gamma = 1
# rho = 1
# lamda = .5
# phi = .1
# varrho = 10

# iota = 1
# beta = .1
# sigma = .5


# Time grid for simulation, integration, and ODE solver:

# T_idx = 1000

# timeGrid = np.linspace(0, T, T_idx)

# A simulation is done like:

# signalRateArray = simulateSignalRatePath(beta, sigma, iota, T, timeGrid)

# signalRate = interp1d(np.linspace(0, T+.5, T_idx), signalRateArray)

# signalArray = cumulative_trapezoid(signalRateArray, timeGrid, initial = 0)

# signal = interp1d(np.linspace(0, T+1, T_idx), signalArray)

# \Define the eigenvalues $\tilde\nu_1, \tilde\nu_2, \tilde\nu_3$ and their differences $\tilde\nu_1-\tilde\nu_2, \tilde\nu_1-\tilde\nu_3, \tilde\nu_2-\tilde\nu_3$: 

def aTilde(kappa, rho, gamma, lamda, phi, varrho):
  
  return (2 * lamda * rho + gamma * kappa)/(2 * lamda)

  def bTilde(kappa, rho, gamma, lamda, phi, varrho):
  
  return (-phi/lamda)

def cTilde(kappa, rho, gamma, lamda, phi, varrho):
  
  return (-phi * rho/lamda)

def pTilde(kappa, rho, gamma, lamda, phi, varrho):
  
  return bTilde(kappa, rho, gamma, lamda, phi, varrho) - 1/3 * aTilde(kappa, rho, gamma, lamda, phi, varrho)**2


def qTilde(kappa, rho, gamma, lamda, phi, varrho):
  
  return 2/27 * aTilde(kappa, rho, gamma, lamda, phi, varrho)**3 - \
          1/3 * aTilde(kappa, rho, gamma, lamda, phi, varrho) * bTilde(kappa, rho, gamma, lamda, phi, varrho) + \
          cTilde(kappa, rho, gamma, lamda, phi, varrho)

def nuTilde1(kappa, rho, gamma, lamda, phi, varrho):

  return (-1) * np.sqrt(-4/3 * pTilde(kappa, rho, gamma, lamda, phi, varrho)) * \
          np.cos(1/3 * np.arccos(-qTilde(kappa, rho, gamma, lamda, phi, varrho)/2 * \
                                 np.sqrt(-27/pTilde(kappa, rho, gamma, lamda, phi, varrho)**3)) + np.pi/3) - \
          aTilde(kappa, rho, gamma, lamda, phi, varrho)/3

def nuTilde2(kappa, rho, gamma, lamda, phi, varrho):
  
  return np.sqrt(-4/3 * pTilde(kappa, rho, gamma, lamda, phi, varrho)) * \
          np.cos(1/3 * np.arccos(-qTilde(kappa, rho, gamma, lamda, phi, varrho)/2 * \
                                 np.sqrt(-27/pTilde(kappa, rho, gamma, lamda, phi, varrho)**3))) - \
          aTilde(kappa, rho, gamma, lamda, phi, varrho)/3

def nuTilde3(kappa, rho, gamma, lamda, phi, varrho):

  return (-1) * np.sqrt(-4/3 * pTilde(kappa, rho, gamma, lamda, phi, varrho)) * \
          np.cos(1/3 * np.arccos(-qTilde(kappa, rho, gamma, lamda, phi, varrho)/2 * \
                                 np.sqrt(-27/pTilde(kappa, rho, gamma, lamda, phi, varrho)**3)) - np.pi/3) - \
          aTilde(kappa, rho, gamma, lamda, phi, varrho)/3

def nuTilde12(kappa, rho, gamma, lamda, phi, varrho):

  return nuTilde1(kappa, rho, gamma, lamda, phi, varrho) - nuTilde2(kappa, rho, gamma, lamda, phi, varrho)
  

def nuTilde13(kappa, rho, gamma, lamda, phi, varrho):

  return nuTilde1(kappa, rho, gamma, lamda, phi, varrho) - nuTilde3(kappa, rho, gamma, lamda, phi, varrho)
  

def nuTilde23(kappa, rho, gamma, lamda, phi, varrho):

  return nuTilde2(kappa, rho, gamma, lamda, phi, varrho) - nuTilde3(kappa, rho, gamma, lamda, phi, varrho)
  

# Define the auxiliary functions $\tilde K_1, \tilde K_2, \tilde K_3$ from (11.22), (11.23), (11.24):

def KTilde1(t, T, kappa, rho, gamma, lamda, phi, varrho):

  return phi/(2 * lamda**2) * \
          ( (2 * varrho * (nuTilde1(kappa, rho, gamma, lamda, phi, varrho) + rho) + \
             kappa * gamma * nuTilde1(kappa, rho, gamma, lamda, phi, varrho) + \
             2 * lamda * nuTilde1(kappa, rho, gamma, lamda, phi, varrho) * \
             (nuTilde1(kappa, rho, gamma, lamda, phi, varrho) + rho)) / \
           (nuTilde1(kappa, rho, gamma, lamda, phi, varrho) * \
            nuTilde12(kappa, rho, gamma, lamda, phi, varrho) * \
            nuTilde13(kappa, rho, gamma, lamda, phi, varrho)) * \
           np.exp(nuTilde1(kappa, rho, gamma, lamda, phi, varrho) * (T - t)) \
           - (2 * varrho * (nuTilde2(kappa, rho, gamma, lamda, phi, varrho) + rho) + \
             kappa * gamma * nuTilde2(kappa, rho, gamma, lamda, phi, varrho) + \
             2 * lamda * nuTilde2(kappa, rho, gamma, lamda, phi, varrho) * \
             (nuTilde2(kappa, rho, gamma, lamda, phi, varrho) + rho)) / \
           (nuTilde2(kappa, rho, gamma, lamda, phi, varrho) * \
            nuTilde12(kappa, rho, gamma, lamda, phi, varrho) * \
            nuTilde23(kappa, rho, gamma, lamda, phi, varrho)) * \
           np.exp(nuTilde2(kappa, rho, gamma, lamda, phi, varrho) * (T - t)) \
           + (2 * varrho * (nuTilde3(kappa, rho, gamma, lamda, phi, varrho) + rho) + \
             kappa * gamma * nuTilde3(kappa, rho, gamma, lamda, phi, varrho) + \
             2 * lamda * nuTilde3(kappa, rho, gamma, lamda, phi, varrho) * \
             (nuTilde3(kappa, rho, gamma, lamda, phi, varrho) + rho)) / \
           (nuTilde3(kappa, rho, gamma, lamda, phi, varrho) * \
            nuTilde13(kappa, rho, gamma, lamda, phi, varrho) * \
            nuTilde23(kappa, rho, gamma, lamda, phi, varrho)) * \
           np.exp(nuTilde3(kappa, rho, gamma, lamda, phi, varrho) * (T - t)))

def KTilde2(t, T, kappa, rho, gamma, lamda, phi, varrho):

  return 1/(2 * gamma * lamda * rho) * \
      (((2 * varrho * (nuTilde1(kappa, rho, gamma, lamda, phi, varrho) + rho) + \
         kappa * gamma * nuTilde1(kappa, rho, gamma, lamda, phi, varrho) + \
         2 * lamda * nuTilde1(kappa, rho, gamma, lamda, phi, varrho) * \
         (nuTilde1(kappa, rho, gamma, lamda, phi, varrho) + rho)) * \
        (nuTilde2(kappa, rho, gamma, lamda, phi, varrho) + rho) * \
        (nuTilde3(kappa, rho, gamma, lamda, phi, varrho) + rho)) / \
       (nuTilde12(kappa, rho, gamma, lamda, phi, varrho) * \
        nuTilde13(kappa, rho, gamma, lamda, phi, varrho)) * \
       np.exp(nuTilde1(kappa, rho, gamma, lamda, phi, varrho) * (T - t)) \
       - ((2 * varrho * (nuTilde2(kappa, rho, gamma, lamda, phi, varrho) + rho) + \
           kappa * gamma * nuTilde2(kappa, rho, gamma, lamda, phi, varrho) + \
           2 * lamda * nuTilde2(kappa, rho, gamma, lamda, phi, varrho) * \
           (nuTilde2(kappa, rho, gamma, lamda, phi, varrho) + rho)) * \
          (nuTilde1(kappa, rho, gamma, lamda, phi, varrho) + rho) * \
          (nuTilde3(kappa, rho, gamma, lamda, phi, varrho) + rho)) / \
       (nuTilde12(kappa, rho, gamma, lamda, phi, varrho) * \
        nuTilde23(kappa, rho, gamma, lamda, phi, varrho)) * \
       np.exp(nuTilde2(kappa, rho, gamma, lamda, phi, varrho) * (T - t)) \
       + ((2 * varrho * (nuTilde3(kappa, rho, gamma, lamda, phi, varrho) + rho) + \
           kappa * gamma * nuTilde3(kappa, rho, gamma, lamda, phi, varrho) + \
           2 * lamda * nuTilde3(kappa, rho, gamma, lamda, phi, varrho) * \
           (nuTilde3(kappa, rho, gamma, lamda, phi, varrho) + rho)) * \
          (nuTilde1(kappa, rho, gamma, lamda, phi, varrho) + rho) * \
          (nuTilde2(kappa, rho, gamma, lamda, phi, varrho) + rho)) / \
       (nuTilde13(kappa, rho, gamma, lamda, phi, varrho) * \
        nuTilde23(kappa, rho, gamma, lamda, phi, varrho)) * \
       np.exp(nuTilde3(kappa, rho, gamma, lamda, phi, varrho) * (T - t)))

def KTilde3(t, T, kappa, rho, gamma, lamda, phi, varrho):

  return 1/(2 * lamda) * \
          (-(2 * varrho * (nuTilde1(kappa, rho, gamma, lamda, phi, varrho) + rho) + \
             kappa * gamma * nuTilde1(kappa, rho, gamma, lamda, phi, varrho) + \
             2 * lamda * nuTilde1(kappa, rho, gamma, lamda, phi, varrho) * \
             (nuTilde1(kappa, rho, gamma, lamda, phi, varrho) + rho)) / \
           (nuTilde12(kappa, rho, gamma, lamda, phi, varrho) * \
            nuTilde13(kappa, rho, gamma, lamda, phi, varrho)) * \
           np.exp(nuTilde1(kappa, rho, gamma, lamda, phi, varrho) * (T - t)) \
           + (2 * varrho * (nuTilde2(kappa, rho, gamma, lamda, phi, varrho) + rho) + \
              kappa * gamma * nuTilde2(kappa, rho, gamma, lamda, phi, varrho) + \
              2 * lamda * nuTilde2(kappa, rho, gamma, lamda, phi, varrho) * \
              (nuTilde2(kappa, rho, gamma, lamda, phi, varrho) + rho)) / \
           (nuTilde12(kappa, rho, gamma, lamda, phi, varrho) * \
            nuTilde23(kappa, rho, gamma, lamda, phi, varrho)) * \
           np.exp(nuTilde2(kappa, rho, gamma, lamda, phi, varrho) * (T - t)) \
           - (2 * varrho * (nuTilde3(kappa, rho, gamma, lamda, phi, varrho) + rho) + \
              kappa * gamma * nuTilde3(kappa, rho, gamma, lamda, phi, varrho) + \
              2 * lamda * nuTilde3(kappa, rho, gamma, lamda, phi, varrho) * \
              (nuTilde3(kappa, rho, gamma, lamda, phi, varrho) + rho)) / \
           (nuTilde13(kappa, rho, gamma, lamda, phi, varrho) * \
            nuTilde23(kappa, rho, gamma, lamda, phi, varrho)) * \
            np.exp(nuTilde3(kappa, rho, gamma, lamda, phi, varrho) * (T - t)))

# Define the auxiliary functions $\tilde w_1, \tilde w_2$ from (3.17):

def wTilde1(t, T, kappa, rho, gamma, lamda, phi, varrho):
  
  return (-1) * KTilde1(t, T, kappa, rho, gamma, lamda, phi, varrho)/ \
                  KTilde3(t, T, kappa, rho, gamma, lamda, phi, varrho)

def wTilde2(t, T, kappa, rho, gamma, lamda, phi, varrho):
  
  return (-1) * KTilde2(t, T, kappa, rho, gamma, lamda, phi, varrho)/ \
                  KTilde3(t, T, kappa, rho, gamma, lamda, phi, varrho)
                  

#  Define the auxiliary functions $R$ and $R'$ from (3.19):

def R(t, T, kappa, rho, gamma, lamda, phi, varrho):

  return np.sqrt(phi/lamda) * np.cosh(np.sqrt(phi/lamda) * (T - t)) + \
          varrho/lamda * np.sinh(np.sqrt(phi/lamda) * (T - t))

def Rprime(t, T, kappa, rho, gamma, lamda, phi, varrho):

  return phi/lamda * np.sinh(np.sqrt(phi/lamda) * (T - t)) + \
          varrho/lamda * np.sqrt(phi/lamda) * np.cosh(np.sqrt(phi/lamda) * (T - t))

# Introduce the process $(\tilde\zeta_t)_{0 \leq t \leq T}$ defined as 

# $$ \zeta_t := -\frac{1}{2\lambda} \mathbb{E}_t\left[\int_t^T \frac{\tilde K_3(T-s)}{\tilde K_3(T-t)} dA_s \right] \quad (0 \leq t \leq T),$$

# where $dA_s = I_s ds$ and $dI_s = -\beta I_s ds + \sigma dW_s$ (cf. equation (3.18)).
#           

def zetaTilde(t, T, kappa, rho, gamma, lamda, phi, varrho, beta, path):

  return (-1)/(4 * lamda**2 * KTilde3(t, T, kappa, rho, gamma, lamda, phi, varrho)) * path(t) * \
      ((2 * varrho * (nuTilde1(kappa, rho, gamma, lamda, phi, varrho) + rho) + \
        kappa * gamma * nuTilde1(kappa, rho, gamma, lamda, phi, varrho) + \
        2 * lamda * nuTilde1(kappa, rho, gamma, lamda, phi, varrho) * \
        (nuTilde1(kappa, rho, gamma, lamda, phi, varrho) + rho)) / \
       (nuTilde12(kappa, rho, gamma, lamda, phi, varrho) * \
        nuTilde13(kappa, rho, gamma, lamda, phi, varrho) * \
        (nuTilde1(kappa, rho, gamma, lamda, phi, varrho) + beta)) * \
       (np.exp(-beta * (T - t)) - np.exp((T - t) * nuTilde1(kappa, rho, gamma, lamda, phi, varrho))) \
       - (2 * varrho * (nuTilde2(kappa, rho, gamma, lamda, phi, varrho) + rho) + \
          kappa * gamma * nuTilde2(kappa, rho, gamma, lamda, phi, varrho) + \
          2 * lamda * nuTilde2(kappa, rho, gamma, lamda, phi, varrho) * \
          (nuTilde2(kappa, rho, gamma, lamda, phi, varrho) + rho)) / \
       (nuTilde12(kappa, rho, gamma, lamda, phi, varrho) * \
        nuTilde23(kappa, rho, gamma, lamda, phi, varrho) * \
        (nuTilde2(kappa, rho, gamma, lamda, phi, varrho) + beta)) * \
       (np.exp(-beta * (T - t)) - np.exp((T - t) * nuTilde2(kappa, rho, gamma, lamda, phi, varrho))) \
       + (2 * varrho * (nuTilde3(kappa, rho, gamma, lamda, phi, varrho) + rho) + \
          kappa * gamma * nuTilde3(kappa, rho, gamma, lamda, phi, varrho) + \
          2 * lamda * nuTilde3(kappa, rho, gamma, lamda, phi, varrho) * \
          (nuTilde3(kappa, rho, gamma, lamda, phi, varrho) + rho)) / \
       (nuTilde13(kappa, rho, gamma, lamda, phi, varrho) * \
        nuTilde23(kappa, rho, gamma, lamda, phi, varrho) * \
        (nuTilde3(kappa, rho, gamma, lamda, phi, varrho) + beta)) * \
       (np.exp(-beta*(T - t)) - np.exp((T - t) * nuTilde3(kappa, rho, gamma, lamda, phi, varrho))))

# Simulate a trajectory of the signal rate process

# $$ dI_t = -\beta I_t dt + \sigma dW_t, \qquad I_0 = \iota $$

# on $[0,T]$ and compute the signal

# $$ A_t = \int_0^t I_s ds \quad (0 \leq t \leq T) $$

def simulateSignalRatePath(beta, sigma, iota, T, timeGrid):

  dt = T/len(timeGrid)

  noise = np.random.normal(0, sigma * np.sqrt(dt), len(timeGrid))

  signalRateArray = np.zeros(len(timeGrid)) 

  signalRateArray[0] = iota

  for i in range(len(timeGrid)-1):

    signalRateArray[i+1] = signalRateArray[i] - beta * signalRateArray[i] * dt + noise[i]  

  return signalRateArray

# Define the (random) 3-dimensional linear ODE system

# $$ \begin{align}
# \frac{d\tilde X^{\tilde\nu}_t}{dt} = & - \tilde{\nu}_t = - \tilde w_1(T-t) \tilde X^{\tilde\nu}_t - \tilde w_2(T-t) \tilde Y^{\tilde\nu}_t - \tilde\zeta_t(\omega) \\
# \frac{d\tilde Y^{\tilde\nu}_t}{dt} = & + \gamma \tilde{\nu}_t - \rho \tilde Y^{\tilde\nu}_t = \gamma \tilde w_1(T-t) \tilde X^{\tilde\nu}_t + \left( \gamma \tilde w_2(T-t) - \rho \right) \tilde Y^{\tilde\nu}_t + \gamma \tilde\zeta_t(\omega) \\
# \frac{d X^{\hat v^i}_t}{dt} = & - \tilde{\nu}_t + \frac{R'(T-t)}{R(T-t)} \left( \tilde X^{\tilde\nu}_t - X^{\hat v^i}_t \right)
# = \left( \frac{R'(T-t)}{R(T-t)} - \tilde w_1(T-t) \right) \tilde X^{\tilde\nu}_t - \tilde w_2(T-t) \tilde Y^{\tilde\nu}_t - \frac{R'(T-t)}{R(T-t)} X^{\hat v^i}_t - \tilde\zeta_t(\omega)
# \end{align}
# $$

# using equation (3.18), the dynamics of $\tilde Y^{\tilde\nu}$, and equation (3.21).

def odesystem(z, t, T, kappa, rho, gamma, lamda, phi, varrho, beta, path):

  x = z[0]
  y = z[1]

  dxdt = -wTilde1(t, T, kappa, rho, gamma, lamda, phi, varrho) * x \
          - wTilde2(t, T, kappa, rho, gamma, lamda, phi, varrho) * y \
          - zetaTilde(t, T, kappa, rho, gamma, lamda, phi, varrho, beta, path)
  
  dydt = gamma * wTilde1(t, T, kappa, rho, gamma, lamda, phi, varrho) * x \
          + (gamma * wTilde2(t, T, kappa, rho, gamma, lamda, phi, varrho) - rho) * y \
          + gamma * zetaTilde(t, T, kappa, rho, gamma, lamda, phi, varrho, beta, path)
  
  dx_dt = list()

  for i in range(2,len(z)):
    dx_dt.append((Rprime(t, T, kappa, rho, gamma, lamda, phi, varrho)/R(t, T, kappa, rho, gamma, lamda, phi, varrho) - \
                    wTilde1(t, T, kappa, rho, gamma, lamda, phi, varrho)) * x - \
                    wTilde2(t, T, kappa, rho, gamma, lamda, phi, varrho) * y - \
                    Rprime(t, T, kappa, rho, gamma, lamda, phi, varrho)/R(t, T, kappa, rho, gamma, lamda, phi, varrho) * z[i] - \
                    zetaTilde(t, T, kappa, rho, gamma, lamda, phi, varrho, beta, path))
  
  return [dxdt, dydt] + dx_dt