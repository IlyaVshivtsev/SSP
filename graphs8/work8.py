#! /usr/bin/python3

import numpy as np
from scipy.special import jv
from scipy.integrate import quad

import matplotlib.pyplot as plt
import sys

def func(p, omega, gamma):
    return np.cos(p*(omega**2-0.5-gamma)) * jv(0, p/2) * jv(0, gamma*p/2)**2

def g1(omega, gamma):
    return 2*omega/np.pi * quad(func, 0, 100, args=(omega, gamma))[0]

def g2(omega, gamma):
    if omega < np.pi * (3*gamma / (4*np.pi))**(1/3):
        return 4 * omega**2 / (np.pi**2 * gamma)
    else:
        return 0

def main(args):
    gamma = 1/float(args[1])
    print(gamma)
    
    omega_data = np.linspace(0, 2, 1024)
    g1_data = np.zeros(1024)
    g2_data = np.zeros(1024)

    for i, omega in np.ndenumerate(omega_data):
        g1_data[i] = g1(omega, gamma)
        g2_data[i] = g2(omega, gamma)

    plt.plot(omega_data, g1_data, omega_data, g2_data)
    plt.show()

if __name__ == '__main__':
    main(sys.argv)
