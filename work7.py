#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

N = 1000
gamma2 = 1. / (2. * np.sqrt(2.))

def Cxx(fx, fy):

    return 4. * (np.sin(fx/2.)**2 + gamma2 * np.sin((fx-fy)/2.)**2
            + gamma2 * np.sin((fx+fy)/2.)**2)

def Cyy(fx, fy):

    return 4. * (np.sin(fy/2.)**2 + gamma2 * np.sin((fx-fy)/2.)**2
            + gamma2 * np.sin((fx+fy)/2.)**2)

def Cxy(fx, fy):

    return 4.*gamma2 * (np.sin((fx-fy)/2.)**2 - np.sin((fx+fy)/2.)**2)

def omega1(fx, fy):

    omega_sq = (Cxx(fx, fy) + Cyy(fx, fx))/2. - np.sqrt((Cxx(fx, fy) + Cyy(fx, fy))**2/4. -
            (Cxx(fx, fy)*Cyy(fx, fy) - Cxy(fx, fy)*Cxy(fx, fy)))

    return np.sqrt(omega_sq)

def omega2(fx, fy):

    omega_sq = (Cxx(fx, fy) + Cyy(fx, fx))/2. + np.sqrt((Cxx(fx, fy) + Cyy(fx, fy))**2/4. -
            (Cxx(fx, fy)*Cyy(fx, fy) - Cxy(fx, fy)*Cxy(fx, fy)))

    return np.sqrt(omega_sq)

def main():
    '''
    main function
    '''
    f_data = np.zeros(3*N+1)
    omega1_data = np.zeros(3*N+1)
    omega2_data = np.zeros(3*N+1)

    for index in range(N):
        f_data[index] = index * np.pi/N
        omega1_data[index] = omega1(index * np.pi/N, 0.)
        omega2_data[index] = omega2(index * np.pi/N, 0.)

    for index in range(N):
        f_data[N + index] = np.pi + index * np.pi/N
        omega1_data[N + index] = omega1(np.pi, index * np.pi/N)
        omega2_data[N + index] = omega2(np.pi, index * np.pi/N)

    for index in range(N+1):
        f_data[2*N + index] = 2.*np.pi + index * np.pi/N
        omega1_data[2*N + index] = omega1(np.pi - index * np.pi/N, np.pi - index * np.pi/N)
        omega2_data[2*N + index] = omega2(np.pi - index * np.pi/N, np.pi - index * np.pi/N)

    plt.plot(f_data, omega1_data, f_data, omega2_data)

    plt.show()

if __name__ == '__main__':
    main()
