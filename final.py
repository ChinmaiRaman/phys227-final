#! /usr/bin/env python

"""
File: final.py
Copyright (c) 2016 Chinmai Raman
License: MIT
Course: PHYS227
Assignment: Final
Date: May 21, 2016
Email: raman105@mail.chapman.edu
Name: Chinmai Raman
Description: Final
"""

from __future__ import division
from unittest import TestCase
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Rossler():

    def __init__(self, c, dt = 0.001, T0 = 250, T = 500):
        self.dt = float(dt)
        self.T = float(T)
        self.T0 = T0
        self.c = float(c)
        self.t = np.linspace(0.0, self.T, self.T / self.dt)
        self.x = np.zeros(len(self.t))
        self.y = np.zeros(len(self.t))
        self.z = np.zeros(len(self.t))
        self.x0 = 0
        self.y0 = 0
        self.z0 = 0

    def f1(self, x, y, z, t):
        return -1 * y - 1 * z

    def f2(self, x, y, z, t):
        return x + 0.2 * y

    def f3(self, x, y, z, t):
        return 0.2 + z * (x - self.c)

    def run(self):
        """
        Implements the fourth order Runge-Kutta method of differentiation.
        """
        dt = self.dt
        x = self.x
        y = self.y
        z = self.z
        t = self.t
        f1 = self.f1
        f2 = self.f2
        f3 = self.f3

        for i in np.arange(0, len(t) - 1):
            k1_x = dt * f1(x[i], y[i], z[i], t[i])
            k1_y = dt * f2(x[i], y[i], z[i], t[i])
            k1_z = dt * f3(x[i], y[i], z[i], t[i])
            k2_x = dt * f1(x[i] + 0.5 * k1_x, y[i] + 0.5 * k1_y, z[i] + 0.5 * k1_z, t[i] + 0.5 * dt)
            k2_y = dt * f2(x[i] + 0.5 * k1_x, y[i] + 0.5 * k1_y, z[i] + 0.5 * k1_z, t[i] + 0.5 * dt)
            k2_z = dt * f3(x[i] + 0.5 * k1_x, y[i] + 0.5 * k1_y, z[i] + 0.5 * k1_z, t[i] + 0.5 * dt)

            k3_x = dt * f1(x[i] + 0.5 * k2_x, y[i] + 0.5 * k2_y, z[i] + 0.5 * k2_z, t[i] + 0.5 * dt)
            k3_y = dt * f2(x[i] + 0.5 * k2_x, y[i] + 0.5 * k2_y, z[i] + 0.5 * k2_z, t[i] + 0.5 * dt)
            k3_z = dt * f3(x[i] + 0.5 * k2_x, y[i] + 0.5 * k2_y, z[i] + 0.5 * k2_z, t[i] + 0.5 * dt)

            k4_x = dt * f1(x[i] + 0.5 * k3_x, y[i] + 0.5 * k3_y, z[i] + 0.5 * k3_z, t[i+1])
            k4_y = dt * f2(x[i] + 0.5 * k3_x, y[i] + 0.5 * k3_y, z[i] + 0.5 * k3_z, t[i+1])
            k4_z = dt * f3(x[i] + 0.5 * k3_x, y[i] + 0.5 * k3_y, z[i] + 0.5 * k3_z, t[i+1])

            x[i+1] = x[i] + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
            y[i+1] = y[i] + (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
            z[i+1] = z[i] + (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6

    def plotx(self):

        t = self.t
        T = self.T
        x = self.x

        fig, ax = plt.subplots()
        ax.grid(True)

        plt.plot(t, x, 'b-')
        plt.xlabel('t')
        plt.ylabel('x(t)')
        plt.title('x(t) vs t')
        plt.show(fig)
        plt.close(fig)

    def ploty(self):

        t = self.t
        T = self.T
        y = self.y

        fig, ax = plt.subplots()
        ax.grid(True)

        plt.plot(t, y, 'b-')
        plt.xlabel('t')
        plt.ylabel('y(t)')
        plt.title('y(t) vs t')
        plt.show(fig)
        plt.close(fig)

    def plotz(self):

        t = self.t
        T = self.T
        z = self.z

        fig, ax = plt.subplots()
        ax.grid(True)

        plt.plot(t, z, 'b-')
        plt.xlabel('t')
        plt.ylabel('z(t)')
        plt.title('z(t) vs t')
        plt.show(fig)
        plt.close(fig)

    def plotxy(self):

        t = self.t
        T0 = self.T0
        x = self.x[np.where(t >= T0)]
        y = self.y[np.where(t >= T0)]

        fig, ax = plt.subplots()
        ax.grid(True)

        plt.plot(x, y, 'b-')
        plt.xlabel('x(t)')
        plt.ylabel('y(t)')
        plt.title('y(t) vs x(t)')
        ax.axis([-12, 12, -12, 12])
        plt.show(fig)
        plt.close(fig)

    def plotyz(self):

        t = self.t
        T0 = self.T0
        y = self.y[np.where(t >= T0)]
        z = self.z[np.where(t >= T0)]

        fig, ax = plt.subplots()
        ax.grid(True)

        plt.plot(y, z, 'b-')
        plt.xlabel('y(t)')
        plt.ylabel('z(t)')
        plt.title('z(t) vs y(t)')
        ax.axis([-12, 12, 0, 25])
        plt.show(fig)
        plt.close(fig)

    def plotxz(self):

        t = self.t
        T0 = self.T0
        x = self.x[np.where(t >= T0)]
        z = self.z[np.where(t >= T0)]

        fig, ax = plt.subplots()
        ax.grid(True)

        plt.plot(x, z, 'b-')
        plt.xlabel('x(t)')
        plt.ylabel('z(t)')
        plt.title('z(t) vs x(t)')
        ax.axis([-12, 12, 0, 25])
        plt.show(fig)
        plt.close(fig)

    def plotxyz(self):

        t = self.t
        T0 = self.T0
        x = self.x[np.where(t >= T0)]
        y = self.y[np.where(t >= T0)]
        z = self.z[np.where(t >= T0)]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.grid(True)

        plt.plot(x, y, z, 'b-')
        plt.xlabel('x(t)')
        plt.ylabel('y(t)')
        ax.set_zlabel("z(t)")
        plt.title('z(t) vs y(t) vs x(t)')
        ax.axis([-12, 12, -12, 12])
        ax.set_zlim((0, 25))
        plt.show(fig)
        plt.close(fig)

def findmaxima(c, dim):
    """
    finds the local maxima of x given a particular c
    """
    ros = Rossler(c)
    ros.run()
    
    if dim == 'x':
        var = ros.x
    elif dim == 'y':
        var = ros.y
    elif dim == 'z':
        var = ros.z

    values = var[np.where(ros.t >= ros.T0)]
    local_max = values[np.where((np.r_[True, values[1:] > values[:-1]] & np.r_[values[:-1] > values[1:], True]) == True)]
    return local_max[local_max > 0]

def plotmaxima(dim):
    """
    plots local maxima of x,y, or z vs c
    """
    c_values = np.linspace(2, 6, 41)
    var = [findmaxima(c, dim)[-17:] for c in c_values]
    fig = plt.figure(1)
    plt.plot(c_values, [elem for elem in var], 'b-')
    plt.xlabel('c')
    plt.ylabel(dim)
    plt.ylim([3,12])
    plt.title(dim + ' local maxes vs. c')
    plt.show()

class Test_Ros(TestCase):
    def test_ros(self):
        
        T = 500
        dt = 0.001
        
        x_test = dt * np.arange(0, T / dt)
        y_test = dt * np.arange(0, T / dt)
        z_test = dt * np.arange(0, T / dt)
        
        def f1(x, y, z, t):
            return 1
        
        def f2(x, y, z, t):
            return 1
        
        def f3(x, y, z, t):
            return 1
        
        test = Rossler(2)
        test.f1 = f1
        test.f2 = f2
        test.f3 = f3
        test.run()
        
        print test.x[-10:]
        print x_test[-10:]
        
        assert (abs(test.x - x_test) < 1e-3).all() and (abs(test.y - y_test) < 1e-3).all() and (abs(test.z - z_test) < 1e-3).all(), 'Failure'