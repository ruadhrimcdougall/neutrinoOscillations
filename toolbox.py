#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:51:07 2022

@author: ruadhri
"""

import numpy as np

def new_min(x012, y012):
    x_0, x_1, x_2 = x012
    y_0, y_1, y_2 = y012
    top = ((x_2**2 - x_1**2) * y_0 
           + (x_0**2 - x_2**2) * y_1 
           + (x_1**2 - x_0**2) * y_2)
    bottom = ((x_2 - x_1) * y_0 
           + (x_0 - x_2) * y_1 
           + (x_1 - x_0) * y_2)
    x_3 = 0.5 * top / bottom
    return x_3

def p2(x012, y012, x):
    x0 = x012[0]
    x1 = x012[1]
    x2 = x012[2]
    y0 = y012[0]
    y1 = y012[1]
    y2 = y012[2]
    first = ((x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2))) * y0
    second = ((x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2))) * y1
    third = ((x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1))) * y2
    return first + second + third

def p2_first_deriv(x012, y012, x):
    x0 = x012[0]
    x1 = x012[1]
    x2 = x012[2]
    y0 = y012[0]
    y1 = y012[1]
    y2 = y012[2]
    d = (x1 - x0) * (x2 - x0) * (x2 - x1)
    first = ((x - x1) + (x - x2)) * (x2 - x1) * y0 / d
    second = ((x - x0) + (x - x2)) * (x0 - x2) * y1 / d
    third = ((x - x0) + (x - x1)) * (x1 - x0) * y2 / d
    return first + second + third

def p2_second_deriv(x012, y012):
    x0 = x012[0]
    x1 = x012[1]
    x2 = x012[2]
    y0 = y012[0]
    y1 = y012[1]
    y2 = y012[2]
    d = (x1 - x0) * (x2 - x0) * (x2 - x1)
    first = 2 * (x2 - x1) * y0 / d
    second = 2 * (x0 - x2) * y1 / d
    third = 2 * (x1 - x0) * y2 / d
    return first + second + third

def curvature(x012, y012, x):
    first_deriv = p2_first_deriv(x012, y012, x)
    second_deriv = p2_second_deriv(x012, y012)
    curv = np.abs(second_deriv) / (1 + (first_deriv)**2)**(3/2)
    return curv

def secant(func, xi, xi_old):
    x_new = xi - func(xi) * (xi - xi_old) / (func(xi) - func(xi_old))
    return x_new