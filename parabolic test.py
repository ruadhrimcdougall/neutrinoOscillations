#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 13:02:52 2022

@author: ruadhri
"""

import numpy as np
import toolbox as tbx

def parabolic_min(interval, stop, func, iters=1000, 
                  std=False, curv=False):

    x_012 = np.linspace(interval[0], interval[1], 3)
    y_012 = np.zeros(3)
    new_mins = []
    for a in range(iters):
        y_012 = func(x_012)
        para_min = tbx.new_min(x_012, y_012)
        new_mins.append(para_min)
        if a > 0 and np.abs(new_mins[-1] - new_mins[-2]) < stop:
            print('minimum found after ' + str(a+1) + ' iterations')
            print('stopping condition met')
            print('value that minimises NLL is ' + str(para_min))
            return para_min, x_012, y_012
        if a != (iters-1):
            x_012 = np.array([x_012[-2], x_012[-1], para_min])
    print('minimum found after ' + str(iters) + ' iterations')
    print('yet to reach stopping condition')
    #print('value that minimises NLL is ' + str(np.around(para_min, 4)))
    print('Value that minimises NLL is ' + str(para_min))
    return para_min, x_012, y_012

def quad(x):
    return x**2 + 2*x - 1

#test on sine function

minimum_cos = parabolic_min([0.2, 2*np.pi-0.1], 1e-4, np.cos)
# converges after 5 iterations to very close to pi :)

#test on quadratic function x^2 + 2x - 1

minimum_cos = parabolic_min([-4, 1], 1e-4, quad)
# converges after 2 iterations to exactly -1 as required :)
