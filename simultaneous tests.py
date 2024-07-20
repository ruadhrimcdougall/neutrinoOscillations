#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 13:20:43 2022

@author: ruadhri
"""

import toolbox as tbx
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

#%%

def test_func(x, y):
    return 4*x**2 + 2*x*y + 3*y**2 + 2*x + 2*y + 1

def grad_test_func(x, y):
    x_deriv = 8*x + 2*y + 2
    y_deriv = 2*x + 6*y + 2
    grad = np.array([x_deriv, y_deriv])
    return grad

def grad_search(initial_guess, learn_rate, func, grad, stop, iters=1000):
    minimum = np.array(initial_guess)
    trial_NLL = []
    mins = [initial_guess]
    for i in range(iters):
        #if len(initial_guess) == 2:
        change = learn_rate * grad(minimum[0], minimum[1])
        new_nll = func(minimum[0],minimum[1])
        #elif len(initial_guess) == 3:
        #    change = learn_rate * self.nll_grad(minimum[0], minimum[1], minimum[2])
        #    new_nll = self.nll(self.new_lambda(minimum[0], 
        #                                       minimum[1], 
        #                                       minimum[2])
        #                       )
        minimum = minimum - change
        trial_NLL.append(new_nll)
        mins.append(minimum)
        if i > 0 and np.abs(trial_NLL[-2] - trial_NLL[-1]) < stop:
            print('Converged after ' + str(i+1) + ' iterations! :)')
            #if len(initial_guess) == 2:
                #print('Has not converged after ' + str(iters) + 'iterations :(')
            print('Min. location is x = ' + str(minimum[0]) 
                  + ' and y = ' + str(minimum[-1]))
            #elif len(initial_guess) == 3:
            #    print('Min. location is theta = ' + str(minimum[0]) 
            #          + ' \nand mass = ' + str(minimum[1])
            #          + ' \nand alpha = ' + str(minimum[2]))
            mins = np.array([mins]).reshape(len(mins), len(initial_guess)).T
            return minimum, mins
    print('Has not converged after ' + str(iters) + 'iterations :(')
    #if len(initial_guess) == 2:
    print('Min. location is x = ' + str(minimum[0]) 
          + ' and y = ' + str(minimum[-1]))
    #elif len(initial_guess) == 3:
    #     print('Min. location is theta = ' + str(minimum[0]) 
    #           + ' \nand mass = ' + str(minimum[1])
    #           + ' \nand alpha = ' + str(minimum[2]))
    mins = np.array([mins]).reshape(len(mins), len(initial_guess)).T
    return minimum, mins

def quasi_newton(initial_guess, learn_rate, func, grad, stop, iters=1000):
    def update_step(update_matrix, delta, gamma):
        new_update = (update_matrix 
                      + (np.outer(delta, delta) / np.dot(gamma, delta))
                      - (np.dot(update_mat, 
                                np.dot(np.outer(gamma, gamma), 
                                       update_matrix))
                         / (np.dot(gamma, np.dot(update_matrix, gamma)))
                         )
                      )
        return new_update
    
    minimum = np.array(initial_guess)
    update_mat = np.identity(len(initial_guess))
    mins = [minimum]
    #if len(initial_guess) == 2:
    grads = [grad(minimum[0], minimum[1])]
    #elif len(initial_guess) == 3:
    #    grads = [self.nll_grad(minimum[0], minimum[1], minimum[2])]
    #else:
    #    raise ValueError('initial_guess must be a list of length 2 or 3')
    trial_NLL = []
    for i in range(iters):
        #if len(initial_guess) == 2:
        change = learn_rate * np.dot(update_mat, 
                                     grad(minimum[0], minimum[1])
                                     )
        minimum = minimum - change
        new_grad = grad(minimum[0], minimum[1])
        new_nll = func(minimum[0], minimum[1])
        #elif len(initial_guess) == 3:
        #    change = learn_rate * np.dot(update_mat, 
        #                                 self.nll_grad(minimum[0], minimum[1], minimum[2])
        #                                 )
        #    minimum = minimum - change
        #    new_grad = self.nll_grad(minimum[0], minimum[1], minimum[2])
        #    new_nll = self.nll(self.new_lambda(minimum[0], 
        #                                       minimum[1], 
        #                                       minimum[2])
        #                       )
        mins.append(minimum)
        grads.append(new_grad)
        delt = mins[-1] - mins[-2]
        gamm = grads[-1] - grads[-2]
        update_mat = update_step(update_mat, delt, gamm)
        trial_NLL.append(new_nll)
        if i > 0 and np.abs(trial_NLL[-2] - trial_NLL[-1]) < stop:
            print(trial_NLL[-2] - trial_NLL[-1])
            #change = np.abs(mins[-2] - mins[-1])
            #err = change
            print('Converged after ' + str(i+1) + ' iterations! :)')
           # if len(initial_guess) == 2:
            print('Min. location is x = ' + str(minimum[0])# + ' +/- ' + str(err[0]) 
                  + 'and y = ' + str(minimum[-1]))# + ' +/- ' + str(err[-1]))
            #elif len(initial_guess) == 3:
            #    print('Min. location is theta = ' + str(minimum[0])# + ' +/- ' + str(err[0]) 
            #          + '\nand mass = ' + str(minimum[1])# + ' +/- ' + str(err[1])
            #          + '\nand alpha = ' + str(minimum[2]))# + ' +/- ' + str(err[2]))
            mins = np.array([mins]).reshape(len(mins), len(initial_guess)).T
            #print(mins.shape)
            #plt.plot(mins[:, 0], mins[:,1]*1e3, color='black')
            return minimum, mins#, err
    #err = change
    print('Has not converged after ' + str(iters) + 'iterations :(')
    #if len(initial_guess) == 2:
    print('Min. location is x = ' + str(minimum[0])# + ' +/- ' + str(err[0]) 
          + 'and y = ' + str(minimum[-1]))# + ' +/- ' + str(err[-1]))
    #elif len(initial_guess) == 3:
    #    print('Min. location is theta = ' + str(minimum[0])# + ' +/- ' + str(err[0]) 
    #          + '\nand mass = ' + str(minimum[1])# + ' +/- ' + str(err[1])
    #          + '\nand alpha = ' + str(minimum[2]))# + ' +/- ' + str(err[2]))
    mins = np.array([mins]).reshape(len(mins), len(initial_guess)).T
    return minimum, mins#, err

#%% check where the minimum should be

test_x = np.linspace(-1, 1, 101)
test_y = np.linspace(-1, 1, 101)
X, Y = np.meshgrid(test_x, test_y)
Z = test_func(X, Y)

#fig = plt.figure(figsize=(6,5))
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(X, Y, Z, cmap=cm.jet)

min_est_ind = np.where(Z == np.amin(Z))
print(min_est_ind)

print(X[min_est_ind])
print(Y[min_est_ind])

#%% check grad descent works

print()
print('Grad Search')
grad_min, grad_min_trials = grad_search([-10, -10], 1e-1, test_func, grad_test_func, 1e-9)
# converges really quickly and to high accuracy :)
# converges to x = -0.18181... and y = -0.27272... as expected :)

#%% check quasi-newton works

print()
print('Quasi Newton')
quasi_min, quasi_min_trials = quasi_newton([-10, -10], 1e-1, test_func, grad_test_func, 1e-9)
# also converges really quickly and to high accuracy :)
# also converges to x = -0.18181... and y = -0.27272... as expected :)

# grad search seems to converge a bit quicker for similar accuracy
# but my methods both work
# may need to check derivatives of nll in that case


