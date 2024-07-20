#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:44:06 2022

@author: ruadhri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import toolbox as tbx


class Minimiser:
    '''
    Units
    -----
    Energy - GeV
    Distance - km
    '''
    def __init__(self, path='Data/neutrino_data.txt', dist=295):
        data_df = pd.read_csv(path, header=None)
        data_np = data_df.to_numpy()
        self.__events = data_np[1:201].reshape(-1).astype('int')
        self.__flux_sim = data_np[202:].astype('float64').reshape(-1)
        self.__bin_edges = np.arange(0, 10.05, 0.05)
        self.__num_bins = len(self.__events)
        self.__bin_centres = np.linspace(0.025, 9.975, self.__num_bins)
        fact = np.zeros(self.__num_bins)
        for i in range(self.__num_bins):
            fact[i] = np.math.factorial(self.__events[i])
        self.__log_m_fact = np.log(fact)
        self.__dist = dist
        #print(self.__log_m_fact)
    
    def events(self):
        return self.__events
    
    def flux(self):
        return self.__flux_sim
    
    def centres(self):
        return self.__bin_centres
    
    def initial_hists(self):
        plt.figure()
        plt.stairs(self.__events, self.__bin_edges, fill=True)
        plt.xlabel('Energy (GeV)')
        plt.ylabel('Observed Events')
        plt.tight_layout()

        plt.figure()
        plt.stairs(self.__flux_sim, self.__bin_edges, fill=True)
        plt.xlabel('Energy (GeV)')
        plt.ylabel('Simulated Unoscillated Flux')
        plt.tight_layout()
    
    def prob_oscillation(self, energy, mixing_ang, mass_sqrd):
        prob = (1 
                - np.square(np.sin(2 * mixing_ang)) 
                * np.square(np.sin(1.267 * mass_sqrd * self.__dist / energy))
                )
        return prob
    
    def event_prediction(self, mixing_ang_guess, mass_sqrd_guess):
        if isinstance(mixing_ang_guess, np.ndarray) and isinstance(mass_sqrd_guess, np.ndarray):
            if mixing_ang_guess.ndim == 2 and mass_sqrd_guess.ndim == 2:
                lambda_i = (self.__flux_sim[:, None, None] 
                            * self.prob_oscillation(self.__bin_centres[:, None, None], 
                                                    mixing_ang_guess, 
                                                    mass_sqrd_guess)
                            )
                return lambda_i
        else:
            if isinstance(mixing_ang_guess, np.ndarray):
                if mixing_ang_guess.ndim == 1:
                    mixing_ang_guess = mixing_ang_guess.reshape(len(mixing_ang_guess), 1)
            elif isinstance(mass_sqrd_guess, np.ndarray):
                if mass_sqrd_guess.ndim == 1:
                    mass_sqrd_guess = mass_sqrd_guess.reshape(len(mass_sqrd_guess), 1)
            lambda_i = self.__flux_sim * self.prob_oscillation(self.__bin_centres, 
                                                               mixing_ang_guess, 
                                                               mass_sqrd_guess)
            return lambda_i
    
    def nll(self, exp_events):
        if exp_events.ndim == 3:
            nll = (exp_events 
                   - self.__events[:,None,None] * np.log(exp_events) 
                   + self.__log_m_fact[:,None,None])
        else:
            nll = (exp_events 
                   - self.__events * np.log(exp_events) 
                   + self.__log_m_fact)
        if nll.ndim == 1:
            return np.sum(nll)
        elif nll.ndim == 2:
            return np.sum(nll, axis=1)
        elif nll.ndim == 3:
            return np.sum(nll, axis=0)
        else:
            print("nll wasn't calculated :(")
    
    def parabolic_min(self, interval, stop, mass_sqrd_guess, iters=1000, 
                      std=False, curv=False):
        
        def sec_std(min_val, stop, guess_shift=0.05, iters=100000):
            def func(theta):
                func_plus = (self.nll(self.event_prediction(theta, 
                                                            mass_sqrd_guess)) 
                             - self.nll(self.event_prediction(min_val, 
                                                              mass_sqrd_guess)) 
                             - 0.5)
                return func_plus
            old_x_plus = min_val + guess_shift
            old_x_minus = min_val - guess_shift
            oldold_x_plus = min_val
            oldold_x_minus = min_val
            guess = []
            its = []
            for i in range(iters):
                new_zero_plus = tbx.secant(func, old_x_plus, oldold_x_plus)
                new_zero_minus = tbx.secant(func, old_x_minus, oldold_x_minus)
                change1 = np.abs(new_zero_plus - old_x_plus)
                change2 = np.abs(new_zero_minus - old_x_minus)
                guess.append(np.abs(new_zero_plus - new_zero_minus))
                its.append(i)
                if change1 < stop or change2 < stop:
                #if np.abs(change) < stop:
                    print('Error iters - ' + str(i+1))
                    #return np.abs(new_zero_plus - min_val)*2
                    plt.plot(its, guess)
                    return np.abs(new_zero_plus - new_zero_minus)
                else:
                    oldold_x_plus = old_x_plus
                    old_x_plus = new_zero_plus
                    oldold_x_minus = old_x_minus
                    old_x_minus = new_zero_minus
            print('Error iters - ' + str(i+1))
            #return np.abs(new_zero_plus - min_val)*2
            #plt.plot(its, guess)
            return np.abs(new_zero_plus - new_zero_minus)
        
        def std_newton(min_val, stop, guess_shift=0.01, iters=100000):
            old_zero_plus = min_val + guess_shift
            old_zero_minus = min_val - guess_shift
            guess = []
            its = []
            for i in range(iters):
                func_plus = (self.nll(self.event_prediction(old_zero_plus, 
                                                            mass_sqrd_guess)) 
                             - self.nll(self.event_prediction(min_val, 
                                                              mass_sqrd_guess)) 
                             - 0.5)
                new_zero_plus = (old_zero_plus 
                                - func_plus 
                                / self.dnll_dtheta_23(old_zero_plus, 
                                                      mass_sqrd_guess))
                func_minus = (self.nll(self.event_prediction(old_zero_minus, 
                                                             mass_sqrd_guess)) 
                             - self.nll(self.event_prediction(min_val, 
                                                              mass_sqrd_guess))
                             - 0.5) 
                new_zero_minus = (old_zero_minus 
                                 - func_minus 
                                 / self.dnll_dtheta_23(old_zero_minus, 
                                                       mass_sqrd_guess))
                change1 = np.abs(new_zero_plus - old_zero_plus)
                change2 = np.abs(new_zero_minus - old_zero_minus)
                guess.append(np.abs(new_zero_plus - new_zero_minus))
                its.append(i)
                if change1 < stop or change2 < stop:
                #if np.abs(change) < stop:
                    print('Error iters - ' + str(i+1))
                    #return np.abs(new_zero_plus - min_val)*2
                    plt.plot(its, guess)
                    return np.abs(new_zero_plus - new_zero_minus)
                else:
                    old_zero_plus = new_zero_plus
                    old_zero_minus = new_zero_minus
            print('Error iters - ' + str(i+1))
            #return np.abs(new_zero_plus - min_val)*2
            #plt.figure()
            plt.plot(its, guess)
            return np.abs(new_zero_plus - new_zero_minus)
            
        
        def curv_err(x012, y012):
            curv = tbx.p2_second_deriv(x012, y012)
            err = 1 / curv #* (x - x012[0]) * (x - x012[1]) * (x - x012[2])
            #err = np.abs((self.nll(self.event_prediction(x, mass_sqrd_guess)) - tbx.p2(x012, y012, x)) / self.nll(self.event_prediction(x, mass_sqrd_guess))) * x
            return err
        
        x_012 = np.linspace(interval[0], interval[1], 3)
        y_012 = np.zeros(3)
        new_mins = []
        for a in range(iters):
            lambdas = self.event_prediction(x_012, 
                                            mass_sqrd_guess)
            y_012 = self.nll(lambdas)
            para_min = tbx.new_min(x_012, y_012)
            new_mins.append(para_min)
            if a > 0 and np.abs(new_mins[-1] - new_mins[-2]) < stop:
                if std == True and curv == True:
                    raise ValueError('std and curv cannot both be True')
                elif std == True:
                    #err = std_newton(para_min, stop)
                    err = sec_std(para_min, stop)
                    #err = std_newton(para_min, stop)
                elif curv == True:
                    err = curv_err(x_012, y_012)
                else:
                    raise ValueError('std and curv cannot both be False')
                print('minimum found after ' + str(a+1) + ' iterations')
                print('stopping condition met')
                print('value that minimises NLL is ' + str(para_min) + ' +/- ' 
                      + str(err))
                return para_min, err, x_012, y_012
            if a != (iters-1):
                x_012 = np.array([x_012[-2], x_012[-1], para_min])
        print('minimum found after ' + str(iters) + ' iterations')
        print('yet to reach stopping condition')
        #print('value that minimises NLL is ' + str(np.around(para_min, 4)))
        print('Value that minimises NLL is ' + str(para_min))
        if std == True and curv == True:
            raise ValueError('std and curv cannot both be True')
        elif std == True:
            err = sec_std(para_min, stop)
        elif curv == True:
            err = curv_err(x_012, y_012, para_min)
        else:
            raise ValueError('std and curv cannot both be False')
        return para_min, err, x_012, y_012
    
    def univariate(self, theta_intv, mass_intv, stop, stop_err, iters=1000):
        # I choose the midpoint of the interval as the first guess for the minimum
        t_012 = np.linspace(theta_intv[0], theta_intv[1], 3)
        m_012 = np.linspace(mass_intv[0], mass_intv[1], 3)
        new_mins_t = []
        new_mins_m = [m_012[1]]
        #nlls = []
        #nlls_t = []
        y_012_t = np.zeros(3)
        y_012_m = np.zeros(3)
        stop_m = False
        stop_t = False
        for a in range(iters):
            for i in range(3):
                if stop_m != True and a>0:
                    lam_m = self.event_prediction(new_mins_t[-1], 
                                                  m_012[i])
                    y_012_m[i] = self.nll(lam_m)
                if stop_t != True:
                    lam_t = self.event_prediction(t_012[i], 
                                                  new_mins_m[-1])
                    y_012_t[i] = self.nll(lam_t)
            if stop_m != True and a>0:
                para_min_m = tbx.new_min(m_012, y_012_m)
                #new_nll = self.nll()
                new_mins_m.append(para_min_m)
                #if np.abs(new_mins_m[-1] - new_mins_m[-2])/new_mins_m[-2] < stop:
                if np.abs(new_mins_m[-1] - new_mins_m[-2]) < stop:
                #if a>1 and np.abs(nlls[-1] - nlls[-2]) < stop:
                    stop_m = True
                    print('mass min found after ' + str(a+1) + ' iterations')
                m_012 = np.array([m_012[-2], m_012[-1], para_min_m])
            if stop_t != True:
                para_min_t = tbx.new_min(t_012, y_012_t)
                new_mins_t.append(para_min_t)
                #if a>0 and np.abs(new_mins_t[-1] - new_mins_t[-2])/new_mins_t[-2] < stop:
                if a>0 and np.abs(new_mins_t[-1] - new_mins_t[-2]) < stop:
                #if a>1 and np.abs(nlls[-1] - nlls[-2]) < stop:
                    # based off percentage change in theta vals
                    stop_t = True
                    print('theta min found after ' + str(a+1) + ' iterations')
                t_012 = np.array([t_012[-2], t_012[-1], para_min_t])
            #nlls.append(self.nll(self.event_prediction(new_mins_t[-1], new_mins_m[-1])))
            #print(nlls[-1])
            if stop_m and stop_t:
                err = self.simult_error([para_min_t, para_min_m], stop_err)
                print('minimum found after ' + str(a+1) + ' iterations')
                print('stopping condition met')
                print('values that minimise NLL are theta = ' + str(para_min_t) + ' +/- ' + str(err[0])
                      + ' and m = ' + str(para_min_m) + ' +/- ' + str(err[1]))
                return [para_min_t, para_min_m], err, new_mins_m, new_mins_t
            #else:
            #    print('Weird Error')
        err = None
        print('minimum found after ' + str(iters) + ' iterations')
        print('yet to reach stopping condition')
        print('values that minimise NLL are theta = ' + str(para_min_t) + ' and m = ' + str(para_min_m))
        return [para_min_t, para_min_m], err, new_mins_m, new_mins_t
    
    def dnll_dtheta_23(self, mixing_ang, mass_sqrd, alpha=None):
        mass_arg = 1.267 * mass_sqrd * self.__dist / self.__bin_centres
        osc_prob = self.prob_oscillation(self.__bin_centres, mixing_ang, mass_sqrd)
        deriv_lambda = (-4 * self.__flux_sim * (np.sin(mass_arg)**2) 
                        * np.sin(2*mixing_ang) * np.cos(2*mixing_ang))
        if alpha != None:
            deriv = deriv_lambda * alpha * self.__bin_centres - (self.__events / (osc_prob * self.__flux_sim)) * deriv_lambda
        elif alpha == None:
            deriv = deriv_lambda - (self.__events / (osc_prob * self.__flux_sim)) * deriv_lambda
        #print(deriv.shape)
        return np.sum(deriv)
    
    def dnll_dm2_23(self, mixing_ang, mass_sqrd, alpha=None):
        mass_arg = 1.267 * mass_sqrd * self.__dist / self.__bin_centres
        osc_prob = self.prob_oscillation(self.__bin_centres, mixing_ang, mass_sqrd)
        deriv_lambda = (-2 * 1.267 * self.__flux_sim * self.__dist * (np.sin(2*mixing_ang))**2
                        * np.cos(mass_arg)*np.sin(mass_arg) 
                        / self.__bin_centres)
        if alpha != None:
            deriv = deriv_lambda * alpha - (self.__events / (osc_prob * self.__flux_sim)) * deriv_lambda
        elif alpha == None:
            deriv = deriv_lambda - (self.__events / (osc_prob * self.__flux_sim)) * deriv_lambda 
        #print(deriv.shape)
        return np.sum(deriv)
    
    def dnll_dalpha(self, mixing_ang, mass_sqrd, alpha):
        deriv = (self.__bin_centres * self.event_prediction(mixing_ang, 
                                                            mass_sqrd) 
                 - self.__events / alpha)
        return np.sum(deriv)
    
    def nll_grad(self, mixing_ang, mass_sqrd, alpha=None):
        if alpha == None:
            grad = np.array([self.dnll_dtheta_23(mixing_ang, mass_sqrd),
                             self.dnll_dm2_23(mixing_ang, mass_sqrd)])
        elif alpha != None:
            grad = np.array([self.dnll_dtheta_23(mixing_ang, mass_sqrd, alpha),
                            self.dnll_dm2_23(mixing_ang, mass_sqrd, alpha),
                            self.dnll_dalpha(mixing_ang, mass_sqrd, alpha)])
        return grad
    
    def grad_search(self, initial_guess, learn_rate, stop, err_stop, iters=1000):
        minimum = np.array(initial_guess)
        trial_NLL = []
        mins = [initial_guess]
        for i in range(iters):
            if i%1000 == 0:
                print(minimum)
            if len(initial_guess) == 2:
                change = learn_rate * self.nll_grad(minimum[0], minimum[1])
                new_nll = self.nll(self.event_prediction(minimum[0], 
                                                         minimum[1])
                                   )
            elif len(initial_guess) == 3:
                change = learn_rate * self.nll_grad(minimum[0], minimum[1], minimum[2])
                new_nll = self.nll(self.new_lambda(minimum[0], 
                                                   minimum[1], 
                                                   minimum[2])
                                   )
            minimum = minimum - change
            trial_NLL.append(new_nll)
            mins.append(minimum)
            if i > 0 and np.abs(trial_NLL[-2] - trial_NLL[-1]) < stop:
                print('Converged after ' + str(i+1) + ' iterations! :)')
                err = self.simult_error(minimum, stop=err_stop)
                if len(initial_guess) == 2:
                    #print('Has not converged after ' + str(iters) + 'iterations :(')
                    print('Min. location is theta = ' + str(minimum[0]) + ' +/- ' + str(err[0])
                          + ' and mass = ' + str(minimum[-1]) + ' +/- ' + str(err[1]))
                elif len(initial_guess) == 3:
                    print('Min. location is theta = ' + str(minimum[0]) + ' +/- ' + str(err[0])
                          + ' \nand mass = ' + str(minimum[1])  + ' +/- ' + str(err[1])
                          + ' \nand alpha = ' + str(minimum[2]) + ' +/- ' + str(err[2]))
                mins = np.array([mins]).reshape(len(mins), len(initial_guess)).T
                return minimum, err, mins
        print('Has not converged after ' + str(iters) + 'iterations :(')
        err = self.simult_error(minimum, stop=err_stop)
        #err = None
        if len(initial_guess) == 2:
            print('Min. location is theta = ' + str(minimum[0])  + ' +/- ' + str(err[0])
                  + ' and mass = ' + str(minimum[-1]) + ' +/- ' + str(err[1]))
        elif len(initial_guess) == 3:
             print('Min. location is theta = ' + str(minimum[0]) + ' +/- ' + str(err[0])
                   + ' \nand mass = ' + str(minimum[1]) + ' +/- ' + str(err[1])
                   + ' \nand alpha = ' + str(minimum[2]) + ' +/- ' + str(err[2]))
        mins = np.array([mins]).reshape(len(mins), len(initial_guess)).T
        return minimum, err, mins
    
    def quasi_newton(self, initial_guess, learn_rate, stop, err_stop, iters=1000):
        def update_step(update_matrix, delta, gamma):
            new_update = (update_matrix 
                          + (np.outer(delta, delta) / np.dot(gamma, delta))
                          - (np.dot(update_matrix, 
                                    np.dot(np.outer(gamma, gamma), 
                                           update_matrix))
                             / (np.dot(gamma, np.dot(update_matrix, gamma)))
                             )
                          )
            return new_update
        
        minimum = np.array(initial_guess)
        update_mat = np.identity(len(initial_guess))
        mins = [minimum]
        if len(initial_guess) == 2:
            grads = [self.nll_grad(minimum[0], minimum[1])]
        elif len(initial_guess) == 3:
            grads = [self.nll_grad(minimum[0], minimum[1], minimum[2])]
        else:
            raise ValueError('initial_guess must be a list of length 2 or 3')
        trial_NLL = []
        for i in range(iters):
            if len(initial_guess) == 2:
                change = learn_rate * np.dot(update_mat, 
                                             self.nll_grad(minimum[0], minimum[1])
                                             )
                #print(change)
                minimum = minimum - change
                new_grad = self.nll_grad(minimum[0], minimum[1])
                new_nll = self.nll(self.event_prediction(minimum[0], 
                                                         minimum[1])
                                   )
            elif len(initial_guess) == 3:
                change = learn_rate * np.dot(update_mat, 
                                             self.nll_grad(minimum[0], minimum[1], minimum[2])
                                             )
                minimum = minimum - change
                new_grad = self.nll_grad(minimum[0], minimum[1], minimum[2])
                new_nll = self.nll(self.new_lambda(minimum[0], 
                                                   minimum[1], 
                                                   minimum[2])
                                   )
            mins.append(minimum)
            grads.append(new_grad)
            delt = mins[-1] - mins[-2]
            gamm = grads[-1] - grads[-2]
            update_mat = update_step(update_mat, delt, gamm)
            print(update_mat)
            trial_NLL.append(new_nll)
            if i > 0 and np.abs(trial_NLL[-2] - trial_NLL[-1]) < stop:
                print(trial_NLL[-2] - trial_NLL[-1])
                #change = np.abs(mins[-2] - mins[-1])
                err = self.simult_error(minimum, err_stop)
                print('Converged after ' + str(i+1) + ' iterations! :)')
                if len(initial_guess) == 2:
                    print('Min. location is theta = ' + str(minimum[0]) + ' +/- ' + str(err[0]) 
                          + '\nand mass = ' + str(minimum[-1]) + ' +/- ' + str(err[-1]))
                elif len(initial_guess) == 3:
                    print('Min. location is theta = ' + str(minimum[0]) + ' +/- ' + str(err[0]) 
                          + '\nand mass = ' + str(minimum[1]) + ' +/- ' + str(err[1])
                          + '\nand alpha = ' + str(minimum[2]) + ' +/- ' + str(err[2]))
                mins = np.array([mins]).reshape(len(mins), len(initial_guess)).T
                #print(mins.shape)
                #plt.plot(mins[:, 0], mins[:,1]*1e3, color='black')
                return minimum, err, mins
        #err = change
        print('Has not converged after ' + str(iters) + 'iterations :(')
        err = self.simult_error(minimum, err_stop)
        #err = None
        if len(initial_guess) == 2:
            print('Min. location is theta = ' + str(minimum[0])# + ' +/- ' + str(err[0]) 
                  + '\nand mass = ' + str(minimum[-1]))# + ' +/- ' + str(err[-1]))
        elif len(initial_guess) == 3:
            print('Min. location is theta = ' + str(minimum[0])# + ' +/- ' + str(err[0]) 
                  + '\nand mass = ' + str(minimum[1])# + ' +/- ' + str(err[1])
                  + '\nand alpha = ' + str(minimum[2]))# + ' +/- ' + str(err[2]))
        mins = np.array([mins]).reshape(len(mins), len(initial_guess)).T
        return minimum, err, mins
    
    def new_lambda(self, mixing_ang_guess, mass_sqrd_guess, alpha):
        new = (self.event_prediction(mixing_ang_guess, mass_sqrd_guess) 
               * alpha 
               * self.__bin_centres)
        return new
    
    def simult_error(self, min_locs, stop, guess_shift=0.05, iters=100000):
        def secant_method(min_val, func, stop, guess_shift, iters):
            old_x_plus = min_val + guess_shift
            old_x_minus = min_val - guess_shift
            oldold_x_plus = min_val
            oldold_x_minus = min_val
            for i in range(iters):
                new_zero_plus = tbx.secant(func, old_x_plus, oldold_x_plus)
                new_zero_minus = tbx.secant(func, old_x_minus, oldold_x_minus)
                change1 = np.abs(new_zero_plus - old_x_plus)
                change2 = np.abs(new_zero_minus - old_x_minus)
                if change1 < stop or change2 < stop:
                    print('Error iters - ' + str(i+1))
                    return np.abs(new_zero_plus - new_zero_minus)
                else:
                    oldold_x_plus = old_x_plus
                    old_x_plus = new_zero_plus
                    oldold_x_minus = old_x_minus
                    old_x_minus = new_zero_minus
            print('Error iters - ' + str(i+1))
            return np.abs(new_zero_plus - new_zero_minus)
        def root_func2_theta(theta):
            func_plus = (self.nll(self.event_prediction(theta, 
                                                        min_locs[1])) 
                         - self.nll(self.event_prediction(min_locs[0], 
                                                          min_locs[1])) 
                         - 0.5)
            return func_plus
        def root_func2_mass(mass):
            func_plus = (self.nll(self.event_prediction(min_locs[0], 
                                                        mass)) 
                         - self.nll(self.event_prediction(min_locs[0], 
                                                          min_locs[1])) 
                         - 0.5)
            return func_plus
        def root_func3_theta(theta):
            func_plus = (self.nll(self.new_lambda(theta, 
                                                  min_locs[1],
                                                  min_locs[2])) 
                         - self.nll(self.new_lambda(min_locs[0], 
                                                    min_locs[1],
                                                    min_locs[2])) 
                         - 0.5)
            return func_plus
        def root_func3_mass(mass):
            func_plus = (self.nll(self.new_lambda(min_locs[0], 
                                                  mass,
                                                  min_locs[2])) 
                         - self.nll(self.new_lambda(min_locs[0], 
                                                    min_locs[1],
                                                    min_locs[2])) 
                         - 0.5)
            return func_plus
        def root_func3_alpha(alpha):
            func_plus = (self.nll(self.new_lambda(min_locs[0], 
                                                  min_locs[1],
                                                  alpha)) 
                         - self.nll(self.new_lambda(min_locs[0], 
                                                    min_locs[1],
                                                    min_locs[2])) 
                         - 0.5)
            return func_plus
        if len(min_locs) == 2:
            theta_err = secant_method(min_locs[0], root_func2_theta, stop, guess_shift, iters)
            mass_err = secant_method(min_locs[1], root_func2_mass, stop, guess_shift, iters)
            return [theta_err, mass_err]
        elif len(min_locs) == 3:
            theta_err = secant_method(min_locs[0], root_func3_theta, stop, guess_shift, iters)
            mass_err = secant_method(min_locs[1], root_func3_mass, stop, guess_shift, iters)
            alpha_err = secant_method(min_locs[2], root_func3_alpha, stop, guess_shift, iters)
            return [theta_err, mass_err, alpha_err]


''' Functions before addition of cross section term
    
    def dnll_dtheta_23(self, mixing_ang, mass_sqrd, alpha=None):
        mass_arg = 1.267 * mass_sqrd * self.__dist / self.__bin_centres
        osc_prob = self.prob_oscillation(self.__bin_centres, mixing_ang, mass_sqrd)
        deriv_lambda = (-2 * 1.267 * self.__flux_sim * self.__dist * (np.sin(2*mixing_ang))**2
                        * np.cos(mass_arg)*np.sin(mass_arg) 
                        / self.__bin_centres)
        
        deriv = deriv_lambda - (self.__events / osc_prob) * deriv_lambda
        return np.sum(deriv)
    
    def dnll_dm2_23(self, mixing_ang, mass_sqrd, alpha=None):
        mass_arg = 1.267 * mass_sqrd * self.__dist / self.__bin_centres
        osc_prob = self.prob_oscillation(self.__bin_centres, mixing_ang, mass_sqrd)
        deriv_lambda = (-4 * self.__flux_sim * (np.sin(mass_arg)**2) 
                        * np.sin(2*mixing_ang) * np.cos(2*mixing_ang))
        
        deriv = deriv_lambda - (self.__events / osc_prob) * deriv_lambda 
        return np.sum(deriv)
    
    def nll_grad(self, mixing_ang, mass_sqrd, alpha=None):
        #if alpha += None:
        grad = np.array([self.dnll_dtheta_23(mixing_ang, mass_sqrd),
                         self.dnll_dm2_23(mixing_ang, mass_sqrd)])
        #elif alpha != None:
        #    grad = np.array([self.dnll_dtheta_23(mixing_ang, mass_sqrd, alpha),
        #                    self.dnll_dm2_23(mixing_ang, mass_sqrd, alpha)])
        return grad
    
    def grad_search(self, initial_guess, learn_rate, stop, iters=1000):
        minimum = np.array(initial_guess)
        trial_NLL = []
        for i in range(iters):
            minimum = minimum - learn_rate * self.nll_grad(minimum[0], minimum[1])
            trial_NLL.append(self.nll(self.event_prediction(minimum[0], 
                                                            minimum[1])
                                      )
                             )
            if i > 0 and np.abs(trial_NLL[-2] - trial_NLL[-1]) < stop:
                print('Converged after ' + str(i+1) + ' iterations! :)')
                print('Min. location is theta = ' + str(minimum[0]) 
                      + ' and mass = ' + str(minimum[-1]))
                return minimum
        print('Has not converged after ' + str(iters) + 'iterations :(')
        print('Min. location is theta = ' + str(minimum[0]) 
              + ' and mass = ' + str(minimum[-1]))
        return minimum
    
    def quasi_newton(self, initial_guess, learn_rate, stop, iters=1000):
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
        update_mat = np.identity(2)
        mins = [minimum]
        grads = [self.nll_grad(minimum[0], minimum[1])]
        trial_NLL = []
        for i in range(iters):
            change = learn_rate * np.dot(update_mat, 
                                         self.nll_grad(minimum[0], minimum[1])
                                         )
            minimum = minimum - change
            mins.append(minimum)
            grads.append(self.nll_grad(minimum[0], minimum[1]))
            print(grads[-1])
            delt = mins[-1] - mins[-2]
            gamm = grads[-1] - grads[-2]
            update_mat = update_step(update_mat, delt, gamm)
            trial_NLL.append(self.nll(self.event_prediction(minimum[0], 
                                                            minimum[1])
                                      )
                             )
            if i > 0 and np.abs(trial_NLL[-2] - trial_NLL[-1]) < stop:
                #change = np.abs(mins[-2] - mins[-1])
                err = change
                print('Converged after ' + str(i+1) + ' iterations! :)')
                print('Min. location is theta = ' + str(minimum[0]) + ' +/- ' + str(err[0]) 
                      + '\nand mass = ' + str(minimum[-1]) + ' +/- ' + str(err[-1]))
                mins = np.array([mins]).reshape(len(mins),2)
                #print(mins.shape)
                #plt.plot(mins[:, 0], mins[:,1]*1e3, color='black')
                return minimum, err
        err = change
        print('Has not converged after ' + str(iters) + 'iterations :(')
        print('Min. location is theta = ' + str(minimum[0]) + ' +/- ' + str(err[0]) 
              + '\nand mass = ' + str(minimum[-1]) + ' +/- ' + str(err[-1]))
        return minimum
           
    def sec_deriv(min_val, last_min_val):
        step = np.abs(min_val - last_min_val)
        x_plus_h = min_val
        x_minus_step = last_min_val - step
        x_vals = np.array([x_plus_h, last_min_val, x_minus_step])
        y_vals = np.zeros(3)
        lams = self.event_prediction(x_vals, 
                                     mass_sqrd_guess)
        y_vals = self.nll(lams)
        second_deriv = (y_vals[0] - 2 * y_vals[1] + y_vals[2]) / step**2
        return np.abs(second_deriv)
    
    def univariate_mf(self, theta_intv, mass_intv, stop, iters=1000):
        # I choose the midpoint of the interval as the first guess for the minimum
        t_012 = np.linspace(theta_intv[0], theta_intv[1], 3)
        m_012 = np.linspace(mass_intv[0], mass_intv[1], 3)
        new_mins_t = [t_012[1]]
        new_mins_m = []
        y_012_t = np.zeros(3)
        y_012_m = np.zeros(3)
        #para_min_t = 0
        stop_m = False
        stop_t = False
        for a in range(iters):
            for i in range(3):
                if stop_t != True and a>0:
                    lam_t = self.event_prediction(t_012[i], 
                                                  new_mins_m[-1])
                    y_012_t[i] = self.nll(lam_t)
                if stop_m != True:
                    lam_m = self.event_prediction(new_mins_t[-1], 
                                                  m_012[i])
                    y_012_m[i] = self.nll(lam_m)
            if stop_t != True and a>0:
                para_min_t = tbx.new_min(t_012, y_012_t)
                new_mins_t.append(para_min_t)
                if np.abs(new_mins_t[-1] - new_mins_t[-2])/new_mins_t[-2] < stop:
                    stop_t = True
                    print('theta min found after ' + str(a+1) + ' iterations')
                t_012 = np.array([t_012[-2], t_012[-1], para_min_t])
            elif stop_m != True:
                para_min_m = tbx.new_min(m_012, y_012_m)
                new_mins_m.append(para_min_m)
                if a>0 and np.abs(new_mins_m[-1] - new_mins_m[-2])/new_mins_m[-2] < stop:
                    stop_m = True
                    print('mass min found after ' + str(a+1) + ' iterations')
                m_012 = np.array([m_012[-2], m_012[-1], para_min_m])
            elif stop_m and stop_t:
                print('minimum found after ' + str(a+1) + ' iterations')
                print('stopping condition met')
                print('values that minimise NLL are theta = ' + str(para_min_t) + ' and m = ' + str(para_min_m))
                return para_min_t, para_min_m
            else:
                print('Weird Error')
        print('minimum found after ' + str(iters) + ' iterations')
        print('yet to reach stopping condition')
        print('values that minimise NLL are theta = ' + str(para_min_t) + ' and m = ' + str(para_min_m))
        return para_min_t, para_min_m

    def std_newton(min_val, stop, guess_shift=0.01, iters=100000):
        old_zero_plus = min_val + guess_shift
        old_zero_minus = min_val - guess_shift
        guess = []
        its = []
        for i in range(iters):
            func_plus = (self.nll(self.event_prediction(old_zero_plus, 
                                                        mass_sqrd_guess)) 
                         - self.nll(self.event_prediction(min_val, 
                                                          mass_sqrd_guess)) 
                         - 0.5)
            new_zero_plus = (old_zero_plus 
                            - func_plus 
                            / self.dnll_dtheta_23(old_zero_plus, 
                                                  mass_sqrd_guess))
            func_minus = (self.nll(self.event_prediction(old_zero_minus, 
                                                         mass_sqrd_guess)) 
                         - self.nll(self.event_prediction(min_val, 
                                                          mass_sqrd_guess))
                         - 0.5) 
            new_zero_minus = (old_zero_minus 
                             - func_minus 
                             / self.dnll_dtheta_23(old_zero_minus, 
                                                   mass_sqrd_guess))
            change1 = np.abs(new_zero_plus - old_zero_plus)
            change2 = np.abs(new_zero_minus - old_zero_minus)
            guess.append(np.abs(new_zero_plus - new_zero_minus))
            its.append(i)
            if change1 < stop or change2 < stop:
            #if np.abs(change) < stop:
                print('Error iters - ' + str(i+1))
                #return np.abs(new_zero_plus - min_val)*2
                return np.abs(new_zero_plus - new_zero_minus)
            else:
                old_zero_plus = new_zero_plus
                old_zero_minus = new_zero_minus
        print('Error iters - ' + str(i+1))
        #return np.abs(new_zero_plus - min_val)*2
        #plt.figure()
        plt.plot(its, guess)
        return np.abs(new_zero_plus - new_zero_minus)
'''    
            
