#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:23:51 2022

@author: ruadhri
"""

import minimisation as ms
import toolbox as tbx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import seaborn as sns
import scipy.stats as sci

sns.set()
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.5)
sns.set_palette('BuGn',1)

#%%

minim = ms.Minimiser()
#filepath = 'Data/neutrino_data.txt'
event_data = minim.events()
flux_data = minim.flux()
energies = minim.centres()
mixing_angle = np.pi/4.7
mass_squared = 2.8e-3
distance = 295


#%% 3.1 - The Data

minim.initial_hists()

#%% 3.2 - Fit Function (Oscillation Probability)

mix_ang = np.pi/4
mass_sq = 2.4e-3
energy_range = np.logspace(-2, 1, 1000)
survival_prob = minim.prob_oscillation(energy_range, mix_ang, mass_sq)
label=r'$\theta_{23} = \pi/4$\n$\Delta m_{23}^2 = 2.4\times10^{-3}$\n$L=295'
plt.figure()
plt.semilogx(energy_range, survival_prob)#, label=r'$\theta_{23} = \pi/4, \Delta m_{23}^2 = 2.4\times10^{-3}, L=295$')
plt.xlabel('Energy (GeV)')
plt.ylabel('Oscillation Probability')
plt.legend()
plt.tight_layout()

#%% 3.2 - Fit Function (Oscillated Event Rate Prediction)

#minim = ms.Minimiser_1D()
#event_data = minim.events()
#flux_data = minim.flux()
#log_fact = np.log(np.math.factorial(event_data))
#energies = minim.centres()

lambda_init = minim.event_prediction(mixing_angle, mass_squared)

plt.figure()
plt.plot(energies, lambda_init, label='Expected Events', linestyle='dashed', color='black')
plt.stairs(event_data, np.append(energies, 10.025)-0.025, fill=True, label='Measured Events')
#plt.plot(energies, event_data, label='Measured Events')
plt.xlabel('Energy (GeV)')
plt.ylabel('Events')
#plt.title(r'$\theta_{23} = \pi/4.7, \Delta m_{23}^2 = 2.8\times10^{-3}, L=295$')
plt.legend()
plt.tight_layout()

#%% 3.3 - Likelihood Function

#minim = ms.Minimiser_1D()
#event_data = minim.events()
#flux_data = minim.flux()
#energies = minim.centres()

mass_squared_guess = 2.8e-3
distance_guess = 295

#thetas = np.linspace(0.715, 0.741, 200)
thetas = np.linspace(0, np.pi/2, 200)

#loglikes = np.zeros((len(thetas)))
#for i in range(len(thetas)):
loglikes = minim.nll(minim.event_prediction(thetas, mass_squared_guess))

plt.figure()
plt.plot(thetas, loglikes)
#plt.axhline(y=minim.nll(minim.event_prediction(0.7276504758535954, mass_squared_guess)) + 0.5, color='black', linestyle='dashed')
plt.xlabel(r'Mixing Angle $\theta_{23}$ (rads)')
plt.ylabel('NLL')
plt.tight_layout()

thetas = np.linspace(0.715, 0.741, 200)
loglikes = minim.nll(minim.event_prediction(thetas, mass_squared_guess))

plt.figure()
plt.plot(thetas, loglikes)
plt.axhline(y=minim.nll(minim.event_prediction(0.7276504758535954, mass_squared_guess)) + 0.5, color='black', linestyle='dashed')
plt.xlabel(r'Mixing Angle $\theta_{23}$ (rads)')
plt.ylabel('NLL')
plt.tight_layout()

#print('Min. min location - ' + str(thetas[49]))
#print('Max. min location - ' + str(thetas[50]))

#%% 3.4 - Minimise

#inits = np.linspace(0, np.pi/2, 3)#.reshape(3,1)
#minim = ms.Minimiser_1D()
#event_data = minim.events()
#flux_data = minim.flux()
#energies = minim.centres()

#lambdas = minim.event_prediction(energies, flux_data, inits, mass_squared, distance)
#loglikes = minim.nll(event_data, lambdas, axis=1)

#minimum, err = minim.parabolic_min([thetas[49], thetas[50]], 1e-11, mass_squared, distance, curv=True)

#%% 3.5 - Finding the Accuracy of Fit Result (standard deviation)
print('Parabolic Method')
#inits = np.linspace(0, np.pi/2, 3)#.reshape(3,1)
#minim = ms.Minimiser_1D()
plt.figure()
minimum, err, x012, y012 = minim.parabolic_min([0, np.pi/2], 1e-11, mass_squared, curv=True)
minimum, err, x012, y012 = minim.parabolic_min([0, np.pi/2], 1e-11, mass_squared, std=True)
plt.xlabel('Iterations')
plt.ylabel('Standard Deviation')
plt.tight_layout()

thetas = np.linspace(0, np.pi/2, 200)
loglikes = minim.nll(minim.event_prediction(thetas, mass_squared))
p2s = tbx.p2(x012, y012, thetas)
plt.figure()
plt.plot(thetas , loglikes, label='Calculated NLL')
plt.plot(thetas, p2s, label='2nd Order Lagrange Poly. Fit', linestyle='dashed')
plt.xlabel(r'Mixing Angle $\theta_{23}$ (rads)')
plt.ylabel('NLL')
plt.tight_layout()
plt.legend()

#%% 4.1 - Univariate Method (First plotting NLL as a function of mass squared)

m23 = np.linspace(0, 0.005,300)
min_angle = np.pi/4 #0.8247994090528387#np.pi/4
#nll_m = np.zeros(m23.shape)
#for m in range(len(m23)):
nll_m = minim.nll(minim.event_prediction(min_angle, m23))

plt.figure()
plt.plot(m23, nll_m)
plt.xlabel(r'$\Delta m_{23}^2$ ($eV^2$)')
plt.ylabel('NLL')
plt.tight_layout()

#%% 4.1 - Univariate Method (Second plotting NLL as a function of both oscillation parameters)
print()
print('Minimum Location Estimates')

m23 = np.linspace(0, 0.005,300)
t23 = np.linspace(0, np.pi/2, 300)
theta_vals, m_vals = np.meshgrid(t23, m23)

nlls = np.zeros(m_vals.shape)
#for t in range(len(t23)):
#    for m in range(len(m23)):
nlls = minim.nll(minim.event_prediction(theta_vals, m_vals))

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta_vals, m_vals*1e3, nlls, cmap=cm.jet)
plt.xlabel(r'$\theta_{23}$ (rads)')
plt.ylabel(r'$\Delta m_{23}^2$ ($eV^2 \times10^{-3}$)')
ax.set_zlabel(r'NLL')

plt.figure()
plt.contourf(theta_vals, m_vals*1e3, nlls, cmap=cm.jet)
plt.xlabel(r'$\theta_{23}$ (rads)')
plt.ylabel(r'$\Delta m_{23}^2$ ($eV^2 \times10^{-3}$)')

# Manual check for quick estimate of minimum location
min_est_ind = np.where(nlls == np.amin(nlls))
min_1 = tuple(min_est_ind[0])
min_2 = tuple(min_est_ind[1])
minima1 = [theta_vals[min_2], m_vals[min_1]]
#minima2 = [theta_vals[min_2], m_vals[min_1]]
#min
print('Approx. min. NLL at angle = ' + str(minima1[0])
      + ' and mass = ' + str(minima1[1]))
#print('2st Min. NLL at angle = ' + str(minima2[0]) 
#      + ' and mass = ' + str(minima2[1]))

#%% 4.1 - Univariate Method (Testing the algorithm)
print()
print('Univariate Method')

theta_int = [0, np.pi/2]
mass_int = [0, 0.003]

min_coords, err, mass_atts, theta_atts = minim.univariate(theta_int, mass_int, 1e-4, stop_err=1e-7, iters=100000)
min_coords, err, mass_atts, theta_atts = minim.univariate(theta_int, mass_int, 5e-10, stop_err=1e-7, iters=100000)
#print(mass_atts)
new_masses = np.ones(len(theta_atts) - len(mass_atts)) * mass_atts[-1]
#print(new_masses)
mass_atts = np.append(np.array(mass_atts), new_masses)
#theta_min_mf, mass_min_mf = minim.univariate_mf(theta_int, mass_int, 1e-2)

# hypothesis: NLL as a function of theta is much smoother so better to do theta first
# conclusion: theta first converges closer to the expected value from prev. section
#             mass first meets stopping condition earlier
#             theta first is better

m23 = np.linspace(0.001, 0.0045,300)
t23 = np.linspace(0.6, 1.2, 300)
theta_vals, m_vals = np.meshgrid(t23, m23)
nlls = minim.nll(minim.event_prediction(theta_vals, m_vals))
plt.figure()
plt.contourf(theta_vals, m_vals*1e3, nlls, cmap=cm.jet)
plt.plot(theta_atts, mass_atts*1e3, linestyle='dashed', marker='o', color='black')
plt.colorbar()
plt.xlabel(r'$\theta_{23}$ (rads)')
plt.ylabel(r'$\Delta m_{23}^2$ ($eV^2 \times10^{-3}$)')
plt.tight_layout()

#%% 4.2 - Simultaneous Minimisation (gradient search)

print()
print('Gradient Search')
print('This code block is very slow so apologies')
initial_guess = [0.8, 0.003]
#its = 50000
its = 100000000
#rate = 2e-9
rate=1e-10

minimum, err, trials = minim.grad_search(initial_guess, learn_rate=rate, stop=1e-8, err_stop=1e-7, 
                            iters=its)

its = 4261707
plot_its = int(its/100)# - 100
plot_trials = np.zeros((2, plot_its))
for i in range(plot_its):
    #print(i)
    plot_trials[:,i] = trials[:, i*100]

m23 = np.linspace(0.0022, 0.0032,300)
t23 = np.linspace(0.79, 0.89, 300)
theta_vals, m_vals = np.meshgrid(t23, m23)
nlls = minim.nll(minim.event_prediction(theta_vals, m_vals))
plt.figure()
plt.plot(plot_trials[0], plot_trials[1]*1e3, linestyle='dashed', marker='o', color='white')
plt.contourf(theta_vals, m_vals*1e3, nlls, cmap=cm.jet)
#plt.plot(theta_atts, mass_atts*1e3, linestyle='dashed', marker='o', color='black')
plt.colorbar()
plt.xlabel(r'$\theta_{23}$ (rads)')
plt.ylabel(r'$\Delta m_{23}^2$ ($eV^2 \times10^{-3}$)')
plt.tight_layout()

plt.figure()
plt.plot(trials[0], trials[1]*1e3, linestyle='dashed', marker='o', color='black')

#%% 4.2 - Simultaneous Minimisation (Quasi-Newton)
# Doesn't converge unfortunately :(
print()
print('Quasi-Newton')
#m23 = np.linspace(0.0025, 0.00275,300)
#t23 = np.linspace(0.8, 0.9, 300)
#theta_vals, m_vals = np.meshgrid(t23, m23)
#nlls = minim.nll(minim.event_prediction(theta_vals, m_vals))
its = 100
rate = 1e-9

#initial_guess = [0.7, 2.4e-3]
initial_guess = [0.8, 0.003]
minimum, err, trials = minim.quasi_newton(initial_guess, learn_rate=rate, stop=1e-9, err_stop=1e-7, iters=its)

plot_its = int(its/100)# - 100
plot_trials = np.zeros((2, plot_its))
for i in range(plot_its):
    #print(i)
    plot_trials[:,i] = trials[:, i*100]

m23 = np.linspace(0.0022, 0.0032,300)
t23 = np.linspace(0.79, 0.89, 300)
theta_vals, m_vals = np.meshgrid(t23, m23)
nlls = minim.nll(minim.event_prediction(theta_vals, m_vals))
plt.figure()
plt.plot(plot_trials[0], plot_trials[1]*1e3, linestyle='dashed', marker='o', color='white')
plt.contourf(theta_vals, m_vals*1e3, nlls, cmap=cm.jet)
plt.xlabel(r'$\theta_{23}$ (rads)')
plt.ylabel(r'$\Delta m_{23}^2$ ($eV^2 \times10^{-3}$)')
plt.tight_layout()

#%% 5 - Neutrino Interaction Cross Section (plotting NLL as a function of alpha)

# First plotting NLL as a function of alpha at the determined minimum.
#print(minima1)
print()
print('Check alpha min location')
alphas = np.linspace(0.1, 10, 10000)
alphas = alphas.reshape(len(alphas), 1)
logs1 = minim.nll(minim.new_lambda(minima1[0], minima1[1], alphas))
#logs2 = minim.nll(minim.new_lambda(theta_vals[min_2], m_vals[min_2], alphas))

plt.figure()
plt.plot(alphas, logs1)
#plt.plot(alphas, logs2)
plt.xlabel(r'$\alpha$')
plt.ylabel('NLL')
plt.tight_layout()

min_alpha_ind1 = tuple(np.where(logs1 == np.amin(logs1))[0])
#min_alpha_ind2 = tuple(np.where(logs2 == np.amin(logs2))[0])
print('Alpha that minimises NLL at min(theta, mass) - ' + str(alphas[min_alpha_ind1][0]))
#print('Alpha that minimises NLL at 2st min(theta, mass) - ' + str(alphas[min_alpha_ind2][0]))

# Minimum NLL values for alpha are different at each minimum, so this should
# mean that there is only 1 globa minimum, not 2.

#%% 5 - Neutrino Interaction Cross Section (now try minimising)

print()
print('Neutrino Cross Section and Final Params')
init = [0.8, 0.003, 1.18]
rate = 1e-9
its = 100000000
# Quasi newton didn't work
#final_min, trials = minim.quasi_newton(initial_guess=init, learn_rate=0.8e-9, stop=5.8e-7, iters=100000)
#init = [0.85, 0.005, 1.2]
#final_min = minim.quasi_newton(initial_guess=init, learn_rate=1e-8, stop=1e-8, iters=100000)

#trying gradient descent instead
final_min, final_err, trials = minim.grad_search(init, learn_rate=rate, stop=1e-8, err_stop=1e-7, iters=its)


#%% Extension - Reproducing Event Data using new oscillation probability function

print()
print('Reproducing Event Data')
print('Final oscillation parameters are...')
print('theta_23 = ' + str(final_min[0]))
print('mass^2_23 = ' + str(final_min[1]))
print('alpha = ' + str(final_min[2]))
lambda_result = minim.new_lambda(mixing_ang_guess=final_min[0], 
                                 mass_sqrd_guess=final_min[1], 
                                 alpha = final_min[2])

plt.figure()
plt.plot(energies, lambda_result, label=r'Expected Events ($\lambda^{new}$)', linestyle='dashed', color='black')
#plt.scatter(energies, event_data, label='Measured Events', color='black', marker='x')
plt.stairs(event_data, np.append(energies, 10.025)-0.025, fill=True, label='Measured Events')
plt.xlabel('Energy (GeV)')
plt.ylabel('Events')
#plt.title(r'$\theta_{23} = \pi/4.7, \Delta m_{23}^2 = 2.8\times10^{-3}, L=295$')
plt.legend()
plt.tight_layout()

#%% Extension - Chi Squared Analysis

print()
print('Chi-Squared Analysis')
#print(np.sum(lambda_result))
#print(np.sum(event_data))
#N_dof = len(event_data) - 3
chi_squared_final = np.sum((event_data - lambda_result)**2 / (lambda_result))
#chi_squared_sci, p_value_sci = sci.chisquare(event_data, lambda_result, 197)
#print(chi_squared_sci)
N_dof = len(event_data) - 3
print('Degrees of freedom - ' + str(N_dof))
print('Chi Squared Value for the fit - ' + str(chi_squared_final))
print('Reduced Chi Squared Value for the fit - ' + str(chi_squared_final / N_dof))
#print('P Value - ' + str(p_value_sci))

chi_squared_init = np.sum((event_data - lambda_init)**2 / (lambda_init))
#chi_squared_sci, p_value_sci = sci.chisquare(event_data, lambda_result, 197)
#print(chi_squared_sci)
N_dof = len(event_data) - 3
print()
#print('Degrees of freedom - ' + str(N_dof))
print('Chi Squared Value for the inital guess fit - ' + str(chi_squared_init))
print('Reduced Chi Squared Value for the inital guess fit - ' + str(chi_squared_init / N_dof))

#reduced_chi = chi_squared / N_dof
#print('Reduced Chi Squared Value for the fit - ' + str(reduced_chi))

#%% Extension - Survival Probability

#energy_range = np.logspace(-2, 1, 10000)
#final_probs = minim.prob_oscillation(energy_range, final_min[0], final_min[1])
#plt.figure()
#plt.semilogx(energy_range, final_probs)
#plt.xlabel('Energy (GeV)')
#plt.ylabel('Oscillation Probability')
#plt.legend()
#plt.tight_layout()

#check = minim.new_lambda(minima1[0], minima1[1], 1.165)
#plt.figure()
#plt.plot(energies, check, label=r'Expected Events ($\lambda^{new}$)', linestyle='dashed')
#plt.scatter(energies, event_data, label='Measured Events')
#plt.xlabel('Energy (GeV)')
#plt.ylabel('Events')
#plt.title(r'$\theta_{23} = \pi/4.7, \Delta m_{23}^2 = 2.8\times10^{-3}, L=295$')
#plt.legend()
#plt.tight_layout()

#print(np.sum(check))
#print(np.sum(lambda_result))
#print(np.sum(event_data))
#chi_squared_sci, p_value_sci = sci.chisquare(event_data, lambda_result, ddof=N_dof)








