#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 21:26:10 2022

@author: ruadhri
"""

This project involves developing a Negative Log Likelihood (NLL) fit to extract neutrino oscillation parameters from simulated T2K data. The tasks include:

Data Handling: Reading experimental and unoscillated event rate data from provided files.

Probability Calculation: Coding the oscillation probability using a given formula and applying it to predict event rates.

NLL Computation: Writing a function to compute the NLL and creating graphs to identify minima.

Minimisation: Implementing a parabolic minimiser to find the optimal oscillation parameters.

Accuracy Estimation: Estimating the error of the fit results using NLL shape analysis and curvature.

Two-dimensional Fitting: Extending the minimisation to handle two parameters simultaneously and validating results.

Cross-section Modelling: Incorporating the neutrino interaction cross-section as a free parameter in the NLL minimiser and estimating its effect on the results.

The goal is to accurately determine the oscillation parameters and assess the rate of increase in the neutrino interaction cross-section with energy.

The relevant files for this project are as follows,
    
1) minimisation.py
        
        This module contains all classes and functions used for minimisation.

        The main class reads in and stores the data upon initialisation, and
        allows for more convenient manipulation of data.
    
2) toolbox.py
        
        Contains certain mathematical functions that were used a lot in the
        project, but not directly related to minimisation, so were stored here
        to avoid clutter.
    
3) results.py
        
        This python script utilises the minimisation.py module to obtain all 
        plots contained in the report as specified in the project script.
        
        It can be ran from top to botton to return these, however I reccomend
        running the gradient search code blocks separately as they will take 
        quite a long time.
        
        Commented out code has been kept as evidence of continuous testing.
        
4) parabolic test.py
        
        Contains evidence of testing done for the parabolic interpolation
        method on trial functions.
        
5) simultaneous tests.py
        
        Contains evidence of testing done for the simultaneous minimisation
        methods on trial functions.
        
5) Data Folder
        
        Contains the data generated for my particular project as a text file,
        neutrino_data.txt