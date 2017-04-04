#!/usr/bin/env python

'''
Estimate performances of several CMB projects
wrt. their delensing power, their ability to separate components
as well as constrain cosmological parameters such as neutrino mass, w0/wa, r and nT
'''

import numpy as np
import pylab as pl
import healpy as hp
import forecasting_cosmo_self_consistent_forecast_smf as fc
import fnmatch
import operator
import residuals_computation_loc_calibration_errors as residuals_comp
from scipy import polyval, polyfit, optimize
import sys
import os
import python_camb_self_consistent
import argparse
import time
import matplotlib.cm as cm
from matplotlib import rc
import copy 
import glob
import pickle
from scipy.ndimage import gaussian_filter1d
import random
import string
import subprocess as sp
from CMB4cast_utilities import *
import CMB4cast_noise
import CMB4cast_compsep
import CMB4cast_delens
import CMB4cast_Fisher
from collections import OrderedDict


def pissoffpython():
    import ctypes as ct
    dl = np.ctypeslib.load_library('libdelens', '.')
    dl.delensing_performance.argtypes = [ct.c_int, ct.c_int, \
                                        ct.POINTER(ct.c_double), \
                                        ct.POINTER(ct.c_double), \
                                        ct.POINTER(ct.c_double), \
                                        ct.POINTER(ct.c_double), \
                                        ct.POINTER(ct.c_double), \
                                        ct.POINTER(ct.c_double), \
                                        ct.POINTER(ct.c_double), \
                                        ct.c_double, ct.c_bool, \
                                        ct.POINTER(ct.c_double), \
                                        ct.POINTER(ct.c_double)]
    def delens_est(l_min, l_max, c_l_ee_u, c_l_ee_l, c_l_bb_u, \
                c_l_bb_l, c_l_pp, f_l_cor, n_l_ee_bb, thresh, \
                no_iteration, n_l_pp, c_l_bb_res):
        return dl.delensing_performance(ct.c_int(l_min), ct.c_int(l_max), \
                c_l_ee_u.ctypes.data_as(ct.POINTER(ct.c_double)), \
                c_l_ee_l.ctypes.data_as(ct.POINTER(ct.c_double)), \
                c_l_bb_u.ctypes.data_as(ct.POINTER(ct.c_double)), \
                c_l_bb_l.ctypes.data_as(ct.POINTER(ct.c_double)), \
                c_l_pp.ctypes.data_as(ct.POINTER(ct.c_double)), \
                f_l_cor.ctypes.data_as(ct.POINTER(ct.c_double)),\
                n_l_ee_bb.ctypes.data_as(ct.POINTER(ct.c_double)), \
                ct.c_double(thresh), ct.c_bool(no_iteration), \
                n_l_pp.ctypes.data_as(ct.POINTER(ct.c_double)), \
                c_l_bb_res.ctypes.data_as(ct.POINTER(ct.c_double)))

    # common parameters
    l_min = 2
    l_max = 4000
    thresh = 0.01
    no_iteration = False
    f_cor = 0.0
    cib_delens = False
    if (cib_delens):
        f_l_cor_raw = np.genfromtxt('f_l_cor_planck_545.dat')
        f_l_cor = f_l_cor_raw[0: l_max - l_min + 1, 1].flatten()
    else:
        f_l_cor = np.ones(l_max - l_min + 1) * f_cor
    c_l_u=np.genfromtxt('fiducial_lenspotentialCls.dat')
    c_l_l=np.genfromtxt('fiducial_lensedtotCls.dat') 
    ell = c_l_u[0: l_max - l_min + 1, 0].flatten()
    d_l_conv = 2.0 * np.pi / ell / (ell + 1.0)
    c_l_ee_u = c_l_u[0: l_max - l_min + 1, 2].flatten() * d_l_conv
    c_l_ee_l = c_l_l[0: l_max - l_min + 1, 2].flatten() * d_l_conv
    c_l_bb_u = c_l_u[0: l_max - l_min + 1, 3].flatten() * d_l_conv
    c_l_bb_l = c_l_l[0: l_max - l_min + 1, 3].flatten() * d_l_conv
    c_l_pp = c_l_u[0: l_max - l_min + 1, 5].flatten() / \
		 (ell * (ell + 1.0)) ** 2 * 2.0 * np.pi
    f_sky_new = 0.75
    l_min_exp_new = int(np.ceil(2.0 * np.sqrt(np.pi / f_sky_new)))
    spp = np.sqrt(2.0) * 0.58
    beam = 1.0
    beam =  beam / 60.0 / 180.0 * np.pi
    beam_area = beam * beam
    beam_theta = beam / np.sqrt(8.0 * np.log(2.0))
    n_l_ee_bb = np.zeros(l_max - l_min + 1)
    for i in range(0, l_max - l_min + 1):
        bl=np.exp(beam_theta * beam_theta * (l_min + i) * (l_min + i + 1))
        n_l_ee_bb[i] = (beam_area * spp * spp * bl)
    n_l_pp = np.zeros(l_max - l_min + 1)
    c_l_bb_res = np.zeros(l_max - l_min + 1)
    delens_est(l_min_exp_new, l_max, c_l_ee_u[l_min_exp_new - l_min:], \
		   c_l_ee_l[l_min_exp_new - l_min:], \
		   c_l_bb_u[l_min_exp_new - l_min:], \
		   c_l_bb_l[l_min_exp_new - l_min:], \
		   c_l_pp[l_min_exp_new - l_min:], \
		   f_l_cor[l_min_exp_new - l_min:], \
		   n_l_ee_bb[l_min_exp_new - l_min:], \
		   thresh, no_iteration, n_l_pp[l_min_exp_new - l_min:], \
		   c_l_bb_res[l_min_exp_new - l_min:])


############################################################################################################################################################################################################
############################################################################################################################################################################################################
# plotting stuff
pl.rc('font', family = 'serif')
pl.rcParams['text.latex.preamble'] = [r'\boldmath']
pl.rcParams['axes.linewidth'] = 1.5
pl.rcParams['lines.linewidth'] = 1.5
#t20cb = [(0, 107, 164), (255, 128, 14), (171, 171, 171), (89, 89, 89), \
t20cb = [(0, 107, 164), (255, 128, 14), (0, 0, 0), (89, 89, 89), \
		 (95, 158, 209), (200, 82, 0), (137, 137, 137), (163, 200, 236), \
		 (255, 188, 121), (207, 207, 207)]
cb_colors = [ ]
for col in range(len(t20cb)):
	r, g, b = t20cb[col]
	cb_colors.append((r / 255., g / 255., b / 255.))

# @TODO - could tidy up these so that the same keys are used, but whatever...
locs = {}
locs['ground']  = 1
locs['balloon'] = 2
locs['space']   = 3
locs['cross']   = 4
loc_markers = {}
loc_markers[locs['ground']]  = 'v'
loc_markers[locs['balloon']] = 'o'
loc_markers[locs['space']]   = '^'
loc_markers[locs['cross']]   = 'x'
############################################################################################################################################################################################################
############################################################################################################################################################################################################

##############
## SETUP COSMOLOGIES. fiducial cosmology matches Planck 2015 TT, TE, EE + lowP +
## lensing + BAO + JLA + H0 constraints, c.f. arxiv:1502.01589v2.
## note that regarding neutrinos Planck gives "constraints assuming three
## species of degenerate massive neutrinos, neglecting the small differences in
## mass expected from the observed mass splittings. At the level of sensitivity
## of Planck this is an accurate approximation, but note that it does not quite
## match continuously on to the base \LambdaCDM model (which assumes two
## massless and one massive neutrino with \Sigma m_\nu = 0.06 eV)"
params_fid = {}
params_fid['h'] = 67.74
params_fid['ombh2'] = 0.02230
params_fid['omch2'] = 0.1188
params_fid['omnuh2'] = 0.0006451439 # 1 massive neutrino, \Sigma M_\nu = 0.06 eV #100.0/93000 #85.0/93000 #
params_fid['omk'] = 0.0
params_fid['YHe'] = 0.2453 # "in all cases Y_p is predicted by BBN, with posterior mean 0.2453"
params_fid['Neff'] = 3.046
params_fid['w'] = -1.0
params_fid['wa'] = 0.0
params_fid['tau'] = 0.066
params_fid['As'] = 2.142e-9
params_fid['ns'] = 0.9667
params_fid['alphas'] = 0.0
params_fid['r'] = 0.001
params_fid['nT'] = -params_fid['r']/8.0
params_fid['A'] = 0.1 ## residuals amplitude @ ell= 1
params_fid['b'] = -0.8 ## ell dependence of the residuals
params_fid['k_scalar'] = 0.05#05
params_fid['k_tensor'] = 0.002#05
params_fid['A_fgs_res'] = 1.0 ## residuals global amplitude
params_fid['b_fgs_res'] = -2.0 ## residuals global ell dependence
						
						
# cosmology with massless neutrinos
params_fid_mnu_0 = copy.copy(params_fid)
params_fid_mnu_0['omnuh2'] = 0.0

# cosmologies with varying r. assume consistency relation throughout
params_fid_r_0p1 = copy.copy(params_fid)
params_fid_r_0p03 = copy.copy(params_fid)
params_fid_r_0p01 = copy.copy(params_fid)
params_fid_r_0 = copy.copy(params_fid)
params_fid_r_0p1['r']    = 0.100
params_fid_r_0p03['r']   = 0.030
params_fid_r_0p01['r']   = 0.010
params_fid_r_0['r']      = 0.000
for params_fid_loc in [params_fid_r_0p1, params_fid_r_0p03, params_fid_r_0p01, params_fid_r_0]:
	params_fid_loc['nT']  = -params_fid_loc['r']/8.0

################
## define priors, from Table IV and V of 1502.01589v2 Planck 2015 XIII: Cosmological parameters
params_fid_prior = {}
params_fid_prior['h'] = 0.46
params_fid_prior['ombh2'] = 0.00014
params_fid_prior['omch2'] = 0.0010
params_fid_prior['omnuh2'] = 0.194/93000  
params_fid_prior['omk'] = 0.0040
params_fid_prior['YHe'] = 0.026 
params_fid_prior['Neff'] = 0.33
params_fid_prior['w'] = 0.080
params_fid_prior['wa'] = 1.0
params_fid_prior['tau'] = 0.012
params_fid_prior['As'] = 4.9e-11
params_fid_prior['ns'] = 0.0040
params_fid_prior['alphas'] = 0.013
params_fid_prior['r'] = 0.113
params_fid_prior['nT'] = 1e4 ## ? not published
params_fid_prior['A_fgs_res'] = 1e8 ## residuals global amplitude
params_fid_prior['b_fgs_res'] = 1e8 ## residuals global ell dependence

# priors from external data sets 
params_fid_prior_ext = {}
params_fid_prior_ext['h'] = 2.5e-2
params_fid_prior_ext['ombh2'] = 1e8
params_fid_prior_ext['omch2'] = 1e8
params_fid_prior_ext['omnuh2'] = 1e8
params_fid_prior_ext['omk'] = 1e8
params_fid_prior_ext['YHe'] = 1e8
params_fid_prior_ext['Neff'] = 1e8
params_fid_prior_ext['w'] = 1e8
params_fid_prior_ext['wa'] = 1e8
params_fid_prior_ext['tau'] = 1e8
params_fid_prior_ext['As'] = 1e8
params_fid_prior_ext['ns'] = 1e8
params_fid_prior_ext['alphas'] =1e8
params_fid_prior_ext['r'] = 1e8
params_fid_prior_ext['nT'] = 1e8 ## ? not published
params_fid_prior_ext['A_fgs_res'] = 1e8 ## residuals global amplitude
params_fid_prior_ext['b_fgs_res'] = 1e8 ## residuals global ell dependence

############################################################################################################################################################################################################
############################################################################################################################################################################################################

##############
## SETUP MODELS
## i.e., which parameters we vary together
## may want to vary Y_He along with N_eff as they're degenerate
## may also want to consider adding in a prior on H_0 on occasion?
## params_dev = ['alphas', 'r', 'nT', 'ns', 'As', 'tau', 'h', 'ombh2', \
##               'omch2', 'omnuh2', 'omk', 'YHe', 'Neff', 'w', 'wa']
params_dev_full          = ['ns', 'As', 'tau', 'h', 'ombh2', 'omch2', \
							'alphas', 'r', 'nT', 'omk', 'omnuh2', 'Neff', \
							'YHe', 'w', 'wa' ]
params_dev_LCDM_r        = ['ns', 'As', 'tau', 'h', 'ombh2', 'omch2', 'r']
params_dev_LCDM_r_nt     = ['ns', 'As', 'tau', 'h', 'ombh2', 'omch2', 'r', 'nT']
params_dev_LCDM_full_inf = ['ns', 'As', 'tau', 'h', 'ombh2', 'omch2', \
							'alphas', 'r', 'nT', 'omk']
params_dev_LCDM_Neff     = ['ns', 'As', 'tau', 'h', 'ombh2', 'omch2', 'Neff']
params_dev_LCDM_mnu      = ['ns', 'As', 'tau', 'h', 'ombh2', 'omch2', 'omnuh2']
params_dev_wwaCDM        = ['ns', 'As', 'tau', 'h', 'ombh2', 'omch2', 'w', 'wa']
params_dev_wCDM          = ['ns', 'As', 'tau', 'h', 'ombh2', 'omch2', 'w']
params_dev_LCDM_snu      = ['ns', 'As', 'tau', 'h', 'ombh2', 'omch2', \
							'omnuh2', 'Neff']
params_dev_LCDM_k        = ['ns', 'As', 'tau', 'h', 'ombh2', 'omch2', 'omk']
params_dev_test_r_only   = ['ns', 'As', 'tau', 'h', 'ombh2', 'omch2', 'r']
#params_dev_LCDM_k_alone = ['omk']

# priors should be specified in same order as model parameters
params_priors_full          = [params_fid_prior[p] for p in params_dev_full]
params_priors_LCDM_r        = [params_fid_prior[p] for p in params_dev_LCDM_r]
params_priors_LCDM_r_nt     = [params_fid_prior[p] for p in params_dev_LCDM_r_nt]
params_priors_LCDM_full_inf = [params_fid_prior[p] for p in params_dev_LCDM_full_inf]
params_priors_LCDM_Neff     = [params_fid_prior[p] for p in params_dev_LCDM_Neff]
params_priors_LCDM_mnu      = [params_fid_prior[p] for p in params_dev_LCDM_mnu]
params_priors_wwaCDM        = [params_fid_prior[p] for p in params_dev_wwaCDM]
params_priors_LCDM_snu      = [params_fid_prior[p] for p in params_dev_LCDM_snu]
params_priors_LCDM_k        = [params_fid_prior[p] for p in params_dev_LCDM_k]

# priors from external data sets 
params_priors_full_ext 			= [params_fid_prior_ext[p] for p in params_dev_full]
params_priors_LCDM_r_ext     	= [params_fid_prior_ext[p] for p in params_dev_LCDM_r]
params_priors_LCDM_r_nt_ext		= [params_fid_prior_ext[p] for p in params_dev_LCDM_r_nt]
params_priors_LCDM_full_inf_ext	= [params_fid_prior_ext[p] for p in params_dev_LCDM_full_inf]
params_priors_LCDM_Neff_ext		= [params_fid_prior_ext[p] for p in params_dev_LCDM_Neff]
params_priors_LCDM_mnu_ext		= [params_fid_prior_ext[p] for p in params_dev_LCDM_mnu]
params_priors_wwaCDM_ext		= [params_fid_prior_ext[p] for p in params_dev_wwaCDM]
params_priors_LCDM_snu_ext		= [params_fid_prior_ext[p] for p in params_dev_LCDM_snu]
params_priors_LCDM_snu_k		= [params_fid_prior_ext[p] for p in params_dev_LCDM_k]


############################################################################################################################################################################################################
############################################################################################################################################################################################################

##############
## SETUP EXPERIMENTAL CONFIGURATIONS
## e.g.

expts = {}

expts['LiteBIRD_baseline'] = {}
expts['LiteBIRD_baseline']['freqs'] = [60.0, 78.0, 100.0, 140.0, 195.0, 280.0]
expts['LiteBIRD_baseline']['uKCMBarcmin'] = [15.7, 9.9, 7.1, 5.6, 4.7, 5.7] #[ 10.3, 6.5, 4.7, 3.7, 3.1, 3.8 ]
expts['LiteBIRD_baseline']['FWHM'] = [54.1, 55.5, 56.8, 40.5, 38.4, 37.7] #[75.0, 58.0, 45.0, 32.0, 24.0, 16.0]
expts['LiteBIRD_baseline']['fsky'] = 0.7
expts['LiteBIRD_baseline']['bandpass'] = 0.3*np.ones(len(expts['LiteBIRD_baseline']['freqs']))
expts['LiteBIRD_baseline']['ell_min'] = lmin_computation(expts['LiteBIRD_baseline']['fsky'], 'space')
expts['LiteBIRD_baseline']['ell_max'] = 1500
expts['LiteBIRD_baseline']['prior_dust'] = 2*0.02
expts['LiteBIRD_baseline']['prior_sync'] = 2*0.2
expts['LiteBIRD_baseline']['loc'] = locs['space']
expts['LiteBIRD_baseline']['alpha_knee'] = 0.0*np.ones(len(expts['LiteBIRD_baseline']['freqs']))
expts['LiteBIRD_baseline']['ell_knee'] = 0.0*np.ones(len(expts['LiteBIRD_baseline']['freqs']))

expts['LiteBIRD_extended'] = {}
expts['LiteBIRD_extended']['freqs'] = [40.0, 50.0, 60.0, 68.4, 78.0, 88.5, 100.0, 118.9, 140.0, 166.0, 195.0, 234.9, 280.0, 337.4, 402.1] #[60.0, 78.0, 100.0, 140.0, 195.0, 280.0]
expts['LiteBIRD_extended']['uKCMBarcmin'] = [42.4586227 ,  25.80081497,  20.16471461,  15.65583433, 12.52466746,  10.14498064,  11.89843409,   9.51874727, 7.64004715,   6.76332043,   5.13511366,   6.38758041, 10.14498064,  10.14498064,  19.16274122] #[ 10.3, 6.5, 4.7, 3.7, 3.1, 3.8 ]
expts['LiteBIRD_extended']['FWHM'] = [108, 86, 72, 63, 55, 49, 43, 36, 31, 26, 22, 18, 37, 31, 26] #[75.0, 58.0, 45.0, 32.0, 24.0, 16.0]
expts['LiteBIRD_extended']['fsky'] = 0.7
expts['LiteBIRD_extended']['bandpass'] = 0.3*np.ones(len(expts['LiteBIRD_extended']['freqs']))
expts['LiteBIRD_extended']['ell_min'] = lmin_computation(expts['LiteBIRD_extended']['fsky'], 'space')
expts['LiteBIRD_extended']['ell_max'] = 3000
expts['LiteBIRD_extended']['prior_dust'] = 2*0.02
expts['LiteBIRD_extended']['prior_sync'] = 2*0.2
expts['LiteBIRD_extended']['loc'] = locs['space']
expts['LiteBIRD_extended']['alpha_knee'] = 0.0*np.ones(len(expts['LiteBIRD_extended']['freqs']))
expts['LiteBIRD_extended']['ell_knee'] = 0.0*np.ones(len(expts['LiteBIRD_extended']['freqs']))

expts['LiteBIRD_update'] = {}
expts['LiteBIRD_update']['freqs'] = [40.0, 50.0, 60.0, 68.4, 78.0, 88.5, 100.0, 118.9, 140.0, 166.0, 195.0, 234.9, 280.0, 337.4, 402.1] #[60.0, 78.0, 100.0, 140.0, 195.0, 280.0]
expts['LiteBIRD_update']['uKCMBarcmin'] = [56.715, 34.019, 26.148, 20.263, 15.658, 12.569, 15.546, 12.389, 8.061, 8.353, 6.344, 8.054, 21.352, 25.462, 61.172]
expts['LiteBIRD_update']['FWHM'] = [108, 86, 72, 63, 55, 49, 43, 36, 31, 26, 22, 18, 37, 31, 26] #[75.0, 58.0, 45.0, 32.0, 24.0, 16.0]
expts['LiteBIRD_update']['fsky'] = 0.7
expts['LiteBIRD_update']['bandpass'] = 0.3*np.ones(len(expts['LiteBIRD_update']['freqs']))
expts['LiteBIRD_update']['ell_min'] = lmin_computation(expts['LiteBIRD_update']['fsky'], 'space')
expts['LiteBIRD_update']['ell_max'] = 3000
expts['LiteBIRD_update']['prior_dust'] = 2*0.02
expts['LiteBIRD_update']['prior_sync'] = 2*0.2
expts['LiteBIRD_update']['loc'] = locs['space']
expts['LiteBIRD_update']['alpha_knee'] = 0.0*np.ones(len(expts['LiteBIRD_update']['freqs']))
expts['LiteBIRD_update']['ell_knee'] = 0.0*np.ones(len(expts['LiteBIRD_update']['freqs']))

expts['Planck'] = {}
expts['Planck']['freqs'] = [ 30.0, 44.0, 70.0, 100.0, 143.0, 217.0, 353.0 ]
expts['Planck']['uKCMBarcmin'] = np.array([7.5, 7.5, 4.8, 1.3, 1.1, 1.6, 6.9])*40.0
expts['Planck']['FWHM'] = [  33.16,  28.09,  13.08, 9.66,  7.27,  5.01,  4.86 ] 
expts['Planck']['fsky'] = 0.5
expts['Planck']['bandpass'] = 0.3*np.ones(len(expts['Planck']['freqs']))
expts['Planck']['ell_min'] = lmin_computation(expts['Planck']['fsky'], 'space')
expts['Planck']['ell_max'] = 3999
expts['Planck']['information_channels'] = ['Tu', 'Eu', 'd']
expts['Planck']['prior_dust'] = 0.0
expts['Planck']['prior_sync'] = 0.0
expts['Planck']['loc'] = locs['space']
expts['Planck']['alpha_knee'] = 0.0*np.ones(len(expts['Planck']['freqs']))
expts['Planck']['ell_knee'] = 0.0*np.ones(len(expts['Planck']['freqs']))

# derived from SA NETs so that the noise after component separation was about 1uK.arcmin in polarization
expts['Stage-IV'] = {}
expts['Stage-IV']['freqs'] = [40.0, 90.0, 150.0, 220.0, 280.0]
expts['Stage-IV']['uKCMBarcmin'] = [ 3.0, 1.5, 1.5, 5.0, 9.0 ]
expts['Stage-IV']['FWHM'] = [ 11.0, 5.0, 3.0, 2.0, 1.5 ]
expts['Stage-IV']['fsky'] = 0.5
# expts['Stage-IV']['bandpass'] = 0.3*np.ones(len(expts['Stage-IV']['freqs']))
expts['Stage-IV']['bandpass'] = 0.0*np.ones(len(expts['Stage-IV']['freqs']))
expts['Stage-IV']['ell_min'] = lmin_computation(expts['Stage-IV']['fsky'], 'ground')
expts['Stage-IV']['ell_max'] = 3999
expts['Stage-IV']['prior_dust'] = 2*0.02
expts['Stage-IV']['prior_sync'] = 2*0.2
expts['Stage-IV']['loc'] = locs['ground']
expts['Stage-IV']['alpha_knee'] = 0.0*np.ones(len(expts['Stage-IV']['freqs']))
expts['Stage-IV']['ell_knee'] = 0.0*np.ones(len(expts['Stage-IV']['freqs']))

# C-BASS: informations about C-BASS
# http://www.astro.caltech.edu/cbass/posters/Dickinson_CBASS_Okinawa_June2013.pdf
expts['C-BASS'] = {}
expts['C-BASS']['freqs'] = np.array([ 5.0 ] )
expts['C-BASS']['uKCMBarcmin'] = np.array([ 100*45.0 ]) #[ 100*(45.0/pix_size_1024)*residuals_computation.BB_factor_computation( 5.0 ) ]
expts['C-BASS']['FWHM'] = [45.0]
expts['C-BASS']['fsky'] = 0.8
expts['C-BASS']['bandpass'] = 0.2*np.ones(len(expts['C-BASS']['freqs']))
expts['C-BASS']['ell_min'] = lmin_computation(expts['C-BASS']['fsky'], 'ground')
expts['C-BASS']['ell_max'] = 3999
expts['C-BASS']['prior_dust'] = 2*0.02
expts['C-BASS']['prior_sync'] = 2*0.2
expts['C-BASS']['loc'] = locs['ground']
expts['C-BASS']['uKRJ/pix'] = expts['C-BASS']['uKCMBarcmin'] * residuals_comp.BB_factor_computation(  expts['C-BASS']['freqs'] ) / pix_size_map_arcmin
expts['C-BASS']['alpha_knee'] = 0.0*np.ones(len(expts['C-BASS']['freqs']))
expts['C-BASS']['ell_knee'] = 0.0*np.ones(len(expts['C-BASS']['freqs']))

# Quijote from http://max.ifca.unican.es/EWASS2015/TalksOnTheWeb/Number04_GenovaSantos.pdf
expts['Quijote'] = {}
expts['Quijote']['freqs'] = np.array([ 11.0, 13.0, 17.0, 19.0, 30.0, 42.0 ] )
#expts['Quijote']['uKCMBarcmin'] = np.array([ 4.7, 4.7, 4.7, 4.7, 0.84, 0.84 ])*60.0
# from http://arxiv.org/pdf/1401.4690.pdf, section 4.1 -- wide survey = 18,000deg2 ~ 44%
# 14 uK per 1 sq deg beam for 11-19 GHz, and ~3uK per beam for 30-40 GHz.
expts['Quijote']['uKCMBarcmin'] = np.array([ 14.0, 14.0, 14.0, 14.0, 3*0.37, 3*0.28 ])*60.0
expts['Quijote']['FWHM'] = np.array([ 0.92, 0.92, 0.6, 0.6, 0.37, 0.28 ])*60.0
expts['Quijote']['fsky'] = 18000.0 * (np.pi/180.0)**2 / (4.0*np.pi)
expts['Quijote']['bandpass'] =  np.array([2,2,2,2,8,10]) / expts['Quijote']['freqs']
expts['Quijote']['ell_min'] = lmin_computation(expts['Quijote']['fsky'], 'ground')
expts['Quijote']['ell_max'] = 3999
expts['Quijote']['prior_dust'] = 2*0.02
expts['Quijote']['prior_sync'] = 2*0.2
expts['Quijote']['loc'] = locs['ground']
expts['Quijote']['uKRJ/pix'] = expts['Quijote']['uKCMBarcmin'] * residuals_comp.BB_factor_computation(  expts['Quijote']['freqs'] ) / pix_size_map_arcmin
expts['Quijote']['alpha_knee'] = 0.0*np.ones(len(expts['Quijote']['freqs']))
expts['Quijote']['ell_knee'] = 0.0*np.ones(len(expts['Quijote']['freqs']))


# selection to consider

# configurations = dict((expt, expts[expt]) for expt in ('LiteBIRD_extended', \
													   # 'LiteBIRD_update'))
configurations = {}
configurations['Stage-IV'] = copy.deepcopy(expts['Stage-IV'])

############################################################################################################################################################################################################
#########################################################################
#
#  COMPONENT SEPARATION 
#  
# 
# you choose sky components and their ANALYTIC scaling law below
analytic_expr_per_template = OrderedDict([ \
         ('Qcmb','(nu / cst) ** 2 * ( exp ( nu / cst ) ) / ( ( exp ( nu / cst ) - 1 ) ** 2 )'),\
         ('Ucmb','(nu / cst) ** 2 * ( exp ( nu / cst ) ) / ( ( exp ( nu / cst ) - 1 ) ** 2 )'),\
         ('Qdust','( exp( nu_ref / (Td / h_over_k ) ) - 1 ) / ( exp( nu / ( Td / h_over_k ) ) - 1 ) * ( nu / nu_ref ) ** ( 1 + Bd + drun * log( nu / nu_ref ) )'),\
         ('Udust','( exp( nu_ref / (Td / h_over_k ) ) - 1 ) / ( exp( nu / ( Td / h_over_k ) ) - 1 ) * ( nu / nu_ref ) ** ( 1 + Bd + drun * log( nu / nu_ref ) )'),\
         # ('dBQdust','( exp( nu_ref / (Td / h_over_k ) ) - 1 ) / ( exp( nu / ( Td / h_over_k ) ) - 1 ) * log( nu / nu_ref ) * ( nu / nu_ref ) ** ( 1 + Bd + drun * log( nu / nu_ref ) )'),\
         # ('dBUdust','( exp( nu_ref / (Td / h_over_k ) ) - 1 ) / ( exp( nu / ( Td / h_over_k ) ) - 1 ) * log( nu / nu_ref ) * ( nu / nu_ref ) ** ( 1 + Bd + drun * log( nu / nu_ref ) )'),\
         ('Qsync','( nu / nu_ref ) ** (Bs + srun * log( nu / nu_ref ) )'),\
         ('Usync','( nu / nu_ref ) ** (Bs + srun * log( nu / nu_ref ) )') ])#,\
         # ('dBQsync','log( nu / nu_ref ) * ( nu / nu_ref ) ** (Bs + srun * log( nu / nu_ref ) )'),\
         # ('dBUsync','log( nu / nu_ref ) * ( nu / nu_ref ) ** (Bs + srun * log( nu / nu_ref ) )')\
          # ] )

# you pick here the numerical values for the spectral parameters
spectral_parameters = { 'nu_ref':150.0, 'Bd':1.59, 'Td':19.6, 'h_over_k':h_over_k, 'drun':0.0, 'Bs':-3.1, 'srun':0.0, 'cst':cst }

# by-default priors -> these are set to zero
prior_spectral_parameters = { 'Bd':0.0, 'Td':0.0, 'drun':0.0, 'Bs':0.0, 'srun':0.0 }

# spectral parameters to constraint with data sets 
drv = ['Bd', 'Bs']

############################################################################################################################################################################################################
########################################################################
# compute noise, delta_beta and Clres from noise per channel in RJ
#components_v = ['cmb-only', 'dust', 'sync', 'sync+dust' ]
components_v = ['cmb-only', 'sync+dust']#, 'sync+dust+dust']
# components_v = ['cmb-only']

params_fid_v = [params_fid, params_fid_r_0p1, params_fid_r_0p03, params_fid_r_0p01, params_fid_r_0]

params_fid_names_v = ['fid', 'fid_r_0p1', 'fid_r_0p03', \
					  'fid_r_0p01', 'fid_r_0']

params_dev_v =  [params_dev_LCDM_r, params_dev_LCDM_r_nt, \
				params_dev_LCDM_full_inf, params_dev_LCDM_mnu, \
				params_dev_LCDM_Neff, params_dev_full, \
				params_dev_wCDM, params_dev_LCDM_k]

params_dev_names_v = ['L_r', 'L_r_nt', \
					  'L_inf', 'L_mnu', \
					  'L_Neff', 'full', \
					  'L_w', 'L_k']

information_channels= ['Tu', 'Eu', 'Bu', 'd']
delensing_option_v = ['','CMBxCMB','CMBxCIB', 'CMBxLSS']

delensing_z_max = 3.5


############################################################################################################################################################################################################


#####################################################
## wrapper to call core_function from command line ##
#####################################################
def initialize():
	
	#######################################################################
	# parse the command-line options and set up others
	args = grabargs()
	params_fid_sel = [params_fid, params_fid_r_0p1]
	params_dev_sel = [params_dev_LCDM_r, params_dev_LCDM_r_nt, \
					  params_dev_LCDM_full_inf, params_dev_LCDM_mnu, \
					  params_dev_LCDM_Neff, params_dev_wCDM, params_dev_LCDM_k]
	params_prior_sel = [ params_priors_full_ext ]
	
	#######################################################################
	# build up the filename for results: just one file per run, for now
	res_file = '_Ab_'.join(configurations.keys())
	if args.cbass: res_file += '_xCBASS'
	if args.quijote: res_file += '_xQUIJOTE'
	res_file += '_' + '_'.join(information_channels)
	res_file += '_' + '_'.join(components_v)
	res_file += '_Bd_1.59_Td_19.6_Bs_-3.1'
	if args.stolyarov:
		res_file += '_stol_d'
	elif args.stolyarov_sync:
		res_file += '_stol_ds'
	else:
		res_file += '_np'
	if args.calibration_error!=0.0:
		res_file += '_calib_error_'+str(args.calibration_error)
	res_file += '_' + '_'.join(['none' if x=='' else x for x in delensing_option_v])
	if 'CMBxLSS' in delensing_option_v:
		res_file += '_z_max_' + str(delensing_z_max)
	for i in range(0, len(params_fid_sel)):
		for j in range(0, len(params_fid_v)):
			if params_fid_sel[i] == params_fid_v[j]:
				res_file += '_' + params_fid_names_v[j]
	for i in range(0, len(params_dev_sel)):
		for j in range(0, len(params_dev_v)):
			if params_dev_sel[i] == params_dev_v[j]:
				res_file += '_' + params_dev_names_v[j]
	res_file = 'fc_' + res_file

	#######################################################################
	# perform forecast and save results, or load pre-calculated results
	print '#################### forecast ####################'
	print 'looking for', res_file + '.pkl'
	res_path = glob.glob(res_file + '.pkl')
	if 1 or (not res_path) or args.fgs_vs_freq_vs_ell or args.power_spectrum_figure or args.fgs_power_spectrum or  args.delens_power_spectrum or args.combo_power_spectrum:
		print '#################### no existing results: forecasting! ####################'
		foregrounds, sigmas, Nl, Cls_fid = core_function( configurations=configurations, \
			components_v=components_v, camb=args.camb, params_fid_v=params_fid_sel, \
			params_dev_v=params_dev_sel, information_channels=information_channels, \
			delensing_option_v=delensing_option_v, delensing_z_max=delensing_z_max, \
			param_priors_v=[], cross_only=args.cross_only, Bd=1.59, Td=19.6, Bs=-3.1, \
			stolyarov=args.stolyarov, stolyarov_sync=args.stolyarov_sync,\
			cbass=args.cbass, quijote=args.quijote, \
			delens_command_line=args.delens_command_line, calibration_error=args.calibration_error,\
			path2maps=args.path2maps, path2Cls=args.path2Cls )

		save_obj('./', res_file, (foregrounds, sigmas, Nl))
	else:
		print '#################### loading existing results! ####################'
		foregrounds, sigmas, Nl = load_obj('./', res_path[0])
    
	#######################################################################
	## output LaTeX table for each instrument
	# which fiducial cosmologies, which params_dev for which sigma(parameter)...
	ind_pfid_r, ind_pfid_nT, ind_pfid_ns, ind_pfid_as, ind_pfid_mnu, ind_pfid_neff, ind_pfid_w, ind_pfid_omk = 0,1,0,0,0,0,0,0
	# from Stephen's email:
	#		-sigma(r) comes from LCDM+r
	#		-sigma(n_T) comes from LCDM+r+n_T [WITH r = 0.1!]
	#		-sigma(n_s) and sigma(alpha_s) come from LCDM+r+n_T+alpha_s (or did you want sigma(r, n_T) -- which presumably change a lot -- from this too?)
	#		-sigma(M_nu) comes from LCDM+M_nu
	#		-sigma(N_eff) comes from LCDM+N_eff
	#		-sigma(w_0) comes from wCDM
	#		-sigma(Omega_k) comes from LCDM+k
	ind_pdev_r, ind_pdev_nT, ind_pdev_ns, ind_pdev_as, ind_pdev_mnu, ind_pdev_neff, ind_pdev_w, ind_pdev_omk = 0,1,2,2,3,4,5,6

	exps_table = foregrounds.keys()
	ind = -1
	for exp in exps_table:
		
		filename = 'latex_table_sigmas_'+str(exp)
		filename += '_Bd_1.59_Td_19.6_Bs_-3.1'
		print ' CREATING AND SAVING ', filename
		if args.stolyarov_sync:
			filename += '_A_expansion'
		if args.cbass:
			filename += '_CBASS'
		if args.quijote:
			filename += '_QUIJOTE'
		if args.calibration_error!=0.0:
			filename += '_calib_error_'+str(args.calibration_error)

		output = r"""\begin{tabular}{|l|l||l|l||l|l||l|l||l|l|}		
\hline
\multicolumn{10}{|c|}{""" + str(exp) + r"""} \\
\hline
\hline
\multicolumn{2}{|l||}{Delensing option $\rightarrow$} &  \multicolumn{2}{c||}{no} & \multicolumn{2}{c||}{CMB x} &  \multicolumn{2}{c||}{CMB x} &  \multicolumn{2}{c|}{CMB x} \\
\multicolumn{2}{|l||}{$\downarrow$ comp. sep. option} &  \multicolumn{2}{c||}{delensing} & \multicolumn{2}{c||}{CMB} & \multicolumn{2}{c||}{CIB} & \multicolumn{2}{c|}{LSS} \\
\hline
"""
		
		for components in components_v:

			label1 = ' iterative delensing  CMBxCMB '
			label2 = ' iterative delensing  CMBxCIB '
			label3 = ' iterative delensing  CMBxLSS '
			label0 = ' no delensing '

			if components=='cmb-only' :
				label0 += ' no comp-sep '
				label1 += ' no comp-sep '
				label2 += ' no comp-sep '
				label3 += ' no comp-sep '
				label_alpha1 = 'alpha_CMBxCMB'
				label_alpha2 = 'alpha_CMBxCIB'
				label_alpha3 = 'alpha_CMBxLSS'
			else:
				label0 += ' + post-comp-sep '
				label1 += ' + post-comp-sep '
				label2 += ' + post-comp-sep '
				label3 += ' + post-comp-sep '
				label_alpha1 = 'alpha_CMBxCMB_post_comp_sep'
				label_alpha2 = 'alpha_CMBxCIB_post_comp_sep'
				label_alpha3 = 'alpha_CMBxLSS_post_comp_sep'

			delta_loc = ltx_round(foregrounds[exp][components]['delta'])
			reff_loc = ltx_round(foregrounds[exp][components]['r_eff'])

			## no delensing
			sigma_r_loc_none = ltx_round(sigmas[exp][components][label0]['marginalized']['r'][ind_pfid_r,ind_pdev_r])
			sigma_nT_loc_none = ltx_round(sigmas[exp][components][label0]['marginalized']['nT'][ind_pfid_nT,ind_pdev_nT])
			sigma_ns_loc_none = ltx_round(sigmas[exp][components][label0]['marginalized']['ns'][ind_pfid_ns,ind_pdev_ns])
			sigma_as_loc_none = ltx_round(sigmas[exp][components][label0]['marginalized']['alphas'][ind_pfid_as,ind_pdev_as])
			sigma_Mnu_loc_none = ltx_round(sigmas[exp][components][label0]['marginalized']['omnuh2'][ind_pfid_mnu,ind_pdev_mnu]*93.0)
			sigma_w_loc_none = ltx_round(sigmas[exp][components][label0]['marginalized']['w'][ind_pfid_w,ind_pdev_w])
			sigma_Neff_loc_none = ltx_round(sigmas[exp][components][label0]['marginalized']['Neff'][ind_pfid_neff,ind_pdev_neff])
			sigma_OmK_loc_none = ltx_round(sigmas[exp][components][label0]['marginalized']['omk'][ind_pfid_omk,ind_pdev_omk])

			## CMBCMB delensing
			sigma_r_loc_CMBxCMB = ltx_round(sigmas[exp][components][label0]['marginalized']['r'][ind_pfid_r,ind_pdev_r])
			sigma_nT_loc_CMBxCMB = ltx_round(sigmas[exp][components][label0]['marginalized']['nT'][ind_pfid_nT,ind_pdev_nT])
			sigma_ns_loc_CMBxCMB = ltx_round(sigmas[exp][components][label0]['marginalized']['ns'][ind_pfid_ns,ind_pdev_ns])
			sigma_as_loc_CMBxCMB = ltx_round(sigmas[exp][components][label0]['marginalized']['alphas'][ind_pfid_as,ind_pdev_as])
			sigma_Mnu_loc_CMBxCMB = ltx_round(sigmas[exp][components][label0]['marginalized']['omnuh2'][ind_pfid_mnu,ind_pdev_mnu]*93.0)
			sigma_w_loc_CMBxCMB = ltx_round(sigmas[exp][components][label0]['marginalized']['w'][ind_pfid_w,ind_pdev_w])
			sigma_Neff_loc_CMBxCMB = ltx_round(sigmas[exp][components][label0]['marginalized']['Neff'][ind_pfid_neff,ind_pdev_neff])
			sigma_OmK_loc_CMBxCMB = ltx_round(sigmas[exp][components][label0]['marginalized']['omk'][ind_pfid_omk,ind_pdev_omk])

			## CMBxCIB delensing
			sigma_r_loc_CMBxCIB = ltx_round(sigmas[exp][components][label0]['marginalized']['r'][ind_pfid_r,ind_pdev_r])
			sigma_nT_loc_CMBxCIB = ltx_round(sigmas[exp][components][label0]['marginalized']['nT'][ind_pfid_nT,ind_pdev_nT])
			sigma_ns_loc_CMBxCIB = ltx_round(sigmas[exp][components][label0]['marginalized']['ns'][ind_pfid_ns,ind_pdev_ns])
			sigma_as_loc_CMBxCIB = ltx_round(sigmas[exp][components][label0]['marginalized']['alphas'][ind_pfid_as,ind_pdev_as])
			sigma_Mnu_loc_CMBxCIB = ltx_round(sigmas[exp][components][label0]['marginalized']['omnuh2'][ind_pfid_mnu,ind_pdev_mnu]*93.0)
			sigma_w_loc_CMBxCIB = ltx_round(sigmas[exp][components][label0]['marginalized']['w'][ind_pfid_w,ind_pdev_w])
			sigma_Neff_loc_CMBxCIB = ltx_round(sigmas[exp][components][label0]['marginalized']['Neff'][ind_pfid_neff,ind_pdev_neff])
			sigma_OmK_loc_CMBxCIB = ltx_round(sigmas[exp][components][label0]['marginalized']['omk'][ind_pfid_omk,ind_pdev_omk])

			## CMBxLSS delensing
			sigma_r_loc_CMBxLSS = ltx_round(sigmas[exp][components][label0]['marginalized']['r'][ind_pfid_r,ind_pdev_r])
			sigma_nT_loc_CMBxLSS = ltx_round(sigmas[exp][components][label0]['marginalized']['nT'][ind_pfid_nT,ind_pdev_nT])
			sigma_ns_loc_CMBxLSS = ltx_round(sigmas[exp][components][label0]['marginalized']['ns'][ind_pfid_ns,ind_pdev_ns])
			sigma_as_loc_CMBxLSS = ltx_round(sigmas[exp][components][label0]['marginalized']['alphas'][ind_pfid_as,ind_pdev_as])
			sigma_Mnu_loc_CMBxLSS = ltx_round(sigmas[exp][components][label0]['marginalized']['omnuh2'][ind_pfid_mnu,ind_pdev_mnu]*93.0)
			sigma_w_loc_CMBxLSS = ltx_round(sigmas[exp][components][label0]['marginalized']['w'][ind_pfid_w,ind_pdev_w])
			sigma_Neff_loc_CMBxLSS = ltx_round(sigmas[exp][components][label0]['marginalized']['Neff'][ind_pfid_neff,ind_pdev_neff])
			sigma_OmK_loc_CMBxLSS = ltx_round(sigmas[exp][components][label0]['marginalized']['omk'][ind_pfid_omk,ind_pdev_omk])


			#############
			output += r' \multirow{4}{*}{'+components+r'} & \multirow{3}{*}{$\Delta='+delta_loc+r'$} & \multicolumn{2}{c||}{$\alpha=1.0$} & \multicolumn{2}{c||}{$\alpha='+ltx_round(Nl[exp][components][label_alpha1])+r'$} & \multicolumn{2}{c||}{$\alpha='+ltx_round(Nl[exp][components][label_alpha2])+r'$} & \multicolumn{2}{c|}{$\alpha='+ltx_round(Nl[exp][components][label_alpha3])+r'$}\\'
			output +=  r' & & $\sigma(r)='+ltx_round(sigmas[exp][components][label0]['marginalized']['r'][ind_pfid_r,ind_pdev_r])+r'$ & $\sigma(\mnu)='+ltx_round(sigmas[exp][components][label0]['marginalized']['omnuh2'][ind_pfid_mnu,ind_pdev_mnu]*93000)+r'$ & $\sigma(r)='+ltx_round(sigmas[exp][components][label1]['marginalized']['r'][ind_pfid_r,ind_pdev_r])+r'$ & $\sigma(\mnu)='+ltx_round(sigmas[exp][components][label1]['marginalized']['omnuh2'][ind_pfid_mnu,ind_pdev_mnu]*93000)+r'$ & $\sigma(r)='+ltx_round(sigmas[exp][components][label2]['marginalized']['r'][ind_pfid_r,ind_pdev_r])+r'$ & $\sigma(M_\nu)='+ltx_round(sigmas[exp][components][label2]['marginalized']['omnuh2'][ind_pfid_mnu,ind_pdev_mnu]*93000)+r'$ & $\sigma(r)='+ltx_round(sigmas[exp][components][label3]['marginalized']['r'][ind_pfid_r,ind_pdev_r])+r'$ & $\sigma(\mnu)='+ltx_round(sigmas[exp][components][label3]['marginalized']['omnuh2'][ind_pfid_mnu,ind_pdev_mnu]*93000)+r'$ \\'
			#output +=  r' \multirow{4}{*}{'+components+r'} & \multirow{2}{*}{$\Delta='+ltx_round(foregrounds[exp][components]['delta'])+r'$} & $\sigma(r)='+ltx_round(sigmas[exp][components][label0]['marginalized']['r'][ind_pfid_r,ind_pdev_r])+r'$ & $\sigma(M_\nu)='+ltx_round(sigmas[exp][components][label0]['marginalized']['omnuh2'][ind_pfid_mnu,ind_pdev_mnu]*93.0)+r'$ & $\sigma(r)='+ltx_round(sigmas[exp][components][label1]['marginalized']['r'][ind_pfid_r,ind_pdev_r])+r'$ & $\sigma(M_\nu)='+ltx_round(sigmas[exp][components][label1]['marginalized']['omnuh2'][ind_pfid_mnu,ind_pdev_mnu]*93.0)+r'$ & $\sigma(r)='+ltx_round(sigmas[exp][components][label2]['marginalized']['r'][ind_pfid_r,ind_pdev_r])+r'$ & $\sigma(M_\nu)='+ltx_round(sigmas[exp][components][label2]['marginalized']['omnuh2'][ind_pfid_mnu,ind_pdev_mnu]*93.0)+r'$ & $\sigma(r)='+ltx_round(sigmas[exp][components][label3]['marginalized']['r'][ind_pfid_r,ind_pdev_r])+r'$ & $\sigma(M_\nu)='+ltx_round(sigmas[exp][components][label3]['marginalized']['omnuh2'][ind_pfid_mnu,ind_pdev_mnu]*93.0)+r'$ \\'
			output +=  r' & & $\sigma(\nt)='+ltx_round(sigmas[exp][components][label0]['marginalized']['nT'][ind_pfid_nT,ind_pdev_nT])+r'$ & $\sigma(\w)='+ltx_round(sigmas[exp][components][label0]['marginalized']['w'][ind_pfid_w,ind_pdev_w])+r'$ & $\sigma(\nt)='+ltx_round(sigmas[exp][components][label1]['marginalized']['nT'][ind_pfid_nT,ind_pdev_nT])+r'$ & $\sigma(\w)='+ltx_round(sigmas[exp][components][label1]['marginalized']['w'][ind_pfid_w,ind_pdev_w])+r'$ & $\sigma(\nt)='+ltx_round(sigmas[exp][components][label2]['marginalized']['nT'][ind_pfid_nT,ind_pdev_nT])+r'$ & $\sigma(\w)='+ltx_round(sigmas[exp][components][label2]['marginalized']['w'][ind_pfid_w,ind_pdev_w])+r'$ & $\sigma(\nt)='+ltx_round(sigmas[exp][components][label3]['marginalized']['nT'][ind_pfid_nT,ind_pdev_nT])+r'$ & $\sigma(\w)='+ltx_round(sigmas[exp][components][label3]['marginalized']['w'][ind_pfid_w,ind_pdev_w])+r'$ \\'
			output +=  r' & \multirow{2}{*}{$\reff='+ltx_round(foregrounds[exp][components]['r_eff'])+r'$} & $\sigma(\ns)='+ltx_round(sigmas[exp][components][label0]['marginalized']['ns'][ind_pfid_ns,ind_pdev_ns])+r'$ & $\sigma(\neff)='+ltx_round(sigmas[exp][components][label0]['marginalized']['Neff'][ind_pfid_neff,ind_pdev_neff])+r'$ & $\sigma(\ns)='+ltx_round(sigmas[exp][components][label1]['marginalized']['ns'][ind_pfid_ns,ind_pdev_ns])+r'$ & $\sigma(\neff)='+ltx_round(sigmas[exp][components][label1]['marginalized']['Neff'][ind_pfid_neff,ind_pdev_neff])+r'$ & $\sigma(\ns)='+ltx_round(sigmas[exp][components][label2]['marginalized']['ns'][ind_pfid_ns,ind_pdev_ns])+r'$ & $\sigma(\neff)='+ltx_round(sigmas[exp][components][label2]['marginalized']['Neff'][ind_pfid_neff,ind_pdev_neff])+r'$ & $\sigma(\ns)='+ltx_round(sigmas[exp][components][label3]['marginalized']['ns'][ind_pfid_ns,ind_pdev_ns])+r'$ & $\sigma(\neff)='+ltx_round(sigmas[exp][components][label3]['marginalized']['Neff'][ind_pfid_neff,ind_pdev_neff])+r'$ \\'
			#output +=  r' & & $\sigma(\alpha_s)='+ltx_round(sigmas[exp][components][label0]['marginalized']['alphas'][ind_pfid_as,ind_pdev_as])+r'$ & $\alpha= 1.0 $ & $\sigma(\alpha_s)='+ltx_round(sigmas[exp][components][label1]['marginalized']['alphas'][ind_pfid_as,ind_pdev_as])+r'$ & $\alpha='+ltx_round(Nl[exp][components][label_alpha1])+r'$ & $\sigma(\alpha_s)='+ltx_round(sigmas[exp][components][label2]['marginalized']['alphas'][ind_pfid_as,ind_pdev_as])+r'$ & $\alpha='+ltx_round(Nl[exp][components][label_alpha2])+r'$ & $\sigma(\alpha_s)='+ltx_round(sigmas[exp][components][label3]['marginalized']['alphas'][ind_pfid_as,ind_pdev_as])+r'$ & $\alpha='+ltx_round(Nl[exp][components][label_alpha3])+r'$ \\'
			output +=  r' & & $\sigma(\alphas)='+ltx_round(sigmas[exp][components][label0]['marginalized']['alphas'][ind_pfid_as,ind_pdev_as])+r'$ & $\sigma(\omk)='+ltx_round(sigmas[exp][components][label0]['marginalized']['omk'][ind_pfid_omk,ind_pdev_omk])+r'$ & $\sigma(\alphas)='+ltx_round(sigmas[exp][components][label1]['marginalized']['alphas'][ind_pfid_as,ind_pdev_as])+r'$ & $\sigma(\omk)='+ltx_round(sigmas[exp][components][label1]['marginalized']['omk'][ind_pfid_omk,ind_pdev_omk])+r'$ & $\sigma(\alphas)='+ltx_round(sigmas[exp][components][label2]['marginalized']['alphas'][ind_pfid_as,ind_pdev_as])+r'$ & $\sigma(\omk)='+ltx_round(sigmas[exp][components][label2]['marginalized']['omk'][ind_pfid_omk,ind_pdev_omk])+r'$ & $\sigma(\alphas)='+ltx_round(sigmas[exp][components][label3]['marginalized']['alphas'][ind_pfid_as,ind_pdev_as])+r'$ & $\sigma(\omk)='+ltx_round(sigmas[exp][components][label3]['marginalized']['omk'][ind_pfid_omk,ind_pdev_omk])+r'$ \\'
			output +=  r"""
\hline"""
		output += """
\end{tabular}"""
		path = './'
		text_file = open(os.path.join( path, filename+".tex"), "w")
		text_file.write("%s" % output )
		text_file.close()

	print ' code successfully finished ! '
	exit()


############################################################################################################################################################################################################


#####################################################
##                   example call                  ##
#####################################################
#@app.task
def forecast( fsky=0.1, freqs=[95, 150, 220], uKCMBarcmin=[10.0, 10.0, 10.0], FWHM=[5.0, 3.0, 2.0], \
	ell_max=2000, ell_min=20, Bd=1.59, Td=19.6, Bs=-3.1, \
	components_v=[0,1,0,0], delensing_option_v=[0,1,0,0], \
	params_dev_v=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], information_channels_v=[1,1,1,1]):

	configurations_loc= {}
	configurations['my_exp'] = {}
	configurations['my_exp']['freqs'] = freqs
	configurations['my_exp']['uKCMBarcmin'] = uKCMBarcmin
	configurations['my_exp']['FWHM'] = FWHM
	configurations['my_exp']['fsky'] = fsky
	configurations['my_exp']['bandpass'] = 0.3*np.ones(len(configurations['my_exp']['freqs']))
	configurations['my_exp']['ell_min'] = ell_min
	configurations['my_exp']['ell_max'] = ell_max

	############
	camb_loc = '/home/josquin/softwares/camb/./camb'
	information_channels_ini =['Tu', 'Eu', 'Bu', 'd']
	delensing_option_v_ini = ['','CMBxCMB','CMBxCIB', 'CMBxLSS']
	params_dev_full_ini   = ['ns', 'As', 'tau', 'h', 'ombh2', 'omch2', \
							'alphas', 'r', 'nT', 'omk', 'omnuh2', 'Neff', \
							'YHe', 'w', 'wa' ]
	components_v_ini = ['cmb-only', 'dust', 'sync', 'sync+dust' ]

	############
	params_dev_v_loc = []
	for i in range( len(params_dev_v) ):
		if params_dev_v[i] == 1:
			params_dev_v_loc.append(params_dev_full_ini[i] )
	delensing_option_v_loc = []
	for i in range( len(delensing_option_v) ):
		if delensing_option_v[i] == 1:
			delensing_option_v_loc.append(delensing_option_v_ini[i] )
	information_channels_loc = []
	for i in range( len(information_channels_v) ):
		if information_channels_v[i] == 1:
			information_channels_loc.append(information_channels_ini[i] )
	components_v_loc = []
	for i in range( len(components_v) ):
		if components_v[i] == 1:
			components_v_loc.append(components_v_ini[i] )

	############
	# actual computation
	foregrounds, sigmas, Nl, Cls_fid = core_function(configurations_loc, components_v=components_v_loc, \
		camb=camb_loc, fgs_scatter=False, delens_scatter=False, params_fid_v=[params_fid_v[0]], \
		params_dev_v=[params_dev_v_loc], information_channels=information_channels_loc, \
		delensing_option_v=delensing_option_v_loc, param_priors_v=[], cross_only=False, Bd=Bd, Td=Td, Bs=Bs )	

	############
	# printing output
	output = ''
	for exp in sigmas.keys():
		for components in sigmas[exp].keys():
			output += ' components = '+components 
			for label in sigmas[exp][components].keys():
				output +=  'label = '+label
				for parameter in  sigmas[exp][components][label]['marginalized'].keys():
					output +=  'sigma('+parameter+') = '+ str( sigmas[exp][components][label]['marginalized']['r'] )
				output +=  'effective level of foregrounds residuals, r_eff = '+str( foregrounds[exp][components]['r_eff'] )
				output +=  'degradation of the noise after comp. sep. = '+str( foregrounds[exp][components]['delta'] )
	
	return sigmas 


#####################################################
##                core functionalities             ##
#####################################################
def core_function(configurations, components_v, camb, \
	params_fid_v, params_dev_v, information_channels, delensing_option_v, \
	delensing_z_max=-1.0, param_priors_v=[], cross_only=False, Bd=1.59, \
	Td=19.6, Bs=-3.1, correlation_2_dusts=0.0, stolyarov=False, stolyarov_sync=False, cbass=False, quijote=False, \
	delens_command_line=False, comp_sep_only=False, calibration_error=0.0, np_nside=4, no_lensing=False, A_lens=1.0, \
	resolution=False, DESI=False, mpi_safe=False, path2maps='/Users/josquin1/Documents/Dropbox/planck_maps',\
	path2Cls='/Users/josquin1/Documents/Dropbox/self_consistent_forecast/codes/', spectral_parameters=spectral_parameters, \
    analytic_expr_per_template=analytic_expr_per_template, bandpass_channels={}, drv=drv, \
    prior_spectral_parameters=prior_spectral_parameters ):
    
	################ entry checker
	if Bd == 0.0 or Td == 0.0:
		print  'Wrong dust grey body parameters'
		exit()
	for exp in configurations.keys():
		if configurations[exp]['ell_min'] >= configurations[exp]['ell_max']:
			print  'Wrong ell_min or ell_max'
			exit()

	#####################################################
	## in preparation of the delensing for each experiment 
	# compute fiducial Cls computed between ell = 2 and 6000
	print '################################ computing fiducial Cls ... ####################################'
	name_fid = 'fidCls'
	fnames_fid = glob.glob( os.path.join(path2Cls, name_fid+'*.pkl' ))

	output = {}
	for file_ in fnames_fid:
		output[file_] = 0
	for p in range(len(params_fid_v[0].keys())):
		name_fid = '_'+str(params_fid_v[0].keys()[p])+'_'+str(params_fid_v[0][params_fid_v[0].keys()[p]] )+'_'
		for file_ in fnames_fid:
			if name_fid in file_:
				output[file_] += 1

	fnames_fid = [max(output.iteritems(), key=operator.itemgetter(1))[0]]

	if output[fnames_fid[0]] < len(params_fid_v[0].keys())-1 :
		fnames_fid = []

	if not fnames_fid:
		print '################# computing Cls file because it does not seem to be on disk #######################'
		Cls_fid=python_camb_self_consistent.submit_camb( h=params_fid_v[0]['h'], ombh2=params_fid_v[0]['ombh2'], omch2=params_fid_v[0]['omch2'], \
			omnuh2=params_fid_v[0]['omnuh2'], omk=params_fid_v[0]['omk'], YHe=params_fid_v[0]['YHe'], Neff=params_fid_v[0]['Neff'], w=params_fid_v[0]['w'], \
			wa=params_fid_v[0]['wa'], tau=params_fid_v[0]['tau'],As=params_fid_v[0]['As'], ns=params_fid_v[0]['ns'], alphas=params_fid_v[0]['alphas'], nT=params_fid_v[0]['nT'], \
			r=params_fid_v[0]['r'], k_scalar=params_fid_v[0]['k_scalar'] , k_tensor=params_fid_v[0]['k_tensor'], eta=1.0, lensing_z_max=delensing_z_max, exe = camb)
		save_obj('./', name_fid, Cls_fid)
	else:
		print '################################ loading already existing Cls file ####################################'
		Cls_fid = load_obj('/global/homes/j/josquin/FORECAST/self_consistent_forecast/codes/', fnames_fid[0])

	if 'BuBu_r1' not in Cls_fid.keys():
		Cls_fid['BuBu_r1'] = Cls_fid['BuBu'] / params_fid_v[0]['r']

	experiments = sorted(configurations.keys(), reverse=True)
	## computing uKRJ/pix for each experiment
	for exp in experiments:
		nfreqs = len(configurations[exp]['freqs'])
		configurations[exp]['uKRJ/pix'] = np.zeros(nfreqs)
		for f in range( nfreqs ):
			uKCMB_perpixel_f = configurations[exp]['uKCMBarcmin'][f] / pix_size_map_arcmin
			uKRJ_perpixel_f = uKCMB_perpixel_f*residuals_comp.BB_factor_computation( configurations[exp]['freqs'][f] )
			configurations[exp]['uKRJ/pix'][f] = uKRJ_perpixel_f 
			del uKCMB_perpixel_f, uKRJ_perpixel_f

	foregrounds = {}
	ells = {}

	ind = -1
	for exp1 in experiments:
		
		ind += 1

		for exp2 in experiments[ind:]:

			if exp1 == exp2 :
				# experiment alone, exp is the name of the instrument that we study
				exp = exp1

				if cross_only: continue

			else:
				# combination of two experiments
				exp = exp1+' x '+exp2
				configurations[exp] = {}
				configurations[exp]['uKCMBarcmin'] = np.hstack(( configurations[exp1]['uKCMBarcmin'] , configurations[exp2]['uKCMBarcmin'] ))
				configurations[exp]['FWHM'] = np.hstack(( configurations[exp1]['FWHM'] , configurations[exp2]['FWHM'] ))
				configurations[exp]['uKRJ/pix'] = np.hstack(( configurations[exp1]['uKRJ/pix'] , configurations[exp2]['uKRJ/pix'] ))
				configurations[exp]['freqs'] = np.hstack(( configurations[exp1]['freqs'] , configurations[exp2]['freqs'] ))
				configurations[exp]['ell_min'] = np.max( [ configurations[exp1]['ell_min'] , configurations[exp2]['ell_min']])
				configurations[exp]['ell_max'] = np.min( [ configurations[exp1]['ell_max'], configurations[exp2]['ell_max']])
				configurations[exp]['bandpass'] = np.hstack(( configurations[exp1]['bandpass'], configurations[exp2]['bandpass'] ))
				configurations[exp]['fsky'] = np.min([ configurations[exp1]['fsky'], configurations[exp2]['fsky'] ])
				configurations[exp]['prior_dust'] = np.min([configurations[exp1]['prior_dust'], configurations[exp2]['prior_dust'] ])
				configurations[exp]['prior_sync'] = np.min([configurations[exp1]['prior_sync'], configurations[exp2]['prior_sync'] ])
				configurations[exp]['ell_knee'] = np.hstack(( configurations[exp1]['ell_knee']*np.ones(len(configurations[exp1]['freqs'])) , configurations[exp2]['ell_knee']*np.ones(len(configurations[exp2]['freqs'])) ))
				configurations[exp]['alpha_knee'] = np.hstack(( configurations[exp1]['alpha_knee'], configurations[exp2]['alpha_knee'] ))
	
			ells[exp] =np.array(range(ell_min_camb, configurations[exp]['ell_max']+1))

			# comp sep !
			Cls_fid_r = fc.derivatives_computation(Cls_fid, ['r'], params_fid, information_channels, exe=camb, path2Cls = path2Cls)['r']
			foregrounds[exp] = {}
			# foregrounds[exp] = CMB4cast_compsep.CMB4cast_compsep(configurations=configurations, components_v=components_v, exp=exp, \
							# cbass=cbass, quijote=quijote, np_nside=np_nside, ell_min_camb=ell_min_camb, Cls_fid=Cls_fid,\
							# stolyarov=stolyarov, stolyarov_sync=stolyarov_sync, calibration_error=calibration_error, Bd=Bd, \
							# Td=Td, Bs=Bs, camb=camb, ells_exp=ells[exp], path2maps=path2maps, path2Cls=path2Cls, Cls_fid_r=Cls_fid_r )
			foregrounds[exp] = CMB4cast_compsep.CMB4cast_compsep(configurations=configurations, components_v=components_v, exp=exp, \
									np_nside=np_nside, ell_min_camb=ell_min_camb, Cls_fid=Cls_fid, spectral_parameters=spectral_parameters, \
									analytic_expr_per_template=analytic_expr_per_template, camb=camb, drv=drv, \
									prior_spectral_parameters=prior_spectral_parameters, ells_exp=ells[exp], \
									path2files=path2maps )
			pissoffpython()

	########################################
	## COMPUTE Nls for each experimental setup (NlTT, NlEE, NlBB, Nldd)
	Nl = CMB4cast_noise.noise_computation( configurations=configurations, foregrounds=foregrounds, components_v=components_v, \
						resolution=resolution, stolyarov=stolyarov, stolyarov_sync=stolyarov_sync, ell_min_camb=ell_min_camb,\
						Cls_fid=Cls_fid, ells_exp=ells[exp], experiments=experiments)

	if comp_sep_only:
		return foregrounds

	##########################################################
	#
	#   D E L E N S
	# 
	#########################################################
	# set up correlation data for use in CIB/LSS delensing
	# and default delensing settings
	converge = 0.01
	if delensing_z_max > 0.0:
		f_l_cor_cib, f_l_cor_lss = CMB4cast_delens.f_l_cor_setup(path2Cls, \
																 lss = True, \
																 Cls_fid = Cls_fid)
	else:
		f_l_cor_cib = CMB4cast_delens.f_l_cor_setup(path2Cls)

	# if you ask for no delensing but don't have CMBxCMB
	if ('' in delensing_option_v) and ('CMBxCMB' not in delensing_option_v):
		delensing_option_v_loc = copy.copy( delensing_option_v )
		delensing_option_v_loc.append('CMBxCMB')
	else:
		delensing_option_v_loc = copy.copy(delensing_option_v)
	Nl_in = copy.copy(Nl)
    
	# choose appropriate delensing forecast function
	if delens_command_line:
		CMB4cast_delens.cmd_delens(experiments, configurations, \
								   components_v, \
								   delensing_option_v_loc, \
								   Cls_fid, f_l_cor_cib, f_l_cor_lss, Nl, \
								   converge = converge, \
								   cross_only = cross_only, \
								   mpi_safe = mpi_safe)
	else:
		CMB4cast_delens.delens(experiments, configurations, \
							   components_v, delensing_option_v_loc, \
							   Cls_fid, f_l_cor_cib, f_l_cor_lss, Nl,\
							   converge = converge, \
							   cross_only = cross_only)

	##############################################################################
	## F O R E C A S T I N G
	##############################################################################

	sigmas = CMB4cast_Fisher.forecast_fisher(params_fid_v=params_fid_v, camb=camb, params_dev_full=params_dev_full, information_channels=information_channels, \
				configurations=configurations, components_v=components_v, delensing_option_v=delensing_option_v, Nl=Nl,\
				path2Cls=path2Cls, cross_only=cross_only, no_lensing=no_lensing, params_dev_v=params_dev_v, DESI=DESI, \
				param_priors_v=param_priors_v, ell_min_camb=ell_min_camb, ells=ells, foregrounds=foregrounds,\
				experiments=experiments, A_lens=A_lens )

	return foregrounds, sigmas, Nl, Cls_fid

#####################################################################################################
#####################################################
##           command-line options parser           ##
#####################################################
def grabargs():

	# check whether user has specified a path to CAMB
	parser = argparse.ArgumentParser()
	parser.add_argument("--camb", \
						help = "full path of CAMB executable (default = ./camb)", \
						default = "./camb")
	parser.add_argument("--stolyarov", action='store_true',\
						help = "the component separation uses the Stolyarov approach i.e. considering spatially varying spectral indices", \
						default = False)
	parser.add_argument("--stolyarov_sync", action='store_true',\
						help = "the component separation uses the Stolyarov approach i.e. considering spatially varying spectral indices for BOTH DUST AND SYNCHROTRON", \
						default = False)
	parser.add_argument("--calibration_error", type=float ,\
						help = "sigma(omega) -> error on calibration is included for the estimation of foregroudns residuals", \
						default = 0.0)
	parser.add_argument("--cbass", action='store_true',\
						help = "adding CBASS to any considered instrument", \
						default = False)
	parser.add_argument("--quijote", action='store_true',\
						help = "adding Quijote to any considered instrument", \
						default = False)
	parser.add_argument("--cross_only", action='store_true',\
						help = "just compute cross-instruments only", \
						default = False)
	parser.add_argument("--delens_command_line", action='store_true',\
						help = "force the use of command-line delensing instead of the f2py-compiled module", \
						default = False)
	parser.add_argument("--mpi_safe", action='store_true',\
			    help = "make any temporary files MPI-safe by randomising name", \
			    default = False)
	parser.add_argument("--path2Cls", \
						help = "full path to Cls/derivatives", \
						default ="/Users/josquin1/Documents/Dropbox/self_consistent_forecast/codes/")
	parser.add_argument("--path2maps", \
						help = "full path to foregrounds and CMB maps", \
						default = "/Users/josquin1/Documents/Dropbox/planck_maps")

	args = parser.parse_args()
	if not os.path.isfile(args.camb):
		raise IOError("specified camb executable (" + args.camb + ") does not exist")
		
	return args

if __name__ == "__main__":

	initialize()

