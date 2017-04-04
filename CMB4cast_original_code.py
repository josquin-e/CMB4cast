#!/usr/bin/env python

'''
Estimate performances of several CMB projects
wrt. their delensing power, their ability to separate components
as well as constrain cosmological parameters such as neutrino mass, w0/wa, r and nT
'''

#from celery import Celery
#app = Celery('main_merged', backend='mongodb://localhost/turkeycalltest', broker='mongodb://localhost/turkeycalltest')
import numpy as np
import pylab as pl
import healpy as hp
import forecasting_cosmo_self_consistent_forecast_smf as fc
import fnmatch
import operator
import residuals_computation_loc_calibration_errors as residuals_comp
import scipy
from scipy import polyval, polyfit, optimize
import sys
import os
#import pyfits
import python_camb_self_consistent
import argparse
import time
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter
import matplotlib.cm as cm
from matplotlib import rc
import copy 
import glob
import pickle
from scipy.ndimage import gaussian_filter1d
from math import log10, floor
import copy
host = os.environ['NERSC_HOST']
if "cori" in host:
	from pydelens import delens_tools as pd
elif "edison" in host:
	from pydelens_edison import delens_tools as pd
else:
	print ' host name not resolved : ', host
	exit()	
import residuals_computation_extended_self_consistent_forecast_calibration_errors as res_ext
import random
import string
import subprocess as sp

############################################################################################################################################################################################################
############################################################################################################################################################################################################
## useful functions

def save_obj(path, name, obj):
	with open(os.path.join( path, name + '.pkl' ), 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path, name):
	print 'loading ... ', os.path.join( path, name )
	with open(os.path.join( path, name ), 'r') as f:
		return pickle.load(f)

def round_sig(x, sig=3):
	if x != 0.0 and x==x:
		a  = round(x, sig-int(floor(log10(x)))-1)
		return "%.2e" % ( a )
	else:
		return round( x )

# slightly different version of above function. "sig" is the number of
# significant figures to display; if abs(log10(x)) is larger than explim,
# the number is displayed in exponential format
def ltx_round(x, explim = 2, sig = 2):
	if x == 0.0:
		return str(x)
	if np.abs(np.log10(x)) > explim:
		fmt = '{:.'+'{:d}'.format(sig-1)+'e}'
		pys = fmt.format(x)
		pos = pys.find('e')
		return pys[0:pos] + r'\times10^{{{:}}}'.format(int(pys[pos+1:]))
	else:
		fmt = '{:.'+'{:d}'.format(sig-1-int(np.floor(np.log10(x))))+'f}'
		return fmt.format(x)

def ticks_format(value, index):
	"""
	get the value and returns the value as:
		integer: [0,99]
		1 digit float: [0.1, 0.99]
		n*10^m: otherwise
	To have all the number of the same size they are all returned as latex strings
	"""
	exp = np.floor(np.log10(value))
	base = value/10**exp
	if exp == 0 or exp < 3:
		return '${0:d}$'.format(int(value))
	if exp == -1:
		return '${0:.1f}$'.format(value)
	else:
		return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))

def eformat(x, y):
    e = floor(np.log10(x))
    m = x / 10.0 ** e#round(x / 10.0 ** e)
    if e == 0:
        return r'$%1.1f$' % m
    else:
		return '${0:2.1f}\\times10^{{{1:d}}}$'.format(m, int(e))

def lmin_computation(fsky, loc=None):
	fsky_rad = fsky*4*np.pi
	lmin = int(np.ceil( np.pi/(2*np.sqrt( fsky )) ))
	if ((loc == 'ground') or (loc == 'balloon')) and (lmin < 20):
		lmin = 20
	return lmin

def logl( Cl0, Cl1 ):
	return np.sum( np.log(Cl0) + Cl1/Cl0 )

############################################################################################################################################################################################################
############################################################################################################################################################################################################
## few constants

fullsky_arcmin2 = 4*np.pi*(180.0*60/np.pi)**2
arcmin_to_radian = np.pi/(180.0*60.0)
pix_size_2048_arcmin = np.sqrt( 4.0*np.pi*(180*60/np.pi)**2 / (12*2048**2) )
pix_size_1024_arcmin = np.sqrt( 4.0*np.pi*(180*60/np.pi)**2 / (12*1024**2) ) 
pix_size_512_arcmin = np.sqrt( 4.0*np.pi*(180*60/np.pi)**2 / (12*512**2) ) 
seconds_per_year = 365.25*24*3600
ell_min_camb = 2
ell_max_abs = 4000
common_nside = 128 ## check the spsp matrices used in foregrounds section
pix_size_map_arcmin = hp.nside2resol(common_nside, arcmin=True)

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
expts['Stage-IV']['bandpass'] = 0.3*np.ones(len(expts['Stage-IV']['freqs']))
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

configurations = dict((expt, expts[expt]) for expt in ('LiteBIRD_extended', \
													   'LiteBIRD_update'))


##############################################################################
## bar plot of sensitivities vs frequencies for post-2020 instruments
#pl.figure()
#pl.figure()
#pl.loglog( configurations['PIXIE']['freqs'], configurations['PIXIE']['uKCMBarcmin'], 'ko', alpha=0.5)
#pl.ylabel('$\mu$K-arcmin', fontsize=20)
#pl.xlabel('GHz', fontsize=20)
#pl.show()
#exit()
"""
f, axes = pl.subplots(len(configurations.keys()), sharex=True, sharey=True, figsize=(13,11))
for i in range(len(configurations.keys())):
	#axes[i].set_title( configurations.keys()[i], fontsize=18)
	axes[i].text( 200, 100, configurations.keys()[i], fontsize=24 )
	#axes[i].text( 200, 30, configurations.keys()[i], fontsize=24 )
	axes[i].bar( configurations[configurations.keys()[i]]['freqs'], configurations[configurations.keys()[i]]['uKCMBarcmin'], np.array(configurations[configurations.keys()[i]]['bandpass'])*np.array(configurations[configurations.keys()[i]]['freqs']), alpha=0.5, color=cb_colors[i] )
	axes[i].set_xscale("log")
	axes[i].set_yscale("log")
	#print np.max(configurations[configurations.keys()[i]]['uKCMBarcmin'])
	axes[i].set_ylim([ 1.0, 400 ])
	#axes[i].set_ylim([ 1.0, 50 ])
	axes[i].set_xlim([ 35, 550])
	axes[i].set_ylabel('sensitivity\n [$\mu$K-arcmin]', fontsize=24)
	for tick in axes[i].xaxis.get_major_ticks():
		tick.label.set_fontsize(0)
	for tick in axes[i].xaxis.get_minor_ticks():
		tick.label.set_fontsize(0)
	for tick in axes[i].yaxis.get_major_ticks():
		tick.label.set_fontsize(26)
axes[i].set_xlabel('frequency [GHz]', fontsize=24)
subs = [ 1.0, 2.0, 5.0 ]  # ticks to show per decade
#from matplotlib.ticker import LogLocator
#axes[i].xaxis.set_major_locator(LogLocator(base = 10.0))
#start, stop = axes[i].get_xlim()
#ticks = np.arange(start, stop + None, None)
#axes[i].set_xticks(ticks)
axes[i].xaxis.set_major_locator( ticker.LogLocator( subs=subs ) )
axes[i].xaxis.set_minor_locator( ticker.LogLocator( subs=subs ) ) #set the ticks position
axes[i].xaxis.set_major_formatter( ticker.NullFormatter() )   # remove the major ticks
axes[i].xaxis.set_minor_formatter( ticker.FuncFormatter(eformat) )  #add the custom tick
#for i in range(len(configurations.keys())-1):
#	axes[i].get_xaxis().set_visible(False)
#ax = pl.gca()
for tick in axes[i].xaxis.get_major_ticks():
	tick.label.set_fontsize(26)
for tick in axes[i].xaxis.get_minor_ticks():
	tick.label.set_fontsize(26)
#pl.savefig('./LB_sensitivities_vs_freq.pdf')
pl.savefig('./LB_sensitivities_vs_freq.pdf')
pl.show()
exit()
"""
"""
f, axes = pl.subplots(1, sharex=True, sharey=True, figsize=(12,8 ))
axes.text( 200, 100, 'Planck', fontsize=24 )
axes.bar( configurations['Planck']['freqs'], configurations['Planck']['uKCMBarcmin'], np.array(configurations['Planck']['bandpass'])*np.array(configurations['Planck']['freqs']), alpha=0.5, color='DarkGray' )
axes.set_xscale("log")
axes.set_yscale("log")
axes.set_ylim([ 1.0, 400 ])
axes.set_xlim([ 25, 550])
axes.set_ylabel('sensitivity\n [$\mu$K-arcmin]', fontsize=24)
axes.set_xlabel('frequency [GHz]', fontsize=24)
box = axes.get_position()
axes.set_position([box.x0, box.y0+0.2, box.width, box.height*0.8])
subs = [ 1.0, 2.0, 5.0 ]  # ticks to show per decade
axes.xaxis.set_major_locator( ticker.LogLocator( subs=subs ) )
axes.xaxis.set_minor_locator( ticker.LogLocator( subs=subs ) ) #set the ticks position
axes.xaxis.set_major_formatter( ticker.NullFormatter() )   # remove the major ticks
axes.xaxis.set_minor_formatter( ticker.FuncFormatter(eformat) )  #add the custom tick
for tick in axes.yaxis.get_major_ticks():
	tick.label.set_fontsize(26)
for tick in axes.xaxis.get_major_ticks():
	tick.label.set_fontsize(26)
for tick in axes.xaxis.get_minor_ticks():
	tick.label.set_fontsize(26)
pl.savefig('./LB_sensitivities_vs_freq_Planck.pdf')
pl.show()
exit()
"""

###########################################################################
# figures multipole coverage vs frequency
"""
fig = pl.figure(num=None, figsize=(12,10), facecolor='w', edgecolor='k')
from matplotlib.patches import Rectangle
ind=0
if ('Simons Array' in configurations.keys()):
	stage = 'III'
else:
	stage = 'IV'
print 'plotting stage {:} experiments'.format(stage)
filename='instruments_ell_and_freq_coverage'
filename+='_stage'+stage

if stage=='III':
	pl.title('pre-2020 projects' + '\n', fontsize=32)
	n_leg_col = 3
else:
	pl.title('post-2020 projects' + '\n', fontsize=32)
	n_leg_col = 2

patches = []
exp_labs = []
ax = pl.gca()
linewidth = 5.0
alpha= 0.8
for inst in configurations.keys():
	ell_min_loc = expts[inst]['ell_min']
	freq_min_loc = np.min(expts[inst]['freqs'])


	n_ell_pol = np.zeros(ell_max_abs - ell_min_camb + 1)
	w_inv = ( np.array(expts[inst]['uKCMBarcmin'][:] ) * arcmin_to_radian) ** 2
	for ell in range(ell_min_camb, ell_max_abs + 1):
		beam_l = np.exp((np.array(expts[inst]['FWHM'][:]) * arcmin_to_radian / np.sqrt(8.0*np.log(2.0))) ** 2 * (ell * (ell + 1.0)))
		n_ell_pol[ell - ell_min_camb] = ( (ell * (ell + 1.0) / (2.0 * np.pi)) / np.sum( 1.0 / w_inv / beam_l) )
		n_ell_pol[n_ell_pol > 1.0e20] = 1.0e20
	cross = np.where(n_ell_pol > 1.0e4)[0]
	if not len(cross):
		ell_max_loc = ell_max_abs
	else:
		ell_max_loc = cross[0]
	edgecolor = cb_colors[ind]
	if expts[inst]['loc'] == 3:
		linestyle='dashed'
	else:
		linestyle='solid'
	patch = ax.add_patch(Rectangle((ell_min_loc, freq_min_loc), ell_max_loc-ell_min_loc, \
					     np.max(expts[inst]['freqs'])-freq_min_loc, fill=None, \
					     edgecolor=edgecolor, linewidth=linewidth, \
					     alpha=alpha, linestyle=linestyle))
	exp_labs.append(inst)
	patches.append(patch)

	print '==========='
	print inst, ell_max_loc, np.min(expts[inst]['freqs']), np.max(expts[inst]['freqs'])
	print '==========='

	ind+=1
pl.xlim([1,ell_max_abs])
pl.ylim([25,6010])
ax.set_yscale('log')
ax.set_xscale('log')
for tick in ax.xaxis.get_major_ticks():
	tick.label.set_fontsize(26)
for tick in ax.yaxis.get_major_ticks():
	tick.label.set_fontsize(26)
exp_labs, patches = zip(*sorted(zip(exp_labs, patches), key=lambda t: (t[0], t[1])))
lgd = pl.legend(patches, exp_labs, loc = 'lower center', fontsize = 20, ncol = n_leg_col, \
					bbox_to_anchor = (0.5, -0.3), numpoints = 1)
fr = lgd.get_frame()
fr.set_lw(1.5)
fr.set_facecolor('white')
ax.grid(True)
pl.ylabel('frequency [GHz]', fontsize=32)
pl.xlabel('$\ell$', fontsize=32)
pl.savefig('../article/'+filename+'.pdf', \
			bbox_inches='tight')
pl.show()
exit()
"""

############################################################################################################################################################################################################
########################################################################
# compute noise, delta_beta and Clres from noise per channel in RJ
#components_v = ['cmb-only', 'dust', 'sync', 'sync+dust' ]
components_v = ['cmb-only', 'sync+dust']#, 'sync+dust+dust']

foregrounds = {}
ells = {}

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
			components_v=components_v, camb=args.camb, fgs_scatter=args.fgs_scatter, \
			delens_scatter=args.delens_scatter, params_fid_v=params_fid_sel, \
			params_dev_v=params_dev_sel, information_channels=information_channels, \
			delensing_option_v=delensing_option_v, delensing_z_max=delensing_z_max, \
			param_priors_v=[], cross_only=args.cross_only, Bd=1.59, Td=19.6, Bs=-3.1, \
			#param_priors_v=[], cross_only=args.cross_only, Bd=1.7, Td=10.0, Bs=-2.7, \
			fgs_vs_freq_vs_ell=args.fgs_vs_freq_vs_ell, \
			power_spectrum_figure=args.power_spectrum_figure, \
			fgs_power_spectrum=args.fgs_power_spectrum, \
			delens_power_spectrum=args.delens_power_spectrum, \
			combo_power_spectrum=args.combo_power_spectrum, \
			stolyarov=args.stolyarov, stolyarov_sync=args.stolyarov_sync,\
			cbass=args.cbass, quijote=args.quijote, \
			delens_command_line=args.delens_command_line, calibration_error=args.calibration_error)
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
	return sigmas.keys()


############################################################################################################################################################################################################


#####################################################
##                core functionality               ##
#####################################################
def core_function(configurations, components_v, camb, fgs_scatter, delens_scatter, \
	params_fid_v, params_dev_v, information_channels, delensing_option_v, \
	delensing_z_max=-1.0, param_priors_v=[], cross_only=False, Bd=1.59, \
	Td=19.6, Bs=-3.1, correlation_2_dusts=0.0, fgs_vs_freq_vs_ell=False, \
	power_spectrum_figure=False, fgs_power_spectrum=False, delens_power_spectrum=False, \
	combo_power_spectrum=False, stolyarov=False, stolyarov_sync=False, cbass=False, quijote=False, \
	delens_command_line=False, comp_sep_only=False, calibration_error=0.0, np_nside=4, no_lensing=False,\
	resolution=False, DESI=False,\
	mpi_safe=False):

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
	'''
	name_fid = 'fidCls'
	for p in range(len(params_fid_v[0].keys())):
		name_fid += '_'+str(params_fid_v[0].keys()[p])+'_'+str(params_fid_v[0][params_fid_v[0].keys()[p]] )
	print 'looking for ', name_fid+'.pkl'
	fnames_fid = glob.glob( name_fid+'.pkl' )
	'''
	name_fid = 'fidCls'
	fnames_fid = glob.glob( os.path.join('/global/homes/j/josquin/FORECAST/self_consistent_forecast/codes/', name_fid+'*.pkl' ))

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

	experiments = sorted(configurations.keys(),reverse=True)
	## computing uKRJ/pix for each experiment
	for exp in experiments:
		nfreqs = len(configurations[exp]['freqs'])
		configurations[exp]['uKRJ/pix'] = np.zeros(nfreqs)
		for f in range( nfreqs ):
			uKCMB_perpixel_f = configurations[exp]['uKCMBarcmin'][f] / pix_size_map_arcmin
			uKRJ_perpixel_f = uKCMB_perpixel_f*residuals_comp.BB_factor_computation( configurations[exp]['freqs'][f] )
			configurations[exp]['uKRJ/pix'][f] = uKRJ_perpixel_f 
			del uKCMB_perpixel_f, uKRJ_perpixel_f

	ind = -1
	#pl.figure()
	for exp1 in experiments:
		
		ind += 1

		for exp2 in experiments[ind:]:

			if exp1 == exp2 :
				# experiment alone, exp is the name of the instrument that we study
				exp = exp1
				#if 'information_channels' in configurations[exp].keys():
				#	print ' specific set of information channels ! '
				#	information_channels =  copy.copy( configurations[exp]['information_channels'] )			

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
	
				### TO BE CODED ... the case of cross correlation btw instruments having different information_channels
				# just keep the intersection of information channels ?
				#if 'information_channels' in configurations[exp1].keys():
				#	if 'information_channels' in configurations[exp2].keys():
				#		information_channels = list(set(configurations[exp1]['information_channels']).intersection(configurations[exp1]['information_channels']))
				#	else:
				#		information_channels = 
				#elif 'information_channels' in configurations[exp2].keys():
				#	information_channels

			foregrounds[exp] = {}

			# definition of ell range
			ells[exp]=np.array(range(ell_min_camb, configurations[exp]['ell_max']+1))

			# assumed that nside=8 pixels over the sky have independent spectral parameters that have to be estimated
			npatch = int( configurations[exp]['fsky']*12*np_nside**2 )
			if npatch == 0 : npatch=1

			###############################################################################
			## FOREGROUNDS MAPS INFOS 
			# grabbing the correct maps and spectra for the specified fsky
			path_to_planck = os.path.abspath('/global/homes/j/josquin/FORECAST/self_consistent_forecast/planck_maps')


			fskys = np.array([20, 40, 60, 70, 80, 90, 97, 99])
			print 'for exp=', exp, ', fsky= ', configurations[exp]['fsky']
			indf=np.argmin(np.abs( configurations[exp]['fsky'] - np.array(fskys)/100.0 ))
			print' and indf =', indf
			
			if not (stolyarov or stolyarov_sync):
				spsp_in = np.load( os.path.join(path_to_planck,'spsp_uKRJ_fsky_'+str( fskys[indf] )+'_nside_'+str(common_nside)+'.npy' ))	

			elif stolyarov_sync:
				name_spsp = os.path.join(path_to_planck,'spsp_uKRJ_STOLYAROV_fsky_'+str( fskys[indf] )+'_nside_'+str(common_nside)+'.npy' )

				if os.path.exists( name_spsp ):
					spsp_in = np.load( name_spsp )	
				else:
					print '/!\ error : you need to run exploit_planck_maps.py first, with the relevant nside'
					exit()
			
			else:
				name_spsp = os.path.join(path_to_planck,'spsp_uKRJ_STOLYAROV_dust_only_fsky_'+str( fskys[indf] )+'_nside_'+str(common_nside)+'.npy' )

				if os.path.exists( name_spsp ):
					spsp_in = np.load( name_spsp )	
				else:
					print '/!\ error : you need to run exploit_planck_maps.py first, with the relevant nside'
					exit()


			fskys_ext = np.hstack(( 0.0, 10.0, fskys )) #, 100.0 ))
			
			spsp_in_tot = np.zeros((spsp_in.shape[0], spsp_in.shape[1], len(fskys_ext) ))


			for f in range(len(fskys_ext))[2:]:

				if not (stolyarov or stolyarov_sync):
					spsp_in_loc = np.load( os.path.join(path_to_planck,'spsp_uKRJ_fsky_'+str( fskys[f-2] )+'_nside_'+str(common_nside)+'.npy' ))	
				elif stolyarov_sync:
					spsp_in_loc = np.load( os.path.join(path_to_planck,'spsp_uKRJ_STOLYAROV_fsky_'+str( fskys[f-2] )+'_nside_'+str(common_nside)+'.npy' ) )
				else:
					spsp_in_loc = np.load( os.path.join(path_to_planck,'spsp_uKRJ_STOLYAROV_dust_only_fsky_'+str( fskys[f-2] )+'_nside_'+str(common_nside)+'.npy' ) )

				for i in range(spsp_in.shape[0]):
					for j in range(spsp_in.shape[1]):
						spsp_in_tot[i,j,f] = spsp_in_loc[i,j]

			for i in range(spsp_in.shape[0]):
				for j in range(spsp_in.shape[1]):
					spsp_in_tot[i,j,0] = 0.0

			for i in range(spsp_in.shape[0]):
				for j in range(spsp_in.shape[1]):
					spsp_in_tot[i,j,1] = spsp_in_tot[i,j,2]*(10.0/20.0)

			fskys_int = np.linspace(0.005, 0.99, num=200)
			spsp_in_int = np.zeros((spsp_in.shape[0], spsp_in.shape[1], len(fskys_int) ))

			for i in range(spsp_in.shape[0]):
				for j in range(spsp_in.shape[1]):

					#N = 6
					#p  = polyfit( fskys_ext/100.0, spsp_in_tot[i,j,:], N)
					#for k in range( N ):
					#	spsp_in_int[i,j,:] = a*fskys_int**4  + b*fskys_int**3 + c*fskys_int**2 + d*fskys_int + e
					#	spsp_in_int[i,j,:] += p[k]*fskys_int**(N-k)
					f = scipy.interpolate.interp1d(fskys_ext/100.0, spsp_in_tot[i,j,:], kind='cubic', bounds_error=False)
					spsp_in_int[i,j,:] = f(fskys_int)

					#pl.figure()
					#pl.title( str(i)+' / '+str(j))
					#pl.plot( fskys_ext/100.0, spsp_in_tot[i,j,:] , 'ko' )
					#pl.plot( fskys_int, spsp_in_int[i,j,:], 'r--'  ) 
					#pl.show()


			indf = np.argmin(np.abs( configurations[exp]['fsky'] - fskys_int ))

			spsp_in = np.squeeze(spsp_in_int[:,:,indf])*1.0

			if not (stolyarov or stolyarov_sync):
				if (spsp_in[4,4] < np.max(spsp_in)*1e-8) or (spsp_in[2,2] < np.max(spsp_in)*1e-8):
					correlation_dustxsync = 0.0
				else:
					correlation_dustxsync = spsp_in[4,2]/np.sqrt( spsp_in[4,4]*spsp_in[2,2] )
			elif stolyarov_sync:
				if (spsp_in[7,7] < np.max(spsp_in)*1e-8) or (spsp_in[2,2] < np.max(spsp_in)*1e-8):
					correlation_dustxsync = 0.0
				else:
					correlation_dustxsync = spsp_in[7,2]/np.sqrt( spsp_in[7,7]*spsp_in[2,2] )
			else:
				if (spsp_in[7,7] < np.max(spsp_in)*1e-8) or (spsp_in[2,2] < np.max(spsp_in)*1e-8):
					correlation_dustxsync = 0.0
				else:
					correlation_dustxsync = spsp_in[7,2]/np.sqrt( spsp_in[7,7]*spsp_in[2,2] )

			###############################################################################
			## FOREGROUNDS POWER SPECTRA 
			#Cls_dust0 = np.load( os.path.join(path_to_planck,'Cls_dust_uKRJ2_fsky_'+str( fskys[indf] )+'_nside_512.npy' ))
			#Cls_sync0 = np.load( os.path.join(path_to_planck,'Cls_sync_uKRJ2_fsky_'+str( fskys[indf] )+'_nside_512.npy' ))
			#Cls_dxs0 = np.load( os.path.join(path_to_planck,'Cls_dxs_uKRJ2_fsky_'+str( fskys[indf] )+'_nside_512.npy' ))
			
			fskys_Planck = np.array([0.0, 5.0, 7.0, 10.0, 20.0, 24, 33, 42, 53, 63, 72, 80, 99])
			ind_70 = np.argmin(np.abs(72 - fskys_ext ))
			ind_80 = np.argmin(np.abs(80 - fskys_ext ))
			ind_99 = np.argmin(np.abs(99 - fskys_ext ))
			factor_80 = np.sum(spsp_in_tot[2:4,2:4,ind_80])/np.sum(spsp_in_tot[2:4,2:4,ind_70])
			factor_99 = np.sum(spsp_in_tot[2:4,2:4,ind_99])/np.sum(spsp_in_tot[2:4,2:4,ind_70])
			AEE_Planck = np.array([0.0, 4.5, 7.5, 10.0, 30.0, 37.5, 51.0, 78.6, 124.2, 197.1, 328.0, 328.0*factor_80, 328.0*factor_99 ] )

			f = scipy.interpolate.interp1d(fskys_Planck/100.0, AEE_Planck, kind='linear', bounds_error=False)
			AEE_Planck_int = f(fskys_int)

			indf_Planck = np.argmin(np.abs( configurations[exp]['fsky']-np.array(fskys_int) ))

			A350 = residuals_comp.A_element_computation(350, 1.59, -3.1, squeeze=True)
			A350 /= residuals_comp.BB_factor_computation(350)
			A150 = residuals_comp.A_element_computation(150, 1.59, -3.1, squeeze=True)
			A150 /= residuals_comp.BB_factor_computation(150)
			A70 = residuals_comp.A_element_computation(70, 1.59, -3.1, squeeze=True)
			A70 /= residuals_comp.BB_factor_computation(70)
			s70_dust = A70[1]/A350[1]
			s150_dust = A150[1]/A350[1]
			s150_sync = A150[2]/A70[2]

			Cls_dust = ( AEE_Planck_int[indf_Planck]*s150_dust**2/2 )*(Cls_fid['ell']*1.0/80.0)**(-0.4)
			Cls_dust_70GHz = ( AEE_Planck_int[indf_Planck]*s70_dust**2/2 )*(Cls_fid['ell']*1.0/80.0)**(-0.4)	
			ind_loc = np.argmin(np.abs(Cls_fid['ell'] - 200))
			Cls_sync_70GHz = Cls_dust_70GHz[ind_loc]*(Cls_fid['ell']*1.0/200.0 )**(-0.6)
			Cls_sync = Cls_sync_70GHz*s150_sync**2
			Cls_dxs = correlation_dustxsync*np.sqrt( Cls_dust * Cls_sync )
			norm = Cls_fid['ell']*(Cls_fid['ell'] + 1)/(2*np.pi)
			
			#A100 = residuals_comp.A_element_computation(100, 1.59, -3.1, squeeze=True)
			#A100 /= residuals_comp.BB_factor_computation(100)
			#A150 = residuals_comp.A_element_computation(150, 1.59, -3.1, squeeze=True)
			#A150 /= residuals_comp.BB_factor_computation(150)
			#s100_dust = A100[1]/A150[1]
			#s100_sync = A100[2]/A150[2]
			Cls_fid['dust@150GHz'] = Cls_dust#*s100_dust**2
			Cls_fid['sync@150GHz'] = Cls_sync#*s100_sync**2

			# AEE is in uK_CMB**2, going back to RJ units 
			Cls_dust *= residuals_comp.BB_factor_computation(150.0)**2/norm
			Cls_sync *= residuals_comp.BB_factor_computation(150.0)**2/norm
			Cls_dxs *= residuals_comp.BB_factor_computation(150.0)**2/norm

			######################################################################################
			## doing the left panel of Fig. 2	
			if power_spectrum_figure:

				ells_loc = Cls_fid['ell']#np.arange(2, 4000)
				ind_l_min = np.argmin(np.abs(Cls_fid['ell'] - 2))
				ind_l_max = np.argmin(np.abs(Cls_fid['ell'] - 3000))
				fsky = 70

				A353 = residuals_comp.A_element_computation(353, 1.59, -3.1, squeeze=True)
				A353 /= residuals_comp.BB_factor_computation(353)
				A150 = residuals_comp.A_element_computation(150, 1.59, -3.1, squeeze=True)
				A150 /= residuals_comp.BB_factor_computation(150)
				A70 = residuals_comp.A_element_computation(70, 1.59, -3.1, squeeze=True)
				A70 /= residuals_comp.BB_factor_computation(70)
				A100 = residuals_comp.A_element_computation(100, 1.59, -3.1, squeeze=True)
				A100 /= residuals_comp.BB_factor_computation(100)
				A200 = residuals_comp.A_element_computation(200, 1.59, -3.1, squeeze=True)
				A200 /= residuals_comp.BB_factor_computation(200)
				s100_dust = A100[1]/A353[1]
				s100_sync = A100[2]/A70[2]
				s150_dust = A150[1]/A353[1]
				s150_sync = A150[2]/A70[2]
				s200_dust = A200[1]/A353[1]
				s200_sync = A200[2]/A70[2]
				s70_dust = A70[1]/A353[1]

				Cl_dust_100GHz_fsky5 = (AEE_Planck_int[0]*s100_dust**2/2)*(ells_loc*1.0/80.0)**(-0.4)
				Cl_dust_100GHz_fsky75 = (AEE_Planck_int[-1]*s100_dust**2/2)*(ells_loc*1.0/80.0)**(-0.4)
				Cl_dust_200GHz_fsky5 = (AEE_Planck_int[0]*s200_dust**2/2)*(ells_loc*1.0/80.0)**(-0.4)
				Cl_dust_200GHz_fsky75 = (AEE_Planck_int[-1]*s200_dust**2/2)*(ells_loc*1.0/80.0)**(-0.4)
				Cl_dust_70GHz_fsky5 = (AEE_Planck_int[0]*s70_dust**2/2)*(ells_loc*1.0/80.0)**(-0.4)	
				Cl_dust_70GHz_fsky75 = (AEE_Planck_int[-1]*s70_dust**2/2)*(ells_loc*1.0/80.0)**(-0.4)	

				Cl_sync_70GHz_fsky5 = Cl_dust_70GHz_fsky5[198]*(ells_loc*1.0/200.0)**(-0.6)
				Cl_sync_70GHz_fsky75 = Cl_dust_70GHz_fsky75[198]*(ells_loc*1.0/200.0)**(-0.6)
				Cl_sync_100GHz_fsky5 = Cl_sync_70GHz_fsky5*s100_sync**2
				Cl_sync_150GHz_fsky5 = Cl_sync_70GHz_fsky5*s150_sync**2
				Cl_sync_100GHz_fsky75 = Cl_sync_70GHz_fsky75*s100_sync**2
				Cl_sync_200GHz_fsky5 = Cl_sync_70GHz_fsky5*s200_sync**2
				Cl_sync_200GHz_fsky75 = Cl_sync_70GHz_fsky75*s200_sync**2
			
				fig = pl.figure(num=None, figsize=(12,10), facecolor='w', edgecolor='k')
				pl.loglog(1e-10, 1e-10, color='DarkBlue', linewidth=3.0, label=r'primordial $B$ modes', alpha=0.5)
				pl.loglog(ells_loc[ind_l_min:ind_l_max], Cls_fid['EE'][ind_l_min:ind_l_max], color='red', linewidth=3.0, alpha=0.75, label=r'$E$ modes')
				pl.loglog(ells_loc[ind_l_min:ind_l_max], Cls_fid['TT'][ind_l_min:ind_l_max], color='black', linewidth=3.0, alpha=0.75, label=r'temperature')
				# BB lens
				pl.loglog(ells_loc[ind_l_min:ind_l_max], Cls_fid['BlBl'][ind_l_min:ind_l_max], color='DarkGray', linewidth=3.0, label=r'lensing $B$ modes', alpha=0.75)
				# dust
				pl.loglog(  ells_loc[ind_l_min:ind_l_max], Cl_dust_100GHz_fsky5[ind_l_min:ind_l_max]+Cl_sync_100GHz_fsky5[ind_l_min:ind_l_max], color='DarkOrange', alpha=0.75, linewidth=2.0)
				pl.loglog(  ells_loc[ind_l_min:ind_l_max], Cl_dust_100GHz_fsky75[ind_l_min:ind_l_max]+Cl_sync_100GHz_fsky75[ind_l_min:ind_l_max], color='DarkOrange', alpha=0.75, linewidth=2.0)
				pl.loglog(  ells_loc[ind_l_min:ind_l_max], Cl_dust_200GHz_fsky5[ind_l_min:ind_l_max]+Cl_sync_200GHz_fsky5[ind_l_min:ind_l_max], color='OrangeRed', linestyle='--', alpha=0.75, linewidth=2.0)
				pl.loglog(  ells_loc[ind_l_min:ind_l_max], Cl_dust_200GHz_fsky75[ind_l_min:ind_l_max]+Cl_sync_200GHz_fsky75[ind_l_min:ind_l_max], color='OrangeRed', linestyle='--', alpha=0.75, linewidth=2.0)
				pl.fill_between( ells_loc[ind_l_min:ind_l_max], Cl_dust_100GHz_fsky5[ind_l_min:ind_l_max]+Cl_sync_100GHz_fsky5[ind_l_min:ind_l_max], Cl_dust_100GHz_fsky75[ind_l_min:ind_l_max]+Cl_sync_100GHz_fsky75[ind_l_min:ind_l_max], color='DarkOrange', alpha=0.25 )
				pl.fill_between( ells_loc[ind_l_min:ind_l_max], Cl_dust_100GHz_fsky5[ind_l_min:ind_l_max]+Cl_sync_100GHz_fsky5[ind_l_min:ind_l_max], Cl_dust_100GHz_fsky75[ind_l_min:ind_l_max]+Cl_sync_100GHz_fsky75[ind_l_min:ind_l_max], color='none', alpha=0.7, hatch="/", edgecolor="DarkGray", linewidth=1.0)
				pl.fill_between( ells_loc[ind_l_min:ind_l_max], Cl_dust_100GHz_fsky5[ind_l_min:ind_l_max]+Cl_sync_100GHz_fsky5[ind_l_min:ind_l_max], Cl_dust_100GHz_fsky75[ind_l_min:ind_l_max]+Cl_sync_100GHz_fsky75[ind_l_min:ind_l_max], color='none', alpha=0.7, hatch="/", edgecolor="DarkOrange", linewidth=1.0)
				pl.fill_between( ells_loc[ind_l_min:ind_l_max], Cl_dust_200GHz_fsky5[ind_l_min:ind_l_max]+Cl_sync_200GHz_fsky5[ind_l_min:ind_l_max], Cl_dust_200GHz_fsky75[ind_l_min:ind_l_max]+Cl_sync_200GHz_fsky75[ind_l_min:ind_l_max], color='OrangeRed', alpha=0.1)

				from matplotlib.font_manager import FontProperties
				font0 = FontProperties()
				font = font0.copy()
				#font.set_weight( 'bold' )
				ind_l = np.argmin(np.abs(ells_loc-300))
				pl.text(  ells_loc[ind_l], 0.45*(Cl_dust_100GHz_fsky5[ind_l]+Cl_sync_100GHz_fsky5[ind_l]), '       $C_\ell^{\mathrm{dust+synchrotron}}$'+'\n'+'$f_{sky}=1\%$ @ 100GHz', fontsize=18, color='black', rotation=-6, fontproperties=font )
				ind_l = np.argmin(np.abs(ells_loc-3))
				pl.text(  ells_loc[ind_l], 0.45*(Cl_dust_100GHz_fsky75[ind_l]+Cl_sync_100GHz_fsky75[ind_l]), '    $C_\ell^{\mathrm{dust+synchrotron}}$'+'\n'+'$f_{sky}=90\%$ @ 100GHz', fontsize=18, color='black', rotation=-6, fontproperties=font )

				ind_l = np.argmin(np.abs(ells_loc-300))
				pl.text(  ells_loc[ind_l], 0.45*(Cl_dust_200GHz_fsky5[ind_l]+Cl_sync_200GHz_fsky5[ind_l]), '    $C_\ell^{\mathrm{dust+synchrotron}}$'+'\n'+'$f_{sky}=1\%$ @ 200GHz', fontsize=18, color='black', rotation=-6, fontproperties=font )
				ind_l = np.argmin(np.abs(ells_loc-300))
				pl.text(  ells_loc[ind_l], 0.45*(Cl_dust_200GHz_fsky75[ind_l]+Cl_sync_200GHz_fsky75[ind_l]), '      $C_\ell^{\mathrm{dust+synchrotron}}$'+'\n'+'$f_{sky}=90\%$ @ 200GHz', fontsize=18, color='black', rotation=-6, fontproperties=font )

				ind_l = np.argmin(np.abs(ells_loc-3))
				pl.text( 0.7*ells_loc[ind_l], 0.5*Cls_fid['BuBu'][ind_l]*0.1/params_fid['r'], 'r=0.1', fontsize=16, color='DarkBlue', fontproperties=font, alpha=0.5 )
				pl.text( 0.7*ells_loc[ind_l], 0.5*Cls_fid['BuBu'][ind_l]*0.01/params_fid['r'], 'r=0.01', fontsize=16, color='DarkBlue', fontproperties=font, alpha=0.5 )
				pl.text( 0.7*ells_loc[ind_l], 0.5*Cls_fid['BuBu'][ind_l]*0.001/params_fid['r'], 'r=0.001', fontsize=16, color='DarkBlue', fontproperties=font, alpha=0.5)

				## BB for r=0.1
				pl.loglog(ells_loc[ind_l_min:ind_l_max], Cls_fid['BuBu'][ind_l_min:ind_l_max]*0.1/params_fid['r'], color='DarkBlue', linewidth=3.0, alpha=0.5)
				#pl.fill_between( ells_loc[ind_l_min:ind_l_max], Cls_fid['BuBu'][ind_l_min:ind_l_max]*0.1/params_fid['r']-CVr01[ind_l_min:ind_l_max]/2.0, Cls_fid['BuBu'][ind_l_min:ind_l_max]*0.1/params_fid['r']+CVr01[ind_l_min:ind_l_max]/2.0, alpha=0.5 , color='DarkGray')#, label='cosmic variance')
				pl.loglog(ells_loc[ind_l_min:ind_l_max], Cls_fid['BuBu'][ind_l_min:ind_l_max]*0.01/params_fid['r'], color='DarkBlue', linewidth=3.0, alpha=0.5)
				#pl.fill_between( ells_loc[ind_l_min:ind_l_max], Cls_fid['BuBu'][ind_l_min:ind_l_max]*0.01/params_fid['r']-CVr001[ind_l_min:ind_l_max]/2.0, Cls_fid['BuBu'][ind_l_min:ind_l_max]*0.01/params_fid['r']+CVr001[ind_l_min:ind_l_max]/2.0, alpha=0.5 , color='DarkGray')#, label='cosmic variance')	
				pl.loglog(ells_loc[ind_l_min:ind_l_max], Cls_fid['BuBu'][ind_l_min:ind_l_max]*0.001/params_fid['r'], color='DarkBlue', linewidth=3.0, alpha=0.5)
				#pl.fill_between( ells_loc[ind_l_min:ind_l_max],  Cls_fid['BuBu'][ind_l_min:ind_l_max]*0.001/params_fid['r']-CV[ind_l_min:ind_l_max]/2.0, Cls_fid['BuBu'][ind_l_min:ind_l_max]*0.001/params_fid['r']+CV[ind_l_min:ind_l_max]/2.0, alpha=0.5 , color='DarkGray')#, label='cosmic variance')
				#pl.plot( 1e-10, 1e-10, alpha=0.5 , linewidth=10.0, color='DarkGray', label='cosmic variance')
				#pl.loglog(ells[ind_l_min:ind_l_max], np.sqrt(Cls_fid['BBr01'][ind_l_min:ind_l_max]), color='DarkGray', linewidth=3.0, alpha=0.75)
				#pl.loglog(ells[ind_l_min:ind_l_max], np.sqrt(Cls_fid['BBr001'][ind_l_min:ind_l_max]), color='DarkGray', linewidth=3.0, alpha=0.75)
				#pl.loglog(ells[ind_l_min:ind_l_max], np.sqrt(Cls_fid['BBr0001'][ind_l_min:ind_l_max]), color='DarkGray', linewidth=3.0, alpha=0.75)
				pl.loglog(ells_loc[ind_l_min:ind_l_max], Cls_fid['BlBl'][ind_l_min:ind_l_max], color='DarkGray', linewidth=3.0, alpha=0.75 )

				pl.xlim( [2, 3000] )
				pl.ylim([5e-6, 1e4])
				ax = pl.gca()
				for tick in ax.xaxis.get_major_ticks():
					tick.label.set_fontsize(26)
				for tick in ax.yaxis.get_major_ticks():
					tick.label.set_fontsize(26)

				legend1 = ax.legend( loc='upper left', prop={'size':20}, ncol=1, labelspacing=0.5)
				frame1 = legend1.get_frame()
				frame1.set_edgecolor('white')
				legend1.get_frame().set_alpha(0.9)

				pl.ylabel('$ (\ell(\ell+1)/2\pi )\, C_\ell\ [\mu \mathrm{K}^2] $', fontsize=32)
				pl.xlabel('$\ell$', fontsize=32)
				pl.savefig('../article/Power_Spectrum_figure_showing_foregrounds.pdf', \
							bbox_inches='tight')
				pl.show()
				exit()

			######################################################################################
			## doing the right panel of Fig. 2	
			if fgs_vs_freq_vs_ell:

				frequencies = np.logspace(np.log10(4), np.log10(600), 200, base=10.0)
				Cls_fgs = np.zeros((len(frequencies), len( Cls_fid['ell'] ) ))
				norm_loc = Cls_fid['ell']*(Cls_fid['ell']+1)/(2*np.pi)
				for i_f in range(len(frequencies)):
					Af = residuals_comp.A_element_computation(frequencies[i_f], 1.59, -3.1, squeeze=True)
					Af /= residuals_comp.BB_factor_computation(frequencies[i_f])
					s_dust = Af[1]/A150[1]
					s_sync = Af[2]/A150[2]
					Cls_fgs[i_f,:] = Cls_dust[:]*( s_dust )**2 + Cls_sync[:]*( s_sync )**2
				
				ratio = Cls_fgs*0.0
				for i_l in range(len(Cls_fid['ell'])):
					#ratio[:,i_l] = Cls_fgs[:,i_l]*norm[i_l]/Cls_fid['BB'][i_l]
					ratio[:,i_l] = ( Cls_fgs[:,i_l]*norm_loc[i_l] + Cls_fid['BlBl'][i_l] )/(Cls_fid['BuBu'][i_l]/params_fid['r'])

				ind_min_0, ind_min_1 = np.unravel_index(ratio.argmin(), ratio.shape)
				print ind_min_0, ind_min_1 
				print frequencies[ind_min_0], Cls_fid['ell'][ind_min_1]
				min_freq, min_ell = int(round(frequencies[ind_min_0])), int(round(Cls_fid['ell'][ind_min_1]))
				#exit()

				fig = pl.figure(num=None, figsize=(12,10), facecolor='w', edgecolor='k')
				#pl.title('ratio of foregrounds over total B-modes angular power spectra:\n $C_\ell^{dust+synchrotron}/C_\ell^{BB}$ with $r=0.001$', fontsize=20)
				levels = [0.1, 1.0, 10.0, 100.0, 1e3, 1e6, 1e9, 1e12]
				#levels = np.logspace( np.log10(np.min(ratio)), np.log10(np.max(ratio)), 100 )
				#ratio = scipy.ndimage.zoom(ratio, 3)
				ratio = scipy.ndimage.filters.gaussian_filter(ratio, sigma=2.0)
				CS = pl.contourf( Cls_fid['ell'], frequencies, ratio, 1000, cmap=cm.Greys, origin='lower', locator=ticker.LogLocator()) 
				pl.axhline(frequencies[ind_min_0], color='r', linestyle='--', alpha=1.0, linewidth=3.0)
				pl.text(5.0, frequencies[ind_min_0]*1.1, str(min_freq)+' GHz', color='r', alpha=1.0, fontsize=20)
				pl.axvline(Cls_fid['ell'][ind_min_1], color='r', linestyle='--', alpha=1.0, linewidth=3.0)
				pl.text(Cls_fid['ell'][ind_min_1]*1.1, 120.0, '$\ell=$'+str(min_ell), color='r', alpha=1.0, fontsize=20, rotation=90)
				#CS = pl.contour( ells[exp], frequencies, ratio, levels, linewidths=3.0, linestyles='solid', alpha=0.9, colors='DarkGray', hold='on', origin='lower')
				fmt = ticker.LogFormatterMathtext()
				ax = pl.gca()
				ax.set_xscale('log')
				ax.set_yscale('log')
				#pl.axis(v)
				subs = [ 1.0, 2.0, 3.0, 5.0 ]  # ticks to show per decade
				ax.yaxis.set_minor_locator( ticker.LogLocator( subs=subs ) ) #set the ticks position
				ax.yaxis.set_major_formatter( ticker.NullFormatter() )   # remove the major ticks
				ax.yaxis.set_minor_formatter( ticker.FuncFormatter(ticks_format) )  #add the custom tick
				#ax.xaxis.set_minor_locator( ticker.LogLocator( subs=subs ) ) #set the ticks position
				#ax.xaxis.set_major_formatter( ticker.NullFormatter() )   # remove the major ticks
				#ax.xaxis.set_minor_formatter( '1.1f' )  #add the custom tick
				for tick in ax.yaxis.get_major_ticks():
					tick.label.set_fontsize(26)
				for tick in ax.yaxis.get_minor_ticks():
					tick.label.set_fontsize(26)
				for tick in ax.xaxis.get_major_ticks():
					tick.label.set_fontsize(26)
				for tick in ax.xaxis.get_minor_ticks():
					tick.label.set_fontsize(26)
				pl.ylabel('frequency [GHz]', fontsize=32)
				pl.xlabel('$\ell$', fontsize=32)
				cb = pl.colorbar()
				cb.set_label('$(C_\ell^{\mathrm{dust+sync}} + C_\ell^{BB,\,{\mathrm{lens}}})/C_\ell^{BB,\,{\mathrm{prim}}}(r=1)$', fontsize=32)
				cb.ax.tick_params(labelsize=24) 
				#pl.clabel(CS, inline=1, fontsize=18, fmt=fmt, manual=True )
				CS2= pl.contour( Cls_fid['ell'], frequencies, ratio, CS.levels, linewidths=1.0, linestyles='solid', alpha=0.9, colors='black', hold='on', origin='lower')
				fmt = {}
				indl=0
				for l in CS2.levels:
					print CS2.levels[indl]
					fmt[l] = '$r='+str(CS2.levels[indl])[:-2]+'$'
					indl+=1
				pl.clabel(CS2, inline=1, fontsize=18, fmt=fmt )
				pl.savefig('../article/fgs_over_BB_vs_frequency_vs_ell_contourf_NEW_DEF.pdf', \
							bbox_inches='tight')
				pl.show()
				exit()

			######################################################################################
			######################################################################################

			for components in components_v:

				foregrounds[exp][components] = {}

				if not stolyarov and not stolyarov_sync:

					print ' ################################################## '
					print ' ####### COMP. SEP. WITH N_patches APPROACH ####### '
					print ' ################################################## '

					if (components == 'dust+dust') or (components=='sync+dust+dust'):
						two_dusts = True
					else:
						two_dusts = False
					
					if 'dust' not in components:
						spsp = spsp_in[np.ix_([0,1,4,5],[0,1,4,5])]
					elif 'sync' not in components:
						spsp = spsp_in[np.ix_([0,1,2,3],[0,1,2,3])]
						if two_dusts:
							spsp = np.vstack(( spsp_in[np.ix_([0,1,2,3],[0,1,2,3])], np.hstack((spsp_in[:2,2].T, np.zeros(2).T )), np.hstack(( spsp_in[:2,3].T, 0.0, 0.0)) ))
							b = np.hstack(( spsp[4:,:2], np.zeros((2,2)), spsp[2:4, 2:4] ))
							spsp = np.hstack(( spsp, b.T ))
							# correlated 2 dust components ? 
							spsp[2:4, 4:6] *= correlation_2_dusts
							spsp[4:6, 2:4] *= correlation_2_dusts
					else:
						
						if two_dusts:				
							## should be cmb, dust#1, sync, dust#2
							spsp = np.vstack(( spsp_in[np.ix_([0,1,2,3],[0,1,2,3])], np.hstack((spsp_in[:2,2].T, spsp_in[2:4,2].T*correlation_2_dusts )), np.hstack(( spsp_in[:2,3].T, spsp_in[2:4,3].T*correlation_2_dusts)) ))
							b = np.hstack(( spsp[4:,:2], spsp[2:4, 2:4]*correlation_2_dusts, spsp[2:4, 2:4] ))
							spsp = np.hstack(( spsp, b.T ))
							# correlated 2 dust components ? 
							#spsp[2:4, 4:6] *= correlation_2_dusts
							#spsp[4:6, 2:4] *= correlation_2_dusts
							# add synchrotron
							spsp = np.vstack(( spsp, np.hstack((spsp_in[:4,4].T, spsp_in[2:4,4].T)), np.hstack((spsp_in[:4,5].T, spsp_in[2:4,5].T )) ))
							b = np.hstack(( spsp[6:8,:2].T, spsp[6:8,2:4].T, spsp[6:8,4:6].T, spsp_in[2:4,2:4].T ))
							spsp = np.hstack(( spsp, b.T ))
						else:
							spsp = spsp_in

					# evaluating residuals power spectrum and noise in the final CMB map
					if components == 'dust':
						components_loc = ['dust']
					elif components == 'dust+dust':
						components_loc = ['dust']
					elif components == 'sync+dust':
						components_loc = ['sync','dust']
					elif components == 'sync+dust+dust':
						components_loc = ['sync','dust', 'dust']
					elif components == 'cmb-only':
						components_loc = ['']
					elif components == 'sync':
						components_loc = ['sync']
					else:
						print ' /!\  I do not understand this component: ', components
						exit()


					if cbass:
						print ' ||||||||||| ADDING C-BASS ||||||||||||'
						uKRJperpix_loc = np.hstack((configurations[exp]['uKRJ/pix'], expts['C-BASS']['uKRJ/pix'] ))
						freqs_loc = np.hstack((configurations[exp]['freqs'], expts['C-BASS']['freqs'] ))
						bandpass_loc = np.hstack(( configurations[exp]['bandpass'],  expts['C-BASS']['bandpass'] ))
					else:
						uKRJperpix_loc = configurations[exp]['uKRJ/pix']
						freqs_loc = configurations[exp]['freqs']
						stolyarov_args_loc = [ 'Bd' ]
						bandpass_loc = configurations[exp]['bandpass']


					if components == 'cmb-only':
						Cl_res = Cls_dust*0.0
						sqrtAtNAinv_00 = 0.0
						delta_betas = 0.0
					else:

						if calibration_error != 0.0:
							calibration_fixed = False
						else:
							calibration_fixed = True


						if two_dusts:
							Cl_res, sqrtAtNAinv_00, delta_betas = residuals_comp.Cl_res_computation( np.ones(len(freqs_loc)), uKRJperpix_loc, freqs_loc, spsp, calibration_error, Cls_dust*0.0, Cls_dust, Cls_sync, Cls_dxs, int(np.min(Cls_fid['ell'])), int(np.max(Cls_fid['ell'])), ells_input=Cls_fid['ell'], calibration_fixed=calibration_fixed, npatch=npatch, components=components_loc, prior_dust=configurations[exp]['prior_dust'], prior_sync=configurations[exp]['prior_sync'], bandpass_channels=bandpass_loc, everything_output=True, T_second_grey_body=15.72, Bd2=2.70, Bd=1.67, Temp=9.15, Bs=-3.1, correlation_2_dusts=correlation_2_dusts )
							#Cl_res, sqrtAtNAinv_00, delta_betas = residuals_comp.Cl_res_computation( np.ones(len(configurations[exp]['freqs'])), configurations[exp]['uKRJ/pix'], configurations[exp]['freqs'], spsp, calibration_error, Cls_dust*0.0, Cls_dust, Cls_sync, Cls_dxs, int(np.min(Cls_fid['ell'])), int(np.max(Cls_fid['ell'])), ells_input=Cls_fid['ell'], calibration_fixed=calibration_fixed, npatch=npatch, components=components_loc, prior_dust=configurations[exp]['prior_dust'], prior_sync=configurations[exp]['prior_sync'], bandpass_channels=configurations[exp]['bandpass'], everything_output=True, T_second_grey_body=15.72, Bd2=2.70, Bd=1.67, Temp=9.15, Bs=-3.1, correlation_2_dusts=correlation_2_dusts )
						else:
							Cl_res, sqrtAtNAinv_00, delta_betas = residuals_comp.Cl_res_computation( np.ones(len(freqs_loc)), uKRJperpix_loc, freqs_loc, spsp, calibration_error, Cls_dust*0.0, Cls_dust, Cls_sync, Cls_dxs, int(np.min(Cls_fid['ell'])), int(np.max(Cls_fid['ell'])), ells_input=Cls_fid['ell'], calibration_fixed=calibration_fixed, npatch=npatch, components=components_loc, prior_dust=configurations[exp]['prior_dust'], prior_sync=configurations[exp]['prior_sync'], bandpass_channels=bandpass_loc, everything_output=True, Temp=Td, Bd=Bd, Bs=Bs)
							#Cl_res, sqrtAtNAinv_00, delta_betas = residuals_comp.Cl_res_computation( np.ones(len(configurations[exp]['freqs'])), configurations[exp]['uKRJ/pix'], configurations[exp]['freqs'], spsp, calibration_error, Cls_dust*0.0, Cls_dust, Cls_sync, Cls_dxs, int(np.min(Cls_fid['ell'])), int(np.max(Cls_fid['ell'])), ells_input=Cls_fid['ell'], calibration_fixed=calibration_fixed, npatch=npatch, components=components_loc, prior_dust=configurations[exp]['prior_dust'], prior_sync=configurations[exp]['prior_sync'], bandpass_channels=configurations[exp]['bandpass'], everything_output=True, Temp=Td, Bd=Bd, Bs=Bs)

				else:

					print ' ############################################################################## '
					if stolyarov_sync:
						print ' ########### COMP. SEP. WITH STOLYAROV APPROACH FOR DUST + SYNC ############ '
					else:
						print ' ########### COMP. SEP. WITH STOLYAROV APPROACH FOR DUST ONLY ############ '
					print ' ############################################################################### '

					if (components == 'dust+dust') or (components=='sync+dust+dust'):
						two_dusts = True
					else:
						two_dusts = False

					if two_dusts: 
						print '/!\ the possibility of having two dusts in the framework of Stolyarov has not been implemented yet !!'
						exit()
					
					if components == 'cmb-only':
						Cl_res = Cls_dust*0.0
						sqrtAtNAinv_00 = 0.0
						delta_betas = 0.0

					else:

						if 'dust' not in components:
							if stolyarov_sync:
								spsp = spsp_in[np.ix_([0,1,6,7,8,9],[0,1,6,7,8,9])]
							else:
								spsp = spsp_in[np.ix_([0,1,6,7],[0,1,6,7])]
						elif 'sync' not in components:
							spsp = spsp_in[np.ix_([0,1,2,3,4,5],[0,1,2,3,4,5])]
						else:
							spsp = spsp_in

						# evaluating residuals power spectrum and noise in the final CMB map
						if components == 'dust':
							components_loc = ['dust']
						elif components == 'dust+dust':
							components_loc = ['dust']
						elif components == 'sync+dust':
							components_loc = ['sync','dust']
						elif components == 'sync+dust+dust':
							components_loc = ['sync','dust', 'dust']
						elif components == 'cmb-only':
							components_loc = ['']
						elif components == 'sync':
							components_loc = ['sync']
						else:
							print ' /!\  I do not undestand this component: ', components
							exit()

						if components == 'cmb-only':
							
							Cl_res = Cls_dust*0.0
							sqrtAtNAinv_00 = 0.0
							delta_betas = 0.0

						else:

							fskys_masks = np.array( [20.0, 40.0, 60.0, 70.0, 80.0] )

							if os.path.exists(os.path.join(path_to_planck,'mask_files_'+str(common_nside)+'.npy')):
								mask_files = np.load(os.path.join(path_to_planck,'mask_files_'+str(common_nside)+'.npy'))
								if hp.npix2nside(len(mask_files[0]))!= common_nside: mask_files = hp.ud_grade(mask_files, nside_out=common_nside)
							else:
								mask_files = hp.read_map(os.path.join( path_to_planck, 'HFI_Mask_GalPlane-apo2_2048_R2.00.fits'), field=(0,1,2,3,4))
								if hp.npix2nside(len(mask_files[0]))!= common_nside: mask_files = hp.ud_grade(mask_files, nside_out=common_nside)
								np.save(os.path.join(path_to_planck,'mask_files_'+str(common_nside)+'.npy'), mask_files)
							
							ind_fsky = np.argmin(np.abs(fskys_masks - configurations[exp]['fsky']*100.0))
							mask = mask_files[ind_fsky]

							if os.path.exists(os.path.join(path_to_planck,'dust_params_'+str(common_nside)+'.npy')):
								dust_params = np.load(os.path.join(path_to_planck,'dust_params_'+str(common_nside)+'.npy'))
								if hp.npix2nside(len(dust_params[0]))!= common_nside: dust_params = hp.ud_grade(dust_params, nside_out=common_nside)
							else:
								dust_params = hp.read_map( os.path.join( path_to_planck, 'HFI_CompMap_ThermalDustModel_2048_R1.20.fits'), field=(0,1,2, 3, 4, 5, 6,7 ))
								if hp.npix2nside(len(dust_params[0]))!= common_nside: dust_params = hp.ud_grade(dust_params, nside_out=common_nside)
								np.save(os.path.join(path_to_planck,'dust_params_'+str(common_nside)+'.npy'), dust_params)

							if os.path.exists(os.path.join(path_to_planck, 'Beta_s_'+str(common_nside)+'.npy')):
								Beta_s = np.load(os.path.join(path_to_planck,'Beta_s_'+str(common_nside)+'.npy'))
								if hp.npix2nside(len(Beta_s))!= common_nside: Beta_s = hp.ud_grade(Beta_s, nside_out=common_nside)
							else:
								beta_s_tot = hp.read_map( os.path.join(path_to_planck,'COM_CompMap_Lfreqfor-commrul_0256_R1.00.fits'), field=(0,1,2,3) )
								Beta_s = beta_s_tot[2]
								if hp.npix2nside(len(Beta_s))!= common_nside: Beta_s = hp.ud_grade(Beta_s, nside_out=common_nside)
								np.save(os.path.join(path_to_planck,'Beta_s_'+str(common_nside)+'.npy'), Beta_s)

							if os.path.exists(os.path.join(path_to_planck,'cmb_'+str(common_nside)+'.npy')):
								cmb = np.load(os.path.join(path_to_planck,'cmb_'+str(common_nside)+'.npy'))
								if hp.npix2nside(len(cmb[0]))!= common_nside: cmb = hp.ud_grade(cmb, nside_out=common_nside)
							else:
								cmb = hp.read_map(os.path.join(path_to_planck,'COM_CMB_IQU-smica-field-Int_2048_R2.00.fits'), field=(0,1,2))
								if hp.npix2nside(len(cmb[0]))!= common_nside: cmb = hp.ud_grade(cmb, nside_out=common_nside)
								np.save(os.path.join(path_to_planck,'cmb_'+str(common_nside)+'.npy'), cmb)

							if os.path.exists(os.path.join(path_to_planck,'dust_'+str(common_nside)+'.npy')):
								dust = np.load(os.path.join(path_to_planck,'dust_'+str(common_nside)+'.npy'))
								if hp.npix2nside(len(dust[0]))!= common_nside: dust = hp.ud_grade(dust, nside_out=common_nside)
							else:
								dust = hp.read_map(os.path.join(path_to_planck,'COM_CompMap_DustPol-commander_1024_R2.00.fits'), field=(0,1))
								if hp.npix2nside(len(dust[0]))!= common_nside: dust = hp.ud_grade(dust, nside_out=common_nside)
								np.save(os.path.join(path_to_planck,'dust_'+str(common_nside)+'.npy'), dust)
							
							if os.path.exists(os.path.join(path_to_planck,'sync_'+str(common_nside)+'.npy')):
								sync = np.load(os.path.join(path_to_planck,'sync_'+str(common_nside)+'.npy'))
								if hp.npix2nside(len(sync[0]))!= common_nside: sync = hp.ud_grade(sync, nside_out=common_nside)
							else:
								sync = hp.read_map(os.path.join(path_to_planck,'COM_CompMap_SynchrotronPol-commander_0256_R2.00.fits'), field=(0,1))
								if hp.npix2nside(len(sync[0]))!= common_nside: sync = hp.ud_grade(sync, nside_out=common_nside)						
								np.save(os.path.join(path_to_planck,'sync_'+str(common_nside)+'.npy'), sync)
							
							####################################################
							print 'assigning spectral indices'
							# Beta dust
							Bd = dust_params[6]
							err_Bd =  dust_params[7]
							# Temperature dust
							Td = dust_params[4]
							err_Td = dust_params[5]
							# Beta synchrotron
							Bs =  Beta_s
							## computation of delta betas
							delta_Bs =  Bs*1.0 - np.mean( Bs*mask*1.0 )
							delta_Td =  Td*1.0 - np.mean( Td*mask*1.0 )
							delta_Bd =  Bd*1.0 - np.mean( Bd*mask*1.0 )

							#####################################################

							print 'grabing dust and sync maps'
							A353 = res_ext.A_element_computation(353, 1.59, 18.0, -3.1, squeeze=True)
							A30 = res_ext.A_element_computation(30, 1.59, 18.0, -3.1, squeeze=True)
							A150 = res_ext.A_element_computation(150, 1.59, 18.0, -3.1, squeeze=True)
							from_dust_353RJ_150RJ = A150[1]/A353[1]
							from_sync_30RJ_150RJ = A150[4]/A30[4]
							from_cmb_KCMB_to_150_uKRJ = A150[0]*1e6
							Q_dust, U_dust = dust[0], dust[1]
							Q_dust *= from_dust_353RJ_150RJ
							U_dust *= from_dust_353RJ_150RJ
							Q_sync, U_sync = sync[0], sync[1]
							Q_sync *= from_sync_30RJ_150RJ
							U_sync *= from_sync_30RJ_150RJ

							######################################
							# prior from Planck
							prior_Bd = configurations[exp]['prior_dust']*1.0
							prior_Td = 2.0
							prior_Bs = configurations[exp]['prior_sync']*1.0
							###########################################

							ell_max_comp_sep = np.min([np.max(ells[exp]), 4*common_nside])

							if stolyarov_sync:
								if cbass:
									print ' ||||||||||| ADDING C-BASS ||||||||||||'
									uKRJperpix_loc = np.hstack((configurations[exp]['uKRJ/pix'], expts['C-BASS']['uKRJ/pix'] ))
									freqs_loc = np.hstack((configurations[exp]['freqs'], expts['C-BASS']['freqs'] ))
									stolyarov_args_loc = [ 'Bd', 'Bs' ]
									bandpass_loc = np.hstack(( configurations[exp]['bandpass'],  expts['C-BASS']['bandpass'] ))
								elif quijote:
									print ' ||||||||||| ADDING QUIJOTE ||||||||||||'
									uKRJperpix_loc = np.hstack((configurations[exp]['uKRJ/pix'], expts['Quijote']['uKRJ/pix'] ))
									freqs_loc = np.hstack((configurations[exp]['freqs'], expts['Quijote']['freqs'] ))
									stolyarov_args_loc = [ 'Bd', 'Bs' ]
									bandpass_loc = np.hstack(( configurations[exp]['bandpass'],  expts['Quijote']['bandpass'] ))
								else:
									uKRJperpix_loc = configurations[exp]['uKRJ/pix']
									freqs_loc = configurations[exp]['freqs']				
									stolyarov_args_loc = [ 'Bd', 'Bs' ]
									bandpass_loc = configurations[exp]['bandpass']
							else:
								if cbass:
                                                                        print ' ||||||||||| ADDING C-BASS ||||||||||||'
                                                                        uKRJperpix_loc = np.hstack((configurations[exp]['uKRJ/pix'], expts['C-BASS']['uKRJ/pix'] ))
                                                                        freqs_loc = np.hstack((configurations[exp]['freqs'], expts['C-BASS']['freqs'] ))
                                                                        stolyarov_args_loc = [ 'Bd', 'Bs' ]
                                                                        bandpass_loc = np.hstack(( configurations[exp]['bandpass'],  expts['C-BASS']['bandpass'] ))
								else:
									uKRJperpix_loc = configurations[exp]['uKRJ/pix']
									freqs_loc = configurations[exp]['freqs']				
									stolyarov_args_loc = [ 'Bd' ]
									bandpass_loc = configurations[exp]['bandpass']

							Cls_res_loc, sqrtAtNAinv_00, delta_betas, Cls_res_p_loc, Cls_res_m_loc = res_ext.main_Cl_res_computation( uKRJperpix_loc, \
								freqs_loc, spsp, calibration_error*1.0, Cls_dust*0.0, Cls_dust, Cls_sync, Cls_dxs, np.min(ells[exp]), ell_max_comp_sep, \
								delta_Bd=delta_Bd*mask, delta_Bs=delta_Bs*mask, delta_Td=delta_Bd*0.0, \
								Q_dust=Q_dust*mask, U_dust=U_dust*mask, Q_sync=Q_sync*mask, U_sync=U_sync*mask, \
								Bd=1.59, Td=19.6, Bs=-3.1, bandpass_channels=bandpass_loc, \
								prior_Bd=prior_Bd, prior_Td=0.0, prior_Bs=prior_Bs, components=components, stolyarov_args=stolyarov_args_loc,\
								Td_fixed=True, fsky=int( configurations[exp]['fsky']*100.0 ), err_Bd=None, err_Td=None, path_to_planck=path_to_planck,\
								calibration_error=calibration_error)

							# we should extrapolate Cl res to the full extension of ells[exp] !!!! 
							def residuals_ell_dependence( ell, p0, p1 ):
								return p0 * (ell**p1)
							def logl_model( p ):
								A, b = p
								return logl( residuals_ell_dependence( np.arange(np.min(ells[exp]), ell_max_comp_sep)[20:200], A, b ), np.abs(Cls_res_loc)[20:200] )
							
							popt = optimize.fmin( logl_model, x0=[1,-1] )
							
							Cl_res = residuals_ell_dependence( Cls_fid['ell'], popt[0], popt[1] )



							'''
							#np.save('ells', Cls_fid['ell'])
							#np.save('Cl_res', Cl_res)
							#exit()
							ells2 = np.load('ells.npy' )
							Cl_res2 = np.load('Cl_res.npy')
							pl.figure()
							#pl.loglog( np.arange(np.min(ells[exp]), ell_max_comp_sep), np.abs(Cls_res_loc), 'ko', alpha=0.3)
							#pl.loglog( np.arange(np.min(ells[exp]), ell_max_comp_sep)[20:200], np.abs(Cls_res_loc)[20:200], 'rx', alpha=0.3)
							pl.loglog( Cls_fid['ell'], Cl_res, 'k--' )
							pl.loglog( ells2, Cl_res2, 'r:')
							#pl.figure()
							#pl.semilogx(Cl_res2/Cl_res)
							pl.show()
							exit()
							'''

				#############################################################################################


				#if components!= 'cmb-only':
				#	print np.mean( Cl_res )
				#	exit()

				## NOISE POST COMP SEP 
				if components == 'cmb-only':
					foregrounds[exp][components]['uKCMB/pix_postcompsep'] = \
							1.0/np.sqrt( np.sum( 1.0 / np.array(configurations[exp]['uKCMBarcmin'])**2) )
				else:
					foregrounds[exp][components]['uKCMB/pix_postcompsep'] = sqrtAtNAinv_00*1.0

				print ' for exp=',exp,' >>> uK_CMB arcmin after comp. sep.  = ', foregrounds[exp][components]['uKCMB/pix_postcompsep']*pix_size_map_arcmin
				print '  				 while uK_CMB arcmin before comp. sep.  = ',  1.0/np.sqrt( np.sum( 1.0 / np.array(configurations[exp]['uKCMBarcmin'])**2) ) 

				## RESIDUALS ########
				foregrounds[exp][components]['Cl_res'] = Cl_res*(Cls_fid['ell']*(Cls_fid['ell']+1))/(2*np.pi)
				# flattening of residuals at ell <= 10:
				#i10 = np.argmin(np.abs(Cls_fid['ell']-10))
				#foregrounds[exp][components]['Cl_res'][:i10] = foregrounds[exp][components]['Cl_res'][i10]*np.ones(len(foregrounds[exp][components]['Cl_res'][:i10]))
				#foregrounds[exp][components]['Cl_res'] = gaussian_filter1d( foregrounds[exp][components]['Cl_res'], 3.0, mode='nearest' )
				
				# computation of r_eff
				ind0 = np.argmin(np.abs( Cls_fid['ell'] - 20 ))
				ind1 = np.argmin(np.abs( Cls_fid['ell'] - 200 ))
				ind0_ = np.argmin(np.abs( Cls_fid['ell'] - 20 ))
				ind1_ = np.argmin(np.abs( Cls_fid['ell'] - 200 ))
				if components != 'cmb-only':
					dCldp = fc.derivatives_computation(Cls_fid, ['r'], params_fid, information_channels, exe = camb)
					#foregrounds[exp][components]['r_eff'] = np.sum(foregrounds[exp][components]['Cl_res'][ind0:ind1] / Cls_fid['ell'][ind0:ind1] / (Cls_fid['ell'][ind0:ind1]+1.0) * 2.0 * np.pi) / np.sum(Cls_fid['BuBu'][ind0_:ind1_] / params_fid_v[0]['r'] / Cls_fid['ell'][ind0_:ind1_] / (Cls_fid['ell'][ind0_:ind1_]+1.0) * 2.0 * np.pi) 
					foregrounds[exp][components]['r_eff'] = np.sum(foregrounds[exp][components]['Cl_res'][ind0:ind1] / Cls_fid['ell'][ind0:ind1] / (Cls_fid['ell'][ind0:ind1]+1.0) * 2.0 * np.pi) / np.sum(dCldp['r']['BuBu'][ind0_:ind1_] / Cls_fid['ell'][ind0_:ind1_] / (Cls_fid['ell'][ind0_:ind1_]+1.0) * 2.0 * np.pi) 
				else:
					foregrounds[exp][components]['r_eff'] = 0.0

				print '  				and the effective level of residuals is reff = ', foregrounds[exp][components]['r_eff']

				'''
				if components != 'cmb-only': 
					pl.figure()
					pl.loglog( Cls_fid['ell'][ind0:ind1], foregrounds[exp][components]['Cl_res'][ind0:ind1], 'k' )
					pl.show()
					print  foregrounds[exp][components]['Cl_res'][ind0:ind1]
					print foregrounds[exp][components]['r_eff'] 
					exit()
				'''

				#pl.figure()
				#pl.loglog( np.arange(2, len(Cls_dust)+2) , Cls_dust )
				#pl.loglog( np.arange(2, len(Cls_dust)+2) , Cls_sync )
				#pl.loglog( np.arange(2, len(Cls_dust)+2) , Cls_dxs )
				#if components != 'cmb-only': 
				#	pl.loglog(Cls_fid['ell'], Cl_res, label=exp+' for components='+str(components))
				#pl.title( exp+' and components = '+str(components))
				#pl.show()

				## CONSTRAINTS ON SPECTRAL INDICES
				if 'dust' not in components:
					foregrounds[exp][components]['delta_beta_sync'] = delta_betas
				elif 'sync' not in components:
					if two_dusts:
						foregrounds[exp][components]['delta_beta_dust_1'] = delta_betas[0,0]
						foregrounds[exp][components]['delta_beta_dust_2'] = delta_betas[1,1]
					else:
						foregrounds[exp][components]['delta_beta_dust'] = delta_betas
				else:
					foregrounds[exp][components]['delta_beta_dust'] = delta_betas[0,0]
					foregrounds[exp][components]['delta_beta_sync'] = delta_betas[1,1]

	############################################################################################################################################################################################################
	############################################################################################################################################################################################################

	#pl.legend()
	#pl.show()
	#exit()

	##############
	## COMPUTE Nls for each experimental setup (NlTT, NlEE, NlBB, Nldd)
	Nl={}

	ind = -1
	for exp1 in experiments:
		ind += 1
		for exp2 in experiments[ind:]:
			if exp1 == exp2 :
				# experiment alone
				exp = exp1
			else:
				# combination of two experiments
				exp = exp1+' x '+exp2

			if cross_only and exp1==exp2: continue

			Nl[exp] = {}

			for components in components_v:

				Nl[exp][components] = {}

				if components == 'cmb-only':
					# calculate quadratic noise combo including beams, then weight by 
					# ratio of white noise levels pre- and post-comp.-sep.
					Nl[exp][components]['TT'] = np.zeros(configurations[exp]['ell_max'] - ell_min_camb + 1)
					w_inv = ( np.array(configurations[exp]['uKCMBarcmin'][:] ) * arcmin_to_radian) ** 2
					for ell in range(ell_min_camb, configurations[exp]['ell_max'] + 1):
						beam_l = np.zeros(len(w_inv))
						for k in range(len(beam_l)):
							if ((configurations[exp]['alpha_knee'][k]!=0.0) and (configurations[exp]['ell_knee'][k]!=0.0)):
								factor = ( 1.0 + pow(configurations[exp]['ell_knee'][k]*1.0/ell, configurations[exp]['alpha_knee'][k]) )
							else:
								factor = 1.0
							beam_l[k] = factor*np.exp((np.array(configurations[exp]['FWHM'][k]) * arcmin_to_radian / np.sqrt(8.0*np.log(2.0))) ** 2 * (ell * (ell + 1.0)))
						Nl[exp][components]['TT'][ell - ell_min_camb] = ( (ell * (ell + 1.0) / (2.0 * np.pi)) / np.sum( 1.0 / w_inv / beam_l) )/2.0
						#if ell_knee != 0.0 :
						#	if ell==ell_min_camb: print 'ell_knee is different from 0 -> we inject 1/ell noise !'
						#	Nl[exp][components]['TT'][ell - ell_min_camb] *= ( 1.0 + pow(ell_knee*1.0/ell, alpha_knee) )

					Nl[exp][components]['EE'] = Nl[exp][components]['TT'] * 2.0
					Nl[exp][components]['BB'] = Nl[exp][components]['TT'] * 2.0
					foregrounds[exp][components]['delta'] = 1.0
					foregrounds[exp][components]['sigma_CMB'] = 1.0/np.sqrt(np.sum( 1.0 / w_inv ))/arcmin_to_radian

				else:
					# noise after component separation
					w_inv_post = (foregrounds[exp][components]['uKCMB/pix_postcompsep'] * pix_size_map_arcmin * arcmin_to_radian) ** 2

					if resolution:
						print ' resolution effect on noise ... '
						# take into account the resolution of the various frequency channel in the comp sep process 
						Nl[exp][components]['TT_post_comp_sep'] = Nl[exp]['cmb-only']['TT'] * 0.0
						sensitivity_uK_per_chan = configurations[exp]['uKRJ/pix']*1.0
						nch = len( configurations[exp]['freqs'] )
						if components == 'dust':
							components_loc = ['dust']
						elif components == 'dust+dust':
							components_loc = ['dust']
						elif components == 'sync+dust':
							components_loc = ['sync','dust']
						elif components == 'sync+dust+dust':
							components_loc = ['sync','dust', 'dust']
						elif components == 'cmb-only':
							components_loc = ['']
						elif components == 'sync':
							components_loc = ['sync']
						else:
							print ' /!\  I do not undestand this component: ', components
							exit()
						for ell in range(ell_min_camb, configurations[exp]['ell_max'] + 1):
							# Computation of the inverse covariance, in 1/uK**2 per frequency channel
							Ninv = np.zeros( (2*nch, 2*nch) )
							for k in range( 2*nch ):
								if ((configurations[exp]['alpha_knee'][ int(k/2) ]!=0.0) and (configurations[exp]['ell_knee'][ int(k/2) ]!=0.0)):
									factor = ( 1.0 + pow(configurations[exp]['ell_knee'][ int(k/2) ]*1.0/ell, configurations[exp]['alpha_knee'][ int(k/2) ]) )
								else:
									factor = 1.0
								# this Ninv, so the 1/ell factor should be at the denominator ... 
								Ninv[ k, k ]  = (1.0/factor)*((1.0/sensitivity_uK_per_chan[ int(k/2) ])**2) * np.exp(  - ell*(ell+1.0)*( configurations[exp]['FWHM'][ int(k/2) ]*arcmin_to_radian )**2/(8*np.log(2)) )
							# A is a 6 x Nch matrix. 6 = 2x(CMB, Dust, Synchrotron)
							if stolyarov:
								stolyarov_loc = ['Bd']
							elif stolyarov_sync:
								stolyarov_loc = ['Bd', 'Bs']
							else:
								stolyarov_loc = ['']

							A, dAdBd, dAdTd, dAdBs = res_ext.A_and_dAdB_matrix_computation( configurations[exp]['freqs'], 1.59, 19.6, -3.1,\
							 		components=components_loc, stolyarov_args=stolyarov_loc, Td_fixed=True, bandpass_channels=None  )
							# A**T N**-1 ** A
							AtNA =  A.T.dot(Ninv).dot(A)
							# A**T N**-1 ** A
							try:
								AtNAinv = np.linalg.inv( AtNA )
							except np.linalg.linalg.LinAlgError:
								AtNAinv = 1e6*np.ones( AtNA.shape )  

							Nl[exp][components]['TT_post_comp_sep'][ell - ell_min_camb] = (ell*(ell+1.0)/(2*np.pi))*AtNAinv[0,0] * (pix_size_map_arcmin * arcmin_to_radian)**2/2.0
							#if ell_knee != 0.0 :
                                                	#	if ell==ell_min_camb: print 'ell_knee is different from 0 -> we inject 1/ell noise !'
                                                	#	Nl[exp][components]['TT_post_comp_sep'][ell - ell_min_camb] *= ( 1.0 + pow(ell_knee*1.0/ell, alpha_knee) )
							del AtNA
					else:
						# in this case, the noise after comp sep is taken as a rescaling of the simple quadratic noise
						# rescaling the cmb-only white noise curve and adding the foregrounds residuals 
						Nl[exp][components]['TT_post_comp_sep'] = Nl[exp]['cmb-only']['TT'] * w_inv_post * np.sum(1.0 / w_inv ) * 1.0

					
					# taking the post comp sep noise on E/B as twice the noise on T
					Nl[exp][components]['EE_post_comp_sep'] = Nl[exp][components]['TT_post_comp_sep'] * 2.0 
					Nl[exp][components]['BB_post_comp_sep'] = Nl[exp][components]['TT_post_comp_sep'] * 2.0 
		
					ind0=np.argmin(np.abs(Cls_fid['ell'] - np.min(ells[exp])))
					ind1=np.argmin(np.abs(Cls_fid['ell'] - np.max(ells[exp])-1))
					#pl.figure()
					#pl.loglog(Cls_fid['ell'], foregrounds[exp][components]['Cl_res'], 'r')
					#pl.loglog(Cls_fid['ell'],Cls_fid['BB'],'b')
					#pl.loglog(ells[exp], Nl[exp][components]['TT_post_comp_sep'], 'k--')
					#pl.loglog(ells[exp], Nl[exp][components]['TT_post_comp_sep2'], 'k:')
					#pl.show()

					Nl[exp][components]['TT_post_comp_sep'] +=  foregrounds[exp][components]['Cl_res'][ind0:ind1]
					Nl[exp][components]['EE_post_comp_sep'] +=  foregrounds[exp][components]['Cl_res'][ind0:ind1]
					Nl[exp][components]['BB_post_comp_sep'] +=  foregrounds[exp][components]['Cl_res'][ind0:ind1] 

					# computation of the noise degradation
					foregrounds[exp][components]['delta'] = w_inv_post*(np.sum(1.0 / w_inv))
					foregrounds[exp][components]['sigma_CMB'] = foregrounds[exp][components]['uKCMB/pix_postcompsep']*pix_size_map_arcmin
					#if exp == 'PIXIE':
					#	print 'component=', components
					#	print 'delta = ', foregrounds[exp][components]['delta'] 
					#	print 'reff = ', foregrounds[exp][components]['r_eff']
					#	exit()

	if comp_sep_only:
		return foregrounds

	#####################################################
	### FIGURE RESIDUALS VS B-MODES AND NOISE
	if fgs_scatter:

		# need different options for different experiment groups
		print ' DOING THE OTHER FOREGROUNDS FIGURE '
		if ('Simons Array' in experiments):
			stage = 'III'
		else:
			stage = 'IV'
		print 'plotting stage {:} experiments'.format(stage)
		
		# loop through non-CMB-only components and plot
		fg_comps = [comp for comp in components_v if comp not in ['cmb-only']]
		n_fg = len(fg_comps)
		fig, axes = pl.subplots(1, n_fg, sharey=True, figsize=(n_fg * 4,4), \
								facecolor='w', edgecolor='k')

		fg_ind = 0
		for fg_comp in fg_comps:
			 
			# keep track of plotted points and labels for legend, 
			# and experiments and locations plotted
			exp_ind = 0
			points = []
			exp_labs = []
			exp_locs = []
			loc_counts = dict.fromkeys(loc_markers.keys(), 0)
			for exp1 in experiments:
				
				# deal with experimental combinations
				for exp2 in experiments[exp_ind:]:
					if exp1 == exp2 :
						exp = exp1
						exp_lab = exp
						loc = configurations[exp]['loc']
					else:
						if (stage == 'III'):
							if  (exp1 != 'Planck') and (exp2 != 'Planck'):
								continue
								#exp_lab = exp1+' x '+exp2
							elif (exp1 == 'Planck'):
								exp_lab = exp1+' x '+exp2
							else:
								exp_lab = exp2+' x '+exp1
						else:
							exp_lab = ' x '.join(sorted((exp1, exp2), key=lambda t: t[0]))
						exp = exp1+' x '+exp2
						loc = locs['cross']

					print exp_lab, foregrounds[exp][fg_comp]['r_eff'], \
												   foregrounds[exp][fg_comp]['delta']
					if stolyarov or stolyarov_sync:
						point, = axes[fg_ind].loglog(foregrounds[exp][fg_comp]['r_eff'], \
												   foregrounds[exp][fg_comp]['delta'], \
												   linestyle = 'None', \
												   marker = loc_markers[loc], \
												   markerfacecolor = 'None', \
												   markeredgecolor = cb_colors[loc_counts[loc]], \
												   markeredgewidth = 2, markersize = 8)
					else:
						point, = axes[fg_ind].semilogx(foregrounds[exp][fg_comp]['r_eff'], \
												   foregrounds[exp][fg_comp]['delta'], \
												   linestyle = 'None', \
												   marker = loc_markers[loc], \
												   markerfacecolor = 'None', \
												   markeredgecolor = cb_colors[loc_counts[loc]], \
												   markeredgewidth = 2, markersize = 8)
					
					if exp1==exp2=='SPIDER':
						print foregrounds[exp][fg_comp]['r_eff'], \
								foregrounds[exp][fg_comp]['delta']


					points.append(point)
					exp_labs.append(exp_lab)
					exp_locs.append(loc)
					loc_counts[loc] +=1
				exp_ind += 1

			# label plot and set limits
			if stolyarov:
				if stage == 'III':
					ymax = 100.0
					xmin = 10.0**(-7.75)
					xmax = 10.0**(-2.25)
					r_ticks = [ 1e-7, 1e-6, 1e-5, 1e-4, 1e-3 ]
				else:
					ymax = 33.0
					xmin = 10.0**(-8.75)
					xmax = 10.0**(-4.25)
					r_ticks = [ 1e-8, 1e-7, 1e-6, 1e-5 ]
			elif stolyarov_sync:
				if stage == 'III':
					ymax = 100.0
					xmin = 10.0**(-6.75)
					xmax = 10.0**(-1.25)
					r_ticks = [ 1e-6, 1e-5, 1e-4, 1e-3, 1e-2 ]
				else:
					ymax = 100.0
					xmin = 10.0**(-9.75)
					xmax = 10.0**(-4.25)
					r_ticks = [ 1e-9,1e-8, 1e-7, 1e-6, 1e-5 ]
			else:
				if stage  == 'III':
					ymax = 5.0
					xmin = 10.0**(-6.75)
					xmax = 10.0**(-2.25)
					r_ticks = [ 1e-6, 1e-5, 1e-4, 1e-3 ]
				else:
					ymax = 3.0
					xmin = 10.0**(-8.25)
					xmax = 10.0**(-3.75)
					r_ticks = [ 1e-8, 1e-7, 1e-6, 1e-5, 1e-4 ]
			axes[fg_ind].text(0.98, 0.98, fg_comp, fontsize = 16, \
							  transform = axes[fg_ind].transAxes, \
							  horizontalalignment = 'right', \
							  verticalalignment = 'top')
			axes[fg_ind].set_ylim([0.8, ymax])
			axes[fg_ind].set_xlim([xmin, xmax])
			axes[fg_ind].set_xticks(r_ticks)

			fg_ind += 1
		
		# set overall axis labels
		axes[1].set_xlabel(r'foreground residuals $r_{\rm eff}$', fontsize = 22)
		axes[0].set_ylabel(r'noise degradation $\Delta$', fontsize = 22)
		if stolyarov or stolyarov_sync:
			figtitle = '$\mathbf{A}$-expansion approach, '
		else:
			figtitle = '$n_p$ approach, '
		if stage == 'III':
			figtitle += 'pre-2020'
		else:
			figtitle += 'post-2020'
		if cbass:
			figtitle += ' + C-BASS'
		if quijote:
			figtitle += ' + Quijote'
		axes[1].set_title(figtitle + '\n', fontsize=24)

		# create legend: lots of faff to ensure it's not cropped by savefig
		# sort legend to make readable
		#exp_labs, points = zip(*sorted(zip(exp_labs, points), key=lambda t: t[0]))
		exp_locs, exp_labs, points = zip(*sorted(zip(exp_locs, exp_labs, points), key=lambda t: (t[0], t[1])))
		lgd = pl.legend(points, exp_labs, loc = 'center', fontsize = 10, \
						bbox_to_anchor = (1.4, 0.5), numpoints = 1)
		fr = lgd.get_frame()
		fr.set_lw(1.5)
		
		# remove whitespace and save figure
		fig.subplots_adjust(wspace = 0)
		figfile = '../article/component_sep_performance_s' + stage
		if stolyarov:
			figfile += '_stol_d'
		if stolyarov_sync:
			figfile += '_stol_ds'
		if cbass:
			figfile += '_cbass'
		if quijote:
			figfile += '_quijote'
		pl.savefig(figfile + '.pdf', bbox_inches='tight', bbox_extra_artists=(lgd,))
		pl.show()
		exit()
		
		'''
		pl.figure(num=None, figsize=(20,6), facecolor='w', edgecolor='k')
		ind = 0
		for exp in experiments:
			print exp, np.min(ells[exp]), np.max(ells[exp])
			ind += 1
			ind12 = 0
			if ind > 1:
				ax = pl.subplot(1, len(experiments), ind, sharey=ax )
				pl.setp( ax.get_yticklabels(), visible=False)
			else:
				ax = pl.subplot(1, len(experiments), ind)
			colors = ['DarkOrange', 'DarkBlue', 'DarkGray']
			pl.loglog(Cls_fid['ell'], Cls_fid['BlBl'], linewidth=3.0, alpha=0.7, color='DarkGray' )
			pl.loglog(Cls_fid['ell'], Cls_fid['BuBu']*0.1/params_fid_v[0]['r'], linewidth=2.0, alpha=0.7, color='Black' )
			pl.loglog(Cls_fid['ell'], Cls_fid['BuBu']*0.01/params_fid_v[0]['r'], linewidth=2.0, alpha=0.7, color='Black' )
			pl.loglog(Cls_fid['ell'], Cls_fid['BuBu']*0.001/params_fid_v[0]['r'], linewidth=2.0, alpha=0.7, color='Black' )
			for components in components_v:
				if components == 'cmb-only':
					pl.loglog(ells[exp][components][configurations[exp]['ell_min']-ell_min_camb:configurations[exp]['ell_max']-ell_min_camb], Nl[exp][components]['BB'][configurations[exp]['ell_min']-ell_min_camb:configurations[exp]['ell_max']-ell_min_camb], color='k', linestyle=':', linewidth=3.0 )
				else:
					pl.loglog(ells[exp][configurations[exp]['ell_min']-ell_min_camb:configurations[exp]['ell_max']-ell_min_camb], foregrounds[exp][components]['Cl_res'][configurations[exp]['ell_min']-ell_min_camb:configurations[exp]['ell_max']-ell_min_camb], color=colors[ind12], linewidth=3.0 )
					pl.loglog(ells[exp][configurations[exp]['ell_min']-ell_min_camb:configurations[exp]['ell_max']-ell_min_camb], Nl[exp][components]['BB_post_comp_sep'][configurations[exp]['ell_min']-ell_min_camb:configurations[exp]['ell_max']-ell_min_camb], color=colors[ind12], linestyle='--', linewidth=3.0 )
				ind12 += 1
			pl.title( exp , fontsize=24)
			pl.xlim([10, 2000])
			pl.ylim([1e-7, 2e-1])
			#ax = pl.gca()
			if ind == 1: pl.ylabel(r'$C_\ell^{BB}\ [\mu K^2]$', fontsize=24)
			pl.xlabel('$\ell$', fontsize=24)
			for tick in ax.xaxis.get_major_ticks():
				tick.label.set_fontsize(14)
			for tick in ax.yaxis.get_major_ticks():
				tick.label.set_fontsize(14)
		pl.tight_layout()
		pl.savefig('../article/Power_Spectrum_Residuals_each_instrument_vs_B_modes_and_Nl.pdf')
		pl.show()
		exit()
		'''
	if fgs_power_spectrum:

		# need different options for different experiment groups
		print ' DOING THE FOREGROUNDS FIGURE '
		if ('Simons Array' in experiments):
			stage = 'III'
		else:
			stage = 'IV'
		print 'plotting stage {:} experiments'.format(stage)
        
		colors = [ 'OrangeRed', 'Orange', 'DarkRed' ]
		n_exp = len(experiments)
		n_combo = n_exp * (n_exp + 1) / 2
		fig, axes = pl.subplots(1, n_combo, sharey=True, figsize=(n_combo*4,4), \
								facecolor='w', edgecolor='k')
		ind1 = 1
		for exp1 in experiments:
			ind2 = 1
			for exp2 in experiments:
				
				if exp1 == exp2 :
					exp = exp1
					sp_ind = ind1-1
				else:
					exp = exp1+' x '+exp2
					sp_ind = ind1*(2*n_exp-ind1+1)/2+ind2-ind1-1
				
				# do not re-plot cross correlations
				if ind2 >= ind1:
					i_min = configurations[exp]['ell_min'] - ell_min_camb
					i_max = configurations[exp]['ell_max'] - ell_min_camb + 1
					axes[sp_ind].loglog(Cls_fid['ell'], Cls_fid['BlBl'], \
										linewidth=3.0, alpha=0.7, color=cb_colors[3], \
										linestyle = '-')
					axes[sp_ind].loglog(Cls_fid['ell'], \
										Cls_fid['BuBu']*0.1/params_fid_v[0]['r'], \
										linewidth=2.0, alpha=0.7, color='k', linestyle = '-')
					axes[sp_ind].loglog(Cls_fid['ell'], \
										Cls_fid['BuBu']*0.01/params_fid_v[0]['r'], \
										linewidth=2.0, alpha=0.7, color='k', linestyle = '-')
					axes[sp_ind].loglog(Cls_fid['ell'], \
										Cls_fid['BuBu']*0.001/params_fid_v[0]['r'], \
										linewidth=2.0, alpha=0.7, color='k', linestyle = '-')
					ind12 = 0
					for components in components_v:
						if components == 'cmb-only':
							axes[sp_ind].loglog(ells[exp][i_min:i_max], \
								Nl[exp][components]['BB'][i_min:i_max], \
								color='k', linewidth=3.0, linestyle='--', dashes = [10, 2])
						else:
							axes[sp_ind].loglog(ells[exp][i_min:i_max], \
								foregrounds[exp][components]['Cl_res'][i_min:i_max], \
								color=colors[ind12], linewidth=3.0, alpha=0.8, \
								linestyle='--', dashes = [4, 2])
							axes[sp_ind].loglog(ells[exp][i_min:i_max], \
								Nl[exp][components]['BB_post_comp_sep'][i_min:i_max], \
								color=colors[ind12], linewidth=3.0, alpha=0.8, \
								linestyle='--', dashes = [10, 2])
							ind12 += 1
					if len(exp) > 20:
						axes[sp_ind].set_title(exp.replace(' x ', ' x\n'), fontsize=22 )
					else:
						axes[sp_ind].set_title( exp, fontsize=22 )
					axes[sp_ind].set_xlim([2, 2000])
					axes[sp_ind].set_ylim([1e-8, 10])
					axes[sp_ind].tick_params(axis='both', which='major', labelsize=16)
					axes[sp_ind].tick_params(axis='both', which='minor', labelsize=16)
					axes[sp_ind].grid(True)
				ind2 +=1
			ind1 += 1
		axes[1].set_xlabel(r'$\ell$', fontsize = 22)
		axes[0].set_ylabel(r'$\ell(\ell+1) C_\ell^{BB} / 2\pi \ [\mu {{\rm K}}^2]$', \
						   fontsize = 22)
		fig.subplots_adjust(wspace = 0)
		if stage == 'III':
			pl.savefig('../article/Power_Spectrum_Residuals_vs_B_modes_and_Nl_SIII.pdf', \
					   bbox_inches='tight')
		else:
			pl.savefig('../article/Power_Spectrum_Residuals_vs_B_modes_and_Nl_SIV.pdf', \
					   bbox_inches='tight')
		pl.show()
		exit()

		'''
		print ' DOING THE FOREGROUNDS FIGURE '
		pl.figure(num=None, figsize=(12,12), facecolor='w', edgecolor='k')
		ind1 = 0
		for exp1 in experiments:
			ind2 = 0
			for exp2 in experiments:
				
				if exp1 == exp2 :
					exp = exp1
				else:
					exp = exp1+' x '+exp2
				
				# do not plot the lower-left panel
				ind_subplot = ind1*len(experiments)+ind2+1
				if ind2 >= ind1:	
					ax = pl.subplot(2, len(experiments), ind_subplot)#, sharey=ax, sharex=ax )
					pl.title( exp, fontsize=28 )
					colors = ['DarkOrange', 'DarkBlue', 'DarkGray']
					i_min = configurations[exp]['ell_min'] - ell_min_camb
					i_max = configurations[exp]['ell_max'] - ell_min_camb + 1

					ind12 = 0
					pl.loglog(Cls_fid['ell'], Cls_fid['BlBl'], linewidth=3.0, alpha=0.7, color='DarkGray' )
					pl.loglog(Cls_fid['ell'], Cls_fid['BuBu']*0.1/params_fid_v[0]['r'], linewidth=2.0, alpha=0.7, color='Black' )
					pl.loglog(Cls_fid['ell'], Cls_fid['BuBu']*0.01/params_fid_v[0]['r'], linewidth=2.0, alpha=0.7, color='Black' )
					pl.loglog(Cls_fid['ell'], Cls_fid['BuBu']*0.001/params_fid_v[0]['r'], linewidth=2.0, alpha=0.7, color='Black' )
					for components in components_v:
						if components == 'cmb-only':
							pl.loglog(ells[exp][configurations[exp]['ell_min']-ell_min_camb:configurations[exp]['ell_max']-ell_min_camb], \
								Nl[exp][components]['BB'][configurations[exp]['ell_min']-ell_min_camb:configurations[exp]['ell_max']-ell_min_camb], \
								color='k', linestyle=':', linewidth=3.0 )
						else:
							pl.loglog(ells[exp][configurations[exp]['ell_min']-ell_min_camb:configurations[exp]['ell_max']-ell_min_camb], \
								4*foregrounds[exp][components]['Cl_res'][configurations[exp]['ell_min']-ell_min_camb:configurations[exp]['ell_max']-ell_min_camb], \
								color=colors[ind12], linewidth=3.0, alpha=0.8 )
							pl.loglog(ells[exp][configurations[exp]['ell_min']-ell_min_camb:configurations[exp]['ell_max']-ell_min_camb], \
								Nl[exp][components]['BB_post_comp_sep'][configurations[exp]['ell_min']-ell_min_camb:configurations[exp]['ell_max']-ell_min_camb], \
								color=colors[ind12], linestyle='--', linewidth=3.0, alpha=0.8 )
							ind12 += 1
					pl.xlim([2, 2000])
					pl.ylim([1e-8, 10])
					#ax = pl.gca()
					if ind_subplot!=2: 
						pl.ylabel(r'$\ell(\ell + 1 ) C_\ell^{BB} / 2\pi \ [\mu K^2]$', fontsize=28)
						pl.xlabel('$\ell$', fontsize=28)

					for tick in ax.xaxis.get_major_ticks():
						tick.label.set_fontsize(18)
					for tick in ax.yaxis.get_major_ticks():
						tick.label.set_fontsize(18)
					ax = pl.gca()
					ax.grid(True)
				ind2 +=1
			ind1 += 1
		pl.savefig('../article/Power_Spectrum_Residuals_each_instrument_vs_B_modes_and_Nl_v08142015_sIV.pdf')
		pl.show()
		exit()
		'''

	#####################################################
	## set up CIB correlation data for use in LSS delensing
    # read in CIB correlation and linearly extrapolate to overall ell_max
	f_l_cor_cib_data = np.load(os.path.join('/global/homes/j/josquin/FORECAST/self_consistent_forecast/codes/','corr545.pkl'))
	f_l_cor_cib_data[0] = np.append(f_l_cor_cib_data[0], ell_max_abs)
	f_l_cor_cib_data[1] = np.append(f_l_cor_cib_data[1], \
									f_l_cor_cib_data[1][-1] * 2.0 - \
									f_l_cor_cib_data[1][-2])

	# fit low-ell portion of correlation with polynomial
	fit_order = 6
	fit_range = [52, 400]
	fit_l = np.arange(fit_range[0], fit_range[1])
	f_l_cor_cib_fit_par = np.polyfit(f_l_cor_cib_data[0][fit_l], \
									 f_l_cor_cib_data[1][fit_l], \
									 fit_order)
	f_l_cor_cib_fit = np.polyval(f_l_cor_cib_fit_par, f_l_cor_cib_data[0])

	# extrapolate by concatenating low-ell fit with measurements
	f_l_cor_cib = np.zeros(ell_max_abs - 1)
	f_l_cor_cib[:] = f_l_cor_cib_fit[ell_min_camb:]
	f_l_cor_cib[fit_range[0]-ell_min_camb:] = f_l_cor_cib_data[1][fit_range[0]:]

	# optionally zero out range of ells (e.g., Blake says to only use in range
	# 60 <= ell <= 1000)
	zero_range = [60, 2500]
	f_l_cor_cib[0:zero_range[0]-ell_min_camb] = 0.0
	f_l_cor_cib[zero_range[1]-ell_min_camb:] = 0.0
		
	#####################################################
	## set up LSS correlation data for use in perfect LSS delensing
	# correlation coefficient is sqrt(C_l^phiphi,lo / C_l^phiphi)
	if delensing_z_max > 0.0:	
		f_l_cor_lss = np.sqrt(Cls_fid['ddzmax'] / Cls_fid['dd'])

	# plot, if you're into that kind of thing
	if False:
		line_data, = pl.semilogx(f_l_cor_cib_data[0], f_l_cor_cib_data[1])
		line_fit, = pl.semilogx(f_l_cor_cib_data[0], f_l_cor_cib_fit)
		line_out, = pl.semilogx(f_l_cor_cib_data[0][ell_min_camb:], f_l_cor_cib)
		line_perf, = pl.semilogx(Cls_fid['ell'], f_l_cor_lss)
		pl.ylim([0, 1])
		pl.ylabel('$ f_\ell^{\\rm cor}$', fontsize=14)
		pl.xlabel('$\ell$', fontsize=14)
		pl.legend([line_data, line_fit, line_out, line_perf], \
				  ['CIB input', 'CIB low-$\ell$ fit', \
				   'CIB output', 'perfect'], loc=3)
		pl.show()

	# if you ask for no delensing but don't have CMBxCMB
	if ('' in delensing_option_v) and ('CMBxCMB' not in delensing_option_v):
		delensing_option_v_loc = copy.copy( delensing_option_v )
		delensing_option_v_loc.append('CMBxCMB')
	else:
		delensing_option_v_loc = copy.copy(delensing_option_v)
	Nl_in = copy.copy(Nl)

	# delensing settings and useful arrays
	converge = 0.01
	Dl_conv = 2.0 * np.pi / Cls_fid['ell'] / (Cls_fid['ell'] + 1.0)
	pp_dd_conv = Cls_fid['ell'] * (Cls_fid['ell'] + 1.0)

	#####################################################
    ## use command-line version of the delensing code
	if delens_command_line:

		print ' ........... COMMAND LINE DELENSING ............ '

		# EDISON ONLY: point to the correct delens_est code but save in cwd ($SCRATCH)
		delexe = '/global/homes/s/sfeeney/Software_Packages/cmb_pol_forecast/delens_est'

		# optionally generate a random string identifier to allow concurrent runs
                prefix = 'temp_delens'
                if mpi_safe:
                    prefix += '_' + ''.join(random.choice(string.ascii_uppercase + \
			                                  string.digits) for _ in range(10))

		# output to file for passing to delensing code. define array from ell_min_camb
		output = np.column_stack((f_l_cor_cib_data[0][ell_min_camb:].flatten(), \
		                          f_l_cor_cib.flatten()))
		np.savetxt(prefix + '_f_l_cor_planck_545.dat', output, \
		           fmt = ' %5.1i   %12.5e')

		# output to file for passing to delensing code. define array from ell_min_camb
		output = np.column_stack((f_l_cor_cib_data[0][ell_min_camb:].flatten(), \
								  f_l_cor_lss.flatten()))
		np.savetxt(prefix + '_f_l_cor_perfect.dat', output, \
		           fmt = ' %5.1i   %12.5e')

		# using command-line delensing code which needs fiducial C_ls written to file.
		# could pickle random string in with C_ls in order to pass them directly; for
		# now, resave them
		n_ell_camb = len(Cls_fid['TT_tot'])
		output = np.column_stack((ell_min_camb + np.arange(n_ell_camb).flatten(), \
		                          Cls_fid['TT_tot'].flatten(), \
		                          Cls_fid['EE_tot'].flatten(), \
		                          Cls_fid['BB_tot'].flatten(), \
		                          Cls_fid['TE_tot'].flatten()))
		np.savetxt(prefix + '_fidCls_lensedtotCls.dat', output, \
		           fmt = ' %5.1i   %12.5e   %12.5e   %12.5e   %12.5e')
		output = np.column_stack((ell_min_camb + np.arange(n_ell_camb).flatten(), \
		                          Cls_fid['TuTu'].flatten(), \
		                          Cls_fid['EuEu'].flatten(), \
		                          Cls_fid['BuBu'].flatten(), \
		                          Cls_fid['TuEu'].flatten(), \
		                          Cls_fid['dd'].flatten(), \
		                          Cls_fid['Tud'].flatten(), \
		                          np.zeros(n_ell_camb).flatten()))
		np.savetxt(prefix + '_fidCls_lenspotentialCls.dat', output, \
		           fmt = ' %5.1i   %12.5e   %12.5e   %12.5e   %12.5e' + \
                                 '   %12.5e   %12.5e   %12.5e')

		# determine delensing performance
		ind = -1
		for exp1 in experiments:
			ind += 1
			for exp2 in experiments[ind:]:
				if exp1 == exp2 :
					# experiment alone
					exp = exp1
					if cross_only: continue
				else:
					# combination of two experiments
					exp = exp1+' x '+exp2
				
				i_min = configurations[exp]['ell_min'] - ell_min_camb
				i_max = configurations[exp]['ell_max'] - ell_min_camb + 1
				ell_count = configurations[exp]['ell_max'] - ell_min_camb + 1

				for components in components_v:

					if components == 'cmb-only':

						## WITHOUT FOREGROUNDS

						# define a bunch of post-delensing arrays and their ranges. the delensing 
						# code outputs C_ls in the range [ell_min: ell_max + 1]; we want D_ls in the
						# range [ell_min_camb: ell_max + 1]
						Nl[exp][components]['dd'] = np.zeros(ell_count)
						Nl[exp][components]['BB_delens'] = np.zeros(ell_count)
						Nl[exp][components]['dd_CIB'] = np.zeros(ell_count)
						Nl[exp][components]['BB_CIB_delens'] = np.zeros(ell_count)
						Nl[exp][components]['dd_lss'] = np.zeros(ell_count)
						Nl[exp][components]['BB_lss_delens'] = np.zeros(ell_count)
					
				    	        # write BB noise to file to pass to command-line delensing F90 code
						exp_prefix = prefix + '_' + str(exp).replace(' ', '_')
						output = np.column_stack((ells[exp].flatten(), Nl_in[exp][components]['BB'].flatten()))
						np.savetxt(exp_prefix + '_n_l_ee_bb.dat', output, fmt = ' %5.1i   %12.5e')
					    
						# call iterative CMB EB delensing code using raw noise
						if 'CMBxCMB' in delensing_option_v_loc:
							com = [delexe, '-f_sky', str(configurations[exp]['fsky']), \
								       '-l_min', str(configurations[exp]['ell_min']), \
								       '-l_max', str(configurations[exp]['ell_max']), \
								       '-lensed_path', prefix + '_fidCls_lensedtotCls.dat', \
								       '-unlensed_path', prefix + '_fidCls_lenspotentialCls.dat', \
								       '-noise_path', exp_prefix + '_n_l_ee_bb.dat', \
								       '-prefix', exp_prefix]
                                                        try:
                                                                sp.check_call(com)
                                                        except:
                                                                print('external_program failed for CMBxCMB w/o foregrounds')
							
							print delexe + " -f_sky " + str(configurations[exp]['fsky']) + \
							      " -l_min " + str(configurations[exp]['ell_min']) + \
							      " -l_max " + str(configurations[exp]['ell_max']) + \
							      " -lensed_path " + prefix + "_fidCls_lensedtotCls.dat" + \
							      " -unlensed_path " + prefix + "_fidCls_lenspotentialCls.dat" + \
							      " -noise_path " + exp_prefix + "_n_l_ee_bb.dat" + \
							      " -prefix " + exp_prefix
							data = np.genfromtxt(exp_prefix + '_c_l_bb_res.dat')
							Nl[exp][components]['BB_delens'][i_min:i_max] = data[:, 1]*1.0 / Dl_conv[i_min:i_max]
							data = np.genfromtxt(exp_prefix + '_n_l_dd.dat')
							Nl[exp][components]['dd'][i_min:i_max] = data[:, 1]*1.0 / Dl_conv[i_min:i_max]

						# call CMB x CIB delensing code using raw noise
						if 'CMBxCIB' in delensing_option_v_loc:
							com = [delexe, '-f_sky', str(configurations[exp]['fsky']), \
								       '-l_min', str(configurations[exp]['ell_min']), \
								       '-l_max', str(configurations[exp]['ell_max']), \
								       '-lensed_path', prefix + '_fidCls_lensedtotCls.dat', \
								       '-unlensed_path', prefix + '_fidCls_lenspotentialCls.dat', \
								       '-noise_path', exp_prefix + '_n_l_ee_bb.dat', \
								       '-f_l_cor_path', prefix + '_f_l_cor_planck_545.dat', \
								       '-prefix', exp_prefix]
                                                        try:
                                                                sp.check_call(com)
                                                        except:
                                                                print('external_program failed for CMBxCIB w/o foregrounds')
							data = np.genfromtxt(exp_prefix + '_c_l_bb_res.dat')
							Nl[exp][components]['BB_CIB_delens'][i_min:i_max] = data[:,1]*1.0 / Dl_conv[i_min:i_max]
							data = np.genfromtxt(exp_prefix + '_n_l_dd.dat')
							Nl[exp][components]['dd_CIB'][i_min:i_max] = data[:,1]*1.0 / Dl_conv[i_min:i_max]
							
						# call PERFECT delensing code using raw noise
						if 'CMBxLSS' in delensing_option_v_loc:
							com = [delexe, '-f_sky', str(configurations[exp]['fsky']), \
							               '-l_min', str(configurations[exp]['ell_min']), \
							               '-l_max', str(configurations[exp]['ell_max']), \
							               '-lensed_path', prefix + '_fidCls_lensedtotCls.dat', \
							               '-unlensed_path', prefix + '_fidCls_lenspotentialCls.dat', \
							               '-noise_path', exp_prefix + '_n_l_ee_bb.dat', \
							               '-f_l_cor_path', prefix + '_f_l_cor_perfect.dat', \
								       '-prefix', exp_prefix]
                                                        try:
                                                                sp.check_call(com)
                                                        except:
                                                                print('external_program failed for CMBxLSS w/o foregrounds')
							data = np.genfromtxt(exp_prefix + '_c_l_bb_res.dat')
							Nl[exp][components]['BB_lss_delens'][i_min:i_max] = data[:, 1]*1.0 / Dl_conv[i_min:i_max]
							data = np.genfromtxt(exp_prefix + '_n_l_dd.dat')
							Nl[exp][components]['dd_lss'][i_min:i_max] = data[:, 1]*1.0 / Dl_conv[i_min:i_max]
							
						####################################################      
						l0, l1 = 20, 200
						i0, i1 = np.argmin(np.abs( Cls_fid['ell'] - l0 )), np.argmin(np.abs( Cls_fid['ell'] - l1 ))
						i0_ = i0 + configurations[exp]['ell_min']
						i1_ = i1 + configurations[exp]['ell_min']
						if 'CMBxLSS' in delensing_option_v: Nl[exp][components]['alpha_CMBxLSS'] = round_sig( np.sum(Nl[exp][components]['BB_lss_delens'][i0_:i1_]) / np.sum(Cls_fid['BlBl'][i0:i1]) )
						if 'CMBxCIB' in delensing_option_v: Nl[exp][components]['alpha_CMBxCIB'] = round_sig( np.sum(Nl[exp][components]['BB_CIB_delens'][i0_:i1_]) / np.sum(Cls_fid['BlBl'][i0:i1]) )
						if 'CMBxCMB' in delensing_option_v: Nl[exp][components]['alpha_CMBxCMB'] = round_sig( np.sum(Nl[exp][components]['BB_delens'][i0_:i1_]) / np.sum(Cls_fid['BlBl'][i0:i1]) )
						####################################################

					else:

						## WITH FOREGROUNDS 
						# define a bunch of post-delensing arrays and their ranges. the delensing 
						# code outputs C_ls in the range [ell_min: ell_max + 1]; we want D_ls in the
						# range [ell_min_camb: ell_max + 1]
						Nl[exp][components]['dd_post_comp_sep'] = np.zeros(ell_count)
						Nl[exp][components]['BB_delens_post_comp_sep'] = np.zeros(ell_count)
						Nl[exp][components]['dd_CIB_post_comp_sep'] = np.zeros(ell_count)
						Nl[exp][components]['BB_CIB_delens_post_comp_sep'] = np.zeros(ell_count)
						Nl[exp][components]['dd_lss_post_comp_sep'] = np.zeros(ell_count)
						Nl[exp][components]['BB_lss_delens_post_comp_sep'] = np.zeros(ell_count)

						# write BB noise to file to pass to command-line delensing F90 code
						output = np.column_stack((ells[exp].flatten(), Nl[exp][components]['BB_post_comp_sep'].flatten()))
						np.savetxt(exp_prefix + '_n_l_ee_bb_cs.dat', output, fmt = ' %5.1i   %12.5e')

						# call iterative CMB EB delensing code using post-comp.sep. noise
						if 'CMBxCMB' in delensing_option_v_loc:
							com = [delexe, '-f_sky', str(configurations[exp]['fsky']), \
							               '-l_min', str(configurations[exp]['ell_min']), \
								       '-l_max', str(configurations[exp]['ell_max']), \
							               '-lensed_path', prefix + '_fidCls_lensedtotCls.dat', \
							               '-unlensed_path', prefix + '_fidCls_lenspotentialCls.dat', \
							               '-noise_path', exp_prefix + '_n_l_ee_bb_cs.dat', \
								       '-prefix', exp_prefix]
                                                        try:
                                                                sp.check_call(com)
                                                        except:
                                                                print('external_program failed for CMBxCMB')
							data = np.genfromtxt(exp_prefix + '_c_l_bb_res.dat')
							Nl[exp][components]['BB_delens_post_comp_sep'][i_min:i_max] = data[:, 1]*1.0 / Dl_conv[i_min:i_max]
							data = np.genfromtxt(exp_prefix + '_n_l_dd.dat')
							Nl[exp][components]['dd_post_comp_sep'][i_min:i_max] = data[:, 1]*1.0 / Dl_conv[i_min:i_max]

						# call CMB x CIB delensing code using post-comp.sep. noise
						if 'CMBxCIB' in delensing_option_v_loc:
							com = [delexe, '-f_sky', str(configurations[exp]['fsky']), \
							               '-l_min', str(configurations[exp]['ell_min']), \
							               '-l_max', str(configurations[exp]['ell_max']), \
							               '-lensed_path', prefix + '_fidCls_lensedtotCls.dat', \
							               '-unlensed_path', prefix + '_fidCls_lenspotentialCls.dat', \
							               '-noise_path', exp_prefix + '_n_l_ee_bb_cs.dat', \
							               '-f_l_cor_path', prefix + '_f_l_cor_planck_545.dat', \
								       '-prefix', exp_prefix]
                                                        try:
                                                                sp.check_call(com)
                                                        except:
                                                                print('external_program failed for CMBxCIB')
							data = np.genfromtxt(exp_prefix + '_c_l_bb_res.dat')
							Nl[exp][components]['BB_CIB_delens_post_comp_sep'][i_min:i_max] = data[:, 1]*1.0 / Dl_conv[i_min:i_max]
							data = np.genfromtxt(exp_prefix + '_n_l_dd.dat')
							Nl[exp][components]['dd_CIB_post_comp_sep'][i_min:i_max] = data[:, 1]*1.0 / Dl_conv[i_min:i_max]

						# call PERFECT delensing code using raw noise
						if 'CMBxLSS' in delensing_option_v_loc:
							com = [delexe, '-f_sky', str(configurations[exp]['fsky']), \
								       '-l_min', str(configurations[exp]['ell_min']), \
								       '-l_max', str(configurations[exp]['ell_max']), \
								       '-lensed_path', prefix + '_fidCls_lensedtotCls.dat', \
								       '-unlensed_path', prefix + '_fidCls_lenspotentialCls.dat', \
								       '-noise_path', exp_prefix + '_n_l_ee_bb.dat', \
								       '-f_l_cor_path', prefix + '_f_l_cor_perfect.dat', \
								       '-prefix', exp_prefix]
                                                        try:
                                                                sp.check_call(com)
                                                        except:
                                                                print('external_program failed for CMBxLSS')
							data = np.genfromtxt(exp_prefix + '_c_l_bb_res.dat')
							Nl[exp][components]['BB_lss_delens_post_comp_sep'][i_min:i_max] = data[:, 1]*1.0 / Dl_conv[i_min:i_max]
							data = np.genfromtxt(exp_prefix + '_n_l_dd.dat')
							Nl[exp][components]['dd_lss_post_comp_sep'][i_min:i_max] = data[:, 1]*1.0 / Dl_conv[i_min:i_max]

						####################################################      
						l0, l1 = 20, 200
						i0, i1 = np.argmin(np.abs( Cls_fid['ell'] - l0 )), \
								 np.argmin(np.abs( Cls_fid['ell'] - l1 ))
						i0_ = i0 + configurations[exp]['ell_min']
						i1_ = i1 + configurations[exp]['ell_min']
						if 'CMBxLSS' in delensing_option_v: Nl[exp][components]['alpha_CMBxLSS_post_comp_sep'] = round_sig( np.sum(Nl[exp][components]['BB_lss_delens_post_comp_sep'][i0_:i1_]) / np.sum(Cls_fid['BlBl'][i0:i1]) )
						if 'CMBxCIB' in delensing_option_v: Nl[exp][components]['alpha_CMBxCIB_post_comp_sep'] = round_sig( np.sum(Nl[exp][components]['BB_CIB_delens_post_comp_sep'][i0_:i1_]) / np.sum(Cls_fid['BlBl'][i0:i1]) )
						if 'CMBxCMB' in delensing_option_v: Nl[exp][components]['alpha_CMBxCMB_post_comp_sep'] = round_sig( np.sum(Nl[exp][components]['BB_delens_post_comp_sep'][i0_:i1_]) / np.sum(Cls_fid['BlBl'][i0:i1]) )


	#####################################################
    ## use f2py compiled delensing
	else:

		# determine delensing performance
		ind = -1
		for exp1 in experiments:
			ind += 1
			for exp2 in experiments[ind:]:
				if exp1 == exp2 :
					# experiment alone
					exp = exp1
					if cross_only: continue
				else:
					# combination of two experiments
					exp = exp1+' x '+exp2
				
				i_min = configurations[exp]['ell_min'] - ell_min_camb
				i_max = configurations[exp]['ell_max'] - ell_min_camb + 1
				ell_count = configurations[exp]['ell_max'] - ell_min_camb + 1
				
				for components in components_v:

					print exp,' >>', components,' >>', delensing_option_v_loc
					print '///// with a noise post-comp of ', foregrounds[exp][components]['uKCMB/pix_postcompsep']*pix_size_map_arcmin
					print '///// and an effective amplitude of residuals of ', foregrounds[exp][components]['r_eff']
					if components == 'cmb-only':

						## WITHOUT FOREGROUNDS

						# define a bunch of post-delensing arrays and their ranges. the delensing 
						# code outputs C_ls in the range [ell_min: ell_max + 1]; we want D_ls in the
						# range [ell_min_camb: ell_max + 1]
						# NB: delensing code outputs N_l^pp, so must convert to N_l^dd
						Nl[exp][components]['dd'] = np.zeros(ell_count)
						Nl[exp][components]['BB_delens'] = np.zeros(ell_count)
						Nl[exp][components]['dd_CIB'] = np.zeros(ell_count)
						Nl[exp][components]['BB_CIB_delens'] = np.zeros(ell_count)
						Nl[exp][components]['dd_lss'] = np.zeros(ell_count)
						Nl[exp][components]['BB_lss_delens'] = np.zeros(ell_count)
						
						print configurations[exp]['ell_min'], configurations[exp]['ell_max'], np.sum(Cls_fid['EuEu'][i_min:i_max] * Dl_conv[i_min:i_max]), np.sum(Cls_fid['EE_tot'][i_min:i_max] * Dl_conv[i_min:i_max]), np.sum(Cls_fid['BuBu'][i_min:i_max] * Dl_conv[i_min:i_max]), np.sum(Cls_fid['BB_tot'][i_min:i_max] * Dl_conv[i_min:i_max]), np.sum(Cls_fid['dd'][i_min:i_max] * Dl_conv[i_min:i_max] ** 2 / 2.0 / np.pi), np.sum(np.zeros(i_max-i_min)), np.sum(Nl_in[exp][components]['BB'][i_min:i_max] * Dl_conv[i_min:i_max]), converge, False


						# call iterative CMB EB delensing code using raw noise
						if 'CMBxCMB' in delensing_option_v_loc:
							Nl[exp][components]['dd'][i_min:i_max], Nl[exp][components]['BB_delens'][i_min:i_max] = pd.delensing_performance(configurations[exp]['ell_min'], configurations[exp]['ell_max'], Cls_fid['EuEu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['EE_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BuBu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BB_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['dd'][i_min:i_max] * Dl_conv[i_min:i_max] ** 2 / 2.0 / np.pi, np.zeros(i_max-i_min), Nl_in[exp][components]['BB'][i_min:i_max] * Dl_conv[i_min:i_max], converge, False)
							Nl[exp][components]['BB_delens'][i_min:i_max] /= Dl_conv[i_min:i_max]
							Nl[exp][components]['dd'][i_min:i_max] /= Dl_conv[i_min:i_max] ** 2 / \
																	  (2.0 * np.pi)

						# call CMB x CIB delensing code using raw noise
						if 'CMBxCIB' in delensing_option_v_loc:
							Nl[exp][components]['dd_CIB'][i_min:i_max], Nl[exp][components]['BB_CIB_delens'][i_min:i_max] = pd.delensing_performance(configurations[exp]['ell_min'], configurations[exp]['ell_max'], Cls_fid['EuEu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['EE_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BuBu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BB_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['dd'][i_min:i_max] * Dl_conv[i_min:i_max] ** 2 / 2.0 / np.pi, f_l_cor_cib[i_min:i_max], Nl_in[exp][components]['BB'][i_min:i_max] * Dl_conv[i_min:i_max], converge, False)
							Nl[exp][components]['BB_CIB_delens'][i_min:i_max] /= Dl_conv[i_min:i_max]
							Nl[exp][components]['dd_CIB'][i_min:i_max] /= Dl_conv[i_min:i_max] ** 2 /\
																		  (2.0 * np.pi)

						# call PERFECT delensing code using raw noise
						if 'CMBxLSS' in delensing_option_v_loc:
							Nl[exp][components]['dd_lss'][i_min:i_max], Nl[exp][components]['BB_lss_delens'][i_min:i_max] = pd.delensing_performance(configurations[exp]['ell_min'], configurations[exp]['ell_max'], Cls_fid['EuEu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['EE_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BuBu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BB_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['dd'][i_min:i_max] * Dl_conv[i_min:i_max] ** 2 / 2.0 / np.pi, f_l_cor_lss[i_min:i_max], Nl_in[exp][components]['BB'][i_min:i_max] * Dl_conv[i_min:i_max], converge, False)
							Nl[exp][components]['BB_lss_delens'][i_min:i_max] /= Dl_conv[i_min:i_max]
							Nl[exp][components]['dd_lss'][i_min:i_max] /= Dl_conv[i_min:i_max] ** 2 /\
																		  (2.0 * np.pi)
							
						####################################################
						# calculate delensing factor
						l0, l1 = 20, 200
						i0, i1 = np.argmin(np.abs( Cls_fid['ell'] - l0 )), np.argmin(np.abs( Cls_fid['ell'] - l1 ))
						if 'CMBxLSS' in delensing_option_v: Nl[exp][components]['alpha_CMBxLSS'] = np.sum(Nl[exp][components]['BB_lss_delens'][i0:i1]*Dl_conv[i0:i1]) / np.sum(Cls_fid['BlBl'][i0:i1]*Dl_conv[i0:i1])
						if 'CMBxCIB' in delensing_option_v: Nl[exp][components]['alpha_CMBxCIB'] = np.sum(Nl[exp][components]['BB_CIB_delens'][i0:i1]*Dl_conv[i0:i1]) / np.sum(Cls_fid['BlBl'][i0:i1]*Dl_conv[i0:i1])
						if 'CMBxCMB' in delensing_option_v: Nl[exp][components]['alpha_CMBxCMB'] = np.sum(Nl[exp][components]['BB_delens'][i0:i1]*Dl_conv[i0:i1]) / np.sum(Cls_fid['BlBl'][i0:i1]*Dl_conv[i0:i1])
						####################################################
						
					else:

						## WITH FOREGROUNDS 
						# define a bunch of post-delensing arrays and their ranges. the delensing 
						# code outputs C_ls in the range [ell_min: ell_max + 1]; we want D_ls in the
						# range [ell_min_camb: ell_max + 1]
						# NB: delensing code outputs N_l^pp, so must convert to N_l^dd
						Nl[exp][components]['dd_post_comp_sep'] = np.zeros(ell_count)
						Nl[exp][components]['BB_delens_post_comp_sep'] = np.zeros(ell_count)
						Nl[exp][components]['dd_CIB_post_comp_sep'] = np.zeros(ell_count)
						Nl[exp][components]['BB_CIB_delens_post_comp_sep'] = np.zeros(ell_count)
						Nl[exp][components]['dd_lss_post_comp_sep'] = np.zeros(ell_count)
						Nl[exp][components]['BB_lss_delens_post_comp_sep'] = np.zeros(ell_count)

						# call iterative CMB EB delensing code using post-comp.sep. noise
						if 'CMBxCMB' in delensing_option_v_loc:
							Nl[exp][components]['dd_post_comp_sep'][i_min:i_max], Nl[exp][components]['BB_delens_post_comp_sep'][i_min:i_max] = pd.delensing_performance(configurations[exp]['ell_min'], configurations[exp]['ell_max'], Cls_fid['EuEu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['EE_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BuBu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BB_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['dd'][i_min:i_max] * Dl_conv[i_min:i_max] ** 2 / 2.0 / np.pi, np.zeros(i_max-i_min), Nl[exp][components]['BB_post_comp_sep'][i_min:i_max] * Dl_conv[i_min:i_max], converge, False)
							Nl[exp][components]['BB_delens_post_comp_sep'][i_min:i_max] /= \
							  Dl_conv[i_min:i_max]
							Nl[exp][components]['dd_post_comp_sep'][i_min:i_max] /= \
							  Dl_conv[i_min:i_max] ** 2 / (2.0 * np.pi)

						# call CMB x CIB delensing code using post-comp.sep. noise
						if 'CMBxCIB' in delensing_option_v_loc:
							Nl[exp][components]['dd_CIB_post_comp_sep'][i_min:i_max], Nl[exp][components]['BB_CIB_delens_post_comp_sep'][i_min:i_max] = pd.delensing_performance(configurations[exp]['ell_min'], configurations[exp]['ell_max'], Cls_fid['EuEu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['EE_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BuBu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BB_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['dd'][i_min:i_max] * Dl_conv[i_min:i_max] ** 2 / 2.0 / np.pi, f_l_cor_cib[i_min:i_max], Nl[exp][components]['BB_post_comp_sep'][i_min:i_max] * Dl_conv[i_min:i_max], converge, False)
							Nl[exp][components]['BB_CIB_delens_post_comp_sep'][i_min:i_max] /= \
							  Dl_conv[i_min:i_max]
							Nl[exp][components]['dd_CIB_post_comp_sep'][i_min:i_max] /= \
							  Dl_conv[i_min:i_max] ** 2 / (2.0 * np.pi)

						# call PERFECT delensing code using post-comp.sep. noise
						if 'CMBxLSS' in delensing_option_v_loc:
							Nl[exp][components]['dd_lss_post_comp_sep'][i_min:i_max], Nl[exp][components]['BB_lss_delens_post_comp_sep'][i_min:i_max] = pd.delensing_performance(configurations[exp]['ell_min'], configurations[exp]['ell_max'], Cls_fid['EuEu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['EE_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BuBu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BB_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['dd'][i_min:i_max] * Dl_conv[i_min:i_max] ** 2 / 2.0 / np.pi, f_l_cor_lss[i_min:i_max], Nl[exp][components]['BB_post_comp_sep'][i_min:i_max] * Dl_conv[i_min:i_max], converge, False)
							Nl[exp][components]['BB_lss_delens_post_comp_sep'][i_min:i_max] /= \
							  Dl_conv[i_min:i_max]
							Nl[exp][components]['dd_lss_post_comp_sep'][i_min:i_max] /= \
							  Dl_conv[i_min:i_max] ** 2 / (2.0 * np.pi)
						

						####################################################
						# calculate delensing factor
						l0, l1 = 20, 200
						i0, i1 = np.argmin(np.abs( Cls_fid['ell'] - l0 )), \
								 np.argmin(np.abs( Cls_fid['ell'] - l1 ))
						if 'CMBxLSS' in delensing_option_v: Nl[exp][components]['alpha_CMBxLSS_post_comp_sep'] = np.sum(Nl[exp][components]['BB_lss_delens_post_comp_sep'][i0:i1]*Dl_conv[i0:i1]) / np.sum(Cls_fid['BlBl'][i0:i1]*Dl_conv[i0:i1])
						if 'CMBxCIB' in delensing_option_v: Nl[exp][components]['alpha_CMBxCIB_post_comp_sep'] = np.sum(Nl[exp][components]['BB_CIB_delens_post_comp_sep'][i0:i1]*Dl_conv[i0:i1]) / np.sum(Cls_fid['BlBl'][i0:i1]*Dl_conv[i0:i1])
						if 'CMBxCMB' in delensing_option_v: Nl[exp][components]['alpha_CMBxCMB_post_comp_sep'] = np.sum(Nl[exp][components]['BB_delens_post_comp_sep'][i0:i1]*Dl_conv[i0:i1]) / np.sum(Cls_fid['BlBl'][i0:i1]*Dl_conv[i0:i1])
						####################################################



	######################################################################################################
	######################################################################################################
	## delensing performance versus BB-only sigma(r=0)
	if delens_scatter:

		# need different options for different experiment groups
		print ' DOING THE OTHER FOREGROUNDS FIGURE '
		if ('Simons Array' in experiments):
			stage = 'III'
		else:
			stage = 'IV'
		print 'plotting stage {:} experiments'.format(stage)

		# generate fiducial tensors: same cosmology with r = 1
		print ' DOING THE OTHER DELENSING FIGURE '
		params_fid_t = dict(params_fid_v[0])
		params_fid_t['r'] = 1.0
		print '##### computing fiducial tensor Cls #####'
		name_fid_t = 'fidCls'
		for p in range(len(params_fid_t.keys())):
			name_fid_t += '_'+str(params_fid_t.keys()[p])+'_'+str(params_fid_t[params_fid_t.keys()[p]] )
		print 'looking for ', name_fid_t+'.pkl'
		fnames_fid_t = glob.glob( name_fid_t+'.pkl' )
		if not fnames_fid_t:
			print '##### computing Cls file because it does not seem to be on disk #####'
			Cls_fid_t=python_camb_self_consistent.submit_camb(h=params_fid_t['h'], \
				ombh2=params_fid_t['ombh2'], omch2=params_fid_t['omch2'], \
				omnuh2=params_fid_t['omnuh2'], omk=params_fid_t['omk'], \
				YHe=params_fid_t['YHe'], Neff=params_fid_t['Neff'], w=params_fid_t['w'], \
				wa=params_fid_t['wa'], tau=params_fid_t['tau'], As=params_fid_t['As'], \
				ns=params_fid_t['ns'], alphas=params_fid_t['alphas'], \
				nT=params_fid_t['nT'], r=params_fid_t['r'], \
				k_scalar=params_fid_t['k_scalar'], k_tensor=params_fid_t['k_tensor'], \
				eta=1.0, lensing_z_max=-1, exe = camb)
			save_obj('./', name_fid_t, Cls_fid_t)
		else:
			print '##### loading already existing Cls file #####'
			Cls_fid_t = load_obj('./', fnames_fid_t[0])
			
		# loop through delensing types and plot
		fg_comp = 'sync+dust'
		del_opts = [del_opt for del_opt in delensing_option_v_loc if del_opt not in ['']]
		n_del_opts = len(del_opts)
		fig, axes = pl.subplots(1, n_del_opts, sharey=True, figsize=(n_del_opts * 4,4), \
								facecolor='w', edgecolor='k')
		del_ind = 0
		for del_opt in del_opts :
			 
			# keep track of plotted points and labels for legend, 
			# and experiments and locations plotted
			exp_ind = 0
			points = []
			exp_labs = []
			exp_locs = []
			loc_counts = dict.fromkeys(loc_markers.keys(), 0)
			for exp1 in experiments:
				
				# deal with experimental combinations
				for exp2 in experiments[exp_ind:]:
					if exp1 == exp2 :
						exp = exp1
						exp_lab = exp
						loc = configurations[exp]['loc']
					else:
						if (stage == 'III'):
							if  (exp1 != 'Planck') and (exp2 != 'Planck'):
								continue
							elif (exp1 == 'Planck'):
								exp_lab = exp1+' x '+exp2
							else:
								exp_lab = exp2+' x '+exp1
						else:
							exp_lab = ' x '.join(sorted((exp1, exp2), key=lambda t: t[0]))
						exp = exp1+' x '+exp2
						loc = locs['cross']
			
					# find the correct value of alpha for the delensing option
					if del_opt == 'CMBxLSS':
		   				alpha_loc = Nl[exp][fg_comp]['alpha_CMBxLSS_post_comp_sep']
					if del_opt == 'CMBxCIB':
		   				alpha_loc = Nl[exp][fg_comp]['alpha_CMBxCIB_post_comp_sep']
					if del_opt == 'CMBxCMB':
		   				alpha_loc = Nl[exp][fg_comp]['alpha_CMBxCMB_post_comp_sep']
			
					# quickly calculate a BB-only sigma_r
					i_min = configurations[exp]['ell_min'] - ell_min_camb
					i_max = configurations[exp]['ell_max'] - ell_min_camb + 1
					ell_count = configurations[exp]['ell_max'] - ell_min_camb + 1
					Nl_tot_loc = np.zeros(ell_count)
					Nl_tot_loc += Nl[exp][fg_comp]['BB_post_comp_sep']*1.0
					if del_opt=='CMBxLSS':
						Nl_tot_loc += Nl[exp][fg_comp]['BB_lss_delens_post_comp_sep']*1.0
					elif del_opt=='CMBxCIB':
						Nl_tot_loc += Nl[exp][fg_comp]['BB_CIB_delens_post_comp_sep']*1.0
					elif del_opt=='CMBxCMB':
						Nl_tot_loc += Nl[exp][fg_comp]['BB_delens_post_comp_sep']*1.0
					sigma_loc = 1.0/np.sqrt(np.sum((2.0 * ells[exp][i_min:i_max] + 1) * \
												   configurations[exp]['fsky'] / 2.0 * \
												   (Cls_fid_t['BuBu'][i_min:i_max] / \
													Nl_tot_loc[i_min:i_max]) ** 2))
#													(Cls_fid['BuBu'][i_min:i_max] + \
#													 Nl_tot_loc[i_min:i_max])) ** 2))
					
					point, = axes[del_ind].semilogx(sigma_loc, alpha_loc, \
												    linestyle = 'None', \
												    marker = loc_markers[loc], \
												    markerfacecolor = 'None', \
												    markeredgecolor = cb_colors[loc_counts[loc]], \
												    markeredgewidth = 2, markersize = 8)
					points.append(point)
					exp_labs.append(exp_lab)
					exp_locs.append(loc)
					loc_counts[loc] +=1
					
				exp_ind += 1

			# label plot and set limits
			if stolyarov_sync:
				if stage == 'III':
					ymin = 0.3
					xmin = 10.0**(-2.75)
					xmax = 10.0**(-0.25)
					r_ticks = [ 1e-2, 1e-1 ]
				else:
					ymin = 0.3
					xmin = 2e-5#10.0**(-4.5)
					xmax = 10.0**(-1.5)
					r_ticks = [ 1.0e-4, 1.0e-3, 1.0e-2 ]
			else:
				if stage == 'III':
					ymin = 0.3
					xmin = 10.0**(-2.75)
					xmax = 10.0**(-0.25)
					r_ticks = [ 1e-2, 1e-1 ]
				else:
					axes[del_ind].set_xscale('linear')
					ymin = 0.1
					xmin = 2.0e-5
					xmax = 2.5e-4
					r_ticks = [ 5e-5, 1.0e-4, 1.5e-4, 2.0e-4 ]
			axes[del_ind].text(0.98, 0.02, del_opt, fontsize = 16, \
							   transform = axes[del_ind].transAxes, \
							   horizontalalignment = 'right', \
							   verticalalignment = 'bottom')
			axes[del_ind].set_ylim([ymin, 1.0])
			axes[del_ind].set_xlim([xmin, xmax])
			axes[del_ind].set_xticks(r_ticks)
			if stage == 'IV' and not stolyarov_sync:
				axes[del_ind].xaxis.set_major_formatter(ticker.FuncFormatter(eformat))
				axes[del_ind].xaxis.set_minor_formatter(ticker.FuncFormatter(eformat))
			#pl.tick_params(axis='both', which='major', labelsize=24)
			#pl.tick_params(axis='both', which='minor', labelsize=24)
			
			del_ind += 1
		
		# set overall axis labels
		axes[1].set_xlabel(r'$\sigma(r = 0)$', fontsize = 22)
		axes[0].set_ylabel(r'delensing factor $\alpha$', fontsize = 22)
		if stolyarov or stolyarov_sync:
			figtitle = '$\mathbf{A}$-expansion approach, '
		else:
			figtitle = '$n_p$ approach, '
		if stage == 'III':
			figtitle += 'pre-2020'
		else:
			figtitle += 'post-2020'
		if cbass:
			figtitle += ' + C-BASS'
		if quijote:
			figtitle += ' + Quijote'
		axes[1].set_title(figtitle + '\n', fontsize=24)

		# create legend: lots of faff to ensure it's not cropped by savefig
		# sort legend to make readable
		#exp_labs, points = zip(*sorted(zip(exp_labs, points), key=lambda t: t[0]))
		exp_locs, exp_labs, points = zip(*sorted(zip(exp_locs, exp_labs, points), key=lambda t: (t[0], t[1])))
		lgd = pl.legend(points, exp_labs, loc = 'center', fontsize = 10, \
						bbox_to_anchor = (1.4, 0.5), numpoints = 1)
		fr = lgd.get_frame()
		fr.set_lw(1.5)
		
		# remove whitespace and save figure
		fig.subplots_adjust(wspace = 0)
		figfile = '../article/alpha_vs_sigma_r_S' + stage
		if stolyarov:
			figfile += '_stol_d'
		if stolyarov_sync:
			figfile += '_stol_ds'
		if cbass:
			figfile += '_cbass'
		if quijote:
			figfile += '_quijote'
		
		figfile += 'COrE_corr'		
		pl.savefig(figfile + '.pdf', bbox_inches='tight', bbox_extra_artists=(lgd,))
		pl.show()
		exit()
	
	## plots of delensing performance as a function of ell
	if delens_power_spectrum:
		'''
		fig = pl.figure(num=None, figsize=(12,15), facecolor='w', edgecolor='k')
		ind1 = 1
		for exp1 in experiments:
			ind2 = 1
			for exp2 in experiments:
				if exp1 == exp2 :
					exp = exp1
				else:
					#exp = exp1+' x '+exp2	
					continue

				if ind2 >= ind1:

					i_min = configurations[exp]['ell_min'] - ell_min_camb
					i_max = configurations[exp]['ell_max'] - ell_min_camb + 1

					ax = pl.subplot2grid(( len(experiments)+1, len(experiments)+1), (ind1,ind2))
					if exp1 == exp2:
						pl.ylabel(  exp, fontsize=18 )
					elif ind1 == 1:
						pl.title( exp2, fontsize=18  )
					#pl.loglog(ells[exp][i_min:i_max], Cls_fid['BB'][i_min:i_max], 'k')
					pl.loglog(ells[exp][i_min:i_max], Cls_fid['BuBu'][i_min:i_max], 'k--')
					pl.loglog(ells[exp][i_min:i_max], Cls_fid['BuBu'][i_min:i_max]/10, 'k--')
					pl.loglog(ells[exp][i_min:i_max], Cls_fid['BlBl'][i_min:i_max], 'k')

					pl.loglog(ells[exp][i_min:i_max], Nl[exp]['BB'][i_min:i_max], 'r:')
					pl.loglog(ells[exp][i_min:i_max], Nl[exp]['BB_post_comp_sep'][i_min:i_max], 'r')
					pl.fill_between(ells[exp][i_min:i_max], Nl[exp]['BB_delens_post_comp_sep'][i_min:i_max], Nl[exp]['BB_delens'][i_min:i_max], color='DarkOrange', alpha=0.4  )
					pl.fill_between(ells[exp][i_min:i_max], Nl[exp]['BB_lss_delens_post_comp_sep'][i_min:i_max], Nl[exp]['BB_lss_delens'][i_min:i_max], color='DarkGRay', alpha=0.4  )
					pl.fill_between(ells[exp][i_min:i_max], Nl[exp]['BB_CIB_delens_post_comp_sep'][i_min:i_max], Nl[exp]['BB_CIB_delens'][i_min:i_max], color='DarkBlue', alpha=0.4  )
					ax.xaxis.set_ticklabels([])
					ax.yaxis.set_ticklabels([])
					pl.xlim([10, 3000])
					pl.ylim([1e-4, 1e0])

				ind2 += 1
			ind1 += 1

		ax = pl.gca()
		pl.ylabel('$ (\ell(\ell+1)/2\pi )\, C_\ell\ [\mu K^2] $', fontsize=28)
		pl.xlabel('$\ell$', fontsize=28)
		for tick in ax.xaxis.get_major_ticks():
			tick.label.set_fontsize(24)
		for tick in ax.yaxis.get_major_ticks():
			tick.label.set_fontsize(24)
		#pl.savefig('../article/Delensing_performances_experiments_w_wo_comp_sep.pdf')
		pl.show()
		exit()
		'''
		'''
		print ' DOING THE DELENSING FIGURE '
		pl.figure(num=None, figsize=(20,6), facecolor='w', edgecolor='k')
		ind = 0
		for exp in experiments:
			print exp, np.min(ells[exp]), np.max(ells[exp])
			ind += 1
			ind12 = 0
			if ind > 1:
				ax = pl.subplot(1, len(experiments), ind, sharey=ax )
				pl.setp( ax.get_yticklabels(), visible=False)
			else:
				ax = pl.subplot(1, len(experiments), ind)
			colors = ['DarkOrange', 'DarkBlue', 'DarkGray']
			#pl.loglog(Cls_fid['ell'], Cls_fid['BlBl'], linewidth=3.0, alpha=0.7, color='DarkGray' )
			#pl.loglog(Cls_fid['ell'], Cls_fid['BuBu']*0.1/params_fid['r'], linewidth=2.0, alpha=0.7, color='Black' )
			#pl.loglog(Cls_fid['ell'], Cls_fid['BuBu']*0.01/params_fid['r'], linewidth=2.0, alpha=0.7, color='Black' )
			#pl.loglog(Cls_fid['ell'], Cls_fid['BuBu']*0.001/params_fid['r'], linewidth=2.0, alpha=0.7, color='Black' )
			#pl.loglog(ells[exp][configurations[exp]['ell_min']-ell_min_camb:configurations[exp]['ell_max']-ell_min_camb], Nl[exp]['BB'][configurations[exp]['ell_min']-ell_min_camb:configurations[exp]['ell_max']-ell_min_camb], color='k', linestyle=':', linewidth=3.0 )
			i_min = configurations[exp]['ell_min'] - ell_min_camb
			i_max = configurations[exp]['ell_max'] - ell_min_camb + 1
			for components in components_v:
				if components == 'cmb-only':
					pl.semilogx(ells[exp][i_min:i_max],  np.abs(Nl[exp][components]['BB_delens'][i_min:i_max])*100/Cls_fid['BlBl'][i_min:i_max], color='k', linestyle='-', linewidth=3.0 )
					pl.semilogx(ells[exp][i_min:i_max],  np.abs(Nl[exp][components]['BB_lss_delens'][i_min:i_max])*100/Cls_fid['BlBl'][i_min:i_max], color='k', linestyle=':', linewidth=3.0 )
					pl.semilogx(ells[exp][i_min:i_max],  np.abs(Nl[exp][components]['BB_CIB_delens'][i_min:i_max])*100/Cls_fid['BlBl'][i_min:i_max], color='k', linestyle='--', linewidth=3.0 )
				else:
					pl.semilogx(ells[exp][i_min:i_max],  np.abs(Nl[exp][components]['BB_delens_post_comp_sep'][i_min:i_max])*100/Cls_fid['BlBl'][i_min:i_max], linestyle='-', color=colors[ind12], linewidth=3.0 )
					pl.semilogx(ells[exp][i_min:i_max],  np.abs(Nl[exp][components]['BB_lss_delens_post_comp_sep'][i_min:i_max])*100/Cls_fid['BlBl'][i_min:i_max], linestyle=':', color=colors[ind12], linewidth=3.0 )
					pl.semilogx(ells[exp][i_min:i_max],  np.abs(Nl[exp][components]['BB_CIB_delens_post_comp_sep'][i_min:i_max])*100/Cls_fid['BlBl'][i_min:i_max], linestyle='--', color=colors[ind12], linewidth=3.0 )
				#pl.fill_between(ells[exp][i_min:i_max], Nl[exp][components]['BB_delens_post_comp_sep'][i_min:i_max], Nl[exp]['BB_delens'][i_min:i_max], color='DarkOrange', alpha=0.4  )
				#pl.fill_between(ells[exp][i_min:i_max], Nl[exp][components]['BB_lss_delens_post_comp_sep'][i_min:i_max], Nl[exp]['BB_lss_delens'][i_min:i_max], color='DarkGRay', alpha=0.4  )
				#pl.fill_between(ells[exp][i_min:i_max], Nl[exp][components]['BB_CIB_delens_post_comp_sep'][i_min:i_max], Nl[exp]['BB_CIB_delens'][i_min:i_max], color='DarkBlue', alpha=0.4  )
				#pl.loglog(ells[exp][configurations[exp]['ell_min']-ell_min_camb:configurations[exp]['ell_max']-ell_min_camb], foregrounds[exp][components]['Cl_res'][configurations[exp]['ell_min']-ell_min_camb:configurations[exp]['ell_max']-ell_min_camb], color=colors[ind12], linewidth=3.0 )
				#pl.loglog(ells[exp][configurations[exp]['ell_min']-ell_min_camb:configurations[exp]['ell_max']-ell_min_camb], Nl[exp][components]['BB_post_comp_sep'][configurations[exp]['ell_min']-ell_min_camb:configurations[exp]['ell_max']-ell_min_camb], color=colors[ind12], linestyle='--', linewidth=3.0 )
				ind12 += 1
			pl.title( exp , fontsize=24)
			pl.xlim([10, 2000])
			pl.ylim([-5.0, 105.0])
			#ax = pl.gca()
			if ind == 1: pl.ylabel(r'$C_\ell^{BB,\,delens}/C_\ell^{BB,\,lens}\ [ \% ]$', fontsize=24)
			pl.xlabel('$\ell$', fontsize=24)
			for tick in ax.xaxis.get_major_ticks():
				tick.label.set_fontsize(14)
			for tick in ax.yaxis.get_major_ticks():
				tick.label.set_fontsize(14)
		pl.savefig('../article/Delensing_performances_experiments_w_wo_comp_sep_v04042015.pdf')
		pl.show()
		'''

		# need different options for different experiment groups
		print ' DOING THE DELENSING FIGURE '
		if ('Simons Array' in experiments):
			stage = 'III'
		else:
			stage = 'IV'
		print 'plotting stage {:} experiments'.format(stage)
        
		colors = [ 'OrangeRed', 'Orange', 'DarkRed' ]
		n_exp = len(experiments)
		n_combo = n_exp * (n_exp + 1) / 2
		fig, axes = pl.subplots(1, n_combo, sharey=True, figsize=(n_combo*4,4), \
								facecolor='w', edgecolor='k')
		ind1 = 1
		for exp1 in experiments:
			ind2 = 1
			for exp2 in experiments:
				
				if exp1 == exp2 :
					exp = exp1
					sp_ind = ind1-1
				else:
					exp = exp1+' x '+exp2
					sp_ind = ind1*(2*n_exp-ind1+1)/2+ind2-ind1-1
				
				# do not re-plot cross correlations
				if ind2 >= ind1:
					i_min = configurations[exp]['ell_min'] - ell_min_camb
					i_max = configurations[exp]['ell_max'] - ell_min_camb + 1
					ind12 = 0
					for components in components_v:
						if components == 'cmb-only':
							axes[sp_ind].semilogx(ells[exp][i_min:i_max], \
								np.abs(Nl[exp][components]['BB_delens'][i_min:i_max])*100/\
								Cls_fid['BlBl'][i_min:i_max], color='k', linestyle='-', \
								linewidth=3.0 )
							axes[sp_ind].semilogx(ells[exp][i_min:i_max],
								np.abs(Nl[exp][components]['BB_lss_delens'][i_min:i_max])*100/\
								Cls_fid['BlBl'][i_min:i_max], color='k', linestyle='--', \
								dashes = [4, 2], linewidth=3.0 )
							axes[sp_ind].semilogx(ells[exp][i_min:i_max], \
								np.abs(Nl[exp][components]['BB_CIB_delens'][i_min:i_max])*100/\
								Cls_fid['BlBl'][i_min:i_max], color='k', linestyle='--', \
								dashes = [10, 2], linewidth=3.0 )
						else:
							axes[sp_ind].semilogx(ells[exp][i_min:i_max], \
								np.abs(Nl[exp][components]['BB_delens_post_comp_sep'][i_min:i_max])*100/\
								Cls_fid['BlBl'][i_min:i_max], linestyle='-', \
								color=colors[ind12], linewidth=3.0, alpha=0.8)
							axes[sp_ind].semilogx(ells[exp][i_min:i_max], \
								np.abs(Nl[exp][components]['BB_lss_delens_post_comp_sep'][i_min:i_max])*100/\
								Cls_fid['BlBl'][i_min:i_max], linestyle='--', \
								dashes = [4, 2], color=colors[ind12], linewidth=3.0, alpha=0.8)
							axes[sp_ind].semilogx(ells[exp][i_min:i_max], \
								np.abs(Nl[exp][components]['BB_CIB_delens_post_comp_sep'][i_min:i_max])*100/\
								Cls_fid['BlBl'][i_min:i_max], linestyle='--', \
								dashes = [10, 2], color=colors[ind12], linewidth=3.0, alpha=0.8)
							ind12 += 1
					if len(exp) > 20:
						axes[sp_ind].set_title(exp.replace(' x ', ' x\n'), fontsize=22 )
					else:
						axes[sp_ind].set_title( exp, fontsize=22 )
					axes[sp_ind].set_xlim([2, 2000])
					#axes[sp_ind].set_ylim([-5.0, 105.0])
					if stage == 'III':
						axes[sp_ind].set_ylim([35.0, 105.0])
					else:
						axes[sp_ind].set_ylim([10.0, 105.0])
					axes[sp_ind].tick_params(axis='both', which='major', labelsize=16)
					axes[sp_ind].tick_params(axis='both', which='minor', labelsize=16)
					axes[sp_ind].grid(True)
				ind2 +=1
			ind1 += 1
		axes[1].set_xlabel(r'$\ell$', fontsize = 22)
		axes[0].set_ylabel(r'$C_\ell^{BB,\,{{\rm delens}}}/C_\ell^{BB,\,{{\rm lens}}}\ [ \% ]$', \
						   fontsize = 22)
		fig.subplots_adjust(wspace = 0)
		if stage == 'III':
			pl.savefig('../article/Delensing_performances_experiments_w_wo_comp_sep_SIII.pdf', \
					   bbox_inches='tight')
		else:
			pl.savefig('../article/Delensing_performances_experiments_w_wo_comp_sep_SIV.pdf', \
					   bbox_inches='tight')
		pl.show()
		exit()

		'''
		print ' DOING THE DELENSING FIGURE '
		pl.figure(num=None, figsize=(12,12), facecolor='w', edgecolor='k')
		ind1 = 0
		for exp1 in experiments:
			ind2 = 0
			for exp2 in experiments:
				
				if exp1 == exp2 :
					exp = exp1
				else:
					exp = exp1+' x '+exp2
				
				# do not plot the lower-left panel
				ind_subplot = ind1*len(experiments)+ind2+1
				if ind2 >= ind1:	
					ax = pl.subplot(2, len(experiments), ind_subplot)#, sharey=ax, sharex=ax )
					pl.title( exp, fontsize=28 )
					colors = ['DarkOrange', 'DarkBlue', 'DarkGray']
					i_min = configurations[exp]['ell_min'] - ell_min_camb
					i_max = configurations[exp]['ell_max'] - ell_min_camb + 1

					ind12 = 0
					for components in components_v:
						if components == 'cmb-only':
							pl.semilogx(ells[exp][i_min:i_max],  np.abs(Nl[exp][components]['BB_delens'][i_min:i_max])*100/Cls_fid['BlBl'][i_min:i_max], color='k', linestyle='-', linewidth=3.0 )
							pl.semilogx(ells[exp][i_min:i_max],  np.abs(Nl[exp][components]['BB_lss_delens'][i_min:i_max])*100/Cls_fid['BlBl'][i_min:i_max], color='k', linestyle=':', linewidth=3.0 )
							pl.semilogx(ells[exp][i_min:i_max],  np.abs(Nl[exp][components]['BB_CIB_delens'][i_min:i_max])*100/Cls_fid['BlBl'][i_min:i_max], color='k', linestyle='--', linewidth=3.0 )
						else:
							pl.semilogx(ells[exp][i_min:i_max],  np.abs(Nl[exp][components]['BB_delens_post_comp_sep'][i_min:i_max])*100/Cls_fid['BlBl'][i_min:i_max], linestyle='-', color=colors[ind12], linewidth=3.0, alpha=0.8)
							pl.semilogx(ells[exp][i_min:i_max],  np.abs(Nl[exp][components]['BB_lss_delens_post_comp_sep'][i_min:i_max])*100/Cls_fid['BlBl'][i_min:i_max], linestyle=':', color=colors[ind12], linewidth=3.0, alpha=0.8)
							pl.semilogx(ells[exp][i_min:i_max],  np.abs(Nl[exp][components]['BB_CIB_delens_post_comp_sep'][i_min:i_max])*100/Cls_fid['BlBl'][i_min:i_max], linestyle='--', color=colors[ind12], linewidth=3.0, alpha=0.8)
							ind12 += 1
					pl.xlim([2, 2000])
					pl.ylim([-5.0, 105.0])
					#ax = pl.gca()
					if ind_subplot!=2: 
						pl.ylabel(r'$C_\ell^{BB,\,delens}/C_\ell^{BB,\,lens}\ [ \% ]$', fontsize=28)
						pl.xlabel('$\ell$', fontsize=28)

					for tick in ax.xaxis.get_major_ticks():
						tick.label.set_fontsize(18)
					for tick in ax.yaxis.get_major_ticks():
						tick.label.set_fontsize(18)
					ax = pl.gca()
					ax.grid(True)
				ind2 +=1
			ind1 += 1
		pl.savefig('../article/Delensing_performances_experiments_w_wo_comp_sep_v08142015_sIII.pdf')
		pl.show()
		exit()
		'''
	if combo_power_spectrum:

		# need different options for different experiment groups
		print ' DOING THE FOREGROUNDS & DELENSING FIGURE... '
		if ('Simons Array' in experiments or 'EBEX10K' in experiments or \
			'Planck' in experiments):
			stage = 'III'
		else:
			stage = 'IV'
		print 'plotting stage {:} experiments'.format(stage)
		
		# use generic labelling?
		generic = True

		# information for foreground curves
		ells_fg = Cls_fid['ell']
		i_min_fg = np.argmin(np.abs(Cls_fid['ell'] - 2))
		i_max_fg = np.argmin(np.abs(Cls_fid['ell'] - 3000))
		A353 = residuals_comp.A_element_computation(353, 1.59, -3.1, squeeze=True)
		A353 /= residuals_comp.BB_factor_computation(353)
		A70 = residuals_comp.A_element_computation(70, 1.59, -3.1, squeeze=True)
		A70 /= residuals_comp.BB_factor_computation(70)
		s70_dust = A70[1]/A353[1]
		
		# plot! do both individual and combined plots
		colors = [ 'OrangeRed', 'Orange', 'DarkRed' ]
		n_exp = len(experiments)
		n_combo = n_exp * (n_exp + 1) / 2
		fig, axes = pl.subplots(1, n_combo, sharey=True, figsize=(n_combo*4,4), \
								facecolor='w', edgecolor='k')


		ind1 = 1
		for exp1 in experiments:
			ind2 = 1
			for exp2 in experiments:
				
				if exp1 == exp2 :
					exp = exp1
					sp_ind = ind1-1
					loc = configurations[exp]['loc']
				else:
					exp = exp1+' x '+exp2
					sp_ind = ind1*(2*n_exp-ind1+1)/2+ind2-ind1-1
					loc = locs['cross']
				
				# do not re-plot cross correlations
				if ind2 >= ind1:

					# generate f_sky-specific foreground curves to overplot
					ind_fsky = np.argmin(np.abs(configurations[exp]['fsky'] - \
												np.array(fskys_int)/100.0))
					Cl_dust_70GHz_fsky_exp = (AEE_Planck_int[ind_fsky] * s70_dust ** 2 / 2) * \
											 (ells_fg * 1.0 / 80.0) ** (-0.4)
					Cl_sync_70GHz_fsky_exp = Cl_dust_70GHz_fsky_exp[198] * \
											 (ells_fg * 1.0 / 200.0) ** (-0.6)
					fig_i = pl.figure(figsize=(6,5), facecolor='w', edgecolor='k')
					ax_i = fig_i.gca()
					for ax in [ax_i, axes[sp_ind]]:

						i_min = configurations[exp]['ell_min'] - ell_min_camb
						i_max = configurations[exp]['ell_max'] - ell_min_camb + 1
						ax.loglog(Cls_fid['ell'], Cls_fid['BlBl'], \
								  linewidth=3.0, alpha=0.7, color='k', \
								  linestyle = '-')
						ax.fill_between(Cls_fid['ell'], 1.0e-10, \
										Cls_fid['BuBu']*0.01/params_fid_v[0]['r'], \
										linewidth=3.0, alpha=0.7, color='darkgrey')
						ax.loglog(ells_fg[i_min_fg:i_max_fg], \
								  Cl_dust_70GHz_fsky_exp[i_min_fg:i_max_fg] + \
								  Cl_sync_70GHz_fsky_exp[i_min_fg:i_max_fg], \
								  linewidth=3.0, alpha=0.7, color='k', \
								  linestyle='--', dashes = [4, 2])
						for components in components_v:
							if components == 'cmb-only':
								ax.loglog(ells[exp][i_min:i_max], \
										  Nl[exp][components]['BB'][i_min:i_max], \
										  color='k', linewidth=3.0, linestyle='--', \
										  dashes = [10, 2])
							elif components == 'sync+dust':
								ax.loglog(ells[exp][i_min:i_max], \
										  foregrounds[exp][components]['Cl_res'][i_min:i_max], \
										  color=colors[2], linewidth=3.0, alpha=0.8, \
										  linestyle='--', dashes = [4, 2])
								ax.loglog(ells[exp][i_min:i_max], \
										  Nl[exp][components]['BB_post_comp_sep'][i_min:i_max], \
										  color=colors[2], linewidth=3.0, alpha=0.8, \
										  linestyle='--', dashes = [10, 2])
								if stage == 'III':
									ax.loglog(ells[exp][i_min:i_max], \
											  Nl[exp][components]['BB_CIB_delens_post_comp_sep'][i_min:i_max], \
											  color=colors[2], linewidth=3.0, alpha=0.8, \
											  linestyle='-')
								else:
									ax.loglog(ells[exp][i_min:i_max], \
											  Nl[exp][components]['BB_delens_post_comp_sep'][i_min:i_max], \
											  color=colors[2], linewidth=3.0, alpha=0.8, \
											  linestyle='-')
						if generic:
							if loc == 1:
								exp_lab = 'Ground'
							elif loc == 2:
								exp_lab = 'Balloon'
							elif loc == 3:
								exp_lab = 'Space'
							elif loc == 4:
								exp_lab = 'Cross'
						else:
							if len(exp) > 20:
								exp_lab = exp.replace(' x ', ' x\n')
							else:
								exp_lab = exp
						ax.set_title(exp_lab, fontsize=22)
						ax.set_xlim([2, 2000])
						ax.set_ylim([1e-8, 10])
						ax.tick_params(axis='both', which='major', labelsize=16)
						ax.tick_params(axis='both', which='minor', labelsize=16)
						ax.grid(True)
					ax_i.set_xlabel(r'$\ell$', fontsize = 22)
					ax_i.set_ylabel(r'$\ell(\ell+1) C_\ell^{BB} / 2\pi \ [\mu {{\rm K}}^2]$', \
									fontsize = 22)

					box = ax_i.get_position()
					ax_i.set_position([box.x0, box.y0, box.width, box.height*0.8])

					fig_i.savefig('../article/combined_FG+DL_C_l_performance_' + \
								  exp.replace(' ', '_') + '.pdf', \
								  bbox_inches='tight')
					fig_i.show()
					pl.close(fig_i)
				ind2 +=1
			ind1 += 1
		axes[1].set_xlabel(r'$\ell$', fontsize = 22)
		axes[0].set_ylabel(r'$\ell(\ell+1) C_\ell^{BB} / 2\pi \ [\mu {{\rm K}}^2]$', \
						   fontsize = 22)


		fig.subplots_adjust(wspace = 0)

		# building the legend
		pl.plot(1e-6, 1e-6, linewidth=3.0, alpha=0.7, color='k', linestyle = '-', \
                label=r'lensing $B$ modes')
		pl.plot(1e-6, 1e-6, linewidth=6.0, alpha=0.7, color='darkgrey', \
                label=r'primordial $B$ modes ($r \leq 10^{-3}$)')
		pl.plot(1e-6, 1e-6, linewidth=3.0, alpha=0.7, color='k', linestyle='--', \
                dashes = [4, 2], label=r'$C_\ell^{\rm{dust+sync}}$ @ 70GHz')
		pl.plot(1e-6, 1e-6, color='k', linewidth=3.0, linestyle='--', \
                dashes = [10, 2], label=r'raw $N_\ell^{BB}$')
		pl.plot(1e-6, 1e-6, color=colors[2], linewidth=3.0, alpha=0.8, \
                linestyle='--', dashes = [4, 2], label=r'foreground residuals')
		pl.plot(1e-6, 1e-6, color=colors[2], linewidth=3.0, alpha=0.8, \
                linestyle='--', dashes = [10, 2], \
                label=r'post-cleaning $N_\ell^{BB}$')
		pl.plot(1e-6, 1e-6, color=colors[2], linewidth=3.0, alpha=0.8, \
                linestyle='-', label=r'delensing residuals')

		# add the legend if stage-4
		if stage == 'IV':
			legend = pl.gca().legend(loc='lower center', bbox_to_anchor=(-0.5, -0.7), ncol=2, prop={'size':14})

		# saving on disk
		fig.savefig('../article/combined_FG+DL_C_l_performance_S{:}.pdf'.format(stage), \
					bbox_inches='tight')
		pl.show()
		exit()

	#################################################################################################################################################################################
	#################################################################################################################################################################################
	##############
	## FORECASTING

	# loop through base cosmologies
	sigmas = {}
	ind_pfid = 0
	for params_fid_loc in params_fid_v:

		# compute fiducial C_ls if not already pre-computed
		print '################################ computing fiducial Cls ... ####################################'
		'''
		name_fid = 'fidCls'
		for p in range(len(params_fid_loc.keys())):
			name_fid += '_'+str(params_fid_loc.keys()[p])+'_'+str(params_fid_loc[params_fid_loc.keys()[p]] )
		print 'looking for ', name_fid+'.pkl'
		fnames_fid = glob.glob( name_fid+'.pkl' )
		'''

		name_fid = 'fidCls'
		fnames_fid = glob.glob( os.path.join('/global/homes/j/josquin/FORECAST/self_consistent_forecast/codes/',name_fid+'*.pkl' ))

		output = {}
		for file_ in fnames_fid:
			output[file_] = 0
		for p in range(len(params_fid_loc.keys())):
			name_fid = '_'+str(params_fid_loc.keys()[p])+'_'+str(params_fid_loc[params_fid_loc.keys()[p]] )+'_'
			for file_ in fnames_fid:
				if name_fid in file_:
					output[file_] += 1
		fnames_fid = [max(output.iteritems(), key=operator.itemgetter(1))[0]]

		if output[fnames_fid[0]] < len(params_fid_loc.keys())-1 :
			fnames_fid = []

		if not fnames_fid:
			print '################# computing Cls file because it does not seem to be on disk #######################'
			Cls_fid_loc=python_camb_self_consistent.submit_camb( h=params_fid_loc['h'], ombh2=params_fid_loc['ombh2'], omch2=params_fid_loc['omch2'], omnuh2=params_fid_loc['omnuh2'], omk=params_fid_loc['omk'], YHe=params_fid_loc['YHe'], Neff=params_fid_loc['Neff'], w=params_fid_loc['w'], wa=params_fid_loc['wa'], tau=params_fid_loc['tau'],As=params_fid_loc['As'], ns=params_fid_loc['ns'], alphas=params_fid_loc['alphas'], nT=params_fid_loc['nT'], r=params_fid_loc['r'], k_scalar=params_fid_loc['k_scalar'] , k_tensor=params_fid_loc['k_tensor'], eta=1.0, exe = camb)
			save_obj('./', name_fid, Cls_fid_loc)
		else:
			print '################################ loading already existing Cls file ####################################'
			Cls_fid_loc = load_obj('/global/homes/j/josquin/FORECAST/self_consistent_forecast/codes/', fnames_fid[0])

		# compute derivatives once for all parameters
		dCldp = fc.derivatives_computation(Cls_fid_loc, params_dev_full, params_fid_loc, information_channels, exe = camb)

		# loop through experimental combinations
		ind = -1
		for exp1 in experiments:
			ind += 1
			for exp2 in experiments[ind:]:
				
				if exp1 == exp2 :
					exp = exp1
					if cross_only: continue
				else:
					exp = exp1+' x '+exp2

				## find the exp which has the largest fsky <-- we want to combine fisher matrices outside the overlapping fsky
				if configurations[exp1]['fsky'] != configurations[exp2]['fsky']:
					#####
					if configurations[exp1]['fsky'] > configurations[exp2]['fsky']:
						option_Fisher_combined_on_exp1 = True
						option_Fisher_combined_on_exp2 = False
					else:
						option_Fisher_combined_on_exp1 = False						
						option_Fisher_combined_on_exp2 = True
					#####
				else:
					option_Fisher_combined_on_exp1 = False
					option_Fisher_combined_on_exp2 = False

				## inputs for F'', the fisher matrix evaluated over the lowest ells
				if exp1 != exp2 :
					if configurations[exp1]['ell_min'] < configurations[exp2]['ell_min']:
						delta_ell_for_exp1 = configurations[exp2]['ell_min']-configurations[exp1]['ell_min']
						delta_ell_for_exp2 = 0
					elif configurations[exp1]['ell_min'] > configurations[exp2]['ell_min']:
						delta_ell_for_exp1 = 0
						delta_ell_for_exp2 = configurations[exp1]['ell_min']-configurations[exp2]['ell_min']
					else:
						delta_ell_for_exp1 = 0
						delta_ell_for_exp2 = 0
				else:
					delta_ell_for_exp1 = 0
					delta_ell_for_exp2 = 0

				# loop over delensing and component-separation scenarios
				if ind_pfid==0: sigmas[exp]={}

				###########################################
				# loops initiating sigmas 
				for components in components_v:

					if ind_pfid==0: sigmas[exp][components] = {}

				for components in components_v:

					for delensing in delensing_option_v:

							## building label
							if delensing != '':
								label = ' iterative delensing '
							else:
								label = ' no delensing '

							if delensing=='CMBxCMB':
								label += ' CMBxCMB '
							elif delensing=='CMBxCIB':
								label += ' CMBxCIB '
							elif delensing=='CMBxLSS':
								label += ' CMBxLSS '

							if components!='cmb-only' :
								label += ' + post-comp-sep '
							else:
								label += ' no comp-sep '

							if ind_pfid==0: sigmas[exp][components][label] = {}

				###########################################
				# DELENSING 
				for delensing in delensing_option_v:

					###########################################
					# FOREGROUNDS
					for components in components_v:

						## building label
						if delensing != '':
							label = ' iterative delensing '
						else:
							label = ' no delensing '

						if delensing=='CMBxCMB':
							label += ' CMBxCMB '
						elif delensing=='CMBxCIB':
							label += ' CMBxCIB '
						elif delensing=='CMBxLSS':
							label += ' CMBxLSS '

						if components!='cmb-only' :
							label += ' + post-comp-sep '
						else:
							label += ' no comp-sep '

						print 'label is', label

						###########
						if option_Fisher_combined_on_exp1 :
							exp_loc_v = [ exp, exp1 ]
						elif option_Fisher_combined_on_exp2:
							exp_loc_v = [ exp, exp2 ]
						else:
							exp_loc_v = [ exp ]

						if delta_ell_for_exp2:
							exp_loc_v.append( 'exp2_low_ell' )
						elif delta_ell_for_exp1:
							exp_loc_v.append( 'exp1_low_ell' )

						## loop over experiments for which we can combine fisher matrices afterwards.
						indfisher = 0
						for exp_loc_loop in exp_loc_v:

							#####
							if exp_loc_loop == 'exp1_low_ell':
								exp_loc = exp1
							elif exp_loc_loop == 'exp2_low_ell':
								exp_loc = exp2
							else:
								exp_loc = exp_loc_loop
							#####

							if (exp_loc == exp) or (exp_loc_loop == 'exp1_low_ell') or (exp_loc_loop == 'exp2_low_ell'):
								fsky_loc = configurations[exp_loc]['fsky']
							else:
								fsky_loc = np.abs(configurations[exp1]['fsky'] - configurations[exp2]['fsky'])

							# build local Nl and Cls_fid (delensing, LSS, comp. sep, etc.)
							Nl_loc = {}
							if components!='cmb-only' :
								## for the post comp sep cases
								if exp_loc == exp:
									Nl_loc['TT'], Nl_loc['EE'], Nl_loc['BB'] = \
									  Nl[exp_loc][components]['TT_post_comp_sep']*1.0, \
									  Nl[exp_loc][components]['EE_post_comp_sep']*1.0, \
									  Nl[exp_loc][components]['BB_post_comp_sep']*1.0
								else:
									Cl_fgs_res_fsky_correction = (fsky_loc/configurations[exp_loc]['fsky'] - 1 )*foregrounds[exp_loc][components]['Cl_res']
									ind0=np.argmin(np.abs(Cls_fid['ell'] - np.min(ells[exp_loc])))
									ind1=np.argmin(np.abs(Cls_fid['ell'] - np.max(ells[exp_loc])-1))
									Nl_loc['TT'], Nl_loc['EE'], Nl_loc['BB'] = \
									  Nl[exp_loc][components]['TT_post_comp_sep']*1.0 + Cl_fgs_res_fsky_correction[ind0:ind1], \
									  Nl[exp_loc][components]['EE_post_comp_sep']*1.0 + Cl_fgs_res_fsky_correction[ind0:ind1], \
									  Nl[exp_loc][components]['BB_post_comp_sep']*1.0 + Cl_fgs_res_fsky_correction[ind0:ind1]

								if delensing=='CMBxLSS':
									Nl_loc['dd'] = Nl[exp_loc][components]['dd_lss_post_comp_sep']*1.0
									Cls_fid_loc['BB_delens'] = Nl[exp_loc][components]['BB_lss_delens_post_comp_sep']*1.0
								elif delensing=='CMBxCIB':
									Nl_loc['dd'] = Nl[exp_loc][components]['dd_CIB_post_comp_sep']*1.0
									Cls_fid_loc['BB_delens'] = Nl[exp_loc][components]['BB_CIB_delens_post_comp_sep']*1.0
								elif delensing=='CMBxCMB':
									Nl_loc['dd'] = Nl[exp_loc][components]['dd_post_comp_sep']*1.0
									Cls_fid_loc['BB_delens'] = Nl[exp_loc][components]['BB_delens_post_comp_sep']*1.0									
								else:
									Nl_loc['dd'] = Nl[exp_loc][components]['dd_post_comp_sep']*1.0
									Cls_fid_loc['BB_delens'] = Cls_fid_loc['BlBl']*1.0
							else:
								## without post comp sep
								Nl_loc['TT'], Nl_loc['EE'], Nl_loc['BB'] = \
								  Nl[exp_loc][components]['TT']*1.0, \
								  Nl[exp_loc][components]['EE']*1.0, \
								  Nl[exp_loc][components]['BB']*1.0
								if delensing=='CMBxLSS':
									Nl_loc['dd'] = Nl[exp_loc][components]['dd_lss']*1.0
									Cls_fid_loc['BB_delens'] = Nl[exp_loc][components]['BB_lss_delens']*1.0
								elif delensing=='CMBxCIB':
									Nl_loc['dd'] = Nl[exp_loc][components]['dd_CIB']*1.0
									Cls_fid_loc['BB_delens'] = Nl[exp_loc][components]['BB_CIB_delens']*1.0
								elif delensing=='CMBxCMB':
									Nl_loc['dd'] = Nl[exp_loc][components]['dd']*1.0
									Cls_fid_loc['BB_delens'] = Nl[exp_loc][components]['BB_delens']*1.0
								else:
									Nl_loc['dd'] = Nl[exp_loc][components]['dd']*1.0
									Cls_fid_loc['BB_delens'] = Cls_fid_loc['BlBl']*1.0

							if no_lensing:
								Cls_fid_loc['BB_delens'] = Cls_fid_loc['BB_delens']*0.0

							# compute the alpha factor from Sherwin et al.
							delens_factor=np.sum( Cls_fid_loc['BB_delens'][configurations[exp_loc]['ell_min']-2:configurations[exp_loc]['ell_max']-1] ) / \
										np.sum( Cls_fid_loc['BlBl'][configurations[exp_loc]['ell_min']-2:configurations[exp_loc]['ell_max']-1] )
							print 'delensing factor = ', delens_factor
							if delens_factor != delens_factor:
								print 'Cl_del = ', Cls_fid_loc['BB_delens'][configurations[exp_loc]['ell_min']-2:configurations[exp_loc]['ell_max']-1]
								print 'Cl_fid = ', Cls_fid_loc['BlBl'][configurations[exp_loc]['ell_min']-2:configurations[exp_loc]['ell_max']-1]
									
							# setting the delensing_option for the covariance matrix computation
							if delensing=='CMBxCMB' or delensing=='CMBxCIB' or delensing=='CMBxLSS': 
								delensing_option=1
							else:
								delensing_option=0

							# calculate Fisher covariance matrix once per experiment
							Cov, Cov_inv = fc.Cov_computation(information_channels, Cls_fid_loc, Nl_loc, configurations[exp_loc]['ell_min'], configurations[exp_loc]['ell_max'], delensing=delensing_option)

							## marginalization over foregrounds residuals amplitude
							if components!='cmb-only' :
								# add A_fgs_res in the parameters to be marginalized over
								params_dev_full_loc = copy.copy(params_dev_full)
								params_dev_full_loc.append('A_fgs_res')
								params_dev_full_loc.append('b_fgs_res')
								params_dev_loc = copy.deepcopy( params_dev_v )
								for m in range(len(params_dev_loc)):
									params_dev_loc[m].append('A_fgs_res')
									params_dev_loc[m].append('b_fgs_res')

								# update fiducial value for b_fgs_res (this depends on the actual shape of foregrounds residuals)
								def residuals_ell_dependence( ell, p0, p1 ):
									return p0 * (ell**p1)
								def logl_model_b( p ):
									A, b = 1.0, p
									return logl( residuals_ell_dependence( Cls_fid['ell'], A, b ), foregrounds[exp_loc][components]['Cl_res'] )							
								params_fid_loc['b_fgs_res'] = optimize.fmin( logl_model_b, x0=-2.0 )

								# rescale of derivatives for residuals amplitude
								dCldp['A_fgs_res'] = {}
								dCldp['b_fgs_res'] = {}
								for key_loc in dCldp['ns'].keys():
									if 'd' not in key_loc:
										dCldp['A_fgs_res'][key_loc] = foregrounds[exp_loc][components]['Cl_res'][0]*(Cls_fid['ell']/Cls_fid['ell'][0])**( params_fid_loc['b_fgs_res'] )
										dCldp['b_fgs_res'][key_loc] = foregrounds[exp_loc][components]['Cl_res'][0]*params_fid_loc['A_fgs_res']*np.log(Cls_fid['ell']/Cls_fid['ell'][0])*(Cls_fid['ell']/Cls_fid['ell'][0])**( params_fid_loc['b_fgs_res'] )
									else:
										dCldp['A_fgs_res'][key_loc] = Cls_fid['ell']*0
										dCldp['b_fgs_res'][key_loc] = Cls_fid['ell']*0
							else:
								params_dev_full_loc= copy.copy( params_dev_full )
								params_dev_loc = copy.copy( params_dev_v )

							# calculate Fisher matrix for all parameters
							if indfisher == 0: 
								F = fc.Fisher_computation(Cov_inv, dCldp, params_dev_full_loc, fsky_loc, information_channels, configurations[exp_loc]['ell_min'], configurations[exp_loc]['ell_max'], Cls_fid_loc, params_fid_loc, Cov, DESI_BAO=DESI)
							else: 
								# if there is an iteration over fisher matrices, we add them todether.
								if (exp_loc_loop != 'exp2_low_ell') and (exp_loc_loop!='exp1_low_ell') :
									print 'COMBINATION OF 2 FISHER MATRICES ! '
									F += fc.Fisher_computation(Cov_inv, dCldp, params_dev_full_loc, fsky_loc, information_channels, configurations[exp_loc]['ell_min'], configurations[exp_loc]['ell_max'], Cls_fid_loc, params_fid_loc, Cov, DESI_BAO=DESI )
								else:
									print 'adding a low ell fisher for exp = '
									if exp_loc_loop == 'exp2_low_ell':
										print exp2, ' which has a ell_min = ',configurations[exp2]['ell_min']
										print 'the delta ell is = ', delta_ell_for_exp2
										print ' and fsky = ', fsky_loc
										F += fc.Fisher_computation(Cov_inv, dCldp, params_dev_full_loc, fsky_loc, information_channels, configurations[exp2]['ell_min'], configurations[exp2]['ell_min']+delta_ell_for_exp2, Cls_fid_loc, params_fid_loc, Cov, DESI_BAO=DESI )
									elif exp_loc_loop == 'exp1_low_ell':
										print exp1, ' which has a ell_min = ',configurations[exp1]['ell_min']
										print 'the delta ell is = ', delta_ell_for_exp1
										print ' and fsky = ', fsky_loc
										F += fc.Fisher_computation(Cov_inv, dCldp, params_dev_full_loc, fsky_loc, information_channels, configurations[exp1]['ell_min'], configurations[exp1]['ell_min']+delta_ell_for_exp1, Cls_fid_loc, params_fid_loc, Cov, DESI_BAO=DESI )
									else:
										print 'something is wrong :( '
										exit()
							indfisher +=1

						###########

						# initiate output
						if ind_pfid==0: sigmas[exp][components][label]['marginalized'] = {}
						if ind_pfid==0: sigmas[exp][components][label]['conditional'] = {}
						for m in range(len(params_dev_loc)):
							for p in params_dev_loc[m]:
								if ind_pfid==0: sigmas[exp][components][label]['marginalized'][p] = np.zeros((len(params_fid_v), len(params_dev_loc)))
								if ind_pfid==0: sigmas[exp][components][label]['conditional'][p] = np.zeros((len(params_fid_v), len(params_dev_loc)))

						# loop through the models we want to investigate
						ind_pdev = 0
						for m in range(len(params_dev_loc)):

							# extract relevant Fisher submatrix
							F_loc = fc.Fisher_submatrix(F, params_dev_full_loc, params_dev_loc[m])

							# add in prior information on a model-by-model basis
							if param_priors_v:
								param_priors_v_loc	= [ params_fid_prior_ext[p] for p in params_dev_full_loc ]
								prior_full_matrix = np.diag( 1.0/np.array(param_priors_v_loc )**2 )
								prior_matrix_loc = fc.Fisher_submatrix(prior_full_matrix, params_dev_full_loc, params_dev_loc[m])
								F_loc += prior_matrix_loc

							# invert Fisher submatrix and extract errors
							F_loc_inv = np.linalg.inv(F_loc)
							
							'''
							pl.figure()
							for j in dCldp['r'].keys():
								pl.semilogx( dCldp['r'][j], label=j)
							pl.legend()
							#pl.loglog(Cls_fid_loc['BuBu']/params_fid_loc['r'], 'k--' )
							pl.show()
							exit()
							'''

							sigmas_loc = {}
							sigmas_diag_loc = {}
							for p in range(len(params_dev_loc[m])):
								sigmas_loc[params_dev_loc[m][p]] = np.sqrt(F_loc_inv[p, p])
								sigmas_diag_loc[params_dev_loc[m][p]] = 1.0 / np.sqrt(F_loc[p, p])

							print '___________________________________________________________'
							print exp, ' | del: ', delensing, ' | components: ',components
							print 'params_dev_v=', params_dev_loc[m]
							print 'params_fid_loc=', params_fid_loc
							print '      _____    ' 
							print '        |      ' 
							print '        V      ' # nice drawing eh?!  
							#print sigmas[exp][components].keys()
							for p in params_dev_loc[m]:
								sigmas[exp][components][label]['marginalized'][p][ind_pfid, ind_pdev] = sigmas_loc[p]
								print '---->>> marginalized $\sigma$(',p,'=',params_fid_loc[p],') = ',sigmas[exp][components][label]['marginalized'][p][ind_pfid, ind_pdev]
								sigmas[exp][components][label]['conditional'][p][ind_pfid, ind_pdev] = sigmas_diag_loc[p]
								#if (exp=='COrE+') and (components=='cmb-only') and (p=='nT'):
								#	exit()
							# test: report a quick BB-only r constraint
							if 'r' in params_dev_loc[m]:
								i_min = configurations[exp]['ell_min']-ell_min_camb
								i_max = configurations[exp]['ell_max']-ell_min_camb
								sigmas[exp][components][label]['BB-only sigma_r'] = \
								  1.0/np.sqrt(np.sum((2.0 * ells[exp][i_min:i_max+1] + 1) * \
													 configurations[exp]['fsky'] / 2.0 * \
													 (Cls_fid_loc['BuBu'][i_min:i_max+1] / \
													  (Cls_fid_loc['BuBu'][i_min:i_max+1] + \
													   Cls_fid_loc['BB_delens'][i_min:i_max+1] + \
													  Nl_loc['BB'][i_min:i_max+1])) ** 2)) * \
													 params_fid_loc['r']
								sigmas[exp][components][label]['BB-only sigma_r'] = sigmas[exp][components][label]['BB-only sigma_r']
								print 'BB-only sigma_r: ', sigmas[exp][components][label]['BB-only sigma_r']

							# clean up
							del F_loc, F_loc_inv, sigmas_loc, sigmas_diag_loc
						
							# increment index of marginalized parameters
							ind_pdev += 1

						# clean up
						del Nl_loc, F, label, Cov, Cov_inv, delensing_option

						del params_dev_loc, params_dev_full_loc
		# increment index of cosmologies
		ind_pfid += 1

	del params_fid_loc

	return foregrounds, sigmas, Nl, Cls_fid


############################################################################################################################################################################################################


#####################################################
##           command-line options parser           ##
#####################################################
def grabargs():

	# check whether user has specified a path to CAMB
	parser = argparse.ArgumentParser()
	parser.add_argument("--camb", \
						help = "full path of CAMB executable (default = ./camb)", \
						default = "./camb")
	parser.add_argument("--fgs_scatter", action='store_true',\
						help = "stops at the foregrounds section and produce the foregrounds scatter article figure", \
						default = False)
	parser.add_argument("--fgs_power_spectrum", action='store_true',\
						help = "stops at the foregrounds section and produce the foregrounds power spectrum article figure", \
						default = False)
	parser.add_argument("--delens_scatter", action='store_true',\
						help = "stops at the delensing section and produces the delensing scatter article figure", \
						default = False)
	parser.add_argument("--delens_power_spectrum", action='store_true',\
						help = "stops at the delensing section and produces the delensing power spectrum article figure", \
						default = False)
	parser.add_argument("--combo_power_spectrum", action='store_true',\
						help = "stops at the delensing section and produces combined foreground and delensing power spectrum figure", \
						default = False)
	parser.add_argument("--fgs_vs_freq_vs_ell", action='store_true',\
						help = "stops at the foregrounds section and produce the article fgs/BB vs ell vs frequency figure", \
						default = False)
	parser.add_argument("--power_spectrum_figure", action='store_true',\
						help = "stops at the foregrounds section and produce the article power spectrum figure showing BB and input foregrounds", \
						default = False)
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
	args = parser.parse_args()
	if not os.path.isfile(args.camb):
		raise IOError("specified camb executable (" + args.camb + ") does not exist")
		
	return args

if __name__ == "__main__":

	initialize()

