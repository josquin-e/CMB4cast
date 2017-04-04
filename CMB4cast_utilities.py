#!/usr/bin/env python

'''
CMB4cast utility functions and settings
'''

import numpy as np
import healpy as hp
import os
import pickle
from math import log10, floor


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
		a  = round(x, sig-int(np.floor(np.log10(x)))-1)
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
    e = np.floor(np.log10(x))
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

#########################################
## COMPONENT SEPARATION RELATED QUANTITIES
# h/k
h_over_k =  0.0479924
# cst for CMB = h/kT
cst = 56.8
frequency_stokes_default = ['Q', 'U']


