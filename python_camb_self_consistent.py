#!/usr/bin/env python

'''
Grab spectra from a CAMB run
given a fiducial cosmology
'''

import os
import numpy as np
import pylab as pl
import random
import string
import time

def params_ini_edit(output_root, l_max_scalar, k_eta_max_scalar,l_max_tensor,k_eta_max_tensor,h,ombh2,omch2,omnuh2,omk,YHe,Neff,w,wa,tau,As,ns,alphas,nT,r,k_scalar,k_tensor,lensing_z_max,rand_string):

	global_string = '''
output_root = '''+output_root+'''
get_scalar_cls = T
get_vector_cls = F
get_tensor_cls = T
get_transfer = F
do_lensing = T
do_nonlinear = 3
l_max_scalar = '''+str(l_max_scalar)+'''
k_eta_max_scalar = '''+str(k_eta_max_scalar)+'''
l_max_tensor = '''+str(l_max_tensor)+'''
k_eta_max_tensor = '''+str(k_eta_max_tensor)+'''
use_physical = T
hubble = '''+str(h)+'''
temp_cmb = 2.7255
ombh2 = '''+str(ombh2)+'''
omch2 = '''+str(omch2)+'''
omnuh2 = '''+str(omnuh2)+'''
omk = '''+str(omk)+'''
nu_mass_eigenstates = 1
nu_mass_degeneracies = 1
nu_mass_fractions = 1
share_delta_neff = T
helium_fraction = '''+str(YHe)+'''
massless_neutrinos = '''+str(Neff-1)+'''
massive_neutrinos = 1
w = '''+str(w)+'''
wa = '''+str(wa)+'''
cs2_lam = 1
reionization = T
re_use_optical_depth = T
re_optical_depth = '''+str(tau)+'''
re_redshift = 11
re_delta_redshift = 0.5
re_ionization_frac = -1
initial_power_num = 1
scalar_amp(1) = '''+str(As)+'''
scalar_spectral_index(1) = '''+str(ns)+'''
scalar_nrun(1) = '''+str(alphas)+'''
scalar_nrunrun(1) = 0.0
tensor_spectral_index(1) = '''+str(nT)+'''
initial_ratio(1) = '''+str(r)+'''
initial_condition = 1
initial_vector = -1 0 0 0 0
vector_mode = 0
COBE_normalize = F
CMB_outputscale = 7.42835025e12
pivot_scalar = '''+str(k_scalar)+'''
pivot_tensor = '''+str(k_tensor)+'''
transfer_high_precision = T
transfer_interp_matterpower = T
transfer_kmax = 2
transfer_k_per_logint = 0
transfer_num_redshifts = 1
transfer_redshift(1) = 0
feedback_level = 0
lensing_method = 1
lensing_z_max = '''+str(lensing_z_max)+'''
accurate_BB = T
recombination=1
accurate_reionization = T
do_tensor_neutrinos = T
massive_nu_approx = 0
accurate_polarization = T
do_late_rad_truncation = T
RECFAST_fudge = 1.14
RECFAST_fudge_He = 0.86
RECFAST_Heswitch = 6
RECFAST_Hswitch = T
accuracy_boost = 2
l_accuracy_boost = 1
l_sample_boost = 1
high_accuracy_default=T
scalar_output_file = scalcls_'''+rand_string+'''.dat
vector_output_file = vectcls_'''+rand_string+'''.dat
tensor_output_file = tenscls_'''+rand_string+'''.dat
total_output_file  = totcls_'''+rand_string+'''.dat
lensed_output_file = lensedcls_'''+rand_string+'''.dat
lensed_total_output_file = lensedtotcls_'''+rand_string+'''.dat
lens_potential_output_file = lenspotentialcls_'''+rand_string+'''.dat
#FITS_filename      = scalcls_'''+rand_string+'''.fits
#transfer_filename(1)    = transfer_out_z0_'''+rand_string+'''.dat
#transfer_matterpower(1) = matterpower_z0_'''+rand_string+'''.dat
do_lensing_bispectrum = F
do_primordial_bispectrum = F
derived_parameters = T
number_of_threads = 0
'''
	return global_string


#def submit_camb(output_root='test', l_max_scalar=5000, k_eta_max_scalar=35000,l_max_tensor=6000, k_eta_max_tensor=35000,h=70,ombh2=0.0226,omch2=0.114,omnuh2=0,omk=0,YHe=0.24,Neff=3.04,w=-1,wa=0.0,tau=0.09,As=2.46e-9,ns=0.96,alphas=0,nT=-0.01,r=0.2, k_scalar=0.005, k_tensor=0.005, eta=1.0, exe = './camb'):
def submit_camb(output_root='test', l_max_scalar=4100, k_eta_max_scalar=20000,l_max_tensor=4100, k_eta_max_tensor=20000,h=70,ombh2=0.0226,omch2=0.114,omnuh2=0,omk=0,YHe=0.24,Neff=3.04,w=-1,wa=0.0,tau=0.09,As=2.46e-9,ns=0.96,alphas=0,nT=-0.01,r=0.2, k_scalar=0.005, k_tensor=0.005, eta=1.0, lensing_z_max = -1.0, exe = './camb'):

	# generate random number for output files
	rand_string = ''.join( random.choice(string.ascii_uppercase + string.digits) for _ in range(10) )


	# create string of parameters
	global_string = params_ini_edit(output_root, l_max_scalar, k_eta_max_scalar,l_max_tensor,k_eta_max_tensor,h,ombh2,omch2,omnuh2,omk,YHe,Neff,w,wa,tau,As,ns,alphas,nT,r,k_scalar,k_tensor,lensing_z_max,rand_string)

	# save it to a .ini file
	text_file = open("./params_camb_"+rand_string+".ini", "w")
	text_file.write( global_string )
	text_file.close()

	# submit command to camb
	#cmd = '/Users/josquin1/Documents/Dropbox/postdoc\@LBL/PB_forecast/CAMB/camb/./camb ./params_camb.ini'
	print ' >>>>>>>> CAMB RUN LAUNCHED  '
	#SMF	cmd = '/Users/josquin1/Documents/Dropbox/camb/./camb ./params_camb_'+rand_string+'.ini'
	cmd = exe + ' ./params_camb_'+rand_string+'.ini'
	print 'cmd is : ', cmd
	os.system(cmd)

	print ' sleeping for 5 secs .. '
	time.sleep( 5.0 )

	# read output from CAMB
	'''
	dt = np.dtype( [ ('',int), ('',float), ('',float), ('',float), ('',float), ('',float), ('',float), ('',float)] ) 
	lenspotentialcls = np.fromfile('./test_lenspotentialcls_'+rand_string+'.dat', dtype=dt)
	print np.shape( lenspotentialcls )
	lenspotentialcls = lenspotentialcls.reshape((8, len(lenspotentialcls)/8)).T
	lensedcls = np.fromfile('./test_lensedcls_'+rand_string+'.dat', dtype=float)
	lensedcls = lensedcls.reshape((5, len(lensedcls)/5)).T
	totcls = np.fromfile('./test_totcls_'+rand_string+'.dat', dtype=float)
	totcls = totcls.reshape((5, len(totcls)/5)).T
	print np.shape(lenspotentialcls)
	print np.shape(lensedcls)
	print np.shape(totcls)
	'''
	lenspotentialcls = np.loadtxt('./test_lenspotentialcls_'+rand_string+'.dat' ) #, converters = {0: lambda s: ( float(s.replace("E+", "b")), float(s.replace("+", "E+")), float(s.replace("b", "E+")) ), 1: lambda s: ( float(s.replace("E+", "b")), float(s.replace("+", "E+")), float(s.replace("b", "E+")) ), 2: lambda s: ( float(s.replace("E+", "b")), float(s.replace("+", "E+")), float(s.replace("b", "E+")) ), 3: lambda s: ( float(s.replace("E+", "b")), float(s.replace("+", "E+")), float(s.replace("b", "E+")) ), 4: lambda s: ( float(s.replace("E+", "b")), float(s.replace("+", "E+")), float(s.replace("b", "E+")) ), 5: lambda s: ( float(s.replace("E+", "b")), float(s.replace("+", "E+")), float(s.replace("b", "E+")) ), 6: lambda s: ( float(s.replace("E+", "b")), float(s.replace("+", "E+")), float(s.replace("b", "E+")) ), 7: lambda s: ( float(s.replace("E+", "b")), float(s.replace("+", "E+")), float(s.replace("b", "E+")) )})
	lensedcls = np.loadtxt('./test_lensedcls_'+rand_string+'.dat' ) #, converters = {0: lambda s: ( float(s.replace("E+", "b")), float(s.replace("+", "E+")), float(s.replace("b", "E+")) ), 1: lambda s: ( float(s.replace("E+", "b")), float(s.replace("+", "E+")), float(s.replace("b", "E+")) ), 2: lambda s: ( float(s.replace("E+", "b")), float(s.replace("+", "E+")), float(s.replace("b", "E+")) ), 3: lambda s: ( float(s.replace("E+", "b")), float(s.replace("+", "E+")), float(s.replace("b", "E+")) ), 4: lambda s: ( float(s.replace("E+", "b")), float(s.replace("+", "E+")), float(s.replace("b", "E+")) ) })
	totcls = np.loadtxt('./test_totcls_'+rand_string+'.dat' ) #, converters = {0: lambda s: ( float(s.replace("E+", "b")), float(s.replace("+", "E+")), float(s.replace("b", "E+")) ), 1: lambda s: ( float(s.replace("E+", "b")), float(s.replace("+", "E+")), float(s.replace("b", "E+")) ), 2: lambda s: ( float(s.replace("E+", "b")), float(s.replace("+", "E+")), float(s.replace("b", "E+")) ), 3: lambda s: ( float(s.replace("E+", "b")), float(s.replace("+", "E+")), float(s.replace("b", "E+")) ), 4: lambda s: ( float(s.replace("E+", "b")), float(s.replace("+", "E+")), float(s.replace("b", "E+")) ) })
	lensedtotcls = np.loadtxt('./test_lensedtotcls_'+rand_string+'.dat' )

	# fill the different fields
	print l_max_tensor-1, lenspotentialcls.shape[0], lensedcls.shape[0], totcls.shape[0]
	nell = np.min([ l_max_tensor-1, lenspotentialcls.shape[0], lensedcls.shape[0], totcls.shape[0]])
	print '-----> nell is ', nell
	Cls = {}
	ell_v = lenspotentialcls[0:nell,0]

	## T, E, B unlensed 
	Cls['TuTu'] = lenspotentialcls[0:nell,1]
	Cls['EuEu'] = lenspotentialcls[0:nell,2]
	Cls['BuBu'] = lenspotentialcls[0:nell,3]
	Cls['TuEu'] = lenspotentialcls[0:nell,4]
	Cls['EuTu'] = Cls['TuEu'] 

	## T, E, B lensed
	Cls['TT'] = lensedcls[0:nell,1]
	Cls['EE'] = lensedcls[0:nell,2]
	Cls['BB'] = lenspotentialcls[0:nell,3]+eta*lensedcls[0:nell,3]
	Cls['TE'] = lensedcls[0:nell,4]
	Cls['ET'] = Cls['TE']

	## T, E, B lensed total (i.e., scalar + tensor)
	Cls['TT_tot'] = lensedtotcls[0:nell,1]
	Cls['EE_tot'] = lensedtotcls[0:nell,2]
	Cls['BB_tot'] = lensedtotcls[0:nell,3]
	Cls['TE_tot'] = lensedtotcls[0:nell,4]
	Cls['ET_tot'] = Cls['TE_tot']

	## T x d
	Cls['dd'] = lenspotentialcls[0:nell,5]
	#Cls['Td'] = lenspotentialcls[0:nell,6]
	Cls['Tud'] = lenspotentialcls[0:nell,6]
	#Cls['dT'] = Cls['Td']
	Cls['dTu'] = Cls['Tud']

	## E x d
	#Cls['dE'] = lenspotentialcls[0:nell,7]
	#Cls['Ed'] = Cls['dE']

	Cls['BlBl'] = eta*lensedcls[0:nell,3]
	Cls['ell'] = ell_v

	## B x d
	#Cls['Bd'] = np.sqrt( lensedcls[0:nell,3]*Cls['dd'] ) 
	#Cls['dB'] = Cls['Bd']
	'''
	print 'eta = ', eta
	pl.figure()
	pl.loglog( lenspotentialcls[0:nell,3] )
	pl.loglog(lensedcls[0:nell,3] )
	pl.figure()
	pl.loglog(Cls['ell'], Cls['TuTu'], label='TT')
	pl.loglog(Cls['ell'], Cls['TuEu'], label='TE')
	pl.loglog(Cls['ell'], Cls['EuEu'], label='EE')
	pl.loglog(Cls['ell'], Cls['BB'], label='BB')
	pl.loglog(Cls['ell'], Cls['BuBu'], label='BuBu for r='+str(r))
	pl.loglog(Cls['ell'], Cls['dd'], label='dd')
	pl.legend(loc='best')
	pl.show()
	exit()
	'''
	
	# if present, read deflection power spectrum assuming truncation
	# at some finite z_max
	if lensing_z_max > 0.0:
		Cls['ddzmax'] = lenspotentialcls[0:nell,8]
	
	
	return Cls

if __name__ == "__main__":

	submit_camb()

