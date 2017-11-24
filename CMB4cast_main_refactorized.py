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
import CMB4cast_utilities as CMB4U


############################################################################################################################################################################################################


#####################################################
##                core functionalities             ##
#####################################################
def core_function(configurations, components_v, camb, \
		params_fid_v, params_dev_v, information_channels, delensing_option_v, \
			delensing_z_max=-1.0, param_priors_v=[], cross_only=False, correlation_2_dusts=0.0, stolyarov=False, stolyarov_sync=False, cbass=False, quijote=False, \
					delens_command_line=False, comp_sep_only=False, calibration_error=0.0, np_nside=4, no_lensing=False, A_lens=1.0, \
						resolution=False, DESI=False, mpi_safe=False, path2maps='/Users/josquin1/Documents/Dropbox/planck_maps',\
							path2Cls='/Users/josquin1/Documents/Dropbox/self_consistent_forecast/codes/', spectral_parameters={}, \
								analytic_expr_per_template={}, bandpass_channels={}, drv={}, \
									prior_spectral_parameters={} ):
    
	################ entry checker

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
		Cls_fid=python_camb_self_consistent.submit_camb( h=params_fid_v['h'], ombh2=params_fid_v['ombh2'], omch2=params_fid_v['omch2'], \
			omnuh2=params_fid_v['omnuh2'], omk=params_fid_v['omk'], YHe=params_fid_v['YHe'], Neff=params_fid_v['Neff'], w=params_fid_v['w'], \
			wa=params_fid_v['wa'], tau=params_fid_v['tau'],As=params_fid_v['As'], ns=params_fid_v['ns'], alphas=params_fid_v['alphas'], nT=params_fid_v['nT'], \
			r=params_fid_v['r'], k_scalar=params_fid_v['k_scalar'] , k_tensor=params_fid_v['k_tensor'], eta=1.0, lensing_z_max=delensing_z_max, exe = camb)
		save_obj('./', name_fid, Cls_fid)
	else:
		print '################################ loading already existing Cls file ####################################'
		Cls_fid = load_obj('/global/homes/j/josquin/FORECAST/self_consistent_forecast/codes/', fnames_fid)

	if 'BuBu_r1' not in Cls_fid.keys():
		Cls_fid['BuBu_r1'] = Cls_fid['BuBu'] / params_fid_v['r']

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
			# Cls_fid_r = fc.derivatives_computation(Cls_fid, ['r'], params_fid, information_channels, exe=camb, path2Cls = path2Cls)['r']
			foregrounds[exp] = {}
			# foregrounds[exp] = CMB4cast_compsep.CMB4cast_compsep(configurations=configurations, components_v=components_v, exp=exp, \
							# cbass=cbass, quijote=quijote, np_nside=np_nside, ell_min_camb=ell_min_camb, Cls_fid=Cls_fid,\
							# stolyarov=stolyarov, stolyarov_sync=stolyarov_sync, calibration_error=calibration_error, Bd=Bd, \
							# Td=Td, Bs=Bs, camb=camb, ells_exp=ells[exp], path2maps=path2maps, path2Cls=path2Cls, Cls_fid_r=Cls_fid_r )
			foregrounds[exp] = CMB4cast_compsep.CMB4cast_compsep(configurations=configurations, components_v=components_v, exp=exp, \
									np_nside=np_nside, ell_min_camb=ell_min_camb, Cls_fid=Cls_fid, spectral_parameters=spectral_parameters, \
									analytic_expr_per_template=analytic_expr_per_template, camb=camb, drv=drv, \
									prior_spectral_parameters=prior_spectral_parameters, ells_exp=ells[exp], \
									path2files=path2maps, r_fid=params_fid_v[0]['r'] )
			# pissoffpython()

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

	parser.add_argument("--path_to_parameters", \
						help = "full path to the parameters file", \
						default ="/./CMB4cast_parameter_files.py")

	args = parser.parse_args()
	# if not os.path.isfile(args.path_to_parameters):
		# raise IOError("specified path to parameters (" + args.path_to_parameters + ") does not exist")
		
	return args

###########################################################
if __name__ == "__main__":

	args = grabargs()

	############
	# actual computation	

	## load parameter file
	import imp
	CMBP = imp.load_source('CMBP', args.path_to_parameters )

	configurations = {}
	configurations['my_exp'] = {}
	configurations['my_exp']['freqs'] = CMBP.frequencies
	configurations['my_exp']['uKCMBarcmin'] = CMBP.uKCMBarcmin
	configurations['my_exp']['FWHM'] = CMBP.FWHM
	configurations['my_exp']['fsky'] = CMBP.fsky
	configurations['my_exp']['bandpass'] = 0.3*np.ones(len(configurations['my_exp']['freqs']))
	configurations['my_exp']['ell_min'] = CMBP.ell_min
	configurations['my_exp']['ell_max'] = CMBP.ell_max
	configurations['my_exp']['alpha_knee'] = CMBP.alpha_knee
	configurations['my_exp']['ell_knee'] = CMBP.ell_knee

	foregrounds, sigmas, Nl, Cls_fid = core_function( configurations=configurations, \
			components_v=CMBP.sky_components, camb=CMBP.path_to_camb, params_fid_v=CMBP.params_fid, \
			params_dev_v=CMBP.params_dev, information_channels=CMBP.information_channels, \
			delensing_option_v=CMBP.delensing_option_v, delensing_z_max=3.5, \
			param_priors_v=CMBP.prior_spectral_parameters,\
			stolyarov=CMBP.stolyarov, stolyarov_sync=CMBP.stolyarov_sync,\
			cbass=CMBP.cbass, quijote=CMBP.quijote, \
			calibration_error=None,\
			path2maps=CMBP.path2maps, path2Cls=CMBP.path2Cls,\
			spectral_parameters=CMBP.spectral_parameters,\
			analytic_expr_per_template=CMBP.analytic_expr_per_template,\
			prior_spectral_parameters=CMBP.prior_spectral_parameters )

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
	
	############
	exit()

