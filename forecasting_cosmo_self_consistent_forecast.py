#!/usr/bin/env python

'''
Forecasting cosmological parameters constraints performances 
given CMB instrumental parameters


NB: any Cls shared between function are l(l+1)/2pi Cls, but they can be sometimes *locally* re-normalized.


Author: Josquin Errard
josquin.errard@gmail.com
'''

import sys
import os
import argparse
import scipy
from scipy import interpolate
import python_camb_self_consistent as python_camb
import numpy as np 
import pickle
import glob
#SMF import FUTURCMB_lensnoise
import smith_delensing_self_consistent_forecast
import pylab as pl

arcmin_to_radian = np.pi/(180.0*60.0)

params_fid = {}
params_fid['h'] = 70.0
params_fid['ombh2'] = 0.0226
params_fid['omch2'] = 0.114
params_fid['omnuh2'] = 0.001069 #100.0/93000
params_fid['omk'] = 0.00
params_fid['YHe'] = 0.24
params_fid['Neff'] = 3.04
params_fid['w'] = -1.0
params_fid['wa'] = 0.0
params_fid['tau'] = 0.09
params_fid['As'] = 2.46e-9
params_fid['ns'] = 0.96
params_fid['alphas'] = 0.0
params_fid['r'] = 0.1
params_fid['nT'] = - params_fid['r']/8
params_fid['A'] = 0.1 ## residuals amplitude @ ell= 1
params_fid['b'] = -0.8 ## ell dependence of the residuals
params_fid['k_scalar'] = 0.005
params_fid['k_tensor'] = 0.005

information_channels_tot = ['Tu', 'Eu', 'Bu', 'T', 'E', 'B', 'd']


def main():
	args = grabargs()
	w = sensitivity_pol_computation(args.NETs, args.time_of_integration, args.Yield, args.fsky, args.Nbolos)

	print params_fid
	Core(w = w, fsky = args.fsky , FWHMs = args.FWHMs, eta=args.eta, delensing=args.delensing, params_fid=params_fid, ell_max = args.ell_max, T_only_in_d=args.T_only_in_d, P_only_in_d=args.P_only_in_d)

def Core(ell_min=2, ell_max=3000, information_channels=['Tu', 'Eu', 'B', 'd'], w=1, fsky=0.8, FWHMs=3.5, eta=1.0, Cl_noise=[], fid_r =params_fid['r'], params_dev =['alphas', 'r','nT','ns','As', 'tau', 'h', 'ombh2', 'omch2','omnuh2', 'omk','YHe','Neff','w', 'wa', 'A', 'b'], DESI=False, DESI_BAO=False, delensing=0, params_fid=params_fid, T_only_in_d=False, P_only_in_d=False, short_output=False, LSS=False, use_Planck_T_noise=False, other_marginalizations_option=False, r_n=1.0, fcorr=0.0, CIB=False, prim_BB_CV=True, exe = './camb'):

	print '  			eta = ', eta

	if DESI and DESI_BAO:
		print ' you have to choose between desi or desi_bao'
		exit()

	if T_only_in_d & P_only_in_d:
		print 'you have to choose between T only or P only'
		exit()


	params_fid['r'] = fid_r
	params_fid['nT'] = - params_fid['r']/8.0

	if Cl_noise:
			w = Cl_noise[2]


	################################################
	# compute fiducial Cls computed between ell = 2 and 6000
	print '################################ computing fiducial Cls ... ####################################'
	name_fid = 'fidCls_eta_'+str(eta)+'_'
	for p in range(len(params_fid.keys())):
		name_fid += str(params_fid.keys()[p])+'_'+str(params_fid[params_fid.keys()[p]] )+'_'
	print 'looking for ', name_fid
	fnames_fid = glob.glob( name_fid+'.pkl' )
	if not fnames_fid:
		print '################# computing Cls file because it does not seem to be on disk #######################'
		Cls_fid = python_camb.submit_camb( h=params_fid['h'], ombh2=params_fid['ombh2'], omch2=params_fid['omch2'], omnuh2=params_fid['omnuh2'], omk=params_fid['omk'], YHe=params_fid['YHe'], Neff=params_fid['Neff'], w=params_fid['w'], wa=params_fid['wa'], tau=params_fid['tau'],As=params_fid['As'], ns=params_fid['ns'], alphas=params_fid['alphas'], nT=params_fid['nT'], r=params_fid['r'], k_scalar=params_fid['k_scalar'] , k_tensor=params_fid['k_tensor'], eta=eta, exe=exe)
		save_obj('./', name_fid, Cls_fid)
	else:
		print '################################ loading already existing Cls file ####################################'
		Cls_fid = load_obj('./', fnames_fid[0])

	################################################
	# noise computation computed between ell = 2 and 6000
	print 'computing noise Nls ...'
	Nls = Nls_computation( information_channels, FWHMs, w, Cls_fid, ell_max, Cl_noise, T_only_in_d, P_only_in_d, delensing, use_Planck_T_noise, r_n)

	################################################
	# adding astrophysical residuals to the signal variance

	if 'A' in params_dev and 'b' in params_dev:
		nch = len(information_channels)
		for ch1 in range(nch):
			for ch2 in range(nch):
				if not (information_channels[ch1] == 'd' or information_channels[ch2] =='d'):
					key = information_channels[ch1]+information_channels[ch2]
					Nls[key] += params_fid['A']*Cls_fid['ell']**params_fid['b']

	################################################
	# Compute derivatives computed between ell = 2 and 5000
	print 'computation of the derivatives ...'
	dCldp = derivatives_computation(Cls_fid, params_dev, params_fid, information_channels, eta, exe=exe)

	if short_output:
		return  Cls_fid, Nls, dCldp

	if delensing:
		Cls_fid = smith_delensing_self_consistent_forecast.smith_delensing_python( Cls_fid, Nls, ell_max, LSS=LSS, fcorr=fcorr, CIB=CIB)

	if 'BB_delens' not in Cls_fid.keys():
		print 'BB_delens key does not exit! ' 
		Cls_fid['BB_delens'] = Cls_fid['BlBl']

	################################################
	# build covariance matrix computed between ell_min and ell_max
	print 'building covariance matrices ...'
	Cov, Cov_inv = Cov_computation(information_channels, Cls_fid, Nls, ell_min, ell_max, prim_BB_CV, delensing)

	################################################
	# build covariance matrix computed between ell_min and ell_max
	print 'computation of the Fisher matrix ... using '+str(information_channels)+' information channels only'
	sigmas, sigmas_diag, sigma_dic_reduced, F = Fisher_computation(Cov_inv, dCldp, params_dev, fsky, information_channels, ell_min, ell_max, Cls_fid, params_fid, Cov, DESI, DESI_BAO, other_marginalizations_option)

	################################################
	# results
	print 'Results are: '
	for p in range(len(params_dev)):
		
		if params_fid[params_dev[p]] != 0.0:
			if params_dev[p] == 'ns':
				sentence = 'sigma( '+str(params_dev[p])+' - 1 = '+str(params_fid[params_dev[p]]-1.0)+' ) = '+str(sigmas[params_dev[p]])
				sentence += ' >>> detection with a significance = '+str(np.abs( (params_fid[params_dev[p]] - 1.0)/sigmas[params_dev[p]]))
			elif  params_dev[p] == 'Neff':
				sentence = 'sigma( '+str(params_dev[p])+' - 3 = '+str(params_fid[params_dev[p]]-3.0)+' ) = '+str(sigmas[params_dev[p]])
				sentence += ' >>> detection with a significance = '+str(np.abs( (params_fid[params_dev[p]] - 3.0)/sigmas[params_dev[p]]))
			else:
				sentence = 'sigma( '+str(params_dev[p])+' = '+str(params_fid[params_dev[p]])+' ) = '+str(sigmas[params_dev[p]])
				sentence += ' >>> detection with a significance = '+str(np.abs(params_fid[params_dev[p]]/sigmas[params_dev[p]]))
		else:
			sentence = 'sigma( '+str(params_dev[p])+' = '+str(params_fid[params_dev[p]])+' ) = '+str(sigmas[params_dev[p]])
		print sentence
	

		#########################################
		print '			and taking the parameter only leads to: '
		if params_fid[params_dev[p]] != 0.0:
			if params_dev[p] == 'ns':
				sentence = '			sigma( '+str(params_dev[p])+' - 1 = '+str(params_fid[params_dev[p]]-1.0)+' ) = '+str(sigmas_diag[params_dev[p]])
			elif  params_dev[p] == 'Neff':
				sentence = '			sigma( '+str(params_dev[p])+' - 3 = '+str(params_fid[params_dev[p]]-3.0)+' ) = '+str(sigmas_diag[params_dev[p]])
			else:
				sentence = '			sigma( '+str(params_dev[p])+' = '+str(params_fid[params_dev[p]])+' ) = '+str(sigmas_diag[params_dev[p]])
		else:
			sentence = '			sigma( '+str(params_dev[p])+' = '+str(params_fid[params_dev[p]])+' ) = '+str(sigmas_diag[params_dev[p]])
		print sentence

	if other_marginalizations_option:

		#########################################
		params_added = sigma_dic_reduced.keys()
		print ' ======================================= '
		print ' NOW: Marginalizing over', ['r','nT','ns','As', 'ombh2', 'omch2'], '... and adding successively: ', params_added
		for p in range(len(params_added)):
			if params_added[p] == 'None':
				print ' IF ALL PARAMETERS ARE FIXED EXCEPT ', ['r','nT','ns','As', 'ombh2', 'omch2'], ' here is what we get : '
				free_params = sigma_dic_reduced[params_added[p]].keys()
				for p2 in range(len(free_params)):
					print free_params[p2], params_fid[free_params[p2]]
					if params_fid[free_params[p2]] != 0.0:
						if free_params[p2] == 'ns':
							sentence = ' sigma( '+str(free_params[p2])+' - 1 = '+str(params_fid[free_params[p2]]-1.0)+' ) = '+str(sigma_dic_reduced['None'][free_params[p2]])
						elif  free_params[p2] == 'Neff':
							sentence = ' sigma( '+str(free_params[p2])+' - 3 = '+str(params_fid[free_params[p2]]-3.0)+' ) = '+str(sigma_dic_reduced['None'][free_params[p2]])
						else:
							sentence = 'sigma( '+str(free_params[p2])+' = '+str(params_fid[free_params[p2]])+' ) = '+str(sigma_dic_reduced['None'][free_params[p2]])
					else:
						sentence = ' sigma( '+str(free_params[p2])+' = '+str(params_fid[free_params[p2]])+' ) = '+str(sigma_dic_reduced['None'][free_params[p2]])
					print sentence
			elif params_fid[params_added[p]] != 0.0:
				print ' IF ALL PARAMETERS ARE FIXED EXCEPT ', params_added[p] , ' + ',['r','nT','ns','As', 'ombh2', 'omch2']
				if params_added[p] == 'ns':
					sentence = ' sigma( '+str(params_added[p])+' - 1 = '+str(params_fid[params_added[p]]-1.0)+' ) = '+str(sigma_dic_reduced[params_added[p]][params_added[p]])
				elif  params_added[p] == 'Neff':
					sentence = ' sigma( '+str(params_added[p])+' - 3 = '+str(params_fid[params_added[p]]-3.0)+' ) = '+str(sigma_dic_reduced[params_added[p]][params_added[p]])
				else:
					sentence = 'sigma( '+str(params_added[p])+' = '+str(params_fid[params_added[p]])+' ) = '+str(sigma_dic_reduced[params_added[p]][params_added[p]])
				print sentence
			else:
				print ' IF ALL PARAMETERS ARE FIXED EXCEPT ', params_added[p] , ' + ',['r','nT','ns','As', 'ombh2', 'omch2']
				sentence = 'sigma( '+str(params_added[p])+' = '+str(params_fid[params_added[p]])+' ) = '+str(sigma_dic_reduced[params_added[p]][params_added[p]])
				print sentence
		#########################################


	return sigmas, F, w, fsky, Cls_fid, Nls


def Fisher_computation(Cov_inv, dCldp, params_dev, fsky, information_channels, ell_min, ell_max, Cls_fid, params_fid, Cov, DESI=False, DESI_BAO=False, other_marginalizations_option=False):

	#ell_v = Cls_fid['ell'][ell_min-2:ell_max-1] #np.arange(ell_min, ell_max, 1)
	ell_v = np.arange(ell_min-2, ell_max-1, 1)
	nell = ell_max - ell_min + 1
	F = np.zeros((len(params_dev),len(params_dev)))

	if DESI or DESI_BAO:
		print ' ///////////////////////////////////////// COMBINING WITH DESI ///////////////////////////////////////////'
		if DESI:
			print ' 						and you are combining with FULL DESI '
			Fbb1 = np.loadtxt( '/Users/josquin1/Documents/Dropbox/postdoc@LBL/PRISM-COrE-ESA-M4/foregrounds_python/newfish/bb14lz_1_3_all_0.1.zahnfish' )
			#Fbb2 = np.loadtxt( '/Users/josquin1/Documents/Dropbox/postdoc@LBL/PRISM-COrE-ESA-M4/foregrounds_python/newfish/bb14lz_1_3_all_0.2.zahnfish' )
			Fbb = Fbb1 #+ Fbb2
			#Fbb[2,:] /= 100.0
			#Fbb[:,2] /= 100.0
		elif DESI_BAO:
			print ' 						and you are combining with DESI BAO '
			#Fbb = np.loadtxt( '/Users/josquin1/Documents/Dropbox/postdoc@LBL/PRISM-COrE-ESA-M4/foregrounds_python/newfish/bb14lz_1_3_all_0.1.zahnfish' )
			Fbb = np.loadtxt( '/Users/josquin1/Documents/Dropbox/postdoc@LBL/PRISM-COrE-ESA-M4/foregrounds_python/newfish/bb14lz_1_3_bao_0.zahnfish' )
			#Fbb = Fbb1 + Fbb2
			#stored as scalar_amp(1) scalar_spectral_index(1) hubble re_optical_depth omnuh2 ombh2 omch2 w Nnu
			# ANZE: omch2 ombh2 omnuh2 w scalar_amp(1) scalar_spectral_index(1) re_optical_depth hubble
			# 		As ns hubble tau omnuh2 ombh2 omch2 w Neff
			#Fbb[7,:] /= 100.0
			#Fbb[:,7] /= 100.0
		#Fbb *= 1000.0
		#Fbb_params_dev = ['omch2', 'ombh2', 'omnuh2', 'w', 'As', 'ns', 'tau', 'h', 'Neff']
		Fbb_params_dev =  ['As', 'ns', 'h', 'tau', 'omnuh2', 'ombh2', 'omch2', 'w', 'Neff']
		#Fbb[2,:] *= 100.0
		#Fbb[:,2] *= 100.0

	print 'FISHER COMPUTATION'# >>> ell max is ', ell_max 
	for i in range(len(params_dev)):
		dCldp_m_i = matrix_form(information_channels, ell_min, ell_max, dCldp[params_dev[i]])
		for j in range(len(params_dev)):
			dCldp_m_j = matrix_form(information_channels, ell_min, ell_max, dCldp[params_dev[j]])
			for l in range(nell):
				if Cov_inv.shape[0] < 2:
					F[i,j] += ((2*ell_v[l] + 1)*fsky/2)*( np.squeeze(dCldp_m_i[:,:,l]) )*(np.squeeze(Cov_inv[:,:,l]))*(np.squeeze(dCldp_m_j[:,:,l]))*(np.squeeze(Cov_inv[:,:,l])) 
				else:
					F[i,j] += ((2*ell_v[l] + 1)*fsky/2)*np.trace( (np.squeeze(dCldp_m_i[:,:,l])).dot(np.squeeze(Cov_inv[:,:,l])).dot(np.squeeze(dCldp_m_j[:,:,l])).dot(np.squeeze(Cov_inv[:,:,l])) )
			if DESI or DESI_BAO:
				key1 = params_dev[i]
				key2 = params_dev[j]
				if key1 in Fbb_params_dev and key2 in Fbb_params_dev:
					ind1 = Fbb_params_dev.index( key1 )
					ind2 = Fbb_params_dev.index( key2 )
					Fbb_loc = Fbb[ ind1, ind2 ]
					F[i,j] += Fbb_loc

	Finv = np.linalg.inv(F)
	sigmas = np.sqrt(np.diag( Finv ))

	##########################################################################
	sigma_dic = {}
	for p in range(len(params_dev)):
		sigma_dic[params_dev[p]] = sigmas[p]

	##########################################################################
	### taking only the diagonal of F .... 
	#print '>> taking only the diagonal of F  '
	sigmas_diag = np.diag( np.sqrt( np.linalg.inv(np.diag(np.diag(F))) ) )
	sigma_dic_diag = {}
	for p in range(len(params_dev)):
		sigma_dic_diag[params_dev[p]] = sigmas_diag[p]


	sigma_dic_reduced = {}
	if other_marginalizations_option:

		##########################################################################
		### taking only a part of the Fisher matrix
		#print '>> taking only a part of the Fisher matrix '
		
		params_dev_reduced = ['r','nT','ns','As', 'ombh2', 'omch2', 'h']
		other_parameters = ['None', 'alphas','omnuh2','omk','YHe','Neff']
		import copy
		for op in other_parameters:
			#print ' >> starting loop over '
			params_dev_reduced_loc = copy.copy(params_dev_reduced)
			if not op == 'None':
				params_dev_reduced_loc.append( op )

			F_reduced = np.zeros(( len(params_dev_reduced_loc), len(params_dev_reduced_loc) ))

			# matching indices of the usual Fisher matrix
			indices_intersection = []
			for p in range(len(params_dev_reduced_loc)):
				if not params_dev_reduced_loc[p] == 'None':
					indices_intersection.append( params_dev.index( params_dev_reduced_loc[p] ))

			## grabing the useful part of the big Fisher matrix
			for i in range(len(indices_intersection)):
				for j in range(len(indices_intersection)):
					F_reduced[i,j] = F[indices_intersection[i], indices_intersection[j]]
			F_reduced_inv = np.linalg.inv( F_reduced )
			sigmas_reduced_0 = np.sqrt(np.diag( F_reduced_inv ))

			## filling in the dictionary of constraints
			sigma_dic_reduced[ op ] = {}
			for p1 in range(len(params_dev_reduced_loc)):
				sigma_dic_reduced[ op ][ params_dev_reduced_loc[p1] ] = sigmas_reduced_0[p1]

			del params_dev_reduced_loc, indices_intersection, F_reduced, sigmas_reduced_0

	return sigma_dic, sigma_dic_diag, sigma_dic_reduced, F


def derivatives_computation(Cls_fid, params_dev, params_fid, information_channels, eta=1.0, exe='./camb'):

	nell = len(Cls_fid[information_channels[0]+information_channels[0]])# ell_max - ell_min
	nparam = len(params_dev)
	nch = len(information_channels_tot)
	dCldp = {}

	'''
	subname_search ='deriv_'
	subname ='deriv_'
	for i in range(len(information_channels)):
		subname_search += '*'+information_channels[i]
		subname += information_channels[i]+'_'
	# derivative over each parameter
	for p in range( nparam ):
		# check if it's already computed then continue if it is the case
		subname_search += '*'+str(params_dev[p])+'_'+str(params_fid[params_dev[p]])
		subname += str(params_dev[p])+'_'+str(params_fid[params_dev[p]])+'_'

	print 'looking for files which have ', subname_search+'*.pkl' , ' in their name '
	print subname_search
	fnames = glob.glob(  subname_search+'*.pkl' )
	if fnames:
		print '################################ loading already existing file ####################################'
		dCldp = load_obj('./', fnames[0]) # loading the first file of fnames list
		return dCldp
	'''

	for p in range( nparam ):

		## look for already computed and save on disk derivatives
		
		## subname_search ='deriv_'
		## subname ='deriv_'
		## k_name = 'kscalar_'+str(params_fid['k_scalar'])+'_ktensor_'+str(params_fid['k_tensor'])
		## for i in range(len(information_channels)):
		## 	subname_search += '*'+information_channels[i][0]
		## 	subname += information_channels[i][0]+'_'
		## subname_search += '*'+str(params_dev[p])+'_'+str(params_fid[params_dev[p]])+'_'+k_name
		## subname += str(params_dev[p])+'_'+str(params_fid[params_dev[p]])+'_'+k_name

		subname_search ='deriv_'
		subname ='deriv_'
		subname_search += '*'+str(params_dev[p])
		subname += str(params_dev[p])
		for i in range(len(information_channels)):
			subname_search += '*'+information_channels[i][0]
			subname += '_' + information_channels[i][0]
		for q in range(len(params_fid.keys())):
			subname += '_'+str(params_fid.keys()[q])+'_'+str(params_fid[params_fid.keys()[q]])
			subname_search += '_'+str(params_fid.keys()[q])+'_'+str(params_fid[params_fid.keys()[q]])        
        
		print 'looking for files which have ', subname_search +'*.pkl' , ' in their name '
		fnames = glob.glob(  subname_search+'*.pkl' )
		if fnames:
			print '################################ loading already existing file ####################################'
			dCldp_loc = load_obj('./', fnames[0]) # loading the first file of fnames list
			dCldp[params_dev[p]] = dCldp_loc[params_dev[p]]
			del dCldp_loc
		else:
			print '################################ derivative of Cls wrt. '+params_dev[p]+' #################################### '
			if params_dev[p] == 'A':
				dCldp[params_dev[p]] = {}
				for ch1 in range(nch):
					for ch2 in range(nch):
						key = information_channels_tot[ch1]+information_channels_tot[ch2]
						if 'T' in key or 'Tu' in key or 'd' in key or 'E' in key or 'Eu' in key:
							dCldp['A'][key] = Cls_fid['ell']*0.0
						else:
							dCldp['A'][key] = Cls_fid['ell']**params_fid['b']
				save_obj('./', subname, dCldp)
				continue
			elif params_dev[p] == 'b':
				dCldp[params_dev[p]] = {}
				for ch1 in range(nch):
					for ch2 in range(nch):
						key = information_channels_tot[ch1]+information_channels_tot[ch2]
						print key
						if 'T' in key or 'Tu' in key or 'd' in key or 'E' in key or 'Eu' in key:
							dCldp['b'][key] = Cls_fid['ell']*0.0
						else:
							dCldp['b'][key] = params_fid['A']*params_fid['b']*Cls_fid['ell']**( params_fid['b'] - 1 )
				save_obj('./', subname, dCldp)
				continue
			elif params_dev[p] == 'omk':
				#########################################################################
				print 'special derivatives for OmK'
				params_v = [ 0.0, 0.001 ]
				Cls_tot_loc = np.zeros(( len(params_v), nch, nch, nell ))
				params_loc = params_fid.copy()
				params_loc[params_dev[p]] = 0.001
			
				Cls_loc = python_camb.submit_camb( h=params_loc['h'], ombh2=params_loc['ombh2'], omch2=params_loc['omch2'], omnuh2=params_loc['omnuh2'], omk=params_loc['omk'], YHe=params_loc['YHe'], Neff=params_loc['Neff'], w=params_loc['w'], wa=params_loc['wa'],  tau=params_loc['tau'],As=params_loc['As'], ns=params_loc['ns'], alphas=params_loc['alphas'], nT=params_loc['nT'], r=params_loc['r'], k_scalar=params_fid['k_scalar'] , k_tensor=params_fid['k_tensor'], eta=1.0, exe=exe)
			
				dCldp[params_dev[p]] = {}

				print np.shape(Cls_tot_loc), np.shape(Cls_loc['TT'])
				
				for ch1 in range(nch):
					for ch2 in range(nch):
						key = information_channels_tot[ch1]+information_channels_tot[ch2]
						if not key in Cls_fid.keys():
							Cls_tot_loc[0, ch1, ch2,:] = 0.0
							Cls_tot_loc[1, ch1, ch2,:] = 0.0
						else:
							# 0th point is the fiducial Cls
							Cls_tot_loc[0, ch1, ch2, :] = Cls_fid[key][:]*1.0
							# point #1 is the increment cosmology
							Cls_tot_loc[1, ch1, ch2, :] = Cls_loc[key][:]*1.0

						dCldp[params_dev[p]][key] = np.zeros(nell)	
						if not key in Cls_fid.keys():
							dCldp[params_dev[p]][key] = np.zeros(nell)
						else:
							for l in range(nell):
								x=[ params_fid[params_dev[p]]+params_v[1], params_fid[params_dev[p]]+params_v[0] ]
								dCldp_loc = Cls_tot_loc[1,ch1,ch2,l] - Cls_tot_loc[0,ch1,ch2,l]
								dCldp_loc /= ( x[1]-x[0] )
								if dCldp_loc != dCldp_loc: dCldp_loc = 0.0
								dCldp[params_dev[p]][key][l] = dCldp_loc

				save_obj('./', subname, dCldp)
				continue
				#########################################################################
			elif params_fid[params_dev[p]] < 0.0:
				print 'fiducial parameter is negative'
				params_v = [ 1.1, 1.05, 1.0, 0.95, 0.90 ]
			elif params_fid[params_dev[p]] == 0.0:
				print 'fiducial parameter is null'
				params_v =[ -0.1, -0.05, 0.0, 0.05, 0.1 ]
			elif params_fid[params_dev[p]] < 1e-10:
				print 'fiducial parameter is super small, i am considering slightly larger steps'
				params_v =[  0.8, 0.9, 1.0, 1.1, 1.2 ]
			else:
				#params_v = [ 0.980, 0.990, 1.000, 1.010, 1.020 ]
				params_v = [ 0.90, 0.95, 1.0, 1.05, 1.1 ]

			Cls_tot_loc = np.zeros(( len(params_v), nch, nch, nell ))
			for i in range( len(params_v) ):
				# central point
				if i == 2:
					for ch1 in range(nch):
						for ch2 in range(nch):
							key = information_channels_tot[ch1]+information_channels_tot[ch2]
							if not key in Cls_fid.keys():
								Cls_tot_loc[i, ch1, ch2,:] = 0.0
							else:
								Cls_tot_loc[i, ch1, ch2,:] = Cls_fid[key][:]*1.0
				else:
					## two points smaller, two point higher
					# make the right change
					params_loc = params_fid.copy()
					if params_fid[params_dev[p]] != 0.0:
						params_loc[params_dev[p]] *= params_v[i]
					else:
						params_loc[params_dev[p]] += params_v[i]
					# compute the Cls
					Cls_loc = python_camb.submit_camb( h=params_loc['h'], ombh2=params_loc['ombh2'], omch2=params_loc['omch2'], omnuh2=params_loc['omnuh2'], omk=params_loc['omk'], YHe=params_loc['YHe'], Neff=params_loc['Neff'], w=params_loc['w'], wa=params_loc['wa'],  tau=params_loc['tau'],As=params_loc['As'], ns=params_loc['ns'], alphas=params_loc['alphas'], nT=params_loc['nT'], r=params_loc['r'], k_scalar=params_fid['k_scalar'] , k_tensor=params_fid['k_tensor'], eta=1.0, exe=exe)
					for ch1 in range(nch):
						for ch2 in range(nch):
							key = information_channels_tot[ch1]+information_channels_tot[ch2]
							if not key in Cls_fid.keys():
								Cls_tot_loc[i, ch1, ch2,:] = 0.0
							else:
								Cls_tot_loc[i, ch1, ch2, :] = Cls_loc[key][:]*1.0
					del params_loc, Cls_loc, ch1, ch2, key

			########################################################
			#if params_dev[p] == 'b' or params_dev[p] == 'A':
			#	continue
			########################################################

			print '################################ interpolation of the derivative ####################################'
			dCldp[params_dev[p]] = {}		
			for ch1 in range(nch):
				for ch2 in range(nch):

					key = information_channels_tot[ch1]+information_channels_tot[ch2]
					## tune steps for different keys?
					delta_step = 0.001

					print key
					dCldp[params_dev[p]][key] = np.zeros(nell)	
					if not key in Cls_fid.keys():
						dCldp[params_dev[p]][key] = np.zeros(nell)
					else:
						for l in range(nell):
							# interpolate the points
							params_loc_v = np.zeros(len(params_v))
							for k in range(len(params_v)):
								if params_fid[params_dev[p]] != 0.0:
									params_loc_v[k] = params_fid[params_dev[p]]*params_v[k]
								else:
									params_loc_v[k] = params_fid[params_dev[p]] + params_v[k]

							f = scipy.interpolate.interp1d(params_loc_v , np.squeeze(Cls_tot_loc[:,ch1,ch2,l]), kind='cubic')

							if params_fid[params_dev[p]] != 0.0:
									dCldp_loc = f( params_fid[params_dev[p]]*(1.0 + delta_step) ) - f( params_fid[params_dev[p]]*(1.0 - delta_step) )
									dCldp_loc /= ( params_fid[params_dev[p]]*2*delta_step )
							else:
									dCldp_loc = f( delta_step ) - f( - delta_step )
									dCldp_loc /= ( 2*delta_step )


							if dCldp_loc != dCldp_loc: dCldp_loc = 0.0

							dCldp[params_dev[p]][key][l] = dCldp_loc

							#######################################
							'''
							if l == 100 or l==1000:
								if l== 100:
									pl.figure()
									pl.subplot(211)
									pl.title(key+' @ l=100 and der = '+str(dCldp_loc))
									pl.plot(params_loc_v , np.squeeze(Cls_tot_loc[:,ch1,ch2,l]), 'ko')
									x = np.arange(np.min(params_loc_v)*1.001, np.max(params_loc_v)*0.999, params_fid[params_dev[p]]*0.0001)
									pl.plot(x, f(x), 'k')
									pl.plot( params_fid[params_dev[p]]*(1.0 + delta_step), f( params_fid[params_dev[p]]*(1.0 + delta_step) ), 'rx')
									pl.plot( params_fid[params_dev[p]]*(1.0 - delta_step), f( params_fid[params_dev[p]]*(1.0-delta_step) ), 'rx' )
									der = dCldp_loc*x + ( f(params_fid[params_dev[p]])- dCldp_loc*params_fid[params_dev[p]])
									pl.plot(x, der, 'k--')
								if l==1000:
									pl.subplot(212)
									pl.title(key+' @ l=100 and der = '+str(dCldp_loc))
									pl.plot(params_loc_v , np.squeeze(Cls_tot_loc[:,ch1,ch2,l]), 'ro')
									pl.plot(x, f(x), 'r')
									pl.plot( params_fid[params_dev[p]]*(1.0 + delta_step), f( params_fid[params_dev[p]]*(1.0 + delta_step) ), 'kx')
									pl.plot( params_fid[params_dev[p]]*(1.0 - delta_step), f( params_fid[params_dev[p]]*(1.0-delta_step) ), 'kx')
									der2 = dCldp_loc*x + ( f(params_fid[params_dev[p]])- dCldp_loc*params_fid[params_dev[p]])
									pl.plot(x, der2, 'r--')
							'''
							###############################################

							del f, params_loc_v, dCldp_loc
						del l, key, delta_step
			del ch1, ch2
			########################################################
			print '################################ saving derivatives wrt.', params_dev[p],' to disk ####################################'
			save_obj('./', subname, dCldp)
			########################################################

	return dCldp

def save_obj(path, name, obj):
	with open(os.path.join( path, name + '.pkl' ), 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path, name):
	print 'loading ... ', os.path.join( path, name )
	with open(os.path.join( path, name ), 'r') as f:
		return pickle.load(f)

def Cov_computation(information_channels, Cls, Nls, ell_min, ell_max, prim_BB_CV=True, delensing=False):
	
	nch = len(information_channels)
	nell = ell_max - ell_min + 1
	Cov = np.zeros((nch, nch, nell))
	#pl.figure()
	for ch1 in range(nch):
		for ch2 in range(nch):

			key = information_channels[ch1]+information_channels[ch2]
			keyNl = information_channels[ch1][0]+information_channels[ch2][0]

			if not key in Cls.keys():
				Cov[ch1, ch2,:] = 0.0
			else:
				if ch1 == ch2 :
					Cov[ ch1 , ch2 , : ] = Nls[keyNl][ell_min-2:ell_max-1]
					if keyNl != 'BB' or (keyNl == 'BB' and prim_BB_CV):
						Cov[ ch1 , ch2 , : ] += Cls[key][ell_min-2:ell_max-1]
					if key == 'BuBu':
						Cov[ ch1 , ch2 , : ] += Cls['BB_delens'][ell_min-2:ell_max-1] #+ Cls['BuBu'][ell_min-2:ell_max-1] 
				else:
					Cov[ ch1 , ch2 , : ] = Cls[key][ell_min-2:ell_max-1]
				#if key =='Bd' or key =='dd' or key =='BB':
				#if key == 'dd':
				#pl.loglog(Cls['BlBl'], 'r--')
				#pl.loglog(Cov[ ch1 , ch2 , : ] , 'k')
				#pl.figure()
				#pl.plot(Cls['BlBl'][ell_min-2:ell_max-1]/Cov[ ch1 , ch2 , : ] , 'k--')
				#if keyNl=='BB':
				#	ells = Cls['ell'][ell_min-2:ell_max-1]
				#	norm = ells*(ells+1)/(2*np.pi)
				#	pl.loglog(Cls[key][ell_min-2:ell_max-1], '--' )
				#	pl.loglog(Nls[keyNl][ell_min-2:ell_max-1], label=key)
				#	#pl.loglog(Nls['phiphi'][ell_min-2:ell_max-1], 'r--')
	#pl.ylim([1e-5, 1e5])
	#pl.legend(loc='best')
	#pl.show()
	#exit()

	Cov_inv = Cov*0.0
	#eig = 0.0
	for l in range(nell):
		if Cov_inv.shape[0] < 2:
			Cov_inv[:,:,l] = 1.0/np.squeeze( Cov[:,:,l] )
		else:
			Cov_inv[:,:,l] = np.linalg.inv( np.squeeze( Cov[:,:,l] ))
		#eig, eigv = np.linalg.eig( np.squeeze( Cov[:,:,l] ))
		#eig += np.min(eig)/np.max(eig)
	#print eig/nell

	return Cov, Cov_inv

def matrix_form(information_channels, ell_min, ell_max, dCls):
	nch = len(information_channels)
	nell = ell_max - ell_min + 1
	M = np.zeros((nch, nch, nell))
	for ch1 in range(nch):
		for ch2 in range(nch):
			key = information_channels[ch1]+information_channels[ch2]
			M[ ch1 , ch2 , : ] = dCls[key][ell_min-2:ell_max-1]
	return M

def Nls_computation(information_channels, FWHM, w, Cls_fid, ell_max_cut, Cl_noise=[], T_only_in_d=False, P_only_in_d=False, delensing=False, use_Planck_T_noise=False, r_n=1.0):
	Nl = {}
	import copy

	if ('T' in information_channels) or ('Tu' in information_channels): 
		if use_Planck_T_noise:
			print 'T noise from Planck'
			Nl['TT'], ells = noise_computation(7.1, 45.0)
			ell_max = np.min([ len(ells), len(Cls_fid['ell']) ])
		elif Cl_noise:
			print ' T noise is from Cl_noise'
			ells=copy.copy(Cl_noise[1])
			Nl['TT'] = Cl_noise[0]*1.0
			ell_max = np.min([ len(Nl['TT']), len(Cls_fid['ell']) ])
		else:
			print ' T noise is computed on the fly'
			Nl['TT'], ells = noise_computation(FWHM, w)
			ell_max = np.min([ len(ells), len(Cls_fid['ell']) ])
		## rescale NlTT
		for l in range(ell_max):
			if l==0: print 're-normalizing NlTT'
			Nl['TT'][l] *= ells[l]*(ells[l]+1)/(2*np.pi)
			Nl['TT'][l] /= 2.0

	if ('E' in information_channels) or ('Eu' in information_channels) :
		if 'TT' in Nl.keys() and not use_Planck_T_noise:
			print 'computing NlEE from TT'
			Nl['EE'] = Nl['TT']*2.0
		else:
			if Cl_noise:
				print ' E noise is from Cl_noise'
				ells=copy.copy(Cl_noise[1])
				Nl['EE'] = Cl_noise[0]*1.0
				ell_max = np.min([ len(Nl['EE']), len(Cls_fid['ell']) ])
			else:
				print ' E noise is computed on the fly'
				Nl['EE'], ells = noise_computation(FWHM, w)
				ell_max = np.min([ len(ells), len(Cls_fid['ell']) ])
			for l in range(ell_max):
				if l==0: print 're-normalizing NlEE'
				Nl['EE'][l] *= ells[l]*(ells[l]+1)/(2*np.pi)

	if ('B' in information_channels) or ('Bu' in information_channels) :
		if 'TT' in Nl.keys() and not use_Planck_T_noise:
			print 'computing NlBB from TT'
			Nl['BB'] = Nl['TT']*2.0
		elif 'EE' in Nl.keys():
			print 'computing NlBB from EE'
			Nl['BB'] = Nl['EE']*1.0
		else:
			if Cl_noise:
				print ' B noise is from Cl_noise'
				ells=copy.copy(Cl_noise[1])
				Nl['BB'] = Cl_noise[0]*1.0
				ell_max = np.min([ len(Nl['BB']), len(Cls_fid['ell']) ])
			else:
				print 'B noise is computed on the fly'
				Nl['BB'], ells = noise_computation(FWHM, w)
				ell_max = np.min([ len(ells), len(Cls_fid['ell']) ])
			for l in range(ell_max):
				if l==0: print 're-normalizing NlBB'
				Nl['BB'][l] *= ells[l]*(ells[l]+1)/(2*np.pi)
	## degraded TT noise due to atmosphere
	if r_n != 1.0: 
		if ('T' in information_channels) or ('Tu' in information_channels): 
			print ' 													/!\ TT is degraded /!\ Are you working on atmosphere stuff ?? '
			Nl['TT'][:] /= (r_n**2)

	########################################################################################

	if ('d'  in information_channels ) or delensing:

		name_Nldd_search ='Nldd_FWHM_'+str(FWHM)+'_w_'+str(w)+'_ellmax_'+str(ell_max_cut)
		name_Nldd_search+='_rn_'+str(r_n)
		# derivative over each parameter
		# check if it's already computed then continue if it is the case
		print 'looking for files which have ', name_Nldd_search, ' in their name '
		fnames = glob.glob( name_Nldd_search+'*.pkl' )
		if fnames:
			print '################################ loading already existing Nldd ####################################'
			Nl['dd'] = load_obj('./', fnames[0]) # loading the first file of fnames list

			print ' .............. and computing Nldd ...'
			if P_only_in_d:
				print ' ............... without information on T '
			if T_only_in_d:
				print ' ............... without information on P '
			if 'TT' not in Nl.keys() : 
				print 'computing Nldd but T was not computed'
				if Cl_noise:
					print ' T noise is from Cl_noise'
					ells=copy.copy(Cl_noise[1])
					Nl['TT'] = Cl_noise[0]*1.0
					ell_max = np.min([ len(Nl['TT']), len(Cls_fid['ell']) ])
				else:
					Nl['TT'], ells = noise_computation(FWHM, w)
					ell_max = np.min([ len(ells), len(Cls_fid['ell']) ])
			if 'EE' not in Nl.keys() : 
				if Cl_noise:
					print ' E noise is from Cl_noise'
					ells=copy.copy(Cl_noise[1])
					Nl['EE'] = Cl_noise[0]*1.0
					ell_max = np.min([ len(Nl['EE']), len(Cls_fid['ell']) ])
				else:
					print 'computing Nldd but E was not computed'
					Nl['EE'] = Nl['TT']*2
					ell_max = np.min([ len(Nl['EE']), len(Cls_fid['ell']) ])
		else:
					
			######################
			assert ell_max_cut*1.2 <= ell_max
			ell_max_loc = int(ell_max_cut*1.2) # to deal with high ell cut in FUTURCMB
			if ell_max_loc <= ell_max: ell_max = ell_max_loc
			######################

			if 'TT' not in Nl.keys() : 
				print 'computing Nldd but T was not computed'
				if Cl_noise:
					print ' T noise is from Cl_noise'
					ells=copy.copy(Cl_noise[1])
					Nl['TT'] = Cl_noise[0]*1.0
					ell_max = np.min([ len(Nl['TT']), len(Cls_fid['ell']) ])
				else:
					Nl['TT'], ells = noise_computation(FWHM, w)
					ell_max = np.min([ len(ells), len(Cls_fid['ell']) ])
			if 'EE' not in Nl.keys() : 
				if Cl_noise:
					print ' E noise is from Cl_noise'
					ells=copy.copy(Cl_noise[1])
					Nl['EE'] = Cl_noise[0]*1.0
					ell_max = np.min([ len(Nl['EE']), len(Cls_fid['ell']) ])
				else:
					print 'computing Nldd but E was not computed'
					Nl['EE'] = Nl['TT']*2
					ell_max = np.min([ len(Nl['EE']), len(Cls_fid['ell']) ])

			Cls_TT_unlensed = Cls_fid['TuTu'][:ell_max]*0.0
			Cls_EE_unlensed = Cls_fid['EuEu'][:ell_max]*0.0
			Cls_TE_unlensed = Cls_fid['TuEu'][:ell_max]*0.0
			Cls_dd_unlensed = Cls_fid['dd'][:ell_max]*0.0
			Cls_Td_unlensed = Cls_fid['Tud'][:ell_max]*0.0
			Cls_TT_lensed = Cls_fid['TT'][:ell_max]*0.0
			Cls_EE_lensed = Cls_fid['EE'][:ell_max]*0.0
			Cls_TE_lensed = Cls_fid['TE'][:ell_max]*0.0
			Cls_BB_lensed = Cls_fid['BB'][:ell_max]*0.0
			NlT =  Nl['TT'][:ell_max]*0.0
			NlP =  Nl['EE'][:ell_max]*0.0

			# locally re-normalizing Cls ( from Dls to Cls )
			for l in range( ell_max ):
				norm = Cls_fid['ell'][l]*(Cls_fid['ell'][l]+1)/(2*np.pi)
				Cls_TT_unlensed[l] = Cls_fid['TuTu'][l]/norm 
				Cls_EE_unlensed[l] = Cls_fid['EuEu'][l]/norm
				Cls_TE_unlensed[l] = Cls_fid['TuEu'][l]/norm
				Cls_dd_unlensed[l] = Cls_fid['dd'][l]/norm
				Cls_Td_unlensed[l] = Cls_fid['Tud'][l]/norm
				Cls_TT_lensed[l] = Cls_fid['TT'][l]/norm
				Cls_EE_lensed[l] = Cls_fid['EE'][l]/norm
				Cls_TE_lensed[l] = Cls_fid['TE'][l]/norm
				Cls_BB_lensed[l] = Cls_fid['BB'][l]/norm
				if P_only_in_d:
					NlT[l] = Nl['TT'][l]*1e15/norm
				else:
					## degraded TT noise due to atmosphere
					NlT[l] = Nl['TT'][l]*1.0/norm
				if T_only_in_d:
					NlP[l] = Nl['EE'][l]*1e15/norm
				else:
					NlP[l] = Nl['EE'][l]*1.0/norm
				del norm

			# !! Cl non lentilles : 1=T, 2=E, 3=TE, 4=dd, 5=Td
			# !! Cl lentilles : 1=T, 2=E, 3=TE, 4=B
			# !! Bruit instru : 1=T, 2=P

			Cls_unlensed = np.vstack(( Cls_TT_unlensed, Cls_EE_unlensed, Cls_TE_unlensed, Cls_dd_unlensed, Cls_Td_unlensed ))
			Cls_lensed = np.vstack(( Cls_TT_lensed, Cls_EE_lensed, Cls_TE_lensed, Cls_BB_lensed ))
			Nls_instru = np.vstack(( NlT, NlP ))

			nldd  = Nl['TT'][:ell_max]*0.0

			assert Cls_unlensed.shape[1] == Cls_lensed.shape[1] == Nls_instru.shape[1] == len(nldd)
			# calc_Nldd (incls, inlcls, clnoise, jjmaxTmax, lmaxM, nldd)
#SMF			FUTURCMB_lensnoise.lensnoise.calc_Nldd(Cls_unlensed,  Cls_lensed,  Nls_instru, 164, len(nldd), nldd )
			#del Cls_unlensed, Cls_lensed, Nls_instru
			Nl['dd'] = nldd
			#del nldd
			#import pylab as pl
			#pl.figure()
			#pl.loglog(Cls_unlensed.T, '-')
			#pl.loglog(Cls_lensed.T, '--')
			#pl.loglog(NlT, 'r:')
			#pl.loglog(NlP, 'b:')
			#pl.loglog(nldd.T, 'k--')
			#pl.show()

			# normalizing Nldd to real 
			for l in range( ell_max ):
				# l*(l+1)nldd is the true (l*(l+1))Nldd/2pi
				norm = Cls_fid['ell'][l]*(Cls_fid['ell'][l]+1)
				Nl['dd'][l] *= (norm)/(2*np.pi)
				del norm

			#pl.loglog(nldd.T, 'k:')
			#pl.figure()
			#pl.loglog( Nl['TT'], 'k--')
			#pl.loglog( Nl['EE'], 'r--')
			#pl.show()
			#exit()

			save_obj('./', name_Nldd_search,  Nl['dd'] )

	return Nl

def noise_computation(FWHM, w):

	print '____________________'
	print 'noise computation with FWHM = ', FWHM, ' and w = ', w, ' in uK.arcmin '
	print '____________________'

	ell_max = 5000
	ell_min = 2
	ell = range(ell_min, ell_max)
	nell = ell_max - ell_min + 1
	Cl_noise = np.zeros( nell )
	Cl_noise_inv = 0.0
	if isinstance(FWHM, list) and len(FWHM) >1:
		print 'not implemented yet'
		exit()
	else:
		if isinstance(FWHM, list): FWHM = FWHM[0]
		sigma_b = (FWHM*arcmin_to_radian)/(np.sqrt(8.0*np.log(2.0)))

	Bl_inv = np.zeros( nell )
	for l in range( nell ):
		l2 = l+ell_min
		Bl_inv[l] = np.exp( -((sigma_b**2)*l2*(l2+1)) ) 
	Cl_noise_inv = (1.0/(w*arcmin_to_radian)**2)*Bl_inv # this is in 1.0/(uK.arcmin)**2

	Cl_noise = np.zeros( nell )
	Cl_noise[:] = 1.0/Cl_noise_inv[:] # this is in (uK.arcmin)**2 because w_ch is in 1.0/(uK.arcmin)**2
	#Cl_noise *= (arcmin_to_radian**2) # this is in (uK.rad)**2

	return Cl_noise, ell


def sensitivity_pol_computation(NET, time_of_integration, efficiency, fsky, Nbolos):
	if isinstance(NET, list) and len(NET) >1:
		print 'not implemented yet'
		exit()
	else:
		if isinstance(NET, list): NET = NET[0]
		skyam = 4.0*np.pi*fsky/(arcmin_to_radian**2) ## fsky [arcmin**2]
		EffectiveDetectorSeconds = time_of_integration * efficiency
		w_eff = (EffectiveDetectorSeconds)/( NET**2 * skyam )

	print ' The effective sensitivity for this experiment is :', 1.0/np.sqrt(w_eff), ' uK.arcmin '

	return 1.0/np.sqrt(w_eff)

def grabargs():
	parser = argparse.ArgumentParser(description='computation of Cl noise given instrumental parameters')
	parser.add_argument('--Nbolos', dest='Nbolos', action='store', type=int, help='number of bolos in each channel',required=False, default=1)
	parser.add_argument('--NETs', dest='NETs', action='store', type=float, help='NETs in uK.rs',required=False, default=20)
	parser.add_argument('--FWHMs', dest='FWHMs', action='store', type=float, help='FWHMs in arcmin',required=False, default=7.1)
	parser.add_argument('--time_of_integration', dest='time_of_integration', type=float, action='store', help='time of integration in year',required=False, default=3)
	parser.add_argument('--fsky', dest='fsky', action='store', type=float, help='fraction of the sky between 0.0 and 1.0',required=False, default=0.75)
	parser.add_argument('--Yield', dest='Yield', action='store', type=float, help='Yield',required=False, default = 1.0)
	parser.add_argument('--eta', dest='eta', action='store', type=float, help='eta',required=False, default = 1.0)
	parser.add_argument('--ell_max', dest='ell_max', action='store', type=int, help='ell_max',required=False, default = 3000)
	parser.add_argument('--delensing', dest='delensing', action='store_true', required=False, default=0)
	parser.add_argument('--T_only_in_d', dest='T_only_in_d', action='store_true', required=False, default=0)
	parser.add_argument('--P_only_in_d', dest='P_only_in_d', action='store_true', required=False, default=0)

	args = parser.parse_args()
	args.time_of_integration *= 31536000.0

	return args

if __name__ == "__main__":

	main()
