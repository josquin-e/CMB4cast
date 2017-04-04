#!/usr/bin/env python

import os 
import scipy
from scipy import interpolate
import numpy as np
import CMB4cast_utilities as CMB4U
import CMB4cast_compsep as CMB4CS
import pickle
from collections import OrderedDict
import healpy as hp
import copy 
import scipy
from scipy import polyval, polyfit, optimize
import pylab as pl

########################################################################################

'''
CMB4cast computations prior to the component separation code
It computes the spsp matrix as well as the angular power spectra 
for the principal sky components = CMB, dust, synchrotron, and the first order
expansion along beta_dust, beta_sync. These quantities are evaluated for several
galactic masks and interpolated between fraction of skies.
'''

############################################################################

def logl( Cl0, Cl1 ):
	return np.sum( np.log(Cl0) + Cl1/Cl0 )

########################################################################################

def interpolation_of_Cell(in_ell=0, in_Cell=0, out_ell=0,type_int='power_law'):

	if type_int=='power_law':
		# interpolate the C_ell assuming a power law
		def Cell_ell_dependence( ell, p0, p1 ):
			return p0 * (ell**p1)
		def logl_model_for_Cl( p ):
			A, b = p
			return logl( Cell_ell_dependence( in_ell, A, b ), in_Cell )
		popt_loc = optimize.fmin( logl_model_for_Cl, x0=[1,-1] )
		Cell_interpolated = Cell_ell_dependence( out_ell, popt_loc[0], popt_loc[1] )
	elif type_int=='extrapolated_quad':
		# interpolate the C_ell assuming a second ord
		def Cell_ell_dependence2( ell, p0, p1, p2 ):
			return 10**( np.log10(p0) + p1*np.log10(ell) + \
					p2*(np.log10(ell))**2 )
		def logl_model_for_Cl2( p ):
			A, b, c = p
			return logl( Cell_ell_dependence2(in_ell, A, b, c), in_Cell )
		popt_loc2 = optimize.fmin( logl_model_for_Cl2, x0=[1,-1,-1] )
		Cell_interpolated = Cell_ell_dependence2( out_ell, popt_loc2[0], \
								popt_loc2[1], popt_loc2[2] )
	elif type_int=='extrapolated_cubic':
		# interpolate the C_ell assuming a second ord
		def Cell_ell_dependence2( ell, p0, p1, p2, p3 ):
			return 10**( np.log10(p0) + p1*np.log10(ell) + \
					p2*(np.log10(ell))**2 + p3*(np.log10(ell))**3 )
		def logl_model_for_Cl2( p ):
			A, b, c, d = p
			return logl( Cell_ell_dependence2(in_ell, A, b, c, d), in_Cell )
		popt_loc2 = optimize.fmin( logl_model_for_Cl2, x0=[1,-1,-1,-1] )
		Cell_interpolated = Cell_ell_dependence2( out_ell, popt_loc2[0], \
								popt_loc2[1], popt_loc2[2], popt_loc2[3] )
	else:
		print 'type_int not understood ... '
		exit()

	return Cell_interpolated

########################################################################################

def maps_template( common_nside=128, \
				 analytic_expr_per_template_sky = {},\
				 fiducial_spectral_parameters_sky={},\
				 path2maps='./', r_fid=0.001, no_Cl_interpolation=False):
	""" 
	@brief: this function aims at being replace with whatever foregrounds/c,b templates
	and corresponding angular power spectra are provided.
	Given a fsky, sky resolution (nside), and fiducial cosmology, the function returns
	the sum_{sky pixel} s.s^T matrix, as well as all the auto- and cross-angular power spectra
	"""

	mask = hp.read_map( os.path.join( path2maps, 'HFI_Mask_GalPlane-apo2_2048_R2.00.fits'), field=(0,1,2,3,4,5))
	# mask_2 = hp.read_map( os.path.join( path2maps, '4f_depth_uKamin.fits'), field=(0))
	mask_5 = hp.read_map( os.path.join( path2maps, 'mask_02000.fits'), field=(0))
	mask_10 = hp.read_map( os.path.join( path2maps, 'mask_04000.fits'), field=(0))
	if hp.npix2nside(len(mask[0]))!= common_nside: mask = hp.ud_grade(copy.deepcopy(mask), nside_out=common_nside)
	# if hp.npix2nside(len(mask_2))!= common_nside: mask_2 = hp.ud_grade(copy.deepcopy(mask_2), nside_out=common_nside)
	if hp.npix2nside(len(mask_5))!= common_nside: mask_5 = hp.ud_grade(copy.deepcopy(mask_5), nside_out=common_nside)
	if hp.npix2nside(len(mask_10))!= common_nside: mask_10 = hp.ud_grade(copy.deepcopy(mask_10), nside_out=common_nside)
	# ind = np.where((mask_2 != 0.0)&(mask_2>0.0)&(mask_2<50.0))[0]
	# mask_2 = np.zeros(mask_2.shape)
	# mask_2[ ind ] = 1.0
	# np.save('/Users/josquin1/Documents/Dropbox/CNRS-CR2/POLARBEAR/PB_large_patch_analysis/mask_2', mask_2)
	# exit()
	mask_tot = np.zeros((9, len(mask_5)))
	mask_tot[0,:]=mask_5*0.0
	mask_tot[1,:]=mask_5
	mask_tot[2,:]=mask_10
	mask_tot[3:,:]=mask

	mask = mask_tot*1.0
	del mask_tot

	templates_sky = analytic_expr_per_template_sky.keys()

	dust = hp.read_map( os.path.join( path2maps, 'COM_CompMap_DustPol-commander_1024_R2.00.fits'), field=(0,1,2))
	sync = hp.read_map( os.path.join( path2maps, 'COM_CompMap_SynchrotronPol-commander_0256_R2.00.fits'), field=(0,1,2))
	if (('dBQdust' in  templates_sky) or ('dBUdust' in templates_sky)): 
		dust_params = hp.read_map(os.path.join( path2maps, 'thermaldust_spec_1.fits'))
	if (('dBQsync' in  templates_sky) or ('dBUsync' in templates_sky)):		
		sync_params = hp.read_map(os.path.join( path2maps, 'synchrotron_specind.fits'))

	#### synfast a CMB map with correct cosmology
	print 'GENERATING A POLARIZED CMB MAP WITH THE CORRECT COSMOLOGY'
	##########################################################################################################################################
	## loading some cosmological power spectra
	all_Cls_CMB = CMB4U.load_obj(path2maps, 'fidCls_A_0.1_tau_0.055_A_fgs_res_1.0_Neff_3.046_k_scalar_0.05_wa_0.0_alphas_0.0_omk_0.0_h_67.74_b_-0.8_omnuh2_0.0006451439_YHe_0.2453_As_2.142e-09_r_0.001_omch2_0.1188_b_fgs_res_-2.0_w_-1.0_k_tensor_0.002_ns_0.9667_nT_-0.000125_ombh2_0.0223.pkl')
	# renormalization of CMB power spectra
	all_Cls_CMB['BuBu_r1'] = all_Cls_CMB['BuBu']/0.001
	all_Cls_CMB['BuBu'] = all_Cls_CMB['BuBu']*r_fid/0.001
	all_Cls_CMB['BB'] = all_Cls_CMB['BuBu'] + all_Cls_CMB['BlBl']
	# TT, EE, BB, TE, EB, TB 
	Cls_loc = {}
	for key1 in ['TT', 'TE', 'EE', 'BB', 'ell', 'BuBu', 'BuBu_r1']:
		if key1 != 'ell':
			Cls_loc[key1] = np.hstack(( [0.0, 0.0], all_Cls_CMB[key1][:] ))
		else:
			Cls_loc[key1] = np.hstack(( [0, 1], all_Cls_CMB[key1][:] ))
	
	norm_loc = Cls_loc['ell']*(Cls_loc['ell']+1)/(2*np.pi)
	
	cls_loc = (Cls_loc['TT']/norm_loc,Cls_loc['EE']/norm_loc, \
				Cls_loc['BB']/norm_loc,Cls_loc['TE']/norm_loc, \
						Cls_loc['TT']*0.0,Cls_loc['TT']*0.0 )
	# norm = all_Cls_CMB['ell']*(all_Cls_CMB['ell']+1)/(2*np.pi)
	# cls = (all_Cls_CMB['TT']/norm,all_Cls_CMB['EE']/norm,all_Cls_CMB['BB']/norm,all_Cls_CMB['TE']/norm,all_Cls_CMB['TT']*0.0,all_Cls_CMB['TT']*0.0 )
	cmb = hp.synfast( cls_loc, nside=common_nside, pol=True, new=True, verbose=True)

	print 'converting maps to a common nside'
	if hp.npix2nside(len(cmb[0]))!= common_nside: cmb = hp.ud_grade(copy.deepcopy(cmb), nside_out=common_nside)
	if hp.npix2nside(len(sync[0]))!= common_nside: sync = hp.ud_grade(copy.deepcopy(sync), nside_out=common_nside)
	if hp.npix2nside(len(dust[0]))!= common_nside: dust = hp.ud_grade(copy.deepcopy(dust), nside_out=common_nside)
	if (('dBQdust' in  templates_sky) or ('dBUdust' in templates_sky)): 
		if hp.npix2nside(len(dust_params))!= common_nside: dust_params = hp.ud_grade(copy.deepcopy(dust_params), nside_out=common_nside)
	if (('dBQsync' in  templates_sky) or ('dBUsync' in templates_sky)):
		if hp.npix2nside(len(sync_params))!= common_nside: sync_params = hp.ud_grade(copy.deepcopy(sync_params), nside_out=common_nside)

	## check if stolyarov components for dust or synchrotron
	if (('dBQdust' in  templates_sky) or ('dBUdust' in templates_sky)):
		print 'there is a first order expansion along beta_dust !'
		stolyarov_Bd_true_A = True
	else:
		stolyarov_Bd_true_A = False
	if (('dBQsync' in  templates_sky) or ('dBUsync' in templates_sky)):
		print 'there is a first order expansion along beta_sync !'
		stolyarov_Bs_true_A = True
	else:
		stolyarov_Bs_true_A = False
	# check if any missing templates...
	templates_to_check = ['cmb', 'dust', 'sync']
	if any(temp in templates_sky for temp in templates_to_check):
		print 'there are unknown templates i.e. sky signals which do not have corresponding maps nor power spectra: '
		print templates_sky
		exit()

	analytic_expr_per_template_sky_LOC = OrderedDict([ \
		 ('Qcmb','(nu / cst) ** 2 * ( exp ( nu / cst ) ) / ( ( exp ( nu / cst ) - 1 ) ** 2 )'),\
		 ('Ucmb','(nu / cst) ** 2 * ( exp ( nu / cst ) ) / ( ( exp ( nu / cst ) - 1 ) ** 2 )'),\
		 ('Qdust','( exp( nu_ref / (Td / h_over_k ) ) - 1 ) / ( exp( nu / ( Td / h_over_k ) ) - 1 ) * ( nu / nu_ref ) ** ( 1 + Bd + drun * log( nu / nu_ref ) )'),\
		 ('Udust','( exp( nu_ref / (Td / h_over_k ) ) - 1 ) / ( exp( nu / ( Td / h_over_k ) ) - 1 ) * ( nu / nu_ref ) ** ( 1 + Bd + drun * log( nu / nu_ref ) )'),\
		 ('Qsync','( nu / nu_ref ) ** (Bs + srun * log( nu / nu_ref ) )'),\
		 ('Usync','( nu / nu_ref ) ** (Bs + srun * log( nu / nu_ref ) )') ] )

	# conversion factor from 353 to 150, and from 30 to 150.
	A353 = CMB4CS.A_matrix_builder( analytic_expr_per_template=analytic_expr_per_template_sky_LOC, \
			frequency_stokes=CMB4U.frequency_stokes_default, frequencies=[353.0], drv='',\
			spectral_parameters=fiducial_spectral_parameters_sky, bandpass_channels={})

	A30 = CMB4CS.A_matrix_builder( analytic_expr_per_template=analytic_expr_per_template_sky_LOC, \
			frequency_stokes=CMB4U.frequency_stokes_default, frequencies=[30.0], drv='',\
			spectral_parameters=fiducial_spectral_parameters_sky, bandpass_channels={})

	A150 = CMB4CS.A_matrix_builder( analytic_expr_per_template=analytic_expr_per_template_sky_LOC, \
		frequency_stokes=CMB4U.frequency_stokes_default, frequencies=[150.0], drv='',\
		spectral_parameters=fiducial_spectral_parameters_sky, bandpass_channels={})

	# check if dimensions of these mixing matrices are the same
	assert A353['in'] == A30['in'] == A150['in']
	# find the correct indices
	indi_d = A150['in'].index('Qdust')
	indi_s = A150['in'].index('Qsync')
	indo = A150['out'].index('Q150.0GHz')

	from_dust_353RJ_150RJ = A150['matrix'][indo,indi_d]/A353['matrix'][indo,indi_d] 
	from_sync_30RJ_150RJ =  A150['matrix'][indo,indi_s]/A30['matrix'][indo,indi_s]

	# adding fsky = 0% to the series of masks
	# mask  = np.vstack( (mask[0]*0.0, mask, np.ones(mask[0].shape)) )
	mask  = np.vstack( ( mask, np.ones(mask[0].shape)) )

	# fskys_planck = np.array([0.0,0.2,0.4,0.6,0.7,0.8,0.9,1.0])
	# fskys_planck = np.array([0.02, 0.05, 0.1, 0.2,0.4,0.6,0.7,0.8,0.9,1.0])
	fskys_planck = np.array([0.0, 0.05, 0.1, 0.2, 0.4,0.6,0.7,0.8,0.9,1.0])
	fskys_planck_int = np.arange( 0.01, 1.0, 0.01 ) # estimates spsp for many sky masks with a 1% step
	
	templates = ['Qcmb', 'Ucmb']
	if ('Qdust' in analytic_expr_per_template_sky.keys()):
		templates.append('Qdust') 
	if ('Udust' in analytic_expr_per_template_sky.keys()):
		templates.append('Udust')

	# keep cmb full sky 
	Q_cmb = cmb[1][:]*1.0#/BB_factor_computation(150)
	U_cmb = cmb[2][:]*1.0#/BB_factor_computation(150)	
	sp_unmasked = np.vstack(( Q_cmb.flatten(), U_cmb.flatten() ))
	
	# loop over sky masks
	for m in range(len(fskys_planck)):

		print '-------------------------------- '
		print 'masking maps with a fsky ~ ', len( np.where( mask[m][:]!=0.0 )[0] )*1.0/len( mask[m][:] )
		mask[m][ np.where( mask[m][:]!=0.0 )[0] ] = 1.0
		mask[m][ np.where( mask[m][:]==0.0 )[0] ] = 0.0
		Q_cmb_mask = mask[m]*cmb[1][:]*1.0 
		U_cmb_mask = mask[m]*cmb[2][:]*1.0 
		Q_dust_mask = mask[m]*dust[0]*from_dust_353RJ_150RJ
		U_dust_mask = mask[m]*dust[1]*from_dust_353RJ_150RJ
		delta_Bd_mask = mask[m]*( dust_params - np.mean(dust_params[np.where( mask[m][:]!=0.0 )[0]]) )
		Bd_mask = mask[m]*( dust_params )
		Q_sync_mask = mask[m]*sync[0]*from_sync_30RJ_150RJ
		U_sync_mask = mask[m]*sync[1]*from_sync_30RJ_150RJ
		delta_Bs_mask = mask[m]*( sync_params - np.mean(sync_params[np.where( mask[m][:]!=0.0 )[0]]) )
		Bs_mask = mask[m]*( sync_params )

		# definition of sp
		print 'creation of sp & computation of spsp'
		sp = np.vstack(( Q_cmb_mask.flatten(), U_cmb_mask.flatten() ))
		if ('Qdust' in analytic_expr_per_template_sky.keys()):
			sp = np.vstack(( sp, Q_dust_mask.flatten() ))
		if ('Udust' in analytic_expr_per_template_sky.keys()):
			sp = np.vstack(( sp, U_dust_mask.flatten() ))

		if stolyarov_Bd_true_A:
			print 'you chose stolyarov option, will have two extra rows & columns'
			if ('dBQdust' in analytic_expr_per_template_sky.keys()):
				sp = np.vstack(( sp, (delta_Bd_mask*Q_dust_mask).flatten() ))
				if m==0: 
					templates.append( 'dBQdust' )
			if ('dBUdust' in analytic_expr_per_template_sky.keys()):
				sp = np.vstack(( sp,  (delta_Bd_mask*U_dust_mask).flatten() ))
				if m==0: 
					templates.append( 'dBUdust' ) 
			if ('Qsync' in analytic_expr_per_template_sky.keys()):
				sp = np.vstack(( sp, Q_sync_mask.flatten() ))
				if m==0: 
					templates.append( 'Qsync' )
			if ('Usync' in analytic_expr_per_template_sky.keys()):
				sp = np.vstack(( sp, U_sync_mask.flatten() ))
				if m==0: 
					templates.append( 'Usync' )
		else:
			if ('Qsync' in analytic_expr_per_template_sky.keys()):
				sp = np.vstack(( sp, Q_sync_mask.flatten() ))
				if m==0: 
					templates.append( 'Qsync' )
			if ('Usync' in analytic_expr_per_template_sky.keys()):
				sp = np.vstack(( sp, U_sync_mask.flatten() ))
				if m==0: 
					templates.append( 'Usync' )

		if stolyarov_Bs_true_A:
			if ('dBQsync' in analytic_expr_per_template_sky.keys()):
				sp = np.vstack(( sp, (delta_Bs_mask*Q_sync_mask).flatten() ))
				if m==0: 
					templates.append( 'dBQsync' )
			if ('dBUsync' in analytic_expr_per_template_sky.keys()):
				sp = np.vstack(( sp, (delta_Bs_mask*U_sync_mask).flatten() ))
				if m==0: 
					templates.append( 'dBUsync' ) 

		# computation of spsp
		spsp = np.zeros( (len(sp), len(sp)) )
		for i in range(len(sp)):
			for j in range(len(sp)):
				spsp[i,j] = np.sum( sp[i,:] * sp[j,:] )

		if m == 0:
			spsp_tot = np.zeros((len(mask), spsp.shape[0], spsp.shape[1]))
			sp_tot = np.zeros((len(mask), sp.shape[0], sp.shape[1]))
			Bs_tot =  np.zeros((len(mask), len(delta_Bs_mask)))
			Bd_tot =  np.zeros((len(mask), len(delta_Bs_mask)))
			obs_pix = np.zeros((len(mask), len(delta_Bs_mask)))


		obs_pix[m,np.where( mask[m][:]!=0.0 )[0]] = 1.0
		sp_tot[m,:,:] = copy.deepcopy(sp)*1.0

		# setting the masked sky as input CMB
		ind_masked= np.where( mask[m][:]==0.0 )[0]
		sp_tot[m,:2,ind_masked] = sp_unmasked[:,ind_masked].T*1.0

		spsp_tot[m,:,:] = spsp*1.0
		Bs_tot[m,:] = delta_Bs_mask*1.0
		Bd_tot[m,:] = delta_Bd_mask*1.0

	################## end of loop
	# spsp_out = spsp*0.0
	# 
	sp_output = OrderedDict()
	sp_output['matrix'] = sp_tot*1.0
	sp_output['Bd'] = Bd_tot*1.0
	sp_output['obs_pix'] = obs_pix*1.0

	# sp_output['area_Bd'] = find_constant_areas(sp_output['Bd'], threshold=0.05)

	sp_output['Bs'] = Bs_tot*1.0 #- 2.0
	sp_output['out'] = templates
	sp_output['in'] = 'pixels'

	sp_output['fsky'] = fskys_planck

	# define the output dictionary for spsp
	spsp_output = OrderedDict()
	spsp_output['in'] = templates
	spsp_output['out'] = templates
	spsp_output['matrix'] = np.zeros((len(fskys_planck_int),spsp.shape[0],spsp.shape[1]))
	spsp_output['fsky'] = fskys_planck_int
	spsp_output['obs_pix'] = obs_pix*1.0

	## do the interpolation over fsky
	for i in range(spsp.shape[0]):
		for j in range(spsp.shape[1]):
			f_int = scipy.interpolate.interp1d(fskys_planck, spsp_tot[:,i,j], kind='slinear', bounds_error=False)
			for f in range(len(fskys_planck_int)):
				spsp_output['matrix'][f,i,j] = f_int( fskys_planck_int[f] )*1.0
			del f_int

	## number of observed pixels
	npix = len( np.where( mask[m][:]!=0.0 )[0] )*1.0
	spsp_output['npix']= npix

	return sp_output, spsp_output, npix, obs_pix, all_Cls_CMB

#############################################################################################
#############################################################################################

def Cls_template( common_nside=128,  \
				 analytic_expr_per_template_sky = {},\
				 fiducial_spectral_parameters_sky={},\
				 path2maps='./', r_fid=0.001, no_Cl_interpolation=False,\
				 all_Cls_CMB={}, sp_output={}, spsp_output={} ):
	""" 
	@brief: this function provides the auto- and cross-angular power spectra
	from the sky templates computed a priori
	"""

	## check if stolyarov components for dust or synchrotron
	templates_sky = analytic_expr_per_template_sky.keys()
	if (('dBQdust' in  templates_sky) or ('dBUdust' in templates_sky)):
		print 'there is a first order expansion along beta_dust !'
		stolyarov_Bd_true_A = True
	else:
		stolyarov_Bd_true_A = False
	if (('dBQsync' in  templates_sky) or ('dBUsync' in templates_sky)):
		print 'there is a first order expansion along beta_sync !'
		stolyarov_Bs_true_A = True
	else:
		stolyarov_Bs_true_A = False

	# check if any missing templates...
	templates_to_check = ['cmb', 'dust', 'sync']
	if any(temp in templates_sky for temp in templates_to_check):
		print 'there are unknown templates i.e. sky signals which do not have corresponding maps nor power spectra: '
		print templates_sky
		exit()

	analytic_expr_per_template_sky_LOC = OrderedDict([ \
		 ('Qcmb','(nu / cst) ** 2 * ( exp ( nu / cst ) ) / ( ( exp ( nu / cst ) - 1 ) ** 2 )'),\
		 ('Ucmb','(nu / cst) ** 2 * ( exp ( nu / cst ) ) / ( ( exp ( nu / cst ) - 1 ) ** 2 )'),\
		 ('Qdust','( exp( nu_ref / (Td / h_over_k ) ) - 1 ) / ( exp( nu / ( Td / h_over_k ) ) - 1 ) * ( nu / nu_ref ) ** ( 1 + Bd + drun * log( nu / nu_ref ) )'),\
		 ('Udust','( exp( nu_ref / (Td / h_over_k ) ) - 1 ) / ( exp( nu / ( Td / h_over_k ) ) - 1 ) * ( nu / nu_ref ) ** ( 1 + Bd + drun * log( nu / nu_ref ) )'),\
		 ('Qsync','( nu / nu_ref ) ** (Bs + srun * log( nu / nu_ref ) )'),\
		 ('Usync','( nu / nu_ref ) ** (Bs + srun * log( nu / nu_ref ) )') ] )

	###############################################################
	## FOREGROUNDS POWER SPECTRA
	###############################################################
	# putting everything in the output dictionaries 

	fskys_int = spsp_output['fsky']
	fskys_mask = sp_output['fsky']

	Cls = copy.deepcopy( all_Cls_CMB )
	for key in ['dust', 'sync', 'dxs', 'dBdxdBd', 'dBdxdust', 'dBsxdBs', 'dBsxsync']:
		Cls[key] = np.zeros(( len(fskys_mask), len(all_Cls_CMB['ell']) ))

	ells = Cls['ell']*1.0


	# pl.figure()
	# import matplotlib.cm as cm
	# colors = cm.rainbow(np.linspace(0, 1, len(fskys_mask)))

	for ind_fsky in range( len(fskys_mask) ):

		iQdust = sp_output['out'].index('Qdust')
		iUdust = sp_output['out'].index('Udust')
		iQsync = sp_output['out'].index('Qsync')
		iUsync = sp_output['out'].index('Usync')

		map1 = copy.deepcopy( (sp_output['matrix'][ind_fsky,iQdust,:], sp_output['matrix'][ind_fsky,iQdust,:], sp_output['matrix'][ind_fsky,iUdust,:]))

		a,b,Cls_dust,c,d,e = hp.sphtfunc.anafast( map1=map1, iter=5, lmax=2*common_nside)

		map1 = copy.deepcopy((sp_output['matrix'][ind_fsky,iQsync,:], sp_output['matrix'][ind_fsky,iQsync,:], sp_output['matrix'][ind_fsky,iUsync,:]))

		a,b,Cls_sync,c,d,e = hp.sphtfunc.anafast( map1=map1, iter=5, lmax=2*common_nside )

		map1 = copy.deepcopy((sp_output['matrix'][ind_fsky,iQdust,:], sp_output['matrix'][ind_fsky,iQdust,:], sp_output['matrix'][ind_fsky,iUdust,:]))
		map2 = copy.deepcopy((sp_output['matrix'][ind_fsky,iQsync,:], sp_output['matrix'][ind_fsky,iQsync,:], sp_output['matrix'][ind_fsky,iUsync,:]))

		a,b,Cls_dxs,c,d,e = hp.sphtfunc.anafast( map1=map1, map2=map2, iter=5, lmax=2*common_nside)


		ells_loc = np.arange(0, len(Cls_dust)) 
		norm_loc = ells_loc*(ells_loc+1) / (2*np.pi) / fskys_mask[ind_fsky]
		Cls_dust *= norm_loc
		Cls_sync *= norm_loc
		Cls_dxs *= norm_loc

		# interpolate
		if not no_Cl_interpolation:
			Cls_dust_int = interpolation_of_Cell( in_ell=ells_loc[10:20], \
											in_Cell=np.abs(Cls_dust)[10:20], out_ell=ells ,\
											type_int='power_law' )
			Cls_dxs_int = interpolation_of_Cell( in_ell=ells_loc[10:20], \
											in_Cell=np.abs(Cls_dxs)[10:20], out_ell=ells ,\
											type_int='power_law' )
			Cls_sync_int = interpolation_of_Cell( in_ell=ells_loc[10:20], \
											in_Cell=np.abs(Cls_sync)[10:20], out_ell=ells ,\
											type_int='power_law' )

		# del Cls_dust, Cls_dxs, Cls_sync

		Cls_dust_int = np.hstack( ( Cls_dust_int[2:], np.zeros(np.max(Cls['ell'])-len(Cls_dust_int[2:])-1) ) )
		Cls_sync_int = np.hstack( ( Cls_sync_int[2:], np.zeros(np.max(Cls['ell'])-len(Cls_sync_int[2:])-1) ) )
		Cls_dxs_int = np.hstack( ( Cls_dxs_int[2:], np.zeros(np.max(Cls['ell'])-len(Cls_dxs_int[2:])-1) ) )

		if ind_fsky == 0:
			Cls_dust_int *= 0.0
			Cls_sync_int *= 0.0
			Cls_dxs_int *= 0.0

		Cls['dust'][ind_fsky,:] = copy.deepcopy( Cls_dust_int*1.0 )
		Cls['sync'][ind_fsky,:] = copy.deepcopy( Cls_sync_int*1.0 )
		Cls['dxs'][ind_fsky,:] = copy.deepcopy( Cls_dxs_int*1.0 )

		# pl.loglog( ells,Cls['dust'][ind_fsky,:], color=colors[ind_fsky], linestyle='-', label=fskys_mask[ind_fsky])
		# pl.loglog( ells_loc[10:20], np.abs(Cls_dust)[10:20], color=colors[ind_fsky], linestyle=':', linewidth=3.0 )
		# pl.loglog( ells,Cls['sync'][ind_fsky,:], color=colors[ind_fsky], linestyle='--')
		# pl.loglog( ells_loc[10:20], np.abs(Cls_sync)[10:20], color=colors[ind_fsky], linestyle=':', linewidth=3.0 )
		# pl.loglog( ells,Cls['dxs'][ind_fsky,:], color=colors[ind_fsky], linestyle=':')
		# pl.loglog( ells_loc[10:20], np.abs(Cls_dxs)[10:20], color=colors[ind_fsky], linestyle=':', linewidth=3.0 )

		del Cls_dust_int, Cls_sync_int, Cls_dxs_int

		###############################################################
		## computing other angular power spectra for dust stolyarov

		print 'stolyarov for Bd is ON, computing power spectra for dBd x dust and dBd x dBd '
		# find the good maps
		ind_Qdust = sp_output['out'].index('Qdust')
		ind_dBQdust = sp_output['out'].index('dBQdust')
		ind_Udust = sp_output['out'].index('Udust')
		ind_dBUdust = sp_output['out'].index('dBUdust')

		Q_dBd_dust = sp_output['matrix'][ind_fsky,ind_dBQdust,:]
		U_dBd_dust = sp_output['matrix'][ind_fsky,ind_dBUdust,:]
		Q_dust = sp_output['matrix'][ind_fsky,ind_Qdust,:]
		U_dust = sp_output['matrix'][ind_fsky,ind_Udust,:]

		# angular power spectra computation
		a,b, Cl_dBdxdBd, c,d,e = hp.sphtfunc.anafast( map1=(Q_dBd_dust, Q_dBd_dust, U_dBd_dust), \
								iter=5, lmax=2*common_nside) #, lmax=np.max(ells) )
		a,b, Cl_dBdxdust, c,d,e = hp.sphtfunc.anafast( map1=(Q_dBd_dust, Q_dBd_dust, U_dBd_dust), \
								map2=(Q_dBd_dust, Q_dust, U_dust ), \
								iter=5, lmax=2*common_nside) #, lmax=np.max(ells) )

		# renormalization of the Cls..
		ells_loc = np.arange(0, len(Cl_dBdxdust)) 
		norm = ells_loc*(ells_loc+1)/(2*np.pi) /fskys_mask[ind_fsky]
		Cl_dBdxdBd *= norm
		Cl_dBdxdust *= norm
		
		# interpolate
		if not no_Cl_interpolation:
			Cl_dBdxdBd_int = interpolation_of_Cell( in_ell=ells_loc[10:20], \
								in_Cell=np.abs(Cl_dBdxdBd)[10:20], out_ell=ells,\
								type_int='power_law' )
			Cl_dBdxdust_int = interpolation_of_Cell( in_ell=ells_loc[10:20], \
								in_Cell=np.abs(Cl_dBdxdust)[10:20], out_ell=ells,\
								type_int='power_law' )

		# should pad these till the ell_max 
		Cl_dBdxdBd_int = np.hstack( ( Cl_dBdxdBd_int[2:], np.zeros(np.max(Cls['ell'])-len(Cl_dBdxdBd_int[2:])-1) ) )
		Cl_dBdxdust_int = np.hstack( ( Cl_dBdxdust_int[2:], np.zeros(np.max(Cls['ell'])-len(Cl_dBdxdust_int[2:])-1) ) )
		
		if ind_fsky == 0:
			Cl_dBdxdBd_int *= 0.0
			Cl_dBdxdust_int *= 0.0

		Cls['dBdxdBd'][ind_fsky,:] = Cl_dBdxdBd_int*1.0
		Cls['dBdxdust'][ind_fsky,:] = Cl_dBdxdust_int*1.0

		# pl.loglog( ells,Cls['dBdxdBd'][ind_fsky,:], color=colors[ind_fsky], linestyle='--')
		# pl.loglog( ells_loc[10:20], np.abs(Cl_dBdxdBd)[10:20], color=colors[ind_fsky], linestyle=':', linewidth=3.0 )
		# pl.loglog( ells,Cls['dBdxdust'][ind_fsky,:], color=colors[ind_fsky], linestyle='-')
		# pl.loglog( ells_loc[10:20], np.abs(Cl_dBdxdust)[10:20], color=colors[ind_fsky], linestyle=':', linewidth=3.0 )

		###############################################################
		## computing other angular power spectra for synchrotron stolyarov

		print 'stolyarov for Bs is ON, computing power spectra for dBs x sync and dBs x dBs '
		# find the good maps
		ind_Qsync = sp_output['out'].index('Qsync')
		ind_dBQsync = sp_output['out'].index('dBQsync')
		ind_Usync = sp_output['out'].index('Usync')
		ind_dBUsync = sp_output['out'].index('dBUsync')

		Q_dBs_sync = sp_output['matrix'][ind_fsky,ind_dBQsync,:]
		U_dBs_sync = sp_output['matrix'][ind_fsky,ind_dBUsync,:]
		Q_sync = sp_output['matrix'][ind_fsky,ind_Qsync,:]
		U_sync = sp_output['matrix'][ind_fsky,ind_Usync,:]

		# angular power spectra computation
		a,b, Cl_dBsxdBs, c,d,e = hp.sphtfunc.anafast( map1=(Q_dBs_sync, Q_dBs_sync, U_dBs_sync), \
											iter=5, lmax=2*common_nside)#, lmax=np.max(ells) )
		a,b, Cl_dBsxsync, c,d,e = hp.sphtfunc.anafast( map1=(Q_dBs_sync, Q_dBs_sync, U_dBs_sync), \
											map2=(Q_sync, Q_sync, U_sync ), \
											iter=5, lmax=2*common_nside)#, lmax=np.max(ells) )

		# renormalization of the Cls..
		ells_loc = np.arange(0, len(Cl_dBsxsync)) 
		norm = ells_loc*(ells_loc+1)/(2*np.pi) /fskys_mask[ind_fsky]
		Cl_dBsxdBs *= norm
		Cl_dBsxsync *= norm

		if not no_Cl_interpolation:
			# interpolation of C_ell
			Cl_dBsxdBs_int = interpolation_of_Cell( in_ell=ells_loc[10:20], \
							in_Cell=np.abs(Cl_dBsxdBs)[10:20], out_ell=ells,\
							type_int='power_law' )
			Cl_dBsxsync_int = interpolation_of_Cell( in_ell=ells_loc[10:20], \
							in_Cell=np.abs(Cl_dBsxsync)[10:20], out_ell=ells,\
							type_int='power_law' )
		
		# should pad these till the ell_max 
		Cl_dBsxdBs_int = np.hstack( ( Cl_dBsxdBs_int[2:], np.zeros(np.max(Cls['ell'])-len(Cl_dBsxdBs_int[2:])-1) ) )
		Cl_dBsxsync_int = np.hstack( ( Cl_dBsxsync_int[2:], np.zeros(np.max(Cls['ell'])-len(Cl_dBsxsync_int[2:])-1) ) )
		
		if ind_fsky == 0:
			Cl_dBsxdBs_int *= 0.0
			Cl_dBsxsync_int *= 0.0

		Cls['dBsxdBs'][ind_fsky,:] = Cl_dBsxdBs_int*1.0 
		Cls['dBsxsync'][ind_fsky,:] = Cl_dBsxsync_int*1.0 
	
	# 	pl.loglog( ells,Cls['dBsxdBs'][ind_fsky,:], color=colors[ind_fsky], linestyle='--')
	# 	pl.loglog( ells_loc[10:20], np.abs(Cl_dBsxdBs)[10:20], color=colors[ind_fsky], linestyle=':', linewidth=3.0 )
	# 	pl.loglog( ells,Cls['dBsxsync'][ind_fsky,:], color=colors[ind_fsky], linestyle='-')
	# 	pl.loglog( ells_loc[10:20], np.abs(Cl_dBsxsync)[10:20], color=colors[ind_fsky], linestyle=':', linewidth=3.0 )

	# pl.legend()
	# pl.show()
	# exit()

	##################################
	## INTERPOLATION OVER FSKY 
	print 'interpolation over frequencies for each ell'
	Cls_int = copy.deepcopy(Cls)
	for key in ['dust', 'sync', 'dxs', 'dBdxdBd', 'dBdxdust', 'dBsxdBs', 'dBsxsync']:
		Cls_int[key] = np.zeros((len(fskys_int), Cls[key].shape[1] ))

	for key in ['dust', 'sync', 'dxs', 'dBdxdBd', 'dBdxdust', 'dBsxdBs', 'dBsxsync']:
		for l in range(Cls[key].shape[1]):
			f_int = scipy.interpolate.interp1d(fskys_mask, Cls[key][:,l], kind='slinear', bounds_error=False)
			for f in range(len(fskys_int)):
				Cls_int[key][f,l] = f_int( fskys_int[f] )*1.0
				# print key, f, l, Cls_int[key][f,l]
			del f_int
			# exit()
	##################################
	# print 'plotting ... '
	# import matplotlib.cm as cm
	# colors = cm.rainbow(np.linspace(0, 1, len(fskys_int)))
	# for key in ['dust', 'sync', 'dxs', 'dBdxdBd', 'dBdxdust', 'dBsxdBs', 'dBsxsync']:
	# 	print key
	# 	pl.figure()
	# 	pl.title(key, fontsize=20)
	# 	for f in range(len(fskys_int)):
	# 		# print Cls_int[key][f,:]
	# 		pl.loglog( np.abs(Cls_int[key][f,:]), color=colors[f], linestyle='-')
	# pl.show()
	# exit()
	##################################


	Cls_int['fsky'] = fskys_int*1.0
	Cls_int['foregrounds_keys'] = ['dust', 'sync', 'dxs', 'dBdxdBd', 'dBdxdust', 'dBsxdBs', 'dBsxsync']

	return Cls_int

#############################################################################################
#############################################################################################
# PARAMETERS 

common_nside = CMB4U.common_nside

analytic_expr_per_template_sky = OrderedDict([ \
		 ('Qcmb','(nu / cst) ** 2 * ( exp ( nu / cst ) ) / ( ( exp ( nu / cst ) - 1 ) ** 2 )'),\
		 ('Ucmb','(nu / cst) ** 2 * ( exp ( nu / cst ) ) / ( ( exp ( nu / cst ) - 1 ) ** 2 )'),\
		 ('Qdust','( exp( nu_ref / (Td / h_over_k ) ) - 1 ) / ( exp( nu / ( Td / h_over_k ) ) - 1 ) * ( nu / nu_ref ) ** ( 1 + Bd + drun * log( nu / nu_ref ) )'),\
		 ('Udust','( exp( nu_ref / (Td / h_over_k ) ) - 1 ) / ( exp( nu / ( Td / h_over_k ) ) - 1 ) * ( nu / nu_ref ) ** ( 1 + Bd + drun * log( nu / nu_ref ) )'),\
		 ('dBQdust','( exp( nu_ref / (Td / h_over_k ) ) - 1 ) / ( exp( nu / ( Td / h_over_k ) ) - 1 ) * log( nu / nu_ref ) * ( nu / nu_ref ) ** ( 1 + Bd + drun * log( nu / nu_ref ) )'),\
		 ('dBUdust','( exp( nu_ref / (Td / h_over_k ) ) - 1 ) / ( exp( nu / ( Td / h_over_k ) ) - 1 ) * log( nu / nu_ref ) * ( nu / nu_ref ) ** ( 1 + Bd + drun * log( nu / nu_ref ) )'),\
		 ('Qsync','( nu / nu_ref ) ** (Bs + srun * log( nu / nu_ref ) )'),\
		 ('Usync','( nu / nu_ref ) ** (Bs + srun * log( nu / nu_ref ) )'),\
		 ('dBQsync','log( nu / nu_ref ) * ( nu / nu_ref ) ** (Bs + srun * log( nu / nu_ref ) )'),\
		 ('dBUsync','log( nu / nu_ref ) * ( nu / nu_ref ) ** (Bs + srun * log( nu / nu_ref ) )')])

fiducial_spectral_parameters_sky = { 'nu_ref':150.0, 'Bd':1.59, 'Td':19.6, 'h_over_k':CMB4U.h_over_k,\
					 'drun':0.0, 'Bs':-3.1, 'srun':0.0, 'theta_dust':0.0, 'Nlayers':1, 'cst':CMB4U.cst, 
					 'Bd_p':False, 'Bs_p':False}

r_fid = 0.0
path2maps = '/Users/josquin1/Documents/Dropbox/planck_maps'
path2products = '/Users/josquin1/Documents/Dropbox/planck_maps'
# path2products = '/Users/josquin1/Documents/Dropbox/self_consistent_forecast/CMB4cast/'

#############################################################################################
#############################################################################################

sp, spsp, npix, obs_pix, all_Cls_CMB = maps_template(common_nside=common_nside,
			analytic_expr_per_template_sky = analytic_expr_per_template_sky,\
			fiducial_spectral_parameters_sky=fiducial_spectral_parameters_sky,\
		 	path2maps=path2maps, r_fid=r_fid, no_Cl_interpolation=False) 

with open(os.path.join( path2products, 'spsp.pkl' ), 'wb') as f:
		pickle.dump(spsp, f, pickle.HIGHEST_PROTOCOL)

Cls = Cls_template( common_nside=common_nside, \
		 analytic_expr_per_template_sky = analytic_expr_per_template_sky,\
		 fiducial_spectral_parameters_sky=fiducial_spectral_parameters_sky,\
		 path2maps=path2maps, r_fid=r_fid, no_Cl_interpolation=False,\
		 all_Cls_CMB=all_Cls_CMB, sp_output=sp, spsp_output=spsp )

with open(os.path.join( path2products, 'Cls.pkl' ), 'wb') as f:
		pickle.dump(Cls, f, pickle.HIGHEST_PROTOCOL)

exit()
