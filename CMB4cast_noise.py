#!/usr/bin/env python

'''
CMB4cast noise estimation code
This code gives the noinse angular power spectrum 
for instruments and provided post comp sep white noise levels,
ell_knee and alpha_knee
'''

import residuals_computation_extended_self_consistent_forecast_calibration_errors as res_ext
import numpy as np
import CMB4cast_utilities as CMB4U

#######################################################################################################
### NOISE
########################################################################################################

def noise_computation( configurations={}, foregrounds={}, components_v=[''], resolution=True, \
			stolyarov=False, stolyarov_sync=False, ell_min_camb=2, Cls_fid={}, ells_exp=0.0, experiments=['']):

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

			Nl[exp] = {}

			for components in components_v:

				Nl[exp][components] = {}

				if components == 'cmb-only':
					# calculate quadratic noise combo including beams, then weight by 
					# ratio of white noise levels pre- and post-comp.-sep.
					Nl[exp][components]['TT'] = np.zeros(configurations[exp]['ell_max'] - ell_min_camb + 1)
					w_inv = ( np.array(configurations[exp]['uKCMBarcmin'][:] ) * CMB4U.arcmin_to_radian) ** 2
					for ell in range(ell_min_camb, configurations[exp]['ell_max'] + 1):
						beam_l = np.zeros(len(w_inv))
						for k in range(len(beam_l)):
							if ((configurations[exp]['alpha_knee'][k]!=0.0) and (configurations[exp]['ell_knee'][k]!=0.0)):
								factor = ( 1.0 + pow(configurations[exp]['ell_knee'][k]*1.0/ell, configurations[exp]['alpha_knee'][k]) )
							else:
								factor = 1.0
							beam_l[k] = factor*np.exp((np.array(configurations[exp]['FWHM'][k]) * CMB4U.arcmin_to_radian / np.sqrt(8.0*np.log(2.0))) ** 2 * (ell * (ell + 1.0)))
						Nl[exp][components]['TT'][ell - ell_min_camb] = ( (ell * (ell + 1.0) / (2.0 * np.pi)) / np.sum( 1.0 / w_inv / beam_l) )/2.0

					Nl[exp][components]['EE'] = Nl[exp][components]['TT'] * 2.0
					Nl[exp][components]['BB'] = Nl[exp][components]['TT'] * 2.0
					foregrounds[exp][components]['delta'] = 1.0
					foregrounds[exp][components]['sigma_CMB'] = 1.0/np.sqrt(np.sum( 1.0 / w_inv ))/CMB4U.arcmin_to_radian

				else:
					# noise after component separation
					w_inv_post = (foregrounds[exp][components]['uKCMB/pix_postcompsep'] * CMB4U.pix_size_map_arcmin * CMB4U.arcmin_to_radian) ** 2

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
								Ninv[ k, k ]  = (1.0/factor)*((1.0/sensitivity_uK_per_chan[ int(k/2) ])**2) * np.exp(  - ell*(ell+1.0)*( configurations[exp]['FWHM'][ int(k/2) ]*CMB4U.arcmin_to_radian )**2/(8*np.log(2)) )
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

							Nl[exp][components]['TT_post_comp_sep'][ell - ell_min_camb] = (ell*(ell+1.0)/(2*np.pi))*AtNAinv[0,0] * (CMB4U.pix_size_map_arcmin * CMB4U.arcmin_to_radian)**2/2.0
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
		
					ind0=np.argmin(np.abs(Cls_fid['ell'] - np.min(ells_exp)))
					ind1=np.argmin(np.abs(Cls_fid['ell'] - np.max(ells_exp)-1))

					Nl[exp][components]['TT_post_comp_sep'] +=  foregrounds[exp][components]['Cl_res'][ind0:ind1]
					Nl[exp][components]['EE_post_comp_sep'] +=  foregrounds[exp][components]['Cl_res'][ind0:ind1]
					Nl[exp][components]['BB_post_comp_sep'] +=  foregrounds[exp][components]['Cl_res'][ind0:ind1] 

					# computation of the noise degradation
					foregrounds[exp][components]['delta'] = w_inv_post*(np.sum(1.0 / w_inv))
					foregrounds[exp][components]['sigma_CMB'] = foregrounds[exp][components]['uKCMB/pix_postcompsep']*CMB4U.pix_size_map_arcmin

	return Nl
