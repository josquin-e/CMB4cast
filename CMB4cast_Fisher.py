
#!/usr/bin/env python

'''
CMB4cast noise estimation code
This code gives the noinse angular power spectrum 
for instruments and provided post comp sep white noise levels,
ell_knee and alpha_knee
'''

import glob
import CMB4cast_utilities as CMB4U
import python_camb_self_consistent
import os 
import forecasting_cosmo_self_consistent_forecast_smf as fc
import operator
import numpy as np
import copy 
import scipy
from scipy import optimize

def forecast_fisher(params_fid_v=[], camb='', params_dev_full='', information_channels=[''], configurations={},\
						components_v=[''], delensing_option_v=[''], Nl={}, path2Cls='', cross_only=False, no_lensing=False,\
						params_dev_v=[], DESI=False, param_priors_v=[], ell_min_camb=2, ells={}, foregrounds={},
						experiments=[], A_lens=1.0 ):
	"""
	@ brief: Fisher analysis from CMB power spectra with respect to any set of 
	cosmo parameters handled by CAMB
	"""
	# loop through base cosmologies
	sigmas = {}
	ind_pfid = 0

	for params_fid_loc in params_fid_v:

		# compute fiducial C_ls if not already pre-computed
		print '################################ computing fiducial Cls ... ####################################'
		name_fid = 'fidCls'
		fnames_fid = glob.glob( os.path.join(path2Cls,name_fid+'*.pkl' ))

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
			Cls_fid_loc=python_camb_self_consistent.submit_camb( h=params_fid_loc['h'], ombh2=params_fid_loc['ombh2'], omch2=params_fid_loc['omch2'], \
					omnuh2=params_fid_loc['omnuh2'], omk=params_fid_loc['omk'], YHe=params_fid_loc['YHe'], Neff=params_fid_loc['Neff'], w=params_fid_loc['w'], \
					wa=params_fid_loc['wa'], tau=params_fid_loc['tau'],As=params_fid_loc['As'], ns=params_fid_loc['ns'], alphas=params_fid_loc['alphas'], \
					nT=params_fid_loc['nT'], r=params_fid_loc['r'], k_scalar=params_fid_loc['k_scalar'] , k_tensor=params_fid_loc['k_tensor'], eta=1.0, exe = camb)
			CMB4U.save_obj('./', name_fid, Cls_fid_loc)
		else:
			print '################################ loading already existing Cls file ####################################'
			Cls_fid_loc = CMB4U.load_obj(path2Cls, fnames_fid[0])

		# compute derivatives once for all parameters
		dCldp = fc.derivatives_computation(Cls_fid_loc, params_dev_full, params_fid_loc, information_channels, exe=camb, path2Cls = path2Cls)

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
									ind0=np.argmin(np.abs(Cls_fid_loc['ell'] - np.min(ells[exp_loc])))
									ind1=np.argmin(np.abs(Cls_fid_loc['ell'] - np.max(ells[exp_loc])-1))
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
							
							if A_lens != 1.0:
								Cls_fid_loc['BB_delens'] = Cls_fid_loc['BB_delens']*A_lens

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
									return CMB4U.logl( residuals_ell_dependence( Cls_fid_loc['ell'], A, b ), foregrounds[exp_loc][components]['Cl_res'] )							
								params_fid_loc['b_fgs_res'] = optimize.fmin( logl_model_b, x0=-2.0 )

								# rescale of derivatives for residuals amplitude
								dCldp['A_fgs_res'] = {}
								dCldp['b_fgs_res'] = {}
								for key_loc in dCldp['ns'].keys():
									if 'd' not in key_loc:
										dCldp['A_fgs_res'][key_loc] = foregrounds[exp_loc][components]['Cl_res'][0]*(Cls_fid_loc['ell']/Cls_fid_loc['ell'][0])**( params_fid_loc['b_fgs_res'] )
										dCldp['b_fgs_res'][key_loc] = foregrounds[exp_loc][components]['Cl_res'][0]*params_fid_loc['A_fgs_res']*np.log(Cls_fid_loc['ell']/Cls_fid_loc['ell'][0])*(Cls_fid_loc['ell']/Cls_fid_loc['ell'][0])**( params_fid_loc['b_fgs_res'] )
									else:
										dCldp['A_fgs_res'][key_loc] = Cls_fid_loc['ell']*0
										dCldp['b_fgs_res'][key_loc] = Cls_fid_loc['ell']*0
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
							print '        V      '

							for p in params_dev_loc[m]:
								sigmas[exp][components][label]['marginalized'][p][ind_pfid, ind_pdev] = sigmas_loc[p]
								print '---->>> marginalized $\sigma$(',p,'=',params_fid_loc[p],') = ',sigmas[exp][components][label]['marginalized'][p][ind_pfid, ind_pdev]
								sigmas[exp][components][label]['conditional'][p][ind_pfid, ind_pdev] = sigmas_diag_loc[p]


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

	return sigmas 




