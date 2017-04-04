#!/usr/bin/env python

import residuals_computation_loc_calibration_errors as residuals_comp
import residuals_computation_extended_self_consistent_forecast_calibration_errors as res_ext
import os 
import scipy
from scipy import interpolate
import numpy as np
import CMB4cast_utilities as CMB4U
import forecasting_cosmo_self_consistent_forecast_smf as fc
import healpy as hp
import sympy
from sympy.parsing.sympy_parser import parse_expr
from collections import OrderedDict
import pickle 
import copy

#########################################################################

'''
CMB4cast component separation code
Starting from CMB and foregrounds maps, as well as their associated angular power spectra
The code gives error bars on spectral indices, noise after comp sep and amplitude of statistical fgs residuals.
This is based on Errard et al 2011 and 2012.
'''

############################################################################


class mixing_matrix_builder(list):
    '''	List of SymPy expressions

    example usage:
    >>> from data_modelling import SkyModel
    >>> analytic_expr = ['1', '(nu / nu_ref_sync) ** beta_sync']  # CMB and synchrotron
    >>> A = SkyModel(analytic_expr)
    >>> print A
    [1, (nu/nu_ref_sync)**beta_sync]
    >>> A.subs({'nu_ref_sync': 250})  # Set a variable in the expression
    [1, (nu/250)**beta_sync]
    >>> A.diff('beta_sync')  # Differentiate wrt a variable
    [0, (nu/nu_ref_sync)**beta_sync*log(nu/nu_ref_sync)]
    >>> print A  # Consistently with Sympy expressions A has not been modified
    [1, (nu/nu_ref_sync)**beta_sync]
    >>> A = A.subs({'nu_ref_sync': 250}).diff('beta_sync')  # Now it is modified
    >>> print A
    [0, (nu/250)**beta_sync*log(nu/250)]
    '''

    def __init__(self, expr_or_list_to_be_parsed):
	if not isinstance(expr_or_list_to_be_parsed, list):
	    expr_or_list_to_be_parsed = [expr_or_list_to_be_parsed]
	super(mixing_matrix_builder, self).__init__([parse_expr(e) if isinstance(e, basestring) 
					else e for e in expr_or_list_to_be_parsed])

    def get_mixing_matrix_evaluator(self, parameters, frequencies):
	'''	
	Return a mixing matrix evaluator(`mme`): a function that evaluats (efficiently?)
	the mixing matrix at specific values of the parameters

	parameters: str
	    arguments of the `mme` (order matters!)
	    parameters='alpha beta' implies that the `mme` takes as arguments the values
	    of `alpha` and `beta` (in this order) and return a numpy.matrix equal to the
	    mixing matrix evaluated at these values of the parameters
	    Notice: (for time being) `parameters` has to contain all the parameters of the mixing matrix
	'''
	all_symbols = sympy.symbols('nu '+ parameters)
	force_nu_dependence = lambda f: f if f.diff('nu') != 0 else f + parse_expr('1e-15*nu')
	funcs = [sympy.lambdify(all_symbols, force_nu_dependence(f), modules='numpy') for f in self]
	freqs_array = np.array(frequencies)
	return lambda *args: np.matrix([f(freqs_array, *args) for f in funcs]).T

    def __getattr__(self, *args):
	'''	Call the desired method on each element (i.e. each Sympy expression)
	'''
	def func(*fargs, **fkwargs):
	    a = mixing_matrix_builder([])
	    super(mixing_matrix_builder, a).__init__([getattr(l, args[0])(*fargs, **fkwargs) for l in self])
	    return a

	return func

############################################################################

def tag_maker( stokes='Q', frequency=150.0):
	tag_f = stokes+str(frequency)+'GHz'	
	return tag_f

############################################################################

def A_initiation( analytic_expr_per_template={'Qcmb':'', 'Ucmb':'',\
			 'Qdust':'factor_dust * ( nu / nu_ref ) ** ( 1 + Bd + drun * log( nu / nu_ref ) )',\
			 'Udust':'factor_dust * ( nu / nu_ref ) ** ( 1 + Bd + drun * log( nu / nu_ref ) )'}, \
			 stokes_temp_loc=[], squeeze=False):
	"""
	@brief: this function outputs an initialization for the mixing matrix
	A in order to speed up further operations which require to quickly 
	estimate A
	"""
	if squeeze:
		# setting the output dimensions of the squeezed version
		output_squeezed, vec2sq = from_QU_tags_to_squeeze(full_keys=analytic_expr_per_template.keys())	

	# filling in the mixing matrix
	vec_temp = []
	for stokes_temp in stokes_temp_loc:
		# find a scaling laws when stokes_temp is in [cmb, dust, sync], and not [Qcmb, Udust, etc.]
		if stokes_temp not in analytic_expr_per_template.keys():
			stokes_temp_analytic_expr = vec2sq[stokes_temp]
			# print '		NB: you had not specified scaling laws for ', stokes_temp
			# print '    		but we assumed you would not mind considering'
			# print ' 		the same scaling law used for ', stokes_temp_analytic_expr
		else:
			stokes_temp_analytic_expr = stokes_temp
		
		vec_temp.append(analytic_expr_per_template[stokes_temp_analytic_expr])

	# initiating mixing matrix A
	A = mixing_matrix_builder( vec_temp )

	return A

############################################################################

def from_QU_tags_to_squeeze( full_keys=[] ):
	"""
	@brief: this function produces the set of 
	'squeezed' tags from the full set of tags
	e.g. from 'Qfreq', 'Ufreq' to 'freq'
	"""
	output_squeezed_loc = []

	for key in full_keys:
		key_update = key.replace("Q", "")
		key_update = key_update.replace("U", "")
		output_squeezed_loc.append(key_update)

	# keep only one name per dimensions if squeezed, 
	# while keeping the order of keys (hence no set() function here)
	output_squeezed = []
	vec2sq = OrderedDict() # relation between squeezed scaling laws and the input
	ind = 0
	for i in output_squeezed_loc:
		if i not in output_squeezed:
			output_squeezed.append( i )
			vec2sq[i] = full_keys[ind]
		ind+=1

	return output_squeezed, vec2sq

#########################################################################

def A_matrix_builder( analytic_expr_per_template={'Qcmb':'', 'Ucmb':'',\
			 'Qdust':'factor_dust * ( nu / nu_ref ) ** ( 1 + Bd + drun * log( nu / nu_ref ) )',\
			 'Udust':'factor_dust * ( nu / nu_ref ) ** ( 1 + Bd + drun * log( nu / nu_ref ) )'}, \
			 frequency_stokes=['Q', 'U'], frequencies=[150.0], drv='', drv2='',\
			 spectral_parameters={}, bandpass_channels={}, squeeze=False, \
			 A_init=None ):
	"""
	@brief: return a dictionnary describing the entire mixing matrix
	"""

	if squeeze:
		# setting the output dimensions of the squeezed version
		output_squeezed, vec2sq = from_QU_tags_to_squeeze( full_keys=analytic_expr_per_template.keys() )
	
	# set the keys of the output mixing matrix
	A_output = {}#OrderedDict()
	A_output['out'] = []
	A_output['in'] = []

	# squeezed or not squeezed, that's the question
	if not squeeze:
		frequency_stokes_loc = frequency_stokes
		stokes_temp_loc = analytic_expr_per_template.keys()
	else:
		frequency_stokes_loc = ['']
		stokes_temp_loc = output_squeezed # only one tag over two, remembering only the component name, w/o Q or U

	# this sets the size of the mixing matrix 
	for f in frequencies:
		for f_stokes in frequency_stokes_loc:
			tag_f = tag_maker(stokes=f_stokes, frequency=f)			
			A_output['out'].append(tag_f)
	for stokes_temp in stokes_temp_loc:
		A_output['in'].append( stokes_temp )

	# setting the dimensions of the mixing matrix
	A_output['matrix'] = np.zeros((len(A_output['out']), len(A_output['in'])))

	# initiate A
	A = A_initiation(analytic_expr_per_template=analytic_expr_per_template, \
								stokes_temp_loc=stokes_temp_loc, squeeze=squeeze )

	# first derivative
	if drv!='':
		A = A.diff(drv)
	# second derivative
	if drv2!='':
		A = A.diff(drv2)

	# fill in the mixing matrix
	for freq_loc in frequencies:
		
		for f_stokes in frequency_stokes_loc:

			tag_f = tag_maker(stokes=f_stokes, frequency=freq_loc)			
			indo = A_output['out'].index(tag_f)

			if ((not bandpass_channels.keys()) or (tag_f not in bandpass_channels.keys())):
				# print ' you have not defined bandpasses for ', tag_f, ' -> this is set to zero'
				# bandpass_channels[tag_f] = 0.0
				bandpass = [1.0]
				freq_range = [freq_loc]
			else:
				# building bandpasses 
				bandpass, freq_range = bandpass_computation( nu=freq_loc, \
									bandpass_channels=bandpass_channels[tag_f] )
			# integration over bandpasses 
		 	for i_nu in range(len(freq_range)):
		 		spectral_parameters['nu'] = freq_range[i_nu]*1.0
				# numerical estimation of A
				if i_nu == 0:
					A_bandpass = bandpass[i_nu]*np.array( A.subs( spectral_parameters )[:] )
				else:
					A_bandpass += bandpass[i_nu]*np.array( A.subs( spectral_parameters )[:] )

			if len(bandpass)!=1:
				A_output['matrix'][indo,:] = A_bandpass/np.sum( bandpass )
			else:
				A_output['matrix'][indo,:] = A_bandpass

			del A_bandpass

	# setting to zero off-diagonal (Q,U) elements
	for stokes_temp in stokes_temp_loc:
		for freq_loc in frequencies:
			for f_stokes in frequency_stokes_loc:
					if (f_stokes not in stokes_temp):
						tag_f = tag_maker(stokes=f_stokes, frequency=freq_loc)			
						indo = A_output['out'].index(tag_f)
						indi = A_output['in'].index(stokes_temp)
						A_output['matrix'][indo,indi] = 0.0
	
	return A_output

############################################################################

def bandpass_computation( nu=150.0, bandpass_channels=0.3 ):
	"""
	@brief: this function builds the bandpasses, i.e. it returns 
	the normalized (to 1) value of the bandpass and the corresponding
	frequency value
	"""
	"""
	Delta_f = bandpass_channels/2.0
	# freq_range = np.arange( nu*(1-Delta_f/2), nu*(1+Delta_f/2) )
	# freq_range = np.linspace( nu*(1-Delta_f/2), nu*(1+Delta_f/2), num=100 )
	freq_range = np.arange( nu*(1-Delta_f), nu*(1+Delta_f)+1 )
	# freq_range = np.arange( nu*(1-Delta_f), nu )
	# freq_range = np.arange( nu*(1-Delta_f), nu*(1+Delta_f)+1 )
	# freq_range = [ nu*(1-Delta_f/2), nu, nu*(1+Delta_f/2) ] 
	# freq_range = [ nu*(1-Delta_f), nu, nu*(1+Delta_f) ] 
	# freq_range = [ nu*(1-Delta_f/2.0), nu*(1+Delta_f/2.0) ] 
	bandpass = np.ones(len(freq_range))
	"""
	######## UPDATE ON FEB 22, 2017
	width = nu * bandpass_channels
	nu1 = nu - width / 2.0
	nu2 = nu + width / 2.0
	# print nu1, nu2
	Delta_f = bandpass_channels
	# gridding frequencies between -bandpass -> +bandpass
	if width!=0.0:
		freq_range = np.arange( nu*(1-Delta_f), nu*(1+Delta_f)+1, 0.1 )
		bandpass = np.zeros(len(freq_range))
		indices = np.where((freq_range >= nu1)&(freq_range < nu2))
		bandpass[indices] = 1.0
	else:
		freq_range = [nu*1.0]
		bandpass = [1.0]
	return bandpass, freq_range

############################################################################

def BB_factor_computation(nu):
	"""
	@brief: from CMB to RJ units, computed for a given frequency
	@return: CMB->RJ conversion factor
	"""
	BB_factor = (nu/cst)**2*np.exp(nu/cst)/(np.exp(nu/cst)-1)**2
	return BB_factor

############################################################################

def Ninv_builder( sensitivities={}, squeeze=False ):
	"""
	@brief: build a diagonal noise covariance matrix
	for each frequency channel
	"""
	Ninv_loc = {}#OrderedDict()
	N_loc = {}#OrderedDict()
	Ninv_loc['out'] = []
	Ninv_loc['in'] = []
	N_loc['out'] = []
	N_loc['in'] = []
	if squeeze:
		# we only keep half of the keys ... 
		sensitivities_loc, vec2sq = from_QU_tags_to_squeeze( full_keys=sensitivities.keys() )
	else:
		sensitivities_loc = sensitivities.keys()

	Ninv_loc['matrix'] = np.zeros(( len(sensitivities_loc), len(sensitivities_loc) ))
	N_loc['matrix'] = np.zeros(( len(sensitivities_loc), len(sensitivities_loc) ))
	
	ind_key = 0 

	for key1 in sensitivities_loc:
		Ninv_loc['out'].append(key1)
		Ninv_loc['in'].append(key1)
		N_loc['out'].append(key1)
		N_loc['in'].append(key1)
		
		if squeeze:
			key1_loc = vec2sq[key1]
		else:
			key1_loc = key1

		Ninv_loc['matrix'][ind_key, ind_key] = 1.0/sensitivities[key1_loc]**2
		N_loc['matrix'][ind_key, ind_key] = sensitivities[key1_loc]**2

		ind_key += 1

	return Ninv_loc, N_loc

############################################################################

def invert( Mloc_mat ):
	"""
	@brief: estimate the inverse of any array
	"""
	if isinstance(Mloc_mat, np.ndarray):
		try:
			Mloc_mat_inv = np.linalg.inv( Mloc_mat )
		except np.linalg.linalg.LinAlgError:
			print '!!!PROBLEM DURING INVERSION!!!'
			Mloc_mat_inv = 1e6*np.sign(Mloc_mat)*np.ones( Mloc_mat.shape )
	else:
		Mloc_mat_inv = 1.0/( Mloc_mat )
	
	return Mloc_mat_inv

############################################################################

def AtNAinv_builder( A, Ninv ):
	"""
	@brief: produces (A^T N^-1 A) ^ -1
	"""
	# check if in and out dimensions match
	assert A['out'] == Ninv['in']
	# A^T N^-1 A
	AtNAinv = {}#OrderedDict()
	AtNAinv['in'] = A['in']
	AtNAinv['out'] = A['in']
	AtNinvA_matrix = A['matrix'].T.dot( Ninv['matrix'] ).dot( A['matrix'] )
	# ( A^T N^-1 A) ^ -1
	AtNAinv['matrix'] = invert( AtNinvA_matrix )	
	return AtNAinv

############################################################################

def Sigma_computation( sensitivity_uK_per_chan={}, freqs_loc=[150.0],\
			spectral_parameters={}, analytic_expr_per_template={}, \
			frequency_stokes=[], bandpass_channels=None, Ninv={}, 
			not_analytic=False, CMB_units_only=False, drv=[], \
			prior_spectral_parameters={}, npatch=0, spsp={}):
	"""
	@brief: estimates the error matrix $\Sigma^-1$, as defined in Eq. (??) of Errard, Stivoli and Stompor (2011)
	@return: second derivative of the likelihood at the peak
	"""

	A = A_matrix_builder( analytic_expr_per_template=analytic_expr_per_template, \
			 frequency_stokes=frequency_stokes, frequencies=freqs_loc, drv='',\
			 spectral_parameters=spectral_parameters, bandpass_channels=bandpass_channels )

	dAdB = OrderedDict()
	for drv_loc in drv:
		dAdB[drv_loc] = A_matrix_builder( analytic_expr_per_template=analytic_expr_per_template, \
			 frequency_stokes=frequency_stokes, frequencies=freqs_loc, drv=drv_loc,\
			 spectral_parameters=spectral_parameters, bandpass_channels=bandpass_channels )

	if not Ninv.keys():
		Ninv, N = Ninv_builder( sensitivities=sensitivity_uK_per_chan )

	AtNAinv = AtNAinv_builder( A, Ninv )

	if AtNAinv['matrix'][0,0] <=0 or AtNAinv['matrix'][0,0]>=1e4:
		print 'wrong noise value with AtNAinv = ',  AtNAinv['matrix']
		return 1e6*np.ones((len(drv),len(drv)))

	d2LdBdB = OrderedDict()
	d2LdBdB['in'] = drv
	d2LdBdB['out'] = drv
	d2LdBdB['matrix'] =  np.zeros((len(drv),len(drv)))
	for i in range(len(drv)) :
		drv_i = drv[i]
		for j in range(len(drv)) :
			drv_j = drv[j]

			d2LdBdB['matrix'][i,j] =  np.trace( - dAdB[drv_i]['matrix'].T.dot( Ninv['matrix'] ).dot( A['matrix'] ).dot( AtNAinv['matrix'] ).dot( A['matrix'].T ).dot( Ninv['matrix'] ).dot( dAdB[drv_j]['matrix'] ).dot( spsp )\
							 + dAdB[drv_i]['matrix'].T.dot( Ninv['matrix'] ).dot( dAdB[drv_j]['matrix'] ).dot( spsp )  )
			if ((i == j) and (prior_spectral_parameters[drv[i]] != 0.0)):
				print 'you set a prior on ', drv[i], ' with sigma = ', prior_spectral_parameters[drv[i]] 
				d2LdBdB['matrix'][i,j] += 1.0/prior_spectral_parameters[drv[i]]**2

	### npatch rescaling 
	if npatch < 1: npatch=1.0
	d2LdBdB['matrix'] /= np.sqrt(npatch)

	## computation of (d2LdBdB)^-1
	d2LdBdBinv = copy.deepcopy( d2LdBdB )
	d2LdBdBinv['matrix'] = invert( d2LdBdB['matrix'] )

	return d2LdBdB, d2LdBdBinv, AtNAinv

############################################################################

def alpha_computation( sensitivity_uK_per_chan={}, freqs_loc=[150.0],\
			spectral_parameters={}, analytic_expr_per_template={}, \
			frequency_stokes=[], bandpass_channels=None,  drv=[], prior_spectral_parameters={}):
	"""
	@brief: computes the three-dimensional 'r$\alpha$' object, as defined in Eq. (??) of Errard, Stivoli and Stompor (2011)
	@return: 3-d alpha object
	"""

	A_sq = A_matrix_builder( analytic_expr_per_template=analytic_expr_per_template, \
			 frequency_stokes=frequency_stokes, frequencies=freqs_loc, drv='',\
			 spectral_parameters=spectral_parameters, bandpass_channels=bandpass_channels, squeeze=True )

	dAdB_sq = OrderedDict()
	for drv_loc in drv:
		dAdB_sq[drv_loc] = A_matrix_builder( analytic_expr_per_template=analytic_expr_per_template, \
			 frequency_stokes=frequency_stokes, frequencies=freqs_loc, drv=drv_loc,\
			 spectral_parameters=spectral_parameters, bandpass_channels=bandpass_channels, squeeze=True )

	Ninv_sq, N_sq = Ninv_builder( sensitivities=sensitivity_uK_per_chan, squeeze=True )

	AtNAinv_sq = AtNAinv_builder( A_sq, Ninv_sq )

	alpha = OrderedDict()
	alpha['in'] = A_sq['in']
	alpha['matrix'] = np.zeros(( A_sq['matrix'].shape[1], A_sq['matrix'].shape[1], len(drv) ))
	# alpha['out'] = 
	for i in range( len(drv) ):

		alpha['matrix'][:,:,i] = - AtNAinv_sq['matrix'].dot( A_sq['matrix'].T ).dot( Ninv_sq['matrix'] ).dot( dAdB_sq[drv[i]]['matrix'] )

	return alpha

############################################################################

def Cls_residuals( d2LdBdBinv={}, Cls={}, sensitivity_uK_per_chan={}, freqs_loc=[150.0],\
						spectral_parameters={}, analytic_expr_per_template={}, \
						frequency_stokes=[], bandpass_channels=None,  drv=[], prior_spectral_parameters={} ):
	"""
	@brief: computes $C_\ell^{res}$ as defined in Eq. (??) of Errard, Stivoli and Stompor (2011)
	@return: Cls with the key 'res'
	"""
	

	## computation of alpha
	alpha = alpha_computation( sensitivity_uK_per_chan=sensitivity_uK_per_chan, freqs_loc=freqs_loc,\
			spectral_parameters=spectral_parameters, analytic_expr_per_template=analytic_expr_per_template, \
			frequency_stokes=frequency_stokes, bandpass_channels=bandpass_channels, drv=drv, \
			prior_spectral_parameters=prior_spectral_parameters )

	nparams = len(d2LdBdBinv['matrix'])
	components = alpha['in']

	## power estimation from analytical formula
	Cl_res = Cls['ell']*0.0

	for ell_ind in range( len(Cls['ell']) ):
		
		clkk = np.zeros((nparams,nparams))

		for k1 in range(nparams):
			for k2 in range(nparams):
				
				clkk_loc = 0.0

				for j1 in range(len(components)):
					for j2 in range(len(components)):
						if j1 == j2:
							if components[j1] == 'cmb':
								comp_loc = 'BB'
							elif components[j1] == 'dBdust':
								comp_loc = 'dBdxdBd'
							elif components[j1] == 'dBsync':
								comp_loc = 'dBsxdBs'
							else:
								comp_loc = components[j1]

							clkk_loc += d2LdBdBinv['matrix'][ k1,k2 ] * alpha['matrix'][ 0, j1, k1 ] * alpha['matrix'][ 0, j2, k2] * Cls[comp_loc][ ell_ind ]
						else:
							if not (( (components[j1] == 'cmb') and (components[j2]!='cmb') ) or ((components[j2] == 'cmb') and (components[j1]!='cmb') )):
								comp_loc = components[j1]+'x'+components[j2]
								if (( (components[j1] == 'dust') and (components[j2]=='sync') ) or ((components[j1] == 'sync') and (components[j2]=='dust') )):
									comp_loc = 'dxs'
								if (( (components[j1] == 'dust') and (components[j2]=='dBdust') ) or ((components[j1] == 'dBdust') and (components[j2]=='dust') )):
									comp_loc = 'dBdxdust'
								if (( (components[j1] == 'sync') and (components[j2]=='dBsync') ) or ((components[j1] == 'dBsync') and (components[j2]=='sync') )):
									comp_loc = 'dBsxsync'
								if (( ('dBd' in components[j1]) and ('sync' in components[j2]) ) or \
												( ('dBs' in components[j1]) and ('dust' in components[j2]) ) or \
													( ('sync' in components[j1]) and ('dBd' in components[j2]) ) or \
														( ('dust' in components[j1]) and ('dBs' in components[j2]) ) ):
									continue
								
								clkk_loc += d2LdBdBinv['matrix'][ k1,k2 ] * alpha['matrix'][ 0, j1, k1 ] * alpha['matrix'][ 0, j2, k2] * Cls[comp_loc][ ell_ind ]

				clkk[ k1,k2 ] = clkk_loc

		Cl_res[ell_ind] = np.sum(np.sum( clkk ))

	Cls['res'] = Cl_res*1.0

	return Cls

############################################################################

def get_maps_and_spectra(path2files='./', fsky=0.5, analytic_expr_per_template={}):
	"""
	@brief: this function loads spsp and Cls objects
	@return: gives the spsp and Cls corresponding to the required fsky, in uK_RJ, at 150GHz
	"""
	
	with open(os.path.join( path2files, 'spsp.pkl' ), 'r') as f:
		try:
			spsp = pickle.load(f)
		except IOError:
			print 'cannot open ', f

	with open(os.path.join( path2files, 'Cls.pkl' ), 'r') as f:
		try:
			Cls = pickle.load(f)
		except IOError:
			print 'cannot open ', f		

	good_ind = np.argmin(np.abs(fsky - spsp['fsky']))
	good_ind2 = np.argmin(np.abs(fsky - Cls['fsky']))
	assert good_ind == good_ind2

	spsp_out =  np.squeeze(spsp['matrix'][good_ind,:,:])
	## we should keepd  the relevant dimensions of spsp ! 
	print spsp['in']
	print spsp['out']

	filtered_ind = []
	for temp in analytic_expr_per_template.keys():
		if temp in spsp['in']:
			filtered_ind.append( spsp['in'].index(temp) )
		else:
			print '========================================'
			print 'error, spsp does not contain the template ', temp
			print 'you should consider computing spsp yourself with this new sky map'
			print 'meanwhile, we stop here'
			exit()

	spsp_out_filtered = np.zeros((len(filtered_ind), len(filtered_ind)))
	for f1 in range(len(filtered_ind)):
		for f2 in range(len(filtered_ind)):
			spsp_out_filtered[f1,f2] = spsp_out[filtered_ind[f1], filtered_ind[f2]]
	# filtering out the correct Cls for the fsky of interest
	Cls_out = copy.deepcopy(Cls)
	for key in Cls_out['foregrounds_keys'] :
		if key not in Cls.keys():
			print '========================================'
			print 'error, precomputed Cls does not contain the template ', key
			print 'you should consider computing Cls yourself with this new sky map'
			print 'meanwhile, we stop here'
			exit()
		else:
			Cls_out[key] = Cls[key][good_ind,:]

	return spsp_out_filtered, Cls_out

############################################################################

def CMB4cast_compsep(configurations={}, components_v=[''], exp='', np_nside=4, ell_min_camb=2, Cls_fid={}, \
					spectral_parameters={}, analytic_expr_per_template={}, bandpass_channels={},\
					camb='', drv=[], prior_spectral_parameters={}, ells_exp=0.0, path2files='' ):
	"""
	@brief: main function for the comp sep part of CMB4cast
	This is based on Errard et al (2011) formalism i.e. it gives statistical level of 
	foregrounds residuals as well as the noise after comp sep
	"""

	foregrounds_exp = {}
	npatch = int( configurations[exp]['fsky']*12*np_nside**2 )
	if npatch == 0 : npatch=1


	## creating bandpass
	configurations[exp]['sensitivity_uK_per_chan'] = OrderedDict()
	configurations[exp]['bandpass_dict'] = OrderedDict()
	configurations[exp]['FWHM_dict'] = OrderedDict()
	ind = 0
	for f in configurations[exp]['freqs']:
		for f_stokes in CMB4U.frequency_stokes_default:
			tag = tag_maker(stokes=f_stokes, frequency=f)
			configurations[exp]['sensitivity_uK_per_chan'][tag] = configurations[exp]['uKRJ/pix'][ind]
			configurations[exp]['bandpass_dict'][tag] = configurations[exp]['bandpass'][ind]
			configurations[exp]['FWHM_dict'][tag] = configurations[exp]['FWHM'][ind]
		ind += 1

	###############################################################################
	## FOREGROUNDS MAPS INFOS 
	spsp, Cls = get_maps_and_spectra( path2files=path2files, fsky=configurations[exp]['fsky'],\
										analytic_expr_per_template=analytic_expr_per_template )

	######################################################################################
	## ACTUAL COMPONENT SEPARATION, FOR EACH SKY COMPONENT
	for components in components_v:
		
		foregrounds_exp[components] = {}
		
		if components == 'cmb-only':
			foregrounds_exp[components]['Cl_res'] = Cls['BB']*0.0
			d2LdBdBinv = {}
			d2LdBdBinv['in'] = drv
			d2LdBdBinv['out'] = drv
			d2LdBdBinv['matrix'] = np.zeros((len(drv), len(drv)))
			foregrounds_exp[components]['uKCMB/pix_postcompsep'] = \
							1.0/np.sqrt( np.sum( 1.0 / np.array(configurations[exp]['uKCMBarcmin'])**2) )/CMB4U.pix_size_map_arcmin
		else:
			##################################################################
			## COMPUTATION OF THE SECOND DERIVATIVES OF THE SPECTRAL LIKELIHOOD
			d2LdBdB, d2LdBdBinv, AtNAinv = Sigma_computation( sensitivity_uK_per_chan=configurations[exp]['sensitivity_uK_per_chan'], freqs_loc=configurations[exp]['freqs'],\
										spectral_parameters=spectral_parameters, analytic_expr_per_template=analytic_expr_per_template, \
										frequency_stokes=CMB4U.frequency_stokes_default, bandpass_channels=configurations[exp]['bandpass_dict'], drv=drv, \
										prior_spectral_parameters=prior_spectral_parameters, npatch=npatch, spsp=spsp)

			sqrtAtNAinv00 = np.sqrt( AtNAinv['matrix'][0,0] )
			if sqrtAtNAinv00 != sqrtAtNAinv00:
				sqrtAtNAinv00 = 1e3
			foregrounds_exp[components]['uKCMB/pix_postcompsep'] = sqrtAtNAinv00*1.0

			##################################################################
			## ESTIMATION OF THE STATISTICAL RESIDUALS
			Cls = Cls_residuals( d2LdBdBinv=d2LdBdBinv, Cls=Cls, sensitivity_uK_per_chan=configurations[exp]['sensitivity_uK_per_chan'], freqs_loc=configurations[exp]['freqs'],\
										spectral_parameters=spectral_parameters, analytic_expr_per_template=analytic_expr_per_template, \
										frequency_stokes=CMB4U.frequency_stokes_default, bandpass_channels=configurations[exp]['bandpass_dict'], drv=drv, \
										prior_spectral_parameters=prior_spectral_parameters )

			foregrounds_exp[components]['Cl_res'] = Cls['res']*1.0
			
		#################################
		## NOISE POST COMP SEP 
		print '------------'
		print ' for component = ', components
		print ' for exp=',exp,' >>> uK_CMB arcmin after comp. sep.  = ', foregrounds_exp[components]['uKCMB/pix_postcompsep']*CMB4U.pix_size_map_arcmin
		print '  				 while uK_CMB arcmin before comp. sep.  = ',  1.0/np.sqrt( np.sum( 1.0 / np.array(configurations[exp]['uKCMBarcmin'])**2) ) 

		#################################
		# COMPUTATION OF R_EFF
		ind0 = np.argmin(np.abs( Cls_fid['ell'] - 20 ))
		ind1 = np.argmin(np.abs( Cls_fid['ell'] - 200 ))
		ind0_ = np.argmin(np.abs( Cls_fid['ell'] - 20 ))
		ind1_ = np.argmin(np.abs( Cls_fid['ell'] - 200 ))
		if components != 'cmb-only':
			foregrounds_exp[components]['r_eff'] = np.sum(foregrounds_exp[components]['Cl_res'][ind0:ind1] / Cls_fid['ell'][ind0:ind1] / (Cls_fid['ell'][ind0:ind1]+1.0) * 2.0 * np.pi) / np.sum(Cls_fid['BuBu_r1'][ind0_:ind1_] / Cls_fid['ell'][ind0_:ind1_] / (Cls_fid['ell'][ind0_:ind1_]+1.0) * 2.0 * np.pi) 
		else:
			foregrounds_exp[components]['r_eff'] = 0.0

		print '  				and the effective level of residuals is reff = ', foregrounds_exp[components]['r_eff']

		#################################
		## OUTPUT CONSTRAINTS ON SPECTRAL INDICES
		for param in d2LdBdBinv['in']:
			ind = d2LdBdBinv['in'].index(param)
			foregrounds_exp[components][param] = np.sqrt(d2LdBdBinv['matrix'][ ind,ind ])

	return foregrounds_exp
