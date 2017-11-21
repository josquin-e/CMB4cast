'''
Residuals computation for a given experimental design, i.e. detector distribution and beams/NETs.

Author: Josquin
Contact: josquin.errard@gmail.com
'''

import sys
import os
import argparse
import numpy as np
import healpy as hp
import noise_computation
import pylab as pl
import residuals_computation_extended_self_consistent_forecast_calibration_errors as res_ext

#Bd = 1.65
#Bs = - 3.0

#########################################################################

h_over_k =  0.0479924
nu_ref= 150.0
#Temp=18.0
cst = 56.8

#Temp2=20.0
#nu_dust2= Temp2/h_over_k
#Bd2=1.65

def factor_dust_computation( nu, Temp=18.0 ):
	nu_dust = Temp/h_over_k
	return (np.exp(nu_ref/nu_dust) - 1)/(np.exp(nu/nu_dust)-1)

def factor_dust_computation2( nu, nu_dust2, Temp=18.0  ):
	nu_dust = Temp/h_over_k
	return (np.exp(nu_ref/nu_dust2) - 1)/(np.exp(nu/nu_dust2)-1)

#########################################################################

def A_element_computation(nu, Bd, Bs, adust=0.0, async=0.0, squeeze=False, stolyarov=False,\
							 T_second_grey_body=0.0, Bd2=0.0, bandpass_channels=0.0, Temp=18.0):

	Delta_f = bandpass_channels
	freq_range = np.arange( nu*(1-Delta_f/2), nu*(1+Delta_f/2)+1 )

	nu_dust = Temp/h_over_k

	ind = 0
	for nu_i in freq_range:

		factor_dust = factor_dust_computation( nu_i, Temp )
		a_cmb_i = (nu_i/cst)**(2)*(np.exp(nu_i/cst))/( (np.exp(nu_i/cst)-1)**(2) )
		a_dust_i = factor_dust*(nu_i/nu_ref)**(1 + Bd + adust*np.log(nu_i/nu_ref))
		a_sync_i = (nu_i/nu_ref)**(Bs + async*np.log(nu_i/nu_ref))

		if stolyarov:
			#a_d_dust_i = np.abs(factor_dust*(np.log(nu_i/nu_ref))*(nu_i/nu_ref)**(1+Bd))
			a_d_dust_i = factor_dust*(np.log(nu_i/nu_ref)*(nu_i/nu_ref)**(1 + Bd + adust*np.log(nu_i/nu_ref)))
		elif T_second_grey_body!=0.0:
			nu_dust2= T_second_grey_body/h_over_k
			factor_dust2 = factor_dust_computation2( nu_i, nu_dust2, Temp  )
			a_d_dust_i = factor_dust2*(nu_i/nu_ref)**(1+Bd2)

		if ind == 0:
			a_cmb = a_cmb_i
			a_dust = a_dust_i
			a_sync = a_sync_i
			if stolyarov or T_second_grey_body!=0.0:
				a_d_dust = a_d_dust_i
		else:
			a_cmb += a_cmb_i
			a_dust += a_dust_i
			a_sync += a_sync_i
			if stolyarov or T_second_grey_body!=0.0:
				a_d_dust += a_d_dust_i
		ind+=1
	a_cmb /= ind
	a_dust /= ind
	a_sync /= ind
	if stolyarov or T_second_grey_body!=0.0:
		a_d_dust /= ind

	if squeeze == True:
		if stolyarov or T_second_grey_body!=0.0:
			A = [a_cmb, a_dust, a_d_dust, a_sync]
		else:
			A = [a_cmb, a_dust, a_sync]
	else:
		A_cmb = np.array([[a_cmb, 0],[0, a_cmb]])
		A_dust = np.array([[a_dust, 0],[0, a_dust]])
		A_sync = np.array([[a_sync, 0],[0, a_sync]])
		if stolyarov or T_second_grey_body!=0.0:
			A_d_dust = [[a_d_dust, 0],[0, a_d_dust]]
			A =  np.hstack((A_cmb, A_dust, A_d_dust, A_sync ))
		else:
			A = np.hstack(( A_cmb, A_dust, A_sync ))

	return A 



def dAdB_element_computation(nu, Bd, Bs, adust=0.0, async=0.0,  squeeze=False, stolyarov=False, \
								T_second_grey_body=0.0, Bd2=0.0, bandpass_channels=0.0, Temp=18.0):

	Delta_f = bandpass_channels
	freq_range = np.arange( nu*(1-Delta_f/2), nu*(1+Delta_f/2)+1 )

	nu_dust = Temp/h_over_k

	ind=0
	for nu_i in freq_range:
		factor_dust = factor_dust_computation( nu_i, Temp )
		dadBdi = factor_dust*np.log(nu_i/nu_ref)*(nu_i/nu_ref)**(1 + Bd + adust*np.log(nu_i/nu_ref))

		if T_second_grey_body!=0.0:
			nu_dust2= T_second_grey_body/h_over_k
			factor_dust2 = factor_dust_computation2( nu_i, nu_dust2, Temp )
			dadBd_di = factor_dust2*np.log(nu/nu_ref)*(nu_i/nu_ref)**(1+Bd2)
		elif stolyarov:
			#dadBd_di = np.abs(factor_dust*(np.log(nu_i/nu_ref))**2*(nu_i/nu_ref)**(1+Bd))
			dadBd_di = factor_dust*(np.log(nu_i/nu_ref)**2*(nu_i/nu_ref)**(1 + Bd + adust*np.log(nu_i/nu_ref)))

		dadBsi = np.log(nu_i/nu_ref)*(nu_i/nu_ref)**(Bs + async*np.log(nu_i/nu_ref))

		if squeeze == True:
			if stolyarov:
				dAdBd = np.array( [0, dadBdi, dadBd_di, 0] )
				dAdBs = np.array( [0, 0, 0, dadBsi] )
			elif T_second_grey_body!=0.0:
				dAdBd = np.array( [0, dadBdi, 0, 0] )
				dAdBs = np.array( [0, 0, 0, dadBsi] )
				dAdBd_d = np.array( [0, 0, dadBd_di, 0] )
			else:
				dAdBd = np.array( [0, dadBdi, 0] )
				dAdBs = np.array( [0, 0, dadBsi] )
		else:
			dA_dustdBd = np.array( [[dadBdi, 0],[0, dadBdi]] )
			dA_syncdBs = np.array( [[dadBsi, 0],[0, dadBsi]] )
			if stolyarov:
				dA_dust_d_dBd = np.array( [[dadBd_di, 0],[0, dadBd_di]] )
				dAdBd = np.hstack(( np.zeros((2,2)), dA_dustdBd, dA_dust_d_dBd, np.zeros((2,2)) ))
				dAdBs = np.hstack(( np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), dA_syncdBs ))
			elif T_second_grey_body!=0.0:
				dA_dust_d_dBd = np.array( [[dadBd_di, 0],[0, dadBd_di]] )
				dAdBd = np.hstack(( np.zeros((2,2)), dA_dustdBd, np.zeros((2,2)), np.zeros((2,2)) ))
				dAdBd_d = np.hstack(( np.zeros((2,2)), np.zeros((2,2)), dA_dust_d_dBd, np.zeros((2,2)) ))
				dAdBs = np.hstack(( np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), dA_syncdBs ))
			else:
				dAdBd = np.hstack(( np.zeros((2,2)), dA_dustdBd, np.zeros((2,2)) ))
				dAdBs = np.hstack(( np.zeros((2,2)), np.zeros((2,2)), dA_syncdBs ))

		if ind ==0:
			dAdBd_tot = dAdBd
			dAdBs_tot = dAdBs
			if T_second_grey_body!=0.0:
				dAdBd_d_tot = dAdBd_d
		else:
			dAdBd_tot += dAdBd
			dAdBs_tot += dAdBs
			if T_second_grey_body!=0.0:
				dAdBd_d_tot += dAdBd_d
		ind+=1

	if T_second_grey_body!=0.0:
		return dAdBd_tot/ind, dAdBd_d_tot/ind, dAdBs_tot/ind
	else:
		return np.array(dAdBd_tot)/ind, np.array(dAdBs_tot)/ind


def A_and_dAdB_matrix_computation(freqs, Bd, Bs, async=0.0, adust=0.0, squeeze=False, stolyarov=False, components=['dust', 'sync'], T_second_grey_body=0.0, Bd2=0.0, bandpass_channels=None, Temp=19.6 ):

	if bandpass_channels is None:
		bandpass_channels=np.zeros( len(freqs) )

	
	for f in range( len(freqs) ):

		if T_second_grey_body!=0.0:
			print 'SECOND DUST ! '
			Ai = A_element_computation(freqs[ f ], Bd, Bs, adust=adust, async=async, squeeze=squeeze, T_second_grey_body=T_second_grey_body, Bd2=Bd2, bandpass_channels=bandpass_channels[f], Temp=Temp)
			dAdBdi, dAdBd_di, dAdBsi = dAdB_element_computation(freqs[ f ], Bd, Bs,  adust=adust, async=async, squeeze=squeeze, T_second_grey_body=T_second_grey_body, Bd2=Bd2, bandpass_channels=bandpass_channels[f], Temp=Temp)
			if f == 0:
				A = Ai
				dAdBd = dAdBdi
				dAdBd_d = dAdBd_di
				dAdBs = dAdBsi
			else:
				A = np.vstack(( A, Ai ))
				dAdBd = np.vstack(( dAdBd , dAdBdi ))
				dAdBd_d = np.vstack(( dAdBd_d , dAdBd_di ))
				dAdBs = np.vstack(( dAdBs , dAdBsi ))
		else:
			Ai = A_element_computation(freqs[ f ], Bd, Bs,  adust=adust, async=async, squeeze=squeeze, bandpass_channels=bandpass_channels[f], stolyarov=stolyarov, Temp=Temp )
			dAdBdi, dAdBsi = dAdB_element_computation(freqs[ f ], Bd, Bs,  adust=adust, async=async, squeeze=squeeze, bandpass_channels=bandpass_channels[f], stolyarov=stolyarov, Temp=Temp )
			if f == 0:
				A = Ai
				dAdBd = dAdBdi
				dAdBs = dAdBsi
			else:
				A = np.vstack(( A, Ai ))
				dAdBd = np.vstack(( dAdBd , dAdBdi ))
				dAdBs = np.vstack(( dAdBs , dAdBsi ))
			del Ai, dAdBdi, dAdBsi

	if 'dust' not in components:
		if squeeze==True:
			A = np.delete(A, 1, axis=1 )
			dAdBd = np.delete(dAdBd, 1, axis=1 )
			if T_second_grey_body!=0.0:
				dAdBd_d = np.delete(dAdBd_d, 1, axis=1 )
			dAdBs = np.delete(dAdBs, 1, axis=1 )
		else:
			if T_second_grey_body!=0.0:
				A = np.delete(A, 2, axis=1)
				A = np.delete(A, 2 , axis=1)
				A = np.delete(A, 2, axis=1)
				A = np.delete(A, 2 , axis=1)
				dAdBd = np.delete(dAdBd, 2, axis=1)
				dAdBd = np.delete(dAdBd, 2, axis=1)
				dAdBd = np.delete(dAdBd, 2, axis=1)
				dAdBd = np.delete(dAdBd, 2, axis=1)
				dAdBd_d = np.delete(dAdBd_d, 2, axis=1)
				dAdBd_d = np.delete(dAdBd_d, 2, axis=1)
				dAdBd_d = np.delete(dAdBd_d, 2, axis=1)
				dAdBd_d = np.delete(dAdBd_d, 2, axis=1)
				dAdBs = np.delete(dAdBs, 2, axis=1)
				dAdBs = np.delete(dAdBs, 2, axis=1)
				dAdBs = np.delete(dAdBs, 2, axis=1)
				dAdBs = np.delete(dAdBs, 2, axis=1)
			else:
				A = np.delete(A, 2, axis=1)
				A = np.delete(A, 2 , axis=1)
				dAdBd = np.delete(dAdBd, 2, axis=1)
				dAdBd = np.delete(dAdBd, 2, axis=1)
				if stolyarov or T_second_grey_body!=0.0:
					dAdBd = np.delete(dAdBd, 2, axis=1)
					dAdBd = np.delete(dAdBd, 2, axis=1)
					dAdBd_d = np.delete(dAdBd_d, 2, axis=1)
					dAdBd_d = np.delete(dAdBd_d, 2, axis=1)
				dAdBs = np.delete(dAdBs, 2, axis=1)
				dAdBs = np.delete(dAdBs, 2, axis=1)

	elif 'sync' not in components:
		if squeeze==True:
			if T_second_grey_body!=0.0:	
				A = np.delete(A, 3, axis=1)
				dAdBd = np.delete(dAdBd, 3, axis=1)
				dAdBs = np.delete(dAdBs, 3, axis=1)
				dAdBd_d = np.delete( dAdBd_d, 3, axis=1)
			else:
				A = np.delete(A, 2, axis=1)
				dAdBd = np.delete(dAdBd, 2, axis=1)
				dAdBs = np.delete(dAdBs, 2, axis=1)	
		else:
			if T_second_grey_body!=0.0:
				A = np.delete( A, 6, 1)
				A = np.delete( A, 6, 1)
				dAdBs = np.delete( dAdBs, 6, 1)
				dAdBs = np.delete( dAdBs, 6, 1)
				dAdBd = np.delete( dAdBd, 6, 1)
				dAdBd = np.delete( dAdBd, 6, 1)
				dAdBd_d = np.delete( dAdBd_d, 6, 1)
				dAdBd_d = np.delete( dAdBd_d, 6, 1)
			else:
				A = np.delete( A, 4, 1)
				A = np.delete( A, 4, 1)
				dAdBs = np.delete( dAdBs, 4, 1)
				dAdBs = np.delete( dAdBs, 4, 1)
				dAdBd = np.delete( dAdBd, 4, 1)
				dAdBd = np.delete( dAdBd, 4, 1)

	if T_second_grey_body!=0.0:
		return A, dAdBd, dAdBd_d, dAdBs
	else:
		return A, dAdBd, dAdBs

#########################################################################

def BB_factor_computation(nu):
	BB_factor = (nu/cst)**2*np.exp(nu/cst)/(np.exp(nu/cst)-1)**2
	return BB_factor

#########################################################################

def omega_computation(nch, squeeze=False):
	if squeeze:
		return np.diag(np.ones(nch))
	else:
		return np.diag(np.ones(2*nch))

#########################################################################

def Cl_res_computation(Nbolos, sensitivity_uK_per_chan, freqs, spsp, \
	sigma_omega, Cl_cmb, Cl_dust, Cl_sync, Cl_dxs, ell_min, ell_max, \
	calibration_fixed=False, npatch=1, components=['dust', 'sync'], \
	output='', Bd = 1.59, ells_input=[], stolyarov=False, \
	prior_dust=0.0, prior_sync=0.0, T_second_grey_body=0.0, Bd2=1.65, \
	cmb_noise_output=False, just_cmb_noise_output=False, \
	everything_output=False, bandpass_channels=None, \
	delta_beta_stolyarov=0.1, Q_dust=0.0, U_dust=0.0, Q_sync=0.0, U_sync=0.0, \
	Temp=19.6, Bs=-3.1, async=0.0, adust=0.0, \
	Td_fixed=True, correlation_2_dusts=0.0, return_Cl_res_beta_var=False ):

	nch = len( Nbolos )
	# Computation of the inverse covariance, in 1/uK**2 per frequency channel
	Ninv = np.zeros((2*nch, 2*nch))
	for k in range( 2*nch):
		Ninv[ k, k ]  = ((1.0/sensitivity_uK_per_chan[ int(k/2) ])**2) * Nbolos[ int(k/2) ]

	# A is a 6 x Nch matrix. 6 = 2x(CMB, Dust, Synchrotron)
	if T_second_grey_body!=0.0:
		A, dAdBd, dAdBd_d, dAdBs = A_and_dAdB_matrix_computation( freqs, Bd, Bs, adust=adust, async=async, components=components, \
					T_second_grey_body=T_second_grey_body, Bd2=Bd2, bandpass_channels=bandpass_channels, Temp=Temp  )
	else:
		A, dAdBd, dAdBs = A_and_dAdB_matrix_computation( freqs, Bd, Bs, adust=adust, async=async, components=components, \
					Bd2=Bd2, stolyarov=stolyarov, bandpass_channels=bandpass_channels, Temp=Temp )

	Omega = omega_computation(nch)

	# A**T N**-1 ** A
	OmegaA = Omega.dot(A)
	AtNA =  OmegaA.T.dot(Ninv).dot(OmegaA)

	#print 'AtNA is ', AtNA
	# inversion of (A**T N**-1 ** A)
	try:
		AtNAinv = np.linalg.inv( AtNA )
	
	except np.linalg.linalg.LinAlgError:
		if cmb_noise_output:
			return Cl_dust*1e3, 1e3
		elif everything_output:
			if stolyarov:
				return Cl_dust*1e3, 1e3, np.array([[1e3,1e3,1e3],[1e3,1e3,1e3],[1e3,1e3,1e3]]), Cl_dust*1e3
			else:
				return Cl_dust*1e3, 1e3, np.array([[1e3,1e3,1e3],[1e3,1e3,1e3],[1e3,1e3,1e3]])

	if just_cmb_noise_output:
		return np.sqrt( AtNAinv[0,0] )


	################################################
	## Wiener filtering test
	'''
	npix = 0.7*12*128**2
	Finv = np.linalg.inv( spsp/npix )
	noise_matrix = AtNA + Finv
	noise_matrix_inv = np.linalg.inv( noise_matrix )
	#Finv_noise_matrix_inv = Finv.dot( noise_matrix_inv )
	Finv_noise_matrix_inv = noise_matrix_inv
	wiener_CMB_noise = np.sqrt( Finv_noise_matrix_inv[0,0] )
	print wiener_CMB_noise, np.sqrt( AtNAinv[0,0] )
	exit()
	'''

	################################################
	if AtNAinv[0,0] <=0 or AtNAinv[0,0]>=1e4:
		print 'wrong noise value with AtNAinv = ',  AtNAinv
		print freqs, Bd, Bs, components, Bd2, stolyarov, bandpass_channels, Temp
		if cmb_noise_output:
			return Cl_dust*1e6, 1e6
		elif everything_output:
			if stolyarov:
				return Cl_dust*1e6, 1e6, np.array([[1e6,1e6,1e6],[1e6,1e6,1e6],[1e6,1e6,1e6]]), Cl_dust*1e6
			else:
				return Cl_dust*1e6, 1e6, np.array([[1e6,1e6,1e6],[1e6,1e6,1e6],[1e6,1e6,1e6]])
		else:
			return Cl_dust*1e3
	################################################

	###########
	# second derivative of the likelihood (d2LdBdB) computation
	# DUST x SYNC BLOCK
	OdAdBd = Omega.dot( dAdBd )
	if T_second_grey_body!=0.0:
		OdAdBd_d = Omega.dot( dAdBd_d )
	OdAdBs = Omega.dot( dAdBs )

	#######################
	# filter out components which are not considered
	m00 = np.trace( - OdAdBd.T.dot( Ninv ).dot( OmegaA ).dot( AtNAinv ).dot( OmegaA.T ).dot( Ninv ).dot( OdAdBd ).dot( spsp ) + OdAdBd.T.dot( Ninv ).dot( OdAdBd ).dot( spsp )  )
	m01 = np.trace( - OdAdBd.T.dot( Ninv ).dot( OmegaA ).dot( AtNAinv ).dot( OmegaA.T ).dot( Ninv ).dot( OdAdBs ).dot( spsp ) + OdAdBd.T.dot( Ninv ).dot( OdAdBs ).dot( spsp )  )
	m10 = np.trace( - OdAdBs.T.dot( Ninv ).dot( OmegaA ).dot( AtNAinv ).dot( OmegaA.T ).dot( Ninv ).dot( OdAdBd ).dot( spsp ) + OdAdBs.T.dot( Ninv ).dot( OdAdBd ).dot( spsp )  )
	m11 = np.trace( - OdAdBs.T.dot( Ninv ).dot( OmegaA ).dot( AtNAinv ).dot( OmegaA.T ).dot( Ninv ).dot( OdAdBs ).dot( spsp ) + OdAdBs.T.dot( Ninv ).dot( OdAdBs ).dot( spsp )  )
	if T_second_grey_body!=0.0:
		m02 = np.trace( - OdAdBd.T.dot( Ninv ).dot( OmegaA ).dot( AtNAinv ).dot( OmegaA.T ).dot( Ninv ).dot( OdAdBd_d ).dot( spsp ) + OdAdBd.T.dot( Ninv ).dot( OdAdBd_d ).dot( spsp )  )
		m12 = np.trace( - OdAdBs.T.dot( Ninv ).dot( OmegaA ).dot( AtNAinv ).dot( OmegaA.T ).dot( Ninv ).dot( OdAdBd_d ).dot( spsp ) + OdAdBs.T.dot( Ninv ).dot( OdAdBd_d ).dot( spsp )  )
		m20 = np.trace( - OdAdBd_d.T.dot( Ninv ).dot( OmegaA ).dot( AtNAinv ).dot( OmegaA.T ).dot( Ninv ).dot( OdAdBd ).dot( spsp ) + OdAdBd_d.T.dot( Ninv ).dot( OdAdBd ).dot( spsp )  )
		m21 = np.trace( - OdAdBd_d.T.dot( Ninv ).dot( OmegaA ).dot( AtNAinv ).dot( OmegaA.T ).dot( Ninv ).dot( OdAdBs ).dot( spsp ) + OdAdBd_d.T.dot( Ninv ).dot( OdAdBs ).dot( spsp )  )
		m22 = np.trace( - OdAdBd_d.T.dot( Ninv ).dot( OmegaA ).dot( AtNAinv ).dot( OmegaA.T ).dot( Ninv ).dot( OdAdBd_d ).dot( spsp ) + OdAdBd_d.T.dot( Ninv ).dot( OdAdBd_d ).dot( spsp )  )
		d2LdBdB_fg_block = np.array( [[m00, m01, m02],[m10, m11, m12],[m20, m21, m22]] )
	else:
		d2LdBdB_fg_block = np.array( [[m00, m01],[m10, m11]] )
	d2LdBdB_fg_block /= npatch	

	#calibration_fixed = True
	################### CALIBRATION ERROR ??
	if not calibration_fixed:
		print ' CALIBRATION ERROR INCLUDED, ASSUMING SIGMA(OMEGA)=',sigma_omega
		# OMEGA x OMEGA BLOCK
		dOmegadomega = np.zeros((nch, 2*nch, 2*nch))
		for p in range( nch ):
			for k in range( 2*nch ):
				if round(k/2) == p :
					dOmegadomega[ p,k,k ] = 1.
				else:
					dOmegadomega[ p,k,k ] = 0
		d2LdBdB_omega_omega = np.zeros((nch, nch))
		sigma_omega_v = np.ones(nch)/sigma_omega**2
		####
		## calibration relative to CMB channels for the main instrument
		'''
		for f in range( nch-7 ):
			if ((freqs[f] >= 60) and (freqs[f] <= 200)):
				print 'chosen frequencies ' , freqs[f]
				sigma_omega_v[f] *= 1e24
		'''		
		####
		prior = np.diag( sigma_omega_v )
		for p in range( nch ) :
			dOmegadomega_p_A =  np.squeeze(dOmegadomega[p,:,:]).dot(A)
			for q in range( nch ):
				dOmegadomega_q_A =  np.squeeze(dOmegadomega[q,:,:]).dot(A)
				d2LdBdB_omega_omega[ p,q ] = np.trace( -dOmegadomega_p_A.T.dot( Ninv ).dot( OmegaA ).dot( AtNAinv ).dot( OmegaA.T ).dot(Ninv).dot( dOmegadomega_q_A ).dot( spsp ) \
					+ dOmegadomega_p_A.T.dot( Ninv ).dot( dOmegadomega_q_A ).dot( spsp ) + prior[ p,q ] )	

		# OMEGA x FOREGROUNDS BLOCK
		d2LdBdB_dust_omega = np.zeros((1,nch))
		d2LdBdB_sync_omega = np.zeros((1,nch))
		d2LdBdB_omega_dust = np.zeros((nch,1))
		d2LdBdB_omega_sync = np.zeros((nch,1))
		for j in range( nch ):
			dOmegajA = np.squeeze(dOmegadomega[j,:,:]).dot(A)
			d2LdBdB_dust_omega[ 0, j ] = np.trace(-OdAdBd.T.dot(Ninv).dot(OmegaA).dot(AtNAinv).dot(OmegaA.T).dot(Ninv).dot(dOmegajA).dot(spsp) + OdAdBd.T.dot(Ninv).dot(dOmegajA).dot(spsp))			
			d2LdBdB_sync_omega[ 0, j ] = np.trace(-OdAdBs.T.dot(Ninv).dot(OmegaA).dot(AtNAinv).dot(OmegaA.T).dot(Ninv).dot(dOmegajA).dot(spsp) + OdAdBs.T.dot(Ninv).dot(dOmegajA).dot(spsp))
			d2LdBdB_omega_dust[ j, 0 ] = np.trace(-dOmegajA.T.dot(Ninv).dot(OmegaA).dot(AtNAinv).dot(OmegaA.T).dot(Ninv).dot(OdAdBd).dot(spsp) + dOmegajA.T.dot(Ninv).dot(OdAdBd).dot(spsp))
			d2LdBdB_omega_sync[ j, 0 ] = np.trace(-dOmegajA.T.dot(Ninv).dot(OmegaA).dot(AtNAinv).dot(OmegaA.T).dot(Ninv).dot(OdAdBs).dot(spsp) + dOmegajA.T.dot(Ninv).dot(OdAdBs).dot(spsp))        

		d2LdBdB_dust_omega /= np.sqrt(npatch)
		d2LdBdB_sync_omega /= np.sqrt(npatch)
		d2LdBdB_omega_dust /= np.sqrt(npatch)
		d2LdBdB_omega_sync /= np.sqrt(npatch)

		fg_x_omega = np.vstack(( d2LdBdB_dust_omega, d2LdBdB_sync_omega))
		omega_x_fg = np.hstack(( d2LdBdB_omega_dust, d2LdBdB_omega_sync))
		d2LdBdB = np.vstack(( np.hstack(( d2LdBdB_fg_block, fg_x_omega)),np.hstack(( omega_x_fg ,d2LdBdB_omega_omega))))

		#d2LdBdB[3,:] *= 1e24
		#d2LdBdB[:,3] *= 1e24		
		#pl.matshow(np.log10(np.abs(d2LdBdB)))
		#pl.colorbar()
		#pl.show()
		#exit()

		## filters out Planck # Planck has no calibration error
		# d2LdBdB = d2LdBdB[:nch+2-7,:nch+2-7]

	else:
		d2LdBdB = d2LdBdB_fg_block

	if ('sync' in components) and ('dust' not in components):
		if prior_sync != 0.0:
			print ' /!\ prior on sync /!\  '
			d2LdBdB[1,1] += 1.0/prior_sync**2
		d2LdBdBinv = np.array( 1.0 / d2LdBdB[1,1] )

	elif ('dust' in components) and ('sync' not in components):

		if T_second_grey_body!=0.0: ## d2LdBdB is 3 x 3 
			if prior_dust != 0.0:
				print ' /!\ prior on dust /!\  '
				prior_loc = np.diag(np.ones(len(d2LdBdB))*1.0/prior_dust**2)
				d2LdBdB += prior_loc
			d2LdBdBinv = np.linalg.inv( np.array( [[d2LdBdB[0,0], d2LdBdB[0,2]],[d2LdBdB[2,0], d2LdBdB[2,2]]] ))

		else:
			if prior_dust != 0.0:
				print ' /!\ prior on dust /!\  '
				d2LdBdB[0,0] += 1.0/prior_dust**2
			d2LdBdBinv = np.array( 1.0 / d2LdBdB[0,0] )

	else:
		## inversion of the 3 x 3 d2LdBdB
		if prior_dust != 0.0:
			print ' /!\ prior on dust /!\  '
			d2LdBdB[0,0] += 1/prior_dust**2
		if prior_sync != 0.0:
			print ' /!\ prior on sync /!\  '
			if T_second_grey_body!=0.0: ## d2L
				d2LdBdB[2,2] += 1/prior_sync**2
			else:
				d2LdBdB[1,1] += 1/prior_sync**2
		d2LdBdBinv = np.linalg.inv( d2LdBdB )


	######################################################
	if (len(components)==1 and T_second_grey_body==0.0) or (('sync' in components) and ('dust' not in components)):
		d2LdBdBinv_loc = d2LdBdBinv
	else:
		d2LdBdBinv_loc = d2LdBdBinv[0,0]

	#print d2LdBdBinv_loc
	if d2LdBdBinv_loc <=0 or d2LdBdBinv_loc>=1e7:
		print 'wrong sigma( beta_dust ) value '
		if cmb_noise_output:
			return Cl_dust*1e6, np.sqrt( AtNAinv[0,0] )
		elif everything_output:
			if stolyarov:
				return Cl_dust*1e6, np.sqrt( AtNAinv[0,0] ), np.array([[1e6,1e6,1e6],[1e6,1e6,1e6],[1e6,1e6,1e6]]), Cl_dust*1e6
			else:
				return Cl_dust*1e6, np.sqrt( AtNAinv[0,0] ), np.array([[1e6,1e6,1e6],[1e6,1e6,1e6],[1e6,1e6,1e6]])
		else:
			return Cl_dust*1e6
	#################################################

	if (len(components)==1 and T_second_grey_body==0.0) or (('sync' in components) and ('dust' not in components)):
		print ' and **after** adding any prior, d2LdBdB is a scalar and diag( sqrt( d2LdBdB inv )) = ', np.sqrt( d2LdBdBinv )
	else:
		print ' and **after** adding any prior, diag( sqrt( d2LdBdB inv )) = ', np.sqrt(np.diag(d2LdBdBinv))

	if output == 'delta_beta':
		#if (stolyarov or T_second_grey_body!=0.0) :
		#	if ('dust' not in components) or ('sync' not in components) :
		#		return np.sqrt( d2LdBdBinv )
		#	else:
		#		return np.sqrt( np.diag( d2LdBdBinv) )
		#else:
		return np.sqrt( d2LdBdBinv )


	#######################
	## computation of alpha
	Ninv_sq = np.zeros((nch, nch))
	for k in range(nch):
		Ninv_sq[ k,k ] = ((1.0/sensitivity_uK_per_chan[ int(k) ])**2) * Nbolos[ int(k) ]

	Omega_sq = omega_computation(nch, squeeze=True)
	dOmegadomega_sq = np.zeros((nch, nch, nch))
	if T_second_grey_body!=0.0:
		if 'sync' not in components:
			components_loc = ['dust']
		else:
			components_loc = ['dust', 'sync']
		A_sq, dAdBd_sq, dAdBd_d_sq, dAdBs_sq = A_and_dAdB_matrix_computation(freqs, Bd, Bs, adust=adust, async=async, squeeze=True, T_second_grey_body=T_second_grey_body, bandpass_channels=bandpass_channels, Temp=Temp, Bd2=Bd2, components=components_loc ) 
	else:
		A_sq, dAdBd_sq, dAdBs_sq = A_and_dAdB_matrix_computation(freqs, Bd, Bs, adust=adust, async=async, squeeze=True, stolyarov=stolyarov, bandpass_channels=bandpass_channels, Temp=Temp, components=components) 

	for p in range(nch):
		for k in range(nch):
			if (k == p):
				dOmegadomega_sq[p,k,k] = 1
			else:
				dOmegadomega_sq[p,k,k] = 0
	OmegaA_sq = Omega_sq.dot( A_sq )
	AtNA_sq = OmegaA_sq.T.dot( Ninv_sq ).dot( OmegaA_sq )
	AtNAinv_sq = np.linalg.inv( AtNA_sq )

	if (len(components)==1 and T_second_grey_body==0.0) or (('sync' in components) and ('dust' not in components)):
		ncomp = 1
	else:
		ncomp = len(d2LdBdBinv)

	stolyarov_args = ['']
	Td_fixed=True

	# create channels for the construction of alpha
	if ('Bd' in stolyarov_args):
		if ('Td' in stolyarov_args):
			if ('Bs' in stolyarov_args):
				if 'dust' in components:
					if 'sync' in components:
						channels = ['cmb', 'dust', 'delta_Bd', 'delta_Td', 'sync', 'delta_Bs']
					else:
						channels = ['cmb', 'dust', 'delta_Bd', 'delta_Td' ]
				else:
					if 'sync' in components:
						channels = ['cmb', 'sync', 'delta_Bs']
					else:
						channels = ['cmb' ]
			else:
				if 'dust' in components:
					if 'sync' in components:
						channels = ['cmb', 'dust', 'delta_Bd', 'delta_Td', 'sync']
					else:
						channels = ['cmb', 'dust', 'delta_Bd', 'delta_Td']
				else:
					if 'sync' in components:
						channels = ['cmb', 'sync']
					else:
						channels = ['cmb' ]				
		else:
			if ('Bs' in stolyarov_args):
				if 'dust' in components:
					if 'sync' in components:
						channels = ['cmb', 'dust', 'delta_Bd', 'sync', 'delta_Bs']
					else:
						channels = ['cmb', 'dust', 'delta_Bd']
				else:
					if 'sync' in components:
						channels = ['cmb','sync', 'delta_Bs']
					else:
						channels = ['cmb']
			else:
				if 'dust' in components:
					if 'sync' in components:
						channels = ['cmb', 'dust', 'delta_Bd', 'sync' ]
					else:
						channels = ['cmb', 'dust', 'delta_Bd']
				else:
					if 'sync' in components:
						channels = ['cmb', 'sync' ]
					else:
						channels = ['cmb']
	else:
		if ('Td' in stolyarov_args):
			if ('Bs' in stolyarov_args):
				if 'dust' in components:
					if 'sync' in components:
						channels = ['cmb', 'dust', 'delta_Td', 'sync', 'delta_Bs']
					else:
						channels = ['cmb', 'dust', 'delta_Td']
				else:
					if 'sync' in components:
						channels = ['cmb', 'sync', 'delta_Bs']
					else:
						channels = ['cmb']
			else:
				if 'dust' in components:
					if 'sync' in components:
						channels = ['cmb', 'dust', 'delta_Td', 'sync' ]
					else:
						channels = ['cmb', 'dust', 'delta_Td' ]
				else:
					if 'sync' in components:
						channels = ['cmb', 'sync' ]
					else:
						channels = ['cmb']
		else:
			if ('Bs' in stolyarov_args):
				if 'dust' in components:
					if 'sync' in components:
						channels = ['cmb', 'dust', 'sync', 'delta_Bs']
					else:
						channels = ['cmb', 'dust']
				else:
					if 'sync' in components:
						channels = ['cmb','sync', 'delta_Bs']
					else:
						channels = ['cmb']					
			else:
				if 'dust' in components:
					if 'sync' in components:
						channels = ['cmb', 'dust', 'sync' ]
					else:
						channels = ['cmb', 'dust']
				else:
					if 'sync' in components:
						channels = ['cmb', 'sync' ]
					else:
						channels = ['cmb']
	
	if T_second_grey_body != 0 :
		channels.append('dust')

	# checking if we are not doing things wrong .. 
	#assert  ('cmb' in channels) and (components in channels)

	# if calibration is fixed, the corresponding alpha components would be set to zero
	#if ncomp > 3 or ncomp < 2 :
	#	print 'if Bd, Td and Bs are not varying alltogether, then it is not coded yet ... '
	#	exit()


	alpha = np.zeros(( len(channels),len(channels), ncomp )) # because 6 = CMB + 3 dust + 2 sync

	if 'dust' in components:
		if 'sync' in components:
			# Bd
			alpha[:,:,0] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdBd_sq)
			if not Td_fixed:
				# Td
				alpha[:,:,1] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdTd_sq)
				# Bs
				alpha[:,:,2] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdBs_sq)
				# Bd#2
				if T_second_grey_body != 0.0:
					print ' this is not coded yet ... '
					exit()
			else:
				# Bs
				alpha[:,:,1] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdBs_sq)
				# Bd#2
				if T_second_grey_body != 0.0:
					alpha[:,:,2] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdBd_d_sq)
		else:
			# Bd
			alpha[:,:,0] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdBd_sq)
			if not Td_fixed:
				# Td
				alpha[:,:,1] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdTd_sq)
				# Bd#2
				if T_second_grey_body != 0.0:
					print ' this is not coded yet ... '
					exit()
			# Bd#2
			if T_second_grey_body != 0.0:
				alpha[:,:,1] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdBd_d_sq)
	else:
		# Bs
		alpha[:,:,0] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdBs_sq)
		if not Td_fixed:
			# Td
			alpha[:,:,1] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdTd_sq)


	AlphaAlpha_k1k2 = np.zeros(( len(channels), len(channels), ncomp, ncomp ))
	Cl_j1j2 = np.zeros(( len(channels), len(channels), ell_max-ell_min+1 ))

	for j1 in range(len(channels)):
		s1 = channels[j1]

		for j2 in range(len(channels)):
			s2 = channels[j2]

			# does the power spectrum exist?
			# otherwise compute it
			if s1 =='cmb' and s2=='cmb':
				Cl_j1j2[j1,j2,:] = Cl_cmb[ell_min-2:ell_max-1]
			elif ( s1 =='cmb' and s2!='cmb' ) or ( s2 =='cmb' and s1!='cmb' ):
				Cl_j1j2[j1,j2,:] = Cl_cmb[ell_min-2:ell_max-1]*0.0
			elif ((s1 =='dust') and (s2=='dust')):
				if j1 == j2:
					print 'oh, this is dust x dust and it is a diagonal element, I use the input Cl'
					Cl_j1j2[j1,j2,:] = Cl_dust[ell_min-2:ell_max-1]
				else:
					print 'oh, this is dust x dust and it is a off-diagonal element, I use the input Cl *with* the correlation_2_dusts factor = ', correlation_2_dusts
					Cl_j1j2[j1,j2,:] = correlation_2_dusts**2 * Cl_dust[ell_min-2:ell_max-1]
			elif (s1=='dust' and s2=='sync') or (s2=='dust' and s1=='sync') :
				print 'oh, this is dust x sync, I use the input Cl'
				Cl_j1j2[j1,j2,:] = Cl_dxs[ell_min-2:ell_max-1]
			elif s1=='sync' and s2=='sync':
				print 'oh, this is sync x sync, I use the input Cl!'
				Cl_j1j2[j1,j2,:] = Cl_sync[ell_min-2:ell_max-1]
			else:
				if j2<j1:
					print 'j2<j1 ->> symmetry relation ! '
					Cl_j1j2[j1,j2,:] = Cl_j1j2[j2,j1,:]

			# building the alpha alpha tensor
			for k1 in range(ncomp):
				for k2 in range(ncomp):
					AlphaAlpha_k1k2[j1,j2,k1,k2] = alpha[ 0,j1,k1 ]*alpha[ 0,j2,k2]

	## power estimation from analytical formula
	Cl_res = np.zeros( ell_max-ell_min+1 )

	#### TEST CALIBRATION 
	#ncomp -= len(freqs)

	for ell_ind in range( ell_max-ell_min+1 ) :
		
		clkk = np.zeros((ncomp,ncomp))

		for k1 in range(ncomp):
			for k2 in range(ncomp):

				clkk_loc = 0.0

				for j1 in range(len(channels)):
					
					for j2 in range(len(channels)):
						
						if ncomp == 1:
							d2LdBdBinv_loc = d2LdBdBinv*1.0
						else:
							d2LdBdBinv_loc = d2LdBdBinv[ k1,k2 ]*1.0

						clkk_loc += d2LdBdBinv_loc * AlphaAlpha_k1k2[j1,j2,k1,k2] *  Cl_j1j2[ j1,j2,ell_ind ]

				clkk[ k1,k2 ] = clkk_loc

		Cl_res[ell_ind] = np.sum(np.sum( clkk ))

	sqrtAtNAinv00 = np.sqrt( AtNAinv[0,0] )

	if sqrtAtNAinv00 != sqrtAtNAinv00:
		sqrtAtNAinv00 = 1e3

	####################################################
	if return_Cl_res_beta_var:
		'''
		we estimate here the effect of spatial variations 
		on BB residuals if we don't try to control them
		'''
		print 'ADDING Td IN ADDITION TO OTHER SPECTRAL PARAMETERS ! '
		Td_fixed = False
		A_sq, dAdBd_sq, dAdTd_sq, dAdBs_sq = res_ext.A_and_dAdB_matrix_computation(freqs, Bd, Temp, Bs, squeeze=True, \
						bandpass_channels=bandpass_channels, Td_fixed=Td_fixed, components=components, stolyarov_args=[]) 
		ncomp += 1
		#######

		alpha = np.zeros(( len(channels),len(channels), ncomp )) # because 6 = CMB + 3 dust + 2 sync

		if 'dust' in components:
			if 'sync' in components:
				# Bd
				alpha[:,:,0] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdBd_sq)
				if not Td_fixed:
					# Td
					alpha[:,:,1] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdTd_sq)
					# Bs
					alpha[:,:,2] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdBs_sq)
					# Bd#2
					if T_second_grey_body != 0.0:
						print ' this is not coded yet ... '
						exit()
				else:
					# Bs
					alpha[:,:,1] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdBs_sq)
					# Bd#2
					if T_second_grey_body != 0.0:
						alpha[:,:,2] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdBd_d_sq)
			else:
				# Bd
				alpha[:,:,0] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdBd_sq)
				if not Td_fixed:
					# Td
					alpha[:,:,1] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdTd_sq)
					# Bd#2
					if T_second_grey_body != 0.0:
						print ' this is not coded yet ... '
						exit()
				# Bd#2
				if T_second_grey_body != 0.0:
					alpha[:,:,1] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdBd_d_sq)
		else:
			# Bs
			alpha[:,:,0] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdBs_sq)
			if not Td_fixed:
				# Td
				alpha[:,:,1] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdTd_sq)


		AlphaAlpha_k1k2 = np.zeros(( len(channels), len(channels), ncomp, ncomp ))
		Cl_j1j2 = np.zeros(( len(channels), len(channels), ell_max-ell_min+1 ))

		for j1 in range(len(channels)):
			s1 = channels[j1]

			for j2 in range(len(channels)):
				s2 = channels[j2]

				# does the power spectrum exist?
				# otherwise compute it
				if s1 =='cmb' and s2=='cmb':
					Cl_j1j2[j1,j2,:] = Cl_cmb[ell_min-2:ell_max-1]
				elif ( s1 =='cmb' and s2!='cmb' ) or ( s2 =='cmb' and s1!='cmb' ):
					Cl_j1j2[j1,j2,:] = Cl_cmb[ell_min-2:ell_max-1]*0.0
				elif ((s1 =='dust') and (s2=='dust')):
					if j1 == j2:
						print 'oh, this is dust x dust and it is a diagonal element, I use the input Cl'
						Cl_j1j2[j1,j2,:] = Cl_dust[ell_min-2:ell_max-1]
					else:
						print 'oh, this is dust x dust and it is a off-diagonal element, I use the input Cl *with* the correlation_2_dusts factor = ', correlation_2_dusts
						Cl_j1j2[j1,j2,:] = correlation_2_dusts**2 * Cl_dust[ell_min-2:ell_max-1]
				elif (s1=='dust' and s2=='sync') or (s2=='dust' and s1=='sync') :
					print 'oh, this is dust x sync, I use the input Cl'
					Cl_j1j2[j1,j2,:] = Cl_dxs[ell_min-2:ell_max-1]
				elif s1=='sync' and s2=='sync':
					print 'oh, this is sync x sync, I use the input Cl!'
					Cl_j1j2[j1,j2,:] = Cl_sync[ell_min-2:ell_max-1]
				else:
					if j2<j1:
						print 'j2<j1 ->> symmetry relation ! '
						Cl_j1j2[j1,j2,:] = Cl_j1j2[j2,j1,:]

				# building the alpha alpha tensor
				for k1 in range(ncomp):
					for k2 in range(ncomp):
						AlphaAlpha_k1k2[j1,j2,k1,k2] = alpha[ 0,j1,k1 ]*alpha[ 0,j2,k2]

		#######
		Cl_res_Bd = np.zeros( ell_max-ell_min+1 )
		Cl_res_Bs = np.zeros( ell_max-ell_min+1 )
		Cl_res_Td = np.zeros( ell_max-ell_min+1 )
		Cl_res_BdBsTd = np.zeros( ell_max-ell_min+1 )

		d2LdBdBinv_beta_spat_var_BdBsTd = np.array(np.diag([0.025, 1.5, 0.050])**2)
		d2LdBdBinv_beta_spat_var_Bd = np.array(np.diag([0.025, 0.0, 0.0])**2)
		d2LdBdBinv_beta_spat_var_Bs = np.array(np.diag([0.0, 0.0, 0.050])**2)
		d2LdBdBinv_beta_spat_var_Td = np.array(np.diag([0.0, 1.5, 0.0])**2)

		for ell_ind in range( ell_max-ell_min+1 ) :
			
			clkk_Bd = np.zeros((ncomp,ncomp))
			clkk_Bs = np.zeros((ncomp,ncomp))
			clkk_Td = np.zeros((ncomp,ncomp))
			clkk_BdBsTd = np.zeros((ncomp,ncomp))

			for k1 in range(ncomp):
				for k2 in range(ncomp):

					clkk_loc_Bd = 0.0
					clkk_loc_Bs = 0.0
					clkk_loc_Td = 0.0
					clkk_loc_BdBsTd = 0.0

					for j1 in range(len(channels)):
						
						for j2 in range(len(channels)):

							d2LdBdBinv_loc_Bd = d2LdBdBinv_beta_spat_var_Bd[ k1,k2 ]*1.0
							d2LdBdBinv_loc_Bs = d2LdBdBinv_beta_spat_var_Bs[ k1,k2 ]*1.0
							d2LdBdBinv_loc_Td = d2LdBdBinv_beta_spat_var_Td[ k1,k2 ]*1.0
							d2LdBdBinv_loc_BdBsTd = d2LdBdBinv_beta_spat_var_BdBsTd[ k1,k2 ]*1.0

							clkk_loc_Bd += d2LdBdBinv_loc_Bd * AlphaAlpha_k1k2[j1,j2,k1,k2] *  Cl_j1j2[ j1,j2,ell_ind ]
							clkk_loc_Bs += d2LdBdBinv_loc_Bs * AlphaAlpha_k1k2[j1,j2,k1,k2] *  Cl_j1j2[ j1,j2,ell_ind ]
							clkk_loc_Td += d2LdBdBinv_loc_Td * AlphaAlpha_k1k2[j1,j2,k1,k2] *  Cl_j1j2[ j1,j2,ell_ind ]
							clkk_loc_BdBsTd += d2LdBdBinv_loc_BdBsTd * AlphaAlpha_k1k2[j1,j2,k1,k2] *  Cl_j1j2[ j1,j2,ell_ind ]

					clkk_Bd[ k1,k2 ] = clkk_loc_Bd
					clkk_Bs[ k1,k2 ] = clkk_loc_Bs
					clkk_Td[ k1,k2 ] = clkk_loc_Td
					clkk_BdBsTd[ k1,k2 ] = clkk_loc_BdBsTd

			Cl_res_Bd[ell_ind] = np.sum(np.sum( clkk_Bd ))
			Cl_res_Bs[ell_ind] = np.sum(np.sum( clkk_Bs ))
			Cl_res_Td[ell_ind] = np.sum(np.sum( clkk_Td ))
			Cl_res_BdBsTd[ell_ind] = np.sum(np.sum( clkk_BdBsTd ))

		return Cl_res, sqrtAtNAinv00, np.sqrt( d2LdBdBinv ), Cl_res_Bd, Cl_res_Bs, Cl_res_Td, Cl_res_BdBsTd
	
	else:
	
		return Cl_res, sqrtAtNAinv00, np.sqrt( d2LdBdBinv )

