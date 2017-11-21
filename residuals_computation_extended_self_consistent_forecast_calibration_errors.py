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
import pylab as pl
import noise_computation

#########################################################################

Bd_fiducial = 1.59
Bs_fiducial = -3.1

h_over_k =  0.0479924
nu_ref= 150.0
Temp_fiducial=19.6
cst = 56.8

def factor_dust_computation( nu, Td ):
	nu_dust = Td/h_over_k
	return (np.exp(nu_ref/nu_dust) - 1)/(np.exp(nu/nu_dust)-1)

def dfactor_dustdTd_computation( nu, Td ):
	nu_dust = Td/h_over_k
	factor_dust = factor_dust_computation( nu, Td )
	return h_over_k/Td**2*( - nu_ref * np.exp(nu_ref/nu_dust) +  nu * np.exp( nu/nu_dust )*factor_dust ) / ( np.exp(nu/nu_dust) - 1 )

def d2factor_dustdTd2_computation( nu, Td ):
	nu_dust = Td/h_over_k
	factor_dust = factor_dust_computation( nu, Td )
	#term1 = -1/Td * dfactor_dustdTd_computation( nu, Td )
	# dfactor_dustdTd_computation = term1 x term2
	term1 = h_over_k/Td**2
	term1_d = - h_over_k/Td**3
	# term2 = u / v
	u = - nu_ref * np.exp(nu_ref/nu_dust) +  nu * np.exp( nu/nu_dust )*factor_dust
	v = ( np.exp(nu/nu_dust) - 1 )
	term2 = u/v
	ud =  + h_over_k*nu_ref**2/Td**2 * np.exp(nu_ref/nu_dust) \
				- h_over_k*nu**2/Td**2 * np.exp( nu/nu_dust )*factor_dust  \
				+  nu * np.exp( nu/nu_dust )*dfactor_dustdTd_computation( nu, Td )
	vd =  - h_over_k*nu/Td**2 * np.exp(nu/nu_dust) 
	term2_d = (  ud * v  - u*vd )/v**2
	# 
	return term1_d*term2 + term1*term2_d

#########################################################################

def A_element_computation(nu_loc, Bd, Td, Bs, squeeze=False, bandpass_channels=0, stolyarov_args=['Bd', 'Td', 'Bs'], Td_fixed=False, components=['dust','sync']):

	Delta_f = bandpass_channels
	freq_range = np.arange( nu_loc*(1-Delta_f/2), nu_loc*(1+Delta_f/2)+1 )

	a_cmb = 0.0
	a_dust = 0.0
	a_sync = 0.0
	a_dBd_dust = 0.0
	a_dTd_dust = 0.0
	a_dBs_sync = 0.0


	for nu_i in freq_range:

		nu_dust = Td/h_over_k

		factor_dust = factor_dust_computation( nu_i, Td )
		a_cmb_i = (nu_i/cst)**(2)*(np.exp(nu_i/cst))/( (np.exp(nu_i/cst)-1)**(2) )
		a_dust_i = factor_dust*(nu_i/nu_ref)**(1+Bd)
		a_sync_i = (nu_i/nu_ref)**(Bs)

		a_dBd_dust_i = factor_dust*(np.log(nu_i/nu_ref)*(nu_i/nu_ref)**(1+Bd))

		a_dTd_dust_i = dfactor_dustdTd_computation( nu_i, Td )*(nu_i/nu_ref)**(1+Bd)

		a_dBs_sync_i = np.log(nu_i/nu_ref)*(nu_i/nu_ref)**Bs

		a_cmb += a_cmb_i
		a_dust += a_dust_i
		a_sync += a_sync_i
		a_dBd_dust += a_dBd_dust_i
		a_dTd_dust += a_dTd_dust_i
		a_dBs_sync += a_dBs_sync_i


	a_cmb /= len(freq_range)
	a_dust /= len(freq_range)
	a_sync /= len(freq_range)
	a_dBd_dust /= len(freq_range)
	a_dTd_dust /= len(freq_range)
	a_dBs_sync /= len(freq_range)

	if squeeze == True:
		
		A = a_cmb, a_dust, a_dBd_dust, a_dTd_dust, a_sync, a_dBs_sync

		A = stolyarov_selection( A, stolyarov_args, squeeze=True, components=components )

	else:

		A_cmb = np.array([[a_cmb, 0],[0, a_cmb]])
		A_dust = np.array([[a_dust, 0],[0, a_dust]])
		A_sync = np.array([[a_sync, 0],[0, a_sync]])
		A_dBd_dust = [[a_dBd_dust, 0],[0, a_dBd_dust]]
		A_dTd_dust = [[a_dTd_dust, 0],[0, a_dTd_dust]]
		A_dBs_sync = [[a_dBs_sync, 0],[0, a_dBs_sync]]

		A = A_cmb, A_dust, A_dBd_dust, A_dTd_dust, A_sync, A_dBs_sync 

		A = stolyarov_selection( A, stolyarov_args, squeeze=False, components=components  )

	return A 


def dAdB_element_computation( nu_loc, Bd, Td, Bs, squeeze=False, bandpass_channels=0,\
					 stolyarov_args=['Bd', 'Td', 'Bs'], Td_fixed=False, components=['dust', 'sync'] ):

	Delta_f = bandpass_channels
	freq_range = np.arange( nu_loc*(1-Delta_f/2), nu_loc*(1+Delta_f/2)+1 )

	dAdBd_tot=0.0
	dAdTd_tot=0.0
	dAdBs_tot=0.0

	for nu_i in freq_range:

		nu_dust = Td/h_over_k
	
		factor_dust = factor_dust_computation( nu_i, Td )

		dadBdi = factor_dust*np.log(nu_i/nu_ref)*(nu_i/nu_ref)**(1+Bd)
		dadBd_dBdi = factor_dust*(np.log(nu_i/nu_ref))**2 * (nu_i/nu_ref)**(1+Bd)
		dadBd_dTdi = dfactor_dustdTd_computation( nu_i, Td ) * np.log(nu_i/nu_ref) * (nu_i/nu_ref)**(1+Bd)
		dadTdi =  dfactor_dustdTd_computation( nu_i, Td )*(nu_i/nu_ref)**(1+Bd)
		dadTd_dTdi = d2factor_dustdTd2_computation( nu_i, Td )*(nu_i/nu_ref)**(1+Bd)
		dadTd_dBdi = dfactor_dustdTd_computation( nu_i, Td ) * np.log(nu_i/nu_ref) * (nu_i/nu_ref)**(1+Bd)
		dadBsi = np.log(nu_i/nu_ref)*(nu_i/nu_ref)**(Bs)
		dadBs_dBsi = (np.log(nu_i/nu_ref))**2 * (nu_i/nu_ref)**Bs

		if squeeze == True:
			## cmb, dust, dustBd, dustTd, sync, syncBs
			dAdBd = 0, dadBdi, dadBd_dBdi, dadTd_dBdi, 0, 0 
			dAdTd = 0, dadTdi, dadBd_dTdi, dadTd_dTdi, 0, 0 
			dAdBs = 0, 0, 0, 0, dadBsi, dadBs_dBsi 

			dAdBd = stolyarov_selection(dAdBd, stolyarov_args, squeeze=True, components=components  )
			dAdTd = stolyarov_selection(dAdTd, stolyarov_args, squeeze=True, components=components  )
			dAdBs = stolyarov_selection(dAdBs, stolyarov_args, squeeze=True, components=components  )

		else:
			dA_dustdBd = np.array( [[dadBdi, 0],[0, dadBdi]] )
			dA_dustdTd = np.array( [[dadTdi, 0],[0, dadTdi]] )
			dA_syncdBs = np.array( [[dadBsi, 0],[0, dadBsi]] )
			dA_dust_dBd_dBd = np.array( [[dadBd_dBdi, 0],[0, dadBd_dBdi]] )
			dA_dust_dBd_dTd = np.array( [[dadBd_dTdi, 0],[0, dadBd_dTdi]] )
			dA_dust_dTd_dBd = np.array( [[dadTd_dBdi, 0],[0, dadTd_dBdi]] )
			dA_dust_dTd_dTd = np.array( [[dadTd_dTdi, 0],[0, dadTd_dTdi]] )
			dA_sync_dBs_dBs = np.array( [[dadBs_dBsi, 0],[0, dadBs_dBsi]] )
			## the order of elements is cmb, dust, dust_dBd, dust_dTd, sync, sync_dBs
			dAdBd = np.zeros((2,2)), dA_dustdBd, dA_dust_dBd_dBd, dA_dust_dTd_dBd, np.zeros((2,2)), np.zeros((2,2))
			dAdTd = np.zeros((2,2)), dA_dustdTd, dA_dust_dBd_dTd, dA_dust_dTd_dTd, np.zeros((2,2)), np.zeros((2,2))
			dAdBs = np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), dA_syncdBs, dA_sync_dBs_dBs

			dAdBd = stolyarov_selection(dAdBd, stolyarov_args, squeeze=False, components=components )
			dAdTd = stolyarov_selection(dAdTd, stolyarov_args, squeeze=False, components=components )
			dAdBs = stolyarov_selection(dAdBs, stolyarov_args, squeeze=False, components=components )

		dAdBd_tot += dAdBd
		dAdTd_tot += dAdTd
		dAdBs_tot += dAdBs

	return np.array(dAdBd_tot)/len(freq_range), np.array(dAdTd_tot)/len(freq_range), np.array(dAdBs_tot)/len(freq_range)


def stolyarov_selection( x, stolyarov_args, squeeze=False, components=['dust','sync'] ):
	if squeeze:
		a_cmb, a_dust, a_dBd_dust, a_dTd_dust, a_sync, a_dBs_sync = x
		#################################################################################
		if ('Bd' in stolyarov_args):
			if ('Td' in stolyarov_args):
				if ('Bs' in stolyarov_args):
					if 'dust' in components:
						if 'sync' in components:
							A = np.array([a_cmb, a_dust, a_dBd_dust, a_dTd_dust, a_sync, a_dBs_sync ])
						else:
							A = np.array([a_cmb, a_dust, a_dBd_dust, a_dTd_dust])
					else:
						if 'sync' in components:
							A = np.array([a_cmb, a_sync, a_dBs_sync ])
						else:
							A = np.array([a_cmb])
				else:
					if 'dust' in components:
						if 'sync' in components:
							A = np.array([a_cmb, a_dust, a_dBd_dust, a_dTd_dust, a_sync])
						else:
							A = np.array([a_cmb, a_dust, a_dBd_dust, a_dTd_dust])
					else:
						if 'sync' in components:
							A = np.array([a_cmb, a_sync])
						else:
							A = np.array([a_cmb])
			########################################################################
			else:
				if ('Bs' in stolyarov_args):
					if 'dust' in components:
						if 'sync' in components:
							A = np.array([a_cmb, a_dust, a_dBd_dust, a_sync, a_dBs_sync ])
						else:
							A = np.array([a_cmb, a_dust, a_dBd_dust ])
					else:
						if 'sync' in components:
							A = np.array([a_cmb, a_sync, a_dBs_sync ])
						else:
							A = np.array([ a_cmb ])
				else:
					if 'dust' in components:
						if 'sync' in components:
							A = np.array([a_cmb, a_dust, a_dBd_dust, a_sync])
						else:
							A = np.array([a_cmb, a_dust, a_dBd_dust])
					else:
						if 'sync' in components:
							A = np.array([a_cmb, a_sync])
						else:
							A = np.array([a_cmb ])
		else:
			########################################################################
			if ('Td' in stolyarov_args):
				if ('Bs' in stolyarov_args):
					if 'dust' in components:
						if 'sync' in components:
							A = np.array([a_cmb, a_dust, a_dTd_dust, a_sync, a_dBs_sync ])
						else:
							A = np.array([a_cmb, a_dust, a_dTd_dust ])
					else:
						if 'sync' in components:
							A = np.array([a_cmb, a_sync, a_dBs_sync ])
						else:
							A = np.array([a_cmb])						
				else:
					if 'dust' in components:
						if 'sync' in components:
							A = np.array([a_cmb, a_dust, a_dTd_dust, a_sync])
						else:
							A = np.array([a_cmb, a_dust, a_dTd_dust])
					else:
						if 'sync' in components:
							A = np.array([a_cmb, a_sync])
						else:
							A = np.array([ a_cmb ])
			########################################################################
			else:
				if ('Bs' in stolyarov_args):
					if 'dust' in components:
						if 'sync' in components:
							A = np.array([a_cmb, a_dust, a_sync, a_dBs_sync ])
						else:
							A = np.array([a_cmb, a_dust ])
					else:
						if 'sync' in components:
							A = np.array([a_cmb, a_sync, a_dBs_sync ])
						else:
							A = np.array([a_cmb ])
				else:
					if 'dust' in components:
						if 'sync' in components:
							A = np.array([a_cmb, a_dust, a_sync])
						else:
							A = np.array([a_cmb, a_dust])
					else:
						if 'sync' in components:
							A = np.array([a_cmb, a_sync])
						else:
							A = np.array([a_cmb])
	#################################################################################
	else:
		A_cmb, A_dust, A_dBd_dust, A_dTd_dust, A_sync, A_dBs_sync = x
		if ('Bd' in stolyarov_args):
			if ('Td' in stolyarov_args):
				if ('Bs' in stolyarov_args):
					if 'dust' in components:
						if 'sync' in components:
							A =  np.hstack(( A_cmb, A_dust, A_dBd_dust, A_dTd_dust, A_sync, A_dBs_sync ))
						else:
							A =  np.hstack(( A_cmb, A_dust, A_dBd_dust, A_dTd_dust ))
					else:
						if 'sync' in components:
							A =  np.hstack(( A_cmb, A_sync, A_dBs_sync ))
						else:
							A =  np.hstack(( A_cmb ))
				else:
					if 'dust' in components:
						if 'sync' in components:
							A = np.hstack(( A_cmb, A_dust, A_dBd_dust, A_dTd_dust, A_sync ))
						else:
							A = np.hstack(( A_cmb, A_dust, A_dBd_dust, A_dTd_dust ))
					else:
						if 'sync' in components:
							A = np.hstack(( A_cmb, A_sync ))
						else:
							A = np.hstack(( A_cmb ))
			else:
				if ('Bs' in stolyarov_args):
					if 'dust' in components:
						if 'sync' in components:
							A = np.hstack(( A_cmb, A_dust, A_dBd_dust, A_sync, A_dBs_sync ))
						else:
							A = np.hstack(( A_cmb, A_dust, A_dBd_dust ))
					else:
						if 'sync' in components:
							A = np.hstack(( A_cmb, A_sync, A_dBs_sync ))
						else:
							A = np.hstack(( A_cmb ))
				else:
					if 'dust' in components:
						if 'sync' in components:
							A = np.hstack(( A_cmb, A_dust, A_dBd_dust, A_sync ))
						else:
							A = np.hstack(( A_cmb, A_dust, A_dBd_dust ))
					else:
						if 'sync' in components:
							A = np.hstack(( A_cmb, A_sync ))
						else:
							A = np.hstack(( A_cmb ))
		else:
			if ('Td' in stolyarov_args):
				if ('Bs' in stolyarov_args):
					if 'dust' in components:
						if 'sync' in components:
							A = np.hstack(( A_cmb, A_dust, A_dTd_dust, A_sync, A_dBs_sync ))
						else:
							A = np.hstack(( A_cmb, A_dust, A_dTd_dust ))
					else:
						if 'sync' in components:
							A = np.hstack(( A_cmb, A_sync, A_dBs_sync ))
						else:
							A = np.hstack(( A_cmb ))
				else:
					if 'dust' in components:
						if 'sync' in components:
							A = np.hstack(( A_cmb, A_dust, A_dTd_dust, A_sync ))
						else:
							A = np.hstack(( A_cmb, A_dust, A_dTd_dust ))
					else:
						if 'sync' in components:
							A = np.hstack(( A_cmb, A_sync ))
						else:
							A = np.hstack(( A_cmb ))
			else:
				if ('Bs' in stolyarov_args):
					if 'dust' in components:
						if 'sync' in components:
							A = np.hstack(( A_cmb, A_dust, A_sync, A_dBs_sync ))
						else:
							A = np.hstack(( A_cmb, A_dust ))
					else:
						if 'sync' in components:
							A = np.hstack(( A_cmb, A_sync, A_dBs_sync ))
						else:
							A = np.hstack(( A_cmb ))
				else:
					if 'dust' in components:
						if 'sync' in components:
							A = np.hstack(( A_cmb, A_dust, A_sync ))
						else:
							A = np.hstack(( A_cmb, A_dust ))
					else:
						if 'sync' in components:
							A = np.hstack(( A_cmb, A_sync ))
						else:
							A = np.hstack(( A_cmb ))
	return A

def A_and_dAdB_matrix_computation(freqs, Bd, Td, Bs, squeeze=False, bandpass_channels=0, stolyarov_args=['Bd', 'Td', 'Bs'],\
			 Td_fixed=False, components=['dust','sync'] ):

	if bandpass_channels is 0.0 or bandpass_channels is None:
		bandpass_channels=np.zeros( len(freqs) )

	for f in range( len(freqs) ):

		Ai = A_element_computation( freqs[ f ], Bd, Td, Bs, squeeze, bandpass_channels=bandpass_channels[f],\
			 stolyarov_args=stolyarov_args, Td_fixed=Td_fixed, components=components)
		dAdBdi, dAdTdi, dAdBsi = dAdB_element_computation( freqs[ f ], Bd, Td, Bs, squeeze,\
			 bandpass_channels=bandpass_channels[f], stolyarov_args=stolyarov_args, Td_fixed=Td_fixed, components=components )
		if f == 0:
			A = Ai
			dAdBd = dAdBdi
			dAdTd = dAdTdi
			dAdBs = dAdBsi
		else:
			A = np.vstack(( A, Ai ))
			dAdBd = np.vstack(( dAdBd , dAdBdi ))
			dAdTd = np.vstack(( dAdTd , dAdTdi ))
			dAdBs = np.vstack(( dAdBs , dAdBsi ))

		del Ai, dAdBdi, dAdTdi, dAdBsi

	return A, dAdBd, dAdTd, dAdBs

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

def main_Cl_res_computation( sensitivity_uK_per_chan, freqs, spsp, sigma_omega, \
	Cl_cmb, Cl_dust, Cl_sync, Cl_dxs, ell_min, ell_max, \
	delta_Bd, delta_Bs, delta_Td, Q_dust, U_dust, Q_sync, U_sync, \
	Bd=Bd_fiducial, Td=Temp_fiducial, Bs=Bs_fiducial, \
	bandpass_channels=None, prior_Bd=0.0, prior_Td=0.0, prior_Bs=0.0, \
	components=['dust', 'sync'],\
	stolyarov_args=['Bd', 'Td', 'Bs'], Td_fixed=False, fsky=75, \
	err_Bd=None, err_Td=None, path_to_planck='../planck_maps/', calibration_error=0.0,\
	Cl_res_stolyarov_maps_option=False):

	nch = len( freqs )

	if ('Td' in stolyarov_args) and Td_fixed:
		print 'Td is in varying components and you did not put Td fixed option'
		exit()

	# Computation of the inverse covariance, in 1/uK**2 per frequency channel
	Ninv = np.zeros((2*nch, 2*nch))
	for k in range( 2*nch):
		Ninv[ k, k ]  = ((1.0/sensitivity_uK_per_chan[ int(k/2) ])**2)

	# A is a 6 x Nch matrix. 6 = 2x(CMB, Dust, Synchrotron)
	A, dAdBd, dAdTd, dAdBs = A_and_dAdB_matrix_computation( freqs, Bd, Td, Bs, bandpass_channels=bandpass_channels, stolyarov_args=stolyarov_args, Td_fixed=Td_fixed, components=components)

	# Omega is the identity matrix, joy of Fisher approximation
	Omega = omega_computation(nch)
	
	# A**T N**-1 ** A
	OmegaA = Omega.dot(A)
	AtNA =  OmegaA.T.dot(Ninv).dot(OmegaA)


	#print 'AtNA is ', AtNA
	# inversion of (A**T N**-1 ** A)
	try:
		AtNAinv = np.linalg.inv( AtNA )
	except np.linalg.linalg.LinAlgError:
		return Cl_dust*1e3, 1e3, np.array([[1e3,1e3,1e3],[1e3,1e3,1e3],[1e3,1e3,1e3]]), Cl_dust*1e3, Cl_dust*1e3


	###########
	# second derivative of the likelihood (d2LdBdB) computation
	# DUST x SYNC BLOCK
	OdAdBd = Omega.dot( dAdBd )
	OdAdTd = Omega.dot( dAdTd )
	OdAdBs = Omega.dot( dAdBs )

	#######################
	# filter out components which are not considered
	m00 = np.trace( - OdAdBd.T.dot( Ninv ).dot( OmegaA ).dot( AtNAinv ).dot( OmegaA.T ).dot( Ninv ).dot( OdAdBd ).dot( spsp ) + OdAdBd.T.dot( Ninv ).dot( OdAdBd ).dot( spsp )  )
	m02 = np.trace( - OdAdBd.T.dot( Ninv ).dot( OmegaA ).dot( AtNAinv ).dot( OmegaA.T ).dot( Ninv ).dot( OdAdBs ).dot( spsp ) + OdAdBd.T.dot( Ninv ).dot( OdAdBs ).dot( spsp )  )
	if not Td_fixed:
		m01 = np.trace( - OdAdBd.T.dot( Ninv ).dot( OmegaA ).dot( AtNAinv ).dot( OmegaA.T ).dot( Ninv ).dot( OdAdTd ).dot( spsp ) + OdAdBd.T.dot( Ninv ).dot( OdAdTd ).dot( spsp )  )
		m10 = np.trace( - OdAdTd.T.dot( Ninv ).dot( OmegaA ).dot( AtNAinv ).dot( OmegaA.T ).dot( Ninv ).dot( OdAdBd ).dot( spsp ) + OdAdTd.T.dot( Ninv ).dot( OdAdBd ).dot( spsp )  )
		m11 = np.trace( - OdAdTd.T.dot( Ninv ).dot( OmegaA ).dot( AtNAinv ).dot( OmegaA.T ).dot( Ninv ).dot( OdAdTd ).dot( spsp ) + OdAdTd.T.dot( Ninv ).dot( OdAdTd ).dot( spsp )  )
		m12 = np.trace( - OdAdTd.T.dot( Ninv ).dot( OmegaA ).dot( AtNAinv ).dot( OmegaA.T ).dot( Ninv ).dot( OdAdBs ).dot( spsp ) + OdAdTd.T.dot( Ninv ).dot( OdAdBs ).dot( spsp )  )
		m21 = np.trace( - OdAdBs.T.dot( Ninv ).dot( OmegaA ).dot( AtNAinv ).dot( OmegaA.T ).dot( Ninv ).dot( OdAdTd ).dot( spsp ) + OdAdBs.T.dot( Ninv ).dot( OdAdTd ).dot( spsp )  )
	m20 = np.trace( - OdAdBs.T.dot( Ninv ).dot( OmegaA ).dot( AtNAinv ).dot( OmegaA.T ).dot( Ninv ).dot( OdAdBd ).dot( spsp ) + OdAdBs.T.dot( Ninv ).dot( OdAdBd ).dot( spsp )  )
	m22 = np.trace( - OdAdBs.T.dot( Ninv ).dot( OmegaA ).dot( AtNAinv ).dot( OmegaA.T ).dot( Ninv ).dot( OdAdBs ).dot( spsp ) + OdAdBs.T.dot( Ninv ).dot( OdAdBs ).dot( spsp )  )

	if not Td_fixed:
		d2LdBdB_fg_block = np.array( [[m00, m01, m02],[m10, m11, m12],[m20, m21, m22]] )	
	else:
		d2LdBdB_fg_block = np.array( [[m00, m02],[m20, m22]] )

	################### CALIBRATION ERROR ??
	if calibration_error != 0.0:

		if not Td_fixed:
			print 'this is not coded yet i.e. there is no calibration error + varying Td option '
			exit()

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
		sigma_omega_v = np.ones(nch) / sigma_omega **2
		prior = np.diag(sigma_omega_v)
		for p in range( nch ) :
			dOmegadomega_p_A =  np.squeeze(dOmegadomega[p,:,:]).dot(A)
			for q in range( nch ):
				dOmegadomega_q_A =  np.squeeze(dOmegadomega[q,:,:]).dot(A)
				d2LdBdB_omega_omega[ p,q ] = np.trace(- dOmegadomega_p_A.T.dot( Ninv ).dot( OmegaA ).dot( AtNAinv ).dot( OmegaA.T ).dot(Ninv).dot( dOmegadomega_q_A ).dot( spsp ) \
					+ dOmegadomega_p_A.T.dot( Ninv ).dot( dOmegadomega_q_A ).dot( spsp ) + prior[ p,q ])

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

		fg_x_omega = np.vstack(( d2LdBdB_dust_omega, d2LdBdB_sync_omega))
		omega_x_fg = np.hstack((d2LdBdB_omega_dust,d2LdBdB_omega_sync))
		d2LdBdB = np.vstack(( np.hstack(( d2LdBdB_fg_block, fg_x_omega)),np.hstack(( omega_x_fg ,d2LdBdB_omega_omega))))
	
	else:
		########### END OF CALIBRATION ERROR
		d2LdBdB = d2LdBdB_fg_block


	if Td_fixed:
		if 'dust' in components:
			if 'sync' in components:
				d2LdBdB_out = d2LdBdB*1.0
			else:
				d2LdBdB_out = d2LdBdB[0,0]
		else:
			if 'sync' in components:
				d2LdBdB_out = d2LdBdB[1,1]		
			else:
				print 'components should contain at least 1 component in {dust,synchrotron}'
				exit()
	else:
		print 'not coded yet for varying Td'
		exit()

	if isinstance(d2LdBdB_out, np.ndarray):
		d2LdBdBinv = np.linalg.inv( d2LdBdB_out )
	else:
		d2LdBdBinv = 1.0/( d2LdBdB_out )

	#print ' is the following symmetric? d2LdBdB = ', d2LdBdB
	if isinstance(d2LdBdB_out, np.ndarray):
		print ' before adding any prior, diag( sqrt( d2LdBdB inv )) = ', np.sqrt(np.diag(d2LdBdBinv))
	else:
		print ' before adding any prior, diag( sqrt( d2LdBdB inv )) = ', np.sqrt( d2LdBdBinv )

	#### ADDING PRIORS
	if prior_Bd != 0.0:
		print 'adding prior on Bd'
		d2LdBdB_00 = 1.0/prior_Bd**2
	else:
		d2LdBdB_00 = 0.0
	
	if prior_Td != 0.0:
		print 'adding prior on Td'
		if Td_fixed:
			print 'no point of adding prior on a fixed Td ! '
		else:
			d2LdBdB_11 = 1.0/prior_Td**2
	else:
		d2LdBdB_11 = 0.0

	if prior_Bs != 0.0:
		print 'adding prior on Bs'
		d2LdBdB_22 = 1.0/prior_Bs**2
	else:
		d2LdBdB_22 = 0.0

	if Td_fixed:
		if 'dust' in components:
			if 'sync' in components:
				d2LdBdB_pior = np.diag([d2LdBdB_00, d2LdBdB_22])
			else:
				d2LdBdB_pior = d2LdBdB_00
		else:
			if 'sync' in components:
				d2LdBdB_pior = d2LdBdB_22
			else:
				print 'there should be dust or synchrotron in components'
				exit()
	else:
		if 'dust' in components:
			if 'sync' in components:
				d2LdBdB_pior = np.diag([d2LdBdB_00, d2LdBdB_11, d2LdBdB_22])
			else:
				d2LdBdB_pior = np.diag([d2LdBdB_00, d2LdBdB_11])
		else:
			if 'sync' in components:
				d2LdBdB_pior = d2LdBdB_22
			else:
				print 'there should be dust or synchrotron in components'
				exit()

	if calibration_error != 0.0:
		print d2LdBdB_pior
		d2LdBdB_pior_loc = np.zeros((nch+len(d2LdBdB_pior), nch+len(d2LdBdB_pior)))
		d2LdBdB_pior_loc[:len(d2LdBdB_pior),:len(d2LdBdB_pior)] = d2LdBdB_pior
		d2LdBdB_pior = d2LdBdB_pior_loc*1.0
		del d2LdBdB_pior_loc

	d2LdBdB_out += d2LdBdB_pior

	## computation of Sigma
	if isinstance(d2LdBdB_out, np.ndarray):
		d2LdBdBinv = np.linalg.inv( d2LdBdB_out )
	else:
		d2LdBdBinv = 1.0/( d2LdBdB_out )

	if isinstance(d2LdBdB_out, np.ndarray):
		print ' and **after** adding any prior, diag( sqrt( d2LdBdB inv )) = ', np.sqrt(np.diag(d2LdBdBinv))
	else:
		print ' and **after** adding any prior, diag( sqrt( d2LdBdB inv )) = ', np.sqrt( d2LdBdBinv )

	#######################
	## computation of alpha
	Ninv_sq = np.zeros((nch, nch))
	for k in range(nch):
		Ninv_sq[ k,k ] = ((1.0/sensitivity_uK_per_chan[ int(k) ])**2) 

	Omega_sq = omega_computation(nch, squeeze=True)
	dOmegadomega_sq = np.zeros((nch, nch, nch))
	A_sq, dAdBd_sq, dAdTd_sq, dAdBs_sq = A_and_dAdB_matrix_computation(freqs, Bd, Td, Bs, squeeze=True, bandpass_channels=bandpass_channels, stolyarov_args=stolyarov_args, Td_fixed=Td_fixed, components=components) 

	for p in range(nch):
		for k in range(nch):
			if (k == p):
				dOmegadomega_sq[p,k,k] = 1
			else:
				dOmegadomega_sq[p,k,k] = 0
	OmegaA_sq = Omega_sq.dot( A_sq )
	AtNA_sq = OmegaA_sq.T.dot( Ninv_sq ).dot( OmegaA_sq )
	AtNAinv_sq = np.linalg.inv( AtNA_sq )
	
	### TEST !! 
	if isinstance(d2LdBdB_out, np.ndarray):
		ncomp = len(d2LdBdBinv)
	else:
		ncomp = 1

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
			else:
				# Bs
				alpha[:,:,1] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdBs_sq)
		else:
			# Bd
			alpha[:,:,0] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdBd_sq)
			if not Td_fixed:
				# Td
				alpha[:,:,1] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdTd_sq)
	else:
		# Bd
		alpha[:,:,0] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdBs_sq)
		if not Td_fixed:
			# Td
			alpha[:,:,1] = - AtNAinv_sq.dot(OmegaA_sq.T).dot(Ninv_sq).dot(Omega_sq).dot(dAdTd_sq)

	#######################################################################################
	## computation of residuals map
	if Cl_res_stolyarov_maps_option:
		Q_residuals = Q_dust*0.0
		U_residuals = U_dust*0.0

		for p in range(len(Q_dust)):
			for k in range( ncomp ):
				for j in range(len(channels)):
					s = channels[j]
					if s=='dust':
						Q_loc = alpha[ 0, j, k ] * Q_dust[p] 
						U_loc = alpha[ 0, j, k ] * U_dust[p] 
					elif s=='delta_Bd':
						Q_loc = delta_Bd[p]*alpha[ 0, j, k ] * Q_dust[p]
						U_loc = delta_Bd[p]*alpha[ 0, j, k ] * Q_dust[p]
					elif s=='delta_Td':
						Q_loc = delta_Td[p]*alpha[ 0, j, k ] * Q_dust[p]
						U_loc = delta_Td[p]*alpha[ 0, j, k ] * Q_dust[p]
					elif s=='sync':
						Q_loc = alpha[ 0, j, k ] * Q_sync[p]
						U_loc = alpha[ 0, j, k ] * Q_sync[p]
					elif s=='delta_Bs':	
						Q_loc = delta_Bs[p]*alpha[ 0, j, k ] * Q_sync[p]					
						U_loc = delta_Bs[p]*alpha[ 0, j, k ] * U_sync[p]
					else: 
						Q_loc = 0.0
						U_loc = 0.0
						# print 'i don t get this channel : ', s				
					# Q_residuals[p] += np.sign(d2LdBdBinv[ k,k ])*np.sqrt( np.abs(d2LdBdBinv[ k,k ]) ) *  ( alpha[ 0, 1, k ] * Q_dust[p] + delta_Bd[p]*alpha[ 0, 2, k ] * Q_dust[p] + delta_Td[p]*alpha[ 0, 3, k ] * Q_dust[p]  +  alpha[ 0, 4, k ] * Q_sync[p]  +  delta_Bs[p]*alpha[ 0, 5, k ] * Q_sync[p]  )
					# Q_residuals[p] += np.sign(d2LdBdBinv[ k,k ])*np.sqrt( np.abs(d2LdBdBinv[ k,k ]) ) *  ( alpha[ 0, 1, k ] * Q_dust[p] + delta_Bd[p]*alpha[ 0, 2, k ] * Q_dust[p] + delta_Td[p]*alpha[ 0, 3, k ] * Q_dust[p]  +  alpha[ 0, 4, k ] * Q_sync[p]  +  delta_Bs[p]*alpha[ 0, 5, k ] * Q_sync[p]  )
					Q_residuals[p] += np.sign(d2LdBdBinv[ k,k ])*np.sqrt( np.abs(d2LdBdBinv[ k,k ]) ) *  Q_loc
					U_residuals[p] += np.sign(d2LdBdBinv[ k,k ])*np.sqrt( np.abs(d2LdBdBinv[ k,k ]) ) *  U_loc

			ind = p*100.0/len(Q_dust)
			sys.stdout.write("\r building the residuals maps >>>  %d %% " % ind)
	       	sys.stdout.flush()
		sys.stdout.write("\n")	

		#hp.write_map('LB_QQU_fgs_residuals', (Q_residuals,Q_residuals, U_residuals) )
		#hp.mollview(Q_residuals, title='Q residuals')
		#hp.mollview(U_residuals, title='U residuals')
		#hp.mollview(Q_dust, title='Q dust')
		#hp.mollview(Q_sync, title='Q sync')
		#hp.mollview( delta_Bd*Q_dust, title='delta Bd x Q dust')
		#hp.mollview( delta_Td*Q_dust, title='delta Td x Q dust')
		#hp.mollview( delta_Bs*Q_sync, title='delta Bs x Q sync')
		#pl.show()
		## raw power spectrum estimation from map
		print ' estimating power spectrum from the map '
		ell_loc,b, Cl_res, c,d,e = hp.sphtfunc.anafast( (Q_residuals, Q_residuals, U_residuals), iter=20, lmax=ell_max )
		# ell_loc,b, Cl_res, c,d,e = hp.sphtfunc.anafast( (Q1, Q1, U1), map2=(Q2, Q2, U2) , iter=10, lmax=ell_max )
		# pl.figure()
		# pl.title('residuals BB power spectrum estimated by healpy anafast')
		# pl.loglog( Cl_res, 'k', linewidth=3.0, alpha=0.8)
		# pl.xlabel('$\ell$', fontsize=16)
		# pl.ylabel('$D_\ell$', fontsize=16)
		# pl.show()
		# exit()
		sqrtAtNAinv00 = np.sqrt( AtNAinv[0,0] )
		if sqrtAtNAinv00 != sqrtAtNAinv00:
			sqrtAtNAinv00 = 1e3
		return Cl_res, sqrtAtNAinv00, np.sqrt( d2LdBdBinv ), Cl_res, Cl_res
	#######################################################################################

	AlphaAlpha_k1k2 = np.zeros(( len(channels), len(channels), ncomp, ncomp ))
	Cl_j1j2 = np.zeros(( len(channels), len(channels), ell_max-ell_min ))

	for j1 in range(len(channels)):
		s1 = channels[j1]
		if s1=='dust':
			Q1 = Q_dust
			U1 = U_dust
		elif s1=='delta_Bd':
			Q1 = Q_dust*delta_Bd
			U1 = U_dust*delta_Bd
		elif s1=='delta_Td':
			Q1 = Q_dust*delta_Td
			U1 = U_dust*delta_Td
		elif s1=='sync':
			Q1 = Q_sync
			U1 = U_sync
		elif s1=='delta_Bs':
			Q1 = Q_sync*delta_Bs
			U1 = U_sync*delta_Bs

		for j2 in range(len(channels)):
			s2 = channels[j2]
			if s2=='dust':
				Q2 = Q_dust
				U2 = U_dust
			elif s2=='delta_Bd':
				Q2 = Q_dust*delta_Bd
				U2 = U_dust*delta_Bd
			elif s2=='delta_Td':
				Q2 = Q_dust*delta_Td
				U2 = U_dust*delta_Td
			elif s2=='sync':
				Q2 = Q_sync
				U2 = U_sync
			elif s2=='delta_Bs':
				Q2 = Q_sync*delta_Bs
				U2 = U_sync*delta_Bs

			print ' ---------- '
			print 'estimating the cross spectrum: ', s1, ' x ', s2

			# does the power spectrum exist?
			# otherwise compute it
			if s1 =='cmb' and s2=='cmb':
				Cl_j1j2[j1,j2,:] = Cl_cmb[ell_min-2:ell_max-2]
			elif ( s1 =='cmb' and s2!='cmb' ) or ( s2 =='cmb' and s1!='cmb' ):
				Cl_j1j2[j1,j2,:] = Cl_cmb[ell_min-2:ell_max-2]*0.0
			elif s1 =='dust' and s2=='dust':
				print 'oh, this is dust x dust, I use the input Cl'
				Cl_j1j2[j1,j2,:] = Cl_dust[ell_min-2:ell_max-2]
			elif (s1=='dust' and s2=='sync') or (s2=='dust' and s1=='sync') :
				print 'oh, this is dust x sync, I use the input Cl'
				Cl_j1j2[j1,j2,:] = Cl_dxs[ell_min-2:ell_max-2]
			elif s1=='sync' and s2=='sync':
				print 'oh, this is sync x sync, I use the input Cl!'
				Cl_j1j2[j1,j2,:] = Cl_sync[ell_min-2:ell_max-2]
			else:
				if j2<j1:
					print 'j2<j1 ->> symmetry relation ! '
					Cl_j1j2[j1,j2,:] = Cl_j1j2[j2,j1,:]
				else:
					name = 'Cl_'+s1+'x'+s2+'_'+str(fsky)
					filename = os.path.join( path_to_planck, name+'.npy')
					if os.path.exists(filename):
						print 'loading already computed Cl ... '
						Cl_j1j2[j1,j2,:]  = np.load( filename )[:ell_max-ell_min]
					else:
						a,b, Cl_loc, c,d,e = hp.sphtfunc.anafast( (Q1, Q1, U1), map2=(Q2, Q2, U2) , iter=10, lmax=ell_max )
						Cl_j1j2[j1,j2,:] = Cl_loc[ell_min-2:ell_max-2]
						print 'saving Cl ...'
						np.save( filename,  Cl_loc[ell_min-2:ell_max-2] )

			# building the alpha alpha tensor
			for k1 in range(ncomp):
				for k2 in range(ncomp):
					AlphaAlpha_k1k2[j1,j2,k1,k2] = alpha[ 0,j1,k1 ]*alpha[ 0,j2,k2]


	##### compute error spectra
	if (err_Bd is not None) and ('Bd' in stolyarov_args):
		print 'computing Cl for the error map on beta dust'
		Q1 = err_Bd*Q_dust
		U1 = err_Bd*U_dust
		Q2 = err_Bd*Q_dust
		U2 = err_Bd*U_dust
		s1 = 'err_Bd'
		s2 = 'err_Bd'
		name = 'Cl_'+s1+'x'+s2+'_'+str(fsky)
		filename= os.path.join( path_to_planck, name+'.npy')
		if os.path.exists(filename):
			print 'loading already computed Cl ... '
			Cl_err_Bd  = np.load( filename )
		else:
			a,b, Cl_loc, c,d,e = hp.sphtfunc.anafast( (Q1, Q1, U1), map2=(Q2, Q2, U2) , iter=10 )
			Cl_err_Bd = Cl_loc[ell_min-2:ell_max-2]
			print 'saving Cl ...'
			np.save( filename,  Cl_loc[ell_min-2:ell_max-2] )
	if (err_Td is not None) and ('Td' in stolyarov_args):
		print 'computing Cl for the error map on dust temperature'
		Q1 = err_Td*Q_dust
		U1 = err_Td*U_dust
		Q2 = err_Td*Q_dust
		U2 = err_Td*U_dust
		s1 = 'err_Td'
		s2 = 'err_Td'
		name = 'Cl_'+s1+'x'+s2+'_'+str(fsky)
		filename = os.path.join( path_to_planck, name+'.npy')
		if os.path.exists(filename):
			print 'loading already computed Cl ... '
			Cl_err_Td  = np.load( filename )
		else:
			a,b, Cl_loc, c,d,e = hp.sphtfunc.anafast( (Q1, Q1, U1), map2=(Q2, Q2, U2) , iter=10 )
			Cl_err_Td = Cl_loc[ell_min-2:ell_max-2]
			print 'saving Cl ...'
			np.save( filename,  Cl_loc[ell_min-2:ell_max-2] )

	##  saving
	#np.save('AlphaAlpha_k1k2', AlphaAlpha_k1k2)
	#np.save('Cl_j1j2_it20', Cl_j1j2)
	#np.save('d2LdBdBinv', d2LdBdBinv)

	#pl.figure()
	#pl.loglog( Cl_err_Td )
	#pl.loglog( Cl_err_Bd )
	#for j1 in range(len(channels)):
	#	for j2 in range(len(channels)):
	#		pl.loglog( Cl_j1j2[j1,j2,:] )
	#pl.show()

	## power estimation from analytical formula
	Cl_res = np.zeros( ell_max-ell_min  )
	Cl_res_err_p = np.zeros( ell_max-ell_min  )
	Cl_res_err_m = np.zeros( ell_max-ell_min  )

	Cl_res_X = np.zeros((len(channels),len(channels), ell_max-ell_min ))

	for ell_ind in range( ell_max-ell_min ) :
		
		clkk = np.zeros((ncomp,ncomp))
		clkk_p = np.zeros((ncomp,ncomp))
		clkk_m = np.zeros((ncomp,ncomp))

		for k1 in range(ncomp):
			for k2 in range(ncomp):

				clkk_loc = 0.0
				clkk_p_loc = 0.0
				clkk_m_loc = 0.0

				for j1 in range(len(channels)):
					
					for j2 in range(len(channels)):
						
						if ncomp == 1:
							d2LdBdBinv_loc = d2LdBdBinv*1.0
						else:
							d2LdBdBinv_loc = d2LdBdBinv[ k1,k2 ]*1.0

						clkk_loc += d2LdBdBinv_loc * AlphaAlpha_k1k2[j1,j2,k1,k2] *  Cl_j1j2[ j1,j2,ell_ind ]

						#########################

						if (err_Bd is not None) and ('Bd' in stolyarov_args):
							if (channels[j1] == 'delta_Bd') and (channels[j2] == 'delta_Bd' ):
								clkk_p_loc += d2LdBdBinv_loc * AlphaAlpha_k1k2[j1,j2,k1,k2] *  ( Cl_j1j2[ j1,j2,ell_ind ] + Cl_err_Bd[ell_ind] )
								clkk_m_loc += d2LdBdBinv_loc * AlphaAlpha_k1k2[j1,j2,k1,k2] *  ( Cl_j1j2[ j1,j2,ell_ind ] - Cl_err_Bd[ell_ind] )
							elif (err_Td is not None) and ('Td' in stolyarov_args):
								if (channels[j1] == 'delta_Td') and (channels[j2] == 'delta_Td'):
									clkk_p_loc += d2LdBdBinv_loc * AlphaAlpha_k1k2[j1,j2,k1,k2] *  ( Cl_j1j2[ j1,j2,ell_ind ] + Cl_err_Td[ell_ind] )
									clkk_m_loc += d2LdBdBinv_loc * AlphaAlpha_k1k2[j1,j2,k1,k2] *  ( Cl_j1j2[ j1,j2,ell_ind ] - Cl_err_Td[ell_ind] )
								else:
									clkk_p_loc += d2LdBdBinv_loc * AlphaAlpha_k1k2[j1,j2,k1,k2] *   Cl_j1j2[ j1,j2,ell_ind ]
									clkk_m_loc += d2LdBdBinv_loc * AlphaAlpha_k1k2[j1,j2,k1,k2] *   Cl_j1j2[ j1,j2,ell_ind ]
							else:
								clkk_p_loc += d2LdBdBinv_loc * AlphaAlpha_k1k2[j1,j2,k1,k2] *   Cl_j1j2[ j1,j2,ell_ind ]
								clkk_m_loc += d2LdBdBinv_loc * AlphaAlpha_k1k2[j1,j2,k1,k2] *   Cl_j1j2[ j1,j2,ell_ind ]
						else:
							clkk_p_loc += d2LdBdBinv_loc * AlphaAlpha_k1k2[j1,j2,k1,k2] *   Cl_j1j2[ j1,j2,ell_ind ]
							clkk_m_loc += d2LdBdBinv_loc * AlphaAlpha_k1k2[j1,j2,k1,k2] *   Cl_j1j2[ j1,j2,ell_ind ]
						##############################

						Cl_res_X[j1,j2,ell_ind] += d2LdBdBinv_loc * AlphaAlpha_k1k2[j1,j2,k1,k2] *  Cl_j1j2[ j1,j2,ell_ind ]

				clkk[ k1,k2 ] = clkk_loc
				clkk_p[ k1,k2 ] = clkk_p_loc
				clkk_m[ k1,k2 ] = clkk_m_loc

		Cl_res[ell_ind] = np.sum(np.sum( clkk ))
		Cl_res_err_p[ell_ind] = np.sum(np.sum( clkk_p ))
		Cl_res_err_m[ell_ind] = np.sum(np.sum( clkk_m ))


	'''
	import matplotlib.cm as cm
	from matplotlib import rc
	c_all = cm.rainbow(np.linspace( 0, 1, int(len(channels)*(len(channels)+1)/2) ))
	pl.figure()
	pl.title('contribution of each term to the residuals BB power spectrum, for fsky='+str(fsky))
	ind = 0
	for j1 in range(len(channels)):
		for j2 in range(len(channels)):		
			if j2>=j1:#if ( j1>0 and j2>0 ) and (j1==j2):	
				print channels[j1], channels[j2], np.max(  np.abs(Cl_res_X[j1,j2,:]) ), np.mean(  Cl_res_X[j1,j2,:100] )
				pl.plot(  Cl_res_X[j1,j2,:] , color=c_all[ind], label=channels[j1]+'x'+channels[j2], alpha=0.8, linewidth=2.0)
				ind += 1
	pl.plot( Cl_res2, 'k', linewidth=3.0, alpha=0.8)
	pl.yscale('symlog')
	pl.xscale('log')
	pl.legend(loc='best')
	pl.xlabel('$\ell$', fontsize=16)
	pl.ylabel('$C_\ell$', fontsize=16)
	pl.show()
	'''
	'''
	#ells_loc = np.array(range(len(Cl_res)))
	#norm = ells_loc*(ells_loc+1)/(2*np.pi)
	pl.figure()
	pl.title('residuals BB power spectrum estimated by healpy anafast')
	#pl.loglog( Cl_res, 'k', linewidth=3.0, alpha=0.8)
	pl.loglog( Cl_res2, 'r', linewidth=3.0, alpha=0.8)
	#pl.loglog( Cl_res3, 'go', linewidth=3.0, alpha=0.8)
	pl.loglog( np.sum(np.sum(Cl_res_X,0),0))
	pl.xlabel('$\ell$', fontsize=16)
	pl.ylabel('$C_\ell$', fontsize=16)
	pl.show()
	'''


	'''
	################################################
	## Wiener filtering test
	npix = 0.7*12*128**2
	F = spsp / npix
	# F should be in CMB units .... 
	## cmb-cmb-dust-dust-dust-dust-sync-sync-sync-sync
	#F *= BB_factor_computation(150.0)**2
	Finv = np.linalg.inv( F ) #.dot(A.T.dot(A))
	noise_matrix = AtNA + Finv
	noise_matrix_inv = np.linalg.inv( noise_matrix )
	#Finv_noise_matrix_inv = Finv.dot( noise_matrix_inv )
	Finv_noise_matrix_inv = noise_matrix_inv
	#wiener_CMB_noise = np.sqrt( Finv_noise_matrix_inv[0,0] )
	#print wiener_CMB_noise, np.sqrt( AtNAinv[0,0] )
	#exit()
	AtNAinv = Finv_noise_matrix_inv*1.0
	print AtNAinv
	print '---'
	print Finv_noise_matrix_inv
	pl.matshow(AtNAinv)
	pl.colorbar()
	pl.matshow(Finv_noise_matrix_inv)
	pl.colorbar()
	pl.show()
	exit()
	'''

	sqrtAtNAinv00 = np.sqrt( AtNAinv[0,0] )
	if sqrtAtNAinv00 != sqrtAtNAinv00:
		sqrtAtNAinv00 = 1e3
	

	return Cl_res, sqrtAtNAinv00, np.sqrt( d2LdBdBinv ), Cl_res_err_p, Cl_res_err_m



	