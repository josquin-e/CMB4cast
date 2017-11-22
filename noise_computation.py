#!/usr/bin/env python

'''
Noise computation for a experimental design, 
given its NETs, associated beams, yields and time of integration.

Author: Josquin
Contact: josquin.errard@gmail.com
'''

import sys
import os
import argparse
import numpy as np

arcmin_to_radian = np.pi/(180.0*60.0)

def main():
	args = grabargs()
	NETs = args.NETs
	FWHMs = args.FWHMs
	time_of_integration = args.time_of_integration
	fsky = args.fsky
	Nbolos = args.Nbolos

	if len(Nbolos)>1 :	assert len(NETs) == len(FWHMs)

	if not args.Yield:
		Yield = 1.0
	else:
		Yield = args.Yield

	Cl_noise, ell = Core( Nbolos, NETs, FWHMs, time_of_integration, fsky, Yield )

def Core(Nbolos, NETs, FWHMs, time_of_integration, fsky, Yield):


	ell_max = 10000
	ell = range(0, ell_max)
	Cl_noise = np._zeros(len(ell))
	skyam = 4.0*np.pi*fsky/(arcmin_to_radian**2);
	EffectiveDetectorSeconds = time_of_integration * Yield
	Cl_noise_inv = 0.0

	if not isinstance(NETs, list):
		nch = 1
		sigma_b = (FWHMs*arcmin_to_radian)/(np.sqrt(8.0*np.log(2.0)))
		Bl_inv = np.zeros(ellmax)
		for l in range( ell_max ):
			Bl_inv[l] = np.exp( -((sigma_b**2)*l*(l+1))) 
		w_eff = (EffectiveDetectorSeconds*Nbolos)/(NETs**2 * skyam)
		print ' The effective sensitivity for this experiment is :', 1.0/np.sqrt(w_eff), ' uK.arcmin '
		Cl_noise_inv = w_eff*Bl_inv
	else:
		nch = len(NETs)
		w_eff = 0.0
		for ch in range( nch ):
			sigma_b_ch = (FWHMs[ch]*arcmin_to_radian)/(np.sqrt(8.0*np.log(2.0)))
			Bl_inv_ch = np.zeros(ell_max)
			for l in range( ell_max ):
				Bl_inv_ch[l] = np.exp( -((sigma_b_ch**2)*l*(l+1))) 
			w_ch = (EffectiveDetectorSeconds*Nbolos[ch])/(NETs[ch]**2 * skyam)
			w_eff += w_ch
			Cl_noise_inv += w_ch*Bl_inv_ch
			if ch ==nch-1:
				print ' The effective sensitivity for this multi-channel experiment is :', 1.0/np.sqrt(w_eff), ' uK.arcmin '

	good_ell = np.where(Cl_noise_inv != 0.0)[0]
	Cl_noise = np.zeros(ell_max)
	Cl_noise[good_ell] = 1.0/Cl_noise_inv[good_ell] # this is in (uK.arcmin)**2 because w_ch is in 1.0/(uK.arcmin)**2
	Cl_noise *= (arcmin_to_radian**2) # this is in (uK.rad)**2

	return Cl_noise, ell, 1.0/np.sqrt(w_eff)


def grabargs():
	parser = argparse.ArgumentParser(description='computation of Cl noise given instrumental parameters')
	parser.add_argument('--Nbolos', dest='Nbolos', action='store', type=int, nargs='+', help='number of bolos in each channel',required=True)
	parser.add_argument('--NETs', dest='NETs', action='store', type=float, nargs='+', help='NETs in uK.rs',required=True)
	parser.add_argument('--FWHMs', dest='FWHMs', action='store', type=float, nargs='+', help='FWHMs in arcmin',required=True)
	parser.add_argument('--time_of_integration', dest='time_of_integration', type=float, action='store', help='time of integration in sec',required=True)
	parser.add_argument('--fsky', dest='fsky', action='store', type=float, help='fraction of the sky between 0.0 and 1.0',required=True)
	parser.add_argument('--Yield', dest='Yield', action='store', type=float, help='Yield',required=False)

	args = parser.parse_args()

	return args

if __name__ == "__main__":

	main()