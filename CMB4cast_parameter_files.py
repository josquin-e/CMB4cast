'''
Parameter file for a run with CMB4cast
'''

#################
"""
FIDUCIAL COSMOLOGY
You write below the values of the cosmological parameters 
"""
params_fid = {}
params_fid['h'] = 67.74
params_fid['ombh2'] = 0.02230
params_fid['omch2'] = 0.1188
params_fid['omnuh2'] = 0.0006451439 # 1 massive neutrino, \Sigma M_\nu = 0.06 eV  
params_fid['omk'] = 0.0
params_fid['YHe'] = 0.2453 
params_fid['Neff'] = 3.046
params_fid['w'] = -1.0
params_fid['wa'] = 0.0
params_fid['tau'] = 0.066
params_fid['As'] = 2.142e-9
params_fid['ns'] = 0.9667
params_fid['alphas'] = 0.0
params_fid['r'] = 0.001
params_fid['nT'] = -params_fid['r']/8.0
params_fid['k_scalar'] = 0.05#05
params_fid['k_tensor'] = 0.002#05

#################
"""
COSMOLOGICAL PARAMETERS TO CONSIDER FOR CONSTRAINTS
You can choose here the list of cosmological parameters
that you want to marginalize over when estimating the 
Fisher matrix
"""
params_dev = ['ns', 'As', 'tau', 'h', 'ombh2', 'omch2', 'r']

#################
"""
PRIORS ON COSMOLOGICAL PARAMETERS
Below you define the 
"""
params_fid_prior = {}
params_fid_prior['h'] = 0.46

#################
"""
DEFINITION OF EXPERIMENTS
Fill in the specifications below
"""
frequencies = [60.0, 78.0, 100.0, 140.0, 195.0, 280.0]
uKCMBarcmin = [15.7, 9.9, 7.1, 5.6, 4.7, 5.7]
FWHM = [54.1, 55.5, 56.8, 40.5, 38.4, 37.7] 
fsky = 0.7
bandpass = 0.3*np.ones(len(frequencies))
ell_min = 2
ell_max = 1500
prior_dust = 2*0.02
prior_sync = 2*0.2
alpha_knee = 0.0*np.ones(len(frequencies))
ell_knee = 0.0*np.ones(len(frequencies))

####################
"""
DEFINE BELOW THE SKY TEMPLATES AND THEIR FREQUENCY DEPENDENCE
It is a dictionnary, each entry will correspond to a sky template
Note that Q/U for CMB, dust and synchrotron are available. Adding new 
templates (second dust, polarized AME, etc) will require you to build 
yourself the spsp and Cls quantities below 
"""
analytic_expr_per_template = OrderedDict([ \
     ('Qcmb','(nu / cst) ** 2 * ( exp ( nu / cst ) ) / ( ( exp ( nu / cst ) - 1 ) ** 2 )'),\
     ('Ucmb','(nu / cst) ** 2 * ( exp ( nu / cst ) ) / ( ( exp ( nu / cst ) - 1 ) ** 2 )'),\
     ('Qdust','( exp( nu_ref / (Td / h_over_k ) ) - 1 ) / ( exp( nu / ( Td / h_over_k ) ) - 1 ) * ( nu / nu_ref ) ** ( 1 + Bd + drun * log( nu / nu_ref ) )'),\
     ('Udust','( exp( nu_ref / (Td / h_over_k ) ) - 1 ) / ( exp( nu / ( Td / h_over_k ) ) - 1 ) * ( nu / nu_ref ) ** ( 1 + Bd + drun * log( nu / nu_ref ) )'),\
     # ('dBQdust','( exp( nu_ref / (Td / h_over_k ) ) - 1 ) / ( exp( nu / ( Td / h_over_k ) ) - 1 ) * log( nu / nu_ref ) * ( nu / nu_ref ) ** ( 1 + Bd + drun * log( nu / nu_ref ) )'),\
     # ('dBUdust','( exp( nu_ref / (Td / h_over_k ) ) - 1 ) / ( exp( nu / ( Td / h_over_k ) ) - 1 ) * log( nu / nu_ref ) * ( nu / nu_ref ) ** ( 1 + Bd + drun * log( nu / nu_ref ) )'),\
     ('Qsync','( nu / nu_ref ) ** (Bs + srun * log( nu / nu_ref ) )'),\
     ('Usync','( nu / nu_ref ) ** (Bs + srun * log( nu / nu_ref ) )') ])#,\
     # ('dBQsync','log( nu / nu_ref ) * ( nu / nu_ref ) ** (Bs + srun * log( nu / nu_ref ) )'),\
     # ('dBUsync','log( nu / nu_ref ) * ( nu / nu_ref ) ** (Bs + srun * log( nu / nu_ref ) )') ] )

####################
"""
SPECTRAL PARAMETERS
to be used in the equations above.
Don't touch h_over_k or cst.
"""
spectral_parameters = { 'nu_ref':150.0, \
						'Bd':1.59,\
					    'Td':19.6, \
					    'h_over_k':h_over_k, \
					    'drun':0.0, \
					    'Bs':-3.1, \
					    'srun':0.0,\
					    'cst':cst }

####################
"""
FREE SPECTRAL PARAMETERS
which should be fitted for during the parametric
component separation 
"""
drv = ['Bd', 'Bs']

####################
"""
PRIORS ON SPECTRAL PARAMETERS
to be added on the uncertainties estimated 
during the previous step. 
"""
prior_spectral_parameters = { 'Bd':0.0, \
								'Td':0.0, \
								'drun':0.0, \
								'Bs':0.0, \
								'srun':0.0 }

####################
"""
NUMBER OF INDEPENDENT PATCHES TO ANALYZE
The error bars on estimated spectral parameters
are boosted by sqrt(N) where N is the number of
independent patches to be analyzed.
In the example below, we consider a healpix pixelization
with nside=4. Take 0 or 1 to turn off spatial variability 
of spectral indices
"""
np_nside = 4

####################
"""
CMB "CHANNELS" TO BE USED AMONG TEMPERATURE, E-, B-MODES AND DEFLECTION D
Tu = unlensed temperature
default = ['Tu', 'Eu', 'Bu', 'd']
"""
information_channels= ['Tu', 'Eu', 'Bu', 'd']

####################
"""
DELENSING OPTION AMONG
''= NONE
'CMBxCMB' = standard internal EB iterative delensing
'CMBxCIB' = use of infrared background
'CMBxLSS' = use of LSS imaginary 
"""
delensing_option_v = ['','CMBxCMB','CMBxCIB', 'CMBxLSS']

####################
"""
path to the camb executable ./camb
"""
path_to_camb = 'path_to_the_camb_executable'

####################
"""
path to the spsp quantity that you can either build yourself
or download it at this link: 
"""
path_to_precomputed_spsp = 'path_to_spsp.pkl'
path_to_your_spsp = '' # this should be a 2d array of size n_component x n_component

####################
"""
path to the Cls (angular of foregrounds) that you can either build yourself
or download them at this link:
"""
path_to_precomputed_Cls = 'path_to_Cls.pkl' 
path_to_your_Cls = ''
