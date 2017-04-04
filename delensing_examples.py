import numpy as np
import matplotlib.pyplot as plt
import ctypes as ct


# load delensing C lib
dl = np.ctypeslib.load_library('libdelens', '.')
dl.delensing_performance.argtypes = [ct.c_int, ct.c_int, \
                                     ct.POINTER(ct.c_double), \
                                     ct.POINTER(ct.c_double), \
                                     ct.POINTER(ct.c_double), \
                                     ct.POINTER(ct.c_double), \
                                     ct.POINTER(ct.c_double), \
                                     ct.POINTER(ct.c_double), \
                                     ct.POINTER(ct.c_double), \
                                     ct.c_double, ct.c_bool, \
                                     ct.POINTER(ct.c_double), \
                                     ct.POINTER(ct.c_double)]
dl.delensing_performance_raw.argtypes = [ct.c_int, ct.c_int, \
                                         ct.POINTER(ct.c_double), \
                                         ct.POINTER(ct.c_double), \
                                         ct.POINTER(ct.c_double), \
                                         ct.POINTER(ct.c_double), \
                                         ct.POINTER(ct.c_double), \
                                         ct.POINTER(ct.c_double), \
                                         ct.c_int,
                                         ct.POINTER(ct.c_double), \
                                         ct.POINTER(ct.c_double), \
                                         ct.c_double, ct.c_bool, \
                                         ct.POINTER(ct.c_double), \
                                         ct.POINTER(ct.c_double)]
                                                    
# wrapper functions
def delens_est(l_min, l_max, c_l_ee_u, c_l_ee_l, c_l_bb_u, \
               c_l_bb_l, c_l_pp, f_l_cor, n_l_ee_bb, thresh, \
               no_iteration, n_l_pp, c_l_bb_res):
    return dl.delensing_performance(ct.c_int(l_min), ct.c_int(l_max), \
            c_l_ee_u.ctypes.data_as(ct.POINTER(ct.c_double)), \
            c_l_ee_l.ctypes.data_as(ct.POINTER(ct.c_double)), \
            c_l_bb_u.ctypes.data_as(ct.POINTER(ct.c_double)), \
            c_l_bb_l.ctypes.data_as(ct.POINTER(ct.c_double)), \
            c_l_pp.ctypes.data_as(ct.POINTER(ct.c_double)), \
            f_l_cor.ctypes.data_as(ct.POINTER(ct.c_double)),\
            n_l_ee_bb.ctypes.data_as(ct.POINTER(ct.c_double)), \
            ct.c_double(thresh), ct.c_bool(no_iteration), \
            n_l_pp.ctypes.data_as(ct.POINTER(ct.c_double)), \
            c_l_bb_res.ctypes.data_as(ct.POINTER(ct.c_double)))
def delens_est_raw(l_min, l_max, c_l_ee_u, c_l_ee_l, c_l_bb_u, \
                   c_l_bb_l, c_l_pp, f_l_cor, n_freq, sigma_pix_p, \
                   beam_fwhm, thresh, no_iteration, n_l_pp, \
                   c_l_bb_res):
    return dl.delensing_performance_raw(ct.c_int(l_min), ct.c_int(l_max), \
            c_l_ee_u.ctypes.data_as(ct.POINTER(ct.c_double)), \
            c_l_ee_l.ctypes.data_as(ct.POINTER(ct.c_double)), \
            c_l_bb_u.ctypes.data_as(ct.POINTER(ct.c_double)), \
            c_l_bb_l.ctypes.data_as(ct.POINTER(ct.c_double)), \
            c_l_pp.ctypes.data_as(ct.POINTER(ct.c_double)), \
            f_l_cor.ctypes.data_as(ct.POINTER(ct.c_double)),\
            ct.c_int(n_freq), \
            sigma_pix_p.ctypes.data_as(ct.POINTER(ct.c_double)), \
            beam_fwhm.ctypes.data_as(ct.POINTER(ct.c_double)), \
            ct.c_double(thresh), ct.c_bool(no_iteration), \
            n_l_pp.ctypes.data_as(ct.POINTER(ct.c_double)), \
            c_l_bb_res.ctypes.data_as(ct.POINTER(ct.c_double)))


# plotting stuff
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 1.5

            
# common parameters
l_min = 2
l_max = 4000
thresh = 0.01
no_iteration = False
f_cor = 0.0
cib_delens = False
if (cib_delens):
    # flatten() is needed below otherwise call to
    # ctypes.data_as(ct.POINTER(ct.c_double)) goes loopy and resulting
    # array is incorrect
    f_l_cor_raw = np.genfromtxt('f_l_cor_planck_545.dat')
    f_l_cor = f_l_cor_raw[0: l_max - l_min + 1, 1].flatten()
else:
    f_l_cor = np.ones(l_max - l_min + 1) * f_cor


# read and convert fiducial power spectra (assuming r = 0)
c_l_u=np.genfromtxt('fiducial_lenspotentialCls.dat')
c_l_l=np.genfromtxt('fiducial_lensedtotCls.dat') 
ell = c_l_u[0: l_max - l_min + 1, 0].flatten()
d_l_conv = 2.0 * np.pi / ell / (ell + 1.0)
c_l_ee_u = c_l_u[0: l_max - l_min + 1, 2].flatten() * d_l_conv
c_l_ee_l = c_l_l[0: l_max - l_min + 1, 2].flatten() * d_l_conv
c_l_bb_u = c_l_u[0: l_max - l_min + 1, 3].flatten() * d_l_conv
c_l_bb_l = c_l_l[0: l_max - l_min + 1, 3].flatten() * d_l_conv
c_l_pp = c_l_u[0: l_max - l_min + 1, 5].flatten() / \
         (ell * (ell + 1.0)) ** 2 * 2.0 * np.pi
line_0, = plt.loglog(ell, c_l_bb_l / d_l_conv)


# first try out the "raw" version, which takes basic instrumental
# characteristics. set up a four-frequency (EBEX6K-like) experiment.
f_sky = 0.01212
l_min_exp = int(np.ceil(2.0 * np.sqrt(np.pi / f_sky)))
spp_arr = np.array([0.36769553, 0.53740115, 1.1737973, 4.6951890])
beam_arr = np.array([11.0, 11.0, 11.0, 11.0])
n_l_pp = np.zeros(l_max - l_min + 1)
c_l_bb_res = np.zeros(l_max - l_min + 1)
delens_est_raw(l_min_exp, l_max, c_l_ee_u[l_min_exp - l_min:], \
               c_l_ee_l[l_min_exp - l_min:], \
               c_l_bb_u[l_min_exp - l_min:], \
               c_l_bb_l[l_min_exp - l_min:], \
               c_l_pp[l_min_exp - l_min:], \
               f_l_cor[l_min_exp - l_min:], len(spp_arr), \
               spp_arr, beam_arr, thresh, no_iteration, \
               n_l_pp[l_min_exp - l_min:], \
               c_l_bb_res[l_min_exp - l_min:])
line_1, = plt.loglog(ell[l_min_exp - l_min:], \
                     c_l_bb_res[l_min_exp - l_min:] / \
                     d_l_conv[l_min_exp - l_min:])
                     

# next demonstrate the simplest version, which assumes the user has 
# converted the experimental characteristics into noise power spectra
# and then combined all frequencies into a single effective noise. the
# experiment considered is the most sensitive one in Wu, Errard et al.
# if I remember correctly...
f_sky_new = 0.75
l_min_exp_new = int(np.ceil(2.0 * np.sqrt(np.pi / f_sky_new)))
spp = np.sqrt(2.0) * 0.58
beam = 1.0
beam =  beam / 60.0 / 180.0 * np.pi
beam_area = beam * beam
beam_theta = beam / np.sqrt(8.0 * np.log(2.0))
n_l_ee_bb = np.zeros(l_max - l_min + 1)
for i in range(0, l_max - l_min + 1):
	bl=np.exp(beam_theta * beam_theta * (l_min + i) * (l_min + i + 1))
	n_l_ee_bb[i] = (beam_area * spp * spp * bl)
n_l_pp = np.zeros(l_max - l_min + 1)
c_l_bb_res = np.zeros(l_max - l_min + 1)
delens_est(l_min_exp_new, l_max, c_l_ee_u[l_min_exp_new - l_min:], \
           c_l_ee_l[l_min_exp_new - l_min:], \
           c_l_bb_u[l_min_exp_new - l_min:], \
           c_l_bb_l[l_min_exp_new - l_min:], \
           c_l_pp[l_min_exp_new - l_min:], \
           f_l_cor[l_min_exp_new - l_min:], \
           n_l_ee_bb[l_min_exp_new - l_min:], \
           thresh, no_iteration, n_l_pp[l_min_exp_new - l_min:], \
           c_l_bb_res[l_min_exp_new - l_min:])
line_2, = plt.loglog(ell[l_min_exp_new - l_min:], \
                     c_l_bb_res[l_min_exp_new - l_min:] / \
                     d_l_conv[l_min_exp_new - l_min:])


# finish off plot
plt.xlabel(r'$\ell$', fontsize = 16)
plt.ylabel(r'${\mathcal D}_\ell^{BB}$', fontsize = 16)
plt.title(r'Delensing Residuals')
plt.legend([line_0, line_1, line_2], [r'input', r'EBEX6K', r'Wu et al.'], loc = 'lower right')
plt.show()
