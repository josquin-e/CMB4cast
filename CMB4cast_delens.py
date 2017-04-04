#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pl
import os
import subprocess as sp
import math
import copy

'''
CMB4cast delensing code
To compile CTYPES interface to c code, try
 > cc -O3 -std=c99 -fPIC -Wall -shared -o libdelens.so delensing_performance.c
To compile c command-line executable, try
 > cc -O3 -Wall -fPIC -o delens_est delensing_performance.c
To compile F2PY interface to Fortran90 code, try
 > f2py -c -m pydelens delensing_performance.f90
To compile F90 command-line executable, try
 > gfortran -O3 -m64 -DGFORTRAN -fno-second-underscore delensing_performance.f90 -o delens_est"
'''

def round_sig(x, sig=3):
	if x != 0.0 and x==x:
		a  = round(x, sig-int(math.floor(math.log10(x)))-1)
		return "%.2e" % ( a )
	else:
		return round( x )


# load appropriate modules depending on delensing source code
delensing_source = 'c'
if delensing_source == 'c':
    
    # load CTYPES-based C delensing lib and dependencies
    import ctypes as ct
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
                                                    
    # C wrapper functions
    def delens_est(l_min, l_max, c_l_ee_u, c_l_ee_l, c_l_bb_u, \
                   c_l_bb_l, c_l_pp, f_l_cor, n_l_ee_bb, thresh, \
                   no_iteration, n_l_pp, c_l_bb_res):

        print l_min, l_max, np.sum(c_l_ee_u), np.sum(c_l_ee_l), np.sum(c_l_bb_u), \
                   np.sum(c_l_bb_l), np.sum(c_l_pp), np.sum(f_l_cor), np.sum(n_l_ee_bb), thresh, \
                   no_iteration, np.sum(n_l_pp), np.sum(c_l_bb_res)
        print len(c_l_ee_u), len(c_l_ee_l), len(c_l_bb_u), \
                   len(c_l_bb_l), len(c_l_pp), len(f_l_cor), len(n_l_ee_bb), len(n_l_pp), len(c_l_bb_res)
                   
        return dl.delensing_performance(ct.c_int(l_min),
                ct.c_int(l_max), \
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
        return dl.delensing_performance_raw(ct.c_int(l_min),
                ct.c_int(l_max), \
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

    # wrapper function
    def delens(experiments, configurations, components_v, \
               delensing_option_v, Cls_fid, f_l_cor_cib, f_l_cor_lss, \
               Nl, converge = 0.01, no_iteration = False, \
               ell_min_camb = 2, cross_only = False):
        c_delens(experiments, configurations, components_v, \
                 delensing_option_v, Cls_fid, f_l_cor_cib, \
                 f_l_cor_lss, Nl, converge = converge, \
                 no_iteration = no_iteration, \
                 ell_min_camb = ell_min_camb, cross_only = cross_only)

elif delensing_source == 'f90':
    
    # load F2PY-compiled F90 delensing lib
    from pydelens import delens_tools as dl

    # wrapper function
    def delens(experiments, configurations, components_v, \
               delensing_option_v, Cls_fid, f_l_cor_cib, f_l_cor_lss, \
               Nl, converge = 0.01, no_iteration = False, \
               ell_min_camb = 2, cross_only = False):
        f2py_delens(experiments, configurations, components_v, \
                    delensing_option_v, Cls_fid, f_l_cor_cib, \
                    f_l_cor_lss, Nl, converge = converge, \
                    no_iteration = no_iteration, \
                    ell_min_camb = ell_min_camb, \
                    cross_only = cross_only)

    
######################################################################
            

def f_l_cor_setup(path_to_cls, lss = False, Cls_fid = {}, \
                  ell_min_camb = 2, ell_max_abs = 4000):

    '''
    @brief: read in correlation coefficients for CIB and LSS delensing
    '''
    
    # read in CIB correlation and linearly extrapolate to overall ell_max
    f_l_cor_cib_data = np.load(os.path.join(path_to_cls,'corr545.pkl'))
    f_l_cor_cib_data[0] = np.append(f_l_cor_cib_data[0], ell_max_abs)
    f_l_cor_cib_data[1] = np.append(f_l_cor_cib_data[1], \
                                    f_l_cor_cib_data[1][-1] * 2.0 - \
                                    f_l_cor_cib_data[1][-2])

    # fit low-ell portion of correlation with polynomial
    fit_order = 6
    fit_range = [52, 400]
    fit_l = np.arange(fit_range[0], fit_range[1])
    f_l_cor_cib_fit_par = np.polyfit(f_l_cor_cib_data[0][fit_l], \
                                     f_l_cor_cib_data[1][fit_l], \
                                     fit_order)
    f_l_cor_cib_fit = np.polyval(f_l_cor_cib_fit_par, f_l_cor_cib_data[0])

    # extrapolate by concatenating low-ell fit with measurements
    f_l_cor_cib = np.zeros(ell_max_abs - 1)
    f_l_cor_cib[:] = f_l_cor_cib_fit[ell_min_camb:]
    f_l_cor_cib[fit_range[0]-ell_min_camb:] = f_l_cor_cib_data[1][fit_range[0]:]

    # optionally zero out range of ells (e.g., Blake says to only use in range
    # 60 <= ell <= 1000)
    zero_range = [60, 2500]
    f_l_cor_cib[0:zero_range[0]-ell_min_camb] = 0.0
    f_l_cor_cib[zero_range[1]-ell_min_camb:] = 0.0
    
    # set up LSS correlation data for use in perfect LSS delensing
    # correlation coefficient is sqrt(C_l^phiphi,lo / C_l^phiphi)
    if lss:    
        f_l_cor_lss = np.sqrt(Cls_fid['ddzmax'] / Cls_fid['dd'])

    # plot, if you're into that kind of thing
    if False:
        line_data, = pl.semilogx(f_l_cor_cib_data[0], f_l_cor_cib_data[1])
        line_fit, = pl.semilogx(f_l_cor_cib_data[0], f_l_cor_cib_fit)
        line_out, = pl.semilogx(f_l_cor_cib_data[0][ell_min_camb:], f_l_cor_cib)
        if lss:
            line_perf, = pl.semilogx(Cls_fid['ell'], f_l_cor_lss)
        pl.ylim([0, 1])
        pl.ylabel('$ f_\ell^{\\rm cor}$', fontsize=14)
        pl.xlabel('$\ell$', fontsize=14)
        if lss:
            pl.legend([line_data, line_fit, line_out, line_perf], \
                      ['CIB input', 'CIB low-$\ell$ fit', \
                       'CIB output', 'perfect'], loc=3)
        else:
            pl.legend([line_data, line_fit, line_out], \
                      ['CIB input', 'CIB low-$\ell$ fit', \
                       'CIB output'], loc=3)
        pl.show()

    # return correlation coefficients
    if lss:
        return f_l_cor_cib, f_l_cor_lss
    else:
        return f_l_cor_cib


######################################################################


def c_delens(experiments, configurations, components_v, \
             delensing_option_v, Cls_fid, f_l_cor_cib, f_l_cor_lss, \
             Nl, converge = 0.01, no_iteration = False, \
             ell_min_camb = 2, cross_only = False):

    '''
    @brief: perform CTYPES-interfaced c delensing forecast
    '''

    # determine delensing performance
    Dl_conv = 2.0 * np.pi / Cls_fid['ell'] / (Cls_fid['ell'] + 1.0)
    ind = -1
    for exp1 in experiments:
        ind += 1
        for exp2 in experiments[ind:]:
            if exp1 == exp2 :
                # experiment alone
                exp = exp1
                if cross_only: continue
            else:
                # combination of two experiments
                exp = exp1+' x '+exp2
                
            i_min = configurations[exp]['ell_min'] - ell_min_camb
            i_max = configurations[exp]['ell_max'] - ell_min_camb + 1
            ell_count = configurations[exp]['ell_max'] - ell_min_camb + 1
                
            for components in components_v:
                
                if components == 'cmb-only':

                    ## WITHOUT FOREGROUNDS
                    # define a bunch of post-delensing arrays and their
                    # ranges. the delensing code outputs C_ls in the
                    # range [ell_min: ell_max + 1]; we want D_ls in the
                    # range [ell_min_camb: ell_max + 1]
                    # NB: delensing code outputs N_l^pp, so must
                    # convert to N_l^dd
                    Nl[exp][components]['dd'] = np.zeros(ell_count)
                    Nl[exp][components]['BB_delens'] = np.zeros(ell_count)
                    Nl[exp][components]['dd_CIB'] = np.zeros(ell_count)
                    Nl[exp][components]['BB_CIB_delens'] = np.zeros(ell_count)
                    Nl[exp][components]['dd_lss'] = np.zeros(ell_count)
                    Nl[exp][components]['BB_lss_delens'] = np.zeros(ell_count)

                    # call iterative CMB EB delensing code using raw noise
                    if 'CMBxCMB' in delensing_option_v:
        
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

         
                        
                        print configurations[exp]['ell_min'], \
                              configurations[exp]['ell_max'], \
                              np.sum(Cls_fid['EuEu'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max]),\
                              np.sum(Cls_fid['EE_tot'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max]),\
                              np.sum(Cls_fid['BuBu'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max]),\
                              np.sum(Cls_fid['BB_tot'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max]),\
                              np.sum(Cls_fid['dd'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max] ** 2 / 2.0 / np.pi), \
                              np.sum(np.zeros(i_max-i_min)), \
                              np.sum(Nl[exp][components]['BB'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max]), \
                              converge, no_iteration, \
                              np.sum(Nl[exp][components]['dd'][i_min:i_max]), \
                              np.sum(Nl[exp][components]['BB_delens'][i_min:i_max])
                        print len(Cls_fid['EuEu'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max]),\
                              len(Cls_fid['EE_tot'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max]),\
                              len(Cls_fid['BuBu'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max]),\
                              len(Cls_fid['BB_tot'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max]),\
                              len(Cls_fid['dd'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max] ** 2 / 2.0 / np.pi), \
                              len(np.zeros(i_max-i_min)), \
                              len(Nl[exp][components]['BB'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max]), \
                              len(Nl[exp][components]['dd'][i_min:i_max]), \
                              len(Nl[exp][components]['BB_delens'][i_min:i_max])

                        delens_est(configurations[exp]['ell_min'], \
                                   configurations[exp]['ell_max'], \
                                   Cls_fid['EuEu'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['EE_tot'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['BuBu'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['BB_tot'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['dd'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max] ** 2 / 2.0 / np.pi, \
                                   np.zeros(i_max-i_min), \
                                   Nl[exp][components]['BB'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max], \
                                   converge, no_iteration, \
                                   Nl[exp][components]['dd'][i_min:i_max], \
                                   Nl[exp][components]['BB_delens'][i_min:i_max])
                        Nl[exp][components]['BB_delens'][i_min:i_max] /= Dl_conv[i_min:i_max]
                        Nl[exp][components]['dd'][i_min:i_max] /= Dl_conv[i_min:i_max] ** 2 / (2.0 * np.pi)

                    # call CMB x CIB delensing code using raw noise
                    if 'CMBxCIB' in delensing_option_v:

                        delens_est(configurations[exp]['ell_min'], \
                                   configurations[exp]['ell_max'], \
                                   Cls_fid['EuEu'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['EE_tot'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['BuBu'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['BB_tot'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['dd'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max] ** 2 / 2.0 / np.pi, \
                                   f_l_cor_cib[i_min:i_max], \
                                   Nl[exp][components]['BB'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max], \
                                   converge, no_iteration, \
                                   Nl[exp][components]['dd_CIB'][i_min:i_max], \
                                   Nl[exp][components]['BB_CIB_delens'][i_min:i_max])
                        Nl[exp][components]['BB_CIB_delens'][i_min:i_max] /= Dl_conv[i_min:i_max]
                        Nl[exp][components]['dd_CIB'][i_min:i_max] /= Dl_conv[i_min:i_max] ** 2 / (2.0 * np.pi)

                    # call PERFECT delensing code using raw noise
                    if 'CMBxLSS' in delensing_option_v:

                        delens_est(configurations[exp]['ell_min'], \
                                   configurations[exp]['ell_max'], \
                                   Cls_fid['EuEu'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['EE_tot'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['BuBu'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['BB_tot'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['dd'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max] ** 2 / 2.0 / np.pi, \
                                   f_l_cor_lss[i_min:i_max], \
                                   Nl[exp][components]['BB'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max], \
                                   converge, no_iteration, \
                                   Nl[exp][components]['dd_lss'][i_min:i_max], \
                                   Nl[exp][components]['BB_lss_delens'][i_min:i_max])
                        Nl[exp][components]['BB_lss_delens'][i_min:i_max] /= Dl_conv[i_min:i_max]
                        Nl[exp][components]['dd_lss'][i_min:i_max] /= Dl_conv[i_min:i_max] ** 2 / (2.0 * np.pi)
                            
                    ####################################################
                    # calculate delensing factor
                    l0, l1 = 20, 200
                    i0, i1 = np.argmin(np.abs( Cls_fid['ell'] - l0 )), np.argmin(np.abs( Cls_fid['ell'] - l1 ))
                    if 'CMBxLSS' in delensing_option_v: Nl[exp][components]['alpha_CMBxLSS'] = np.sum(Nl[exp][components]['BB_lss_delens'][i0:i1]*Dl_conv[i0:i1]) / np.sum(Cls_fid['BlBl'][i0:i1]*Dl_conv[i0:i1])
                    if 'CMBxCIB' in delensing_option_v: Nl[exp][components]['alpha_CMBxCIB'] = np.sum(Nl[exp][components]['BB_CIB_delens'][i0:i1]*Dl_conv[i0:i1]) / np.sum(Cls_fid['BlBl'][i0:i1]*Dl_conv[i0:i1])
                    if 'CMBxCMB' in delensing_option_v: Nl[exp][components]['alpha_CMBxCMB'] = np.sum(Nl[exp][components]['BB_delens'][i0:i1]*Dl_conv[i0:i1]) / np.sum(Cls_fid['BlBl'][i0:i1]*Dl_conv[i0:i1])
                    ####################################################
                        
                else:

                    ## WITH FOREGROUNDS 
                    # define a bunch of post-delensing arrays and their
                    # ranges. the delensing code outputs C_ls in the
                    # range [ell_min: ell_max + 1]; we want D_ls in the
                    # range [ell_min_camb: ell_max + 1]
                    # NB: delensing code outputs N_l^pp, so must
                    # convert to N_l^dd
                    Nl[exp][components]['dd_post_comp_sep'] = np.zeros(ell_count)
                    Nl[exp][components]['BB_delens_post_comp_sep'] = np.zeros(ell_count)
                    Nl[exp][components]['dd_CIB_post_comp_sep'] = np.zeros(ell_count)
                    Nl[exp][components]['BB_CIB_delens_post_comp_sep'] = np.zeros(ell_count)
                    Nl[exp][components]['dd_lss_post_comp_sep'] = np.zeros(ell_count)
                    Nl[exp][components]['BB_lss_delens_post_comp_sep'] = np.zeros(ell_count)

                    # call iterative CMB EB delensing code using post-comp.sep. noise
                    if 'CMBxCMB' in delensing_option_v:

                        delens_est(configurations[exp]['ell_min'], \
                                   configurations[exp]['ell_max'], \
                                   Cls_fid['EuEu'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['EE_tot'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['BuBu'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['BB_tot'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['dd'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max] ** 2 / 2.0 / np.pi, \
                                   np.zeros(i_max-i_min), \
                                   Nl[exp][components]['BB_post_comp_sep'][i_min:i_max] *\
                                   Dl_conv[i_min:i_max], \
                                   converge, no_iteration, \
                                   Nl[exp][components]['dd_post_comp_sep'][i_min:i_max], \
                                   Nl[exp][components]['BB_delens_post_comp_sep'][i_min:i_max])
                        Nl[exp][components]['BB_delens_post_comp_sep'][i_min:i_max] /= \
                          Dl_conv[i_min:i_max]
                        Nl[exp][components]['dd_post_comp_sep'][i_min:i_max] /= \
                          Dl_conv[i_min:i_max] ** 2 / (2.0 * np.pi)

                    # call CMB x CIB delensing code using post-comp.sep. noise
                    if 'CMBxCIB' in delensing_option_v:

                        delens_est(configurations[exp]['ell_min'], \
                                   configurations[exp]['ell_max'], \
                                   Cls_fid['EuEu'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['EE_tot'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['BuBu'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['BB_tot'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['dd'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max] ** 2 / 2.0 / np.pi, \
                                   f_l_cor_cib[i_min:i_max], \
                                   Nl[exp][components]['BB_post_comp_sep'][i_min:i_max] *\
                                   Dl_conv[i_min:i_max], \
                                   converge, no_iteration, \
                                   Nl[exp][components]['dd_CIB_post_comp_sep'][i_min:i_max], \
                                   Nl[exp][components]['BB_CIB_delens_post_comp_sep'][i_min:i_max])
                        Nl[exp][components]['BB_CIB_delens_post_comp_sep'][i_min:i_max] /= \
                          Dl_conv[i_min:i_max]
                        Nl[exp][components]['dd_CIB_post_comp_sep'][i_min:i_max] /= \
                          Dl_conv[i_min:i_max] ** 2 / (2.0 * np.pi)

                    # call PERFECT delensing code using post-comp.sep. noise
                    if 'CMBxLSS' in delensing_option_v:

                        delens_est(configurations[exp]['ell_min'], \
                                   configurations[exp]['ell_max'], \
                                   Cls_fid['EuEu'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['EE_tot'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['BuBu'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['BB_tot'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max],\
                                   Cls_fid['dd'][i_min:i_max] * \
                                   Dl_conv[i_min:i_max] ** 2 / 2.0 / np.pi, \
                                   f_l_cor_lss[i_min:i_max], \
                                   Nl[exp][components]['BB_post_comp_sep'][i_min:i_max] *\
                                   Dl_conv[i_min:i_max], \
                                   converge, no_iteration, \
                                   Nl[exp][components]['dd_lss_post_comp_sep'][i_min:i_max], \
                                   Nl[exp][components]['BB_lss_delens_post_comp_sep'][i_min:i_max])
                        Nl[exp][components]['BB_lss_delens_post_comp_sep'][i_min:i_max] /= \
                          Dl_conv[i_min:i_max]
                        Nl[exp][components]['dd_lss_post_comp_sep'][i_min:i_max] /= \
                          Dl_conv[i_min:i_max] ** 2 / (2.0 * np.pi)
                        
                    ####################################################
                    # calculate delensing factor
                    l0, l1 = 20, 200
                    i0, i1 = np.argmin(np.abs( Cls_fid['ell'] - l0 )), \
                             np.argmin(np.abs( Cls_fid['ell'] - l1 ))
                    if 'CMBxLSS' in delensing_option_v: Nl[exp][components]['alpha_CMBxLSS_post_comp_sep'] = np.sum(Nl[exp][components]['BB_lss_delens_post_comp_sep'][i0:i1]*Dl_conv[i0:i1]) / np.sum(Cls_fid['BlBl'][i0:i1]*Dl_conv[i0:i1])
                    if 'CMBxCIB' in delensing_option_v: Nl[exp][components]['alpha_CMBxCIB_post_comp_sep'] = np.sum(Nl[exp][components]['BB_CIB_delens_post_comp_sep'][i0:i1]*Dl_conv[i0:i1]) / np.sum(Cls_fid['BlBl'][i0:i1]*Dl_conv[i0:i1])
                    if 'CMBxCMB' in delensing_option_v: Nl[exp][components]['alpha_CMBxCMB_post_comp_sep'] = np.sum(Nl[exp][components]['BB_delens_post_comp_sep'][i0:i1]*Dl_conv[i0:i1]) / np.sum(Cls_fid['BlBl'][i0:i1]*Dl_conv[i0:i1])


######################################################################
                        

def f2py_delens(experiments, configurations, components_v, \
                delensing_option_v, Cls_fid, f_l_cor_cib, \
                f_l_cor_lss, Nl, converge = 0.01, \
                no_iteration = False, ell_min_camb = 2, \
                cross_only = False):

    '''
    @brief: perform F2PY compiled F90 delensing forecast
    '''

    # determine delensing performance
    Dl_conv = 2.0 * np.pi / Cls_fid['ell'] / (Cls_fid['ell'] + 1.0)
    ind = -1
    for exp1 in experiments:
        ind += 1
        for exp2 in experiments[ind:]:
            if exp1 == exp2 :
                # experiment alone
                exp = exp1
                if cross_only: continue
            else:
                # combination of two experiments
                exp = exp1+' x '+exp2
                
            i_min = configurations[exp]['ell_min'] - ell_min_camb
            i_max = configurations[exp]['ell_max'] - ell_min_camb + 1
            ell_count = configurations[exp]['ell_max'] - ell_min_camb + 1
                
            for components in components_v:
                
                if components == 'cmb-only':

                    ## WITHOUT FOREGROUNDS
                    # define a bunch of post-delensing arrays and their
                    # ranges. the delensing code outputs C_ls in the
                    # range [ell_min: ell_max + 1]; we want D_ls in the
                    # range [ell_min_camb: ell_max + 1]
                    # NB: delensing code outputs N_l^pp, so must
                    # convert to N_l^dd
                    Nl[exp][components]['dd'] = np.zeros(ell_count)
                    Nl[exp][components]['BB_delens'] = np.zeros(ell_count)
                    Nl[exp][components]['dd_CIB'] = np.zeros(ell_count)
                    Nl[exp][components]['BB_CIB_delens'] = np.zeros(ell_count)
                    Nl[exp][components]['dd_lss'] = np.zeros(ell_count)
                    Nl[exp][components]['BB_lss_delens'] = np.zeros(ell_count)

                    # call iterative CMB EB delensing code using raw noise
                    if 'CMBxCMB' in delensing_option_v:
                        Nl[exp][components]['dd'][i_min:i_max], Nl[exp][components]['BB_delens'][i_min:i_max] = dl.delensing_performance(configurations[exp]['ell_min'], configurations[exp]['ell_max'], Cls_fid['EuEu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['EE_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BuBu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BB_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['dd'][i_min:i_max] * Dl_conv[i_min:i_max] ** 2 / 2.0 / np.pi, np.zeros(i_max-i_min), Nl[exp][components]['BB'][i_min:i_max] * Dl_conv[i_min:i_max], converge, False)
                        Nl[exp][components]['BB_delens'][i_min:i_max] /= Dl_conv[i_min:i_max]
                        Nl[exp][components]['dd'][i_min:i_max] /= Dl_conv[i_min:i_max] ** 2 / (2.0 * np.pi)

                    # call CMB x CIB delensing code using raw noise
                    if 'CMBxCIB' in delensing_option_v:
                        Nl[exp][components]['dd_CIB'][i_min:i_max], Nl[exp][components]['BB_CIB_delens'][i_min:i_max] = dl.delensing_performance(configurations[exp]['ell_min'], configurations[exp]['ell_max'], Cls_fid['EuEu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['EE_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BuBu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BB_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['dd'][i_min:i_max] * Dl_conv[i_min:i_max] ** 2 / 2.0 / np.pi, f_l_cor_cib[i_min:i_max], Nl[exp][components]['BB'][i_min:i_max] * Dl_conv[i_min:i_max], converge, False)
                        Nl[exp][components]['BB_CIB_delens'][i_min:i_max] /= Dl_conv[i_min:i_max]
                        Nl[exp][components]['dd_CIB'][i_min:i_max] /= Dl_conv[i_min:i_max] ** 2 / (2.0 * np.pi)

                    # call PERFECT delensing code using raw noise
                    if 'CMBxLSS' in delensing_option_v:
                        Nl[exp][components]['dd_lss'][i_min:i_max], Nl[exp][components]['BB_lss_delens'][i_min:i_max] = dl.delensing_performance(configurations[exp]['ell_min'], configurations[exp]['ell_max'], Cls_fid['EuEu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['EE_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BuBu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BB_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['dd'][i_min:i_max] * Dl_conv[i_min:i_max] ** 2 / 2.0 / np.pi, f_l_cor_lss[i_min:i_max], Nl[exp][components]['BB'][i_min:i_max] * Dl_conv[i_min:i_max], converge, False)
                        Nl[exp][components]['BB_lss_delens'][i_min:i_max] /= Dl_conv[i_min:i_max]
                        Nl[exp][components]['dd_lss'][i_min:i_max] /= Dl_conv[i_min:i_max] ** 2 / (2.0 * np.pi)
                            
                    ####################################################
                    # calculate delensing factor
                    l0, l1 = 20, 200
                    i0, i1 = np.argmin(np.abs( Cls_fid['ell'] - l0 )), np.argmin(np.abs( Cls_fid['ell'] - l1 ))
                    if 'CMBxLSS' in delensing_option_v: Nl[exp][components]['alpha_CMBxLSS'] = np.sum(Nl[exp][components]['BB_lss_delens'][i0:i1]*Dl_conv[i0:i1]) / np.sum(Cls_fid['BlBl'][i0:i1]*Dl_conv[i0:i1])
                    if 'CMBxCIB' in delensing_option_v: Nl[exp][components]['alpha_CMBxCIB'] = np.sum(Nl[exp][components]['BB_CIB_delens'][i0:i1]*Dl_conv[i0:i1]) / np.sum(Cls_fid['BlBl'][i0:i1]*Dl_conv[i0:i1])
                    if 'CMBxCMB' in delensing_option_v: Nl[exp][components]['alpha_CMBxCMB'] = np.sum(Nl[exp][components]['BB_delens'][i0:i1]*Dl_conv[i0:i1]) / np.sum(Cls_fid['BlBl'][i0:i1]*Dl_conv[i0:i1])
                    ####################################################
                        
                else:

                    ## WITH FOREGROUNDS 
                    # define a bunch of post-delensing arrays and their
                    # ranges. the delensing code outputs C_ls in the
                    # range [ell_min: ell_max + 1]; we want D_ls in the
                    # range [ell_min_camb: ell_max + 1]
                    # NB: delensing code outputs N_l^pp, so must
                    # convert to N_l^dd
                    Nl[exp][components]['dd_post_comp_sep'] = np.zeros(ell_count)
                    Nl[exp][components]['BB_delens_post_comp_sep'] = np.zeros(ell_count)
                    Nl[exp][components]['dd_CIB_post_comp_sep'] = np.zeros(ell_count)
                    Nl[exp][components]['BB_CIB_delens_post_comp_sep'] = np.zeros(ell_count)
                    Nl[exp][components]['dd_lss_post_comp_sep'] = np.zeros(ell_count)
                    Nl[exp][components]['BB_lss_delens_post_comp_sep'] = np.zeros(ell_count)

                    # call iterative CMB EB delensing code using post-comp.sep. noise
                    if 'CMBxCMB' in delensing_option_v:
                        Nl[exp][components]['dd_post_comp_sep'][i_min:i_max], Nl[exp][components]['BB_delens_post_comp_sep'][i_min:i_max] = dl.delensing_performance(configurations[exp]['ell_min'], configurations[exp]['ell_max'], Cls_fid['EuEu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['EE_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BuBu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BB_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['dd'][i_min:i_max] * Dl_conv[i_min:i_max] ** 2 / 2.0 / np.pi, np.zeros(i_max-i_min), Nl[exp][components]['BB_post_comp_sep'][i_min:i_max] * Dl_conv[i_min:i_max], converge, False)
                        Nl[exp][components]['BB_delens_post_comp_sep'][i_min:i_max] /= \
                          Dl_conv[i_min:i_max]
                        Nl[exp][components]['dd_post_comp_sep'][i_min:i_max] /= \
                          Dl_conv[i_min:i_max] ** 2 / (2.0 * np.pi)

                    # call CMB x CIB delensing code using post-comp.sep. noise
                    if 'CMBxCIB' in delensing_option_v:
                        Nl[exp][components]['dd_CIB_post_comp_sep'][i_min:i_max], Nl[exp][components]['BB_CIB_delens_post_comp_sep'][i_min:i_max] = dl.delensing_performance(configurations[exp]['ell_min'], configurations[exp]['ell_max'], Cls_fid['EuEu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['EE_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BuBu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BB_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['dd'][i_min:i_max] * Dl_conv[i_min:i_max] ** 2 / 2.0 / np.pi, f_l_cor_cib[i_min:i_max], Nl[exp][components]['BB_post_comp_sep'][i_min:i_max] * Dl_conv[i_min:i_max], converge, False)
                        Nl[exp][components]['BB_CIB_delens_post_comp_sep'][i_min:i_max] /= \
                          Dl_conv[i_min:i_max]
                        Nl[exp][components]['dd_CIB_post_comp_sep'][i_min:i_max] /= \
                          Dl_conv[i_min:i_max] ** 2 / (2.0 * np.pi)

                    # call PERFECT delensing code using post-comp.sep. noise
                    if 'CMBxLSS' in delensing_option_v:
                        Nl[exp][components]['dd_lss_post_comp_sep'][i_min:i_max], Nl[exp][components]['BB_lss_delens_post_comp_sep'][i_min:i_max] = dl.delensing_performance(configurations[exp]['ell_min'], configurations[exp]['ell_max'], Cls_fid['EuEu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['EE_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BuBu'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['BB_tot'][i_min:i_max] * Dl_conv[i_min:i_max], Cls_fid['dd'][i_min:i_max] * Dl_conv[i_min:i_max] ** 2 / 2.0 / np.pi, f_l_cor_lss[i_min:i_max], Nl[exp][components]['BB_post_comp_sep'][i_min:i_max] * Dl_conv[i_min:i_max], converge, False)
                        Nl[exp][components]['BB_lss_delens_post_comp_sep'][i_min:i_max] /= \
                          Dl_conv[i_min:i_max]
                        Nl[exp][components]['dd_lss_post_comp_sep'][i_min:i_max] /= \
                          Dl_conv[i_min:i_max] ** 2 / (2.0 * np.pi)
                        
                    ####################################################
                    # calculate delensing factor
                    l0, l1 = 20, 200
                    i0, i1 = np.argmin(np.abs( Cls_fid['ell'] - l0 )), \
                             np.argmin(np.abs( Cls_fid['ell'] - l1 ))
                    if 'CMBxLSS' in delensing_option_v: Nl[exp][components]['alpha_CMBxLSS_post_comp_sep'] = np.sum(Nl[exp][components]['BB_lss_delens_post_comp_sep'][i0:i1]*Dl_conv[i0:i1]) / np.sum(Cls_fid['BlBl'][i0:i1]*Dl_conv[i0:i1])
                    if 'CMBxCIB' in delensing_option_v: Nl[exp][components]['alpha_CMBxCIB_post_comp_sep'] = np.sum(Nl[exp][components]['BB_CIB_delens_post_comp_sep'][i0:i1]*Dl_conv[i0:i1]) / np.sum(Cls_fid['BlBl'][i0:i1]*Dl_conv[i0:i1])
                    if 'CMBxCMB' in delensing_option_v: Nl[exp][components]['alpha_CMBxCMB_post_comp_sep'] = np.sum(Nl[exp][components]['BB_delens_post_comp_sep'][i0:i1]*Dl_conv[i0:i1]) / np.sum(Cls_fid['BlBl'][i0:i1]*Dl_conv[i0:i1])

                        
######################################################################


def cmd_delens(experiments, configurations, components_v, \
               delensing_option_v, Cls_fid, f_l_cor_cib, \
               f_l_cor_lss, Nl, converge = 0.01, \
               no_iteration = False, ell_min_camb = 2, \
               ell_max_abs = 4000, cross_only = False, \
               mpi_safe = False):

    '''
    @brief: perform command-line delensing forecasting
    '''
    
    # useful arrays
    print ' ........... COMMAND LINE DELENSING ............ '
    Dl_conv = 2.0 * np.pi / Cls_fid['ell'] / (Cls_fid['ell'] + 1.0)

    # EDISON ONLY: point to the correct delens_est code but save in cwd ($SCRATCH)
    # delexe = '/global/homes/s/sfeeney/Software_Packages/cmb_pol_forecast/delens_est'
    delexe='./delens_est'
    
    # optionally generate a random string identifier to allow concurrent runs
    prefix = 'temp_delens'
    if mpi_safe:
        prefix += '_' + ''.join(random.choice(string.ascii_uppercase + \
                                              string.digits) for _ in range(10))

    # output to file for passing to delensing code. define array from ell_min_camb
    ells_out = ell_min_camb + np.arange(ell_max_abs + 1 - ell_min_camb)
    output = np.column_stack((ells_out.flatten(), f_l_cor_cib.flatten()))
    np.savetxt(prefix + '_f_l_cor_planck_545.dat', output, \
               fmt = ' %5.1i   %12.5e')
    
    # output to file for passing to delensing code. define array from ell_min_camb
    output = np.column_stack((ells_out.flatten(), f_l_cor_lss.flatten()))
    np.savetxt(prefix + '_f_l_cor_perfect.dat', output, \
               fmt = ' %5.1i   %12.5e')

    # using command-line delensing code which needs fiducial C_ls written to file.
    # could pickle random string in with C_ls in order to pass them directly; for
    # now, resave them
    n_ell_camb = len(Cls_fid['TT_tot'])
    output = np.column_stack((ell_min_camb + np.arange(n_ell_camb).flatten(), \
                              Cls_fid['TT_tot'].flatten(), \
                              Cls_fid['EE_tot'].flatten(), \
                              Cls_fid['BB_tot'].flatten(), \
                              Cls_fid['TE_tot'].flatten()))
    np.savetxt(prefix + '_fidCls_lensedtotCls.dat', output, \
               fmt = ' %5.1i   %12.5e   %12.5e   %12.5e   %12.5e')
    output = np.column_stack((ell_min_camb + np.arange(n_ell_camb).flatten(), \
                              Cls_fid['TuTu'].flatten(), \
                              Cls_fid['EuEu'].flatten(), \
                              Cls_fid['BuBu'].flatten(), \
                              Cls_fid['TuEu'].flatten(), \
                              Cls_fid['dd'].flatten(), \
                              Cls_fid['Tud'].flatten(), \
                              np.zeros(n_ell_camb).flatten()))
    np.savetxt(prefix + '_fidCls_lenspotentialCls.dat', output, \
               fmt = ' %5.1i   %12.5e   %12.5e   %12.5e   %12.5e' + \
                     '   %12.5e   %12.5e   %12.5e')

    # determine delensing performance
    ind = -1
    for exp1 in experiments:
        ind += 1
        for exp2 in experiments[ind:]:
            if exp1 == exp2 :
                # experiment alone
                exp = exp1
                if cross_only: continue
            else:
                # combination of two experiments
                exp = exp1+' x '+exp2
            
            i_min = configurations[exp]['ell_min'] - ell_min_camb
            i_max = configurations[exp]['ell_max'] - ell_min_camb + 1
            ell_count = configurations[exp]['ell_max'] - ell_min_camb + 1
            ells = np.array(range(ell_min_camb, configurations[exp]['ell_max']+1))

            for components in components_v:

                if components == 'cmb-only':

                    ## WITHOUT FOREGROUNDS
                    # define a bunch of post-delensing arrays and their
                    # ranges. the delensing code outputs C_ls in the
                    # range [ell_min: ell_max + 1]; we want D_ls in the
                    # range [ell_min_camb: ell_max + 1]
                    # NB: delensing code outputs N_l^pp, so must
                    # convert to N_l^dd
                    Nl[exp][components]['dd'] = np.zeros(ell_count)
                    Nl[exp][components]['BB_delens'] = np.zeros(ell_count)
                    Nl[exp][components]['dd_CIB'] = np.zeros(ell_count)
                    Nl[exp][components]['BB_CIB_delens'] = np.zeros(ell_count)
                    Nl[exp][components]['dd_lss'] = np.zeros(ell_count)
                    Nl[exp][components]['BB_lss_delens'] = np.zeros(ell_count)
                    
                    # write BB noise to file to pass to command-line delensing F90 code
                    exp_prefix = prefix + '_' + str(exp).replace(' ', '_')
                    output = np.column_stack((ells.flatten(), \
                                              Nl[exp][components]['BB'].flatten()))
                    np.savetxt(exp_prefix + '_n_l_ee_bb.dat', output, \
                               fmt = ' %5.1i   %12.5e')
                        
                    # call iterative CMB EB delensing code using raw noise
                    if 'CMBxCMB' in delensing_option_v:
                        com = [delexe, '-f_sky', str(configurations[exp]['fsky']), \
                                       '-l_min', str(configurations[exp]['ell_min']), \
                                       '-l_max', str(configurations[exp]['ell_max']), \
                                       '-lensed_path', prefix + '_fidCls_lensedtotCls.dat', \
                                       '-unlensed_path', prefix + '_fidCls_lenspotentialCls.dat', \
                                       '-noise_path', exp_prefix + '_n_l_ee_bb.dat', \
                                       '-prefix', exp_prefix]
                        try:
                            # sp.check_call(com)
                            print 'BLABLABLABLA'
                            os.system("./delens_est -f_sky " + str(configurations[exp]['fsky']) + \
                                " -l_min " + str(configurations[exp]['ell_min']) + \
                                " -l_max " + str(configurations[exp]['ell_max']) + \
                                " -lensed_path "+prefix+"_fidCls_lensedtotCls.dat" + \
                                " -unlensed_path "+prefix+"_fidCls_lenspotentialCls.dat" + \
                                " -noise_path "+exp_prefix+"_n_l_ee_bb.dat"+\
                                " -prefix "+exp_prefix )
                        except:
                            print('external_program failed for CMBxCMB w/o foregrounds')

                        data = np.genfromtxt(exp_prefix + '_c_l_bb_res.dat')
                        Nl[exp][components]['BB_delens'][i_min:i_max] = data[:, 1]*1.0 / Dl_conv[i_min:i_max]
                        data = np.genfromtxt(exp_prefix + '_n_l_dd.dat')
                        Nl[exp][components]['dd'][i_min:i_max] = data[:, 1]*1.0 / Dl_conv[i_min:i_max]

                    # call CMB x CIB delensing code using raw noise
                    if 'CMBxCIB' in delensing_option_v:
                        com = [delexe, '-f_sky', str(configurations[exp]['fsky']), \
                                       '-l_min', str(configurations[exp]['ell_min']), \
                                       '-l_max', str(configurations[exp]['ell_max']), \
                                       '-lensed_path', prefix + '_fidCls_lensedtotCls.dat', \
                                       '-unlensed_path', prefix + '_fidCls_lenspotentialCls.dat', \
                                       '-noise_path', exp_prefix + '_n_l_ee_bb.dat', \
                                       '-f_l_cor_path', prefix + '_f_l_cor_planck_545.dat', \
                                       '-prefix', exp_prefix]
                        try:
                            sp.check_call(com)
                        except:
                            print('external_program failed for CMBxCIB w/o foregrounds')
                        data = np.genfromtxt(exp_prefix + '_c_l_bb_res.dat')
                        Nl[exp][components]['BB_CIB_delens'][i_min:i_max] = data[:,1]*1.0 / Dl_conv[i_min:i_max]
                        data = np.genfromtxt(exp_prefix + '_n_l_dd.dat')
                        Nl[exp][components]['dd_CIB'][i_min:i_max] = data[:,1]*1.0 / Dl_conv[i_min:i_max]
                            
                    # call PERFECT delensing code using raw noise
                    if 'CMBxLSS' in delensing_option_v:
                        com = [delexe, '-f_sky', str(configurations[exp]['fsky']), \
                                       '-l_min', str(configurations[exp]['ell_min']), \
                                       '-l_max', str(configurations[exp]['ell_max']), \
                                       '-lensed_path', prefix + '_fidCls_lensedtotCls.dat', \
                                       '-unlensed_path', prefix + '_fidCls_lenspotentialCls.dat', \
                                       '-noise_path', exp_prefix + '_n_l_ee_bb.dat', \
                                       '-f_l_cor_path', prefix + '_f_l_cor_perfect.dat', \
                                       '-prefix', exp_prefix]
                        try:
                            sp.check_call(com)
                        except:
                            print('external_program failed for CMBxLSS w/o foregrounds')
                        data = np.genfromtxt(exp_prefix + '_c_l_bb_res.dat')
                        Nl[exp][components]['BB_lss_delens'][i_min:i_max] = data[:, 1]*1.0 / Dl_conv[i_min:i_max]
                        data = np.genfromtxt(exp_prefix + '_n_l_dd.dat')
                        Nl[exp][components]['dd_lss'][i_min:i_max] = data[:, 1]*1.0 / Dl_conv[i_min:i_max]
                            
                    ####################################################      
                    l0, l1 = 20, 200
                    i0, i1 = np.argmin(np.abs( Cls_fid['ell'] - l0 )), np.argmin(np.abs( Cls_fid['ell'] - l1 ))
                    i0_ = i0 + configurations[exp]['ell_min']
                    i1_ = i1 + configurations[exp]['ell_min']
                    if 'CMBxLSS' in delensing_option_v: Nl[exp][components]['alpha_CMBxLSS'] = round_sig( np.sum(Nl[exp][components]['BB_lss_delens'][i0_:i1_]) / np.sum(Cls_fid['BlBl'][i0:i1]) )
                    if 'CMBxCIB' in delensing_option_v: Nl[exp][components]['alpha_CMBxCIB'] = round_sig( np.sum(Nl[exp][components]['BB_CIB_delens'][i0_:i1_]) / np.sum(Cls_fid['BlBl'][i0:i1]) )
                    if 'CMBxCMB' in delensing_option_v: Nl[exp][components]['alpha_CMBxCMB'] = round_sig( np.sum(Nl[exp][components]['BB_delens'][i0_:i1_]) / np.sum(Cls_fid['BlBl'][i0:i1]) )
                    ####################################################

                else:

                    ## WITH FOREGROUNDS 
                    # define a bunch of post-delensing arrays and their
                    # ranges. the delensing code outputs C_ls in the
                    # range [ell_min: ell_max + 1]; we want D_ls in the
                    # range [ell_min_camb: ell_max + 1]
                    # NB: delensing code outputs N_l^pp, so must
                    # convert to N_l^dd
                    Nl[exp][components]['dd_post_comp_sep'] = np.zeros(ell_count)
                    Nl[exp][components]['BB_delens_post_comp_sep'] = np.zeros(ell_count)
                    Nl[exp][components]['dd_CIB_post_comp_sep'] = np.zeros(ell_count)
                    Nl[exp][components]['BB_CIB_delens_post_comp_sep'] = np.zeros(ell_count)
                    Nl[exp][components]['dd_lss_post_comp_sep'] = np.zeros(ell_count)
                    Nl[exp][components]['BB_lss_delens_post_comp_sep'] = np.zeros(ell_count)

                    # write BB noise to file to pass to command-line delensing F90 code
                    output = np.column_stack((ells.flatten(), Nl[exp][components]['BB_post_comp_sep'].flatten()))
                    np.savetxt(exp_prefix + '_n_l_ee_bb_cs.dat', output, fmt = ' %5.1i   %12.5e')

                    # call iterative CMB EB delensing code using post-comp.sep. noise
                    if 'CMBxCMB' in delensing_option_v:
                        com = [delexe, '-f_sky', str(configurations[exp]['fsky']), \
                                       '-l_min', str(configurations[exp]['ell_min']), \
                                       '-l_max', str(configurations[exp]['ell_max']), \
                                       '-lensed_path', prefix + '_fidCls_lensedtotCls.dat', \
                                       '-unlensed_path', prefix + '_fidCls_lenspotentialCls.dat', \
                                       '-noise_path', exp_prefix + '_n_l_ee_bb_cs.dat', \
                                       '-prefix', exp_prefix]
                        try:
                            sp.check_call(com)
                        except:
                            print('external_program failed for CMBxCMB')
                        data = np.genfromtxt(exp_prefix + '_c_l_bb_res.dat')
                        Nl[exp][components]['BB_delens_post_comp_sep'][i_min:i_max] = data[:, 1]*1.0 / Dl_conv[i_min:i_max]
                        data = np.genfromtxt(exp_prefix + '_n_l_dd.dat')
                        Nl[exp][components]['dd_post_comp_sep'][i_min:i_max] = data[:, 1]*1.0 / Dl_conv[i_min:i_max]

                    # call CMB x CIB delensing code using post-comp.sep. noise
                    if 'CMBxCIB' in delensing_option_v:
                        com = [delexe, '-f_sky', str(configurations[exp]['fsky']), \
                                       '-l_min', str(configurations[exp]['ell_min']), \
                                       '-l_max', str(configurations[exp]['ell_max']), \
                                       '-lensed_path', prefix + '_fidCls_lensedtotCls.dat', \
                                       '-unlensed_path', prefix + '_fidCls_lenspotentialCls.dat', \
                                       '-noise_path', exp_prefix + '_n_l_ee_bb_cs.dat', \
                                       '-f_l_cor_path', prefix + '_f_l_cor_planck_545.dat', \
                                       '-prefix', exp_prefix]
                        try:
                            sp.check_call(com)
                        except:
                            print('external_program failed for CMBxCIB')
                        data = np.genfromtxt(exp_prefix + '_c_l_bb_res.dat')
                        Nl[exp][components]['BB_CIB_delens_post_comp_sep'][i_min:i_max] = data[:, 1]*1.0 / Dl_conv[i_min:i_max]
                        data = np.genfromtxt(exp_prefix + '_n_l_dd.dat')
                        Nl[exp][components]['dd_CIB_post_comp_sep'][i_min:i_max] = data[:, 1]*1.0 / Dl_conv[i_min:i_max]

                    # call PERFECT delensing code using raw noise
                    if 'CMBxLSS' in delensing_option_v:
                        com = [delexe, '-f_sky', str(configurations[exp]['fsky']), \
                                       '-l_min', str(configurations[exp]['ell_min']), \
                                       '-l_max', str(configurations[exp]['ell_max']), \
                                       '-lensed_path', prefix + '_fidCls_lensedtotCls.dat', \
                                       '-unlensed_path', prefix + '_fidCls_lenspotentialCls.dat', \
                                       '-noise_path', exp_prefix + '_n_l_ee_bb.dat', \
                                       '-f_l_cor_path', prefix + '_f_l_cor_perfect.dat', \
                                       '-prefix', exp_prefix]
                        try:
                            sp.check_call(com)
                        except:
                            print('external_program failed for CMBxLSS')
                        data = np.genfromtxt(exp_prefix + '_c_l_bb_res.dat')
                        Nl[exp][components]['BB_lss_delens_post_comp_sep'][i_min:i_max] = data[:, 1]*1.0 / Dl_conv[i_min:i_max]
                        data = np.genfromtxt(exp_prefix + '_n_l_dd.dat')
                        Nl[exp][components]['dd_lss_post_comp_sep'][i_min:i_max] = data[:, 1]*1.0 / Dl_conv[i_min:i_max]

                    ####################################################      
                    l0, l1 = 20, 200
                    i0, i1 = np.argmin(np.abs( Cls_fid['ell'] - l0 )), \
                             np.argmin(np.abs( Cls_fid['ell'] - l1 ))
                    i0_ = i0 + configurations[exp]['ell_min']
                    i1_ = i1 + configurations[exp]['ell_min']
                    if 'CMBxLSS' in delensing_option_v: Nl[exp][components]['alpha_CMBxLSS_post_comp_sep'] = round_sig( np.sum(Nl[exp][components]['BB_lss_delens_post_comp_sep'][i0_:i1_]) / np.sum(Cls_fid['BlBl'][i0:i1]) )
                    if 'CMBxCIB' in delensing_option_v: Nl[exp][components]['alpha_CMBxCIB_post_comp_sep'] = round_sig( np.sum(Nl[exp][components]['BB_CIB_delens_post_comp_sep'][i0_:i1_]) / np.sum(Cls_fid['BlBl'][i0:i1]) )
                    if 'CMBxCMB' in delensing_option_v: Nl[exp][components]['alpha_CMBxCMB_post_comp_sep'] = round_sig( np.sum(Nl[exp][components]['BB_delens_post_comp_sep'][i0_:i1_]) / np.sum(Cls_fid['BlBl'][i0:i1]) )
