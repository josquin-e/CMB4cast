// module containing main functionality. the functions contained
// should be callable by Python after compilation with
// cc -O3 -std=c99 -fPIC -Wall -shared -o <LIB>.so delensing_performance.c
// see delensing_examples.py for examples of how to run.
// Authors: Stephen Feeney (s.feeney@imperial.ac.uk)
//          Numerical Recipes in C
// References: Smith et al., "Delensing CMB polarization with external
// datasets", JCAP 6 14 (2012), doi://10.1088/1475-7516/2012/06/014},
// arXiv:1010.0048

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

const double pi = 3.141592653589793238462643383279502884197;
const double t_cmb = 2.7256; // K
const double inf_noise = 1.0e10; // uK
const int buf_len = 256;


//////////////////////////////////////////////////////////////////////


/* 
   Given the lower and upper limits of integration x1 and x2, and 
   given n, this routine returns arrays x[1..n] and w[1..n] of length 
   n, containing the abscissas and weights of the Gauss-Legendre
   n-point quadrature formula. CALL WITH x-1, w-1 TO IF YOUR INPUTS
   ARE ZERO-INDEXED!
   
    * \param[in] x1 Lower bound of range.
    * \param[in] x2 Upper bound of range.
    * \param[out] x Node positions (i.e. roots of Legendre polynomials).
    * \param[out] w Corresponding weights.
    * \param[in] n Number of points (note memory must already be
    * allocated for x and w to store n terms).
    * \author Numerical recipes (gauleg).
*/
void gl_quad(double x1, double x2, int n, double x[], double w[]) {
  
  int m,j,i;
  double z1,z,xm,xl,pp,p3,p2,p1; // high precision good idea
  double eps = 1.0e-14; // reduced from 3.0e-11
  
  // the roots are symmetric in the interval, so we only have to
  // find half of them.
  m=(n+1)/2;
  xm=0.5*(x2+x1); 
  xl=0.5*(x2-x1);
  
  // loop over roots
  for (i=1;i<=m;i++) {
    
    // approximate the ith root then refine by Newton's Method
    z=cos(3.141592654*(i-0.25)/(n+0.5));
    do {
      
      // loop up the recurrence relation to get the Legendre 
      // polynomial evaluated at z (the approximate root)
      p1=1.0;
      p2=0.0;
      for (j=1;j<=n;j++) {
        p3=p2;
        p2=p1;
        p1=((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j;
      }
      
      // at this point p1 is the desired Legendre polynomial. compute
      // pp, its derivative, using the standard relation involving 
      // the Legendre polynomial one order lower (p2)
      pp=n*(z*p1-p2)/(z*z-1.0);
      z1=z;
      z=z1-p1/pp; // Newton’s method.
    } while (fabs(z-z1) > eps);
    
    // scale the root into the correct interval and define its 
    // symmetric counterpart
    x[i]=xm-xl*z;
    x[n+1-i]=xm+xl*z;
    
    // define their weights
    w[i]=2.0*xl/((1.0-z*z)*pp*pp);
    w[n+1-i]=w[i];
    
  }
}

//////////////////////////////////////////////////////////////////////


/*
   Use three-term recursion for d^l_mn to generate d^l+1_mn
*/
void little_d_recursion(int l, int m, int n, int n_x, double *x,
                        double *d_l_mn, double *d_lm1_mn) {

  int i1;
  double ld, md, nd, lp1, term_one, term_two, twolp1;
  
  // speed things up a bit
  ld = (double) l;
  md = (double) m;
  nd = (double) n;
  lp1 = ld + 1.0;
  twolp1 = 2.0 * ld + 1.0;
  
  // do the recursion
  for (i1 = 0; i1 < n_x; i1++) {
    term_one = lp1 * twolp1 * x[i1] - md * nd * twolp1 / ld;
    term_two = lp1 * sqrt((ld * ld - md * md) * 
                          (ld * ld - nd * nd)) / ld;
    d_l_mn[i1] = (term_one * d_l_mn[i1] - term_two * d_lm1_mn[i1]) / 
      sqrt((lp1 * lp1 - md * md) * (lp1 * lp1 - nd * nd));
  }
  
}


//////////////////////////////////////////////////////////////////////

/*
   Sum of double array...
*/
double sum(double *x, int n_x) {

  int i1;
  double sum;

  sum = 0.0;
  for (i1 = 0; i1 < n_x; i1++) {
    sum += x[i1];
  }
  return sum;
}

//////////////////////////////////////////////////////////////////////


/*
   Estimate the lensing deflection sensitivity and delensing ability
*/
void delensing_performance(int l_min, int l_max, double *c_l_ee_u,
                           double *c_l_ee_l, double *c_l_bb_t,
                           double *c_l_bb_l, double *c_l_pp, 
                           double *f_l_cor, double *n_l_ee_bb,
                           double conv_thresh, bool no_iteration,
                           double *n_l_pp, double *c_l_bb_r) {
  
  int i, j, k, ind, n_x, n_ell, l_min_abs;
  double *x, *w, *z_p_p, *z_p_m, *z_ep_1p1, *z_ep_1m1, *z_ep_3p1,
          *z_ep_3m1, *z_ep_3p3, *z_ep_3m3, *d_l_1p1, *d_l_1m1, *d_l_3p1,
            *d_l_3m1, *d_l_3p3, *d_l_3m3, *d_lm1_1p1, *d_lm1_1m1, *d_lm1_3p1,
              *d_lm1_3m1, *d_lm1_3p3, *d_lm1_3m3, *d_lm1_2p2, *d_lm1_2m2,
                *z_b_p, *z_b_m, *n_l_pp_last, *temp_1p1, *temp_1m1, *temp_3p1,
                  *temp_3m1, *temp_3p3, *temp_3m3, *d_l_2p2, *d_l_2m2, *temp_2p2,
                    *temp_2m2, *c_l_bb_l_only, *temp;
 
  double alpha, conv_test, el;
  
  // define quadrature points. we're performing the needed integrals
  // via Gauss-Legendre integration using n_x = (3 l_max + 1) / 2 
  // quadrature points. these points are the n_x roots of the (n_x)th
  // Legendre polynomial. use Numerical Recipes code from Jason to 
  // calculate the weights and abscissas
  printf("defining quadrature\n");
  n_x = (int) ceil((3.0 * (double) l_max + 1.0) / 2.0);
  printf("sample count = %d\n", n_x);
  x = (double*)malloc(n_x * sizeof(double));
  w = (double*)malloc(n_x * sizeof(double));
  for(i = 0; i < n_x; i++) {
    x[i] = 0.0 ;
    w[i] = 0.0 ;
  }

  gl_quad(-1.0, 1.0, n_x, x-1, w-1); // -1 fixes stupid NR offset
  
  printf("=====================================\n");
  
  // define correlation functions
  printf("allocating arrays\n");
  printf("=====================================\n");
  z_b_p = (double*)malloc(n_x * sizeof(double));
  z_b_m = (double*)malloc(n_x * sizeof(double));
  z_p_p = (double*)malloc(n_x * sizeof(double));
  z_p_m = (double*)malloc(n_x * sizeof(double));
  z_ep_1p1 = (double*)malloc(n_x * sizeof(double));
  z_ep_1m1 = (double*)malloc(n_x * sizeof(double));
  z_ep_3p1 = (double*)malloc(n_x * sizeof(double));
  z_ep_3m1 = (double*)malloc(n_x * sizeof(double));
  z_ep_3p3 = (double*)malloc(n_x * sizeof(double));
  z_ep_3m3 = (double*)malloc(n_x * sizeof(double));
  
  
  // set up Wigner d functions
  d_l_1p1 = (double*)malloc(n_x * sizeof(double));
  d_l_1m1 = (double*)malloc(n_x * sizeof(double));
  d_l_2p2 = (double*)malloc(n_x * sizeof(double));
  d_l_2m2 = (double*)malloc(n_x * sizeof(double));
  d_l_3p1 = (double*)malloc(n_x * sizeof(double));
  d_l_3m1 = (double*)malloc(n_x * sizeof(double));
  d_l_3p3 = (double*)malloc(n_x * sizeof(double));
  d_l_3m3 = (double*)malloc(n_x * sizeof(double));
  d_lm1_1p1 = (double*)malloc(n_x * sizeof(double));
  d_lm1_1m1 = (double*)malloc(n_x * sizeof(double));
  d_lm1_2p2 = (double*)malloc(n_x * sizeof(double));
  d_lm1_2m2 = (double*)malloc(n_x * sizeof(double));
  d_lm1_3p1 = (double*)malloc(n_x * sizeof(double));
  d_lm1_3m1 = (double*)malloc(n_x * sizeof(double));
  d_lm1_3p3 = (double*)malloc(n_x * sizeof(double));
  d_lm1_3m3 = (double*)malloc(n_x * sizeof(double));
  for (i = 0; i < n_x; i++) {
    d_l_1p1[i] = 0.0 ;
    d_l_1m1[i] = 0.0 ;
    d_lm1_1p1[i] = 0.0 ;
    d_lm1_1m1[i] = 0.0 ;
    d_l_2p2[i] = 0.0 ;
    d_l_2m2[i] = 0.0 ;
  }
  for (i = 0; i < n_x; i++) {
    d_l_1p1[i] = 0.5 * (2.0 * pow(x[i], 2.0) + x[i] - 1.0);
    d_l_1m1[i] = 0.5 * (-2.0 * pow(x[i], 2.0) + x[i] + 1.0);
    d_lm1_1p1[i] = 0.5 * (1.0 + x[i]);
    d_lm1_1m1[i] = 0.5 * (1.0 - x[i]);
    d_l_2p2[i] = pow((0.5 * (1.0 + x[i])), 2.0);
    d_l_2m2[i] = pow((0.5 * (1.0 - x[i])), 2.0);
  }
  
  
  // allocate temporary storage
  temp_1p1 = (double*)malloc(n_x * sizeof(double));
  temp_1m1 = (double*)malloc(n_x * sizeof(double));
  temp_2p2 = (double*)malloc(n_x * sizeof(double));
  temp_2m2 = (double*)malloc(n_x * sizeof(double));
  temp_3p1 = (double*)malloc(n_x * sizeof(double));
  temp_3m1 = (double*)malloc(n_x * sizeof(double));
  temp_3p3 = (double*)malloc(n_x * sizeof(double));
  temp_3m3 = (double*)malloc(n_x * sizeof(double));
  temp = (double*)malloc(n_x * sizeof(double));
  
  // @TODO - can we use L_min = 2 for lensed stuff?
  
  // calculate the true lensed B mode: this is the difference between
  // the lensed and unlensed BB inputs. apologies for the confusing
  // nomenclature //
  n_ell = l_max - l_min + 1;
  c_l_bb_l_only = (double*)malloc((n_ell) * sizeof(double));
  for (i = 0; i < n_ell; i++) {
    c_l_bb_l_only[i] = 0.0 ;
  }
  for (i = 0; i < n_ell; i++) {
    c_l_bb_l_only[i] = c_l_bb_l[i] - c_l_bb_t[i];
  }

  
  if (z_b_p == NULL ||
  z_b_m == NULL ||
  z_p_p == NULL ||
  z_p_m == NULL ||
  z_ep_1p1 == NULL ||
  z_ep_1m1 == NULL ||
  z_ep_3p1 == NULL ||
  z_ep_3m1 == NULL ||
  z_ep_3p3 == NULL ||
  z_ep_3m3 == NULL ||
  d_l_1p1 == NULL ||
  d_l_1m1 == NULL ||
  d_l_2p2 == NULL ||
  d_l_2m2 == NULL ||
  d_l_3p1 == NULL ||
  d_l_3m1 == NULL ||
  d_l_3p3 == NULL ||
  d_l_3m3 == NULL ||
  d_lm1_1p1 == NULL ||
  d_lm1_1m1 == NULL ||
  d_lm1_2p2 == NULL ||
  d_lm1_2m2 == NULL ||
  d_lm1_3p1 == NULL ||
  d_lm1_3m1 == NULL ||
  d_lm1_3p3 == NULL ||
  d_lm1_3m3 == NULL ||
  temp_1p1 == NULL ||
  temp_1m1 == NULL ||
  temp_2p2 == NULL ||
  temp_2m2 == NULL ||
  temp_3p1 == NULL ||
  temp_3m1 == NULL ||
  temp_3p3 == NULL ||
  temp_3m3 == NULL ||
  temp == NULL) {
    printf("MEM FAIL!\n");
    exit(EXIT_FAILURE);
  } else {
    printf("MEM FINE\n");
  }


  printf("%d %d %e %e %e %e %e %e %e %e %d %e %e\n", l_min, l_max,
         sum(c_l_ee_u, n_ell), sum(c_l_ee_l, n_ell), sum(c_l_bb_t, n_ell),
         sum(c_l_bb_l, n_ell), sum(c_l_pp, n_ell), sum(f_l_cor, n_ell),
         sum(n_l_ee_bb, n_ell), conv_thresh, no_iteration,
         sum(n_l_pp, n_ell), sum(c_l_bb_r, n_ell));
  
  // for (i = 0; i < n_ell; i++) {
  //   if (isnan(c_l_ee_u[i])) {
  //     printf("NANANANANANANAN Nl EE U %d\n", i);
  //     exit(EXIT_FAILURE);
  //   }
  // }
  // for (i = 0; i < n_ell; i++) {
  //   if (isnan(c_l_ee_l[i])) {
  //     printf("NANANANANANANAN Cl EE U %d\n", i);
  //     exit(EXIT_FAILURE);
  //   }
  // }
  // for (i = 0; i < n_ell; i++) {
  //   if (isnan(c_l_bb_t[i])) {
  //     printf("NANANANANANANAN Cl BB T %d\n", i);
  //     exit(EXIT_FAILURE);
  //   }
  // }
  // for (i = 0; i < n_ell; i++) {
  //   if (isnan(c_l_bb_l[i])) {
  //     printf("NANANANANANANAN Cl BB l %d\n", i);
  //     exit(EXIT_FAILURE);
  //   }
  // }
  // for (i = 0; i < n_ell; i++) {
  //   if (isnan(c_l_pp[i])) {
  //     printf("NANANANANANANAN Cl PP %d\n", i);
  //     exit(EXIT_FAILURE);
  //   }
  // }
  // for (i = 0; i < n_ell; i++) {
  //   if (isnan(f_l_cor[i])) {
  //     printf("NANANANANANANAN Fl corr %d\n", i);
  //     exit(EXIT_FAILURE);
  //   }
  // }
  // for (i = 0; i < n_ell; i++) {
  //   if (isnan(n_l_ee_bb[i])) {
  //     printf("NANANANANANANAN Nl EE BB %d\n", i);
  //     exit(EXIT_FAILURE);
  //   }
  // }
  // for (i = 0; i < n_ell; i++) {
  //   if (isnan(n_l_pp[i])) {
  //     printf("NANANANANANANAN Nl PP %d\n", i);
  //     exit(EXIT_FAILURE);
  //   }
  // }
  // for (i = 0; i < n_ell; i++) {
  //   if (isnan(c_l_bb_r[i])) {
  //     printf("NANANANANANANAN Cl BB %d\n", i);
  //     exit(EXIT_FAILURE);
  //   }
  // }
         
  
  // loop through l to calculate EE zeta functions, which we only 
  // need to do once: only the BB delensing residuals and phiphi 
  // noise ever change. as the Wigner functions are calculated by 
  // iterating from l = 2, loop over this range but only consider 
  // contributions from l > l_min
  printf("calculating EE correlation functions\n");
  printf("=====================================\n");
  l_min_abs = 2;
  for (i = 0; i < n_x; i++) {
    z_ep_3p3[i] = 0.0;
    z_ep_3m3[i] = 0.0;
    z_ep_3p1[i] = 0.0;
    z_ep_3m1[i] = 0.0;
    z_ep_1p1[i] = 0.0;
    z_ep_1m1[i] = 0.0;
    d_l_3p1[i] = 0.0;
    d_l_3m1[i] = 0.0;
    d_l_3p3[i] = 0.0;
    d_l_3m3[i] = 0.0;
  }

  for (i = l_min_abs; i <= l_max; i++) {
    
    // define C_l index
    ind = i - l_min;
    el = (double) i;

    // d_l_3_blah are zero for l < 3
    if (i == 3) {
      for (k = 0; k < n_x; k++) {
        d_l_3p1[k] = sqrt(15.0) / 8.0 * (1.0 + x[k] - pow(x[k], 2.0) - 
                                                    pow(x[k], 3.0));
        d_l_3m1[k] = sqrt(15.0) / 8.0 * (1.0 - x[k] - pow(x[k], 2.0) + 
                                                    pow(x[k], 3.0));
        d_l_3p3[k] = pow((0.5 * (1.0 + x[k])), 3.0);

        d_l_3m3[k] = pow((0.5 * (1.0 - x[k])), 3.0);
      }
    }
    
    
    // update correlation functions
    if (i >= l_min) {
      
      // N_l^pp functions
      for (k = 0; k < n_x; k++) {
        z_ep_3p3[k] += (2.0 * el + 1.0) * pow(c_l_ee_u[ind], 2.0) / 
          (c_l_ee_l[ind] + n_l_ee_bb[ind]) * (el - 2.0) * (el + 3.0) * d_l_3p3[k];
        z_ep_3m3[k] += (2.0 * el + 1.0) * pow(c_l_ee_u[ind], 2.0) / 
          (c_l_ee_l[ind] + n_l_ee_bb[ind]) * (el - 2.0) * (el + 3.0) * d_l_3m3[k];
        z_ep_3p1[k] += (2.0 * el + 1.0) * pow(c_l_ee_u[ind], 2.0) / 
          (c_l_ee_l[ind] + n_l_ee_bb[ind]) * sqrt((el - 1.0) * (el + 2.0) * (el - 2.0) * (el + 3.0)) * d_l_3p1[k];
        z_ep_3m1[k] += (2.0 * el + 1.0) * pow(c_l_ee_u[ind], 2.0) / 
          (c_l_ee_l[ind] + n_l_ee_bb[ind]) * sqrt((el - 1.0) * (el + 2.0) * (el - 2.0) * (el + 3.0)) * d_l_3m1[k];
        z_ep_1p1[k] += (2.0 * el + 1.0) * pow(c_l_ee_u[ind], 2.0) / 
                                 (c_l_ee_l[ind] + n_l_ee_bb[ind]) * (el - 1.0) * (el + 2.0) * d_l_1p1[k];
        z_ep_1m1[k] += (2.0 * el + 1.0) * pow(c_l_ee_u[ind], 2.0) / 
                                 (c_l_ee_l[ind] + n_l_ee_bb[ind]) * (el - 1.0) * (el + 2.0) * d_l_1m1[k];

        // printf("ind=%d, c_l_ee_l[ind]=%f, n_l_ee_bb[ind]=%f, d_l_3p3[k]=%f\n", ind, c_l_ee_l[ind], n_l_ee_bb[ind], d_l_3p3[k]);
      }
      // if(i>l_min){return;}
      
    }
    
    // update Wigner d functions
    // keep track of Wigner d function for last ell
    for (k = 0; k < n_x; k++) {
      temp_1p1[k] = d_l_1p1[k];
      temp_1m1[k] = d_l_1m1[k];
      temp_3p1[k] = d_l_3p1[k];
      temp_3m1[k] = d_l_3m1[k];
      temp_3p3[k] = d_l_3p3[k];
      temp_3m3[k] = d_l_3m3[k];
    }

    little_d_recursion(i, 1, 1, n_x, x, d_l_1p1, d_lm1_1p1);
    little_d_recursion(i, 1, -1, n_x, x, d_l_1m1, d_lm1_1m1);
    little_d_recursion(i, 3, 1, n_x, x, d_l_3p1, d_lm1_3p1);
    little_d_recursion(i, 3, -1, n_x, x, d_l_3m1, d_lm1_3m1);
    little_d_recursion(i, 3, 3, n_x, x, d_l_3p3, d_lm1_3p3);
    little_d_recursion(i, 3, -3, n_x, x, d_l_3m3, d_lm1_3m3);

    for (k = 0; k < n_x; k++) {
      d_lm1_1p1[k] = temp_1p1[k];
      d_lm1_1m1[k] = temp_1m1[k];
      d_lm1_3p1[k] = temp_3p1[k];
      d_lm1_3m1[k] = temp_3m1[k];
      d_lm1_3p3[k] = temp_3p3[k];
      d_lm1_3m3[k] = temp_3m3[k];
    }

    // printf("%e\n", sum(z_ep_3p3, n_x));
    // for (i = 0; i < n_x; i++) {
    //   if (isnan(z_ep_3p3[i])) {
    //     printf("NANANANANANANAN in z_ep_3p3 .... %d\n", i);
    //     exit(EXIT_FAILURE);
    //   }
    // }

  // end of the loop over i 
  }

  for (i = 0; i < n_x; i++) {
    z_ep_3p3[i] /= 4.0 * pi;
    z_ep_3m3[i] /= 4.0 * pi;
    z_ep_3p1[i] /= 4.0 * pi;
    z_ep_3m1[i] /= 4.0 * pi;
    z_ep_1p1[i] /= 4.0 * pi;
    z_ep_1m1[i] /= 4.0 * pi;
  }
  
  // in LSS-delensing case, assume lensing measurement has come from
  // cross-correlation of noisy lensing estimate with CIB, with some
  // ell-dependent correlation coefficient f_l_cor. in this case the
  // estimator no longer benefits from iteration
  if (sum(f_l_cor, n_ell) > 0.0) {
    
    // calculate the phi zeta function
    printf("LSS lenses: calculating residual BB\n");
    printf("=====================================\n");
    for (i = 0; i < n_x; i++) {
      z_p_p[i] = 0.0;
      z_p_m[i] = 0.0;
      d_l_1p1[i] = 0.5 * (2.0 * pow(x[i], 2.0) + x[i] - 1.0);
      d_l_1m1[i] = 0.5 * (-2.0 * pow(x[i], 2.0) + x[i] + 1.0);
      d_lm1_1p1[i] = 0.5 * (1.0 + x[i]);
      d_lm1_1m1[i] = 0.5 * (1.0 - x[i]);
    }
    for (i = l_min_abs; i <= l_max; i++) {
      
      // define C_l index
      ind = i - l_min;
      el = (double) i;
      
      // calculate phi-phi correlation functions
      if (i >= l_min) {

        for (k = 0; k < n_x; k++) {
          z_p_p[k] += (2.0 * el + 1.0) * pow(f_l_cor[ind], 2.0) *
            c_l_pp[ind] * el * (el + 1.0) * d_l_1p1[k];
          z_p_m[k] += (2.0 * el + 1.0) * pow(f_l_cor[ind], 2.0) *
            c_l_pp[ind] * el * (el + 1.0) * d_l_1m1[k];
        }
        
      }
      
      // update Wigner d functions
      for (k = 0; k < n_x; k++) {
        temp_1p1[k] = d_l_1p1[k];
        temp_1m1[k] = d_l_1m1[k];
      }
      little_d_recursion(i, 1, 1, n_x, x, d_l_1p1, d_lm1_1p1);
      little_d_recursion(i, 1, -1, n_x, x, d_l_1m1, d_lm1_1m1);
      for (k = 0; k < n_x; k++) {
        d_lm1_1p1[k] = temp_1p1[k];
        d_lm1_1m1[k] = temp_1m1[k];
      }
      
    }
    for (i = 0; i < n_x; i++) {
      z_p_p[i] /= 4.0 * pi;
      z_p_m[i] /= 4.0 * pi;
    }
    
    
    // calculate the post-delensing C_l^BB residual
    for (i = 0; i < n_x; i++) {
      d_l_2p2[i] = pow((0.5 * (1.0 + x[i])), 2.0);
      d_l_2m2[i] = pow((0.5 * (1.0 - x[i])), 2.0);
      d_lm1_2p2[i] = 0.0;
      d_lm1_2m2[i] = 0.0;
    }
    for (i = 0; i < n_ell; i++) {
      c_l_bb_r[i] = 0.0;
    }
    for (i = l_min_abs; i <= l_max; i++) {
      
      ind = i - l_min;
      
      // integrate over correlation functions to get C_l: this is 
      // the estimated BB lensing component, so it must be 
      // subtracted from the true lensing component (after the 
      // loop) to calculate the residual
      if (i >= l_min) {
        
        for (k = 0; k < n_x; k++) {
          temp[k] = ((z_ep_3p3[k] * z_p_p[k] + 
                      2.0 * z_ep_3p1[k] * z_p_m[k] + 
                      z_ep_1p1[k] * z_p_p[k]) * d_l_2p2[k] - 
                     (z_ep_3m3[k] * z_p_m[k] + 
                      2.0 * z_ep_3m1[k] * z_p_p[k] + 
                      z_ep_1m1[k] * z_p_m[k]) * d_l_2m2[k]) * 
            w[k] * pi / 4.0;
        }
        c_l_bb_r[ind] = sum(temp, n_x);
        
      }
      
      // update Wigner d function and store last value
      for (k = 0; k < n_x; k++) {
        temp_2p2[k] = d_l_2p2[k];
        temp_2m2[k] = d_l_2m2[k];
      }
      little_d_recursion(i, 2, 2, n_x, x, d_l_2p2, d_lm1_2p2);
      little_d_recursion(i, 2, -2, n_x, x, d_l_2m2, d_lm1_2m2);
      for (k = 0; k < n_x; k++) {
        d_lm1_2p2[k] = temp_2p2[k];
        d_lm1_2m2[k] = temp_2m2[k];
      }
      
    }
    for (i = 0; i < n_ell; i++) {
      c_l_bb_r[i] = c_l_bb_l_only[i] - c_l_bb_r[i];
    }
    
    
    // set corresponding phi-phi noise
    for (i = 0; i < n_ell; i++) {
      if (f_l_cor[i] == 0.0) {
        n_l_pp[i] = inf_noise;
      } else {
        n_l_pp[i] = c_l_pp[i] * (1.0 / pow(f_l_cor[i], 2.0) - 1.0);
      }
    }
    
    
  } else {
    
    
    // calculate the BB zeta function to obtain a first estimate of the
    // noise on the lensing potential. the tensor, lensing and noise BB
    // components all contribute to the variance of the estimator
    printf("calculating phiphi noise\n");
    printf("=====================================\n");
    for (i = 0; i < n_x; i++) {
      z_b_p[i] = 0.0;
      z_b_m[i] = 0.0;
      temp_2p2[i] = 0.0;
      temp_2m2[i] = 0.0;
      d_l_1p1[i] = 0.0;
      d_l_1m1[i] = 0.0;
      d_lm1_1p1[i] = 0.0;
      d_lm1_1m1[i] = 0.0;
    }

    for (i = l_min_abs; i <= l_max; i++) {
      
      ind = i - l_min;
      el = (double) i;
      
      // update correlation functions
      if (i >= l_min) {
        
        for (k = 0; k < n_x; k++) {
          z_b_p[k] += (2.0 * el + 1.0) / 
              (c_l_bb_t[ind] + c_l_bb_l_only[ind] + 
                   n_l_ee_bb[ind]) * d_l_2p2[k];
          z_b_m[k] += (2.0 * el + 1.0) / 
                 (c_l_bb_t[ind] + c_l_bb_l_only[ind] + 
                   n_l_ee_bb[ind]) * d_l_2m2[k];
          // printf(" z_b_m[k]=%f, c_l_bb_t[ind]=%f, c_l_bb_l_only[ind]=%f, n_l_ee_bb[ind]=%f\n",  z_b_m[k], c_l_bb_t[ind], c_l_bb_l_only[ind], n_l_ee_bb[ind]);
          // printf("k=%d, z_b_m[k]=%f,  \n",  k, z_b_m[k] );
        }
        // if(i == l_min){ return; }
      }
      
      // update Wigner d function and store last value
      for (k = 0; k < n_x; k++) {
        temp_2p2[k] = d_l_2p2[k];
        temp_2m2[k] = d_l_2m2[k];
      }
      little_d_recursion(i, 2, 2, n_x, x, d_l_2p2, d_lm1_2p2);
      little_d_recursion(i, 2, -2, n_x, x, d_l_2m2, d_lm1_2m2);
      for (k = 0; k < n_x; k++) {
        d_lm1_2p2[k] = temp_2p2[k];
        d_lm1_2m2[k] = temp_2m2[k];
      }
    // end of the loop over i  
    }

    for (i = 0; i < n_x; i++) {
      z_b_p[i] /= 4.0 * pi;
      z_b_m[i] /= 4.0 * pi;
    }
    
    // calculate lensing potential noise
    for (i = 0; i < n_x; i++) {
      d_l_1p1[i] = 0.5 * (2.0 * pow(x[i], 2.0) + x[i] - 1.0);
      d_l_1m1[i] = 0.5 * (-2.0 * pow(x[i], 2.0) + x[i] + 1.0);
      d_lm1_1p1[i] = 0.5 * (1.0 + x[i]);
      d_lm1_1m1[i] = 0.5 * (1.0 - x[i]);
    }

    for (i = l_min_abs; i <= l_max; i++) {
      
      // define index into C_l array
      ind = i - l_min;
      el = (double) i;
      
      // integrate over correlation functions to get C_l
      if (i >= l_min) {
        // if(i > l_min){ return;}        
        for (k = 0; k < n_x; k++) {
          temp[k] = ((z_ep_3p3[k] * z_b_p[k] - 
                      2.0 * z_ep_3m1[k] * z_b_m[k] + 
                      z_ep_1p1[k] * z_b_p[k]) * d_l_1p1[k] - 
                     (z_ep_3m3[k] * z_b_m[k] - 
                      2.0 * z_ep_3p1[k] * z_b_p[k] + 
                      z_ep_1m1[k] * z_b_m[k]) * d_l_1m1[k]) * 
                              w[k];
        
        // if((k>=0)&&(k<=5)){
        // printf("temp[k]=%f || z_ep_3p3[k]=%f, z_b_p[k]=%f, z_ep_3m1[k]=%f, z_b_m[k]=%f, z_ep_1p1[k]=%f, d_l_1p1[k]=%f, z_ep_3m3[k]=%f, z_ep_3p1[k]=%f, z_ep_1m1[k]=%f, d_l_1m1[k]=%f, w[k]=%f\n", temp[k], z_ep_3p3[k], z_b_p[k], z_ep_3m1[k], z_b_m[k], z_ep_1p1[k], d_l_1p1[k], z_ep_3m3[k], z_ep_3p1[k], z_ep_1m1[k], d_l_1m1[k], w[k]);
        // printf("k=%d, temp[k]=%f \n", k, temp[k]);
        // }
        // if(k >100 ){ return;}
        }
        // printf("sum(temp, n_x)=%f\n", sum(temp, 2000));
        n_l_pp[ind] = 1.0 / (sum(temp, n_x) * pi / 4.0 * el * (el + 1.0));
        // printf("  n_l_pp[ind] = %f\n",   n_l_pp[ind] );
      
      }

      
      // update Wigner d function and store last value
      for (k = 0; k < n_x; k++) {
        temp_1p1[k] = d_l_1p1[k];
        temp_1m1[k] = d_l_1m1[k];
      }
      little_d_recursion(i, 1, 1, n_x, x, d_l_1p1, d_lm1_1p1);
      little_d_recursion(i, 1, -1, n_x, x, d_l_1m1, d_lm1_1m1);
      for (k = 0; k < n_x; k++) {
        d_lm1_1p1[k] = temp_1p1[k];
        d_lm1_1m1[k] = temp_1m1[k];
      }
      
    }
    // iterate to optimise delensing
    // @TODO - there's a lot that can be calculated just once and stored
    //         if there's memory for it
    printf("delensing iterations\n");
    n_l_pp_last = (double*)malloc((n_ell) * sizeof(double));
    j = 0;
    do {
      
      // iteration counter
      j += 1;
      
      // calculate the phi zeta function
      for (i = 0; i < n_x; i++) {
        z_p_p[i] = 0.0;
        z_p_m[i] = 0.0;
        d_l_1p1[i] = 0.5 * (2.0 * pow(x[i], 2.0) + x[i] - 1.0);
        d_l_1m1[i] = 0.5 * (-2.0 * pow(x[i], 2.0)+ x[i] + 1.0);
        d_lm1_1p1[i] = 0.5 * (1.0 + x[i]);
        d_lm1_1m1[i] = 0.5 * (1.0 - x[i]);
      }
      for (i = l_min_abs; i <= l_max; i++) {
        
        // define C_l index
        ind = i - l_min;
        el = (double) i;
        
        // update correlation functions
        // C_l^BB functions
        if (i >= l_min) {
          
          for (k = 0; k < n_x; k++) {
            z_p_p[k] += (2.0 * el + 1.0) * pow(c_l_pp[ind], 2.0) / 
              (c_l_pp[ind] + n_l_pp[ind]) * el * (el + 1.0) * d_l_1p1[k];
            z_p_m[k] += (2.0 * el + 1.0) * pow(c_l_pp[ind], 2.0) / 
              (c_l_pp[ind] + n_l_pp[ind]) * el * (el + 1.0) * d_l_1m1[k];
          }
          
        }
        
        // update Wigner d functions
        for (k = 0; k < n_x; k++) {
          temp_1p1[k] = d_l_1p1[k];
          temp_1m1[k] = d_l_1m1[k];
        }
        little_d_recursion(i, 1, 1, n_x, x, d_l_1p1, d_lm1_1p1);
        little_d_recursion(i, 1, -1, n_x, x, d_l_1m1, d_lm1_1m1);
        for (k = 0; k < n_x; k++) {
          d_lm1_1p1[k] = temp_1p1[k];
          d_lm1_1m1[k] = temp_1m1[k];
        }
        
      }

      for (i = 0; i < n_x; i++) {
        z_p_p[i] /= 4.0 * pi;
        z_p_m[i] /= 4.0 * pi;
      }
      
      
      // calculate the post-delensing C_l^BB residual
      for (i = 0; i < n_x; i++) {
        d_l_2p2[i] = pow((0.5 * (1.0 + x[i])), 2);
        d_l_2m2[i] = pow((0.5 * (1.0 - x[i])), 2);
        d_lm1_2p2[i] = 0.0;
        d_lm1_2m2[i] = 0.0;
      }
      for (i = 0; i < n_ell; i++) {
        c_l_bb_r[i] = 0.0;
      }
      for (i = l_min_abs; i <= l_max; i++) {
        
        ind = i - l_min;
        
        // integrate over correlation functions to get C_l: this is 
        // the estimated BB lensing component, so it must be 
        // subtracted from the true lensing component (after the 
        // loop) to calculate the residual
        if (i >= l_min) {
          
          for (k = 0; k < n_x; k++) {
            temp[k] = ((z_ep_3p3[k] * z_p_p[k] + 
                        2.0 * z_ep_3p1[k] * z_p_m[k] + 
                        z_ep_1p1[k] * z_p_p[k]) * d_l_2p2[k] - 
                       (z_ep_3m3[k] * z_p_m[k] + 
                        2.0 * z_ep_3m1[k] * z_p_p[k] + 
                        z_ep_1m1[k] * z_p_m[k]) * d_l_2m2[k]) * 
              w[k] * pi / 4.0;
          }
          c_l_bb_r[ind] = sum(temp, n_x);
          
        }
        
        // update Wigner d function and store last value
        for (k = 0; k < n_x; k++) {
          temp_2p2[k] = d_l_2p2[k];
          temp_2m2[k] = d_l_2m2[k];
        }
        little_d_recursion(i, 2, 2, n_x, x, d_l_2p2, d_lm1_2p2);
        little_d_recursion(i, 2, -2, n_x, x, d_l_2m2, d_lm1_2m2);
        for (k = 0; k < n_x; k++) {
          d_lm1_2p2[k] = temp_2p2[k];
          d_lm1_2m2[k] = temp_2m2[k];
        }
        
      }
      for (i = 0; i < n_ell; i++) {
        c_l_bb_r[i] = c_l_bb_l_only[i] - c_l_bb_r[i];
      }
      
      
      // terminate now if non-iterative EBEB delensing
      // report projected improvement
      if (no_iteration) {
        
        printf("no_iteration set: terminating\n");
        break;
        
      }
      
      
      // re-calculate the BB zeta function. now the tensor, delensing 
      // residual and noise BB components contribute to the variance 
      // of the estimator
      for (i = 0; i < n_x; i++) {
        z_b_p[i] = 0.0;
        z_b_m[i] = 0.0;
        d_l_2p2[i] = pow((0.5 * (1.0 + x[i])), 2);
        d_l_2m2[i] = pow((0.5 * (1.0 - x[i])), 2);
        d_lm1_2p2[i] = 0.0;
        d_lm1_2m2[i] = 0.0;
      }
      for (i = l_min_abs; i <= l_max; i++) {
        
        ind = i - l_min;
        el = (double) i;
        
        // update correlation functions
        if (i >= l_min) {
          
          for (k = 0; k < n_x; k++) {
            z_b_p[k] += (2.0 * el + 1.0) / 
              (c_l_bb_t[ind] + c_l_bb_r[ind] + n_l_ee_bb[ind]) * 
              d_l_2p2[k];
            z_b_m[k] += (2.0 * el + 1.0) / 
              (c_l_bb_t[ind] + c_l_bb_r[ind] + n_l_ee_bb[ind]) * 
              d_l_2m2[k];
          }
          
        }
        
        // update Wigner d function and store last value
        for (k = 0; k < n_x; k++) {
          temp_2p2[k] = d_l_2p2[k];
          temp_2m2[k] = d_l_2m2[k];
        }
        little_d_recursion(i, 2, 2, n_x, x, d_l_2p2, d_lm1_2p2);
        little_d_recursion(i, 2, -2, n_x, x, d_l_2m2, d_lm1_2m2);
        for (k = 0; k < n_x; k++) {
          d_lm1_2p2[k] = temp_2p2[k];
          d_lm1_2m2[k] = temp_2m2[k];
        }
        
        
      }
      for (i = 0; i < n_x; i++) {
        z_b_p[i] /= 4.0 * pi;
        z_b_m[i] /= 4.0 * pi;
      }
      
      
      // calculate the post-delensing lensing potential noise
      for (i = 0; i < n_x; i++) {
        d_l_1p1[i] = 0.5 * (2.0 * pow(x[i], 2.0) + x[i] - 1.0);
        d_l_1m1[i] = 0.5 * (-2.0 * pow(x[i], 2.0) + x[i] + 1.0);
        d_lm1_1p1[i] = 0.5 * (1.0 + x[i]);
        d_lm1_1m1[i] = 0.5 * (1.0 - x[i]);
      }

      for (i = 0; i < n_ell; i++) {
        n_l_pp_last[i] = n_l_pp[i];
        n_l_pp[i] = 0.0;
      }
      for (i = l_min_abs; i <= l_max; i++) {
        
        // define index into C_l array
        ind = i - l_min;
        el = (double) i;
        
        // integrate over correlation functions to get C_l
        if (i >= l_min) {
          
          for (k = 0; k < n_x; k++) {
            temp[k] = ((z_ep_3p3[k] * z_b_p[k] - 
                        2.0 * z_ep_3m1[k] * z_b_m[k] + 
                          z_ep_1p1[k] * z_b_p[k]) * d_l_1p1[k] - 
                          (z_ep_3m3[k] * z_b_m[k] - 
                            2.0 * z_ep_3p1[k] * z_b_p[k] + 
                              z_ep_1m1[k] * z_b_m[k]) * d_l_1m1[k]) * 
                                w[k];
          }
          n_l_pp[ind] = 1.0 / (sum(temp, n_x) * pi / 4.0 * el * (el + 1.0));
          
        }
        
        // update Wigner d function and store last value
        for (k = 0; k < n_x; k++) {
          temp_1p1[k] = d_l_1p1[k];
          temp_1m1[k] = d_l_1m1[k];
        }
        little_d_recursion(i, 1, 1, n_x, x, d_l_1p1, d_lm1_1p1);
        little_d_recursion(i, 1, -1, n_x, x, d_l_1m1, d_lm1_1m1);
        for (k = 0; k < n_x; k++) {
          d_lm1_1p1[k] = temp_1p1[k];
          d_lm1_1m1[k] = temp_1m1[k];
        }
        
      }
      
      // test for convergence and pretty plot again
      for (i = 0; i < n_ell; i++) {
        temp[i] = (n_l_pp[i] - n_l_pp_last[i]) / n_l_pp[i];
      }


      conv_test = fabs(sum(temp, n_ell));
      printf("%3d: |sum(delta N_\ell^pp)| = %10.3E\n", j, conv_test);
      if (conv_test < conv_thresh) break;
      
      
    } while(true);
    printf("=====================================\n");

  }
  
  
  // report projected improvement
  alpha = (c_l_bb_l_only[0] + n_l_ee_bb[0]) / 
          (c_l_bb_r[0] + n_l_ee_bb[0]);
  printf("sigma_r_0 / sigma_r = %10.3E\n", alpha);
  printf("=====================================\n");
  
  
  // clean everything up
  free(x); free(w); free(z_b_p); free(z_b_m); free(z_p_p);
  free(z_p_m); free(z_ep_1p1); free(z_ep_1m1); free(z_ep_3p1);
  free(z_ep_3m1); free(z_ep_3p3); free(z_ep_3m3); free(d_l_1p1);
  free(d_l_1m1); free(d_l_2p2); free(d_l_2m2); free(d_l_3p1);
  free(d_l_3m1); free(d_l_3p3); free(d_l_3m3); free(d_lm1_1p1);
  free(d_lm1_1m1); free(d_lm1_2p2); free(d_lm1_2m2); free(d_lm1_3p1);
  free(d_lm1_3m1); free(d_lm1_3p3); free(d_lm1_3m3); free(temp_1p1);
  free(temp_1m1); free(temp_2p2); free(temp_2m2); free(temp_3p1);
  free(temp_3m1); free(temp_3p3); free(temp_3m3); free(temp);
  free(c_l_bb_l_only);
  if (sum(f_l_cor, n_ell) == 0.0) free(n_l_pp_last);
  return;
  
}
  
  
//////////////////////////////////////////////////////////////////////


/*
   Wrapper for the main lensing deflection sensitivity subroutine.
   converts a set of raw noise inputs (number of frequencies, beam
   and pixel noise at each frequency) into noise power spectra before
   passing to the delensing performance code
*/
void delensing_performance_raw(int l_min, int l_max, double *c_l_ee_u,
                               double *c_l_ee_l, double *c_l_bb_u,
                               double *c_l_bb_l, double *c_l_pp, 
                               double *f_l_cor, int n_freq,
                               double *sigma_pix_p,
                               double *beam_fwhm_am, 
                               double conv_thresh, bool no_iteration,
                               double *n_l_pp, double *c_l_bb_r) {
  
  int i, j, n_ell = l_max - l_min + 1;
  double *n_l_ee_bb;
  double beam_fwhm[n_freq], beam_theta[n_freq];
  double el, beam_l;
  
  // beam conversions
  for (i = 0; i < n_freq; i++) {
    beam_fwhm[i] = beam_fwhm_am[i] * pi / 60.0 / 180.0;
    beam_theta[i] = beam_fwhm[i] / sqrt(8.0 * log(2.0));
  }
  
  // calculate the noise, including the deconvolved beam, for each
  // frequency and combine: the noise at each l must be added in quadrature
  n_l_ee_bb = (double*)malloc(n_ell * sizeof(double));
  for (i = 0; i < n_ell; i++) {
    
    el = (double) (i + l_min);
    for (j = 0; j < n_freq; j++) {
      beam_l = exp(pow(beam_theta[j], 2.0) * (el * (el + 1.0)));
      n_l_ee_bb[i] += 1.0 / (pow(beam_fwhm[j] * sigma_pix_p[j], 2.0) * beam_l);
    }
    n_l_ee_bb[i] = 1.0 / n_l_ee_bb[i];
  }
  
  // forecast!
  delensing_performance(l_min, l_max, c_l_ee_u,
                        c_l_ee_l, c_l_bb_u,
                        c_l_bb_l, c_l_pp, 
                        f_l_cor, n_l_ee_bb,
                        conv_thresh, no_iteration,
                        n_l_pp, c_l_bb_r);
  free(n_l_ee_bb);
  return;
  
}
  

//////////////////////////////////////////////////////////////////////


/*
   Count the number of components in a vector argument
*/
int count_vec(char *to_parse) {
  
  int len, i, count = 1;
  len = strlen(to_parse);
  for (i = 0; i < len; i++) {
    if (to_parse[i] == ',') count++;
  }
  return count;
  
}
  
  
//////////////////////////////////////////////////////////////////////


/*
   Parse the components of a vector argument
*/
void parse_vec(char *to_parse, double *vec) {

  int i;
  char *to_tok, *tok;
  const char delim[] = ",";
  to_tok = strdup(to_parse);
  
  tok = strtok(to_tok, delim);
  i = 0;
  while(tok) {
    vec[i] = atof(tok);
    tok = strtok(NULL, delim);
    i++;
  }
  free(to_tok);
  return;
  
}


//////////////////////////////////////////////////////////////////////


/*
   Check lengths of vector command-line options
*/
int check_options(int argc, char *argv[]) {
  
  
  int i, n_spp = 0, n_bf = 0;

  for (i = 1; i < argc; i += 2) {
    
    if (i == argc - 1 && strcmp(argv[i], "-help") != 0) {
      printf("option %s has no argument!\n", argv[i]);
      exit(EXIT_FAILURE);
    }
    
    // read each argument in turn
    if (strcmp(argv[i], "-sigma_pix_p") == 0) {

      n_spp = count_vec(argv[i + 1]);
      
    } else if (strcmp(argv[i], "-beam_fwhm") == 0) {

      n_bf = count_vec(argv[i + 1]);
      
    }

  }

  // check the vector options all have the same dimension
  if (n_spp != n_bf) {
    
    printf("ERROR: sigma_pix_p and freq have inconsistent sizes\n");
    exit(EXIT_FAILURE);

  } else {
    
    return n_spp;
    
  }

}


//////////////////////////////////////////////////////////////////////


/*
   Parse command-line options (based on HEALPix)
*/
void parse_options(int argc, char *argv[], int *l_min, int *l_max,
                   char **unlensed_path, char **lensed_path, 
                   char **noise_path, double *f_sky,
                   double *sigma_pix_p, double *beam_fwhm,
                   char **f_l_cor_path, double *f_cor,
                   double *conv_thresh, bool *no_iteration,
                   char **prefix) {
  
  
  int i;

  for (i = 1; i < argc; i += 2) {
    
    if (i == argc - 1 && strcmp(argv[i], "-help") != 0) {
      printf("option %s has no argument!\n", argv[i]);
      exit(EXIT_FAILURE);
    }
    
    // read each argument in turn
    if (strcmp(argv[i], "-help") == 0) {
      
      printf("Usage: ./delens_est [-f_sky f_sky]\n");
      printf("                    [-l_min l_min]\n");
      printf("                    [-l_max l_max]\n");
      printf("                    [-unlensed_path unlensed_path]\n");
      printf("                    [-lensed_path lensed_path]\n");
      printf("                    [-noise_path noise_path]\n");
      printf("                    [-sigma_pix_p sigma_pix_p(0),sigma_pix_p(1)]\n");
      printf("                    [-beam_fwhm beam_fwhm(0),beam_fwhm(1)]\n");
      printf("                    [-f_l_cor_path f_l_cor_path]\n");
      printf("                    [-f_cor f_cor]\n");
      printf("                    [-conv_thresh conv_thresh]\n");
      printf("                    [-no_iteration no_iteration]\n");
      printf("                    [-prefix prefix]\n");
      exit(EXIT_SUCCESS);
      
    } else if (strcmp(argv[i], "-f_sky") == 0) {
      
      *f_sky = atof(argv[i + 1]);
      
    } else if (strcmp(argv[i], "-l_min") == 0) {
      
      *l_min = atoi(argv[i + 1]);
      
    } else if (strcmp(argv[i], "-l_max") == 0) {
      
      *l_max = atoi(argv[i + 1]);
      
    } else if (strcmp(argv[i], "-unlensed_path") == 0) {
      
      *unlensed_path = argv[i + 1];
      
    } else if (strcmp(argv[i], "-lensed_path") == 0) {
      
      *lensed_path = argv[i + 1];
      
    } else if (strcmp(argv[i], "-noise_path") == 0) {
      
      *noise_path = argv[i + 1];
      
    } else if (strcmp(argv[i], "-sigma_pix_p") == 0) {
      
      parse_vec(argv[i + 1], sigma_pix_p);
      
    } else if (strcmp(argv[i], "-beam_fwhm") == 0) {
      
      parse_vec(argv[i + 1], beam_fwhm);
      
    } else if (strcmp(argv[i], "-f_l_cor_path") == 0) {
      
      *f_l_cor_path = argv[i + 1];
      
    } else if (strcmp(argv[i], "-f_cor") == 0) {
      
      *f_cor = atof(argv[i + 1]);
      
    } else if (strcmp(argv[i], "-conv_thresh") == 0) {
      
      *conv_thresh = atof(argv[i + 1]);
      
    } else if (strcmp(argv[i], "-no_iteration") == 0) {
      
      if (strcmp(argv[i + 1], "false") == 0) {
        
        *no_iteration = false;
        
      } else {
        
        *no_iteration = true;

      }
      
    } else if (strcmp(argv[i], "-prefix") == 0) {
      
      *prefix = argv[i + 1];
      
    } else {
      
      printf("Unknown option %s ignored\n", argv[i]);

    }

  }
  return;

}


//////////////////////////////////////////////////////////////////////
    

/*
   Program allowing the delensing performance code to be called from 
   the command line. Try compiling with
   "cc -O3 -Wall -fPIC -o delens_est delensing_performance.c"
   and running with
   ./delens_est -f_sky 0.01212 \
   -sigma_pix_p 0.36769553,0.53740115,1.1737973,4.6951890 \
   -beam_fwhm 11.0,11.0,11.0,11.0
*/
int main(int argc, char *argv[]) {
  
  int i, ind, l_min, l_min_camb, l_max, n_ell, n_freq, n_freq_cl;
  int *ell;
  double dummy_d, f_sky, f_cor, conv_thresh;
  double *c_l_ee_u, *c_l_ee_l,
    *c_l_bb_u, *c_l_bb_l, *c_l_pp, *f_l_cor, *n_l_ee_bb,
    *n_l_pp, *c_l_bb_r, *sigma_pix_p, *beam_fwhm_am;
  bool no_iteration;
  FILE *fp_1, *fp_2;
  char *unlensed_path, *lensed_path, *noise_path, *f_l_cor_path,
    *prefix, *fg_res_stub, *n_l_dd_stub, *fg_res_file, *n_l_dd_file;
  char buffer[buf_len];

  
  // default parameter values
  f_sky = 0.75;
  l_min_camb = 2;
  l_min = (int) ceil(2.0 * sqrt(pi / f_sky));
  l_max = 4000;
  unlensed_path = "fiducial_lenspotentialCls.dat";
  lensed_path = "fiducial_lensedtotCls.dat";
  noise_path = "";
  f_l_cor_path = "";
  fg_res_stub = "c_l_bb_res.dat";
  n_l_dd_stub = "n_l_dd.dat";
  prefix = "";
  n_freq = 1;
  n_freq_cl = 0;
  sigma_pix_p = (double*)malloc(n_freq * sizeof(double));
  beam_fwhm_am = (double*)malloc(n_freq * sizeof(double));
  sigma_pix_p[0] = sqrt(2.0) * 0.58;
  beam_fwhm_am[0] = 1.0;
  f_cor = 0.0;
  conv_thresh = 0.01;
  no_iteration = false;
  
  
  // get and check user choices from command line. the check_options
  // subroutine is required by F2PY to get around allocatable array
  // issues
  n_freq_cl = check_options(argc, argv);
  if (n_freq_cl != 0) {
    
    n_freq = n_freq_cl;
    free(sigma_pix_p);
    free(beam_fwhm_am);
    sigma_pix_p = (double*)malloc(n_freq * sizeof(double));
    beam_fwhm_am = (double*)malloc(n_freq * sizeof(double));
     
  }
  parse_options(argc, argv, &l_min, &l_max, &unlensed_path,
                &lensed_path, &noise_path, &f_sky, sigma_pix_p,
                beam_fwhm_am, &f_l_cor_path, &f_cor, &conv_thresh, 
                &no_iteration, &prefix);


  // report instrument characteristics
  printf("instrument characteristics\n");
  printf("f_sky = %5.3f; l_min = %d\n", f_sky, l_min);
  if (strcmp(noise_path, "") != 0) {
    
    printf("reading noise from %s\n", noise_path);
    printf("=====================================\n");
     
  } else {
    
    for (i = 0; i < n_freq; i++) {
      
      printf("sig_pix = %10.3E uK (P), beam = %10.3E'\n",
             sigma_pix_p[i], beam_fwhm_am[i]);
      printf("=====================================\n");

    }

  }
  

  // allocate required storage
  n_ell = l_max - l_min + 1;
  ell = (int*)malloc(n_ell * sizeof(int));
  c_l_ee_u = (double*)malloc(n_ell * sizeof(double));
  c_l_ee_l = (double*)malloc(n_ell * sizeof(double));
  c_l_bb_u = (double*)malloc(n_ell * sizeof(double));
  c_l_bb_l = (double*)malloc(n_ell * sizeof(double));
  c_l_pp = (double*)malloc(n_ell * sizeof(double));
  f_l_cor = (double*)malloc(n_ell * sizeof(double));
  n_l_ee_bb = (double*)malloc(n_ell * sizeof(double));
  n_l_pp = (double*)malloc(n_ell * sizeof(double));
  c_l_bb_r = (double*)malloc(n_ell * sizeof(double));


  // read CMB power spectra and convert from CAMB D_ls to C_ls
  fp_1 = fopen ("/Users/stephen/Code/fortran/prog/fiducial_lenspotentialCls.dat", "r");
  fp_2 = fopen ("/Users/stephen/Code/fortran/prog/fiducial_lensedtotCls.dat", "r");
  for (i = 0; i < l_min - l_min_camb; i++) {
    fgets(buffer, buf_len, fp_1);
    fgets(buffer, buf_len, fp_2);
  }
  for (i = 0; i < n_ell; i++) {
    c_l_ee_u[i] = 0.0 ;
    c_l_ee_l[i] = 0.0 ;
    c_l_bb_u[i] = 0.0 ;
    c_l_bb_l[i] = 0.0 ;
    c_l_pp[i] = 0.0 ;
  }
  for (i = 0; i < n_ell; i++) {
    fscanf(fp_1, "  %4d   %lf   %lf   %lf   %lf   %lf   %lf   %lf",
           &ell[i], &dummy_d, &c_l_ee_u[i], &c_l_bb_u[i], &dummy_d,
           &c_l_pp[i], &dummy_d, &dummy_d);
    fscanf(fp_2, "  %4d   %lf   %lf   %lf   %lf", &ell[i], &dummy_d,
           &c_l_ee_l[i], &c_l_bb_l[i], &dummy_d);
    c_l_ee_u[i] *= 2.0 * pi / ell[i] / (ell[i] + 1.0);
    c_l_ee_l[i] *= 2.0 * pi / ell[i] / (ell[i] + 1.0);
    c_l_bb_u[i] *= 2.0 * pi / ell[i] / (ell[i] + 1.0);
    c_l_bb_l[i] *= 2.0 * pi / ell[i] / (ell[i] + 1.0);
    c_l_pp[i] *= 2.0 * pi / pow(ell[i] * (ell[i] + 1.0), 2.0);
  }
  fclose(fp_1);
  fclose(fp_2);
  
  
  // if requested, read in noise power spectrum and convert from D_l
  if (strcmp(noise_path, "") != 0) {
    
    fp_1 = fopen (noise_path, "r");
    for (i = 0; i < l_min - l_min_camb; i++) {
      fgets(buffer, buf_len, fp_1);
    }
    for (i = 0; i < n_ell; i++) {
      fscanf(fp_1, "  %4d   %lf", &ell[i], &n_l_ee_bb[i]);
      n_l_ee_bb[i] *= 2.0 * pi / ell[i] / (ell[i] + 1.0);
    }
    fclose(fp_1);
    
  }
  

  // if requested, read in ell-dependent f_cor, otherwise build. note
  // that this input array, unlike the others, is assumed to be C_l,
  // i.e., is not ell-weighted
  if (strcmp(f_l_cor_path, "") != 0) {
    
    fp_1 = fopen (f_l_cor_path, "r");
    for (i = 0; i < l_min - l_min_camb; i++) {
      fgets(buffer, buf_len, fp_1);
    }
    for (i = 0; i < n_ell; i++) {
      fscanf(fp_1, "  %4d   %lf", &ell[i], &f_l_cor[i]);
    }
    fclose(fp_1);
    
  } else{
    
    for (i = 0; i < n_ell; i++) {
      f_l_cor[i] = f_cor;
    }
    
  }
  
  
  // call delensing performance estimation code
  if (strcmp(noise_path, "") != 0) {
    
    delensing_performance(l_min, l_max, c_l_ee_u, c_l_ee_l, c_l_bb_u,
                          c_l_bb_l, c_l_pp, f_l_cor, n_l_ee_bb,
                          conv_thresh, no_iteration, n_l_pp,
                          c_l_bb_r);
    
  } else {
    
    delensing_performance_raw(l_min, l_max, c_l_ee_u, c_l_ee_l,
                              c_l_bb_u, c_l_bb_l, c_l_pp, f_l_cor,
                              n_freq, sigma_pix_p, beam_fwhm_am,
                              conv_thresh, no_iteration, n_l_pp,
                              c_l_bb_r);

  }
  

  // write outputs to file, converting lensing potential power spectrum
  // to deflection power spectrum
  fg_res_file = (char*)malloc(strlen(prefix) +
                              strlen(fg_res_stub) + 2);
  n_l_dd_file = (char*)malloc(strlen(prefix) +
                              strlen(n_l_dd_stub) + 2);
  if (strcmp(prefix, "") != 0) {
    strcpy(fg_res_file, prefix);
    strcat(fg_res_file, "_");
    strcpy(n_l_dd_file, prefix);
    strcat(n_l_dd_file, "_");
  }
  strcat(fg_res_file, fg_res_stub);
  strcat(n_l_dd_file, n_l_dd_stub);
  fp_1 = fopen (fg_res_file, "w");
  fp_2 = fopen (n_l_dd_file, "w");
  for (i = l_min; i <= l_max; i++) {
    
    ind = i - l_min;
    fprintf(fp_1, "%4d %19.12e\n", i, c_l_bb_r[ind]);
    fprintf(fp_2, "%4d %19.12e\n", i, n_l_pp[ind] * ell[ind] * (ell[ind] + 1.0));
  }
  fclose(fp_1);
  fclose(fp_2);

  
  // tidy up
  free(sigma_pix_p); free(beam_fwhm_am); free(ell); free(c_l_ee_u);
  free(c_l_ee_l); free(c_l_bb_u); free(c_l_bb_l); free(c_l_pp);
  free(f_l_cor); free(n_l_ee_bb); free(n_l_pp); free(c_l_bb_r);
  free(fg_res_file); free(n_l_dd_file);
  return 0;

  
}
