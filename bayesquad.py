#!/usr/bin/env python

"""
RS 2017/02/17:  Bayesian quadrature compressions of filter transmission curves

These routines take as input an empirical transmission curve and use Bayesian
quadrature (O'Hagan 1991; Huszar & Duvenaud 2012) as a means for approximating
the integral of that transmission curve against a Gaussian process with known
covariance function.  This will be an important step of making tractable the
problem of inferring a spectral time series against photometry.
"""

import sys
import glob
import dill as pickle
import numpy as np
from scipy import linalg, interpolate, integrate, optimize


class BQFilter(object):
    """
    Implements compression of a filter transfer function using Bayesian
    quadrature.  Chooses evaluation points and weights using a greedy
    algorithm that successively minimizes, at each stage, the mean square
    difference between the original filter function and an approximate
    version as evaluated on functions under a Gaussian process prior.
    Uses the algorithm as outlined in section 3 of
        "Optimally-Weighted Herding is Bayesian Quadrature",
            F. Huszar & D. Duvenaud, Proc. UAI 2012, p. 377.
    """
    # In general, underscores represent internal variables associated
    # with the training points.

    def _vmsg(self, msg):
        """
        Writes unbuffered status messages to stdout.
        """
        if self.verbose:
            print msg
        sys.stdout.flush()

    def __init__(self, _x, _fx, kcov, khyp=[ ], verbose=False):
        """
        Parameters:
            _x:    points at which original filter transfer function is
                   empirically defined (e.g. wavelengths); this will form
                   the probability distribution p for Bayesian quadrature
            _fx:   transfer function values (e.g. filter transmission)
            kcov:  covariance kernel for GP integrand (callable),
                   defined to take two arguments x1, x2
            khyp:  (fixed) hyperparameters for kcov (np.array of floats)
            verbose:  print status messages? (bool)
        """
        # Normalize the training points to zero mean and unit variance,
        # to help stabilize the scipy quadrature routines
        self.verbose = verbose
        self._x, self._fx = _x, _fx
        self._xmu, self._xsig = np.mean(_x, axis=0), np.std(_x, axis=0)
        self._u = self._x2u(self._x)
        self._ulo, self._uhi = self._u.min(), self._u.max()
        # Internal callables for filter transfer function and covariance
        integZu = interpolate.interp1d(self._u, _fx)
        self.Zu, Zu_err = integrate.quad(integZu, self._ulo, self._uhi)
        self._vmsg("__init__:  Zu = {:.3g} +/- {:.3g}".format(self.Zu, Zu_err))
        self._vmsg("__init__:  orig. filter norm = {:.3g}".format(
                   self.Zu * self._xsig))
        self.pu = interpolate.interp1d(self._u, _fx/self.Zu)
        # self.kcov = kcov
        self.kcov, self.khyp = kcov, khyp
        # Internal state
        self.u = np.array([ ])      # quadrature points
        self.zu = np.array([ ])     # quadrature weights
        self.K = np.array([[ ]])    # covariance at quadrature points
        # Starting variance
        self._calc_base_variance_integral()

    def _x2u(self, x):
        return (x - self._xmu)/self._xsig

    def _u2x(self, u):
        return u*self._xsig + self._xmu

    def kxx(self, x1, x2):
        return self.kcov(x1, x2, *(self.khyp))

    def kuu(self, u1, u2):
        return self.kcov(self._u2x(u1), self._u2x(u2), *(self.khyp))

    def _calc_base_variance_integral(self):
        """
        Calculates the integral
            int_dx int_dx' k(x,x') * p(x) * p(x')
        that forms the baseline variance estimate for a BQ filter.
        """
        if False:
            V0, V0_err = 0.101403302569, 1.83139792831e-06
            self.Vn = self.V0 = V0
            return
        # Define a lot of throwaway functions to wrap parts of the problem
        # to match requirements for scipy.integrate.quad.  Rescale x-axis
        # to improve convergence of the integral.
        g = lambda const: (lambda x: const)
        integ_u = lambda u1, u2: self.kuu(u1, u2) * self.pu(u1) * self.pu(u2)
        # Run dblquad -- this should take about 1 min to complete
        self._vmsg("_calc_base_variance_integral:  Calculating...")
        V0, V0_err = integrate.dblquad(
                integ_u, self._ulo, self._uhi, g(self._ulo), g(self._uhi))
        self.Vn = self.V0 = V0
        self._vmsg("_calc_base_variance_integral: V0 = {} +/- {}"
                   .format(V0, V0_err))
        self._vmsg("_calc_base_variance_integral: V0[xval] = {}"
                   .format(V0 * (self._xsig * self.Zu)**2))

    def add_one_point(self):
        """
        Runs optimization for adding a single point to the BQ filter.
        """
        Ktmp = np.array(self.K)
        def u_var(u_n):
            # Wrapper functions for integrands
            integ_u = lambda u: self.kuu(u, u_n) * self.pu(u)
            _zn = integrate.quad(integ_u, self._ulo, self._uhi)[0]
            # Update internal state
            self.u[-1] = u_n
            self.zu[-1] = _zn
            kuu_n = self.kuu(self.u, u_n)
            Ktmp[-1,:] = Ktmp[:,-1] = kuu_n
            # Calculate and return variance = V0 - z.T * inv(K) * z
            self.Kchol = linalg.cholesky(Ktmp, lower=True)
            zeta = linalg.solve_triangular(self.Kchol, self.zu, lower=True)
            self.Vn = self.V0 - np.dot(zeta, zeta)
            return self.Vn
        # Enlarge internal state and optimize over location of new point
        # Since doing this in u, initial guess for new point is 0.0
        n = len(self.u)
        KX = np.atleast_2d([self.u])
        self.u = np.concatenate([self.u, [0.0]])
        self.zu = np.concatenate([self.zu, [0.0]])
        if Ktmp.shape[1] == 0:
            Ktmp = np.array([[1.0]])
        else:
            Ktmp = np.vstack([np.hstack([self.K, KX.T   ]),
                              np.hstack([KX,     [[1.0]]])])
        Ktmp += 1e-8 * np.eye(n+1)
        # As we add more points the Cholesky factor may become more unstable,
        # so add a small nugget -- as small as we can get away with.
        nugget = 1e-07
        self._vmsg("add_one_point:  Optimizing over point #{}...".format(n+1))
        while nugget <= 0.1:
            # try:
                u0 = np.array([0.0])
                bounds = np.array([(self._ulo, self._uhi)])
                cons = [{ 'type': 'ineq', 'fun': lambda u: u - self._ulo },
                        { 'type': 'ineq', 'fun': lambda u: self._uhi - u }]
                result = optimize.minimize(
                        u_var, u0, method='COBYLA', constraints=cons)
                break
            # except Exception as e:
                self._vmsg("add_one_point:  Cholesky factorization failed")
                self._vmsg(str(e))
                self._vmsg("add_one_point:  adding nugget = {} "
                           "to stabilize Cholesky factor".format(nugget))
                Ktmp += nugget * np.eye(n+1)
                nugget *= 10
        if nugget > 0.1:
            self._vmsg("add_one_point:  total Cholesky factorization fail")
            self.u, self.zu = self.u[:-1], self.zu[:-1]
            return
        elif result.success:
            self._vmsg("add_one_point:  Added new point (wt) {} ({}); Vn = {}"
                       .format(self.u[-1], self.zu[-1], self.Vn))
            self.K = Ktmp
        else:
            self._vmsg("add_one_point:  Optimization failed, don't trust me!")
            self._vmsg("Failure message:  {}".format(result.message))
        self.wbq = linalg.cho_solve((self.Kchol, True), self.zu)
        # Return quadrature points and weights for integration transformed
        # back to original x-axis, along with the renormalized variance.
        self.x = self._u2x(self.u)
        self.zx = self.zu * self._xsig * self.Zu
        self.Vxn = self.Vn * (self._xsig * self.Zu)**2

    def add_n_points(self, n=0):
        """
        What it says on the tin:  runs self.add_one_point() n times.
        """
        for i in range(n):
            self.add_one_point()

    def solve_n_points(self, n=0):
        """
        Runs ab initio optimization for an n-point Bayesian quadrature.
        """
        print "self._ulo, self._uhi =", self._ulo, self._uhi
        def u_var(uvec):
            self.u = np.array(uvec)
            self.zu = np.zeros(len(uvec))
            for i, ui in enumerate(uvec):
                integ_u = lambda u: self.kuu(u, ui) * self.pu(u)
                self.zu[i] = integrate.quad(integ_u, self._ulo, self._uhi)[0]
            Ktmp = self.kuu(self.u[:,np.newaxis], self.u[np.newaxis,:])
            Ktmp += 1e-12*np.eye(len(self.u))
            # Calculate and return variance = V0 - z.T * inv(K) * z
            self.Kchol = linalg.cholesky(Ktmp, lower=True)
            zeta = linalg.solve_triangular(self.Kchol, self.zu, lower=True)
            self.Vn = self.V0 - np.dot(zeta, zeta)
            self._vmsg("*** u_var:  uvec =" +
                       ("{:.3f} " * len(uvec)).format(*uvec) +
                       "Vn = {}".format(self.Vn))
            return self.Vn
        # Optimize all the points
        u0 = np.linspace(self._ulo, self._uhi, n+2)[1:-1]
        bounds = np.array([(self._ulo, self._uhi)])
        cons =  [{ 'type': 'ineq', 'fun': lambda u: u[i] - self._ulo }
                 for i in range(n)]
        cons += [{ 'type': 'ineq', 'fun': lambda u: self._uhi - u[i] }
                 for i in range(n)]
        result = optimize.minimize(
                u_var, u0, method='COBYLA', constraints=cons)
        self.wbq = linalg.cho_solve((self.Kchol, True), self.zu)
        # Return quadrature points and weights for integration transformed
        # back to original x-axis, along with the renormalized variance.
        self.x = self._u2x(self.u)
        self.zx = self.zu * self._xsig * self.Zu
        self.Vxn = self.Vn * (self._xsig * self.Zu)**2

    def int_quadz(self, f):
        """
        Uses straight-up quadrature to evaluate integral of f.  In most
        interesting cases f will be an interpolate.interp1d over some
        set of points (for example, an observed supernova spectrum).
        Parameters:
            f: 1-D callable
        """
        integ_u = lambda u: f(self._u2x(u)) * self.pu(u)
        pnorm = self._xsig * self.Zu
        Fu, Fu_err = integrate.quad(integ_u, self._ulo, self._uhi)
        Fx, Fx_err = Fu * pnorm, Fu_err * pnorm
        self._vmsg('int_quadz: F = {} +/- {}'.format(Fx, Fx_err))
        return Fx

    def int_bayes(self, f):
        """
        Uses Bayesian quadrature rule to evaluate integral of f.  The rule
        is derived assuming f is a Gaussian process with a given covariance
        kernel (i.e. fixed hyperparameters).
        Parameters:
            f: 1-D callable
        """
        pnorm = self._xsig * self.Zu
        Fx = np.dot(self.wbq, f(self.x)) * pnorm
        self._vmsg('int_bayes: F = {}'.format(Fx))
        return Fx

def sqexp(x1, x2, l):
    """
    Kernel for the GP, in this case an isotropic square exponential
    """
    return np.exp(-0.5*((x1-x2)/l)**2)

def sqlogexp(x1, x2, logl):
    """
    GP kernel, square exponential in log of variable
    """
    return np.exp(-0.5*((np.log(x1)-np.log(x2))/logl)**2)

def compress_filter(fname, kcov, khyp, n_points):
    """
    Reads in a transfer curve for a filter, and computes an optimal
    Bayesian quadrature rule for a square exponential covariance kernel.
    Parameters:
        fname:  name of two-column text file with (x, y) pairs
        kcov:   covariance kernel for GP integrand (callable),
                defined to take two arguments x1, x2
        khyp:  (fixed) hyperparameters for kcov (np.array of floats)
        n_points:  number of quadrature points desired
    """
    _x, _fx = np.loadtxt(fname, unpack=True)
    bquad = BQFilter(_x, _fx, kcov, khyp, verbose=True)
    bquad.add_n_points(n_points)
    return bquad

def integrate_test_suite(bquad):
    """
    Uses Bayesian quadrature to integrate a bunch of spectra, and compares
    with results from integrating straight against filter.
    """
    fquadz, fbayes = [ ], [ ]
    for fn in glob.glob("testdata/spec*.txt"):
        _x, _fx = np.loadtxt(fn, unpack=True)
        f = interpolate.interp1d(_x, _fx)
        print "Integrating", fn
        try:
            fquadz.append(bquad.int_quadz(f))
            fbayes.append(bquad.int_bayes(f))
        except Exception as e:
            print "...failed:", e
    delta_f = np.array(fbayes)/np.array(fquadz)
    print "bayes/quadz ratio over test data = {:.3f} +/- {:.3f}".format(
            np.mean(delta_f, axis=0), np.std(delta_f, axis=0))

def test_compress_filter():
    """
    Tests against a given dataset
    """
    for filt in ['u', 'B', 'V_9844', 'g', 'r', 'i']:
        filtfname = "CSP_filter_curves/CSP_{}.txt".format(filt)
        pklfname = filtfname.replace('.txt', '_bquad.pkl')
        print "*** Compressing: {} ***".format(filtfname)
        bquad = compress_filter(filtfname, sqlogexp, [0.01], 30)
        print "Writing to", pklfname
        with open(pklfname, 'w') as pklfile:
            pickle.dump(bquad, pklfile, -1)
        integrate_test_suite(bquad)


if __name__ == "__main__":
    test_compress_filter()
