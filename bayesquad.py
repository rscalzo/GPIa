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
        def u_var(u_n):
            # Wrapper functions for integrands
            integ_u = lambda u: self.kuu(u, u_n) * self.pu(u)
            _zn = lambda u: integrate.quad(integ_u, self._ulo, self._uhi)[0]
            # Update internal state
            self.u[-1] = u_n
            self.zu[-1] = _zn(u_n)
            kuu_n = self.kuu(self.u, u_n)
            self.K[-1,:] = self.K[:,-1] = kuu_n
            # Calculate and return variance = V0 - z.T * inv(K) * z
            self.Kchol = linalg.cholesky(self.K, lower=True)
            zeta = linalg.solve_triangular(self.Kchol, self.zu, lower=True)
            self.Vn = self.V0 - np.dot(zeta, zeta)
            return self.Vn
        # Enlarge internal state and optimize over location of new point
        # Since doing this in u, initial guess for new point is 0.0
        n = len(self.u)
        KX = np.atleast_2d([self.u])
        self.u = np.concatenate([self.u, [0.0]])
        self.zu = np.concatenate([self.zu, [0.0]])
        if self.K.shape[1] == 0:
            self.K = np.array([[1.0]])
        else:
            self.K = np.vstack([np.hstack([self.K, KX.T   ]),
                                np.hstack([KX,     [[1.0]]])])
        self._vmsg("add_one_point:  Optimizing over point #{}...".format(n+1))
        result = optimize.minimize(u_var, np.array([0.0]),
                                   bounds=np.array([(self._ulo, self._uhi)]))
        if result.success:
            self._vmsg("add_one_point:  Added new point (wt) {} ({}); Vn = {}"
                       .format(self.u[-1], self.zu[-1], self.Vn))
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
        Fu, Fu_err = lambda u: integrate.quad(integ_u, self._ulo, self._uhi)
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
        Fx = np.dot(self.wbq, f(self._x)) * self.pnorm
        self._vmsg('int_quadz: F = {}'.format(Fx))
        return Fx


def sqexp(x1, x2, l):
    """
    Kernel for the GP, in this case an isotropic square exponential
    """
    return np.exp(-0.5*((x1-x2)/l)**2)

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

def test_compress_filter():
    """
    Tests against a given dataset
    """
    pklfname = "bquad_test.pkl"
    try:
        with open(pklfname) as pklfile:
            bquad = pickle.load(pklfile)
    except:
        print "Regenerating", pklfname, "from scratch"
        bquad = compress_filter('CSP_B.txt', sqexp, [50.0], 20)
        with open(pklfname, 'w') as pklfile:
            pickle.dump(bquad, pklfile, -1)


if __name__ == "__main__":
    test_compress_filter()
