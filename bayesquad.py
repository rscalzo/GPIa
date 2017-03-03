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

def gconst(const):
    """
    Generates function that returns the specified constant.
    (Used in scipy.integrate.dblquad for integration bounds.)
    """
    return lambda x: const

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
    # with the training points.  One annoying design problem with this
    # class is the need for internal lambda functions:  the quadrature
    # schemes in scipy.integrate require a strict function prototype,
    # but the integrands generally require knowledge of internal state
    # apart from the function arguments.  I don't 

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

    def _kuu(self, u1, u2):
        return self.kcov(self._u2x(u1), self._u2x(u2), *(self.khyp))

    def _kuup(self, u1, u2):
        return self._kuu(u1, u2) * self.pu(u1)

    def _kuupp(self, u1, u2):
        return self._kuu(u1, u2) * self.pu(u1) * self.pu(u2)

    def _calc_base_variance_integral(self):
        """
        Calculates the integral
            V0 = int_dx int_dx' k(x,x') * p(x) * p(x')
        that forms the baseline variance estimate for a BQ filter.
        """
        # Run dblquad -- this should take about 1 min to complete
        self._vmsg("_calc_base_variance_integral:  Calculating...")
        V0, V0_err = integrate.dblquad(self._kuupp, self._ulo, self._uhi,
                                       gconst(self._ulo), gconst(self._uhi))
        self.Vn = self.V0 = V0
        self._vmsg("_calc_base_variance_integral: V0 = {} +/- {}"
                   .format(V0, V0_err))
        self._vmsg("_calc_base_variance_integral: V0[xval] = {}"
                   .format(V0 * (self._xsig * self.Zu)**2))

    def Vtot(self):
        """
        Calculates the variance of the n-point Bayesian quadrature scheme:
            Vn = V0 - z.T * inv(K) * z
        where V0 is the base variance (see above), K is the covariance matrix
        the training points, and z is the integral of the covariance kernel
        against the base measure (in our case, the filter transmission).
        Assumes the covariance K and weights z have already been calculated.
        As a side effect, updates the cached Cholesky factor of K.
        """
        self.Kchol = linalg.cholesky(self.K, lower=True)
        zeta = linalg.solve_triangular(self.Kchol, self.zu, lower=True)
        return self.V0 - np.dot(zeta, zeta)

    def Vtot_n(self, u_n):
        """
        In the context of the greedy optimization of a Bayesian quadrature
        scheme, this function wraps self.Vtot() and makes it a function of
        the location of the last point added (the one being optimized over).
        As a side effect, updates the internal state of the class instance,
        including u, zu, K, and its Cholesky factor Kchol.
        """
        z_n = integrate.quad(self._kuup, self._ulo, self._uhi, args=(u_n))[0]
        self.u[-1], self.zu[-1] = u_n, z_n
        self.K[-1,:] = self.K[:,-1] = self._kuu(self.u, u_n)
        self.Vn = self.Vtot()
        return self.Vn

    def Vtot_all(self, uvec):
        """
        In the context of brute-force optimization of a Bayesian quadrature
        scheme, this function wraps self.Vtot() and makes it a function of
        the location of all quadrature points, as a vector to optimize.
        As a side effect, updates the internal state of the class instance,
        including u, zu, K, and its Cholesky factor Kchol.
        """
        self.u, self.zu = np.array(uvec), np.zeros(len(uvec))
        for i, ui in enumerate(uvec):
            self.zu[i] = integrate.quad(
                    self._kuup, self._ulo, self._uhi, args=(ui))[0]
        self.K = self._kuu(self.u[:,np.newaxis], self.u[np.newaxis,:])
        self.K += 1e-12*np.eye(len(self.u))
        self.Vn = self.Vtot()
        uvec_str = ("{:.3f} " * len(uvec)).format(*uvec)
        self._vmsg("*** u_var:  uvec = [{}], Vn = {}".format(uvec_str, self.Vn))
        return self.Vn

    def add_one_point(self):
        """
        Runs optimization for adding a single point to the BQ filter.
        """
        # Enlarge internal state and optimize over location of new point
        # Since doing this in u, initial guess for new point is 0.0
        self.u = np.concatenate([self.u, [0.0]])
        self.zu = np.concatenate([self.zu, [0.0]])
        n = len(self.u)
        Ktmp = np.eye(n)
        Ktmp[:-1,:-1] = self.K
        self.K = Ktmp
        # Use COBYLA for minimization; it seems to work well
        self._vmsg("add_one_point:  Optimizing over point #{}...".format(n))
        try:
            cons = [{ 'type': 'ineq', 'fun': lambda u: u - self._ulo },
                    { 'type': 'ineq', 'fun': lambda u: self._uhi - u }]
            result = optimize.minimize(
                    self.Vtot_n, [0.0], method='COBYLA', constraints=cons)
            cobyla_except = False
        except Exception as e:
            self._vmsg("add_one_point:  exception caught during optimization")
            self._vmsg(str(e))
            cobyla_except = True
        if cobyla_except or not result.success:
            # If we died, back out the changes to the internal state and bail
            self._vmsg("add_one_point:  Optimization failed, don't trust me!")
            if not cobyla_except:
                self._vmsg("optimize.minimize fail message: " + result.message)
            self.u, self.zu = self.u[:-1], self.zu[:-1]
            self.K = self.K[:-1,:-1]
        else:
            # Calculate quadrature weights and transform them back to the
            # original x-axis as a convenience for the user.
            self._vmsg("add_one_point:  Added new point (zu) {} ({}); Vn = {}"
                       .format(self.u[-1], self.zu[-1], self.Vn))
            self.wbq_u = linalg.cho_solve((self.Kchol, True), self.zu)
            self.wbq_x = self.wbq_u * self._xsig * self.Zu
            self.x = self._u2x(self.u)
            self.zx = self.zu * self._xsig * self.Zu

    def add_n_points(self, n=0):
        """
        What it says on the tin:  runs self.add_one_point() n times.
        This is the recommended method for most base measures.
        """
        for i in range(n):
            self.add_one_point()

    def solve_n_points(self, n=0):
        """
        Runs ab initio optimization for an n-point Bayesian quadrature,
        treating all quadrature point locations as a vector to optimize over.
        NB:  this takes a LONG time to run and is not obviously better on a
        practical basis than the greedy algorithm BQFilter.add_n_points(),
        so we strongly recommend the former.
        """
        # Set up an initial guess with points spread out across the support
        # of the base measure, and constraints to stay in that support.
        u0 = np.linspace(self._ulo, self._uhi, n+2)[1:-1]
        cons =  [{ 'type': 'ineq', 'fun': lambda u: u[i] - self._ulo }
                 for i in range(n)]
        cons += [{ 'type': 'ineq', 'fun': lambda u: self._uhi - u[i] }
                 for i in range(n)]
        try:
            result = optimize.minimize(
                    self.Vtot_all, u0, method='COBYLA', constraints=cons)
            cobyla_except = False
        except Exception as e:
            self._vmsg("solve_n_points:  minimization failed")
            self._vmsg(str(e))
            epic_fail = True
        if cobyla_except or not result.success:
            # If we died, report that and bail
            self._vmsg("solve_n_points:  Optimization failed, don't trust me!")
            self._vmsg("optimize.minimize failure message: " + result.message)
        else:
            # Calculate quadrature weights and transform them back to the
            # original x-axis as a convenience for the user.
            self._vmsg("solve_n_points:  Found {} points w/ Vn = {}"
                       .format(len(self.u), self.Vn))
            self._vmsg("quadrature points = {}".format(self.u))
            self.wbq_u = linalg.cho_solve((self.Kchol, True), self.zu)
            self.wbq_x = self.wbq_u * self._xsig * self.Zu
            self.x = self._u2x(self.u)
            self.zx = self.zu * self._xsig * self.Zu

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
        Fx = np.dot(self.wbq_x, f(self.x))
        self._vmsg('int_bayes: F = {}'.format(Fx))
        return Fx

def sqexp(x1, x2, l):
    """
    GP kernel, in this case an isotropic square exponential.
    Parameters:
        x1, x2:  floats or compatible np.ndarrays
        l:       variation scale(s); units, shape compatible with x1 & x2
    """
    return np.exp(-0.5*((x1-x2)/l)**2)

def sqlogexp(x1, x2, logl):
    """
    GP kernel, square exponential in log of variable.  This is useful in
    the case where the function being integrated is a SN Ia spectrum,
    since its variations have a characteristic velocity scale dv = dl/l
    rather than a wavelength scale dl.
    Parameters:
        x1, x2:  strictly *positive* floats or compatible np.ndarrays
        logl:    variation scale(s); units, shape compatible with x1 & x2
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
