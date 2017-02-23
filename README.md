# GPIa

This is a project to infer type Ia supernova (SN Ia) spectrophotometric
templates based on filter photometry *only*.  It is partly an academic
exercise in Bayesian inference, but should also have concrete applications
in the era of wide-area transient surveys, such as LSST.

Context
-------

SNe Ia are used in astronomy to measure distances to faraway galaxies,
since they are luminous (as bright as the galaxy in which they appear)
and because their luminosity can be predicted with 10-15% precision.
They result from thermonuclear explosions of carbon/oxygen white dwarfs,
brightening and fading over timescales of a few weeks.
Distances are measured using the inverse square law, comparing the known
luminosity of a supernova with its apparent brightness as measured.
This technique enables the accurate measurement of the expansion history
of the Universe:  as the Universe expands by a scale factor _a_(_t_),
the wavelength of all light within it is redshifted by a factor
_z_ = 1/_a_ - 1.  Given that light travels at a constant speed _c_,
measuring redshift _z_ as a function of distance _d_ is thus equivalent
to measuring _a_ as a function of time _t = d/c_.  Measurements based on
SNe Ia were used to discover the accelerating expansion of the Universe,
which won the 2011 Nobel Prize in Physics.  The underlying cause of the
expansion (commonly called "dark energy") is still unknown, and SN Ia
cosmology is still a very active area of research.

Measurements of the dark energy density in the Universe require
observations from a large number of SNe Ia spread across a wide range
of redshifts.  Observations are usually made in optical wavelengths
using glass filters, which capture only the average flux within some
wavelength window at a specific point in time; a time series of these
points is called a _light curve_.  The spectrum of a SN Ia is complex,
with many broad atomic lines that also vary in time, so that the observed
brightness depends strongly on the redshift of the SN, the time since
explosion, and the filter used.  Combining observations of SNe Ia for
cosmology can therefore be done only using a full model of the
time-evolving spectral energy distribution, which forms the core of any
parametrized light curve fitting procedure.

The current gold standard for SN Ia light curve fitters, SALT2
([Guy et al. 2007] (http://adsabs.harvard.edu/abs/2007A%26A...466...11G),
 [2010] (http://adsabs.harvard.edu/abs/2010A%26A...523A...7G)),
uses a combination of photometric and spectroscopic data for training.
The photometry captures only average fluxes over intervals.  In principle
a spectrum can be integrated over a set of filter transmission curves to
produce photometric fluxes (_synthetic photometry_), but flux calibration
of spectra is usually highly uncertain, and so only the relative depths
of atomic line features are reliable.  Systematic errors therefore
creep in due to uncertain calibration of the spectra to match the filter
photometry, along with uncertainty in the photometry itself, and the need
for spectroscopic coverage in wavelength ranges that can be difficult or
expensive to obtain (e.g. infrared spectra, which requires long exposures
using large telescopes at very stable sites, and ultraviolet spectra,
which must be obtained from space).  SALT2 and other light curve fitters
are also currently formulated as linear models, but a great deal of rich
structure in the underlying high-dimensional distribution of the data
may remain untapped.

[Guy et al. 2007] (http://adsabs.harvard.edu/abs/2007A%26A...466...11G)
suggest in their paper the future possibility of deconvolving the full
SN Ia spectrophotometric time series.  Photometric datasets of SNe Ia
are now so extensive, and computing power sufficiently cheap and advanced,
that we can now contemplate doing this.  The advantages include:
* A more advanced model structure, using Gaussian processes in place
  of binned spectra and capturing full covariances of all observations
  rather than just mean square errors.
* A fully Bayesian treatment of all aspects of the model, including the
  potential to self-calibrate by treating the filter transmissions and
  calibration spectra as random variables.
* Removal of limitations based on spectroscopic coverage, and inference
  of mean spectroscopic behavior at all wavelengths for subclasses of
  SNe Ia with sufficient photometric coverage.
* Inference over latent parameters, giving a data-driven, low-dimensional
  summary of SN Ia spectroscopic behavior.  Complex latent relationships
  such as graphical models can be considered here, extending the current
  simple linear regression models used.  Such representations may also be
  used to mine SN Ia physics when compared with numerical simulations.
  
Model structure
---------------

The current plan is to use a Gaussian process
(GP; [Rasmussen & Williams 2006] (http://gaussianprocess.org/))
over wavelength and time to represent the (latent) spectral time series.
This latent time series is linked to the observations by transformations
of the GP prior ([Murray-Smith & Pearlmutter 2005]
                 (http://link.springer.com/chapter/10.1007/11559887_7))
corresponding to synthetic photometry; the filter transmission curves
can be compressed via Bayesian quadrature to make this more tractable
(O'Hagan 1991; [Huszar & Duvenaud 2012](https://arxiv.org/abs/1204.1664)).
Eventually the GP parameters themselves will admit additional hyperpriors
(such as a graphical model) to extract a compressed representation of the
full spectrophotometric time series.
