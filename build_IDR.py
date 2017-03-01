#!/usr/bin/env python

"""
RS 2017/03/01:  Simple target class to read in and hold JLA data

This reads in data files on type Ia supernova light curves from
    "Improved cosmological constraints from a joint analysis
     of the SDSS-II and SNLS supernova samples",
        M. Betoule et al., A&A 568, A22 (2014).
I call it an "IDR" for Internal/Importable Data Release, that is, one in
which the light curve format of the original data has been recast in terms
of Python objects the rest of my code can understand.
"""

import os
import re
import glob
import numpy as np
import dill as pickle
import StringIO as StringIO


def _snoopy_filtname(snls_filtname):
    """
    Parse SNLS filter name into SNooPy equivalent
    """
    insthash = { 'MEGACAMPSF': '_m', '4SHOOTER2': 'sh', 'SDSS': '_s',
                 'SWOPE2': '', 'KEPLERCAM': 'k1', 'STANDARD': 'bess', }
    inst, filt = re.search('(\S+)::(\S+)', snls_filtname).groups()
    return "{}{}".format(filt, insthash[inst])


class JLA_SN(object):
    """
    Holds data and metadata for a single JLA SN.  Really just a struct.
    """

    def __init__(self, fname):
        """
        Initializes from a text file with the given filename.
        Parameters:
            fname:   name of the JLA SALT-format light curve file

        NB:  right now some elements of the IDR aren't in the JLA light curve
        files, for example the RA and DEC of SDSS targets or the DayMax of
        nearby targets.  The HST NICMOS candidates will die completely since
        NICMOS is not a proper filter setup in snpy by default.  Let's worry
        about all this if/when we try to fit these SNe using snpy.
        """
        # Parse the light curve file:  preamble header values marked with @,
        # rest of file in multi-column text format
        hdrstr, bodystr = "", ""
        with open(fname) as infile:
            for line in infile:
                if line[0] == '@':
                    hdrstr += line
                else:
                    bodystr += line
        inbuf = StringIO.StringIO(bodystr)
        data = np.genfromtxt(inbuf, dtype=None)
        H = { kw: val for kw, val in re.findall('@(\S+) (\S+)', hdrstr) }
        # Assign global attributes of this SN
        # Known missing keywords:  "RA", "DEC" in SDSS-*, "DayMax" in SN*;
        # these will need to be filled from other sources.
        self.survey = H.get('SURVEY', 'None')
        self.id = H.get('SN', 'None')
        self.ra = np.float(H.get('RA', 0.0))
        self.dec = np.float(H.get('DEC', 0.0))
        self.zcmb = np.float(H.get('Redshift', 0) or H.get('Z_CMB', 0))
        self.zhelio = np.float(H.get('Z_HELIO', 0.0))
        self.mwebv = np.float(H.get('MWEBV', 0.0))
        self.mjdmax = np.float(H.get('DayMax', 0.0))
        if self.survey in ['SDSS', 'SNLS']:
            self.name = "{}-{}".format(self.survey, self.id)
        else:
            self.name = self.id
        # Read in light curve data.  Don't bother sorting into individual
        # light curves, these are all just going in the big GP hopper anyway.
        # Re-evaluate this if later on we want to fit things with SNooPy.
        for i, attr in enumerate(['mjd', 'mag', 'magerr', 'zp']):
            setattr(self, attr, np.array([d[i] for d in data], dtype='float'))
        self.phase = (self.mjd - self.mjdmax)/(1.0 + self.zhelio)
        self.jlafilt = np.array([d[-2] for d in data], dtype='str')
        self.snpyfilt = np.array([_snoopy_filtname(fn) for fn in self.jlafilt])

    def dump_snpy_file(self, fname):
        """
        Writes SN information into file readable by snpy.get_sn().
        """
        outstr = "{} {} {} {}\n".format(
                self.name, self.zhelio, self.ra, self.dec)
        for filt in np.unique(self.snpyfilt):
            outstr += "filter {}\n".format(filt)
            idx = (self.snpyfilt == filt)
            for ti, mi, mierr in zip(
                    self.mjd[idx], self.mag[idx], self.magerr[idx]):
                outstr += "{:10.3f} {:.3f} {:.3f}\n".format(ti, mi, mierr)
        with open(fname, 'w') as outfile:
            outfile.write(outstr)

def cosmo_par_hash(cosmo_par_fname):
    """
    Reads light curve parameters from the JLA cosmology release, most of
    which are not part of the light curve files and need to be merged in.
    """
    # Pull the row data out -- this is easy
    colnames = ['cosmo_id', 'zcmb', 'zhelio', 'z_err', 'MB', 'MB_err',
                'x1', 'x1_err', 'c', 'c_err', 'logMhost', 'logMhost_err',
                'cov_MB_x1', 'cov_MB_c', 'cov_x1_c', 'set']
    lcfitrows = np.genfromtxt(cosmo_par_fname, dtype=None, names=colnames)
    # Now the name needs to be re-formatted, because it isn't consistent
    # with what appears in the light curve files
    def wrangle_name(id):
        mm = re.search("SDSS(\S+)", id)
        if mm:                                          # SDSS target
            return "SDSS-{}.0".format(mm.group(1))
        elif id[0] == '0':                              # SNLS target
            return "SNLS-{}".format(id)
        elif re.search("sn", id):                       # nearby target
            return id
        else:                                           # distant target
            return id
    # Return a hash of light curve parameters
    return { wrangle_name(r['cosmo_id']): r for r in lcfitrows }


def dump_idr(pklfname):
    """
    Dumps an IDR for all JLA SNe Ia into a dill pickle (filename = fname).
    """
    targets = { }
    snpy_dir = "IDR/snpy_phot"
    if not os.path.exists(snpy_dir):
        os.path.mkdir(snpy_dir)
    # Read in the light curve fit results
    lcfitpars = cosmo_par_hash("IDR/jla_cosmo_v1/data/jla_lcparams.txt")
    fill_attrs = ('x1', 'x1_err', 'c', 'c_err')
    # Read in the individual light curve files
    for fname in glob.glob("IDR/jla_light_curves/lc-*.list"):
        try:
            print "Reading {}".format(fname),
            sn = JLA_SN(fname)
            snpy_fname = "{}/{}_phot.txt".format(snpy_dir, sn.name)
            for attr in fill_attrs:
                setattr(sn, attr, lcfitpars[sn.name][attr])
            print "and dumping snpy light curve to", snpy_fname
            sn.dump_snpy_file(snpy_fname)
            targets[sn.name] = sn
        except Exception as e:
            print "...threw exception:"
            print "-->", e
    with open(pklfname, 'w') as pklfile:
        pickle.dump(targets, pklfile, -1)

if __name__ == "__main__":
    dump_idr('IDR/JLA_IDR.pkl')
