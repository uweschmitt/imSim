"""
Kolmogorov and Gaussian PSF using LSST System Engineering modeling.
"""
import numpy as np
import galsim
from galsim.config import RegisterObjectType


def Kolmogorov_and_Gaussian_PSF(airmass, rawSeeing, band, gsparams=None):
    """
    This PSF is based on David Kirkby's presentation to the DESC
    Survey Simulations working group on 23 March 2017.

    https://confluence.slac.stanford.edu/pages/viewpage.action?spaceKey=LSSTDESC&title=SSim+2017-03-23

    (you will need a SLAC Confluence account to access that link)

    Parameters
    ----------
    airmass

    rawSeeing is the FWHM seeing at zenith at 500 nm in arc seconds
    (provided by OpSim)

    band is the bandpass of the observation [u,g,r,i,z,y]
    """
    # This code was provided by David Kirkby in a private communication
    wlen_eff = dict(u=365.49, g=480.03, r=622.20, i=754.06,
                    z=868.21, y=991.66)[band]
    # wlen_eff is from Table 2 of LSE-40 (y=y2)

    FWHMatm = rawSeeing * (wlen_eff / 500.) ** -0.3 * airmass ** 0.6
    # From LSST-20160 eqn (4.1)

    FWHMsys = np.sqrt(0.25**2 + 0.3**2 + 0.08**2) * airmass ** 0.6
    # From LSST-20160 eqn (4.2)

    atm = galsim.Kolmogorov(fwhm=FWHMatm, gsparams=gsparams)
    sys = galsim.Gaussian(fwhm=FWHMsys, gsparams=gsparams)
    return galsim.Convolve((atm, sys))


def Build_KG_PSF(config, base, ingore, gsparams, logger):
    """Build a Kolmogorov_and_Gaussian_PSF from the OpsimMeta information."""
    md = galsim.config.GetInputObj('opsim_meta_dict', config, base, 'OpsimMeta').meta
    band = 'ugrizy'[md['filter']]
    psf = Kolmogorov_and_Gaussian_PSF(md['airmass'], md['seeing'], band,
                                      gsparams=galsim.GSParams(**gsparams) if gsparams else None)
    return psf, False


RegisterObjectType('Kolmogorov_and_Gaussian_PSF', Build_KG_PSF, input_type='opsim_meta_dict')
