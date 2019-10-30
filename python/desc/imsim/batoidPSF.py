import numpy as np

import batoid
import galsim

from lsst.sims.GalSimInterface import PSFbase


class BatoidGSObject(galsim.GSObject):
    _has_hard_edges=False
    def __init__(self, telescope, theta_x, theta_y, wavelength):
        self.telescope = telescope
        self.theta_x = theta_x
        self.theta_y = theta_y
        self.wavelength = wavelength
        self._flux = 1.0
        self._gsparams = galsim.GSParams()
        airy = galsim.Airy(lam=wavelength*1e9, diam=8.36, obscuration=0.61)
        self._stepk = airy.stepk
        self._maxk = airy.maxk

    def _shoot(self, photons, rng):
        n_photons = len(photons)
        rays = batoid.RayVector.asPolar(
            optic=self.telescope,
            wavelength=self.wavelength,
            theta_x=self.theta_x, theta_y=self.theta_y,
            nrandom=n_photons
        )
        self.telescope.traceInPlace(rays)
        # Compute positions wrt chief ray
        chiefRay = batoid.Ray.fromStop(
            0.0, 0.0,
            optic=self.telescope,
            wavelength=self.wavelength,
            theta_x=self.theta_x, theta_y=self.theta_y,
        )
        self.telescope.traceInPlace(chiefRay)

        # Need precise pixel scale going forward?
        photons.x = (rays.x - chiefRay.x)/10e-6*0.2
        photons.y = (rays.y - chiefRay.y)/10e-6*0.2
        photons.flux = 1./n_photons*(~rays.vignetted)
        return photons


class BatoidPSF(PSFbase):
    def __init__(self, telescope, wavelength, atmPSF=None):
        self.telescope = telescope
        self.wavelength = wavelength
        self.atmPSF = atmPSF

    def _getPSF(self, xPupil=None, yPupil=None, gsparams=None):
        theta = (xPupil*galsim.arcsec, yPupil*galsim.arcsec)
        apsf = self.atmPSF.atm.makePSF(
            self.atmPSF.wlen_eff,
            aper=self.atmPSF.aper,
            theta=theta,
            t0=self.atmPSF.t0,
            exptime=self.atmPSF.exptime,
            gsparams=gsparams
        )
        if self.atmPSF.gaussianFWHM > 0.0:
            apsf = galsim.Convolve(
                galsim.Gaussian(fwhm=self.atmPSF.gaussianFWHM, gsparams=gsparams),
                apsf
            )
        theta_x = np.deg2rad(xPupil/3600)
        theta_y = np.deg2rad(yPupil/3600)
        bpsf = BatoidGSObject(self.telescope, theta_x, theta_y, self.wavelength)
        psf = galsim.Convolve(
            apsf,
            bpsf
        )
        return psf
