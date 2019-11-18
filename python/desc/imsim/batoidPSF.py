import numpy as np

import batoid
import galsim

from lsst.sims.GalSimInterface import PSFbase


class BatoidGSObject(galsim.GSObject):
    _has_hard_edges=False
    def __init__(
        self, telescope, theta_x, theta_y, wavelength, parallactic_angle,
        _optics_magnification=1
    ):
        self.telescope = telescope
        self.theta_x = theta_x
        self.theta_y = theta_y
        self.wavelength = wavelength
        self.parallactic_angle = parallactic_angle
        self._optics_magnification = _optics_magnification

        # Some attributes to make GalSim happy...
        self._flux = 1.0
        self._gsparams = galsim.GSParams()
        airy = galsim.Airy(lam=wavelength*1e9, diam=8.36, obscuration=0.61)
        self._stepk = airy.stepk
        self._maxk = airy.maxk

    def _shoot(self, photons, rng):
        n_photons = len(photons)
        rays = batoid.RayVector.asPolar(
            optic=self.telescope,
            # Don't simulate photons closer than 2m to entrance pupil axis
            inner=2.0,
            wavelength=self.wavelength,
            theta_x=self.theta_x, theta_y=self.theta_y,
            projection='gnomonic',
            nrandom=n_photons
        )
        self.telescope.traceInPlace(rays)
        # We'll return ray positions w.r.t. the chief ray, so simulate that.
        chiefRay = batoid.Ray.fromStop(
            0.0, 0.0,  # chief ray => stop position (0.0, 0.0)
            optic=self.telescope,
            wavelength=self.wavelength,
            theta_x=self.theta_x, theta_y=self.theta_y,
            projection='gnomonic'
        )
        self.telescope.traceInPlace(chiefRay)

        xy = np.array([rays.x-chiefRay.x, rays.y-chiefRay.y])
        # Above gives position in meters.  Once photon-shooting in GalSim is
        # refactored this could be the end point.

        # Currently, however, GalSim  expects photon positions in arcseconds on
        # a tangent plane with +y pointing North and +x pointing W.

        # First step is to convert from focal plane coords to batoid field angle
        # coords (which are aligned with alt and az).  batoid.psf.dthdr is
        # exactly the required jacobian.
        jac = batoid.psf.dthdr(
            self.telescope,
            theta_x=self.theta_x, theta_y=self.theta_y,
            projection='gnomonic',
            wavelength=self.wavelength,
            nx=6
        ) # radians/meter
        azalt = np.rad2deg(np.dot(jac, xy))*3600  # arcsec

        # Still need to rotate so +y points North and not +alt.  This is a
        # rotation by the parallactic angle.
        sq = np.sin(-self.parallactic_angle)
        cq = np.cos(-self.parallactic_angle)
        rot = np.array([[cq, -sq], [sq, cq]])

        photons.x, photons.y = np.dot(rot, azalt)
        # Still not quite done.  Need to multiply x by -1 to account for viewing
        # focal plane through the L3 window vs viewing from behind.
        photons.x *= -1
        # Set the fluxes to just 0 or 1 depending on the vignetting, but scaled
        # to the current object flux.
        photons.flux = (self._flux/n_photons)*(~rays.vignetted)

        # Magnify optics if requested
        photons.x *= self._optics_magnification
        photons.y *= self._optics_magnification
        # TODO: set photons dxdz, dydz here...
        return photons


class BatoidPSF(PSFbase):
    def __init__(
        self, telescope, band, atmPSF, rotTelPos,
        parallactic_angle,
        _field_magnification=1,  # Useful for debugging
        _optics_magnification=1  # Useful for debugging
    ):
        self.telescope = telescope
        self.wavelength = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)[band]
        self.atmPSF = atmPSF
        self.rotTelPos = rotTelPos
        self.parallactic_angle = parallactic_angle
        srTP = np.sin(rotTelPos)
        crTP = np.cos(rotTelPos)
        self._jac = np.array([[crTP, -srTP], [srTP, crTP]])
        self._field_magnification = _field_magnification
        self._optics_magnification = _optics_magnification

    def _getPSF(self, xPupil, yPupil, gsparams=None):
        # Convert lsst_sims pupil coordinates to batoid field_angle coordinates
        # (which are different that DM field_angle coordinates)
        thx, thy = -np.dot(self._jac, (xPupil, yPupil))*self._field_magnification

        apsf = self.atmPSF.atm.makePSF(
            self.atmPSF.wlen_eff,
            aper=self.atmPSF.aper,
            theta=(thx*galsim.arcsec, thy*galsim.arcsec),
            t0=self.atmPSF.t0,
            exptime=self.atmPSF.exptime,
            gsparams=gsparams
        )
        if self.atmPSF.gaussianFWHM > 0.0:
            apsf = galsim.Convolve(
                galsim.Gaussian(fwhm=self.atmPSF.gaussianFWHM, gsparams=gsparams),
                apsf
            )
        bpsf = BatoidGSObject(
            self.telescope,
            np.deg2rad(thx/3600),
            np.deg2rad(thy/3600),
            self.wavelength*1e-9,
            parallactic_angle=self.parallactic_angle,
            _optics_magnification=self._optics_magnification
        )
        return galsim.Convolve(apsf, bpsf)
