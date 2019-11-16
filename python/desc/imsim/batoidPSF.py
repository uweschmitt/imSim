import numpy as np

import batoid
import galsim

from lsst.sims.GalSimInterface import PSFbase


class BatoidGSObject(galsim.GSObject):
    _has_hard_edges=False
    def __init__(self, telescope, theta_x, theta_y, wavelength):
        self.telescope = telescope
        self.theta_x = theta_x*18
        self.theta_y = theta_y*18
        # self.theta_x = theta_x
        # self.theta_y = theta_y
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
            inner=2.0,
            wavelength=self.wavelength,
            theta_x=self.theta_x, theta_y=self.theta_y,
            projection='gnomonic',
            nrandom=n_photons
        )
        self.telescope.traceInPlace(rays)
        # Compute positions wrt chief ray
        chiefRay = batoid.Ray.fromStop(
            0.0, 0.0,
            optic=self.telescope,
            wavelength=self.wavelength,
            theta_x=self.theta_x, theta_y=self.theta_y,
            projection='gnomonic'
        )
        # print("self.theta = ", self.theta_x, self.theta_y)
        # print("initial chief ray")
        # print(chiefRay)
        self.telescope.traceInPlace(chiefRay)
        # print("final chief ray")
        # print(chiefRay)

        # Need precise pixel scale going forward?
        # px = (rays.x - chiefRay.x)/10e-6*0.2
        # py = (rays.y - chiefRay.y)/10e-6*0.2
        jac = batoid.psf.dthdr(
            self.telescope,
            theta_x=self.theta_x, theta_y=self.theta_y,
            projection='gnomonic',
            wavelength=self.wavelength,
            nx=6
        ) # rad/m
        # print("jac = ", jac)

        # import matplotlib.pyplot as plt
        # w = ~rays.vignetted
        # plt.scatter(
        #     (rays.x-chiefRay.x)[w],
        #     (rays.y-chiefRay.y)[w],
        #     s=1, alpha=0.01
        # )
        # plt.title(f"{self.theta_x} {self.theta_y}")
        # plt.xlabel("focal dx")
        # plt.ylabel("focal dy")
        # plt.show()

        # Get's pixel scale and rotates to be along zenith direction.
        xy = np.dot(
            jac,
            [rays.x-chiefRay.x, rays.y-chiefRay.y]
        )*206265*100
        # plt.scatter(
        #     xy[0][w], xy[1][w],
        #     s=1, alpha=0.01
        # )
        # plt.title(f"{self.theta_x} {self.theta_y}")
        # plt.xlabel("az")
        # plt.ylabel("alt")
        # plt.show()

        # Need to rotate to point North
        rTP = 0.6988060000275936
        rSP = 2.2904839273531232
        q = rSP - rTP
        q *= -1
        sq, cq = np.sin(q), np.cos(q)
        #
        # srSP = np.sin(np.deg2rad(131.2351))
        # crSP = np.cos(np.deg2rad(131.2351))
        # self.t = np.array([[crSP, -srSP], [srSP, crSP]])
        self.t = np.array([[cq, -sq], [sq, cq]])
        photons.x, photons.y = np.dot(self.t, xy)

        # plt.scatter(
        #     photons.x[w], photons.y[w],
        #     s=1, alpha=0.01
        # )
        # plt.title(f"{self.theta_x} {self.theta_y}")
        # plt.xlabel("E")
        # plt.ylabel("N")
        # plt.show()
        photons.x *= -1
        photons.flux = 1./n_photons*(~rays.vignetted)
        return photons


class BatoidPSF(PSFbase):
    def __init__(self, telescope, band, atmPSF, rotTelPos):
        self.telescope = telescope
        self.wavelength = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)[band]
        self.atmPSF = atmPSF
        self.rotTelPos = rotTelPos
        srTP = np.sin(rotTelPos)
        crTP = np.cos(rotTelPos)
        self._jac = np.array([[crTP, -srTP], [srTP, crTP]])
        # print("rotTelPos = ", self.rotTelPos)

    def _getPSF(self, xPupil, yPupil, gsparams=None):
        # print("pupil = ", (xPupil, yPupil))
        thx, thy = -np.dot(self._jac, (xPupil, yPupil))
        # print("theta = ", (thx, thy))

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
            self.wavelength*1e-9
        )
        psf = galsim.Convolve(
            apsf,
            bpsf
        )
        return psf
