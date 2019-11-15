import lsst.afw.cameraGeom as cameraGeom
from lsst.obs.lsst import LsstCamMapper
import lsst.obs.lsst.cameraTransforms as cameraTransforms
import batoid


class BatoidWCS(galsim.wcs.CelestialWCS):
    """
    Parameters
    ----------
    boresight : galsim.CelestialCoord
    rotSkyPos : galsim.Angle
    rotTelPos : galsim.Angle
    fiducial_telescope : batoid.Optic
        Telescope before application of rotator
    wavelength : float
        Nanometers
    camera : lsst.afw.cameraGeom.camera.camera.Camera
    det : lsst.afw.cameraGeom.detector.detector.Detector
    """
    def __init__(
        self, boresight, rotSkyPos, rotTelPos, fiducial_telescope, wavelength,
        camera, det
    ):
        self.boresight = boresight
        self.rotSkyPos = rotSkyPos
        self.rotTelPos = rotTelPos
        self.fiducial_telescope = fiducial_telescope
        self.wavelength = wavelength
        self.camera = camera
        self.det = det

        # Initialize RaDec -> field
        q = rotTelPos - rotSkyPos
        cq, sq = np.cos(q), np.sin(q)  # +q or -q here?
        jac = [cq, -sq, sq, cq]
        affine = galsim.AffineTransform(*jac)
        self._radec2field = galsim.TanWCS(
            affine, boresight, units=galsim.radians
        )

        # Initialize field -> focal
        _pz = fiducial_telescope.stopSurface.surface.sag(0.0, 0.0)
        transform = batoid.CoordTransform(
            fiducial_telescope.stopSurface.coordSys, batoid.globalCoordSys
        )
        _, _, self._pz = transform.applyForward(0, 0, _pz)
        self._refractive_index = fiducial_telescope.inMedium.getN(
            wavelength*1e-9
        )
        self._telescope = fiducial_telescope.withLocallyRotatedOptic(
            "LSSTCamera", batoid.RotZ(-self.rotTelPos)
        )

        # Initialize focal -> pixel
        lct = cameraTransforms.LsstCameraTransforms(camera)
        center = 509*4 + 0.5, 2000.5
        # Focal plane coords of center of CCD
        _x, _y = lct.ccdPixelToFocalMm(*center, det.getName())
        self._focal_x = _x*1e-3 # meters
        self._focal_y = _y*1e-3 # meters

        # Also calculate field angle of CCD center
        self._field_x, self._field_y = self._focalToField(
            self._focal_x, self._focal_y
        )

        # Required by galsim.BaseWCS
        self._color = None
        self.origin = galsim.PositionD(0, 0)

    def _fieldToFocal(self, field_x, field_y):
        ray = batoid.Ray.fromStop(
            0.0, 0.0,
            optic=self._telescope,
            wavelength=self.wavelength*1e-9,
            theta_x=field_x, theta_y=field_y, projection='gnomonic'
        )
        self._telescope.traceInPlace(ray)
        return ray.x, ray.y

    def _fieldToFocalMany(self, field_x, field_y):
        # Setup rays to trace with batoid.  For each input coordinate, we'll
        # trace exactly 1 ray, the chief ray.  Any displacements between the
        # mean optics/atmosphere and the chief ray can then be considered
        # part of the PSF.
        n = len(field_x)
        dirCos = np.array(batoid.utils.fieldToDirCos(
            field_x, field_y, projection='gnomonic')
        )
        w = np.empty(n); w.fill(self.wavelength*1e-9)
        pz = np.empty(n); pz.fill(self._pz)
        rays = batoid.RayVector.fromArrays(
            # Chief ray -> start at center of the entrance pupil.
            np.zeros(n), np.zeros(n), pz,
            # velocity = dirCos adjusted for refractive index
            *(dirCos/self._refractive_index),
            # time is irrelevant, so just use 0.0
            np.zeros(n),
            w # wavelength
        )
        self._telescope.traceInPlace(rays)
        return rays.x, rays.y

    def _focalToField(self, x, y):
        from scipy.optimize import least_squares

        def _resid(args):
            fx1, fy1 = args
            x1, y1 = self._fieldToFocal(fx1, fy1)
            return np.array([x1-x, y1-y])

        result = least_squares(_resid, np.array([0.0, 0.0]))
        return result.x[0], result.x[1]

    def _focalToFieldMany(self, x, y):
        if not hasattr(self, '_xinterp'):
            from scipy.interpolate import RectBivariateSpline
            # radius of circle circumscribing CCD, plus some fudge
            half_length = 2000*10e-6*np.sqrt(2)*1.2
            N = 4  # seems to be sufficient, kinda surprising...
            focal_xs = self._focal_x + half_length*np.linspace(-1,1,N)
            focal_ys = self._focal_y + half_length*np.linspace(-1,1,N)
            field_xs = np.empty((N, N), dtype=float)
            field_ys = np.empty((N, N), dtype=float)
            for i in range(N):
                for j in range(N):
                    field_xs[i,j], field_ys[i,j] = self._focalToField(
                        focal_xs[i], focal_ys[j]
                    )
            self._xinterp = RectBivariateSpline(focal_xs, focal_ys, field_xs)
            self._yinterp = RectBivariateSpline(focal_xs, focal_ys, field_ys)
        return self._xinterp(x, y, grid=False), self._yinterp(x, y, grid=False)

    def _xy(self, ra, dec, color=None):
        """
        Parameters
        ----------
        ra : array_like
            Right ascension in radians
        dec : array_like
            Declination in radians

        Returns
        -------
        xpix, ypix : array_like
        """
        # Abuse GalSim's TanWCS to convert ra/dec into field angles
        # centered at ra/dec and aligned with alt/az.
        field_x, field_y = self._radec2field.radecToxy(ra, dec, 'rad')

        # Field -> Focal via batoid.
        focal_x, focal_y = self._fieldToFocalMany(field_x, field_y)

        # For the moment, focal -> pixel is just an offset and a scaling by the
        # pixel size.
        pixSize = 10e-6 # m
        pixel_x = (focal_x - self._focal_x)/pixSize + (509*4 + 0.5)
        pixel_y = (focal_y - self._focal_y)/pixSize + 2000.5
        return pixel_x, pixel_y

    def _radec(self, x, y, color=None):
        # reverse of _xy.

        # pixel -> focal
        pixSize = 10e-6
        focal_x = pixSize*(x-(509*4+0.5)) + self._focal_x
        focal_y = pixSize*(y-2000.5) + self._focal_y

        # focal -> field using interpolating function derived from field->focal
        field_x, field_y = self._focalToFieldMany(focal_x, focal_y)

        # field -> ra, dec using GalSim TanWCS
        return self._radec2field.xyToradec(field_x, field_y, units='rad')
