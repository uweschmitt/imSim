"""
Module to map objects on the sky to focalplane locations and bin by
detector name.
"""
import itertools
import numpy as np
from lsst.afw import cameraGeom
from lsst.sims import coordUtils
import desc.imsim

__all__ = ['FocalPlaneBinner']

class Indexer:
    """Class to compute bin index for uniform bins."""
    def __init__(self, dx, x0):
        self.dx = dx
        self.x0 = x0
    def __call__(self, xval):
        return int((xval - self.x0)/self.dx)

class FocalPlaneBinner:
    """
    Class to map object sky positions to focalplane locations and bin
    by detector name.
    """
    def __init__(self, obs_md):
        self.obs_md = obs_md
        self.camera = desc.imsim.get_obs_lsstSim_camera()
        self._set_fp_indexers()
        self._set_det_mapping()

    def __call__(self, xpos, ypos):
        ix = self.x_indexer(xpos)
        iy = self.y_indexer(ypos)
        return self.det_mapping[(ix, iy)]

    def fpFromRaDec(self, ra, dec):
        """Compute focalplane coordinates (in mm) from RA, Dec (in degrees)."""
        return coordUtils.focalPlaneCoordsFromRaDec(ra, dec, camera=self.camera,
                                                    obs_metadata=self.obs_md)

    def get_chip_fp_corners(self, chip_name):
        """Get the corners of a detector in focalplane coordinates."""
        index = ([0, 1, 3, 2, 0],)
        corners = np.array(coordUtils.getCornerRaDec(chip_name, self.camera,
                                                     self.obs_md))[index]
        ra, dec = zip(*corners)
        return self.fpFromRaDec(ra, dec)

    def _set_fp_indexers(self):
        """
        Find the binning parameters in the x- and y-directions by
        analyzing the detector bounds and gaps.  This uses the
        obs_lsstSim focalplane geometry and detector names.
        """
        # Find the binning parameters in the x-direction.
        x = []
        # Iterate over the detectors on the horizontal midline.
        for R_S in itertools.product('R:0,2 R:1,2 R:2,2 R:3,2 R:4,2'.split(),
                                     'S:0,1 S:1,1 S:2,1'.split()):
            det_name = ' '.join(R_S)
            xx, yy = self.get_chip_fp_corners(det_name)
            x.extend([(xx[1] + xx[2])/2., (xx[3] + xx[4])/2.])
        x = np.array(x)
        # Compute the mean detector width.
        dx_detector = np.mean(x[1::2] - x[:-1:2])
        # Compute the mean horizontal gap between detectors.
        dx_gap = np.mean(x[2::2] - x[1:-1:2])

        # Find the binning parameters in the y-direction.
        y = []
        # Iterate over the detectors on the vertical midline.
        for R_S in itertools.product('R:2,0 R:2,1 R:2,2 R:2,3 R:2,4'.split(),
                                     'S:1,0 S:1,1 S:1,2'.split()):
            det_name = ' '.join(R_S)
            xx, yy = self.get_chip_fp_corners(det_name)
            y.extend([(yy[0] + yy[1])/2., (yy[2] + yy[3])/2.])
        y = np.array(y)
        # Compute the mean detector height.
        dy_detector = np.mean(y[1::2] - y[:-1:2])
        # Compute the mean vertical gap between detectors.
        dy_gap = np.mean(y[2::2] - y[1:-1:2])

        self.x_indexer = Indexer(dx_detector + dx_gap, min(x) - dx_gap/2.)
        self.y_indexer = Indexer(dy_detector + dy_gap, min(y) - dy_gap/2.)

    def _set_det_mapping(self):
        """
        Map the detector names by the bin indexes in the x- and
        y-directions.
        """
        self.det_mapping = dict()
        for det in self.camera:
            if det.getType() != cameraGeom.SCIENCE:
                continue
            det_name = det.getName()
            xx, yy = self.get_chip_fp_corners(det_name)
            ix = self.x_indexer(xx[0])
            iy = self.y_indexer(yy[0])
            self.det_mapping[(ix, iy)] = det_name
