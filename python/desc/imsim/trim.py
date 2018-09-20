"""
Function to apply chip-centered acceptance cones on instance catalogs.
"""
from collections import defaultdict
import pickle
import numpy as np
import lsst.sims.coordUtils
from lsst.sims.utils import _angularSeparation
import desc.imsim
from .fp_binner import FocalPlaneBinner

__all__ = ['InstCatTrimmer']


def degrees_separation(ra0, dec0, ra, dec):
    """
    Compute angular separation in degrees.

    Parameters
    ----------
    ra0: float
        Right ascension of reference location in degrees.
    dec0: float
        Declination of reference location in degrees.
    ra: float or numpy.array
        Right ascension of object(s) in degrees
    dec: float or numpy.array
        Declination of object(s) in degrees.
    """
    return np.degrees(_angularSeparation(np.radians(ra0), np.radians(dec0),
                                         np.radians(ra), np.radians(dec)))

class Disaggregator:
    """
    Class to disaggregate instance catalog object lines per chip using
    acceptance cones.
    """
    def __init__(self, object_lines, trimmer):
        """
        Parameters
        ----------
        object_lines: list
            list of object entries from an instance catalog.
        trimmer: InstCatTrimmer
            An instance of the InstCatTrimmer class to provide
            visit-level metadata.
        """
        self.object_lines = object_lines
        self.trimmer = trimmer

        # Extract the ra, dec values for each object.
        self._ids = []
        self._ra = np.zeros(len(object_lines), dtype=np.float)
        self._dec = np.zeros(len(object_lines), dtype=np.float)
        self._sersic = np.zeros(len(object_lines), dtype=np.int)
        self._magnorm = np.zeros(len(object_lines), dtype=np.float)
        for i, line in enumerate(object_lines):
            tokens = line.strip().split()
            self._ids.append(tokens[1])
            self._ra[i] = np.float(tokens[2])
            self._dec[i] = np.float(tokens[3])
            if 'sersic2d' in line:
                self._sersic[i] = 1
            self._magnorm[i] = np.float(tokens[4])
        self._camera = desc.imsim.get_obs_lsstSim_camera()

        # TODO: Sort objects by magnorm.

        # Convert object ra, dec values to focalplane locations.
        xpos, ypos = self.trimmer.fp_binner.fpFromRaDec(self._ra, self._dec)
        # Bin by detector.
        self._obj_lists = defaultdict(list)
        self._sersic_lists = defaultdict(list)
        for i, xx, yy in zip(range(len(xpos)), xpos, ypos):
            det_name = self.trimmer.fp_binner(xx, yy)
            # If object is in already-drawn objects, skip it.
            if (det_name in self.trimmer.drawn_objects
                    and self._ids[i] in self.trimmer.drawn_objects[det_name]):
                continue
            self._obj_lists[det_name].append(self.object_lines[i])
            self._sersic_lists[det_name].append(self._sersic[i])

    def compute_chip_center(self, chip_name):
        """
        Compute the center of the desired chip in focalplane pixel
        coordinates.

        Parameters
        ----------
        chip_name: str
            Name of the CCD, e.g., "R:2,2 S:1,1".

        Returns
        -------
        (float, float): The RA, Dec in degrees of the center of the CCD.
        """
        center_x, center_y = desc.imsim.get_chip_center(chip_name, self._camera)
        return lsst.sims.coordUtils.raDecFromPixelCoords(
            xPix=center_x, yPix=center_y, chipName=chip_name,
            camera=self._camera, obs_metadata=self.trimmer.obs_md, epoch=2000.0,
            includeDistortion=True)

    def get_object_entries(self, chip_name):
        """
        Get the object entries within an acceptance cone centered on
        a specified CCD.

        Parameters
        ----------
        chip_name: str
            Name of the CCD, e.g., "R:2,2 S:1,1".

        Returns
        -------
        list, int: (list of object entries from the original instance catalog,
                    number of sersic objects for minsource application)

        """
        return self._obj_lists[chip_name], sum(self._sersic_lists[chip_name])

class InstCatTrimmer(dict):
    """
    Subclass of dict to provide trimmed instance catalogs for
    acceptance cones centered on CCDs in the LSST focalplane.

    Attributes
    ----------
    instcat_file: str
        Instance catalog filename.
    obs_md: ObservationMetadata
        Observation metadata for the visit.
    minsource: int
        Minimum number of sersic objects to require for a sensor-visit
        to be simulated.

    """
    def __init__(self, instcat, sensor_list, checkpoint_files=None,
                 chunk_size=int(3e5), numRows=None, minsource=None,
                 log_level='INFO'):
        """
        Parameters
        ----------
        instcat: str
            Path to input instance catalog.  The file can have includeobj
            entries.
        sensor_list: list
            List of sensors, e.g., "R:2,2 S:1,1", for which to provide
            object lists.
        checkpoint_files: dict [None]
            Checkpoint files keyed by sensor name, e.g., "R:2,2 S:1,1".
            The instance catalog lines corresponding to drawn_objects in
            the checkpoint files will be skipped on ingest.
        chunk_size: int [int(3e5)]
            Number of lines to read in at a time from the instance catalogs
            to avoid excess memory usage.
        numRows: int [None]
            Maximum number of rows to read in from the instance catalog.
        minsource: int [None]
            Minimum number of galaxies in a given sensor acceptance cone.
            If not None, this overrides the value in the instance catalog.
        log_level: str ['INFO']
            Logging level.
        """
        super(InstCatTrimmer, self).__init__()
        self.logger = desc.imsim.get_logger(log_level, 'InstCatTrimmer')
        self.instcat_file = instcat
        self._read_commands()
        self.fp_binner = FocalPlaneBinner(self.obs_md)
        if minsource is not None:
            self.minsource = minsource
        self._read_drawn_objects(checkpoint_files)
        self._process_objects(sensor_list, chunk_size, numRows=numRows)

    def _read_drawn_objects(self, checkpoint_files):
        """
        Read the drawn objects from the checkpoint files.
        """
        self.drawn_objects = dict()
        if checkpoint_files is None:
            return
        for detname, ckpt_file in checkpoint_files.items():
            with open(ckpt_file, 'rb') as fd:
                ckpt = pickle.load(fd)
                self.drawn_objects[detname] = ckpt['drawn_objects']

    def _process_objects(self, sensor_list, chunk_size, numRows=None):
        """
        Loop over chunks of lines from the instance catalog
        and disaggregate the entries into the separate object lists
        for each sensor using the Disaggregator class to apply the
        acceptance cone cut centered on each sensor.
        """
        num_gals = defaultdict(lambda: 0)
        self.update({sensor: [] for sensor in sensor_list})
        with desc.imsim.fopen(self.instcat_file, mode='rt') as fd:
            nread = 0
            while numRows is None or nread < numRows:
                object_lines = []
                ichunk = 0
                for ichunk, line in zip(range(chunk_size), fd):
                    nread += 1
                    if (not line.startswith('object') or
                            'inf' in line.split()):
                        continue
                    object_lines.append(line)
                disaggregator = Disaggregator(object_lines, self)
                for sensor in self:
                    obj_list, nsersic = disaggregator.get_object_entries(sensor)
                    self[sensor].extend(obj_list)
                    num_gals[sensor] += nsersic
                if ichunk < chunk_size - 1:
                    break
        for sensor in self:
            # Apply minsource criterion on galaxies.
            if self.minsource is not None and num_gals[sensor] < self.minsource:
                self[sensor] = []

    def _read_commands(self):
        """Read in the commands from the instance catalog."""
        max_lines = 50  # There should be fewer than 50, but put a hard
                        # limit to avoid suspect catalogs.
        self.command_lines = []
        phosim_commands = dict()
        with desc.imsim.fopen(self.instcat_file, mode='rt') as input_:
            for line, _ in zip(input_, range(max_lines)):
                if line.startswith('object'):
                    break
                if line.startswith('#'):
                    continue
                self.command_lines.append(line)
                tokens = line.strip().split()
                phosim_commands[tokens[0]] = float(tokens[1])
        try:
            self.minsource = phosim_commands['minsource']
        except KeyError:
            self.minsource = None
        phosim_commands['bandpass'] = 'ugrizy'[int(phosim_commands['filter'])]
        self.obs_md = desc.imsim.phosim_obs_metadata(phosim_commands)
