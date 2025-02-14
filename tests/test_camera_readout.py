"""
Unit tests for electronics readout simulation code.
"""
import os
from pathlib import Path
import itertools
import unittest
import astropy.io.fits as fits
import imsim
import galsim

DATA_DIR = Path(__file__).parent / 'data'

def transpose(image):
    # TODO: This will be a method of galsim.Image in GalSim v2.3.
    #       Can remove this monkey-patch once we're using that.
    b1 = image.bounds
    b2 = galsim.BoundsI(b1.ymin, b1.ymax, b1.xmin, b1.xmax)
    return galsim._Image(image.array.T, b2, image.wcs)
galsim.Image.transpose = transpose

class ImageSourceTestCase(unittest.TestCase):
    "TestCase class for ImageSource."

    def setUp(self):
        # Note: This e-image was originally made using the sizes for ITL sensors.
        #       Now this raft is set to be e2v, so we need to fake the numbers below
        #       to pretend it is still an ITL raft.
        #       This is why we use R20_S11 for the det_name below.
        self.eimage_file = str(DATA_DIR / 'lsst_e_161899_R22_S11_r.fits.gz')
        instcat_file = str(DATA_DIR / 'tiny_instcat.txt')
        self.image = galsim.fits.read(self.eimage_file)
        # Also, this file was made in phosim convention, which swaps x,y relative
        # to the normal convention.  So we need to transpose the image after reading it.
        self.image = self.image.transpose()
        self.config = {
            'image': {'random_seed': 1234},
            'input': {
                'opsim_meta_dict': {'file_name': instcat_file}
            },
            'output': {
                'readout' : {
                    'file_name': 'amp.fits',
                    'readout_time': 3,
                    'dark_current': 0.02,
                    'bias_level': 1000.,
                    'pcti': 1.e-6,
                    'scti': 1.e-6,
                }
            },
            'index_key': 'image_num',
            'image_num': 0,
            'det_name': 'R20_S00',
            'exp_time': 30,
        }

        galsim.config.SetupConfigRNG(self.config)
        self.logger = galsim.config.LoggerWrapper(None)
        self.readout = imsim.CameraReadout()
        self.readout_config = self.config['output']['readout']
        galsim.config.ProcessInput(self.config)
        self.readout.initialize(None,None, self.readout_config, self.config, self.logger)
        self.readout.ensureFinalized(self.readout_config, self.config, [self.image], self.logger)

    def tearDown(self):
        pass

    def test_create_from_eimage(self):
        "Test the .create_from_eimage static method."
        hdus = self.readout.final_data
        self.assertAlmostEqual(hdus[0].header['EXPTIME'], 30.)
        self.assertTupleEqual(self.image.array.shape, (4000, 4072))
        for i in range(1,16):
            self.assertTupleEqual(hdus[i].data.shape, (2048, 544))

    def test_get_amplifier_hdu(self):
        "Test the .get_amplifier_hdu method."
        hdus = self.readout.final_data
        hdu = hdus[1]
        self.assertEqual(hdu.header['EXTNAME'], "Segment10")
        self.assertEqual(hdu.header['DATASEC'], "[4:512,1:2000]")
        self.assertEqual(hdu.header['DETSEC'], "[509:1,4000:2001]")

        hdu = hdus[8]
        self.assertEqual(hdu.header['EXTNAME'], "Segment17")
        self.assertEqual(hdu.header['DATASEC'], "[4:512,1:2000]")
        self.assertEqual(hdu.header['DETSEC'], "[4072:3564,4000:2001]")

    def test_raw_file_headers(self):
        "Test contents of raw file headers."
        outfile = 'raw_file_test.fits'
        self.readout.writeFile(outfile, self.readout_config, self.config, self.logger)
        with fits.open(outfile) as hdus:
            self.assertEqual(hdus[0].header['IMSIMVER'], imsim.__version__)
        os.remove(outfile)

    def test_no_opsim(self):
        "Test running readout without OpsimMeta (e.g. for flats)"
        outfile = 'raw_no_opsim_test.fits'
        config = {  # Same as above, but no input field.
            'image': {'random_seed': 1234},
            'output': {
                'readout' : {
                    'file_name': 'amp.fits',
                    'readout_time': 3,
                    'dark_current': 0.02,
                    'bias_level': 1000.,
                    'pcti': 1.e-6,
                    'scti': 1.e-6,
                }
            },
            'index_key': 'image_num',
            'image_num': 0,
            'det_name': 'R20_S00',
            'exp_time': 30,
        }
        readout = imsim.CameraReadout()
        readout_config = config['output']['readout']
        readout.initialize(None,None, readout_config, config, self.logger)
        readout.ensureFinalized(readout_config, config, [self.image], self.logger)
        readout.writeFile(outfile, self.readout_config, self.config, self.logger)
        with fits.open(outfile) as hdus:
            self.assertEqual(hdus[0].header['IMSIMVER'], imsim.__version__)
            self.assertEqual(hdus[0].header['FILTER'], 'N/A')
            self.assertEqual(hdus[0].header['MJD-OBS'], 51544)
        os.remove(outfile)

        # Filter is option in the readout section to show up in the header.
        readout_config['filter'] = 'r'
        readout.initialize(None,None, readout_config, config, self.logger)
        readout.ensureFinalized(readout_config, config, [self.image], self.logger)
        readout.writeFile(outfile, self.readout_config, self.config, self.logger)
        with fits.open(outfile) as hdus:
            self.assertEqual(hdus[0].header['FILTER'], 'r')

        # Make sure it parses it, not just gets it.
        readout_config['filter'] = '$"Happy Birthday!"[8]'
        readout.initialize(None,None, readout_config, config, self.logger)
        readout.ensureFinalized(readout_config, config, [self.image], self.logger)
        readout.writeFile(outfile, self.readout_config, self.config, self.logger)
        with fits.open(outfile) as hdus:
            self.assertEqual(hdus[0].header['FILTER'], 'r')


if __name__ == '__main__':
    unittest.main()
