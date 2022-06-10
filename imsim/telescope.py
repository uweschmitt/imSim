import galsim
import batoid
from .batoid_utils import load_telescope


def build_telescope(name, band, rotTelPos):
    telescope = load_telescope(name, band)
    if name == 'LSST':
        telescope = telescope.withLocallyRotatedOptic(
            "LSSTCamera",
            batoid.RotZ(rotTelPos)
        )
    # Monkey-patch the band onto the telescope so we can use it later
    # for determining a fiducial wavelength.
    telescope.band = band
    return telescope


class TelescopeLoader(galsim.config.InputLoader):
    def getKwargs(self, config, base, logger):
        req = {
            'name': str,
            'band': str,
            'rotTelPos': galsim.Angle
        }
        kwargs, safe = galsim.config.GetAllParams(
            config, base, req=req
        )
        return kwargs, safe


galsim.config.RegisterInputType(
    'telescope', TelescopeLoader(build_telescope, file_scope=True),
)
