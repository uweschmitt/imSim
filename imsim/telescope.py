import galsim
import batoid
from .batoid_utils import load_telescope


def build_telescope(name, band, rotTelPos, perturb=None):
    if perturb is None:
        perturb = []
    telescope = load_telescope(name, band)
    if name == 'LSST':
        telescope = telescope.withLocallyRotatedOptic(
            "LSSTCamera",
            batoid.RotZ(rotTelPos)
        )
    # Monkey-patch the band onto the telescope so we can use it later
    # for determining a fiducial wavelength.
    for perturbation in perturb:
        for k, v in perturbation.items():
            if k == 'shift':
                for optic, shift in v.items():
                    telescope = telescope.withGloballyShiftedOptic(optic, shift)
            elif k == 'rotate':
                for optic, angles in v.items():
                    telescope = telescope.withLocallyRotatedOptic(
                        optic,
                        batoid.RotZ(angles[2])@batoid.RotY(angles[1])@batoid.RotX(angles[0])
                    )
            elif k == 'Zernike':
                raise NotImplementedError("Zernike perturbation not yet implemented")
            elif k == 'Grid':
                raise NotImplementedError("Grid perturbation not yet implemented")
            elif k == 'AOS_dof':
                raise NotImplementedError("AOS_dof perturbation not yet implemented")
            elif k == 'FEA':
                raise NotImplementedError("FEA perturbation not yet implemented")
            elif k == 'Actuator':
                raise NotImplementedError("Actuator perturbation not yet implemented")
            elif k == 'LUT':
                raise NotImplementedError("LUT perturbation not yet implemented")
            else:
                raise ValueError(f"Unknown perturbation type {k}")

    telescope.band = band
    return telescope


class TelescopeLoader(galsim.config.InputLoader):
    def getKwargs(self, config, base, logger):
        req = {
            'name': str,
            'band': str,
            'rotTelPos': galsim.Angle
        }
        opt = {
            'perturb': list
        }
        kwargs, safe = galsim.config.GetAllParams(
            config, base, req=req, opt=opt
        )
        return kwargs, safe


galsim.config.RegisterInputType(
    'telescope', TelescopeLoader(build_telescope, file_scope=True),
)
