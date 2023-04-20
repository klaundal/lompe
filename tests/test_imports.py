"""Just tests that expected imports work"""


def test_base_package():
    import lompe


def test_model():
    import lompe.model
    from lompe.model import Emodel, Cmodel, Data, lompeplot, model_data_scatterplot


def test_utils():
    import lompe.utils
    from lompe.utils import conductance, geodesy, sunlight, time


def test_dipole():
    import dipole
    from dipole import Dipole


def test_polplot():
    import polplot
    from polplot import Polarplot


def test_secsy():
    import lompe.secsy
