"""Smoke tests: verify all four packages can be imported cleanly."""


def test_import_microcircuits():
    import microcircuits  # noqa: F401


def test_import_neuralsampling():
    import neuralsampling  # noqa: F401


def test_import_stddc():
    import stddc  # noqa: F401


def test_import_symmnet():
    import symmnet  # noqa: F401
