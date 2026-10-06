import sys
import subprocess

import pytest


def _pysam_loaded_after(code: str) -> bool:
    script = code + "\nimport sys\nprint(any(m.split('.')[0] == 'PySAM' for m in sys.modules))"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().splitlines()[-1] == "True"


@pytest.mark.unit
@pytest.mark.parametrize(
    "code",
    [
        "import h2integrate",
        "import h2integrate.converters.solar",
        "import h2integrate.converters.wind",
        "from h2integrate.converters.wind import FlorisWindPlantPerformanceModel",
        "from h2integrate.core.supported_models import supported_models; "
        "supported_models['ATBUtilityPVCostModel']",
    ],
)
def test_no_pysam_import(code):
    assert not _pysam_loaded_after(code)


@pytest.mark.unit
def test_pysam_models_still_importable():
    assert _pysam_loaded_after(
        "from h2integrate.converters.solar import PYSAMSolarPlantPerformanceModel"
    )
