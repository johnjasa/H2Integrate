import numpy as np
import pytest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from h2integrate.transporters.cable import CablePerformanceModel


@pytest.mark.unit
def test_cable_partials():
    plant_config = {"plant": {"simulation": {"n_timesteps": 4}}}
    prob = om.Problem(reports=False)
    prob.model.add_subsystem(
        "cable", CablePerformanceModel(plant_config=plant_config, transport_item="electricity")
    )
    prob.setup()
    prob.set_val("cable.electricity_in", np.arange(4))
    prob.run_model()
    assert_check_partials(prob.check_partials(out_stream=None), atol=1e-8, rtol=1e-8)
