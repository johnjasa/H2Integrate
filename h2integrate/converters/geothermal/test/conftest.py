import numpy as np
import pytest

from test.conftest import (  # noqa: F401
    temp_dir,
    temp_copy_of_example,
    pytest_collection_modifyitems,
)


@pytest.fixture
def plant_config():
    plant = {
        "plant": {
            "plant_life": 30,
            "simulation": {
                "dt": 3600,
                "n_timesteps": 8760,
                "start_time": "01/01/1900 00:30:00",
                "timezone": 0,
            },
        },
        "site": {"latitude": 30.6617, "longitude": -101.7096, "resources": {}},
    }

    return plant


@pytest.fixture
def solar_resource_data():
    """Representative ambient weather resource data used for the geothermal power cycle.

    This mimics the ``solar_resource_data`` dictionary produced by the solar/weather
    resource models, providing the surface ambient conditions (dry-bulb temperature,
    relative humidity, and pressure) that the geothermal power-cycle design uses.
    """
    n = 8760
    hours = np.arange(n)
    # Simple diurnal + seasonal dry-bulb temperature swing around a ~18 C annual mean.
    temperature = (
        18.0
        + 10.0 * np.sin(2 * np.pi * (hours - 8) / 24.0)
        + 8.0 * np.sin(2 * np.pi * (hours - 2160) / n)
    )
    return {
        "site_lat": 30.6617,
        "site_lon": -101.7096,
        "temperature": temperature,
        "relative_humidity": np.full(n, 45.0),
        "pressure": np.full(n, 1013.25),
    }
