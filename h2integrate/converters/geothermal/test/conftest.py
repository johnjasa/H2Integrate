import pytest

from h2integrate import EXAMPLE_DIR

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
def weather_file():
    """Path to a representative ambient weather file used for the geothermal power cycle."""
    weather_dir = EXAMPLE_DIR / "11_hybrid_energy_plant" / "tech_inputs" / "weather" / "solar"
    return str(weather_dir / "30.6617_-101.7096_psmv3_60_2013.csv")
