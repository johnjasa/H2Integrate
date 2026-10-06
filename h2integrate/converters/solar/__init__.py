import importlib


_LAZY_IMPORTS = {
    "PYSAMSolarPlantPerformanceModel": "solar_pysam",
    "ATBUtilityPVCostModel": "atb_utility_pv_cost",
    "ATBResComPVCostModel": "atb_res_com_pv_cost",
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(f"{__name__}.{_LAZY_IMPORTS[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
