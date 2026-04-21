"""
System-level control configuration parsing and validation.

This module provides validation and helper functions for the optional
``system_level_control`` section from ``plant_config``. The actual controller
component and its wiring are handled by methods on ``H2IntegrateModel``.
"""

VALID_ROLES = {"fixed", "curtailable", "dispatchable", "flexible", "storage", "demand"}

VALID_PRODUCER_ROLES = {"fixed", "curtailable", "dispatchable"}
VALID_CONSUMER_ROLES = {"flexible"}


def validate_system_level_control(slc_config, technology_config):
    """Validate the ``system_level_control`` config section.

    Checks:
    - Required keys (``commodity_streams``) are present
    - All referenced tech names exist in ``technology_config["technologies"]``
    - All roles are valid
    - At least one storage tech exists

    Args:
        slc_config (dict): The ``system_level_control`` section from plant_config.
        technology_config (dict): The full technology_config dict.

    Raises:
        ValueError: If any validation check fails.
    """
    declared_techs = set(technology_config["technologies"].keys())

    if "commodity_streams" not in slc_config:
        raise ValueError(
            "system_level_control requires a 'commodity_streams' section "
            "defining at least one commodity stream with participating technologies."
        )

    has_storage = False

    for stream_name, stream_cfg in slc_config["commodity_streams"].items():
        # Validate producers
        for entry in stream_cfg.get("producers", []):
            _validate_tech_entry(entry, stream_name, declared_techs, "producer")
            role = entry.get("role")
            if role and role not in VALID_PRODUCER_ROLES:
                raise ValueError(
                    f"system_level_control: producer '{entry['tech']}' in stream "
                    f"'{stream_name}' has invalid role '{role}'. "
                    f"Valid producer roles: {sorted(VALID_PRODUCER_ROLES)}"
                )

        # Validate consumers
        for entry in stream_cfg.get("consumers", []):
            _validate_tech_entry(entry, stream_name, declared_techs, "consumer")
            role = entry.get("role")
            if role and role not in VALID_CONSUMER_ROLES:
                raise ValueError(
                    f"system_level_control: consumer '{entry['tech']}' in stream "
                    f"'{stream_name}' has invalid role '{role}'. "
                    f"Valid consumer roles: {sorted(VALID_CONSUMER_ROLES)}"
                )

        # Validate storage
        for entry in stream_cfg.get("storage", []):
            _validate_tech_entry(entry, stream_name, declared_techs, "storage")
            has_storage = True

        # Validate demands
        for entry in stream_cfg.get("demands", []):
            _validate_tech_entry(entry, stream_name, declared_techs, "demand")

    if not has_storage:
        raise ValueError(
            "system_level_control requires at least one storage technology "
            "across all commodity streams."
        )

    # Check mutual exclusivity with existing tech_to_dispatch_connections
    get_all_slc_tech_names(slc_config)


def _validate_tech_entry(entry, stream_name, declared_techs, category):
    """Validate a single technology entry in a commodity stream.

    Args:
        entry (dict): Single entry like ``{"tech": "wind", "role": "fixed"}``.
        stream_name (str): Name of the parent commodity stream.
        declared_techs (set): Set of valid technology names.
        category (str): One of "producer", "consumer", "storage", "demand".

    Raises:
        ValueError: If the tech name is missing or not declared.
    """
    if "tech" not in entry:
        raise ValueError(
            f"system_level_control: entry in '{category}' list of stream "
            f"'{stream_name}' is missing required 'tech' key."
        )
    tech = entry["tech"]
    if tech not in declared_techs:
        raise ValueError(
            f"system_level_control references tech '{tech}' in stream "
            f"'{stream_name}', but it is not declared in "
            f"tech_config.technologies. Available technologies: "
            f"{sorted(declared_techs)}"
        )


def get_all_slc_tech_names(slc_config):
    """Extract all unique technology names from the system_level_control config.

    Args:
        slc_config (dict): The ``system_level_control`` section from plant_config.

    Returns:
        set: All technology names referenced in any commodity stream.
    """
    techs = set()
    for stream_cfg in slc_config.get("commodity_streams", {}).values():
        for category in ("producers", "consumers", "storage", "demands"):
            for entry in stream_cfg.get(category, []):
                techs.add(entry["tech"])
    return techs


def get_storage_techs(slc_config):
    """Get all storage technology names from the system_level_control config.

    Args:
        slc_config (dict): The ``system_level_control`` section from plant_config.

    Returns:
        list[dict]: Storage entries, each with at least a ``"tech"`` key.
    """
    storage_entries = []
    for stream_cfg in slc_config.get("commodity_streams", {}).values():
        storage_entries.extend(stream_cfg.get("storage", []))
    return storage_entries


def get_fixed_producers(slc_config):
    """Get all fixed-role producer entries from the SLC config.

    Args:
        slc_config (dict): The ``system_level_control`` section from plant_config.

    Returns:
        list[tuple[str, dict]]: List of (stream_name, entry) tuples.
    """
    results = []
    for stream_name, stream_cfg in slc_config.get("commodity_streams", {}).items():
        results.extend(
            (stream_name, entry)
            for entry in stream_cfg.get("producers", [])
            if entry.get("role") == "fixed"
        )
    return results
