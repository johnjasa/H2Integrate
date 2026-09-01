"""Generic structured-grid surrogate support for technology performance models.

Several technology performance models accept design variables that are physically
integer valued (number of wind turbines, number of electrolyzer clusters, ...) but
are handed to the optimizer as continuous quantities. Rounding those values inside
the model turns the objective into a staircase, which derivative-free optimizers
such as COBYLA handle poorly: the linear trust-region model sees zero slope across
the flat regions and the optimizer stalls at whatever point it happens to occupy.

For a technology whose only non-design-variable inputs are constant over the course
of a run (a wind plant, for example, depends only on its design variables and a
fixed resource time series) the model can be sampled once on a structured grid over
those design variables and replaced by a smooth interpolant for the remainder of the
run. Because the samples are taken at integer values, the interpolant reproduces the
underlying physics exactly at every training point and varies smoothly in between.

``SurrogateMixin`` implements that pattern generically for an arbitrary number of
surrogate inputs. A model opts in by:

1. Listing ``SurrogateMixin`` first in its base classes.
2. Renaming its ``compute`` method to ``compute_physics``.
3. Calling ``self.setup_surrogate()`` at the end of its ``setup`` method.

The surrogate is only valid while every input that is *not* being interpolated over
stays constant. That invariant is enforced by fingerprinting all other inputs at
training time and re-checking the fingerprint on every evaluation.
"""

import hashlib
import itertools

import numpy as np
from attrs import field, define
from scipy.interpolate import RegularGridInterpolator

from h2integrate.core.utilities import BaseConfig
from h2integrate.core.validators import contains


# Minimum number of samples per dimension required by each interpolation method. These
# mirror scipy's own requirement of one more sample than the underlying spline degree.
MIN_POINTS_PER_METHOD = {
    "linear": 2,
    "nearest": 1,
    "slinear": 2,
    "cubic": 4,
    "quintic": 6,
    "pchip": 4,
}


@define(kw_only=True)
class SurrogateModelConfig(BaseConfig):
    """Configuration for a structured-grid surrogate of a performance model.

    Supplied in the technology config under ``model_inputs/surrogate_parameters``.

    Attributes:
        enabled (bool): Whether to replace the model's physics with a surrogate.
            Defaults to False.
        inputs (dict): Mapping of input name to a specification dictionary. Each
            specification must contain ``n_points`` (number of samples along that
            dimension) and may optionally contain ``lower`` and ``upper``. When the
            bounds are omitted they are read from the matching design variable in the
            driver config.
        outputs (list | None): Names of the outputs to interpolate. Defaults to None,
            meaning every continuous output of the model is interpolated.
        method (str): Interpolation method passed to
            ``scipy.interpolate.RegularGridInterpolator``. Defaults to "pchip", which
            is smooth and monotonicity preserving and needs four points per dimension.
        on_invalidation (str): Action taken when an input that is not part of the
            surrogate grid changes after training, which means the surrogate is no
            longer valid. Either "error" (default) or "retrain".
    """

    enabled: bool = field(default=False)
    inputs: dict = field(factory=dict)
    outputs: list | None = field(default=None)
    method: str = field(default="pchip", validator=contains(list(MIN_POINTS_PER_METHOD)))
    on_invalidation: str = field(default="error", validator=contains(["error", "retrain"]))


class _OutputProxy(dict):
    """Stand-in for an OpenMDAO output vector used while sampling a physics model.

    OpenMDAO output vectors coerce whatever a model assigns into the declared array
    shape, so models routinely assign scalars and tuples (PySAM outputs, for example)
    and rely on the vector to convert them. This proxy reproduces that behavior so
    ``compute_physics`` sees no difference between a training sample and a normal
    evaluation.

    Args:
        values (dict): Initial output values, keyed by output name.
        shapes (dict): Declared shape of each output, keyed by output name.
    """

    def __init__(self, values, shapes):
        super().__init__(values)
        self._shapes = shapes

    def __setitem__(self, key, value):
        value = np.asarray(value, dtype=float)
        shape = self._shapes.get(key)
        if shape is not None and value.shape != shape:
            value = np.broadcast_to(value, shape).copy()
        super().__setitem__(key, value)


class SurrogateMixin:
    """Mixin that can replace a performance model's physics with an interpolant.

    Subclasses implement ``compute_physics`` instead of ``compute`` and call
    ``setup_surrogate`` at the end of ``setup``. When the surrogate is disabled the
    mixin simply forwards every evaluation to ``compute_physics``, so a model that
    includes the mixin behaves identically to one that does not until a user turns
    the surrogate on.
    """

    def setup_surrogate(self):
        """Parse the surrogate configuration and initialize the surrogate state.

        Should be called at the end of the subclass ``setup`` method, after the
        model's inputs and outputs have been declared.

        Raises:
            ValueError: The surrogate is enabled but no inputs were given, or an
                input specification is missing or has an invalid ``n_points``.
        """
        surrogate_params = self.options["tech_config"].get("model_inputs", {})
        surrogate_params = surrogate_params.get("surrogate_parameters", {})
        self.surrogate_config = SurrogateModelConfig.from_dict(
            surrogate_params, additional_cls_name=self.__class__.__name__
        )

        self._surrogate = None
        self._surrogate_axes = None
        self._surrogate_fingerprint = None
        self._surrogate_training = False
        self._surrogate_input_names = list(self.surrogate_config.inputs)

        # Inputs that are allowed to change after training without invalidating the
        # surrogate. Curtailment is applied after prediction rather than being baked
        # into the training data, so the command value is exempt.
        exempt = set(getattr(self, "_surrogate_exempt_inputs", ()))
        if getattr(self, "_control_classifier", None) == "flexible":
            exempt.add(f"{self.commodity}_command_value")
        self._surrogate_exempt_inputs = tuple(sorted(exempt))

        if not self.surrogate_config.enabled:
            return

        if not self._surrogate_input_names:
            raise ValueError(
                f"{self.__class__.__name__} has surrogate_parameters/enabled set to true but no "
                "entries under surrogate_parameters/inputs. At least one input must be given."
            )

        min_points = MIN_POINTS_PER_METHOD[self.surrogate_config.method]
        for name, spec in self.surrogate_config.inputs.items():
            if "n_points" not in spec:
                raise ValueError(
                    f"{self.__class__.__name__} surrogate input '{name}' is missing 'n_points'."
                )
            if int(spec["n_points"]) < min_points:
                raise ValueError(
                    f"{self.__class__.__name__} surrogate input '{name}' has n_points="
                    f"{spec['n_points']}, but the '{self.surrogate_config.method}' method "
                    f"requires at least {min_points} points per dimension."
                )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Evaluate the model, using the surrogate when it is enabled.

        Args:
            inputs (om.vectors.default_vector.DefaultVector): OM inputs.
            outputs (om.vectors.default_vector.DefaultVector): OM outputs.
            discrete_inputs (om.core.component._DictValues, optional): OM discrete inputs.
            discrete_outputs (om.core.component._DictValues, optional): OM discrete outputs.

        Raises:
            RuntimeError: An input outside the surrogate grid changed after training
                and ``on_invalidation`` is "error".
        """
        if not getattr(self, "surrogate_config", None) or not self.surrogate_config.enabled:
            self.compute_physics(inputs, outputs, discrete_inputs, discrete_outputs)
            return

        fingerprint = self._surrogate_input_fingerprint(inputs, discrete_inputs)

        if self._surrogate is None:
            self._train_surrogate(inputs, outputs, discrete_inputs, discrete_outputs)
            self._surrogate_fingerprint = fingerprint
        elif fingerprint != self._surrogate_fingerprint:
            if self.surrogate_config.on_invalidation == "error":
                raise RuntimeError(
                    f"The surrogate for {self.msginfo} is no longer valid: an input that is not "
                    f"part of the surrogate grid {self._surrogate_input_names} changed after the "
                    "surrogate was trained. A surrogate may only be used for a technology whose "
                    "remaining inputs are constant. Either add the changing input to "
                    "surrogate_parameters/inputs, disable the surrogate, or set "
                    "surrogate_parameters/on_invalidation to 'retrain'."
                )
            self._train_surrogate(inputs, outputs, discrete_inputs, discrete_outputs)
            self._surrogate_fingerprint = fingerprint

        self._predict_surrogate(inputs, outputs)

    def compute_physics(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Evaluate the underlying physics model.

        Subclasses of ``SurrogateMixin`` implement this instead of ``compute``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} uses SurrogateMixin and must implement compute_physics."
        )

    def _resolve_surrogate_axes(self):
        """Build the sample locations for each surrogate input.

        Bounds come from the input specification when given, and otherwise from the
        matching design variable in the driver config.

        Raises:
            ValueError: Bounds could not be determined for a surrogate input.

        Returns:
            list[np.ndarray]: Sample locations, one array per surrogate input.
        """
        path_parts = self.pathname.split(".")
        tech_name = path_parts[-2] if len(path_parts) > 1 else None
        tech_design_vars = (
            self.options["driver_config"].get("design_variables", {}).get(tech_name, {})
        )

        axes = []
        for name, spec in self.surrogate_config.inputs.items():
            design_var = tech_design_vars.get(name, {})
            lower = spec.get("lower", design_var.get("lower"))
            upper = spec.get("upper", design_var.get("upper"))
            if lower is None or upper is None:
                raise ValueError(
                    f"Could not determine bounds for surrogate input '{name}' of {self.msginfo}. "
                    f"Either declare '{tech_name}.{name}' as a design variable with 'lower' and "
                    "'upper' in the driver config, or give 'lower' and 'upper' directly in "
                    "surrogate_parameters/inputs."
                )
            axes.append(np.linspace(float(lower), float(upper), int(spec["n_points"])))

        return axes

    def _train_surrogate(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """Sample the physics model on a structured grid and build the interpolants.

        Args:
            inputs (om.vectors.default_vector.DefaultVector): OM inputs.
            outputs (om.vectors.default_vector.DefaultVector): OM outputs.
            discrete_inputs (om.core.component._DictValues): OM discrete inputs.
            discrete_outputs (om.core.component._DictValues): OM discrete outputs.

        Raises:
            ValueError: A surrogate input is not an input of this model, or a
                requested surrogate output is not an output of this model.
        """
        input_values = dict(inputs.items())
        missing = [name for name in self._surrogate_input_names if name not in input_values]
        if missing:
            raise ValueError(
                f"Surrogate input(s) {missing} are not inputs of {self.msginfo}. "
                f"Available inputs are {sorted(input_values)}."
            )

        output_shapes = {name: np.shape(value) for name, value in outputs.items()}
        surrogate_output_names = self.surrogate_config.outputs or list(output_shapes)
        missing = [name for name in surrogate_output_names if name not in output_shapes]
        if missing:
            raise ValueError(
                f"Surrogate output(s) {missing} are not outputs of {self.msginfo}. "
                f"Available outputs are {sorted(output_shapes)}."
            )

        self._surrogate_axes = self._resolve_surrogate_axes()
        samples = {name: [] for name in surrogate_output_names}

        # Suppress curtailment while sampling so the surrogate captures the
        # uncurtailed production; curtailment is re-applied after prediction.
        self._surrogate_training = True
        try:
            for point in itertools.product(*self._surrogate_axes):
                sample_inputs = {name: np.array(value) for name, value in input_values.items()}
                for name, value in zip(self._surrogate_input_names, point):
                    sample_inputs[name] = np.full_like(sample_inputs[name], value, dtype=float)

                sample_outputs = _OutputProxy(
                    {name: np.array(value, dtype=float) for name, value in outputs.items()},
                    output_shapes,
                )
                self.compute_physics(
                    sample_inputs, sample_outputs, discrete_inputs, discrete_outputs
                )

                for name in surrogate_output_names:
                    value = np.asarray(sample_outputs[name], dtype=float)
                    samples[name].append(np.broadcast_to(value, output_shapes[name]).copy())
        finally:
            self._surrogate_training = False

        grid_shape = tuple(len(axis) for axis in self._surrogate_axes)
        self._surrogate = {
            name: RegularGridInterpolator(
                self._surrogate_axes,
                np.asarray(values).reshape(grid_shape + output_shapes[name]),
                method=self.surrogate_config.method,
                bounds_error=False,
                fill_value=None,
            )
            for name, values in samples.items()
        }

    def _predict_surrogate(self, inputs, outputs):
        """Set the model outputs by evaluating the trained interpolants.

        Args:
            inputs (om.vectors.default_vector.DefaultVector): OM inputs.
            outputs (om.vectors.default_vector.DefaultVector): OM outputs.
        """
        point = np.array([[float(inputs[name][0]) for name in self._surrogate_input_names]])
        for name, interpolator in self._surrogate.items():
            outputs[name] = interpolator(point)[0]

        # Curtailment depends on a control signal that changes between evaluations, so
        # it is applied to the interpolated production rather than being interpolated.
        if hasattr(self, "apply_curtailment"):
            self.apply_curtailment(outputs)

    def _surrogate_input_fingerprint(self, inputs, discrete_inputs):
        """Hash every input that the surrogate does not interpolate over.

        Args:
            inputs (om.vectors.default_vector.DefaultVector): OM inputs.
            discrete_inputs (om.core.component._DictValues | None): OM discrete inputs.

        Returns:
            str: Hex digest identifying the state of the non-surrogate inputs.
        """
        skip = set(self._surrogate_input_names) | set(self._surrogate_exempt_inputs)

        hasher = hashlib.md5()
        for name, value in sorted(inputs.items()):
            if name in skip:
                continue
            hasher.update(name.encode("utf-8"))
            _update_hash(hasher, value)
        for name, value in sorted((discrete_inputs or {}).items()):
            hasher.update(name.encode("utf-8"))
            _update_hash(hasher, value)

        return hasher.hexdigest()


def _update_hash(hasher, value):
    """Update a hasher with a value, hashing numpy arrays by their raw bytes.

    Recurses through dicts, lists, and tuples so nested containers of arrays (such
    as resource-data discrete inputs) are hashed efficiently and deterministically.

    Args:
        hasher (hashlib._Hash): Hasher to update in place.
        value: The value to fold into the hash.
    """
    if isinstance(value, np.ndarray):
        hasher.update(b"ndarray")
        hasher.update(str(value.dtype).encode("utf-8"))
        hasher.update(str(value.shape).encode("utf-8"))
        hasher.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            hasher.update(str(key).encode("utf-8"))
            _update_hash(hasher, value[key])
    elif isinstance(value, list | tuple):
        hasher.update(b"seq")
        for item in value:
            _update_hash(hasher, item)
    else:
        hasher.update(str(value).encode("utf-8"))
