"""In-process Ipopt solves using a cached PyNumero nonlinear program."""

import os
import sys
import ctypes
import ctypes.util
from numbers import Real, Integral
from pathlib import Path

import numpy as np


_INT = ctypes.c_int
_DOUBLE = ctypes.c_double
_DOUBLE_PTR = ctypes.POINTER(_DOUBLE)
_INT_PTR = ctypes.POINTER(_INT)
_VOID_PTR = ctypes.c_void_p
_EVAL_F = ctypes.CFUNCTYPE(_INT, _INT, _DOUBLE_PTR, _INT, _DOUBLE_PTR, _VOID_PTR)
_EVAL_G = ctypes.CFUNCTYPE(_INT, _INT, _DOUBLE_PTR, _INT, _INT, _DOUBLE_PTR, _VOID_PTR)
_EVAL_JAC = ctypes.CFUNCTYPE(
    _INT, _INT, _DOUBLE_PTR, _INT, _INT, _INT, _INT_PTR, _INT_PTR, _DOUBLE_PTR, _VOID_PTR
)
_EVAL_HESS = ctypes.CFUNCTYPE(
    _INT,
    _INT,
    _DOUBLE_PTR,
    _INT,
    _DOUBLE,
    _INT,
    _DOUBLE_PTR,
    _INT,
    _INT,
    _INT_PTR,
    _INT_PTR,
    _DOUBLE_PTR,
    _VOID_PTR,
)


def _double_ptr(array: np.ndarray) -> ctypes._Pointer:
    return array.ctypes.data_as(_DOUBLE_PTR)


def cached_ipopt_available() -> bool:
    """Return whether the Ipopt shared library exposes the required C interface."""
    probe = CachedIpopt.__new__(CachedIpopt)
    probe._dll_directories = []
    try:
        library = probe._load_library()
    except RuntimeError:
        return False
    return all(
        hasattr(library, name)
        for name in (
            "CreateIpoptProblem",
            "FreeIpoptProblem",
            "IpoptSolve",
            "AddIpoptStrOption",
            "AddIpoptNumOption",
            "AddIpoptIntOption",
        )
    )


class CachedIpopt:
    """Solve a PyNumero NLP in-process through Ipopt's standard C interface.

    The PyNumero NLP is built once by the caller. Parameter variables are fixed by their bounds
    for each solve, so changing their values does not require writing another NL file.

    Args:
        nlp: A PyNumero nonlinear program.
        options (dict): Ipopt solver options.

    Raises:
        RuntimeError: If the Ipopt shared library cannot be loaded.
    """

    def __init__(self, nlp, options: dict):
        self.nlp = nlp
        self.set_options(options)
        self._dll_directories = []
        self._lib = self._load_library()
        try:
            self._configure_library()
        except AttributeError as err:
            raise RuntimeError(
                "The Ipopt library does not expose the required C interface."
            ) from err
        self._jac = nlp.evaluate_jacobian()
        self._hess = nlp.evaluate_hessian_lag()
        self._lower_hess = self._hess.row >= self._hess.col
        self._callback_errors = []
        self._callbacks = self._create_callbacks()
        self._previous_multipliers = None

    def set_options(self, options: dict) -> None:
        """Set the options to use on the next solve."""
        self.options = {
            "print_level": 0,
            "sb": "yes",
            "fixed_variable_treatment": "make_parameter",
            **options,
        }
        if self.options["fixed_variable_treatment"] != "make_parameter":
            raise ValueError("Cached Ipopt requires fixed_variable_treatment='make_parameter'.")

    def _load_library(self):
        from pyomo.common import Executable

        executable = Executable("ipopt").path()
        directories = []
        if executable:
            directories.append(Path(executable).parent)
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            directories.append(Path(local_app_data) / "idaes" / "bin")
        directories.append(Path.home() / ".idaes" / "bin")
        if sys.platform == "win32":
            names = ("libipopt-3.dll", "ipopt.dll", "libipopt.dll")
        elif sys.platform == "darwin":
            names = ("libipopt.dylib",)
        else:
            names = ("libipopt.so", "libipopt.so.3", "libipopt.so.1")

        candidates = [str(directory / name) for directory in directories for name in names]
        found = ctypes.util.find_library("ipopt")
        if found:
            candidates.append(found)
        errors = []
        for candidate in dict.fromkeys(candidates):
            path = Path(candidate)
            if path.is_absolute() and not path.is_file():
                continue
            try:
                if sys.platform == "win32" and path.is_absolute():
                    self._dll_directories.append(os.add_dll_directory(str(path.parent)))
                return ctypes.CDLL(candidate)
            except OSError as err:
                errors.append(f"{candidate}: {err}")
        details = "; ".join(errors) if errors else "no Ipopt shared library was found"
        raise RuntimeError(f"Cached Ipopt requires a loadable Ipopt shared library: {details}")

    def _configure_library(self) -> None:
        lib = self._lib
        lib.CreateIpoptProblem.argtypes = [
            _INT,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _INT,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _INT,
            _INT,
            _INT,
            _EVAL_F,
            _EVAL_G,
            _EVAL_F,
            _EVAL_JAC,
            _EVAL_HESS,
        ]
        lib.CreateIpoptProblem.restype = _VOID_PTR
        lib.AddIpoptStrOption.argtypes = [_VOID_PTR, ctypes.c_char_p, ctypes.c_char_p]
        lib.AddIpoptNumOption.argtypes = [_VOID_PTR, ctypes.c_char_p, _DOUBLE]
        lib.AddIpoptIntOption.argtypes = [_VOID_PTR, ctypes.c_char_p, _INT]
        lib.IpoptSolve.argtypes = [
            _VOID_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _VOID_PTR,
        ]
        lib.IpoptSolve.restype = _INT
        lib.FreeIpoptProblem.argtypes = [_VOID_PTR]
        lib.FreeIpoptProblem.restype = None

    def _create_callbacks(self):
        nlp = self.nlp
        n, m = nlp.n_primals(), nlp.n_constraints()

        def guarded(function):
            def callback(*args):
                # Exceptions cannot unwind through Ipopt's C callback.
                try:
                    function(*args)
                except BaseException as err:  # noqa: BLE001
                    self._callback_errors.append(err)
                    return 0
                return 1

            return callback

        def set_primals(x):
            nlp.set_primals(np.ctypeslib.as_array(x, shape=(n,)))

        @_EVAL_F
        @guarded
        def objective(_n, x, _new, output, _data):
            set_primals(x)
            output[0] = nlp.evaluate_objective()

        @_EVAL_F
        @guarded
        def gradient(_n, x, _new, output, _data):
            set_primals(x)
            np.ctypeslib.as_array(output, shape=(n,))[:] = nlp.evaluate_grad_objective()

        @_EVAL_G
        @guarded
        def constraints(_n, x, _new, _m, output, _data):
            set_primals(x)
            np.ctypeslib.as_array(output, shape=(m,))[:] = nlp.evaluate_constraints()

        @_EVAL_JAC
        @guarded
        def jacobian(_n, x, _new, _m, count, rows, columns, output, _data):
            if not output:
                np.ctypeslib.as_array(rows, shape=(count,))[:] = self._jac.row
                np.ctypeslib.as_array(columns, shape=(count,))[:] = self._jac.col
            else:
                set_primals(x)
                nlp.evaluate_jacobian(out=self._jac)
                np.ctypeslib.as_array(output, shape=(count,))[:] = self._jac.data

        @_EVAL_HESS
        @guarded
        def hessian(
            _n, x, _new, factor, _m, duals, _new_duals, count, rows, columns, output, _data
        ):
            if not output:
                np.ctypeslib.as_array(rows, shape=(count,))[:] = self._hess.row[self._lower_hess]
                np.ctypeslib.as_array(columns, shape=(count,))[:] = self._hess.col[self._lower_hess]
            else:
                set_primals(x)
                nlp.set_duals(np.ctypeslib.as_array(duals, shape=(m,)))
                nlp.set_obj_factor(factor)
                nlp.evaluate_hessian_lag(out=self._hess)
                np.ctypeslib.as_array(output, shape=(count,))[:] = self._hess.data[self._lower_hess]

        return objective, constraints, gradient, jacobian, hessian

    def solve(self, primals: np.ndarray, parameter_indices: np.ndarray, values: np.ndarray):
        """Return primal variables and Ipopt constraint and bound multipliers."""
        n, m = self.nlp.n_primals(), self.nlp.n_constraints()
        if not np.all(np.isfinite(primals)) or not np.all(np.isfinite(values)):
            raise ValueError("Cached Ipopt requires finite initial and parameter values.")

        def bounds(data):
            result = np.asarray(data, dtype=float).copy()
            if np.any(np.isnan(result)):
                raise ValueError("Cached Ipopt cannot use NaN bounds.")
            result[np.isneginf(result)] = -1e300
            result[np.isposinf(result)] = 1e300
            return result

        lower = bounds(self.nlp.primals_lb())
        upper = bounds(self.nlp.primals_ub())
        lower[parameter_indices] = values
        upper[parameter_indices] = values
        constraint_lower = bounds(self.nlp.constraints_lb())
        constraint_upper = bounds(self.nlp.constraints_ub())
        nnz_jac = self._jac.nnz
        nnz_hess = int(np.count_nonzero(self._lower_hess))
        if max(n, m, nnz_jac, nnz_hess) > np.iinfo(np.int32).max:
            raise ValueError("Cached Ipopt requires a problem within 32-bit index limits.")
        problem = self._lib.CreateIpoptProblem(
            n,
            _double_ptr(lower),
            _double_ptr(upper),
            m,
            _double_ptr(constraint_lower),
            _double_ptr(constraint_upper),
            nnz_jac,
            nnz_hess,
            0,
            *self._callbacks,
        )
        if not problem:
            raise RuntimeError("Ipopt could not create the cached problem.")
        try:
            options = self.options.copy()
            warm_start_enabled = options.get("warm_start_init_point") == "yes"
            use_warm_start = warm_start_enabled and self._previous_multipliers is not None
            if warm_start_enabled and not use_warm_start:
                options["warm_start_init_point"] = "no"
            for name, value in options.items():
                key = name.encode("utf-8")
                if isinstance(value, str):
                    success = self._lib.AddIpoptStrOption(problem, key, value.encode("utf-8"))
                elif isinstance(value, Integral):
                    success = self._lib.AddIpoptIntOption(problem, key, int(value))
                elif isinstance(value, Real):
                    success = self._lib.AddIpoptNumOption(problem, key, float(value))
                else:
                    raise TypeError(f"Unsupported Ipopt option value for {name}: {value!r}")
                if not success:
                    raise ValueError(f"Invalid Ipopt option {name}={value!r}.")

            x = np.asarray(primals, dtype=float).copy()
            x[parameter_indices] = values
            constraint_values = np.zeros(m)
            objective = np.zeros(1)
            if use_warm_start:
                duals, z_lower, z_upper = (value.copy() for value in self._previous_multipliers)
            else:
                duals = np.zeros(m)
                z_lower, z_upper = np.zeros(n), np.zeros(n)
            self._callback_errors.clear()
            status = self._lib.IpoptSolve(
                problem,
                _double_ptr(x),
                _double_ptr(constraint_values),
                _double_ptr(objective),
                _double_ptr(duals),
                _double_ptr(z_lower),
                _double_ptr(z_upper),
                None,
            )
            if self._callback_errors:
                self._previous_multipliers = None
                error = self._callback_errors[0]
                if not isinstance(error, Exception):
                    raise error
                raise RuntimeError(
                    "Cached Ipopt could not evaluate the nonlinear program."
                ) from error
            if status != 0:
                self._previous_multipliers = None
                raise RuntimeError(f"Cached Ipopt did not converge (status {status}).")
            if not all(np.all(np.isfinite(v)) for v in (x, duals, z_lower, z_upper)):
                self._previous_multipliers = None
                raise RuntimeError("Cached Ipopt returned nonfinite primal or multiplier values.")
            self._previous_multipliers = (
                (duals.copy(), z_lower.copy(), z_upper.copy()) if warm_start_enabled else None
            )
            return x, duals, z_lower, z_upper
        finally:
            self._lib.FreeIpoptProblem(problem)
