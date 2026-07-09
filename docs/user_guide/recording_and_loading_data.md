# Recording and Loading Data From H2I Simulations
Detailed data from H2Integrate simulations can be saved and loaded later. This section covers:
1. [How to record data from a simulation](setting-recorder-parameters)
2. [How to load data and access recorded data](loading-recorder-files)

(setting-recorder-parameters)=
## Setting Recorder Parameters

Recording data from a simulation can be enabled in the `driver_config` file with the 'recorder' section. The most simple example is shown below:

```yaml
name: "driver_config"
description: "example driver config to show recording options"

general:
  folder_output: outputs #folder to save recorded data to

recorder:
  flag: True #set to True to record data
  file: "cases.sql" #this is the name of the file to record data to.
```
In the above example, data will be saved to the file `outputs/cases.sql`.

The file below would have the same behavior as the above example, but highlights the default behavior:
```yaml
name: "driver_config"
description: "example driver config to show recording options"

general:
  folder_output: outputs #folder to save recorded data to

recorder:
  flag: True #set to True to record data
  file: "cases.sql" #this is the name of the file to record data to.
  overwrite_recorder: False #create a unique recorder for each simulation
  recorder_attachment: "model" # "driver", "model", or "problem"
  includes: ["*"] # include everything
  excludes: ["*resource_data"] # don't include resource data
```

- **overwrite_recorder**: If False or not specified, H2I will make a new filename for the recorder that doesn't yet exist in the `outputs` folder. If `cases.sql` exists, it will make a new file named `cases0.sql`. If `cases.sql` and `cases0.sql` exist, it will make a new file named `cases1.sql`, etc. If set to True, it will overwrite an existing `cases.sql` file from previous runs.
- **recorder_attachment**: Must be `model`, `driver`, or `problem`, defaults to `model`. We recommend attaching the recorder to the driver if running an optimization or design of experiments in parallel as model-level recording cannot be performed in parallel due to limitations in OpenMDAO. Use `problem` to record only the final design point of an optimization case (see [Attaching a recorder to the problem](attaching-a-recorder-to-the-problem)).

You may also define more than one recorder at once by providing a list under `recorder`; see [Configuring multiple recorders](configuring-multiple-recorders).

(attaching-a-recorder-to-the-driver)=
### Attaching a recorder to the driver
We recommend attaching the recorder to the driver if running an optimization or design of experiments in parallel as model-level recording cannot be performed in parallel due to limitations in OpenMDAO.
It can be beneficial if running a design of experiments or optimization in serial as well.
Further documentation on driver recording can be found [here](https://openmdao.org/newdocs/versions/latest/features/recording/driver_recording.html).

```yaml
name: "driver_config"
description: "example driver config to show recording options"

general:
  folder_output: outputs #folder to save recorded data to

recorder:
  flag: True #set to True to record data
  file: "cases.sql" #this is the name of the file to record data to.
  overwrite_recorder: False #create a unique recorder for each simulation
  recorder_attachment: "driver" #"driver" or "model

  # H2I Default recorder options
  includes: ["*"] # include everything
  excludes: ["*resource_data"] # don't include resource data

  # OpenMDAO default for recording options
  record_inputs: True #record inputs
  record_outputs: True #record outputs
  record_constraints: True #record constraints
  record_derivatives: False #record derivatives
  record_desvars: True #record design variables
  record_objectives: True #record objectives
```

(attaching-a-recorder-to-the-model)=
### Attaching a recorder to the model
Further documentation on model recording can be found [here](https://openmdao.org/newdocs/versions/latest/features/recording/system_recording.html). By default, the recorder will be attached to the model unless `recorder_attachment` is set to "driver".

```yaml
name: "driver_config"
description: "example driver config to show recording options"

general:
  folder_output: outputs #folder to save recorded data to

recorder:
  flag: True #set to True to record data
  file: "cases.sql" #this is the name of the file to record data to.
  overwrite_recorder: False #create a unique recorder for each simulation
  recorder_attachment: "model" #"driver" or "model

  # H2I Default recorder options
  includes: ["*"] # include everything
  excludes: ["*resource_data"] # don't include resource data

  # OpenMDAO default for recording options
  record_inputs: True #record inputs
  record_outputs: True #record outputs
  record_residuals: True #record residuals
```

(attaching-a-recorder-to-the-problem)=
### Attaching a recorder to the problem
Attaching a recorder to the problem records only the final design point of an optimization case, rather than every driver iteration. This is useful when you only care about the converged/last design and want a small file that captures the final state. The final case is written after the driver has finished running.
Further documentation on problem recording can be found [here](https://openmdao.org/newdocs/versions/latest/features/recording/problem_recording.html).

```yaml
name: "driver_config"
description: "example driver config to show recording options"

general:
  folder_output: outputs #folder to save recorded data to

recorder:
  flag: True #set to True to record data
  file: "final_point.sql" #this is the name of the file to record data to.
  overwrite_recorder: False #create a unique recorder for each simulation
  recorder_attachment: "problem" #"driver", "model", or "problem"

  # H2I Default recorder options
  includes: ["*"] # include everything
  excludes: ["*resource_data"] # don't include resource data

  # OpenMDAO default for recording options
  record_inputs: True #record inputs
  record_outputs: True #record outputs
  record_residuals: True #record residuals
```

The resulting file contains a single case, which is read from the `"problem"` source:
```python
import openmdao.api as om
from pathlib import Path

fpath = Path.cwd() / "outputs" / "final_point.sql"
cr = om.CaseReader(fpath)

# the problem recorder stores exactly one case: the final design point
final_case = cr.get_cases("problem")[0]
final_case.get_val("finance_subgroup_hydrogen.LCOH")
```

(configuring-multiple-recorders)=
### Configuring multiple recorders
The `recorder` entry may also be a list of recorder configurations, allowing several recorders to run at once. Each recorder writes to its own file and may record different variables via its own `includes`/`excludes` statements and its own `recorder_attachment`. This is a convenient way to keep, for example, the full optimization history on the driver while also saving a compact record of just the final design point on the problem.

```yaml
name: "driver_config"
description: "example driver config to show multiple recorders"

general:
  folder_output: outputs #folder to save recorded data to

recorder:
  # Recorder 1: full optimization history on the driver
  - flag: True
    file: "full_history.sql"
    recorder_attachment: "driver"
    includes: ["*"]
    excludes: ["*resource_data"]

  # Recorder 2: only the final design point on the problem
  - flag: True
    file: "final_point.sql"
    recorder_attachment: "problem"
    includes: ["*"]
    excludes: ["*resource_data"]
```

Each file is loaded independently with `om.CaseReader`, as shown in [Loading Recorder Files](loading-recorder-files). When multiple recorders are configured, `H2IntegrateModel.recorder_paths` holds the paths to all of them, while `H2IntegrateModel.recorder_path` points to the first one for backward compatibility.

(loading-recorder-files)=
## Loading Recorder Files
Detailed documentation on OpenMDAO's case read can be found [here](https://openmdao.org/newdocs/versions/latest/features/recording/case_reader.html).

Example usage of reading and accessing recorded data is shown in Example 8 (`examples/08_wind_electrolyzer/run_wind_electrolyzer.py`).

Below is an example python script that shows how to load recorded data and to access the data available:

```python
import openmdao.api as om
from pathlib import Path
# set the path for the recorder from stuff specified in the driver_config.yaml
fpath = Path.cwd() / "outputs" / "cases.sql"

# load the cases
cr = om.CaseReader(fpath)

# get the cases as a list
cases = list(cr.get_cases())
# access a variable from the problem, this can be anything thats an
# input or output from the models that were run and not specified
# as variables to exclude in the driver_config file

cases[0].get_val("finance_subgroup_default.LCOE",units="USD/(kW*h)")
cases[0].get_val("solar.system_capacity_DC",units="MW")
```
