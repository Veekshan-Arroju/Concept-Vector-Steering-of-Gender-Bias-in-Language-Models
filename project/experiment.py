"""
Experiment namespacing.

Every output of the pipeline lives under an experiment tag, so two experiments (for example the default
random-frame run and a fixed-frame run, or an English run and a Hindi run) can sit side by side
without overwriting each other:

    dataset/processed/<experiment>/            prompts.csv, train_datasets.csv, paired_test.csv, heldout_test.csv, manifest.json
    activations/<model_slug>/<experiment>/     meta.json, layer_neg<k>.pt
    concept_vectors/<model_slug>/<experiment>/ <vector>.pkl, _similarity.pkl
    probe_results/, direction_probe_results/, monitor_results/, control_results/, comparison_results/, results/
                                               <model_slug>/<experiment>/...

Choose the tag with  --experiment <tag>  on any script, or the EXPERIMENT environment variable
(default: "main"). The tag is just a folder name: letters, digits, "_", "-" and ".".
"""

import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_VALID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _check(name: str) -> str:
    if not _VALID.match(name or ""):
        raise ValueError(f"Invalid experiment tag {name!r}: use letters, digits, '_', '-' and '.' (it becomes a folder name).")
    return name


_state = {"experiment": _check(os.getenv("EXPERIMENT", "main"))}


def get_experiment() -> str:
    return _state["experiment"]


def set_experiment(name: str) -> str:
    """Switch the active experiment for this process (and for child processes via the environment)."""
    _state["experiment"] = _check(name)
    os.environ["EXPERIMENT"] = name
    return name


from contextlib import contextmanager


@contextmanager
def using(name: str):
    """Temporarily switch the active experiment:  with using("main"): ...  (restored afterwards, even on errors)."""
    old = get_experiment()
    set_experiment(name)
    try:
        yield
    finally:
        set_experiment(old)


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace("-", "_")


def dataset_dir() -> Path:
    return PROJECT_ROOT / "dataset" / "processed" / get_experiment()


def exp_dir(root: Path, model_id: str) -> Path:
    """<root>/<model_slug>/<experiment>"""
    return Path(root) / model_slug(model_id) / get_experiment()


def add_experiment_arg(parser) -> None:
    parser.add_argument("--experiment", default=None,
                        help="experiment tag that namespaces every output folder (default: $EXPERIMENT or 'main')")


def make_logs_readable() -> None:
    """
    Line-buffer stdout. Python block-buffers stdout when it is redirected to a file (python script.py > log 2>&1), so a
    detached run's log only showed a print when a progress bar happened to start, or at exit: progress counters looked
    stuck at 0 although the job was running. Every command-line script calls this through apply_experiment_arg.
    """
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:       # a captured or replaced stdout (tests, notebooks) has nothing to reconfigure
        pass


def apply_experiment_arg(args) -> str:
    """Call right after parser.parse_args()."""
    make_logs_readable()
    if getattr(args, "experiment", None):
        set_experiment(args.experiment)
    print(f"[experiment] {get_experiment()}")
    return get_experiment()
