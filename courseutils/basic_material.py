"""
basic_material.py
Shared utilities for Signals and Systems.
Students should not modify this file.
All environment setup is opt in via setup_environment().
"""
__version__ = "16.002-0.1"

import sys
import importlib.util
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    from simple_colors import blue, green
except ImportError:
    def blue(msg, *args, **kwargs):
        return str(msg)

    def green(msg, *args, **kwargs):
        return str(msg)

float_formatter = "{:.4f}".format

# -------------------------------
# Version and environment helpers
# -------------------------------
PYTHON_VERSION = sys.version_info

R2D = 180 / np.pi
RPS2HZ = 1 / (2 * np.pi)

# -------------------------------
# Paths
# -------------------------------
DATA_DIR = Path("./data")
FIG_DIR = Path("./figs")

# -------------------------------
# Public setup function
# -------------------------------
def setup_environment(
    *,
    verbose=False,
    set_plot_style=True,
    set_numpy_print=True,
    create_dirs=True,
    check_packages=True
):
    """
    Perform course standard environment setup.
    Parameters
    ----------
    verbose : bool
        Print Python and SymPy versions.
    set_plot_style : bool
        Apply course matplotlib style.
    set_numpy_print : bool
        Apply course NumPy print formatting.
    create_dirs : bool
        Create ./data and ./figs if missing.
    check_packages : bool
        Check that required packages are installed.
    """
    if check_packages:
        _check_required_packages()
    if verbose:
        from platform import python_version
        import sympy as sym
        print("Running Python:", python_version())
        print("Running SymPy:", sym.__version__)
    if set_plot_style:
        _set_plot_style()
    if set_numpy_print:
        _set_numpy_print_options()
    if create_dirs:
        DATA_DIR.mkdir(exist_ok=True)
        FIG_DIR.mkdir(exist_ok=True)
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="This figure includes Axes that are not compatible with tight_layout"
    )

def _set_numpy_print_options():
    np.set_printoptions(formatter={"float": "{: 8.3f}".format})

# -------------------------------
# Plotting style
# -------------------------------

def _set_plot_style():
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams["font.serif"] = "cmr14"
    rcParams.update({"font.size": 18})

    plt.rcParams["figure.figsize"] = [8, 5.0]
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["axes.xmargin"] = 0
    plt.rcParams["axes.grid"] = True
    plt.rcParams["figure.autolayout"] = True

    SMALL = 10
    BIG = 18

    plt.rc("font", size=SMALL)
    plt.rc("axes", titlesize=SMALL)
    plt.rc("axes", labelsize=SMALL)
    plt.rc("xtick", labelsize=SMALL)
    plt.rc("ytick", labelsize=SMALL)
    plt.rc("legend", fontsize=SMALL)
    plt.rc("figure", titlesize=BIG)

# -------------------------------
# Package checks
# -------------------------------

def _require_package(name):
    if importlib.util.find_spec(name) is None:
        raise ImportError(
            f"Required package '{name}' not found. "
            "Please install it following the course instructions."
        )

def _check_required_packages():
    _require_package("matplotlib")
    _require_package("numpy")
    _require_package("scipy")
    _require_package("sympy")
    _require_package("control")

# -------------------------------
# Numeric helpers
# -------------------------------

# -------------------------------
# Plot helpers
# -------------------------------

def get_colors():
    """
    Return a list of Matplotlib-safe color names.
    """
    return [
        "blue", "red", "darkgreen", "magenta", "black",
        "salmon", "brown", "darkblue", "tomato", "violet",
        "tan", "pink", 'SaddleBrown', 'SpringGreen', 'RosyBrown','Silver',]

def print_blue(msg):
    print(blue(msg))
def print_green(msg):
    print(green(msg, 'bold'))
#print(green('hello', ['bold', 'underlined']))

def nicegrid(ax=None, hh=None): #hh is legacy
    """
    Apply standard grid styling to one or more axes.
    """
    if ax is None:
        ax = plt.gca()  # current axes

    for a in np.asarray(ax, dtype=object).ravel():
        _jgrid(a)

def _jgrid(ax):
    ax.grid(True, which="major", color="#666666", linestyle=":")
    ax.grid(True, which="minor", color="#999999", linestyle=":", alpha=0.2)

    if ax.get_yscale() != "log":
        ax.axhline(y=0, color="k", linestyle="-", lw=1)
    else:
        ax.axhline(y=1, color="k", linestyle="--", lw=1)

    if ax.get_xscale() != "log":
        ax.axvline(x=0, color="k", linestyle="-", lw=1)

    ax.minorticks_on()

def caption(txt, fig, xloc=0.5, yloc=-0.05):
    """
    Add a caption below a figure.
    """
    fig.text(xloc, yloc, txt, ha="center", size=14, color="blue")

# -------------------------------
# Line style presets
# -------------------------------

LOOSELY_DOTTED = (0, (1, 10))
DENSELY_DOTTED = (0, (1, 1))
LOOSELY_DASHED = (0, (5, 10))
DENSELY_DASHED = (0, (5, 1))
LOOSELY_DASHDOTTED = (0, (3, 10, 1, 10))
DENSELY_DASHDOTTED = (0, (3, 1, 1, 1))

ORDINALS = [
    "One", "Two", "Three", "Four", "Five",
    "Six", "Seven", "Eight", "Nine", "Ten"]

from itertools import cycle
lines = ["-","--","-.",":"]
linecycler = cycle(lines)

# ------------------------------------------------------------------
# Self-update helper (instructor provided)
# ------------------------------------------------------------------

def ensure_version(
    required_version,
    *,
    url_base="https://raw.githubusercontent.com/mit-acl/Signals_and_Systems/main/courseutils/",
    filename="basic_material.py",
    verbose=True,
):
    """
    Ensure this module matches the required version.

    If the version is missing or mismatched, attempt to download the
    correct file and reload the module.

    Parameters
    ----------
    required_version : str
        Expected __version__ string.
    url_base : str
        Base URL where the file is hosted.
    filename : str
        Local filename of this module.
    verbose : bool
        Print status messages.
    """
    import sys
    import time
    import shutil
    import requests
    import importlib
    from pathlib import Path

    current = globals().get("__version__", None)
    if current == required_version:
        if verbose:
            print(f"basic_material OK (version {current})")
        return

    if verbose:
        print(
            f"basic_material version mismatch "
            f"(found {current!r}, need {required_version!r}). "
            "Attempting update..."
        )

    filename_path = Path(filename)
    path = (
        filename_path
        if filename_path.is_absolute()
        else Path(__file__).resolve().parent / filename_path
    )
    url = url_base.rstrip("/") + "/" + filename_path.name

    # download
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")

    # backup existing file
    if path.exists():
        bak = path.with_suffix(f".bak.{int(time.time())}")
        shutil.copy2(path, bak)
        if verbose:
            print(f"Backed up existing file to {bak.name}")

    # atomic write
    tmp = path.with_suffix(".tmp")
    tmp.write_text(r.text, encoding="utf-8")
    tmp.replace(path)

    # reload module
    mod = sys.modules.get(__name__)
    if mod is None:
        raise RuntimeError("Internal error: module not found in sys.modules")

    importlib.reload(mod)

    new_version = globals().get("__version__", None)
    if new_version != required_version:
        raise RuntimeError(
            f"Update failed: expected version {required_version!r}, "
            f"found {new_version!r} after reload"
        )

    if verbose:
        print(f"basic_material updated successfully to version {new_version}")
