# Avenue LightGBM

**The training backend for Avenue's interpretable machine learning.**

[Avenue Model](https://github.com/lukemuz/Avenue_Model) handles tuning, exact conversion
into editable tables, inspection, scoring and optional GLM refitting. This backend
adds two training penalties that encourage fewer, simpler feature interactions,
so the resulting tables are easier to understand.

Install it alongside stock LightGBM: the package is `avenue-lightgbm`, and the Python
import is `avenue_lightgbm`. Avenue Model automatically prefers this backend when it
creates a dataset through `resolve_lightgbm()`.

## Install

Release wheels are installed with ordinary pip and need no C++ compiler. Download the
wheel for your system from [Releases](https://github.com/lukemuz/avenue-lightgbm/releases):

```sh
python -m pip install ./avenue_lightgbm-VERSION-py3-none-PLATFORM.whl
python -c "import avenue_lightgbm; print(avenue_lightgbm.__version__)"
```

Replace the placeholder filename with the downloaded file. You can also pass its
GitHub release download URL directly to `pip install`.
**Until the first Avenue wheel release is published, use the source installation below.**

The wheel workflow targets Linux x86-64/ARM64, macOS Intel/Apple Silicon, and Windows
x86-64. Wheels bundle their OpenMP dependency. They use the `py3-none` tag because
the native library is loaded through ctypes, without a CPython-specific ABI.
Python 3.9+ is supported; use Python 3.12 or 3.13 with Avenue Model.

Install Avenue Model separately using its
[installation instructions](https://github.com/lukemuz/Avenue_Model#installation),
including its `[tuning]` extra for Optuna. Installing this backend does not replace
stock `lightgbm`; the two libraries have separate package directories.

## Use it through Avenue Model

With your numerical predictors `X`, response `y` and a Polars frame `quotes`:

```python
from avenue_model import from_booster, resolve_lightgbm, tune_lgbm

backend, _ = resolve_lightgbm()  # Prefers avenue_lightgbm when installed.
dataset = backend.Dataset(X, label=y, feature_name=list(quotes.columns))
search = tune_lgbm(dataset, {"objective": "poisson"}, n_trials=50)
print(search.summary())
selected = search.select(max_tables=10)  # Screens mean CV table count.
booster = backend.train({**selected.params, "num_iterations": selected.num_iterations}, dataset)
conversion = from_booster(booster, quotes)
print(conversion.parity)
print(conversion.metadata["complexity"])  # Inspect the final artifact too.
conversion.save("rating_plan")
```

Train and cross-validate on training data; use separate quotes for validation.
Start with the [Avenue workflow guide](https://github.com/lukemuz/Avenue_Model/blob/main/docs/lightgbm.md)
for complete examples, exposure conventions, inspecting tables and GLM refitting.

You can also train directly with `import avenue_lightgbm as lgb`; it provides the
usual LightGBM `Dataset`, `train`, `cv` and `Booster` APIs.

## What the penalties do

| Parameter | Effect |
|---|---|
| `interaction_penalty` | Subtracts a penalty from split gain when adding a feature creates a combination not represented in earlier trees. |
| `interaction_complexity` | Divides split gain by an increasing penalty when introducing a new feature into the current tree. |

Both default to zero. Neither is a hard limit on interaction depth or table count.
Avenue Model tunes them alongside predictive accuracy and measures the resulting
converted tables. These controls apply to CPU training; CUDA ignores them.

## Install from source

Requires Git, a C++17 compiler and an OpenMP runtime. On Ubuntu/Debian install
`build-essential`; on macOS install the Xcode command-line tools and `brew install libomp`;
on Windows use Visual Studio Build Tools with C++ support. Pip supplies CMake and Ninja
as build dependencies where needed.

```sh
git clone --recursive https://github.com/lukemuz/avenue-lightgbm.git
cd avenue-lightgbm
python -m pip install .
```

For an existing checkout, first run `git submodule update --init --recursive`.
Run pip from the repository root: the inherited `python-package/` and `build-python.sh`
entry points retain upstream packaging under the name `lightgbm`.

## Build and release

```sh
python -m pip install build
python -m build
python -m pip install ./dist/ACTUAL_WHEEL_FILENAME.whl lightgbm
python tests/avenue_wheel_smoke.py
```

The smoke check verifies both penalties affect trained models, saved models reload,
and stock LightGBM remains usable in the same process. The **Avenue wheels** GitHub
Actions workflow builds, repairs and tests platform wheels on pull requests or manual
runs; its artifacts can be downloaded before a release. It also rebuilds from the
source archive to verify that users do not need Git submodules when installing it.

To release, update the version in the root `pyproject.toml`, merge, and push the matching
`vVERSION` tag. After all builds pass, the workflow creates a **draft GitHub release**
with wheels, a source archive and checksums. Review its assets and publish the draft.
No PyPI account is required.

[Upstream LightGBM documentation](README.upstream.md) ·
[Research methodology](https://avenue-analytics.com/research/avenue-analytics-methodology.pdf) ·
[MIT license](LICENSE)
