"""Exercise an installed fork wheel, its native penalties, and stock coexistence."""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np


def split_features(node):
    if "split_feature" not in node:
        return set()
    return {node["split_feature"]} | split_features(node["left_child"]) | split_features(node["right_child"])


def main():
    if "--fork-first" in sys.argv:
        print("Import Avenue, then stock LightGBM", flush=True)
        avenue = importlib.import_module("avenue_lightgbm")
        stock = importlib.import_module("lightgbm")
    else:
        print("Import stock, then Avenue LightGBM", flush=True)
        stock = importlib.import_module("lightgbm")
        avenue = importlib.import_module("avenue_lightgbm")
    print("Both imports succeeded", flush=True)
    assert avenue.__version__ == version("avenue-lightgbm")
    assert Path(avenue.__file__).parent.name == "avenue_lightgbm"
    assert Path(stock.__file__).parent.name == "lightgbm"
    assert Path(avenue.basic._LIB._name).resolve() != Path(stock.basic._LIB._name).resolve()
    assert "avenue_lightgbm" in Path(avenue.basic._LIB._name).parts

    rng = np.random.default_rng(47)
    x = rng.normal(size=(512, 3))
    y = 3 * x[:, 0] + 2 * x[:, 1] + rng.normal(scale=0.1, size=512)
    params = {
        "objective": "regression",
        "verbosity": -1,
        "num_threads": 2,
        "num_leaves": 7,
        "min_data_in_leaf": 16,
        "min_gain_to_split": 0.1,
        "seed": 47,
    }
    data = avenue.Dataset(x, label=y, free_raw_data=False)
    print("Train Avenue baseline", flush=True)
    baseline = avenue.train(params, data, num_boost_round=5)
    assert np.std(baseline.predict(x)) > 0.1
    assert any(len(split_features(tree["tree_structure"])) > 1 for tree in baseline.dump_model()["tree_info"])
    # An accepted parameter name alone is insufficient: each penalty must affect fits.
    for name in ("interaction_penalty", "interaction_complexity"):
        print("Train with", name, flush=True)
        penalized = avenue.train({**params, name: 1e12}, data, num_boost_round=5)
        if name == "interaction_penalty":
            assert np.std(penalized.predict(x)) < 1e-8, name
            assert all(tree["num_leaves"] == 1 for tree in penalized.dump_model()["tree_info"]), name
        else:
            # Complexity scales gain, rather than imposing a hard split cutoff.
            assert all(
                len(split_features(tree["tree_structure"])) <= 1 for tree in penalized.dump_model()["tree_info"]
            ), name
            assert not np.allclose(penalized.predict(x), baseline.predict(x)), name

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "booster.txt"
        baseline.save_model(str(path))
        restored = avenue.Booster(model_file=str(path))
        np.testing.assert_allclose(restored.predict(x), baseline.predict(x), rtol=1e-12, atol=1e-12)
    # Stock LightGBM remains independently usable in the same interpreter.
    print("Train stock LightGBM", flush=True)
    reference = stock.train(params, stock.Dataset(x, label=y), num_boost_round=5)
    assert np.isfinite(reference.predict(x)).all()
    assert np.std(reference.predict(x)) > 0.1
    print(f"avenue-lightgbm {avenue.__version__}: both penalties, model reload and stock coexistence passed")


def check_macos_fallback():
    """Exercise the bundled runtime without changing the machine's installed libraries."""
    package = Path(importlib.util.find_spec("avenue_lightgbm").origin).parent
    if not (package / ".dylibs/libomp.dylib").is_file():
        return  # Source installs use the build machine's OpenMP runtime.
    with tempfile.TemporaryDirectory() as directory:
        copied = Path(directory) / "avenue_lightgbm"
        shutil.copytree(package, copied, ignore=shutil.ignore_patterns("__pycache__"))
        native = copied / "lib/lib_lightgbm.dylib"
        for path in ("/opt/homebrew/opt/libomp/lib", "/usr/local/opt/libomp/lib", "/opt/local/lib/libomp"):
            subprocess.run(["install_name_tool", "-delete_rpath", path, str(native)], check=True)
        subprocess.run(["codesign", "--force", "--sign", "-", str(native)], check=True)
        code = """
import ctypes
from pathlib import Path
import avenue_lightgbm as lgb
import numpy as np
assert Path(lgb.__file__).resolve().parent == Path.cwd() / 'avenue_lightgbm'
x = np.arange(100., dtype=float).reshape(-1, 1)
model = lgb.train({'objective': 'regression', 'verbosity': -1, 'num_threads': 2},
                  lgb.Dataset(x, label=x[:, 0]), num_boost_round=3)
assert np.std(model.predict(x)) > 0
# Confirm the fallback library was loaded from this copied wheel.
loader = ctypes.CDLL(None)
loader._dyld_image_count.restype = ctypes.c_uint32
loader._dyld_get_image_name.argtypes = [ctypes.c_uint32]
loader._dyld_get_image_name.restype = ctypes.c_char_p
runtimes = [loader._dyld_get_image_name(i).decode() for i in range(loader._dyld_image_count())
            if loader._dyld_get_image_name(i).decode().endswith('/libomp.dylib')]
assert len(runtimes) == 1 and Path(runtimes[0]).resolve().is_relative_to(Path.cwd()), runtimes
print('Bundled macOS OpenMP fallback passed')
"""
        env = {**os.environ, "PYTHONPATH": directory}
        subprocess.run([sys.executable, "-X", "faulthandler", "-c", code], cwd=directory, env=env, check=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        for order in ("--stock-first", "--fork-first"):
            subprocess.run([sys.executable, "-X", "faulthandler", "-u", __file__, order], check=True)
        if sys.platform == "darwin":
            check_macos_fallback()
