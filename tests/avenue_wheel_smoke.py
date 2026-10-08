"""Exercise an installed fork wheel, its native penalties, and stock coexistence."""
from importlib.metadata import version
from pathlib import Path
import tempfile

import numpy as np
print("Import stock LightGBM", flush=True)
import lightgbm as stock
print("Import Avenue LightGBM", flush=True)
import avenue_lightgbm as avenue
print("Both imports succeeded", flush=True)


def split_features(node):
    if "split_feature" not in node:
        return set()
    return {node["split_feature"]} | split_features(node["left_child"]) | split_features(node["right_child"])


def main():
    assert avenue.__version__ == version("avenue-lightgbm")
    assert Path(avenue.__file__).parent.name == "avenue_lightgbm"
    assert Path(stock.__file__).parent.name == "lightgbm"
    assert Path(avenue.basic._LIB._name).resolve() != Path(stock.basic._LIB._name).resolve()
    assert "avenue_lightgbm" in Path(avenue.basic._LIB._name).parts

    rng = np.random.default_rng(47)
    x = rng.normal(size=(512, 3))
    y = 3 * x[:, 0] + 2 * x[:, 1] + rng.normal(scale=0.1, size=512)
    params = {"objective": "regression", "verbosity": -1, "num_threads": 2,
              "num_leaves": 7, "min_data_in_leaf": 16, "min_gain_to_split": 0.1,
              "seed": 47}
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
            assert all(len(split_features(tree["tree_structure"])) <= 1
                       for tree in penalized.dump_model()["tree_info"]), name
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


if __name__ == "__main__":
    main()
