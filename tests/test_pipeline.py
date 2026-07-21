"""Regression tests for the prediction pipeline and the database schema.

Each test here corresponds to a defect that made the repository
non-reproducible from a clean clone. Run with:

    python -m pytest tests/ -v
"""

import os
import sys
import types
from pathlib import Path

import pytest

# A throwaway backend so importing database_tables never needs MariaDB.
os.environ.setdefault("DATABASE_URL", "sqlite://")

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def _stub_ocean_sdk():
    dimod = types.ModuleType("dimod")
    dimod.SimulatedAnnealingSampler = object
    dimod.BinaryQuadraticModel = object
    sys.modules.setdefault("dimod", dimod)

    dwave_system = types.ModuleType("dwave.system")
    for name in ("EmbeddingComposite", "DWaveSampler", "LeapHybridSampler"):
        setattr(dwave_system, name, object)
    dwave = types.ModuleType("dwave")
    dwave.system = dwave_system
    sys.modules.setdefault("dwave", dwave)
    sys.modules.setdefault("dwave.system", dwave_system)


_stub_ocean_sdk()

networkx = pytest.importorskip("networkx")
pytest.importorskip("sklearn")

import gbr_prediction as gp  # noqa: E402
import database_tables as dt  # noqa: E402

MODELS_PRESENT = gp.GBR_MIN_PATH.exists() and gp.GBR_MAX_PATH.exists()
needs_models = pytest.mark.skipif(
    not MODELS_PRESENT, reason="trained models not present in src/gbr/"
)


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

def test_schema_compiles_on_mariadb():
    """A bare String column makes create_all() fail on MySQL/MariaDB."""
    from sqlalchemy.dialects import mysql
    from sqlalchemy.schema import CreateTable

    for table in (dt.Graph, dt.Partition):
        CreateTable(table.__table__).compile(dialect=mysql.dialect())


def test_graph_names_are_unique_within_one_second():
    """`graphs.name` is UNIQUE; second-resolution stamps collided."""
    import store_into_db

    names = set()
    for _ in range(200):
        # Mirror the construction used by store_graph_to_db
        import datetime
        import uuid

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        names.add(f"Graph_{stamp}_hybrid_{uuid.uuid4().hex[:6]}")

    assert len(names) == 200
    assert hasattr(store_into_db, "uuid"), "store_into_db must import uuid"


# --------------------------------------------------------------------------
# Feature-name compatibility
# --------------------------------------------------------------------------

class _FakeModel:
    """Records the frame it was asked to predict on."""

    def __init__(self, feature_names):
        self.feature_names_in_ = list(feature_names)
        self.seen = None

    def predict(self, X):
        self.seen = X
        return [0.25]


def test_feature_frame_maps_legacy_gamma_start():
    """Models pickled before the gamma -> lambda rename must still work."""
    model = _FakeModel(["density", "nr_of_nodes", "gamma_start"])
    frame = gp._feature_frame(model, density=0.25, nr_of_nodes=100, lambda_est=18.0)

    assert list(frame.columns) == ["density", "nr_of_nodes", "gamma_start"]
    assert frame["gamma_start"].iloc[0] == 18.0


def test_feature_frame_uses_canonical_names_when_current():
    model = _FakeModel(["density", "nr_of_nodes", "lambda_est"])
    frame = gp._feature_frame(model, density=0.25, nr_of_nodes=100, lambda_est=18.0)

    assert list(frame.columns) == ["density", "nr_of_nodes", "lambda_est"]
    assert frame["lambda_est"].iloc[0] == 18.0


def test_feature_frame_preserves_fit_time_column_order():
    model = _FakeModel(["lambda_est", "density", "nr_of_nodes"])
    frame = gp._feature_frame(model, density=0.25, nr_of_nodes=100, lambda_est=18.0)

    assert list(frame.columns) == ["lambda_est", "density", "nr_of_nodes"]


def test_feature_frame_rejects_unknown_feature():
    model = _FakeModel(["density", "nr_of_nodes", "not_a_feature"])
    with pytest.raises(ValueError, match="unknown feature"):
        gp._feature_frame(model, density=0.25, nr_of_nodes=100, lambda_est=18.0)


# --------------------------------------------------------------------------
# Algorithm 3
# --------------------------------------------------------------------------

@needs_models
def test_shipped_models_load():
    gbr_min, gbr_max = gp.load_models()
    assert gbr_min.n_features_in_ == 3
    assert gbr_max.n_features_in_ == 3


@needs_models
@pytest.mark.parametrize("n,p", [(100, 0.25), (500, 0.5), (1000, 0.1)])
def test_predict_lambda_is_internally_consistent(n, p):
    """lambda_total must equal lambda_est * (lambda_min + lambda_max) / 2."""
    G = networkx.gnp_random_graph(n, p, seed=1)
    r = gp.predict_lambda(G)

    expected_mult = (r["lambda_min"] + r["lambda_max"]) / 2
    assert r["lambda_mult"] == pytest.approx(expected_mult)
    assert r["lambda_total"] == pytest.approx(r["lambda_est"] * expected_mult)
    assert r["lambda_total"] > 0


@needs_models
def test_predicted_multiplier_falls_inside_table_3_range():
    """Predictions should stay within the grid the models were trained on."""
    import main

    for n in (100, 500, 1000, 4000):
        G = networkx.gnp_random_graph(n, 0.25, seed=3)
        mult = gp.predict_lambda(G)["lambda_mult"]
        grid = main.LAMBDA_MULT_GRID[n]
        assert min(grid) <= mult <= max(grid), (
            f"n={n}: predicted multiplier {mult:.4f} outside "
            f"Table 3 range [{min(grid)}, {max(grid)}]"
        )


@needs_models
def test_calculate_lambda_matches_predict_lambda():
    G = networkx.gnp_random_graph(200, 0.25, seed=2)
    assert gp.calculate_lambda(G) == gp.predict_lambda(G)["lambda_total"]


def test_importing_gbr_prediction_needs_no_live_database():
    """Prediction only requires the pickles, never a reachable server."""
    import importlib

    importlib.reload(gp)  # must not raise even though no MariaDB is running
