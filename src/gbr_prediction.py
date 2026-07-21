import argparse
import pickle
from pathlib import Path

import pandas as pd
from database_tables import engine,Base,sessionmaker
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from graph_generation import max_degree
import networkx as nx
import numpy as np


# Model paths are resolved relative to this file so the module works
# regardless of the current working directory.
MODEL_DIR = Path(__file__).resolve().parent / "gbr"
GBR_MIN_PATH = MODEL_DIR / "gbr_min.pkl"
GBR_MAX_PATH = MODEL_DIR / "gbr_max.pkl"

# Lazily populated by load_models()
_gbr_min = None
_gbr_max = None

# Feature vector f = (n, rho, lambda_est), in the order used at fit time.
CANONICAL_FEATURES = ["density", "nr_of_nodes", "lambda_est"]

# Models pickled before the gamma -> lambda rename expect the old column
# names. Map any legacy name onto the canonical one so that both the shipped
# models and freshly trained ones can be used without re-exporting.
LEGACY_FEATURE_ALIASES = {
    "gamma_start": "lambda_est",
    "gamma_coef_start": "lambda_est",
}


def _feature_frame(model, density, nr_of_nodes, lambda_est):
    """Build the input frame using the column names *this* model expects."""
    values = {
        "density": density,
        "nr_of_nodes": nr_of_nodes,
        "lambda_est": lambda_est,
    }

    expected = list(getattr(model, "feature_names_in_", CANONICAL_FEATURES))

    columns = {}
    for name in expected:
        canonical = LEGACY_FEATURE_ALIASES.get(name, name)
        if canonical not in values:
            raise ValueError(
                f"Model expects unknown feature {name!r}. "
                f"Known features: {sorted(values)}."
            )
        columns[name] = [values[canonical]]

    # Preserve fit-time column order as well as naming.
    return pd.DataFrame(columns, columns=expected)


def train_gbr():
    # Table creation is deferred to here: prediction only needs the pickled
    # models, so importing this module must not require a live database.
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)  # noqa: F841 - kept for interactive use

    query_for_regression = """
    ## graphs with best partitions and min/max gamma
    with p0 as (
    select g.id as graph_id
    from partitions p
    inner join graphs g on p.graph_id  = g.id
    where nr_of_nodes in (100,200)
    #and edge_probability in (0.1, 0.25, 0.5, 0.75)
    and g.id > 1045
    group by g.id
    having sum(round(lambda_total/lambda_est,4)) >= 0.75
    union all
    select g.id
    from partitions p
    inner join graphs g on p.graph_id  = g.id
    where nr_of_nodes in  (300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000)
    #and edge_probability in (0.1, 0.25, 0.5, 0.75)
    and g.id > 1045
    group by g.id
    having sum(round(lambda_total/lambda_est,4)) >= 0.195 
    ),
    p1 as (
    select min(p.qa_inter_edges) as min_edges,
        p.graph_id
    from p0 
    inner join partitions p on p.graph_id = p0.graph_id
    where 1=1 
    and p.comp_type = "hybrid"
    and p.qa_inter_edges < 1000000
    group by p.graph_id),
    p as (
    select max(p2.lambda_total) as max_lambda_total,
        min(p2.lambda_total) as min_lambda_total,
        p1.graph_id,
        p1.min_edges
    from p1 
    inner join partitions p2 on p1.graph_id = p2.graph_id
                        and p1.min_edges = p2.qa_inter_edges
    where  1=1 
    and p2.qa_inter_edges < 1000000
    group by p1.graph_id, p1.min_edges
    )
    select g.id, g.nr_of_nodes, g.nr_of_edges, 
    g.edge_probability, g.density,
    g.pymetis_inter_edges, g.kernighan_lin_inter_edges,
    p.min_edges, 
    g.lambda_est,p.min_lambda_total, p.max_lambda_total,
    round(p.min_lambda_total / g.lambda_est,3) as min_lambda_mult,
    round(p.max_lambda_total / g.lambda_est,3) as max_lambda_mult,
    case when p.min_edges <= g.pymetis_inter_edges #and p.min_edges <= g.kernighan_lin_inter_edges
    then 1 else 0 end as hybrid_best
    from graphs g
    inner join p on p.graph_id = g.id
    where g.id > 1045
    and g.nr_of_edges is not null 
    and round(p.min_lambda_total / g.lambda_est,3)  is not null
    order by id desc """

    df_for_regression = pd.read_sql(query_for_regression, con=engine)


    # Separate predictors (X) and targets (y)
    X = df_for_regression[['density', 'nr_of_nodes','lambda_est']]
    y_min = df_for_regression['min_lambda_mult']
    y_max = df_for_regression['max_lambda_mult']

    # Split data into training and testing sets
    X_train, X_test, y_min_train, y_min_test, y_max_train, y_max_test = train_test_split(
        X, y_min, y_max, test_size=0.2, random_state=42
    )

    # Initialize Gradient Boosting Regressors
    gbr_min = GradientBoostingRegressor(random_state=42)
    gbr_max = GradientBoostingRegressor(random_state=42)

    # Fit the models
    gbr_min.fit(X_train, y_min_train)
    gbr_max.fit(X_train, y_max_train)

    # Predict intervals
    y_min_pred = gbr_min.predict(X_test)
    y_max_pred = gbr_max.predict(X_test)

    # Combine predictions into an interval
    predicted_intervals = list(zip(y_min_pred, y_max_pred))

    # Display results
    for i, interval in enumerate(predicted_intervals):
        print(f"Sample {i + 1}: Predicted Interval = {interval}")

    # Evaluate for min_value
    mse_min = mean_squared_error(y_min_test, y_min_pred)
    mae_min = mean_absolute_error(y_min_test, y_min_pred)
    rmse_min = np.sqrt(mse_min)
    r2_min = r2_score(y_min_test, y_min_pred)

    print(f"--- min_lambda_mult ---")
    print(f"MAE:  {mae_min:.4f}")
    print(f"MSE:  {mse_min:.4f}")
    print(f"RMSE: {rmse_min:.4f}")
    print(f"R²:   {r2_min:.4f}\n")

    # Evaluate for max_value
    mse_max = mean_squared_error(y_max_test, y_max_pred)
    mae_max = mean_absolute_error(y_max_test, y_max_pred)
    rmse_max = np.sqrt(mse_max)
    r2_max = r2_score(y_max_test, y_max_pred)

    print(f"--- max_lambda_mult ---")
    print(f"MAE:  {mae_max:.4f}")
    print(f"MSE:  {mse_max:.4f}")
    print(f"RMSE: {rmse_max:.4f}")
    print(f"R²:   {r2_max:.4f}")

    # Evaluate for min_value
    mse_min = mean_squared_error(y_min_test, y_min_pred)
    print(f"Mean Squared Error for min_value: {mse_min}")

    # Evaluate for max_value
    mse_max = mean_squared_error(y_max_test, y_max_pred)
    print(f"Mean Squared Error for max_value: {mse_max}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(GBR_MIN_PATH, 'wb') as file:
        pickle.dump(gbr_min, file)

    with open(GBR_MAX_PATH, 'wb') as file:
        pickle.dump(gbr_max, file)

    print(f"Models saved to {MODEL_DIR}")

    return gbr_min, gbr_max


def load_models():
    """Load the trained regressors, caching them for subsequent calls."""
    global _gbr_min, _gbr_max

    if _gbr_min is None or _gbr_max is None:
        for path in (GBR_MIN_PATH, GBR_MAX_PATH):
            if not path.exists():
                raise FileNotFoundError(
                    f"Trained model not found at {path}. "
                    "Run `python gbr_prediction.py --train` first."
                )

        with open(GBR_MIN_PATH, 'rb') as file:
            _gbr_min = pickle.load(file)

        with open(GBR_MAX_PATH, 'rb') as file:
            _gbr_max = pickle.load(file)

    return _gbr_min, _gbr_max


def predict_lambda(G):
    """Predict the penalty parameter for graph G (Algorithm 3).

    Returns a dict with every intermediate quantity, so callers can record
    the predicted bounds alongside the final penalty value:

        lambda_est   initial graph-dependent estimate, Eq. (26)
        lambda_min   predicted lower bound of the multiplier range
        lambda_max   predicted upper bound of the multiplier range
        lambda_mult  (lambda_min + lambda_max) / 2
        lambda_total lambda_est * lambda_mult, i.e. Eq. (28)
    """
    gbr_min, gbr_max = load_models()

    lambda_est = (1 + min(G.number_of_nodes() / 2 - 1, max_degree(G))) / 2
    density = nx.density(G)
    nr_of_nodes = G.number_of_nodes()

    # Predict lower and upper bounds of the effective multiplier range
    min_pred = float(
        gbr_min.predict(_feature_frame(gbr_min, density, nr_of_nodes, lambda_est))[0]
    )
    max_pred = float(
        gbr_max.predict(_feature_frame(gbr_max, density, nr_of_nodes, lambda_est))[0]
    )
    lambda_mult = (min_pred + max_pred) / 2

    return {
        "lambda_est": lambda_est,
        "lambda_min": min_pred,
        "lambda_max": max_pred,
        "lambda_mult": lambda_mult,
        "lambda_total": lambda_est * lambda_mult,
    }


def calculate_lambda(G):
    """Return only the final penalty parameter for graph G (Eq. 28)."""
    return predict_lambda(G)["lambda_total"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or inspect the GBR penalty-multiplier models."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Retrain both regressors from the database and overwrite the pickles.",
    )
    args = parser.parse_args()

    if args.train:
        train_gbr()
    else:
        load_models()
        print(f"Loaded trained models from {MODEL_DIR}")


