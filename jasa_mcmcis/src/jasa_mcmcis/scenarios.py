from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from jasa_mcmcis.problem import PermutationTestProblem
from jasa_mcmcis.statistics import STATISTIC_REGISTRY

ARTICLE_SCENARIO_KEYS: tuple[str, ...] = (
    "gwas_additive_score_ultra_n100",
    "poisson_diffmeans_hep_ultra_n200",
    "gwas_additive_score_sig_n100",
    "poisson_diffmeans_hep_sig_n200",
    "gwas_additive_score_above_n100",
    "poisson_diffmeans_hep_above_n200",
)

GWAS_NEAR_THRESHOLD_SCENARIO_KEYS: tuple[str, ...] = tuple(
    f"gwas_additive_score_near_v{i:02d}_n120" for i in range(1, 51)
)
HEP_NEAR_THRESHOLD_SCENARIO_KEYS: tuple[str, ...] = tuple(
    f"poisson_diffmeans_hep_near_v{i:02d}_n200" for i in range(1, 51)
)
GWAS_ABOVE_THRESHOLD_SCENARIO_KEYS: tuple[str, ...] = tuple(
    f"gwas_additive_score_above_v{i:02d}_n120" for i in range(1, 51)
)
HEP_ABOVE_THRESHOLD_SCENARIO_KEYS: tuple[str, ...] = tuple(
    f"poisson_diffmeans_hep_above_v{i:02d}_n200" for i in range(1, 51)
)
NEAR_THRESHOLD_SCENARIO_KEYS: tuple[str, ...] = (
    GWAS_NEAR_THRESHOLD_SCENARIO_KEYS + HEP_NEAR_THRESHOLD_SCENARIO_KEYS
)
ABOVE_THRESHOLD_SCENARIO_KEYS: tuple[str, ...] = (
    GWAS_ABOVE_THRESHOLD_SCENARIO_KEYS + HEP_ABOVE_THRESHOLD_SCENARIO_KEYS
)
CROSS_METHOD_SCENARIO_KEYS: tuple[str, ...] = (
    ARTICLE_SCENARIO_KEYS + NEAR_THRESHOLD_SCENARIO_KEYS + ABOVE_THRESHOLD_SCENARIO_KEYS
)


@dataclass(frozen=True)
class Scenario:
    key: str
    description: str
    problem: PermutationTestProblem
    statistic_name: str
    exact_method: str
    exact_p_value: float
    tail_hits: int
    n_permutations: int
    notes: str
    extra: dict[str, Any]
    portfolio: dict[str, Any]


def _default_data_root():
    return files("jasa_mcmcis").joinpath("data", "scenarios")


def _read_json_from_root(data_root, relative: str) -> dict[str, Any]:
    resource = data_root.joinpath(relative)
    return json.loads(resource.read_text(encoding="utf-8"))


def _load_array_from_root(data_root, relative: str) -> np.ndarray:
    resource = data_root.joinpath(relative)
    with resource.open("rb") as handle:
        return np.load(handle, allow_pickle=False)


def _resolve_data_root(data_root=None):
    if data_root is None:
        return _default_data_root()
    if isinstance(data_root, (str, Path)):
        return Path(data_root)
    return data_root


def _catalog_records(data_root=None) -> list[dict[str, Any]]:
    root = _resolve_data_root(data_root)
    catalog = _read_json_from_root(root, "catalog.json")
    records = list(catalog.get("scenarios", []))
    return records


def available_scenarios(data_root: str | Path | None = None) -> tuple[str, ...]:
    """Return the scenario keys bundled with the package."""
    return tuple(str(record["key"]) for record in _catalog_records(data_root))


def _record_by_key(data_root=None) -> dict[str, dict[str, Any]]:
    return {str(record["key"]): record for record in _catalog_records(data_root)}


def load_scenario(key: str, data_root: str | Path | None = None) -> Scenario:
    """Load one frozen cross-method simulation scenario."""
    root = _resolve_data_root(data_root)
    records = _record_by_key(root)
    key = str(key)
    if key not in records:
        raise KeyError(f"Unknown scenario '{key}'. Available: {sorted(records)}")
    record = records[key]

    stat_name = str(record["statistic_name"])
    if stat_name not in STATISTIC_REGISTRY:
        raise KeyError(f"Unknown statistic_name '{stat_name}' in scenario metadata.")

    x = _load_array_from_root(root, f"{key}/X.npy")
    y_obs = _load_array_from_root(root, f"{key}/y_obs.npy").astype(np.int8)
    problem = PermutationTestProblem(
        x=x,
        y_obs=y_obs,
        statistic=STATISTIC_REGISTRY[stat_name],
        tail=str(record.get("tail", "right")),
    )
    if not np.isclose(problem.t_obs, float(record["t_obs"]), atol=1e-12, rtol=0.0):
        raise ValueError(f"Loaded data for '{key}' do not match metadata t_obs.")

    return Scenario(
        key=key,
        description=str(record["description"]),
        problem=problem,
        statistic_name=stat_name,
        exact_method=str(record["exact_method"]),
        exact_p_value=float(record["exact_p_value"]),
        tail_hits=int(record["tail_hits"]),
        n_permutations=int(record["n_permutations"]),
        notes=str(record.get("notes", "")),
        extra=dict(record.get("extra", {})),
        portfolio=dict(record.get("portfolio", {})),
    )


def load_scenarios(
    keys: Iterable[str] | None = None,
    data_root: str | Path | None = None,
) -> list[Scenario]:
    """Load several scenarios. By default loads the cross-method inventory."""
    selected = CROSS_METHOD_SCENARIO_KEYS if keys is None else tuple(str(k) for k in keys)
    return [load_scenario(key, data_root=data_root) for key in selected]
