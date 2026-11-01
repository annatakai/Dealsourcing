"""Pipe Agent: condition model, condition score and maintenance priority.

Score structure (all on 0-100, see docs/FRAMEWORK.md section 2):

    A  Condition      D = observed G/A/R (decayed toward prediction when stale)
                          or model-expected state 100*(0.5*P(A) + P(R))
    B  Likelihood     L = w_D*D + w_B*B_breaks   (renormalised if break data absent)
                          * environment multipliers for factors NOT in the model
    C  Consequence    C = weighted mean of diameter, customers, critical facility,
                          road above (renormalised over available components)
       Risk           R = sqrt(L * C)            (rank-equivalent to L*C)
    D  Urgency        U = L (+ recent-break override) -> action window
                          (reported, NOT added to priority: it is the same evidence as L)
    E  Economic       E = 100 * BCR/(1+BCR) if break data, else percentile of R per cost/m
       Priority       P = w_R*R + w_E*E

Age and material enter ONLY through the condition model (or the age proxy when
no model/observation exists) - never as separate additive terms.
L is an index, not a probability: no failure labels are used to calibrate it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import cost_basis
from .costs import annuity_factor, emergency_repair_cost, estimate_pipe_replacement_cost
from .utils import log1p_scale, log_scale, weighted_nanmean, window_from_score

CLASSES = ["G", "A", "R"]
NUMERIC_FEATURES = ["age_years", "diameter_mm", "log_length_m"]
CATEGORICAL_FEATURES = ["material", "region"]

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


# --------------------------------------------------------------------- model
def _features(df: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["age_years"] = df["age_years"].astype(float)
    X["diameter_mm"] = df["diameter_mm"].astype(float)
    X["log_length_m"] = np.log1p(df["length_m"].astype(float))
    for c in categorical:
        X[c] = df[c].astype("string").fillna("UNKNOWN").astype(str) if c in df else "UNKNOWN"
    return X


class PipeConditionModel:
    """Gradient-boosted multiclass model of observed G/A/R condition.

    Output is a *predicted condition state distribution*, not a failure
    probability. Missing numeric values are handled natively by XGBoost.
    """

    def __init__(self, categorical: list[str] | None = None, balance: bool = False,
                 seed: int = 42, **xgb_params):
        self.categorical = categorical or CATEGORICAL_FEATURES
        self.balance = balance
        self.seed = seed
        self.xgb_params = {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.08, "subsample": 0.8,
            "colsample_bytree": 0.8, "min_child_weight": 5, "reg_lambda": 1.0,
            "tree_method": "hist", "n_jobs": 0, **xgb_params,
        }
        self.pipeline: Pipeline | None = None

    def _make(self) -> Pipeline:
        if XGBClassifier is None:
            raise ImportError("xgboost is required for PipeConditionModel")
        pre = ColumnTransformer([
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                  sparse_output=False), self.categorical),
        ])
        clf = XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                            random_state=self.seed, **self.xgb_params)
        return Pipeline([("pre", pre), ("clf", clf)])

    @staticmethod
    def _y(labels: pd.Series) -> np.ndarray:
        return labels.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    def _weights(self, y: np.ndarray):
        if not self.balance:
            return None
        counts = np.bincount(y, minlength=3).astype(float)
        w = len(y) / (3 * np.maximum(counts, 1))
        return w[y]

    def fit(self, df: pd.DataFrame) -> "PipeConditionModel":
        train = df[df["condition"].isin(CLASSES)]
        if train["condition"].nunique() < 3:
            raise ValueError("training data must contain all of G, A and R")
        y = self._y(train["condition"])
        self.pipeline = self._make()
        self.pipeline.fit(_features(train, self.categorical), y, clf__sample_weight=self._weights(y))
        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.pipeline.predict_proba(_features(df, self.categorical))
        return pd.DataFrame(p, index=df.index, columns=[f"p_{c}" for c in CLASSES])

    def explain(self, df: pd.DataFrame, target: str = "R") -> pd.DataFrame:
        """Per-row SHAP contributions (log-odds) to the target class, by input feature."""
        pre, clf = self.pipeline.named_steps["pre"], self.pipeline.named_steps["clf"]
        Xt = pre.transform(_features(df, self.categorical))
        import xgboost as xgb
        contrib = clf.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)
        k = CLASSES.index(target)
        c = contrib[:, k, :-1] if contrib.ndim == 3 else contrib[:, :-1]
        names = pre.get_feature_names_out()
        out = pd.DataFrame(index=df.index)
        for feat in NUMERIC_FEATURES + self.categorical:
            idx = [i for i, n in enumerate(names) if n.split("__", 1)[1].startswith(feat)]
            out[feat] = c[:, idx].sum(axis=1) if idx else 0.0
        return out.rename(columns={"log_length_m": "length_m"})

    def cross_validate(self, df: pd.DataFrame, n_splits: int = 5, group_col: str | None = None
                       ) -> tuple[pd.DataFrame, dict]:
        """Out-of-fold probabilities and metrics. group_col -> GroupKFold (e.g. region)."""
        data = df[df["condition"].isin(CLASSES)]
        y = self._y(data["condition"])
        X = _features(data, self.categorical)
        oof = np.zeros((len(data), 3))
        if group_col:
            groups = data[group_col].astype(str).to_numpy()
            n = min(n_splits, len(np.unique(groups)))
            splits = GroupKFold(n_splits=n).split(X, y, groups)
        else:
            splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed).split(X, y)
        for tr, te in splits:
            m = self._make()
            m.fit(X.iloc[tr], y[tr], clf__sample_weight=self._weights(y[tr]))
            oof[te] = m.predict_proba(X.iloc[te])
        proba = pd.DataFrame(oof, index=data.index, columns=[f"p_{c}" for c in CLASSES])
        return proba, classification_metrics(data["condition"], proba)


def classification_metrics(y_true: pd.Series, proba: pd.DataFrame) -> dict:
    y = y_true.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    p = np.clip(proba[[f"p_{c}" for c in CLASSES]].to_numpy(), 1e-9, 1)
    p = p / p.sum(axis=1, keepdims=True)
    pred = p.argmax(axis=1)
    onehot = np.eye(3)[y]
    is_red = (y == 2).astype(int)
    out = {
        "n": int(len(y)),
        "class_prevalence": {c: float((y == i).mean()) for i, c in enumerate(CLASSES)},
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y, pred, weights="quadratic")),
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "brier_multiclass": float(((p - onehot) ** 2).sum(axis=1).mean()),
        "red_recall_argmax": float(recall_score(is_red, pred == 2, zero_division=0)),
        "red_precision_argmax": float(precision_score(is_red, pred == 2, zero_division=0)),
    }
    if 0 < is_red.sum() < len(is_red):
        out["red_roc_auc"] = float(roc_auc_score(is_red, p[:, 2]))
        out["red_average_precision"] = float(average_precision_score(is_red, p[:, 2]))
        # Calibration of P(Red) by decile
        ranks = pd.Series(p[:, 2]).rank(method="first")
        bins = pd.qcut(ranks, 10, labels=False)
        g = pd.DataFrame({"pred": p[:, 2], "obs": is_red}).groupby(bins.to_numpy())
        cal = g.mean()
        out["red_calibration_deciles"] = [
            {"mean_pred": float(r.pred), "obs_rate": float(r.obs)} for r in cal.itertuples()]
        out["red_ece"] = float(np.average(np.abs(cal["pred"] - cal["obs"]), weights=g.size().to_numpy()))
    return out


def baseline_metrics(df: pd.DataFrame, seed: int = 42, n_splits: int = 5) -> dict:
    """Majority-class and age-only multinomial logistic baselines (OOF)."""
    data = df[df["condition"].isin(CLASSES) & df["age_years"].notna()]
    y = data["condition"]
    prev = y.value_counts(normalize=True).reindex(CLASSES).fillna(0)
    maj = pd.DataFrame(np.tile(prev.to_numpy(), (len(y), 1)), index=y.index,
                       columns=[f"p_{c}" for c in CLASSES])
    out = {"majority_prior": classification_metrics(y, maj)}
    Xa = data[["age_years"]].to_numpy()
    yi = y.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    oof = np.zeros((len(y), 3))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(Xa, yi):
        lr = LogisticRegression(max_iter=500).fit(Xa[tr], yi[tr])
        oof[te] = lr.predict_proba(Xa[te])
    out["age_only_logistic"] = classification_metrics(
        y, pd.DataFrame(oof, index=y.index, columns=[f"p_{c}" for c in CLASSES]))
    return out


# --------------------------------------------------------------- A condition
def score_pipe_condition(pipes: pd.DataFrame, cfg: dict, proba: pd.DataFrame | None = None,
                         use_observed: bool = True) -> pd.DataFrame:
    """Condition score D (0-100) with its source and an explanation.

    Rules
    -----
    * observed rating, inspected recently  -> observed state score
    * observed rating, stale               -> w*observed + (1-w)*predicted,
                                              w = 0.5 ** (years_since / half_life)
    * observed rating, no inspection date  -> observed (flagged 'date_unknown')
    * no observation, model available      -> predicted expected state
    * neither                              -> age / expected service life proxy
    ``use_observed=False`` forces predictions (label-blind validation).
    """
    pc = cfg["pipe"]
    states = pc["condition_state_scores"]
    ref = cfg["reference_year"]
    out = pd.DataFrame(index=pipes.index)
    obs = pipes["condition"].map(states).astype(float) if (use_observed and "condition" in pipes) \
        else pd.Series(np.nan, index=pipes.index)
    if proba is not None:
        p = proba.reindex(pipes.index)
        pred = states["A"] * p["p_A"] + states["R"] * p["p_R"]
    else:
        pred = pd.Series(np.nan, index=pipes.index)
    esl = pipes["material"].map(pc["expected_service_life_years"]).fillna(
        pc["expected_service_life_years"]["default"]) if "material" in pipes else pc["expected_service_life_years"]["default"]
    age_proxy = 100 * np.clip((pipes["age_years"] / esl - 0.5) / 1.0, 0, 1)

    insp = pipes["inspection_year"] if "inspection_year" in pipes else pd.Series(np.nan, index=pipes.index)
    years_since = (ref - insp).clip(lower=0)
    w = np.power(0.5, years_since / pc["observation_half_life_years"])
    w = w.where(insp.notna(), 1.0)

    score = pd.Series(np.nan, index=pipes.index)
    source = pd.Series("unknown", index=pipes.index, dtype=object)
    has_obs = obs.notna()
    has_pred = pred.notna()
    blend = has_obs & has_pred & (w < 0.999)
    score[has_obs] = obs[has_obs]
    source[has_obs] = np.where(insp[has_obs].notna(), "observed", "observed_date_unknown")
    score[blend] = w[blend] * obs[blend] + (1 - w[blend]) * pred[blend]
    source[blend] = "observed_aged_toward_prediction"
    m = ~has_obs & has_pred
    score[m], source[m] = pred[m], "predicted"
    m = score.isna() & age_proxy.notna()
    score[m], source[m] = age_proxy[m], "age_proxy"
    out["condition_observed"] = pipes["condition"] if "condition" in pipes else None
    out["condition_obs_score"] = obs
    out["condition_pred_score"] = pred
    if proba is not None:
        out = out.join(proba.reindex(pipes.index))
    out["observation_weight"] = np.where(has_obs, w, 0.0)
    out["condition_score"] = score
    out["condition_source"] = source
    return out


# ------------------------------------------------------------ B likelihood
def _diameter_band(d: pd.Series) -> pd.Series:
    return pd.cut(d, [0, 150, 300, 600, np.inf], labels=["<=150", "150-300", "300-600", ">600"]).astype(str)


def break_evidence(pipes: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Empirical-Bayes (Poisson-gamma credibility) break rate per km-year.

    lambda_hat = (k + beta * m_cohort) / (E + beta)
    k = observed breaks, E = length_km * exposure years, m_cohort = pooled rate of
    the material x diameter-band cohort, beta = prior pseudo-exposure (km*yr).
    """
    out = pd.DataFrame(index=pipes.index)
    if "n_breaks" not in pipes or pipes["n_breaks"].isna().all():
        out["break_rate_hat"] = np.nan
        out["break_score"] = np.nan
        return out
    k = pipes["n_breaks"].astype(float)
    obs = pipes.get("break_obs_years", pd.Series(np.nan, index=pipes.index)).astype(float)
    years = np.fmin(obs, pipes["age_years"].astype(float)).fillna(obs).fillna(pipes["age_years"]).fillna(10.0)
    E = (pipes["length_m"].astype(float) / 1000.0) * years.clip(lower=0.5)
    valid = k.notna() & E.notna() & (E > 0)
    cohort = pipes["material"].astype(str) + "|" + _diameter_band(pipes["diameter_mm"])
    tmp = pd.DataFrame({"k": k.where(valid), "E": E.where(valid), "cohort": cohort})
    pooled = tmp.groupby("cohort")[["k", "E"]].transform("sum")
    global_rate = tmp["k"].sum() / tmp["E"].sum() if tmp["E"].sum() > 0 else 0.0
    m_cohort = (pooled["k"] / pooled["E"]).where(pooled["E"] > 0, global_rate).fillna(global_rate)
    beta = cfg["pipe"]["break_prior_exposure_km_years"]
    lam = (k + beta * m_cohort) / (E + beta)
    lam = lam.where(valid)
    lo, hi = np.nanpercentile(lam, 5), np.nanpercentile(lam, 99)
    out["break_rate_hat"] = lam
    out["break_exposure_km_yr"] = E
    out["break_score"] = pd.Series(log_scale(lam, lo, hi), index=pipes.index).where(valid)
    return out


def _env_multiplier(pipes: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.Series]:
    pc = cfg["pipe"]
    in_model = set(pc["features_in_model"])
    soil = pd.Series(1.0, index=pipes.index)
    pres = pd.Series(1.0, index=pipes.index)
    if "soil_corrosivity" in pipes and "soil_corrosivity" not in in_model:
        soil = 1 + (pc["soil_corrosivity_multiplier_max"] - 1) * pipes["soil_corrosivity"].fillna(0).clip(0, 1)
    if "pressure_m" in pipes and "pressure_m" not in in_model:
        frac = (pipes["pressure_m"] - pc["pressure_low_m"]) / (pc["pressure_high_m"] - pc["pressure_low_m"])
        pres = 1 + (pc["pressure_multiplier_max"] - 1) * frac.fillna(0).clip(0, 1)
    return soil, pres


# --------------------------------------------------------------- C consequence
def consequence_score(pipes: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    pc = cfg["pipe"]
    comp = pd.DataFrame(index=pipes.index)
    comp["diameter"] = log_scale(pipes["diameter_mm"], pc["diameter_min_mm"], pc["diameter_max_mm"])
    comp.loc[pipes["diameter_mm"].isna(), "diameter"] = np.nan
    if "customers_served" in pipes and pipes["customers_served"].notna().any():
        ref = np.nanpercentile(pipes["customers_served"], 99)
        comp["customers"] = pd.Series(log1p_scale(pipes["customers_served"], ref), index=pipes.index
                                      ).where(pipes["customers_served"].notna())
    else:
        comp["customers"] = np.nan
    if "critical_facility" in pipes:
        comp["critical_facility"] = pipes["critical_facility"].map({True: 100.0, False: 0.0, 1.0: 100.0, 0.0: 0.0})
    else:
        comp["critical_facility"] = np.nan
    if "road_class_above" in pipes:
        comp["road_above"] = pipes["road_class_above"].map(pc["road_class_scores"])
    else:
        comp["road_above"] = np.nan
    score, coverage = weighted_nanmean(comp, pc["consequence_weights"])
    comp["consequence_score"] = score.fillna(50.0)
    comp["consequence_coverage"] = coverage
    comp["consequence_flag"] = np.where(score.isna(), "no_consequence_data_neutral_50", "")
    return comp


# ------------------------------------------------------------------ priority
def score_pipe_priority(pipes: pd.DataFrame, cfg: dict, condition: pd.DataFrame | None = None,
                        costs: pd.DataFrame | None = None, proba: pd.DataFrame | None = None,
                        explanations: bool = True) -> pd.DataFrame:
    """Full Pipe Maintenance Priority Score (0-100) with all sub-scores."""
    pc = cfg["pipe"]
    if condition is None:
        condition = score_pipe_condition(pipes, cfg, proba)
    if costs is None:
        costs = estimate_pipe_replacement_cost(pipes, cfg, n_draws=0)
    out = pd.DataFrame(index=pipes.index)
    out["pipe_id"] = pipes["pipe_id"]
    out = out.join(condition)
    D = condition["condition_score"]

    brk = break_evidence(pipes, cfg)
    out = out.join(brk)
    lw = {"condition": pc["likelihood_weights"]["condition"], "breaks": pc["likelihood_weights"]["breaks"]}
    L, _ = weighted_nanmean(pd.DataFrame({"condition": D, "breaks": brk["break_score"]}), lw)
    soil, pres = _env_multiplier(pipes, cfg)
    out["env_multiplier"] = soil * pres
    out["likelihood_score"] = (L * out["env_multiplier"]).clip(0, 100)
    out["likelihood_basis"] = np.where(brk["break_score"].notna(), "condition+breaks", "condition_only")

    cons = consequence_score(pipes, cfg)
    out = out.join(cons.add_prefix("cons_").rename(columns={
        "cons_consequence_score": "consequence_score", "cons_consequence_coverage": "consequence_coverage",
        "cons_consequence_flag": "consequence_flag"}))
    out["risk_score"] = np.sqrt(out["likelihood_score"] * out["consequence_score"])

    # D. urgency
    out["urgency_score"] = out["likelihood_score"]
    labels, years = window_from_score(out["urgency_score"], pc["action_windows"])
    recent = pd.Series(False, index=pipes.index)
    if "last_break_year" in pipes:
        recent = (cfg["reference_year"] - pipes["last_break_year"]) <= pc["recent_break_years"]
    immediate = (pipes.get("condition") == "R") & recent.fillna(False)
    labels = np.where(immediate, "immediate", labels)
    years = np.where(immediate, 0.0, years)
    out["action_window"] = labels
    out["target_year_offset"] = years

    # E. economic
    out["replacement_cost"] = costs["cost_point"].values
    out["cost_per_m"] = costs["cost_per_m"].values
    out["cost_basis"] = cost_basis(cfg)
    c = cfg["costs"]
    if brk["break_rate_hat"].notna().any():
        reduction = pc.get("replacement_break_reduction", 0.9)
        annual = brk["break_rate_hat"] * pipes["length_m"] / 1000.0 * emergency_repair_cost(pipes["diameter_mm"], cfg)
        pv = annual * reduction * annuity_factor(c["discount_rate"], c["analysis_horizon_years"])
        bcr = pv / out["replacement_cost"]
        out["expected_breaks_per_year"] = brk["break_rate_hat"] * pipes["length_m"] / 1000.0
        out["pv_avoided_break_cost"] = pv
        out["benefit_cost_ratio"] = bcr
        out["economic_score"] = (100 * bcr / (1 + bcr)).fillna(0)
        out["economic_basis"] = "monetised_BCR"
    else:
        ce = out["risk_score"] / out["cost_per_m"]
        out["benefit_cost_ratio"] = np.nan
        out["economic_score"] = (100 * ce.rank(pct=True)).fillna(0)
        out["economic_basis"] = "relative_risk_per_cost_rank"

    w = pc["priority_weights"]
    out["priority_risk_part"] = w["risk"] * out["risk_score"]
    out["priority_econ_part"] = w["economic"] * out["economic_score"]
    out["priority_score"] = (out["priority_risk_part"] + out["priority_econ_part"]).round(2)
    out["priority_rank"] = out["priority_score"].rank(ascending=False, method="first").astype(int)
    out["dq_flags"] = pipes.get("dq_flags", "")
    if explanations:
        out["explanation"] = explain_pipe_rows(out, pipes)
    return out


def explain_pipe_rows(s: pd.DataFrame, pipes: pd.DataFrame) -> list[str]:
    rows = []
    cons_cols = [c for c in s.columns if c.startswith("cons_") and c not in ("cons_consequence_score",)]
    for idx in s.index:
        r = s.loc[idx]
        p = pipes.loc[idx]
        src = r["condition_source"]
        if src.startswith("observed"):
            cond = f"observed {r['condition_observed']} -> D={r['condition_score']:.0f}"
            if src == "observed_aged_toward_prediction":
                cond += f" (obs weight {r['observation_weight']:.2f}, pred {r['condition_pred_score']:.0f})"
        elif src == "predicted":
            cond = f"predicted D={r['condition_score']:.0f} (P(R)={r.get('p_R', np.nan):.2f})"
        else:
            cond = f"{src} D={r['condition_score']:.0f}" if pd.notna(r["condition_score"]) else "condition unknown"
        lik = f"L={r['likelihood_score']:.0f} [{r['likelihood_basis']}"
        if pd.notna(r.get("break_score")):
            lik += f", break score {r['break_score']:.0f}"
        lik += "]"
        cparts = [f"{c[5:]} {r[c]:.0f}" for c in cons_cols
                  if isinstance(r[c], (int, float, np.floating)) and pd.notna(r[c])]
        con = f"C={r['consequence_score']:.0f} ({', '.join(cparts) or 'no data'})"
        attrs = (f"{p.get('material', '?')} {p.get('diameter_mm', np.nan):.0f}mm, "
                 f"age {p.get('age_years', np.nan):.0f}y, {p.get('length_m', np.nan):.0f}m")
        rows.append(
            f"{attrs} | {cond} | {lik} | {con} | Risk={r['risk_score']:.0f} | "
            f"Econ={r['economic_score']:.0f} ({r['economic_basis']}) | "
            f"Priority={r['priority_score']:.1f} = {r['priority_risk_part']:.1f} risk + "
            f"{r['priority_econ_part']:.1f} econ | window {r['action_window']}")
    return rows
