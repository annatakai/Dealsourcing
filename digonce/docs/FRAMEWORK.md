# Dig Once — Evidence-Based Infrastructure Maintenance Scoring Framework

Version 0.1 (hackathon MVP). Everything below is implemented in `dig_once/` and
reproducible with `python scripts/run_pipeline.py`. All parameters live in
`dig_once/config.py` (overridable with YAML).

> **Read this first: what is and isn't real**
> * The **formulas, code and tests** are complete and working.
> * The **unit costs** shipped in the config are **demo placeholders** (round numbers),
>   **not** real-world prices. Every cost output carries `cost_basis = "PLACEHOLDER…"`
>   until you load local costs (Section 5).
> * The **example outputs and validation numbers** in this document were computed on
>   **synthetic data** (`dig_once/synthetic.py`). Your Desktop CSVs were not available
>   in the build environment. They show that the pipeline works and how to read the
>   results. They are **not** findings about any real network.
> * No score in this system is a validated **probability of failure**. No failure labels
>   were used to calibrate anything.

---

## 0. Architecture

```
            ┌─────────────── Pipe Agent ───────────────┐
 pipes ───► │ clean → condition model (XGBoost, G/A/R) │──► pipe scores (A–E, priority)
 breaks ──► │ → condition D → likelihood L → C → risk   │
            └──────────────────────────────────────────┘
            ┌─────────────── Road Agent ───────────────┐
 roads ───► │ clean → PCI_eff → deficiency Dr          │──► road scores (priority, treatment)
            │ → criticality Cr → risk → treatment      │
            └──────────────────────────────────────────┘
            ┌────────────── Budget Agent ──────────────┐
            │ pair matching (geometry / street name)   │──► coordination pairs (COS + ¥ savings)
            │ → coordination score + net savings       │──► budget selection (MIP, OR-Tools SCIP)
            │ → optimise pipe / road / joint options   │
            └──────────────────────────────────────────┘
```

Module map: `schema.py` (cleaning/QA), `pipe.py`, `road.py`, `costs.py`,
`coordination.py`, `optimize.py`, `sampling.py`, `validation.py`, `export.py`.

---

## 1. Literature and standards review (Task 1)

The sources below are real and verifiable. **V** means I checked bibliographic details
against an online record while building this. **K** means a well-established reference
cited from domain knowledge; please check the edition and page numbers before formal use.
I only read abstracts and records, not full texts. Where I describe a method beyond its
abstract, I have marked that description as general knowledge.

| # | Source | Domain | Status |
|---|--------|--------|--------|
| 1 | Shamir, U. & Howard, C.D.D. (1979). An analytic approach to scheduling pipe replacement. *Journal AWWA*, 71(5), 248–258. | Pipe break forecasting, replacement timing | K |
| 2 | Kleiner, Y. & Rajani, B. (2001). Comprehensive review of structural deterioration of water mains: statistical models. *Urban Water*, 3(3), 131–150. | Pipe deterioration models | K |
| 3 | Winkler, D., Haltmeier, M., Kleidorfer, M., Rauch, W. & Tscheikner-Gratl, F. (2018). Pipe failure modelling for water distribution networks using boosted decision trees. *Structure and Infrastructure Engineering*, 14(10), 1402–1411. doi:10.1080/15732479.2018.1443145 | ML failure prediction | V |
| 4 | Snider, B. & McBean, E.A. (2020). Improving urban water security through pipe-break prediction models: machine learning or survival analysis. *J. Environmental Engineering* (ASCE). doi:10.1061/(ASCE)EE.1943-7870.0001657 | ML vs. survival analysis | V |
| 5 | AWWA (2019). *Manual M77: Condition Assessment of Water Mains*. American Water Works Association. | Condition assessment practice | K |
| 6 | US EPA (2008). *Asset Management: A Best Practices Guide*. EPA 816-F-08-014. | Risk = likelihood × consequence; criticality | K |
| 7 | ISO 55000 / 55001 / 55002 (Asset management family; 2014 edition, revised 2024). | Asset-management system requirements | K |
| 8 | ASTM D6433 (current edition). *Standard Practice for Roads and Parking Lots Pavement Condition Index Surveys*. | PCI 0–100 | K |
| 9 | Shahin, M.Y. (2005). *Pavement Management for Airports, Roads, and Parking Lots*, 2nd ed. Springer. | PCI, PMS, deterioration curves | K |
| 10 | Saaty, T.L. (1980). *The Analytic Hierarchy Process*. McGraw-Hill (and Saaty 1977, *J. Math. Psychology* 15(3), 234–281). | AHP weighting | K |
| 11 | Hwang, C.L. & Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications*. Springer (Lecture Notes in Econ. & Math. Systems 186). | TOPSIS | K |
| 12 | Kabir, G., Sadiq, R. & Tesfamariam, S. (2014). A review of multi-criteria decision-making methods for infrastructure management. *Structure and Infrastructure Engineering*, 10(9), 1176–1210. doi:10.1080/15732479.2013.795978 | MCDA review (~300 papers) | V |
| 13 | Walls, J. & Smith, M.R. (1998). *Life-Cycle Cost Analysis in Pavement Design*. FHWA-SA-98-079. | LCCA, discounting, user costs | K |
| 14 | Tscheikner-Gratl, F., Sitzenfrei, R., Rauch, W. & Kleidorfer, M. (2016). Integrated rehabilitation planning of urban infrastructure systems using a street section priority model. *Urban Water Journal*, 13(1), 28–40. doi:10.1080/1573062X.2015.1057174 | Joint water/sewer/road ranking | V |
| 15 | Tscheikner-Gratl, F., Egger, P., Rauch, W. & Kleidorfer, M. (2017). Comparison of multi-criteria decision support methods for integrated rehabilitation prioritization. *Water*, 9(2), 68. doi:10.3390/w9020068 | MCDA comparison (AHP, TOPSIS, …) | V |
| 16 | Carey, B. & Lueke, J. (2013). Optimized holistic municipal right-of-way capital improvement planning. *Canadian Journal of Civil Engineering*, 40, 1244–1251. | Joint ROW optimisation (GA) | V |
| 17 | Shahata, K. & Zayed, T. (2016). Integrated risk-assessment framework for municipal infrastructure. *J. Construction Engineering and Management*, 142(1). doi:10.1061/(ASCE)CO.1943-7862.0001028 | Road–water–sewer integrated risk | V |
| 18 | 厚生労働省 (2009). 水道事業におけるアセットマネジメント（資産管理）に関する手引き; 厚生労働省 (2011). 水道事業の再構築に関する施設更新費用算定の手引き, with the 2024 notice on using it (「活用にあたっての留意点」). | Japanese water AM and renewal cost functions | V (existence/2024 notice); K (content) |
| 19 | 国土交通省. 公共事業評価の費用便益分析に関する技術指針（共通編）: social discount rate 4%. | CBA discounting | K |
| 20 | Pavement-cut damage studies: City of Calgary forensic study (TAC conference paper, 2014), which also reports a 1992 San Francisco study (up to ~50% service-life loss on cut streets); City of Toronto (TAC 2013) and Mississauga (TAC 2025) degradation-fee papers; City of Davis, CA (2022) fee study. | Cost of cutting resurfaced roads | V (via TAC/city records) |

### What each source contributes

**[1] Shamir & Howard 1979.** *Factors:* break history per pipe or pipe group.
*Scoring:* exponential growth of breaks, N(t) = N(t₀)·e^{A(t−t₀)}. The optimal replacement
year is when the present value of future repair costs exceeds the present value of
replacement cost. *Weights:* none, because the method is economic and single-criterion.
*Validation:* fitted to utility break records. It is still a standard reference model.
**Used in Dig Once:** the economic term (E) compares the PV of expected break costs with
replacement cost.

**[2] Kleiner & Rajani 2001.** Reviews deterministic models (time-exponential, linear),
probabilistic multivariate models (proportional hazards, accelerated lifetime) and
cohort-survival approaches. *Factors:* age, material, diameter, length, soil, climate and
previous breaks. *Weights:* these are fitted regression coefficients, not expert weights.
*Validation:* the models it reviews are validated on historical break data, and it shows
that previous breaks are among the strongest predictors. **Used:** the empirical-Bayes
break-rate term B, and the rule that age and material enter only through a fitted model.

**[3] Winkler et al. 2018.** *Factors:* pipe attributes and failure history. *Scoring:*
boosted decision trees with random undersampling, which performed best in their benchmark.
*Validation:* a real network case study. The authors note survival bias, because removed
pipes drop out of the data. **Used:** justifies gradient boosting (XGBoost) for the
condition model and the survival-bias caveat in Section 9.

**[4] Snider & McBean 2020.** Compares a gradient-boosting model with a Weibull
proportional-hazards model for time to next break in cast-iron mains. *Finding (abstract):*
dropping censored observations from the ML model biases it toward predicting breaks too
early, and survival models handle right-censoring better. **Used:** the condition model is a
*state classifier* (G/A/R), not a time-to-failure model. Recommendation R3 is to add a
survival model when break data exist.

**[5] AWWA M77.** Practice manual for direct and indirect condition assessment of mains
(desktop, inspection and testing). **Used:** the rule that observed condition beats
predicted condition, and inspection staleness weighting.

**[6] US EPA 2008.** Five core asset-management questions (current state, level of
service, criticality, lifecycle cost, long-term funding). Criticality is defined as
likelihood of failure × consequence of failure. *Weights:* left to the utility.
**Used:** the L × C risk structure (R = √(L·C) is rank-equivalent).

**[7] ISO 55000 family.** Management-system standard. It requires risk-based,
lifecycle-value decisions and documented decision criteria, but prescribes no scoring
formula. **Used:** auditability (config-driven weights, explanation strings, data-quality
flags).

**[8] ASTM D6433 / [9] Shahin 2005.** *Factors:* distress type, severity and density from
visual surveys. *Scoring:* PCI = 100 − Σ corrected deduct values, from deduct curves.
*Weights:* the deduct curves are empirically derived from expert panel ratings. *Validation:*
the deduct curves were developed against panel ratings, and PCI is an industry standard.
Shahin covers deterioration curves and treatment selection by PCI band. **Used:** PCI is
the road condition input, the treatment decision tree is banded by PCI, and missing or
stale PCI is extrapolated with a deterioration rate.

**[10] Saaty (AHP).** Pairwise comparisons on a 1–9 scale. Weights come from the
principal eigenvector, and a consistency ratio below 0.1 is checked. *Validation:* AHP
weights are elicited preferences, not empirically validated. **Used:** recommended for
the municipal workshop that should replace the default weights (R6). Dig Once does not
pretend its default weights came from an AHP session.

**[11] TOPSIS.** Ranks alternatives by closeness to an ideal point after vector
normalisation and weighting. **Considered and not chosen:** TOPSIS rankings depend on the
set of alternatives, so adding or removing assets changes other assets' scores. That is
undesirable for a network-wide score that must stay stable as data are added.

**[12] Kabir et al. 2014.** Reviews ~300 MCDM applications in infrastructure management
from 1980–2012 and shows how widespread AHP, weighted sum and outranking methods are.
**Used:** justifies the transparent weighted sum with renormalisation as the MVP choice.

**[13] FHWA LCCA (Walls & Smith 1998).** Lifecycle cost = agency costs + user costs
(work-zone delay), discounted to present value, with probabilistic (Monte Carlo) treatment
of uncertain inputs. **Used:** the user-delay term in coordination savings, discounting,
and the Monte Carlo P10/P50/P90 cost ranges.

**[14] Tscheikner-Gratl et al. 2016.** Ranks water and sewer networks *together* at the
street-section level, including surrounding factors (street network, buildings). *Validation
(abstract):* areas flagged by the model were compared with the rehabilitation plans the
utilities actually adopted, and the ranking was used in fixed-budget planning. **Used:**
the coordination concept, i.e. pair-level scoring on street sections.

**[15] Tscheikner-Gratl et al. 2017.** Compares MCDA methods for integrated rehabilitation
prioritisation and finds that different methods yield different rankings. **Used:** the
reason Dig Once ships weight and method sensitivity analysis (Section 8).

**[16] Carey & Lueke 2013.** Holistic right-of-way capital planning across co-located
assets, optimised with genetic algorithms. It is designed to show savings over
single-asset planning. **Used:** joint bundles as optimisation options. Dig Once uses an
exact MIP instead of a GA because the MVP problem is small enough.

**[17] Shahata & Zayed 2016.** Integrated road–water–sewer risk. *Factors:* four
consequence categories with 18 sub-factors. *Scoring:* network indices are combined into
an integrated risk index with K-means clustering. *Validation:* a City of Guelph case
study, where pipe/road size and accessibility had the highest impact. **Used:** diameter
as the dominant consequence proxy, and road class above the pipe as a consequence
component.

**[18] MHLW guidance (Japan).** The AM guidance sets out renewal-demand estimation. The
2011 renewal-cost guide provides cost functions built from national construction records
(FY2011 prices). MHLW's 2024 notice warns that material and labour costs have changed and
must be adjusted, e.g. with an MLIT construction deflator. **Used:** the *structure* of
the pipe cost function (unit cost by diameter × length), plus the instruction to update
the price year. **The actual coefficients were not retrieved** (mhlw.go.jp was not
reachable from the build environment). See Section 5.

**[19] MLIT CBA guideline.** 4% social discount rate. Used as `discount_rate`.

**[20] Pavement-cut studies.** Cut streets lose service life. Calgary reports a cited
reduction of ~8 years in functional life (from a 1987 study) and San Francisco's 1992
study reported up to ~50%. Toronto, Mississauga and Davis turned this into degradation
fees by road class and pavement age. **Used:** the reopen-penalty equation and the
`cut_life_loss_fraction` uncertainty range (default 0.30, documented range 0.1–0.5).

---

## 2. Pipe Maintenance Priority Score (Task 2)

All sub-scores are on 0–100. Notation: `clip01(x)` = min(max(x,0),1).

### A. Physical condition D

Observed ordinal rating → state score: **G = 0, A = 50, R = 100** (configurable; Section 8
shows the effect of moving A to 30 or 70).

| Situation | D |
|-----------|---|
| Observed, inspected this year | `s(obs)` |
| Observed, inspected Δ years ago | `w·s(obs) + (1−w)·D̂`, with `w = 0.5^(Δ/h)` and `h = 10 y` |
| Observed, no inspection date | `s(obs)`, flagged `observed_date_unknown` |
| Not observed, model available | `D̂ = 50·P(A) + 100·P(R)` (expected state) |
| Neither | age proxy `100·clip01(age/ESL − 0.5)` (last resort, flagged) |

`P(·)` comes from the XGBoost condition model (features: age, material, diameter,
log length, region). For rated pipes the pipeline uses **out-of-fold** probabilities, so
no pipe is scored by a model that saw its own label. D̂ is labelled *predicted
condition*, not failure probability.

### B. Likelihood-of-failure index L

Break evidence (empirical-Bayes Poisson–gamma credibility, after [1][2]):

```
E_i       = length_km_i × min(observation_years, age_i)      exposure, km·yr
m_c       = Σk / ΣE in cohort c = material × diameter band   pooled cohort rate
λ̂_i       = (k_i + β·m_c) / (E_i + β)                       β = 2 km·yr prior exposure
B_i       = 100 · clip01( ln(λ̂_i/λ_p5) / ln(λ_p99/λ_p5) )    log scale between network P5–P99
```

Shrinkage stops a 10 m pipe with one break from outranking a 1 km pipe with ten breaks.

```
L = (w_D·D + w_B·B) / (w_D + w_B over available terms)   w_D = w_B = 0.5
L ← min(100, L × m_soil × m_pressure)
```

The environmental multipliers (soil corrosivity up to ×1.20, pressure 30→80 m up to
×1.15) are expert-judgment assumptions. They are applied **only if that variable is not a
model feature** (`features_in_model`). If break data are absent, L = D and the output says
`likelihood_basis = condition_only`.

### C. Consequence of failure C

| Component | Normalisation | Weight |
|-----------|---------------|--------|
| Diameter (flow / service-area proxy) | `100·clip01(ln(D/50)/ln(1000/50))` | 0.35 |
| Customers served | `100·ln(1+n)/ln(1+n_p99)` | 0.25 |
| Critical facility (hospital, etc.) | 100 if yes, else 0 | 0.25 |
| Road above (traffic disruption) | arterial 100 / collector 60 / local 30 / none 10 | 0.15 |

Weights are renormalised per row over the components present. With no data at all,
C = 50 (neutral, flagged).

### Risk, urgency, economics, priority

```
Risk         R = √(L · C)                       same ranking as L×C, stays on 0–100
D. Urgency   U = L  → action window: ≥75 "0–2y", ≥50 "3–5y", ≥25 "6–10y", else ">10y"
             override "immediate" if observed Red AND a break in the last 2 years
E. Economic  with break data:  BCR = PV(avoided break costs) / replacement cost
                PV = λ̂·L_km·c_break(D) · 0.9 · AF(r=4%, 30y);   E = 100·BCR/(1+BCR)
             without:          E = percentile rank of R / (cost per m)   (relative index)
Priority     P = 0.8·R + 0.2·E
```

**Avoiding double counting**
* Age and material appear **only** inside the condition model (or the age proxy), never
  as separate additive terms. A unit test (`test_no_double_counting_of_age`) checks that
  two pipes identical except for age, with the same observed condition, score the same.
* Urgency U is reported and drives scheduling windows, but is **not added** to P,
  because it is the same evidence as L.
* E does reuse L (through λ̂) and R. Its role is a deliberate *tilt toward
  cost-effective interventions at similar risk*. Setting `w_E = 0` gives pure risk
  ranking. Section 8 shows the ranking is insensitive to this weight.
* Pothole history and PCI (roads): see Section 3.

Every output row is additive at the top level (`priority_score = priority_risk_part +
priority_econ_part`). The row also carries all sub-scores, sources and a plain-text
`explanation`.

---

## 3. Road Maintenance Priority Score (Task 3)

**Condition.** PCI_eff (0–100, ASTM D6433 scale):

| Situation | PCI_eff |
|-----------|---------|
| PCI observed ≤ 3 y ago | PCI |
| PCI older | `PCI − r·Δt` (aged forward) |
| PCI missing, resurfacing year known | `100 − r·(years since resurfacing)` |
| Neither | network median (flagged) |

`r` is the PCI loss per year, **fitted from your own data** (median of (100−PCI)/age per
functional class, at least 30 observations). It falls back to a 2.5/yr placeholder if
there are too few observations. Climate scales r by up to ×1.3 at climate_index = 1, and
only when PCI must be extrapolated. Potholes **reported after the PCI survey** subtract
`min(15, 1.0 × potholes/km)`. Counting them only after the survey avoids double counting
distresses the PCI already captured.

`Dr = 100 − PCI_eff` (deficiency).

**Criticality Cr** (weighted mean, renormalised over available components):

| Component | Normalisation | Weight |
|-----------|---------------|--------|
| Traffic | `100·ln(1+AADT)/ln(1+AADT_p99)` | 0.30 |
| Heavy vehicles | same, on AADT × heavy% | 0.15 |
| Functional class | arterial 100 / collector 60 / local 30 / other 20 | 0.20 |
| Emergency route | 100 / 0 | 0.20 |
| Closure disruption | `100·clip01(detour_km / 5)` | 0.15 |

**Missing traffic data.** Missing AADT is imputed with the functional-class median from
the same dataset (flag `aadt_imputed_class_median`). If no AADT exists at all, the
traffic and heavy components are dropped and the rest renormalised. If nothing is known,
Cr = 50 (neutral).

```
Risk      Rr = √(Dr · Cr)
Treatment PCI ≥70 preventive | 50–70 mill & overlay | 25–50 thick overlay | <25 reconstruction
Economic  Er = percentile( ΔPCI × treatment life × lane-km × (1 + ln(1+AADT)/10) / cost )
Priority  Pr = 0.8·Rr + 0.2·Er
Window    on Dr: ≥60 "0–2y", ≥40 "3–5y", ≥20 "6–10y", else ">10y"
```

The treatment bands are illustrative defaults in the spirit of Shahin's PCI-band decision
trees. Calibrate them to your agency's treatment policy.

---

## 4. Coordination Opportunity Score (Task 4)

For each pipe–road pair with geographic overlap, there are **two separate outputs**:

**(1) Estimated net savings (currency, from explicit cost equations):**

```
gross   = trench surface restoration saved on the overlap (road treatment covers it)
        + one mobilisation saved
        + shared traffic control  f_tc · min(direct pipe works on overlap, direct road works)
        + avoided pavement-cut damage (only if separately the road would be done first):
              A_restore · c_overlay · life_loss · max(0, 1 − gap/T)
        + user delay avoided: shared closure days × AADT/1000 × c_delay
timing  = cost of aligning the two jobs, whichever is cheaper:
              advance road:  C_road · min(gap / treatment life, 1)   (wasted surface life)
              advance pipe:  C_pipe · (1 − (1+r)^−gap)               (time value of early capex)
              delay pipe:    gap · expected breaks/yr · c_break      (only if gap ≤ 5 y)
net_savings = gross − timing
```

**(2) Coordination Opportunity Score (unitless decision index):**

```
COS = 100 · o · τ · √(n_min · s) · (1 − ρ)
  o     overlap fraction of the pipe beneath the road
  τ     exp(−gap / 3 y)                 scheduling compatibility of target windows
  n_min min(P_pipe, P_road)/100          the LESS needy asset drives the score
  s     clip01( (net_savings / separate_cost) / 0.30 )
  ρ     (delay / 5 y) · L_pipe/100       risk of delaying the urgent pipe (if chosen)
```

**Gates (COS = 0, with `gate_reason`).** No overlap; either asset's priority below 40
(`pipe_does_not_need_work` / `road_does_not_need_work`); either target beyond the 10-year
horizon; net savings ≤ 0. Joint work is never promoted when one asset doesn't need
maintenance. Its savings are still reported, so a planner can see what was given up.

In the optimiser, a joint bundle is an option whose cost is `joint_cost + timing_cost`
and whose benefit is the sum of the two assets' benefits. It competes with doing either
asset alone, and each asset can be covered by at most one chosen option.

**Matching.** Matching uses geometry when both tables carry WKT lines in a projected CRS:
the overlap is the part of the pipe inside a buffer of (road width/2 + 3 m). Otherwise
same normalised street name (low confidence, overlap assumed 0.5, flagged). **Anonymised
pipes without location are never matched** (`test_no_match_without_location`).

---

## 5. Cost estimation models (Task 5)

### Equations (implemented in `costs.py`)

```
Pipe (open cut), segment of length L [m], diameter D [mm]:
  W = D/1000 + 2·0.3                      trench width [m]
  H = 1.2 + D/1000                        trench depth [m] (cover + pipe)
  c_pipe = a·(D/100)^b                    supply & lay   [¥/m]   ← fit to local bids
  c_exc  = W·H·c_m3                       excavation     [¥/m]
  c_res  = (W + 2·0.15)·c_m2              surface restoration [¥/m]
  total  = [L·(c_pipe + c_exc + c_res) + mobilisation]·(1 + f_tc + f_oh)·(1 + f_cont)
         + reopen_penalty
  reopen_penalty = A_restore · c_overlay · life_loss · max(0, 1 − t_since_resurf / T)

Road:   total = A·c_treatment(PCI band)·(1 + f_tc + f_oh)·(1 + f_cont) + mobilisation
Emergency break:  c_break = base·(D/100)^e + social_cost
Uncertainty: triangular multipliers on each unit cost → Monte Carlo P10/P50/P90
```

`fit_cost_function(diameters, unit_costs)` fits `a, b` by log-log least squares from your
bid tabulations (`test_fit_cost_function_recovers_parameters`).

### Parameters: placeholder values and where real values come from

| Parameter | Placeholder | Uncertainty | Where to get the real value |
|-----------|-------------|-------------|------------------------------|
| `pipe_unit_cost_a`, `_b` | 60,000, 0.9 | ×0.7–1.5 | MHLW 2011 renewal-cost functions (price-adjusted per the 2024 notice); your utility's bid tabulations; completed-project costs. Fit with `fit_cost_function` |
| `excavation_cost_per_m3` | 8,000 | ×0.7–1.6 | MLIT 土木工事標準積算基準書 unit rates; local contracts |
| `trench_restore_cost_per_m2_paved` | 12,000 | ×0.7–1.5 | Road administrator's restoration standard (本復旧) unit prices; bid data |
| `road_treatment_cost_per_m2` | 1,500 / 5,000 / 8,000 / 20,000 | ×0.7–1.5 | Pavement contracts by treatment; PMS unit-cost tables |
| `traffic_control_fraction`, `overhead_fraction`, `contingency_fraction` | 0.08, 0.15, 0.10 | — | Contract breakdowns (諸経費, 交通誘導員 costs) |
| `mobilisation_per_project` | 1.5 M | — | Bid line items |
| `emergency_repair_cost_base` | 1.5 M per break at 100 mm | ×0.5–2.0 | Work-order cost records. **Emergency vs. planned multipliers must come from your own work orders.** |
| `cut_life_loss_fraction` | 0.30 | 0.1–0.5 | Literature range [20]; better, a local forensic study |
| `user_delay_cost_per_day_per_1000_aadt` | 100,000 | — | MLIT time-value unit values × detour delay; FHWA LCCA method [13] |
| `discount_rate` | 0.04 | — | MLIT CBA guideline [19] |

**Real cost anchors found during research** (for context only, not used as defaults):
Davis, CA's 2022 pavement-degradation fee schedule ($1.04–1.14 per ft² by road class/PCI)
and the San Francisco/Calgary service-life-loss findings [20]. I could not retrieve
published Japanese ¥/m unit costs from the build environment, so **no real-world ¥
figure is hard-coded anywhere**. To switch from placeholders, copy
`configs/local_costs_template.yaml`, fill it in, set `is_placeholder: false`, and run
with `--config`.

---

## 6. Applying the framework to the 103k-segment dataset (Task 6)

```bash
cd digonce
python scripts/inspect_csv.py pipes path/to/pipe_condition.csv
python scripts/run_pipeline.py --pipes path/to/pipe_condition.csv --out outputs/run1
```

The first command shows the column mapping. The second runs the pipeline. If the mapping
is wrong, add `--column-map configs/column_map.example.yaml`.

| Step | What happens | Output |
|------|--------------|--------|
| 1 Train | XGBoost multiclass G/A/R on age, material, diameter, log length, region. Missing numerics are handled natively, and rare categories are bucketed. Default is unweighted, to keep probabilities calibrated (`balance=True` available). | model |
| 2 Evaluate | Stratified 5-fold **and** region-grouped 5-fold CV (transfer to unseen regions). Metrics: accuracy, balanced accuracy, macro-F1, quadratic-weighted κ (ordinal), log loss, multiclass Brier, Red recall/precision, Red ROC-AUC and average precision, Red calibration by decile and ECE. Baselines: majority prior and age-only logistic. | `run_report.json` |
| 3 Score | Condition (Section 2A rules) → L → C → R → E → P for **every** segment. SHAP contributions (XGBoost `pred_contribs`) give global importance. | `pipe_scores_all.csv/json` |
| 4 Sample 500 | Proportional stratified sample (region × material × condition × age band, collapsed if too fine), largest-remainder allocation, at least 25 Red/Amber/Green, seed-fixed, with `sample_weight` for re-estimating network totals. | `pipe_scores_sample500.csv/json` |
| 5 Costs | Section 5 equations, P10/P50/P90. | columns `replacement_cost`, `cost_p10…p90` |
| 6 Rank | `priority_rank` (network) and `prototype_rank` (sample). | |
| 7 Export | CSV + JSON (records plus metadata with `data_source` and `cost_basis`). | |

**When to use observed ratings instead of predictions.** Use the observed rating whenever
one exists. Predictions are for (a) unrated segments, (b) ageing a stale rating forward
(observation weight halves every 10 years), (c) label-blind validation, and (d) flagging
disagreements for re-inspection: pipes rated G but with high P(R) are good candidates.
Never overwrite an observed Red with a lower prediction without re-inspection.

Your anonymised dataset has no breaks, customers, critical facilities, geometry or
inspection dates. Consequently: L = D (condition only), C = diameter only, E is the
relative risk-per-cost rank, no road pairs are formed (by design), and every rating is
`observed_date_unknown`. A full-scale dry run on a synthetic 103k-row file with exactly
these columns finished in ~1 m 45 s.

---

## 7. Implementation (Task 7)

| Required function | Location | Returns |
|-------------------|----------|---------|
| `score_pipe_condition()` | `pipe.py` | D, source, observation weight, P(G/A/R) |
| `score_pipe_priority()` | `pipe.py` | all A–E sub-scores, priority, rank, window, `explanation` |
| `score_road_condition()` | `road.py` | PCI_eff, source, rate, pothole penalty, Dr |
| `score_road_priority()` | `road.py` | Cr components, risk, treatment, priority, `explanation` |
| `estimate_pipe_replacement_cost()` | `costs.py` | point, components, P10/P50/P90, flags, basis |
| `estimate_road_repair_cost()` | `costs.py` | treatment, PCI after, life, cost (+ intervals) |
| `calculate_coordination_opportunity()` | `coordination.py` | savings breakdown, net savings, COS, gates, `explanation` |
| `optimize_maintenance_budget()` | `optimize.py` | selected options and solver summary (OR-Tools SCIP MIP; multi-period option; greedy fallback) |

Supporting functions: `PipeConditionModel` (fit / predict_proba / explain / cross_validate),
`match_pipes_to_roads`, `build_options`, `select_representative`, `compare_strategies`,
`weight_sensitivity`, `cost_sensitivity`, `time_split_failure_check`, and `clean_*`
(alias mapping, unit detection, range checks, `dq_flags`).

**Tests** (`pytest tests`, 29 tests, ~5 s) cover:
* **Missing values:** consequence/criticality renormalisation and neutral 50; PCI and AADT
  imputation; age proxy; no condition model; missing diameter in costs.
* **Unusual measurements:** negative or zero diameter and length, PCI > 100, negative AADT,
  inch diameters auto-converted, heavy% given as a fraction, very long segments.
* **Inconsistent records:** duplicate IDs, future install years, inspection before install,
  age vs. install-year mismatch, unrecognised condition labels, unseen materials at
  prediction time.
* **Logic:** coordination gates, bundle preferred by the optimiser only when it pays,
  budget respected, reproducible sampling.

---

## 8. Validation and sensitivity (Task 8). SYNTHETIC results

Run: 20,000 synthetic pipes (G 65% / A 21% / R 14%) and 400 synthetic roads.
Budget = 10% of total network replacement cost (placeholder costs). In `label_blind` mode
every strategy sees only **out-of-fold predicted** condition and is graded against the
observed labels. `label_aware` rows read the labels and are shown only as a reference
ceiling.

**Condition model** (synthetic; stratified CV vs. region-grouped CV vs. baselines):

| | Stratified CV | Region-grouped CV | Age-only logistic | Majority prior |
|---|---|---|---|---|
| Macro-F1 | 0.684 | 0.476 | 0.373 | 0.263 |
| Quadratic κ | 0.742 | 0.461 | 0.226 | 0 |
| Red ROC-AUC | 0.946 | 0.794 | 0.788 | 0.5 |
| Red avg. precision | 0.781 | 0.470 | 0.327 | 0.139 |
| Red ECE (calibration) | 0.007 | 0.064 | — | — |
| Log loss | 0.497 | 0.949 | 0.793 | 0.879 |

The drop under region-grouped CV is the important lesson. Region carries a lot of signal,
so expect much weaker performance when predicting regions with no ratings. Report both
numbers on the real data.

**Strategy comparison at a fixed budget** (synthetic):

| Strategy | Red recall (count) | Red recall (km) | Red precision | Condition-pt·km per 100 M | Simulated risk-km share |
|---|---|---|---|---|---|
| Random | 0.107 | 0.109 | 0.150 | 12.4 | 0.100 |
| Oldest first | 0.281 | 0.276 | 0.389 | 26.2 | 0.143 |
| Worst predicted condition first | **0.610** | 0.600 | **0.767** | 47.2 | 0.208 |
| Break history first | 0.477 | 0.495 | 0.568 | 41.2 | 0.209 |
| Cost-effectiveness (D̂·km/cost) | 0.603 | **0.616** | 0.631 | **53.7** | 0.234 |
| Proposed priority (ranked) | 0.453 | 0.450 | 0.721 | 36.0 | 0.184 |
| Proposed priority per cost (greedy) | 0.510 | 0.566 | 0.562 | 50.9 | **0.249** |
| **Budget-optimised proposed (MIP, OPTIMAL)** | 0.508 | 0.566 | 0.562 | 50.9 | **0.249** |
| *Worst observed condition (label-aware reference)* | *0.770* | *0.762* | *1.000* | *52.8* | *0.205* |

How to read this honestly:
* **Ranking by priority alone is the wrong way to spend a budget.** It favours large,
  expensive, high-consequence pipes and covers fewer km. The budget-optimised version
  (or priority per cost) is the one to use.
* **The proposed score deliberately trades Red recall for consequence.** It finds fewer
  Red pipes than "worst predicted condition first" because it moves money toward
  high-consequence pipes. If the city's goal is "fix the most Red km", use
  worst-condition or cost-effectiveness ranking. If the goal is risk-weighted, use the
  optimised priority. Making that trade-off explicit is the main point of the comparison.
* "Simulated risk-km share" uses the framework's own R definition, so it is **biased in
  favour of the proposed method**. Red recall and condition per yen are independent of
  the formula.
* Nothing here shows that failures were prevented. That requires observed *future*
  failures: rank with data up to year T, then count breaks after T
  (`time_split_failure_check`).
* Failure-risk-first could only be run as "break history first" on synthetic breaks.
  Your anonymised dataset has no failure labels, so that strategy will be skipped on it.

**Weight sensitivity** (50 Monte Carlo draws, every weight perturbed ±50% then
renormalised):

| | Mean | Min |
|---|---|---|
| Spearman ρ vs. base ranking | 0.994 | 0.982 |
| Top-500 overlap | 0.858 | 0.698 |
| Red share of top 500 | 0.905 | 0.776 |

One-at-a-time sweeps (×0.5 and ×1.5): Spearman ≥ 0.989 everywhere and top-500 overlap
0.86–0.96. The top 500 move most when the diameter consequence weight, the road-above
weight or the risk share of priority is halved (overlap 0.86–0.89). Moving the **Amber
state score** has as much effect as any weight: the Red share of the top 500 is 0.97 at
A = 30 and 0.85 at A = 70, against ≈ 0.90 at the default A = 50. On data
that has only G/A/R and diameter (like yours), weights barely matter (ρ ≈ 0.9999). There,
the decisions are driven by the state mapping and the consequence definition, so those
are what to agree with stakeholders.

**Cost sensitivity at a fixed budget:** scaling all unit costs ×0.7 or ×1.5 changes the
selected set substantially (Jaccard 0.66 and 0.64), mostly because more or fewer pipes fit.
Changing the diameter exponent b from 0.9 to 0.7 or 1.2 gives Jaccard 0.85 and 0.75.
Doubling restoration cost gives Jaccard 0.92. **Takeaway:** the size of the programme is
cost-sensitive, while its composition is moderately sensitive to how cost scales with
diameter. Fitting `a, b` to local bids is the single most valuable calibration.

---

## 9. Example output for 10 assets (SYNTHETIC)

Ten pipes from the 500-segment prototype sample, spanning top, middle and bottom
(`outputs/example_synthetic/pipe_scores_sample500.csv`). Costs are placeholders.

| Rank | Pipe | Mat. | Ø mm | m | Age | Obs | D | L | C | R | E | **P** | Window | Cost (P10–P90) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | SYN-P006062 | CI | 500 | 168 | 74 | R | 99 | 100 | 56 | 75 | 24 | **64.3** | 0–2y | 67 M (60–85 M) |
| 2 | SYN-P000004 | CI | 600 | 165 | 82 | R | 97 | 87 | 64 | 75 | 9 | **61.5** | 0–2y | 77 M (69–98 M) |
| 3 | SYN-P003861 | CI | 500 | 97 | 85 | R | 96 | 97 | 53 | 72 | 18 | **61.2** | 0–2y | 40 M (35–50 M) |
| 4 | SYN-P016462 | CI | 200 | 175 | 91 | R | 100 | 100 | 40 | 64 | 50 | **60.9** | 0–2y | 34 M (31–42 M) |
| 151 | SYN-P015827 | AC | 150 | 43 | 31 | A | 34 | 44 | 26 | 34 | 5 | **28.2** | 6–10y | 8.3 M |
| 251 | SYN-P004553 | PVC | 100 | 102 | 35 | A | 30 | 29 | 18 | 23 | 3 | **18.6** | 6–10y | 12.9 M |
| 351 | SYN-P014371 | DI | 400 | 84 | 38 | G | 0 | 7 | 47 | 18 | 1 | **14.5** | >10y | 28.8 M |
| 498 | SYN-P018845 | DI | 800 | 31 | 23 | G | 0 | 0 | 56 | 1 | 0 | **0.9** | >10y | 20.2 M |
| 499 | SYN-P012796 | AC | 500 | 50 | 15 | G | 0 | 0 | 52 | 1 | 0 | **0.9** | >10y | 21.3 M |
| 500 | SYN-P007151 | PVC | 500 | 424 | 11 | G | 0 | 0 | 55 | 0 | 1 | **0.1** | >10y | 167 M |

Example explanation strings (verbatim output):

```
CI 500mm, age 74y, 168m | observed R -> D=99 (obs weight 0.93, pred 90) | L=100 [condition+breaks,
break score 100] | C=56 (diameter 77, customers 79, critical_facility 0, road_above 60) | Risk=75 |
Econ=24 (monetised_BCR) | Priority=64.3 = 59.6 risk + 4.7 econ | window 0-2y

DI 400mm, age 38y, 84m | observed G -> D=0 | L=7 [condition+breaks, break score 14] | C=47 (diameter 69,
customers 84, critical_facility 0, road_above 10) | Risk=18 | Econ=1 (monetised_BCR) | Priority=14.5
```

Road (top) and coordination (top) examples:

```
PCI_eff=0 [observed_aged_forward, pothole -7] -> Dr=100 | Cr=94 (traffic 90, heavy 81, road_class 100,
emergency 100, disruption 100) | Risk=97 | treatment reconstruction | Econ=98 | Priority=97.3 | window 0-2y

overlap 100%, needs P_pipe=50/P_road=80, gap 0y (same_window), gross savings 10,957,527 - timing cost 0
= net 10,957,527 (29% of separate cost) -> COS 68.8
```

Of 5,935 geometry-matched synthetic pairs, 466 passed all gates. 5,134 were gated because
the pipe did not need work, 281 because the road did not, and 54 had no net savings. The
budget optimiser chose 197 joint bundles, 1,425 pipe-only and 199 road-only jobs (SCIP,
optimal, gap < 0.01%).

---

## 10. CSV schemas

### Inputs (canonical names; aliases in `schema.py`, or map with `--column-map`)

**Pipes.** Required: `pipe_id`, `install_year` or `age_years`, `material`, `diameter_mm`,
`length_m`. Recommended: `region`, `condition` (G/A/R in any common spelling),
`inspection_year`. Optional: `n_breaks`, `break_obs_years`, `last_break_year`,
`soil_corrosivity` (0–1), `pressure_m`, `customers_served`, `critical_facility`,
`road_class_above`, `street_name`, `geometry_wkt` (projected metres), `status`.

**Breaks.** `pipe_id` (asset link), `break_date`; optional `street_name`. Breaks without
an asset ID are not guessed onto pipes.

**Roads.** Required: `road_id`, `length_m`. Recommended: `pci`, `pci_year`,
`last_resurface_year`, `aadt`, `functional_class`. Optional: `heavy_pct`, `width_m`,
`lanes`, `emergency_route`, `detour_km`, `pothole_count` (since last survey),
`climate_index` (0–1), `street_name`, `geometry_wkt`.

### Outputs

**`pipe_scores_*.csv`**: `pipe_id, material, diameter_mm, length_m, age_years,
install_year, region, condition_observed, condition_obs_score, condition_pred_score,
p_G, p_A, p_R, observation_weight, condition_score, condition_source, break_rate_hat,
break_exposure_km_yr, break_score, env_multiplier, likelihood_score, likelihood_basis,
cons_diameter, cons_customers, cons_critical_facility, cons_road_above,
consequence_score, consequence_coverage, consequence_flag, risk_score, urgency_score,
action_window, target_year_offset, replacement_cost, cost_per_m, cost_basis,
expected_breaks_per_year, pv_avoided_break_cost, benefit_cost_ratio, economic_score,
economic_basis, priority_risk_part, priority_econ_part, priority_score, priority_rank,
dq_flags, explanation, cost_p10, cost_p50, cost_p90, cost_flags` (+ `sample_weight,
stratum, prototype_rank` in the sample file). Break columns are present only when break
data exist.

**`road_scores.csv`**: `road_id, pci_observed, pci_effective, pci_source,
pci_rate_per_year, pothole_penalty, years_since_resurface, condition_deficiency_score,
pci_rate_source, crit_traffic, crit_heavy, crit_road_class, crit_emergency,
crit_disruption, criticality_score, criticality_coverage, aadt_used, criticality_flag,
risk_score, treatment, repair_cost, cost_basis, benefit_pci_lane_km_years,
economic_score, priority_*, priority_score, priority_rank, action_window,
target_year_offset, dq_flags, explanation`.

**`coordination_pairs.csv`**: `pipe_id, road_id, overlap_fraction, match_method,
overlap_used, overlap_assumed, overlap_length_m, pipe_priority, road_priority,
pipe_target_offset, road_target_offset, timing_gap_years, joint_timing, pipe_cost,
road_cost, separate_cost, joint_cost, save_trench_restoration, save_mobilisation,
save_traffic_control, save_avoided_pavement_cut, save_user_delay, gross_savings,
timing_cost, net_savings, net_savings_ratio, tau, need_min, savings_norm, delay_risk,
coordination_score, gate_reason, cost_basis, explanation`.

**`budget_selection.csv`**: `option_id, kind (pipe|road|joint), covers, cost, benefit, period`.

---

## 11. Limitations and missing-data requirements

1. **No failure ground truth.** G/A/R ratings are condition, not failure. L is an index.
   You need break or work-order records linked to asset IDs, with dates, to calibrate L
   into a probability and to run the time-split test.
2. **Ordinal → interval assumption.** G/A/R = 0/50/100 imposes equal spacing. The A value
   is an explicit, sensitivity-tested assumption.
3. **Rating protocol unknown.** If G/A/R came from desktop scoring that itself used age
   and material, the model partly re-learns that rule, and observed ratings then
   double-count age. Ask how the ratings were produced.
4. **Survival bias.** Already-replaced bad pipes are missing from the data, so old
   surviving pipes look healthier than their cohort really was [3].
5. **Region-transfer weakness.** Predictions for regions with no ratings are much less
   accurate. Use grouped-CV metrics for those.
6. **Placeholder costs.** Programme size and composition depend on costs (Section 8).
   Local costs are mandatory before any planning use.
7. **Consequence is thin in the anonymised data** (diameter only). Customers, critical
   facilities and hydraulic criticality are needed.
8. **Coordination needs location.** Anonymised segments cannot be paired with roads.
   Street-name matching is approximate. Geometry in a projected CRS is required.
9. **Single-period economics.** Deterioration during the planning horizon is
   approximated by action windows. Multi-period optimisation exists, but benefits are
   discounted rather than re-scored each year.
10. **Default weights are expert defaults, not elicited.** They are documented and
    sensitivity-tested, but not yet validated by stakeholders.

**Minimum data for a real deployment:** pipe GIS with IDs, install year, material,
diameter, length and geometry; break and work-order history with asset IDs and dates;
condition ratings with inspection date and protocol; customer counts per segment or
hydraulic-model criticality; road PMS with PCI, survey date, resurfacing year, AADT,
class and geometry; local unit costs (bid tabulations, completed projects); and capital
plans with budgets per year.

## 12. Recommendations for real municipal data

* **R1 — Calibrate costs first.** Fit `a, b` from 3–5 years of local bid tabulations,
  index to the current price year (MLIT construction deflator, per the MHLW 2024 notice),
  and replace the restoration, traffic-control and emergency-repair figures with values
  from work orders.
* **R2 — Link breaks to asset IDs and run the time-split validation**
  (`time_split_failure_check`). It is the only valid evidence of failure-prevention
  potential.
* **R3 — Add a survival model** (Weibull PH or random survival forest) for time to next
  break when break history exists [4]. Use it to turn L into a calibrated annual break
  probability and to replace the action windows with expected-cost timing.
* **R4 — Re-inspect disagreements.** Target segments rated G with high predicted P(R), and
  vice versa. This improves both the data and the model.
* **R5 — Hydraulic consequence.** Replace the diameter proxy with customers isolated
  (valve-isolation analysis) and pressure deficit from the hydraulic model.
* **R6 — Elicit weights with AHP** (consistency ratio < 0.1) in a workshop with water,
  roads and finance staff. Store them in YAML and re-run the sensitivity analysis.
* **R7 — Use the road PMS's own deterioration curves** (fitted per class/pavement type)
  instead of the linear rate.
* **R8 — Multi-year joint programming.** Run `optimize_maintenance_budget(periods=5)`
  with annual budgets, re-scoring each year's deterioration, and add a moratorium
  constraint against cutting roads resurfaced within N years.
* **R9 — Governance.** Version the config and data snapshots (ISO 55001 traceability),
  keep `dq_flags` reports, and publish the explanation strings with each capital plan.
