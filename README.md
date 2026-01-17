This repository provides code for the simulation experiments developed in the paper:

> [Optimal Adjustment Sets for Nonparametric Estimation of Weighted Controlled Direct Effect (WCDE)](https://arxiv.org/pdf/2506.09871)  
> **Authors:** Ruiyang Lin, Yongyi Guo, and Kyra Gan  
> **Conference:** Thirty-Ninth Conference on Neural Information Processing Systems (NeurIPS 2025)

---

# WCDE Simulation Scripts

This repository contains all simulation scripts used in the paper to evaluate the proposed
**Weighted Controlled Direct Effect (WCDE)** estimator.

The experiments cover both **real-world Bayesian networks** and **synthetic nonlinear DAGs**,
and are designed to study how different **valid adjustment sets** affect estimator accuracy
and efficiency under finite samples.

Throughout all simulations, the underlying causal DAG is assumed to be known.

---

## Scripts Overview

| Script | Network / Setting | Description |
|:--------|:------------------|:-------------|
| `asia.py` | **ASIA** network | Small benchmark Bayesian network with a compact DAG structure, commonly used as a sanity-check example for causal identification and adjustment-set analysis. |
| `mildew.py` | **MILDEW** network | Large-scale Bayesian network with a complex DAG and dozens of nodes, representing a realistic high-dimensional causal structure for scalability testing. |
| `sachs.py` | **SACHS** (signaling pathway) | Biological causal network capturing protein signaling pathways, widely used as a benchmark DAG in causal inference and mediation analysis. |
| `bac-sim-nonlinear.py` | **Synthetic nonlinear DAG** *(A–G₁–Y with B₂, G₂)* | Synthetic causal DAG with multiple confounders influencing the treatment–outcome relationship, used to study direct effects under complex backdoor structures. |
| `med-sim-nonlinear.py` | **Synthetic nonlinear DAG** *(A–B₁–G₁–Y with G₂)* | Synthetic causal DAG with multiple mediators on the treatment–outcome pathway, capturing layered mediation structures between the exposure and the outcome. |

---

## Estimation Procedure

All scripts implement an **augmented inverse probability weighting (AIPW)** estimator for WCDE.
The estimator combines:

- outcome regression,
- propensity score weighting, and
- augmentation terms derived from the influence function of the WCDE estimand.

These augmentation terms reflect the marginalization structure of the WCDE target parameter
over non-mediators (confounders) and mediators, without requiring explicit numerical integration.

---

## Output

Each script reports empirical performance metrics of WCDE estimators under different valid
adjustment sets across multiple sample sizes, including measures of estimation accuracy and
efficiency.

---
