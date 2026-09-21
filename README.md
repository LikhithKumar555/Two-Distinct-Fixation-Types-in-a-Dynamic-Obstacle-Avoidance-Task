# Two Distinct Fixation Types in a Dynamic Obstacle-Avoidance Task

**An extension study using Pupil Core eye-tracking and quantile-based clustering (QuClu)**

This repository contains the analysis code, figures, and report for an extension of the *Dodge Asteroids* eye-tracking paradigm introduced by Heinrich et al. [1]. Participants steered a falling spaceship through an obstacle field while a controllable amount of random noise was added to their steering input. Fixations were clustered into data-driven types, and mixed-effects models tested how those types respond to increasing motor uncertainty.

---

## Table of Contents

- [Background](#background)
- [What This Study Adds](#what-this-study-adds)
- [Hypotheses](#hypotheses)
- [Experiment Design](#experiment-design)
- [Analysis Pipeline](#analysis-pipeline)
- [Key Results](#key-results)
- [Figures](#figures)
- [Limitations](#limitations)
- [References](#references)
- [Author](#author)

---

## Background

In Dodge Asteroids, the participant steers a spaceship that falls automatically from top to bottom, moving it left and right to avoid obstacles (comets/asteroids). Input noise is added to the steering, so the environment becomes less predictable without any change to the visual layout.

Earlier work split gaze into *foveal* vs. *peripheral* using a fixed boundary. Heinrich et al. [1] instead used unsupervised clustering on three fixation features and found two distinct fixation types:

| Feature | Description |
|---|---|
| `distance_to_spaceship` | Distance of the fixation from the spaceship (degrees of visual angle) |
| `fixation_duration` | Duration of the fixation (seconds) |
| `Dist_to_closest_obstacles` | Distance of the fixation to the nearest obstacle (degrees of visual angle) |

The original study used 6 participants and only crash-free trials. This repository tests whether the two-cluster structure still holds in a larger sample.

## What This Study Adds

| | Base paper [1] | This extension |
|---|---|---|
| Participants | 6 | 33 |
| Fixations analysed | 31,505 | 117,859 (of 162,882 recorded) |
| Trials analysed | Successful (crash-free) only | **Successful and failed** trials |
| Eye-tracker | Higher-frequency specialised system | Pupil Core, chin rest |
| Clustering | QuClu | QuClu (same method) |

Task, gameplay mechanics, and all 30 layouts were kept identical to the base paper.

## Hypotheses

- **H1**: Fixations cluster into two distinct types (near / far from the spaceship).
- **H2**: With increasing input noise, near-type fixations become shorter and more spatially focused.
- **H3**: With increasing input noise, far-type fixations become longer and move farther from nearby obstacles.
- **H4** (extension): With a larger sample and failed trials included, the two-cluster structure stays stable, but specific noise effects may differ.

## Experiment Design

- **Task:** Steer the spaceship left/right (`Y` / `M` keys on a QWERTZ keyboard) to reach the bottom without crashing. Falling speed is fixed.
- **Layouts:** 30 levels, each played multiple times; a crashed level may be repeated up to three times.
- **Noise manipulation:** Random noise added to the steering input.
- **Eye tracking:** Pupil Core, calibrated before the session, head stabilised on a chin rest.
- **Self-report:** After each level, participants rated their sense of control (1–7).
- **Session length:** about 45–60 minutes.

See [`Participant_Instructions.pdf`](Participant_Instructions.pdf) for the exact instructions given to participants.

## Analysis Pipeline

`Data_analysis.py` runs the full pipeline:

1. **Load** raw fixations (162,882).
2. **Clean:** remove fixations with missing or implausible feature values and those outside the active gameplay screen (117,859 remain, about 27.6% removed).
3. **PCA** on the three fixation features (explained variance: 65.8%, 33.0%, 1.2%).
4. **Quantile-based clustering** (`QuClu.myfunctions.alg_VS`) for K = 2 to 10.
5. **Model selection** with silhouette scores (best at K = 2, silhouette = 0.4034).
6. **Descriptive statistics** per cluster (mean, mode, SD, 95% CI).
7. **Linear mixed models** (`statsmodels`, participant as random intercept) testing the effect of `input_noise` on:
   - number of fixations per trial
   - distance to the spaceship
   - fixation duration
   - distance to the closest obstacle

> **Note:** an earlier version used a distance-based clustering algorithm, which put over 90% of fixations in a single cluster. Switching to QuClu, which handles skewed and heavy-tailed data, resolved this.

## Key Results

### Clusters (K = 2)

| | Cluster 0 (Type 0, near) | Cluster 1 (Type 1, far) |
|---|---|---|
| Fixations | 82,194 (69.7%) | 35,665 (30.3%) |
| Mean distance to spaceship | 6.212° | 24.278° |
| Mean fixation duration | 0.521 s | 0.129 s |
| Mean distance to closest obstacle | 3.987° | 20.824° |
| Modal fixations per trial | ≈ 144.5 | ≈ 35.1 |

The split closely matches the base paper (68.9% vs. 31.2%).

### Effect of input noise (linear mixed models)

| Outcome | Cluster 0 | Cluster 1 |
|---|---|---|
| Fixations per trial | β = 0.000, p = 1.000 | β = 0.000, p = 1.000 |
| Distance to spaceship | β = −0.003, **p = 0.023** | β = 0.000, **p < 0.001** |
| Fixation duration | β = 0.000, p = 0.990 | β = −0.005, p = 0.073 |
| Distance to closest obstacle | β = −0.001, p = 0.797 | β = −0.007, **p < 0.001** (opposite to H3) |

### Hypothesis outcomes

| Hypothesis | Outcome |
|---|---|
| H1 | **Supported.** Two clusters, with proportions nearly identical to the base paper. |
| H2 | **Partial.** Near fixations moved closer to the spaceship with noise; duration effect not significant. |
| H3 | **Partial / mixed.** Far-type distance to spaceship changed (very small effect); duration trend was marginal; distance to obstacles moved in the *opposite* direction. |
| H4 | **Partially confirmed.** Cluster structure stable; noise-related effects shifted. |

Given more than 100,000 fixations, some statistically significant effects are very small and should be interpreted with caution.

## Figures

**Silhouette score by number of clusters**

![Silhouette scores](figures/fig1_silhouette_score.png)

**Distribution of fixations per trial, by cluster** (red line = mode)

![Fixations per trial](figures/fig2_kde_fixations_per_trial.png)

**Summary panel: fixation counts and all three features, both clusters**

![Summary panel](figures/fig3_summary_panel.png)

**Feature distributions by cluster**

![Cluster distributions](figures/cluster_distributions.png)

## Limitations

- **Sampling rate:** Pupil Core samples at a lower frequency than the system used in the base paper, which may explain the weaker duration effects.
- **Failed trials not separated:** Successful and failed trials are pooled in the models, so their individual contribution to the divergent results cannot be isolated.
- **Sample homogeneity:** Participants came from a single university population.
- **Large-N significance:** With over 100,000 fixations, very small effects can reach significance.

## References

1. N. W. Heinrich, A. J. Najafabadi, and J. Perez-Osorio, "Unsupervised clustering uncovers two distinct types of fixational eye-movements in dynamic environments," *Proc. IEEE Signal Processing in Medicine and Biology Symposium*, 2025.
2. R. Engbert and R. Kliegl, "Microsaccades uncover the orientation of covert attention," *Vision Research*, 43(9), 1035–1045, 2003.
3. C. Hennig, C. Viroli, and L. Anderlucci, "Quantile-based clustering," *Electronic Journal of Statistics*, 13(2), 4849–4883, 2019.
4. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, 12, 2825–2830, 2011.
5. S. Seabold and J. Perktold, "Statsmodels: Econometric and statistical modeling with Python," *Proc. 9th Python in Science Conference*, 2010, pp. 92–96.

## Author

**Likhith Kumar Shivakumar**
Bielefeld University, Germany
[likhith.shivakumar@uni-bielefeld.de](mailto:likhith.shivakumar@uni-bielefeld.de)
