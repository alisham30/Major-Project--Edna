# How we built it

This page explains how the implementation was executed, from the design slides to the running web app.

## 1. From slides to a specification

Every numbered box on the *Proposed Design* slides became one equation and one piece of code. Nothing in the model was invented outside the slides; the only deviations are listed in section 5 and were forced by the data.

| Slide step | What it does | Equation | Implemented as |
|---|---|---|---|
| 1 | One-hot encoding | `X[i,c] = 1 if s_i = c`, `X in {0,1}^(150x5)` | `one_hot()` |
| 2-3 | Three convolutions + max-pool | `h_k = ReLU(W_k * X + b_k)`, `g_k[j] = max_i h_k[i,j]` | `SeqBranch`, kernels 3, 5, 7, 64 filters each |
| 4 | DNA fingerprint | `z_s = W_s [g3 g5 g7] + b_s in R^128` | `SeqBranch.fc` |
| 5 | Environment vector | `e = [d_min/D, d_max/D, onehot(month)]` | `env_vector()` (+3 CTD slots) |
| 6 | Context fingerprint | `z_e = W2 ReLU(W1 e + b1) + b2 in R^128` | `EnvBranch` |
| 7 | Fusion | `z = Dropout(ReLU(W_f [z_s z_e] + b_f))`, 256 | `ContextAwareOpenSetNet.fuse` |
| 8-10 | Classification | `p_k = softmax(W_c z + b_c)`, `y = argmax p_k` | `ContextAwareOpenSetNet.cls` |
| 11 | Classification loss | `L_cls = -sum y_k log p_k` | `F.cross_entropy` |
| 12 | Energy score | `E = -log sum_k exp(f_k)` | `-torch.logsumexp(logits)` |
| 13 | Sampling density | `rho(e)` = share of training samples near `e` | `log_density()`, Gaussian kernel |
| 14 | Adaptive threshold | `tau(e) = tau0 + alpha log rho(e)` | learnable `tau0`, `alpha` |
| 15 | Novelty decision | `novel = 1 if E > tau(e)` | evaluation code |
| 16 | Novelty loss | `mean_K max(0, E - tau + m) + mean_U max(0, tau - E + m)` | `novelty_loss()`, margin 1 |
| 17 | Total loss | `L = L_cls + 0.5 L_novel` | training loop |
| 18-19 | Persistence + rank | `P(s) = years seen / Y`, `N(s) = sigmoid(E - tau) P(s)^0.5` | `rank_candidates()` |

## 2. Tools and environment

| Purpose | Tool |
|---|---|
| Language | Python 3.14 |
| Deep learning | PyTorch 2.11 (CPU only, no GPU needed) |
| Data handling | pandas, NumPy |
| Metrics and splits | scikit-learn |
| Statistical tests | SciPy |
| Figures | matplotlib, seaborn, Altair |
| Notebook execution | Jupyter, nbconvert |
| Web front end | Streamlit |

Training one model for 40 epochs takes about two minutes on a laptop CPU. The model has 118,840 trainable parameters.

## 3. How the code is organised

```
build_notebook.py  --generates-->  eDNA_OpenSet_NovelTaxa.ipynb  --nbconvert executes-->  outputs/ (figures, weights, tables)
                                                                                  \-->  eDNA_OpenSet_NovelTaxa.html
app/engine.py  --runs the notebook's own data and model cells, loads outputs/model_full.pt-->  app/app.py (this website)
```

We write the notebook from a Python script instead of editing it by hand. Every step then follows the same caption format (*what happens*, *what we predict*, *what we got*), the whole experiment re-runs with one command, and every change is reviewable in git.

The web app does not re-implement anything. Its engine executes the notebook's own setup, data, preprocessing and model cells, then loads the trained weights. We verified that it reproduces the notebook's test accuracy exactly (74.45 percent).

## 4. Data pipeline

1. **Load** the three source tables named on the Data Description slide: `asv_table.csv`, `taxa_table.csv` and `metadata_table.csv`.
2. **Clean the taxonomy.** The reference pipeline writes placeholders such as `unassigned`, `unknown`, `no_hit`, `g_` and `s_`. These are converted to missing values.
3. **Keep real detections.** The table lists every ASV in every sample, including zero reads. Keeping only reads greater than zero reduces 280,440 rows to 64,903.
4. **Stable ASV identity.** Each sequence is hashed (SHA-1) into an id. Of 4,711 listed ids, 37 share an identical sequence, so there are 4,674 unique ASVs.
5. **Sample metadata.** Dates are parsed into year and month. Temperature, salinity and chlorophyll come from the CTD record of each sample.
6. **A surprise in the data.** Every sample turned out to be a surface sample (0 m). See section 5.

## 5. Deviations from the slides, and why

| Slide design | What we did | Why |
|---|---|---|
| 14-d environment vector (depth + month) | 17-d: the slide's 14 slots + z-scored temperature, salinity, chlorophyll | All samples are at 0 m, so the depth slots carry no information |
| `rho(e)` as a hard count of nearby samples | Gaussian kernel on a continuous view (cyclic month + CTD values) | One-hot months are all equally far apart, so a hard count could only find replicates of the same cruise |
| `tau0`, `alpha` learned by the hinge loss | Learned by the hinge loss, then re-calibrated on never-seen validation genera | The network can satisfy the hinge by moving energies, which leaves the threshold almost untouched |

## 6. Open-set split design

Open-set learning needs organisms the model has never seen, so we hide whole genera.

| Group | Genera | ASVs | Role |
|---|---|---|---|
| Known | 54 | 482 | Labelled. Split by ASV into train, validation and test |
| Hidden-train | 11 | 307 | Seen during training without labels; teaches the novelty loss what "unknown" looks like |
| Hidden-val | 129 | 187 | Never seen. Used only to choose the best epoch and calibrate thresholds |
| Hidden-test | 129 | 187 | Never seen. Used once, for the final numbers |
| Dark | none | 3,508 | No genus in any database. The real-world unknowns we rank |

The split is made **by ASV, not by row**. The same sequence appears in many water samples, so a random row split would put the same DNA in both training and test and inflate the scores.

## 7. Training

- Each step pairs a batch of known rows, which drive `L_cls` and the known side of `L_novel`, with a batch of hidden-train rows, which drive the unknown side of `L_novel`.
- Adam optimiser, learning rate 0.001, weight decay 0.0001, batch size 256, 40 epochs.
- The two threshold scalars `tau0` and `alpha` learn 20 times faster than the network.
- The kept checkpoint is the one with the best validation genus accuracy plus validation AUROC on never-seen genera.
- Two models are trained with identical settings: the full model and a sequence-only ablation with the environment branch switched off.

## 8. Evaluation

- **Classification:** top-1 and top-3 genus accuracy, macro-F1, and family accuracy derived from the predicted genus.
- **Novelty, threshold-free:** AUROC of the energy score for known versus never-seen genera, compared against the classic max-softmax baseline.
- **Novelty, decisions:** four rules are compared: the adaptive threshold as learned, the adaptive threshold calibrated, a fixed threshold, and max-softmax. All are tuned on the same validation data.
- **Metric choice:** novel rows outnumber known test rows about 7 to 1, so the primary metric is balanced accuracy, with the Matthews correlation coefficient (MCC) alongside. F1 would reward a rule that flags everything.

## 9. Robustness and hypothesis testing

- The whole experiment is repeated with **5 random seeds**: training, checkpoint choice and calibration, with identical data splits. Mean and standard deviation are reported for every headline number.
- Seven hypotheses are tested at alpha = 0.05. Each states H0 and H1, justifies its test, and reports the statistic, p-value, effect size and decision. Tests used: Mann-Whitney U, paired t-test across seeds, exact McNemar, paired bootstrap, Spearman correlation and the two-proportion z-test.

## 10. What went wrong along the way, and how we fixed it

This is the part that took the most work. Each problem was found by looking at the numbers and asking whether they made sense.

1. **The threshold never moved.** In the first runs `tau0` and `alpha` stayed near zero and the adaptive rule flagged nothing. The network was satisfying the hinge loss by shifting energies before the threshold could react. *Fix:* a 20 times larger learning rate for the two threshold scalars, plus calibration on validation data.
2. **A leak in calibration.** Tuning thresholds on the hidden-train genera gave almost perfect scores, because the novelty loss had already pushed exactly those energies up. *Fix:* a separate hidden-val split of genera the model never sees, used only for calibration and model selection.
3. **A misleading metric.** With 7 novel rows per known row, F1 favoured a rule that flags everything. *Fix:* balanced accuracy and MCC as primary metrics.
4. **Results that changed with the random seed.** With the data split held fixed, the environment branch's accuracy gain over sequence-only ranged from minus 8 to plus 4.5 points depending only on the seed. The known test set holds only 70 ASVs. *Fix:* 5 seeds, with paired tests across seeds.
5. **A degenerate density.** A kernel on one-hot months only counted replicate filters from the same cruise. *Fix:* a continuous view of the environment (cyclic month and CTD values) for the density term.
6. **Long CPU runs timed out.** Ten trainings took more than an hour. *Fix:* trained seed models are cached, so an interrupted run resumes instead of retraining.

## 11. The web front end

The app has seven pages. *Overview* and *Dataset* describe the problem and the data. The *Live demo* runs the model on any ASV or pasted sequence and walks through every step with real numbers. It also includes a mutation stress test. *Novel taxa discovery* is the ranked shortlist. *Hypothesis testing* and *Results* present the statistics, and this page explains the method.

## 12. Reproduce everything

```bash
python -m pip install -r requirements.txt
python build_notebook.py
python -m nbconvert --to notebook --execute eDNA_OpenSet_NovelTaxa.ipynb --inplace --ExecutePreprocessor.timeout=-1
python -m streamlit run app/app.py
```
