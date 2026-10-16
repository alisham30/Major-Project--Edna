# Discovering Unknown Marine Species from Seawater DNA with a Context-Aware Deep Learning Model

**Exploration-Aware Open-Set Deep Learning for Novel Taxon Discovery in Marine Environmental DNA**

| Roll no. | Name |
|---|---|
| D007 | Alisha Mathias |
| D018 | Deep Naidu |
| D051 | Krish Kansara |
| D127 | Shreyas Mekap |

Under the guidance of Dr. Kanchan Dabre.

> AI that reads ocean DNA, names known species, flags new ones.

---

## 1. What this project does

Between 30 and 90 percent of the DNA sequence variants (ASVs) recovered from seawater cannot be assigned to a known organism because marine reference databases are incomplete. Existing tools (BLAST, Kraken2, Obitools) always return a best-guess label and cannot tell a genuinely new organism from a low-quality read. They also ignore *where and when* the water was sampled.

This repository implements a deep learning model that:

1. reads each ASV's DNA with a convolutional **sequence branch**,
2. reads the sample's **environmental context** (season, temperature, salinity, chlorophyll) with a separate branch,
3. **fuses** both into one representation,
4. predicts the most likely **genus** (classification head), and
5. decides whether the ASV is a **possible novel taxon** using an energy score, a sampling-density estimate and an **adaptive threshold** that is strict in well-studied conditions and lenient in rarely-studied ones (novelty head),
6. ranks candidate novel taxa by combining novelty with **persistence across years**, and
7. **tests every design claim statistically** (seven hypotheses, alpha = 0.05, repeated over five random seeds).

```
eDNA sample -> ASV + environmental data
            -> sequence branch   : one-hot (150 x 5) -> Conv1D k=3,5,7 (64 filters each) + global max-pool -> 192 -> 128   (z_s)
            -> environment branch: [depth_min/D, depth_max/D, onehot(month), temp_z, sal_z, chl_z] -> 64 -> 128             (z_e)
            -> fusion            : [z_s || z_e] = 256 -> Dense 256 + ReLU + Dropout 0.3                                    (z)
            -> classification    : logits f(z) in R^K -> softmax -> genus = argmax                                          (output 1)
            -> novelty           : E = -log sum exp f_k ;  rho(e) = sampling density ;  tau(e) = tau0 + alpha log rho(e)
                                   novel = 1 if E > tau(e)                                                                 (output 2)
            -> ranking           : P(s) = years seen / Y ;  N(s) = sigmoid(E - tau) * P(s)^beta
            -> hypothesis tests  : H1 ... H6
```

Training loss: `L = L_cls + 0.5 * L_novel`, where `L_cls` is cross-entropy over known genera and `L_novel` is a hinge loss that pushes known genera below the threshold and hidden (pretend-novel) genera above it.

## 2. Repository layout

| Path | What it is |
|---|---|
| `eDNA_OpenSet_NovelTaxa.ipynb` | **The deliverable.** Fully executed notebook. Every step has a caption with *what happens*, *what we predict* and *what we got*. |
| `eDNA_OpenSet_NovelTaxa.html` | The same notebook exported to HTML, viewable in any browser without Jupyter. |
| `build_notebook.py` | Source of the notebook. Edit this file, then rebuild (section 4). Do not edit the `.ipynb` by hand. |
| `data/raw/dataset-edna-main/raw/` | The three source tables: `asv_table.csv`, `taxa_table.csv`, `metadata_table.csv` (65 MB, committed). |
| `data/processed/` | Model-ready tables written by Step 1b of the notebook (regenerated on each run, not committed). |
| `outputs/` | Figures (`fig1` to `fig6`, committed), plus model weights, training history, ranked candidate novel taxa, hypothesis-test table and `results.json` (regenerated, not committed). |
| `requirements.txt` | Python dependencies. |

## 3. Dataset

MBARI, **"18S Monterey Bay Time Series: an eDNA data set from Monterey Bay, California, including years 2006, 2013 - 2016"**. Seawater was collected with a CTD rosette at a nearshore station, filtered with a peristaltic pump and metabarcoded for the 18S rRNA V9 region (micro-eukaryotes: plankton, protists). Taxonomy was pre-assigned by the data providers with BLAST against GenBank nr and MEGAN6.

| | |
|---|---|
| Records | 280,440 ASV x sample rows, of which 64,903 have non-zero reads |
| Unique ASVs | 4,674 sequences (4,711 ids; 37 share an identical sequence) |
| Samples | 60 filters = 20 cruises x 3 replicates, years 2006 and 2013 to 2016 |
| Genus assigned | 24.9 percent of ASVs. The other 75.1 percent are "dark taxa" |
| Learnable genera | 54 with at least 3 ASVs; 332 genera in total |
| Published at | OBIS `62b97724-da17-4ca7-9b26-b2a22aeaab51`, GBIF `e0b59ee7-19ae-4eb0-9217-33317fb50d47` |
| Source repo | https://github.com/iobis/dataset-edna (CC-BY 4.0) |
| Related paper | Djurhuus et al. (2020), *Nature Communications* 11, 254 |

If the `data/raw/dataset-edna-main/raw/` folder is missing, download the source repo as a zip and extract it so that the three CSVs sit at that path.

## 4. How to run

Requires Python 3.10 or newer. Tested on Windows 11 with Python 3.14, CPU only (no GPU needed).

Clone and install:

```bash
git clone <this-repo-url>
cd IPD
python -m pip install -r requirements.txt
python -m ipykernel install --user --name python3
```

Option A, just read it: open `eDNA_OpenSet_NovelTaxa.html` in a browser, or open the `.ipynb` in Jupyter / VS Code. All outputs are already there.

Option B, re-run everything (about 15 minutes on a laptop CPU):

```bash
python build_notebook.py
```

```bash
python -m nbconvert --to notebook --execute eDNA_OpenSet_NovelTaxa.ipynb --inplace --ExecutePreprocessor.timeout=-1
```

```bash
python -m nbconvert --to html eDNA_OpenSet_NovelTaxa.ipynb
```

Or open the notebook in Jupyter and choose *Run All*.

Environment variables for quick experiments:

| Variable | Default | Effect |
|---|---|---|
| `EDNA_EPOCHS` | `40` | training epochs per model |
| `EDNA_SEEDS` | `42,7,123,2024,31337` | seeds for the robustness study (the first is the primary seed) |

Example of a 2-minute smoke run: `EDNA_EPOCHS=1 EDNA_SEEDS=42,7` before the nbconvert command (on PowerShell: `$env:EDNA_EPOCHS="1"`).

Trained seed models are cached in `outputs/`, so an interrupted run resumes instead of retraining.

## 5. Notebook walkthrough

| Step | Content |
|---|---|
| 0 | Setup, seeds, all hyper-parameters in one dictionary |
| 1, 1b | Load the three source tables, rebuild model-ready tables, print dataset facts |
| 2 | Exploratory analysis: assignment rate per rank, sequence lengths, sampling effort, read distribution |
| 3 | One-hot encoding, environment vector, open-set split of genera (known / hidden-train / hidden-val / hidden-test), group split by ASV |
| 4 | Model definition, one equation per slide, parameter count, sanity forward pass |
| 5 | Training of the full model and the sequence-only ablation, loss and accuracy curves, learned `tau0` and `alpha` |
| 6 | Evaluation: genus accuracy, open-set AUROC, four novelty decision rules, confusion matrix, energy distributions, ROC, threshold-vs-density plot |
| 6b | Robustness: the whole experiment repeated over 5 seeds, mean and standard deviation of every headline number |
| 7 | Persistence and ranking: ranked shortlist of candidate novel taxa from the real unassigned ASVs |
| 8 | **Hypothesis testing** (see section 6 below) |
| 9 | PCA of the fused embedding, most over-flagged known genera |
| 10 | Scoreboard: every prediction next to its observed value |

## 6. Hypothesis tests

All tests at alpha = 0.05. Each cell prints H0, H1, the statistic, the p-value, an effect size, the decision and a one-line interpretation.

| ID | Claim from the design | Test |
|---|---|---|
| H1 | The energy score separates never-seen genera from known genera | Mann-Whitney U (one-sided), Cliff's delta |
| H2 | Environmental context improves genus accuracy | Paired t-test across 5 seeds; ASV-level McNemar as secondary |
| H3 | The exploration-aware threshold beats a fixed threshold | Paired t-test across 5 seeds on balanced accuracy; paired bootstrap as secondary |
| H4a | Known organisms look less familiar in rarely-sampled conditions | Spearman correlation between energy and `log rho(e)` |
| H4b | Rarely-sampled conditions carry more candidate novel taxa | Spearman correlation per water sample |
| H5 | Persistence carries information that energy does not (we predict no difference) | Mann-Whitney U (two-sided), Spearman |
| H6 | Real dark taxa are flagged as novel more often than known ASVs | Two-proportion z-test |

Results are in the notebook (Step 8) and in `outputs/hypothesis_tests.csv` after a run. See section 8 for the headline numbers.

## 7. Deviations from the slides, and why

- **Environment vector is 17-d, not 14-d.** Every sample at this station is a surface sample (0 m), so the two depth slots are constant. The slide's 14 slots are kept and z-scored temperature, salinity and chlorophyll from the CTD are appended.
- **Sampling density uses a Gaussian kernel on a continuous view of the environment** (cyclic month plus CTD values) instead of a hard count on the one-hot vector. One-hot months are all equidistant, so a hard count could only find replicates of the same cruise.
- **Thresholds are calibrated on never-seen genera.** `tau0`, `alpha`, the fixed cutoff and the max-softmax cutoff are tuned on a hidden-validation split that was never used in training. Calibrating on the hidden-train genera would cheat, because the novelty loss has already pushed their energies up.
- **Balanced accuracy and MCC are the primary novelty metrics**, not F1. Novel rows outnumber known test rows about 7 to 1, so F1 rewards a rule that flags everything.
- **Everything is repeated over 5 seeds.** The known test split holds only about 70 ASVs, so single-run numbers move by several points between seeds.

Everything else follows the slide equations one to one.

## 8. Headline results

Filled in from the executed notebook; see Step 10 for the full scoreboard.

Primary seed 42, test split of never-seen ASVs from known genera; robustness over 5 seeds with identical splits.

| Quantity | Primary seed | Mean +- sd over seeds |
|---|---|---|
| Genus top-1 accuracy (chance 1.9%) | 74.4% (top-3 86.7%, family 72.7%) | 76.6 +- 4.0% |
| Sequence-only ablation accuracy | 77.6% | 77.5 +- 3.1% |
| Open-set AUROC, energy, never-seen genera | 0.774 | 0.763 +- 0.021 |
| Open-set AUROC, max-softmax baseline | 0.757 | 0.786 +- 0.020 |
| Open-set AUROC, known-unknowns (trained-on hidden genera) | 1.000 | - |
| Novelty balanced accuracy: adaptive (calibrated) / fixed / MSP | 0.676 / 0.670 / 0.706 | 0.674 +- 0.031 / 0.676 +- 0.029 / 0.733 +- 0.028 |
| Learned threshold (hinge loss) | tau0 = -0.31, alpha = +0.22 | calibrated alpha_total 0.08 +- 0.76 |
| Dark taxa flagged as possible novel | 61.6% of rows | - |
| Dark ASVs seen in more than one year | 79.1% | - |

Hypothesis tests (alpha = 0.05):

| ID | Claim | p-value | Effect | Decision |
|---|---|---|---|---|
| H1 | Never-seen genera have higher energy than known genera | < 1e-6 | Cliff's delta = +0.549 | REJECT H0 |
| H2 | Adding environmental context (month + CTD) raises genus accuracy | 0.637 | mean accuracy gain (full - seq-only) = -0.009 | fail to reject H0 |
| H3 | Calibrated adaptive tau(e) beats a fixed tau on balanced accuracy | 0.713 | mean balanced-accuracy gain = -0.001 | fail to reject H0 |
| H4a | Known ASVs from well-sampled conditions get lower energy (negative correlation) | 0.317 | Spearman rho = -0.013 | fail to reject H0 |
| H4b | Samples from rarely-explored conditions carry a larger share of candidate novel taxa | 0.778 | Spearman rho = +0.101 | fail to reject H0 |
| H5 | One-off and recurring dark ASVs differ in energy (we predict they do NOT) | 0.63 | Cliff's delta = -0.012 | fail to reject H0 |
| H6 | Real dark taxa are flagged as novel more often than known test ASVs | < 1e-6 | flag-rate difference = +0.440 | REJECT H0 |

3 of 7 predictions were confirmed. The ranked shortlist of candidate novel taxa is in `outputs/candidate_novel_taxa_ranked.csv` after a run, and the top 20 are printed in Step 7 of the notebook.

What this means:

- **Supported.** The model names the genus of never-seen ASVs with about 77 percent top-1 accuracy from one forward pass. The energy score separates never-seen genera from known ones with a large effect (H1), real dark taxa are flagged about four times as often as known ASVs (H6), and persistence is independent of energy (H5), so the ranking score combines two non-redundant signals.
- **Not supported on this dataset.** Environmental context did not raise genus accuracy over five seeds (H2): every ASV in a cruise shares the same context, so it cannot separate genera within a sample. It did raise open-set AUROC slightly (0.763 vs 0.739 sequence-only). The exploration-aware slope had no measurable effect (H3, H4a, H4b) because `log rho(e)` spans barely one unit at a single surface station, and its calibrated sign flips between seeds. Max-softmax is marginally better than energy at ranking never-seen genera (0.786 vs 0.763).
- **Why this is still a result.** The components that failed are exactly the ones that need variation in depth, season coverage or station; the dataset has none. The design is shown to be safe (it never hurts), and the next step is a depth-resolved or multi-station eDNA series where the exploration-aware threshold has room to act.

## 9. Limitations and future work

- Reference labels come from BLAST + MEGAN6 at 95 percent identity; some "known" labels are themselves uncertain.
- Hidden-genus evaluation is a proxy for true novelty; wet-lab confirmation is the real test.
- All samples come from one surface station, so depth, the slide's primary exploration axis, is constant here. Applying the model to MBARI's depth-profile cruises is the natural next step.
- A transformer sequence branch (BarcodeBERT style) and prototype-based explanations (ProtoPNet style) are natural extensions.
