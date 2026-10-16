# Presentation notes

Speaking notes, the live demo script and likely viva questions. These are for the team, not for the audience.

## The 30-second pitch

Up to 90 percent of the DNA we pull out of seawater cannot be named, because reference databases are incomplete. Existing tools always force a best guess, so a genuinely new organism and a noisy read look the same. We built a deep learning model that reads the DNA and the sampling context, names the genus when it knows it, and says "I do not know this one" when it does not. Then it ranks those unknowns by how consistently they come back across years. On the real MBARI Monterey Bay time series it names unseen sequences with about 77 percent accuracy against a 2 percent chance level, and it flags real unnamed DNA four times as often as known DNA.

## Suggested flow, about 10 minutes

| Time | Show | Say |
|---|---|---|
| 0:00 | Slides: Introduction, Problem Definition | The dark-taxa problem; 75 percent of our ASVs have no genus. |
| 1:30 | Slides: Proposed Design | Two branches, fusion, two heads. Name the energy score and the adaptive threshold. |
| 3:00 | App: Overview | The pipeline diagram and the five headline numbers. |
| 3:30 | App: Live demo | Run the demo script below. This is the strongest part, so give it time. |
| 7:00 | App: Novel taxa discovery | The ranked shortlist; why persistence matters. |
| 8:00 | App: Hypothesis testing | Three confirmed, four not, and why (see below). |
| 9:00 | App: How we built it, section 10 | What went wrong and how we fixed it. This shows the depth of the work. |
| 9:45 | Limitations and future work | One station, surface only; depth-profile data is the next test. |

## Live demo script

Start the app before the presentation, because the first load takes about 5 seconds:

```bash
python -m streamlit run app/app.py
```

Open *Live demo* and use the **Quick examples** dropdown.

1. **"Known genus: Paracalanus (copepod)"**
   - Output 1 shows *Paracalanus* at 99.9 percent, with the badge *Correct genus*.
   - Output 2 shows *KNOWN GENUS* with energy about -24, far left of the dashed threshold at about -2.9.
   - Say: the model has never seen this exact sequence, because it is a test ASV, yet it is very sure.
2. **Same ASV, move the mutation slider to 15, then 30, then 50 percent.**
   - The energy climbs towards the threshold and the decision flips to *POSSIBLE NOVEL TAXON*.
   - Then click **Run mutation stress test**. Energy rises from -24 to about +0.7, and flagged copies go from 0 to 92 percent.
   - Say: this is exactly what open-set behaviour should look like. The less familiar the DNA, the louder the shrug meter.
3. **"Never-seen genus: Spirostrombidium (ciliate)"**
   - This genus was never shown to the model, so a correct name is impossible.
   - The best guess is only about 30 percent confident, energy is about +3.8, and it is flagged as a possible novel taxon.
4. **"Never-seen genus that fools softmax (95% confident)"**. This is the punchline.
   - The model's softmax says *Protoperidinium* at about 95 percent, but the true genus is *Dactylodinium*, which it has never seen.
   - The energy score (about +4) still flags it as novel.
   - Say: a plain softmax tool would have confidently given the wrong name. Energy uses the size of the raw scores, not only their ratio, so it catches this.
5. **"Dark taxon: top candidate, seen all 5 years"**
   - This ASV has no genus in any reference database. It is flagged (energy about +7) and was seen in all 5 years, so P(s) = 1.0 and N(s) is close to 1.
   - Say: a recurring, unfamiliar sequence is our best candidate for an organism that is genuinely new to science.
6. Open the **"What just happened, step by step"** panel on any example. It walks through all nine computation steps with the real numbers, including the DNA motifs the filters fired on.

If something fails on the day, open `eDNA_OpenSet_NovelTaxa.html` in a browser. Every result and figure is there.

## How to present the hypotheses that were not confirmed

Say it plainly. It is a strength, not a weakness.

- **H2 (environment did not raise accuracy):** all our samples come from one surface station. Every ASV in a water sample shares the same context, so context cannot tell two genera in the same sample apart. It did slightly improve novelty detection, with AUROC 0.76 against 0.74.
- **H3, H4a, H4b (the exploration-aware threshold had no measurable effect):** the station was sampled 20 times at the same depth, so the sampling density barely varies (`log rho` spans about 1 unit). The component has nothing to adapt to here. It never hurt performance, and it is designed for depth-profile or multi-station data.
- **The key line:** we tested every claim honestly, with five seeds and pre-stated predictions. The claims that failed failed for a reason we can name and test next.

## Numbers to remember

| | |
|---|---|
| Unique ASVs / water samples / years | 4,674 / 59 / 2006 and 2013-2016 |
| ASVs with no genus | 75 percent |
| Learnable genera | 54 (chance level 1.9 percent) |
| Genus accuracy on unseen ASVs | 76.6 percent, plus or minus 4.0 (5 seeds); top-3 about 87 percent |
| AUROC, never-seen genera | 0.76 energy, 0.79 max-softmax |
| AUROC, known-unknowns | 1.00 |
| Dark taxa flagged vs known flagged | about 60 percent vs 16 percent |
| Model size and training time | 118,840 parameters, about 2 minutes per model on CPU |
| Hypotheses confirmed | 3 of 7 (H1, H5, H6) |

## Likely viva questions

**What is an ASV?**
An amplicon sequence variant: one exact DNA sequence recovered after sequencing and error correction. It is a finer unit than an OTU because there is no similarity clustering.

**Why the 18S V9 region?**
It is a short, variable stretch of the small-subunit ribosomal RNA gene. It is present in all eukaryotes, so it captures plankton and protists. Being short (about 127 bp), it has limited resolution, which is one reason genus is the practical target.

**Why predict genus and not species?**
Species-level references are missing for most marine micro-eukaryotes: 85 percent of our ASVs have no species. Genus is a useful rank that the data can still support.

**Why a CNN with kernels 3, 5 and 7?**
DNA signal sits in short motifs. Three window sizes catch motifs of different lengths in parallel, and global max-pooling makes the model care whether a motif is present, not where. This follows DeepBarcoding and similar work.

**What is the energy score, and why not just use the softmax probability?**
`E = -log sum exp(logits)`. Softmax only looks at the ratio between scores, so it can be confident even when every score is low. Energy looks at their absolute size, so it rises when no genus scores strongly. The fourth demo example shows softmax at 95 percent on a genus the model never saw, while energy flags it.

**How is the threshold learned?**
With a hinge loss. It pushes the energy of known genera below `tau - m` and of hidden genera above `tau + m`, with margin m = 1. `tau0` and `alpha` are parameters trained alongside the network, then calibrated on never-seen validation genera.

**Why lambda = 0.5?**
It is the value from our design. It weights novelty at half the classification loss so that learning to name genera stays the main objective.

**What is "open-set" recognition?**
Classification where test inputs may belong to classes never seen in training. The model must name known classes and reject unknown ones, instead of forcing every input into a known class.

**How do you test novelty if you do not know which DNA is truly new?**
We hide whole genera that do have labels. The model never sees hidden-val or hidden-test genera, so we know the right answer is "unknown" and can measure how often the model says so.

**Why split by ASV instead of randomly?**
The same sequence appears in many water samples. A random row split would put identical DNA in training and test, so the score would measure memorisation, not generalisation.

**Why 5 seeds, and why a paired t-test?**
The known test set has only 70 ASVs, so one run can swing accuracy by several points. Each seed is one replicate of the whole experiment on identical splits, so comparing full against sequence-only within each seed is a paired design.

**Why Mann-Whitney U and not a t-test for H1?**
Energy distributions are skewed and multi-modal, not normal. Mann-Whitney compares them by rank and needs no normality assumption. Cliff's delta is its effect size: it is the share of (never-seen, known) pairs where the never-seen ASV has higher energy, minus the share where it has lower energy. Ours is 0.55, a large effect.

**Why balanced accuracy instead of F1?**
Novel rows outnumber known test rows about 7 to 1, so flagging everything already scores a high F1. Balanced accuracy averages the novel-catch rate and the known-keep rate, so it cannot be gamed that way.

**What does persistence add?**
A sequencing error is usually a one-off. A real organism comes back in other years. H5 showed persistence is independent of energy, so combining them in N(s) adds information instead of repeating it.

**Max-softmax had a slightly higher AUROC than energy. Does that undermine the method?**
On never-seen genera the two are close (0.79 against 0.76) and both carry similar information on this short marker. Energy's advantages are that its threshold is learned jointly with the network through the novelty loss, and that it catches confident mistakes like the fourth demo example. We report the comparison honestly.

**How fast is it compared with BLAST?**
After training, prediction is one forward pass with no database search. Scoring all 64,903 ASV-sample rows takes about 5 seconds on a laptop CPU, including encoding.

**What would you do next?**
Apply the model to depth-profile or multi-station eDNA, where the exploration-aware threshold has variation to act on. We would also try a transformer sequence branch (BarcodeBERT style), and confirm top candidates in the lab with long-read sequencing.
