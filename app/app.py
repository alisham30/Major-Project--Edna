"""DeepTaxa-Env: web front end for the exploration-aware open-set eDNA classifier.

Run from the project root:
    python -m streamlit run app/app.py
"""
import ast
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DOCS = ROOT / "docs"

st.set_page_config(page_title="DeepTaxa-Env | Novel Taxon Discovery", page_icon=":material/biotech:", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
.badge {display:inline-block; padding:6px 14px; border-radius:999px; font-weight:700; font-size:1rem; letter-spacing:.02em;}
.badge-novel {background:#fde2e1; color:#9b1c1c; border:1px solid #f5b5b2;}
.badge-known {background:#dcf3e8; color:#0b5d3b; border:1px solid #a9dfc5;}
.badge-grey {background:#eceff3; color:#3b4654; border:1px solid #d3d9e0;}
.card {border:1px solid #dfe5ec; border-radius:12px; padding:16px 18px; background:#fbfcfd; height:100%;}
.card h4 {margin:0 0 6px 0; font-size:1.02rem;}
.card p {margin:0; color:#4a5563; font-size:.93rem;}
.kicker {color:#0b6e99; font-weight:700; text-transform:uppercase; font-size:.78rem; letter-spacing:.08em;}
.small {color:#5b6675; font-size:.88rem;}
.htest {border-left:5px solid #c9d2dc; padding:10px 16px; margin:10px 0; background:#fbfcfd; border-radius:6px;}
.htest.ok {border-left-color:#1f9d63;} .htest.no {border-left-color:#c8a03a;}
</style>
""", unsafe_allow_html=True)


# ============================================================================ data
@st.cache_resource(show_spinner="Loading the trained model and the Monterey Bay dataset...")
def engine():
    from engine import get_engine
    return get_engine()


@st.cache_data
def results():
    R = json.loads((OUT / "results.json").read_text())
    for k in ("novelty_rules", "rob_mean", "rob_std", "log_rho_range"):
        if isinstance(R.get(k), str):
            R[k] = ast.literal_eval(R[k])
    return R


@st.cache_data
def table(name, **kw):
    p = OUT / name
    return pd.read_csv(p, **kw) if p.exists() else None


def figure(name, caption=None):
    p = OUT / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.info(f"{name} is produced by running the notebook.")


def badge(text, kind):
    st.markdown(f'<span class="badge badge-{kind}">{text}</span>', unsafe_allow_html=True)


def card(title, body, kicker=None):
    k = f'<div class="kicker">{kicker}</div>' if kicker else ""
    st.markdown(f'<div class="card">{k}<h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


R = results()
DEMOS = {
    "Known genus: Paracalanus (copepod)": "ASV_76885cc2",
    "Never-seen genus: Spirostrombidium (ciliate)": "ASV_28282fbf",
    "Never-seen genus that fools softmax (95% confident)": "ASV_5b0d3f4c",
    "Dark taxon: top candidate, seen all 5 years": "ASV_79c648a4",
}


# ============================================================================ pages
def page_home():
    st.markdown('<div class="kicker">IPD project | D007 Alisha Mathias, D018 Deep Naidu, D051 Krish Kansara, D127 Shreyas Mekap | Guide: Dr. Kanchan Dabre</div>',
                unsafe_allow_html=True)
    st.title("Discovering Unknown Marine Species from Seawater DNA")
    st.markdown("#### A context-aware, exploration-aware open-set deep learning model for environmental DNA")
    st.write("AI that reads ocean DNA, names known species, and flags new ones.")

    c = st.columns(5)
    c[0].metric("Unique DNA variants (ASVs)", f"{R['n_asvs']:,}")
    c[1].metric("Without a genus in any database", f"{R['pct_unassigned_genus']:.0f}%")
    c[2].metric("Genus accuracy on unseen ASVs", f"{R['rob_mean']['acc_full']*100:.1f}%", help="Mean over 5 seeds. Chance level is 1.9%.")
    c[3].metric("AUROC on never-seen genera", f"{R['rob_mean']['auroc_full']:.2f}", help="Energy score, mean over 5 seeds.")
    c[4].metric("Hypotheses confirmed", "3 of 7")

    st.divider()
    st.subheader("The problem")
    a, b, c3 = st.columns(3)
    with a:
        card("Most ocean DNA is unnamed", "For the 18S V9 marker, 30 to 90 percent of sequence variants cannot be assigned. "
             f"In our Monterey Bay data it is {R['pct_unassigned_genus']:.0f} percent.", "Dark taxa")
    with b:
        card("Existing tools always guess", "BLAST, Kraken2 and Obitools return the closest match. They cannot tell a genuinely "
             "new organism apart from a noisy read.", "No novelty detection")
    with c3:
        card("Context is ignored", "Where and when the water was sampled (season, temperature, salinity, chlorophyll) "
             "is never used, although organisms follow these conditions.", "No environment")

    st.subheader("Our pipeline")
    st.graphviz_chart("""
    digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#9fb3c8", fillcolor="#f4f8fb"];
      edge [color="#7d8fa3"];
      s [label="Seawater sample\\n(CTD rosette, 18S V9)", fillcolor="#e6f2f8"];
      a [label="ASV sequence\\n~127 bp"]; e [label="Environment\\nmonth, temp, salinity, chl"];
      sb [label="Sequence branch\\none-hot 150x5\\nConv1D k=3,5,7 + max-pool\\n-> z_s (128)", fillcolor="#eaf4ea"];
      eb [label="Environment branch\\n17 -> 64 -> 128\\n-> z_e (128)", fillcolor="#eaf4ea"];
      f [label="Fusion\\n[z_s || z_e] 256\\nDense + Dropout 0.3", fillcolor="#fff6e5"];
      c [label="Classification head\\nsoftmax over 54 genera\\n-> genus", fillcolor="#e9eefb"];
      n [label="Novelty head\\nenergy E, density rho(e)\\ntau(e) = tau0 + alpha log rho\\nnovel if E > tau", fillcolor="#fbeaea"];
      r [label="Persistence + ranking\\nN(s) = sigmoid(E - tau) x P(s)^0.5", fillcolor="#f3eafb"];
      s -> a; s -> e; a -> sb; e -> eb; sb -> f; eb -> f; f -> c; f -> n; n -> r; }
    """, width="stretch")

    st.subheader("Explore")
    cols = st.columns(3)
    with cols[0]:
        card("Live demo", "Pick any ASV from the real dataset or paste your own sequence. See the genus, the energy score, the adaptive threshold and the novelty decision.", "Try it")
    with cols[1]:
        card("Novel taxa discovery", "Browse the ranked shortlist of 3,508 dark ASVs scored by novelty and persistence across years.", "Results")
    with cols[2]:
        card("Hypothesis testing", "Seven falsifiable claims, each with H0, H1, the test, the p-value, the effect size and the decision.", "Statistics")


def page_demo():
    eng = engine()
    cat = eng.catalogue
    st.title("Live demo: classify a DNA sequence")
    st.write("The trained model runs live in this page. Everything below is computed on the fly from the weights in `outputs/model_full.pt`.")

    mode = st.radio("Input", ["Pick an ASV from the dataset", "Paste a sequence"], horizontal=True)
    true_info, asv_id = None, None
    if mode == "Pick an ASV from the dataset":
        c1, c2 = st.columns([1, 1])
        with c1:
            demo = st.selectbox("Quick examples for the presentation", ["(choose your own below)"] + list(DEMOS))
        groups = {"Known genus, test ASVs (never trained on)": ("known", "test"),
                  "Never-seen genus (hidden test)": ("hidden_test", None),
                  "Dark taxa (no genus in any database)": ("dark", None),
                  "Known genus, training ASVs": ("known", "train")}
        if demo != "(choose your own below)":
            asv_id = DEMOS[demo]
        else:
            with c2:
                gname = st.selectbox("Group", list(groups))
            g, split = groups[gname]
            sub = cat[(cat.group == g) & ((cat.split == split) if split else True)].sort_values("total_reads", ascending=False)
            label = (sub.asv_id + "  |  " + sub.genus.fillna("no genus") + "  |  " + sub.total_reads.astype(str) + " reads")
            asv_id = st.selectbox(f"ASV ({len(sub):,} in this group, most abundant first)", sub.asv_id, format_func=dict(zip(sub.asv_id, label)).get)
        a = cat.set_index("asv_id").loc[asv_id]
        seq = a.sequence
        sids = eng.samples_of[asv_id]
        meta = eng.metadata.set_index("sample_id").loc[sids]
        sid = st.selectbox(f"Water sample it was found in ({len(sids)} samples)", sids, format_func=meta["label"].get)
        env = eng.env_for_sample(sid)
        m = meta.loc[sid]
        st.caption(f"Conditions: month {int(m.month)}, {m.temp:.2f} C, salinity {m.salinity:.2f}, chlorophyll {m.chlorophyll:.2f}, depth 0 m")
        true_info = dict(genus=a.genus, family=a.family, cls=a["class"], group=a.group, split=a.split)
    else:
        default = cat.set_index("asv_id").loc["ASV_76885cc2", "sequence"]
        seq = eng.clean(st.text_area("DNA sequence (A, C, G, T)", default, height=110))
        d = eng.condition_defaults(); rg = d["ranges"]
        c = st.columns(4)
        month = c[0].slider("Month", 1, 12, 5)
        temp = c[1].slider("Temperature (C)", rg["temp"][0] - 1, rg["temp"][1] + 1, d["temp"], 0.1)
        sal = c[2].slider("Salinity", rg["salinity"][0] - 0.3, rg["salinity"][1] + 0.3, d["salinity"], 0.01)
        chl = c[3].slider("Chlorophyll", 0.0, rg["chlorophyll"][1] + 5, d["chlorophyll"], 0.1)
        env = eng.env_for_conditions(month, temp, sal, chl)

    mut = st.slider("Introduce random mutations (percent of bases changed)", 0, 50, 0,
                    help="Simulates an organism the model has never met. Watch the energy rise and the decision flip.")
    if not seq:
        st.warning("Enter a sequence."); return
    s = eng.mutate(seq, mut, seed=1) if mut else seq
    r = eng.predict_one(s, env)

    st.divider()
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown('<div class="kicker">Output 1 | Classification head</div>', unsafe_allow_html=True)
        st.markdown(f"### Most likely genus: *{r['genus']}*")
        st.progress(min(1.0, r["confidence"]), text=f"softmax confidence {r['confidence']*100:.1f}%")
        top = r["top"].copy(); top["percent"] = top.probability * 100
        st.altair_chart(alt.Chart(top).mark_bar(cornerRadiusEnd=4, color="#2a6f97").encode(
            x=alt.X("percent:Q", title="probability (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("genus:N", sort="-x", title=None), tooltip=["genus", alt.Tooltip("percent:Q", format=".1f")]
        ).properties(height=190), width="stretch")
        if true_info:
            g = true_info["genus"]
            if isinstance(g, str):
                ok = g == r["genus"]
                st.markdown(f"Reference database label: *{g}* ({true_info['family'] if isinstance(true_info['family'], str) else 'family unknown'})")
                if true_info["group"] == "known":
                    badge("Correct genus" if ok else "Wrong genus", "known" if ok else "novel")
                    st.caption("Test ASV: the model never saw this sequence during training." if true_info["split"] == "test" else
                               "Training ASV: the model learned from this sequence." if true_info["split"] == "train" else "")
                else:
                    badge("This genus was never shown to the model", "grey")
                    st.caption("A correct answer is impossible here: the right move is to say 'I do not know this one'.")
            else:
                badge("No genus in any reference database", "grey")
                st.caption(f"Best BLAST/MEGAN6 assignment: class {true_info['cls'] if isinstance(true_info['cls'], str) else 'unknown'}.")
    with right:
        st.markdown('<div class="kicker">Output 2 | Novelty head</div>', unsafe_allow_html=True)
        if r["novel"]:
            badge("POSSIBLE NOVEL TAXON", "novel")
        else:
            badge("KNOWN GENUS", "known")
        c = st.columns(3)
        c[0].metric("Energy E", f"{r['energy']:.2f}", help="E = -log sum_k exp(f_k). Low = confident, high = the model shrugs.")
        c[1].metric("Threshold tau(e)", f"{r['tau']:.2f}", help="tau(e) = tau0 + alpha log rho(e), calibrated on never-seen validation genera.")
        c[2].metric("Sampling density rho(e)", f"{r['rho']:.3f}", help="How well explored these conditions are in the training samples.")
        st.progress(float(r["novel_prob"]), text=f"novelty probability sigmoid(E - tau) = {r['novel_prob']*100:.1f}%")
        ref = eng.reference
        base = alt.Chart(ref).transform_density("energy", groupby=["group"], as_=["energy", "density"], extent=[-32, 14]).mark_area(opacity=0.35).encode(
            x=alt.X("energy:Q", title="energy score (left = familiar, right = unfamiliar)"), y=alt.Y("density:Q", title=None, axis=None),
            color=alt.Color("group:N", scale=alt.Scale(range=["#2a6f97", "#e76f51", "#4c956c"]), legend=alt.Legend(orient="top", title=None)))
        marks = pd.DataFrame({"x": [r["energy"], r["tau"]], "what": ["this sequence", "threshold"]})
        rules = alt.Chart(marks).mark_rule(strokeWidth=2.5).encode(x="x:Q", color=alt.Color("what:N", scale=alt.Scale(range=["#111", "#9b1c1c"]), legend=None),
                                                                   strokeDash=alt.condition(alt.datum.what == "threshold", alt.value([5, 4]), alt.value([1, 0])))
        st.altair_chart((base + rules).properties(height=210), width="stretch")
        st.caption("Black line: this sequence. Dashed red: the novelty threshold. Shaded: energies of the test set.")
        if asv_id:
            yrs, P = eng.persistence(asv_id)
            N = r["novel_prob"] * P ** 0.5
            st.markdown(f"**Persistence** P(s) = {yrs} of {eng.n_years} years = {P:.2f}  |  **Novelty rank score** N(s) = sigmoid(E - tau) x P(s)^0.5 = **{N:.3f}**")

    with st.expander("What just happened, step by step (with the real numbers)"):
        ex = eng.explain(s, env)
        st.markdown(f"""
1. **One-hot encoding.** The {ex['length']} bp sequence became a {ex['onehot_shape'][0]} x {ex['onehot_shape'][1]} matrix (A, C, G, T, other), padded to L = 150{' and truncated' if ex['truncated'] else ''}.
2. **Three convolutions** slid 3-, 5- and 7-letter windows along the DNA with 64 learned patterns each. Global max-pooling kept each pattern's best hit (192 numbers). The strongest hits are below.
3. **Dense 192 -> 128** produced the DNA fingerprint z_s (norm {np.linalg.norm(ex['z_s']):.2f}).
4. **Environment branch** turned the 17-number context vector into the context fingerprint z_e (norm {np.linalg.norm(ex['z_e']):.2f}).
5. **Fusion** concatenated both (256) and mixed them through Dense + ReLU (dropout is off at inference).
6. **Classification head** produced 54 genus scores; softmax turned them into probabilities. Top genus: *{r['genus']}* at {r['confidence']*100:.1f}%.
7. **Energy** E = -log sum exp(scores) = **{r['energy']:.2f}**. Sampling density rho(e) = {r['rho']:.3f}, so log rho = {r['log_rho']:.2f}.
8. **Adaptive threshold** tau(e) = {eng.tau0 + eng.delta:.2f} {eng.alpha + eng.alpha_cal:+.2f} x log rho = **{r['tau']:.2f}**.
9. **Decision:** E {'>' if r['novel'] else '<='} tau, so the sequence is **{'flagged as a possible novel taxon' if r['novel'] else 'accepted as a known genus'}**.
""")
        st.markdown("**Strongest learned DNA motifs in this sequence** (the patterns the convolution filters fired on hardest)")
        st.dataframe(ex["motifs"].round(3), hide_index=True, width="stretch")
        ev = pd.DataFrame({"slot": ["depth_min/D", "depth_max/D"] + [f"month={i}" for i in range(1, 13)] + ["temp (z)", "salinity (z)", "chlorophyll (z)"],
                           "value": np.round(ex["env_vec"], 3)})
        st.markdown("**Environment vector e (17 numbers)**")
        st.dataframe(ev.set_index("slot").T, width="stretch")

    st.divider()
    st.subheader("Stress test: how does the model react to increasingly unfamiliar DNA?")
    st.write("We mutate a growing share of this sequence's bases (12 random versions per level) and re-score each one. A well-behaved open-set model should become less sure and start flagging.")
    if st.button("Run mutation stress test", type="primary"):
        st_df = eng.stress_test(seq, env)
        st_df["flag_pct"] = st_df.flag_rate * 100
        c1, c2 = st.columns(2)
        e_line = alt.Chart(st_df).mark_line(point=True, color="#2a6f97").encode(x=alt.X("mutation_pct:Q", title="bases mutated (%)"), y=alt.Y("energy:Q", title="mean energy"))
        t_line = alt.Chart(st_df).mark_line(strokeDash=[5, 4], color="#9b1c1c").encode(x="mutation_pct:Q", y="threshold:Q")
        c1.altair_chart((e_line + t_line).properties(height=260, title="Mean energy vs mutation (dashed: threshold)"), width="stretch")
        c2.altair_chart(alt.Chart(st_df).mark_bar(color="#e76f51").encode(x=alt.X("mutation_pct:O", title="bases mutated (%)"), y=alt.Y("flag_pct:Q", title="flagged as novel (%)", scale=alt.Scale(domain=[0, 100])))
                        .properties(height=260, title="Copies flagged as novel"), width="stretch")


def page_discovery():
    st.title("Novel taxa discovery")
    st.write("Every dark ASV (no genus in any reference database) scored by the model, then ranked by N(s) = sigmoid(E - tau) x P(s)^0.5. "
             "High energy alone could be a sequencing error; high energy that recurs across years is a real organism we cannot name.")
    df = table("candidate_novel_taxa_ranked.csv")
    if df is None:
        st.info("Run the notebook to produce outputs/candidate_novel_taxa_ranked.csv."); return
    c = st.columns(4)
    c[0].metric("Dark ASVs scored", f"{len(df):,}")
    c[1].metric("Flagged in most samples", f"{(df.flag_rate > 0.5).sum():,}")
    c[2].metric("Seen in all 5 years", f"{(df.years_seen == 5).sum():,}")
    c[3].metric("Strong candidates (N > 0.9)", f"{(df.N_score > 0.9).sum():,}")
    f = st.columns(3)
    yrs = f[0].slider("Minimum years seen", 1, 5, 3)
    pmin = f[1].slider("Minimum novelty probability", 0.0, 1.0, 0.5, 0.05)
    near = f[2].multiselect("Nearest known genus", sorted(df.nearest_genus.unique()))
    sel = df[(df.years_seen >= yrs) & (df.novel_prob >= pmin)]
    if near:
        sel = sel[sel.nearest_genus.isin(near)]
    st.markdown(f"**{len(sel):,} candidates** match the filters.")
    st.altair_chart(alt.Chart(df.sample(min(3000, len(df)), random_state=0)).mark_circle(opacity=0.55).encode(
        x=alt.X("novel_prob:Q", title="novelty probability sigmoid(E - tau)"), y=alt.Y("years_seen:O", title="years seen (of 5)", sort="descending"),
        size=alt.Size("reads:Q", scale=alt.Scale(range=[10, 400]), legend=None), color=alt.Color("N_score:Q", scale=alt.Scale(scheme="magma", reverse=True), title="N(s)"),
        tooltip=["asv_id", "nearest_genus", alt.Tooltip("novel_prob:Q", format=".3f"), "years_seen", "reads", alt.Tooltip("N_score:Q", format=".3f")]
    ).properties(height=300), width="stretch")
    show = sel[["asv_id", "nearest_genus", "nearest_conf", "energy", "novel_prob", "years_seen", "n_samples", "reads", "N_score", "sequence_5prime"]]
    st.dataframe(show.head(500).round(3), hide_index=True, width="stretch",
                 column_config={"N_score": st.column_config.ProgressColumn("N(s)", min_value=0, max_value=1, format="%.3f"),
                                "novel_prob": st.column_config.NumberColumn("novelty prob", format="%.3f")})
    st.download_button("Download the filtered shortlist (CSV)", sel.to_csv(index=False), "candidate_novel_taxa.csv", "text/csv")
    st.caption("'nearest_genus' is the closest known genus in the model's label space. A high novelty probability means the model does not believe it is that genus.")


H_TEXT = {
    "H1": ("Energy of never-seen genera = energy of known genera", "Never-seen genera have higher energy",
           "Energies are not normally distributed, so we compare the two groups by rank.",
           "The energy score works as a 'shrug meter': it rises for organisms the model has never met.", ""),
    "H2": ("Full model and sequence-only model have the same accuracy", "Adding environmental context raises genus accuracy",
           "Same data splits for every seed, so each seed is one paired replicate of the whole experiment.",
           "Environment improves classification.",
           "At one surface station, every ASV in a sample shares the same context, so context cannot separate genera within a sample. It did slightly improve novelty detection (AUROC 0.76 vs 0.74)."),
    "H3": ("Adaptive threshold and fixed threshold perform the same", "Adaptive threshold tau(e) beats a fixed threshold",
           "Both rules are tuned on the same validation data; balanced accuracy because novel rows outnumber known rows 7 to 1.",
           "Exploration awareness improves novelty decisions.",
           "Sampling density barely varies at a single station revisited 20 times (log rho spans about 1 unit), so the density term had no room to act. It never hurt either."),
    "H4a": ("No relation between sampling density and energy of known ASVs", "Known ASVs from rarely-sampled conditions have higher energy",
            "Rank-based, so it detects any monotone trend and is robust to outliers.", "Uncertainty tracks sampling effort.",
            "Energy is driven by the DNA sequence, not by how often the conditions were sampled."),
    "H4b": ("No relation between a sample's density and its share of flagged dark ASVs", "Rarely-sampled conditions carry more candidate novel taxa",
            "One point per water sample, so replicate reads do not inflate the sample size.", "Unexplored conditions hide more novelty.",
            "Same reason as H3: too little variation in sampling effort at one station."),
    "H5": ("One-off and recurring dark ASVs have the same energy", "They differ (we predicted they do NOT)",
           "Two-sided, because here the interesting outcome is no difference; Spearman between energy and years seen is printed as a check.", "",
           "Prediction confirmed: persistence is independent of energy, so combining them in N(s) adds information instead of repeating it."),
    "H6": ("Dark ASVs are flagged as often as known ASVs", "Dark ASVs are flagged more often",
           "Compares two flag rates from large independent groups.",
           "Real unassigned marine DNA looks unfamiliar to the model, as it should if many dark taxa are truly new.", ""),
}


def page_hypotheses():
    st.title("Hypothesis testing")
    st.write("Each design decision became a falsifiable claim, tested at significance level 0.05. A result that fails to reject H0 is reported as honestly as one that succeeds, with the reason.")
    ht = table("hypothesis_tests.csv")
    if ht is None:
        st.info("Run the notebook to produce outputs/hypothesis_tests.csv."); return
    ok = int(ht.matches_prediction.sum())
    c = st.columns(3)
    c[0].metric("Hypotheses tested", len(ht)); c[1].metric("Predictions confirmed", ok); c[2].metric("Significance level", "0.05")
    for _, h in ht.iterrows():
        h0, h1, why, yes, no = H_TEXT.get(h.H, ("", "", "", "", ""))
        p = "p < 0.000001" if h.p_value < 1e-6 else f"p = {h.p_value:.3g}"
        cls = "ok" if h.matches_prediction else "no"
        meaning = (yes if h.decision.startswith("REJECT") else no) or (no or yes)
        st.markdown(f"""<div class="htest {cls}">
<div class="kicker">{h.H} | {'prediction confirmed' if h.matches_prediction else 'prediction not confirmed'}</div>
<b>{h.claim}</b><br>
<span class="small"><b>H0:</b> {h0} &nbsp;|&nbsp; <b>H1:</b> {h1}</span><br>
<span class="small"><b>Test:</b> {h.test}. {why}</span><br>
<span class="small"><b>Result:</b> statistic = {h.statistic:.4g}, {p}, {h.effect_size} = {h.effect:+.3f} &rarr; <b>{h.decision}</b></span><br>
<span class="small"><b>What it means:</b> {meaning}</span>
</div>""", unsafe_allow_html=True)


def page_results():
    st.title("Results")
    rob = table("robustness_seeds.csv")
    t = st.tabs(["Classification", "Novelty detection", "Robustness (5 seeds)", "Training", "Embedding"])
    with t[0]:
        c = st.columns(4)
        c[0].metric("Top-1 genus accuracy", f"{R['top1']*100:.1f}%"); c[1].metric("Top-3", f"{R['top3']*100:.1f}%")
        c[2].metric("Family accuracy", f"{R['family_acc']*100:.1f}%"); c[3].metric("Chance level", f"{100/R['K']:.1f}%")
        figure("fig3_confusion.png", "Confusion matrix of the 15 most frequent test genera")
    with t[1]:
        c = st.columns(3)
        c[0].metric("AUROC, known-unknowns", f"{R['auroc_known_unknown']:.3f}"); c[1].metric("AUROC, never-seen genera", f"{R['auroc']:.3f}")
        c[2].metric("Max-softmax baseline AUROC", f"{R['msp_auroc']:.3f}")
        rules = pd.DataFrame(R["novelty_rules"]).round(3)
        st.dataframe(rules[["balanced_acc", "mcc", "precision", "recall", "known_flag_rate", "novel_catch_rate"]], width="stretch")
        figure("fig4_openset.png", "Energy distributions, ROC curve and the adaptive threshold against sampling density")
    with t[2]:
        if rob is not None:
            st.write("The whole experiment repeated with five random seeds and identical data splits.")
            long = rob.reset_index().melt(id_vars="seed", value_vars=["acc_full", "acc_seq"], var_name="model", value_name="accuracy")
            long["model"] = long.model.map({"acc_full": "full (sequence + environment)", "acc_seq": "sequence only"})
            st.altair_chart(alt.Chart(long).mark_bar().encode(x=alt.X("model:N", title=None, axis=alt.Axis(labels=False)), y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0.6, 0.9])),
                            color=alt.Color("model:N", scale=alt.Scale(range=["#2a6f97", "#bc4749"])), column=alt.Column("seed:N", title="seed")).properties(width=95, height=220))
            st.dataframe(rob.round(3), width="stretch")
    with t[3]:
        figure("fig2_training.png", "Training losses, validation accuracy and validation AUROC per epoch")
        figure("fig2b_threshold_params.png", "Learned threshold parameters tau0 and alpha")
    with t[4]:
        figure("fig6_embedding.png", "PCA of the fused 256-d embedding, and the known genera most often over-flagged")


def page_dataset():
    eng = engine()
    st.title("Dataset: MBARI Monterey Bay 18S V9 time series")
    st.write("Seawater collected by CTD rosette at a nearshore station in Monterey Bay, California, filtered by peristaltic pump and sequenced for the 18S rRNA V9 region. Published through OBIS and GBIF; source repository iobis/dataset-edna.")
    c = st.columns(5)
    c[0].metric("ASV x sample records", f"{R['n_records']:,}"); c[1].metric("Unique ASVs", f"{R['n_asvs']:,}")
    c[2].metric("Water samples", R["n_samples"]); c[3].metric("Years", ", ".join(map(str, R["years"])))
    c[4].metric("Learnable genera", R["K"])
    tax = eng.ns["taxa_table"]
    ranks = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    rk = pd.DataFrame({"rank": ranks, "assigned": [tax[r].notna().mean() * 100 for r in ranks]})
    a, b = st.columns(2)
    a.altair_chart(alt.Chart(rk).mark_bar(color="#2a6f97", cornerRadiusEnd=3).encode(x=alt.X("rank:N", sort=ranks, title=None), y=alt.Y("assigned:Q", title="% of ASVs with an assignment"))
                   .properties(height=260, title="Taxonomy gets darker at finer ranks"), width="stretch")
    meta = eng.metadata
    b.altair_chart(alt.Chart(meta).mark_circle(size=120, opacity=0.8).encode(x=alt.X("month:O", title="month"), y=alt.Y("temp:Q", title="sea-surface temperature (C)", scale=alt.Scale(zero=False)),
                   color=alt.Color("year:N", scale=alt.Scale(scheme="viridis")), tooltip=["label", "temp", "salinity", "chlorophyll", "n_asvs"])
                   .properties(height=260, title="Sampling effort: every dot is a water sample (all at 0 m)"), width="stretch")
    st.subheader("Water samples")
    st.dataframe(meta[["sample_id", "eventDate", "year", "month", "cruise", "replicate", "temp", "salinity", "chlorophyll", "n_asvs", "total_reads"]].round(3),
                 hide_index=True, width="stretch")
    st.subheader("Search the ASVs")
    q = st.text_input("Genus, family or ASV id contains", "Pseudo-nitzschia")
    cat = eng.catalogue
    hit = cat[cat.asv_id.str.contains(q, case=False) | cat.genus.fillna("").str.contains(q, case=False) | cat.family.fillna("").str.contains(q, case=False)] if q else cat
    hit = hit.assign(group=hit.group.map(eng_group_label))
    st.dataframe(hit[["asv_id", "class", "family", "genus", "group", "split", "total_reads", "n_samples", "years_seen", "seq_len"]].head(300), hide_index=True, width="stretch")


def eng_group_label(g):
    from engine import GROUP_LABEL
    return GROUP_LABEL.get(g, g)


def page_method():
    p = DOCS / "HOW_WE_BUILT_IT.md"
    st.markdown(p.read_text(encoding="utf-8") if p.exists() else "docs/HOW_WE_BUILT_IT.md is missing.")


pages = [
    st.Page(page_home, title="Overview", icon=":material/home:", default=True),
    st.Page(page_demo, title="Live demo", icon=":material/science:", url_path="demo"),
    st.Page(page_discovery, title="Novel taxa discovery", icon=":material/travel_explore:", url_path="discovery"),
    st.Page(page_hypotheses, title="Hypothesis testing", icon=":material/functions:", url_path="hypotheses"),
    st.Page(page_results, title="Results", icon=":material/monitoring:", url_path="results"),
    st.Page(page_dataset, title="Dataset", icon=":material/database:", url_path="dataset"),
    st.Page(page_method, title="How we built it", icon=":material/construction:", url_path="method"),
]
with st.sidebar:
    st.markdown("### DeepTaxa-Env")
    st.caption("Exploration-aware open-set deep learning for novel taxon discovery in marine eDNA")
st.navigation(pages).run()
with st.sidebar:
    st.divider()
    st.caption("Model: 118,840 parameters, trained on CPU. Data: MBARI via OBIS / GBIF, CC-BY 4.0.")
