"""Inference engine behind the Streamlit app.

The app does not re-implement the model. It executes the notebook's own setup, data-loading, preprocessing and
model-definition cells (Steps 0, 1, 1b, 3 and 4), so the preprocessing, data splits and architecture are exactly
the code that produced the reported results. It then loads the trained weights from outputs/model_full.pt and the
calibrated threshold from outputs/results.json.
"""
import contextlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")  # the notebook imports pyplot; keep it headless inside the web server

import nbformat
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "eDNA_OpenSet_NovelTaxa.ipynb"
OUT = ROOT / "outputs"

GROUP_LABEL = {
    "known": "Known genus",
    "hidden_train": "Hidden genus (seen unlabelled in training)",
    "hidden_val": "Hidden genus (never seen, validation)",
    "hidden_test": "Hidden genus (never seen, test)",
    "dark": "Dark taxon (no genus in reference database)",
}
BASES = "ACGT"


class Engine:
    def __init__(self):
        self.ns = self._run_notebook_cells()
        ns = self.ns
        self.model = ns["ContextAwareOpenSetNet"](ns["K"]).to("cpu")
        self.model.load_state_dict(torch.load(OUT / "model_full.pt", map_location="cpu"))
        self.model.eval()

        self.results = json.loads((OUT / "results.json").read_text())
        self.delta = float(self.results["delta"])
        self.alpha_cal = float(self.results["alpha_cal"])
        self.tau0 = float(self.model.tau0.item())
        self.alpha = float(self.model.alpha.item())
        self.genera = list(ns["GENERA"])

        self._build_catalogue()
        self._build_reference()

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _run_notebook_cells():
        os.environ["EDNA_ROOT"] = str(ROOT)
        ns = {"display": lambda *a, **k: None, "__name__": "edna_notebook"}
        nb = nbformat.read(NOTEBOOK, as_version=4)
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                for cell in nb.cells:
                    if cell.cell_type != "code":
                        continue
                    src = cell.source
                    if "fig1_eda.png" in src:  # exploratory plots are not needed for inference
                        continue
                    exec(compile(src, "<notebook>", "exec"), ns)
                    if "class ContextAwareOpenSetNet" in src:
                        break
        finally:
            os.chdir(cwd)
        return ns

    def _build_catalogue(self):
        ns = self.ns
        cat = ns["asv_table"].merge(ns["taxa_table"], on="asv_id", how="left")
        cat["group"] = cat["genus"].map(ns["group_of"])
        split_of = {}
        for name in ("train_df", "val_df", "test_df"):
            for a in ns[name].asv_id.unique():
                split_of[a] = name.replace("_df", "")
        cat["split"] = cat["asv_id"].map(split_of).fillna("")
        reads = ns["asv_reads"].merge(ns["metadata_table"][["sample_id", "year"]], on="sample_id")
        cat = cat.merge(reads.groupby("asv_id")["year"].nunique().rename("years_seen"),
                        left_on="asv_id", right_index=True, how="left")
        cat["years_seen"] = cat["years_seen"].fillna(1).astype(int)
        self.n_years = len(self.ns["RESULTS"]["years"])
        self.catalogue = cat
        self.samples_of = ns["asv_reads"].groupby("asv_id")["sample_id"].apply(list).to_dict()
        meta = ns["metadata_table"].copy()
        meta["label"] = (meta["eventDate"].dt.strftime("%d %b %Y") + "  |  cruise " + meta["cruise"].astype(str)
                         + "  |  replicate " + meta["replicate"].astype(str))
        self.metadata = meta

    def _predict_frame(self, frame, bs=1024):
        # Same computation as the notebook's predict(); that helper lives in the training cell, which is not run here.
        out = {"pred": [], "y": [], "energy": []}
        with torch.no_grad():
            for X, e, y in self.ns["loader"](frame, False, bs):
                logits, energy, _, _ = self.model(X, e, self.ns["log_density"](e))
                out["pred"].append(logits.argmax(1)); out["y"].append(y); out["energy"].append(energy)
        return {k: torch.cat(v).numpy() for k, v in out.items()}

    def _build_reference(self):
        """Energy of every known-test, hidden-test and (a sample of) dark row, for 'where does my sequence fall'."""
        ns = self.ns
        pred = lambda _m, frame: self._predict_frame(frame)
        p_te, p_ut = pred(self.model, ns["test_df"]), pred(self.model, ns["utest_df"])
        dark = ns["dark_df"].sample(min(6000, len(ns["dark_df"])), random_state=0)
        p_dk = pred(self.model, dark)
        self.test_accuracy = float((p_te["pred"] == p_te["y"]).mean())  # sanity check vs the notebook
        self.reference = pd.concat([
            pd.DataFrame({"energy": p_te["energy"], "group": "Known genus (test ASVs)"}),
            pd.DataFrame({"energy": p_ut["energy"], "group": "Never-seen genus"}),
            pd.DataFrame({"energy": p_dk["energy"], "group": "Dark taxa"}),
        ], ignore_index=True)

    # ------------------------------------------------------------------ inputs
    def env_for_sample(self, sample_id):
        return np.asarray(self.ns["ENV"][sample_id], dtype=np.float32)

    def env_for_conditions(self, month, temp, salinity, chlorophyll):
        row = SimpleNamespace(depth_min=0.0, depth_max=0.0, month=int(month),
                              temp=float(temp), salinity=float(salinity), chlorophyll=float(chlorophyll))
        return np.concatenate([self.ns["env_vector"](row), self.ns["density_view"](row)]).astype(np.float32)

    def condition_defaults(self):
        m = self.ns["metadata_table"]
        return {c: float(m[c].median()) for c in ("temp", "salinity", "chlorophyll")} | {
            "ranges": {c: (float(m[c].min()), float(m[c].max())) for c in ("temp", "salinity", "chlorophyll")}}

    @staticmethod
    def clean(seq):
        return "".join(ch for ch in seq.upper() if ch.isalpha())

    @staticmethod
    def mutate(seq, pct, seed=0):
        rng = np.random.RandomState(seed)
        s = list(seq)
        n = int(round(len(s) * pct / 100.0))
        for i in rng.choice(len(s), size=n, replace=False) if n else []:
            s[i] = rng.choice([b for b in BASES if b != s[i]])
        return "".join(s)

    # ------------------------------------------------------------------ inference
    def score(self, seqs, env):
        ns = self.ns
        X = torch.tensor(np.stack([ns["one_hot"](s) for s in seqs]))
        E = torch.tensor(np.repeat(env[None], len(seqs), axis=0))
        with torch.no_grad():
            log_rho = ns["log_density"](E)
            logits, energy, tau, _ = self.model(X, E, log_rho)
            prob = F.softmax(logits, dim=1)
        tau_cal = tau + self.delta + self.alpha_cal * log_rho
        return dict(prob=prob.numpy(), energy=energy.numpy(), tau=tau_cal.numpy(),
                    log_rho=log_rho.numpy(), novel=(energy > tau_cal).numpy(),
                    novel_prob=torch.sigmoid(energy - tau_cal).numpy())

    def predict_one(self, seq, env, top=5):
        r = self.score([seq], env)
        p = r["prob"][0]
        order = np.argsort(-p)[:top]
        return dict(
            top=pd.DataFrame({"genus": [self.genera[i] for i in order], "probability": p[order]}),
            genus=self.genera[order[0]], confidence=float(p[order[0]]),
            energy=float(r["energy"][0]), tau=float(r["tau"][0]), log_rho=float(r["log_rho"][0]),
            rho=float(np.exp(r["log_rho"][0])), novel=bool(r["novel"][0]), novel_prob=float(r["novel_prob"][0]),
        )

    def explain(self, seq, env):
        """Intermediate quantities for the 'what just happened' walkthrough."""
        ns, m = self.ns, self.model
        X = torch.tensor(ns["one_hot"](seq))[None]
        E = torch.tensor(env)[None]
        motifs = []
        with torch.no_grad():
            x = X.transpose(1, 2)
            for k, conv in zip(ns["HP"]["kernel_sizes"], m.seq.convs):
                h = F.relu(conv(x))[0]                    # (64 filters, L positions)
                best, pos = h.max(dim=1)                   # global max-pool: best hit per filter
                for j in torch.argsort(best, descending=True)[:2].tolist():
                    p = int(pos[j]); a = max(0, p - k // 2)
                    motifs.append(dict(kernel=k, filter=j, score=float(best[j]), position=p + 1,
                                       motif=seq[a:a + k] if p < len(seq) else "(padding)"))
            z_s = m.seq(X)
            z_e = m.env(E[:, :ns["E_DIM"]])
            z = m.embed(X, E)
        return dict(length=len(seq), truncated=len(seq) > ns["HP"]["L"],
                    onehot_shape=tuple(X.shape[1:]), motifs=pd.DataFrame(motifs),
                    env_vec=env[:ns["E_DIM"]], z_s=z_s[0].numpy(), z_e=z_e[0].numpy(), z=z[0].numpy())

    def stress_test(self, seq, env, levels=range(0, 55, 5), reps=12):
        rows = []
        for pct in levels:
            seqs = [self.mutate(seq, pct, seed=r) for r in range(reps if pct else 1)]
            r = self.score(seqs, env)
            rows.append(dict(mutation_pct=pct, energy=float(np.mean(r["energy"])),
                             energy_sd=float(np.std(r["energy"])), flag_rate=float(np.mean(r["novel"])),
                             threshold=float(np.mean(r["tau"]))))
        return pd.DataFrame(rows)

    def persistence(self, asv_id):
        yrs = int(self.catalogue.set_index("asv_id").loc[asv_id, "years_seen"])
        return yrs, yrs / self.n_years


_ENGINE = None


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine()
    return _ENGINE


if __name__ == "__main__":
    import time
    t0 = time.time()
    eng = get_engine()
    print(f"loaded in {time.time() - t0:.1f}s | K={len(eng.genera)} | test accuracy {eng.test_accuracy:.4f} "
          f"(notebook reported {eng.results['top1']:.4f})")
    cat = eng.catalogue
    for grp in ("known", "hidden_test", "dark"):
        sub = cat[(cat.group == grp) & ((cat.split == "test") if grp == "known" else True)]
        a = sub.sort_values("total_reads", ascending=False).iloc[0]
        sid = eng.samples_of[a.asv_id][0]
        r = eng.predict_one(a.sequence, eng.env_for_sample(sid))
        print(f"{grp:12s} {a.asv_id} true={a.genus} -> {r['genus']} ({r['confidence']:.2f}) "
              f"E={r['energy']:.2f} tau={r['tau']:.2f} novel={r['novel']}")
