# Builds the literature review PDF.
# Run from the project root:  python docs/literature_review/build_literature_review.py
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "Literature_Review_eDNA_Novel_Taxon_Discovery.pdf"

INK = colors.HexColor("#1d2733")
MUTED = colors.HexColor("#5b6675")
ACCENT = colors.HexColor("#0b5d8a")
HEAD_BG = colors.HexColor("#0b5d8a")
ROW_ALT = colors.HexColor("#f2f6f9")
GRID = colors.HexColor("#c9d3dd")
YES, NO, PART = colors.HexColor("#d9efe3"), colors.HexColor("#f8dedd"), colors.HexColor("#fbefd3")

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Times-Roman", fontSize=11, leading=15.2,
                      alignment=TA_JUSTIFY, textColor=INK, spaceAfter=7)
BODY_L = ParagraphStyle("bodyl", parent=BODY, alignment=TA_LEFT)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=14.5, leading=18,
                    textColor=ACCENT, spaceBefore=14, spaceAfter=7)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.8, leading=15,
                    textColor=INK, spaceBefore=9, spaceAfter=4)
TITLE = ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=18, leading=23,
                       textColor=INK, alignment=TA_CENTER, spaceAfter=6)
SUB = ParagraphStyle("sub", parent=BODY, fontName="Helvetica", fontSize=11.5, leading=15, alignment=TA_CENTER,
                     textColor=MUTED, spaceAfter=4)
CELL = ParagraphStyle("cell", parent=BODY, fontName="Helvetica", fontSize=8.4, leading=10.6, alignment=TA_LEFT,
                      spaceAfter=0)
CELL_B = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
CELL_H = ParagraphStyle("cellh", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)
CELL_C = ParagraphStyle("cellc", parent=CELL, alignment=TA_CENTER)
CAPTION = ParagraphStyle("cap", parent=BODY, fontName="Helvetica-Oblique", fontSize=9, leading=11.5,
                         textColor=MUTED, alignment=TA_LEFT, spaceBefore=3, spaceAfter=10)
REF = ParagraphStyle("ref", parent=BODY, fontSize=10, leading=13, alignment=TA_LEFT, leftIndent=18,
                     firstLineIndent=-18, spaceAfter=5)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=4)
ABSTRACT = ParagraphStyle("abs", parent=BODY, fontSize=10.5, leading=14.5, leftIndent=0.6 * cm,
                          rightIndent=0.6 * cm)

story = []
P = lambda t, s=BODY: story.append(Paragraph(t, s))
H = lambda t: story.append(Paragraph(t, H1))
HH = lambda t: story.append(Paragraph(t, H2))
SP = lambda h=6: story.append(Spacer(1, h))


def bullets(items):
    for t in items:
        story.append(Paragraph(t, BULLET, bulletText="•"))


def table(rows, widths, header=True, zebra=True, extra=None, font_cells=CELL):
    data = []
    for r_i, row in enumerate(rows):
        style = CELL_H if (header and r_i == 0) else font_cells
        data.append([c if not isinstance(c, str) else Paragraph(c, style) for c in row])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.5, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4.5), ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
          ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), HEAD_BG))
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))
    if extra:
        st += extra
    t.setStyle(TableStyle(st))
    return t


# ============================================================================ title block
SP(10)
P("Literature Review", ParagraphStyle("kick", parent=SUB, fontName="Helvetica-Bold", textColor=ACCENT, fontSize=11))
P("Exploration-Aware Open-Set Deep Learning for Novel Taxon Discovery in Marine Environmental DNA", TITLE)
P("Identifying a real-world problem, surveying the literature and defining the research gap", SUB)
SP(8)
team = table([["Roll no.", "Name"], ["D007", "Alisha Mathias"], ["D018", "Deep Naidu"], ["D051", "Krish Kansara"],
              ["D127", "Shreyas Mekap"]], [2.6 * cm, 5.4 * cm])
team.hAlign = "CENTER"
story.append(team)
SP(6)
P("Under the guidance of <b>Dr. Kanchan Dabre</b>", SUB)
P("October 2026", SUB)
SP(10)

HH("Abstract")
P("Environmental DNA (eDNA) metabarcoding lets scientists survey marine life by sequencing the DNA that organisms "
  "shed into seawater. Its usefulness depends on matching each recovered sequence variant (ASV) against a reference "
  "database, and for marine micro-eukaryotes those databases are badly incomplete. In the Monterey Bay Aquarium "
  "Research Institute (MBARI) 18S V9 time series examined in this project, 75 percent of 4,674 ASVs have no genus "
  "assignment and 85 percent have no species assignment. These unassigned “dark taxa” are usually discarded, "
  "although some may be organisms new to science. This review surveys reference-based assignment tools, recent deep "
  "learning classifiers for DNA barcodes and eDNA, and open-set recognition methods from machine learning. It finds "
  "that current classifiers assume every input belongs to a known class, use the DNA sequence alone, rarely separate "
  "genuinely novel taxa from low-quality reads, and are seldom evaluated for novelty on real field-collected marine "
  "time series. We define six research gaps and outline an exploration-aware, context-aware open-set approach that "
  "addresses them.", ABSTRACT)
P("<b>Keywords:</b> environmental DNA, metabarcoding, 18S rRNA V9, dark taxa, open-set recognition, "
  "out-of-distribution detection, energy score, deep learning, marine biodiversity", ABSTRACT)

# ============================================================================ 1. introduction
H("1. Introduction and real-world problem")
HH("1.1 Why marine biodiversity monitoring matters")
P("Microscopic plankton and protists form the base of marine food webs. Phytoplankton alone account for roughly half "
  "of global primary production (Field et al., 1998). Changes in these communities affect fisheries, carbon cycling "
  "and human health; for example, blooms of the diatom <i>Pseudo-nitzschia</i> can produce the neurotoxin domoic acid, "
  "which harms marine mammals and closes shellfish harvests. Monitoring these communities by microscopy is slow, "
  "requires expert taxonomists and misses organisms that cannot be identified by shape.")
P("Environmental DNA metabarcoding offers a faster alternative (Taberlet et al., 2012). Seawater is filtered, a short "
  "marker gene is amplified from the captured DNA and sequenced, and the reads are denoised into amplicon sequence "
  "variants (ASVs), each an exact DNA sequence that usually corresponds to one kind of organism (Callahan et al., "
  "2017). For eukaryotic plankton, the V9 region of the 18S rRNA gene is a widely used marker (Amaral-Zettler et al., "
  "2009). The approach was used, for example, to reveal seasonal community shifts in Monterey Bay (Djurhuus et al., 2020).")

HH("1.2 The problem: most marine eDNA cannot be named")
P("Every ASV must be compared with a reference database to receive a name, and for marine micro-eukaryotes those "
  "databases are far from complete. Large ocean surveys have shown that much eukaryotic plankton diversity is only "
  "distantly related to known reference sequences (de Vargas et al., 2015). Our own analysis of the MBARI Monterey Bay "
  "18S V9 time series (nearshore station, 2006 and 2013 to 2016, 60 seawater filters) illustrates the scale of the "
  "problem:")
cov = table([["Taxonomic rank", "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"],
             ["ASVs with an assignment", "88%", "26%", "58%", "53%", "49%", "25%", "15%"]],
            [4.0 * cm] + [1.72 * cm] * 7, zebra=False)
story.append(cov)
P("Table 1. Share of the 4,674 MBARI ASVs assigned at each rank by the data provider's BLAST and lowest-common-ancestor "
  "pipeline. The low phylum value reflects the absence of a phylum rank for many protist groups in the reference "
  "taxonomy.", CAPTION)
P("Three consequences follow. First, three-quarters of the sequences, the dark taxa, are usually discarded, together "
  "with any genuinely new organisms among them. Second, when a sequence is left unassigned, current tools cannot say "
  "<i>why</i>: it may belong to an organism missing from the database, or it may simply be a noisy or chimeric read. "
  "Third, the tools ignore where and when the water was sampled, although related organisms tend to occur under "
  "similar seasonal and oceanographic conditions.")

HH("1.3 Problem statement")
P("<b>How can marine eDNA sequences be classified accurately at a useful taxonomic rank, while distinguishing "
  "potentially novel taxa from sequences that simply lack reliable reference matches?</b>")

HH("1.4 Review questions")
P("To understand how far existing work answers this question, the review addresses four questions:", BODY_L)
bullets(["<b>RQ1.</b> How are eDNA sequences assigned to taxa today, and what are the limits of reference-based methods?",
         "<b>RQ2.</b> How have deep learning models been applied to DNA barcode and eDNA classification?",
         "<b>RQ3.</b> How do these methods handle sequences from taxa absent from the training or reference data?",
         "<b>RQ4.</b> Do they use the environmental context of the sample, and are they evaluated on real field data?"])

# ============================================================================ 2. background
H("2. Background")
HH("2.1 Reference-based taxonomic assignment")
P("The most common approach aligns each sequence against a reference database. BLAST (Altschul et al., 1990) finds "
  "the most similar reference sequences, and a lowest-common-ancestor step, as in MEGAN (Huson et al., 2016), assigns "
  "the deepest rank shared by the best hits. Identity cut-offs decide the rank; the MBARI pipeline, for example, "
  "required 97 percent identity for species and 95 percent for genus. Faster alternatives include k-mer classifiers "
  "such as Kraken 2 (Wood et al., 2019), the naive Bayesian RDP classifier (Wang et al., 2007) and the OBITools "
  "toolkit for metabarcoding (Boyer et al., 2016). Curated databases such as PR2, the Protist Ribosomal Reference "
  "database (Guillou et al., 2013), improve label quality for protists, but cannot contain organisms nobody has yet "
  "sequenced.")
P("All of these methods share two properties. Their output depends entirely on the database: a sequence with no close "
  "match is left unassigned or assigned only at a coarse rank. And they must search the database for every query, "
  "which becomes costly when tens of thousands of ASVs are processed.")

HH("2.2 Open-set recognition and out-of-distribution detection")
P("Standard classifiers assume that every test input belongs to one of the training classes, the closed-set "
  "assumption. Open-set recognition removes that assumption: the model must classify known classes and reject inputs "
  "from unseen ones (Scheirer et al., 2013). A simple baseline is to reject inputs whose maximum softmax probability is "
  "low (Hendrycks &amp; Gimpel, 2017), but softmax probabilities are often overconfident on unfamiliar inputs. The "
  "energy score, the negative log-sum-exponential of the classifier's logits, was proposed as a better-calibrated "
  "alternative (Liu et al., 2020). These ideas are mature in computer vision but, as the survey below shows, are "
  "rarely applied to eDNA.")

# ============================================================================ 3. survey
H("3. Literature survey")
HH("3.1 Scope and selection")
P("We focused on recent studies (2022 to 2026) that apply deep learning to taxonomic assignment of DNA barcodes or "
  "eDNA, or that explicitly address sequences from taxa missing from reference data. Five studies were selected for "
  "detailed review because together they represent the main directions in the field: convolutional classifiers, "
  "self-supervised transformers, interpretable models, higher-rank assignment of unknown sequences and zero-shot "
  "annotation. Each is assessed against the four review questions.")

HH("3.2 Review of key studies")
P("<b>Yang et al. (2022), DeepBarcoding.</b> The authors treat a one-hot encoded DNA barcode as a one-dimensional "
  "image and classify it with a convolutional neural network followed by dense layers. The model reports more than 90 "
  "percent average accuracy across several DNA barcode benchmark datasets, showing that convolutional networks can "
  "learn taxonomically informative sequence patterns. <i>Limitations for our problem:</i> the model uses the sequence "
  "only and is closed-set; every input is forced into a known species, with no mechanism to flag taxa absent from "
  "training.")
P("<b>Millan Arias et al. (2026), BarcodeBERT.</b> A transformer is pre-trained with self-supervision on about 1.5 "
  "million invertebrate COI barcodes and then fine-tuned for downstream tasks. Pre-training outperforms fine-tuned "
  "general-purpose foundation models, especially at genus and species level, and matches BLAST at species-level "
  "classification while running about 55 times faster. <i>Limitations:</i> the model is specific to the COI marker and "
  "to invertebrates, uses no environmental or contextual input, and performs closed-set classification.")
P("<b>Waggoner et al. (2025).</b> This study builds an interpretable eDNA classifier on the prototype-based ProtoPNet "
  "framework, which classifies a sequence by comparing it with learned representative examples and shows the base "
  "patterns most distinctive of each species. The authors report accuracy above previous work on a challenging fish "
  "eDNA dataset while making each prediction explainable. <i>Limitations:</i> detection of unseen or novel species is "
  "left as future work, and no environmental context is used.")
P("<b>Villon et al. (2026).</b> Motivated directly by incomplete marine reference databases, the authors propose a "
  "position-aware convolutional network with five parallel nucleotide-channel heads for the fish 12S (teleo) marker, "
  "aiming to assign sequences reliably at genus and family level even when the species is missing from the database. "
  "They report 94.7 percent genus-level and 86.5 percent family-level accuracy, outperforming OBITools, Kraken 2 and "
  "other assignment tools. <i>Limitations:</i> the model always returns a best-guess label, without a calibrated "
  "“possibly new” flag, and evaluation is in silico.")
P("<b>Stalder et al. (2025).</b> This zero-shot method embeds raw sequences into a phylogeny-informed space and refines "
  "predictions with species co-occurrence priors, so that species absent from training can still be predicted. It "
  "correctly predicts unseen species in about a quarter of tests among more than 31,000 candidates, and its predictions "
  "on real samples largely agree with a traditional pipeline. <i>Limitations:</i> it relies on a phylogeny and on "
  "ecological co-occurrence data rather than physical sampling conditions such as season, temperature or depth, and it "
  "is demonstrated on fish communities rather than micro-eukaryotic plankton.")
P("Other work supports the same picture. Flück et al. (2022), for example, showed that convolutional networks can "
  "annotate short eDNA sequences much faster than alignment-based pipelines, again within a closed set of reference "
  "taxa.")

summary = [["Study", "Marker and taxa", "Method", "Key findings", "Limitations"],
           ["Yang et al., 2022 (IEEE/ACM TCBB)", "DNA barcodes; benchmark datasets",
            "CNN on one-hot DNA treated as a 1-D image", "Over 90% average accuracy across barcode benchmarks",
            "Sequence only; closed-set; no novel-taxon flag"],
           ["Millan Arias et al., 2026 (Bioinformatics Advances)", "COI; invertebrates",
            "Self-supervised transformer pre-trained on 1.5 M barcodes", "Matches BLAST at species level, about 55x faster",
            "COI and invertebrates only; no context; closed-set"],
           ["Waggoner et al., 2025 (Scientific Reports)", "eDNA; fish",
            "Prototype-based interpretable CNN (ProtoPNet)", "Higher accuracy than previous work; explains each prediction",
            "Novel species detection left as future work; no context"],
           ["Villon et al., 2026 (Biology)", "12S teleo; marine fish",
            "Position-aware CNN with five nucleotide-channel heads", "94.7% genus, 86.5% family accuracy; beats OBITools and Kraken 2",
            "Always returns a best guess; no calibrated novelty flag; in silico"],
           ["Stalder et al., 2025 (PLOS Comp. Biol.)", "eDNA; fish",
            "Zero-shot phylogenetic embeddings with co-occurrence priors", "About 24% of unseen species predicted among 31,000+ candidates",
            "Needs phylogeny and co-occurrence data; no physical sampling context"]]
story.append(KeepTogether([Paragraph("3.3 Summary of the reviewed studies", H2),
                           table(summary, [3.3 * cm, 2.6 * cm, 3.6 * cm, 3.6 * cm, 3.6 * cm]),
                           Paragraph("Table 2. Summary of the five studies reviewed in detail.", CAPTION)]))

cap_intro = [Paragraph("3.4 Comparison of capabilities", H2), Paragraph(
    "Table 3 compares the reviewed approaches, together with the standard reference-based tools, on the capabilities "
    "needed to solve the problem stated in Section 1.3.", BODY)]
Y, N, Pa = "Yes", "No", "Partial"
cap = [["Approach", "Uses DNA sequence", "Uses sampling context", "Flags possibly novel taxa",
        "Threshold adapts or is learned", "Separates novelty from noise", "Tested on real field series",
        "No database search per query"],
       ["BLAST + LCA (MEGAN)", Y, N, Pa, N, N, Pa, N],
       ["Kraken 2", Y, N, Pa, N, N, Pa, N],
       ["DeepBarcoding", Y, N, N, N, N, N, Y],
       ["BarcodeBERT", Y, N, N, N, N, N, Y],
       ["Waggoner et al.", Y, N, N, N, N, Pa, Y],
       ["Villon et al.", Y, N, N, N, N, N, Y],
       ["Stalder et al.", Y, Pa, Y, N, N, Pa, Y],
       ["<b>Proposed approach</b>", Y, Y, Y, Y, Y, Y, Y]]
fills = {Y: YES, N: NO, Pa: PART}
extra = []
for r in range(1, len(cap)):
    for c in range(1, len(cap[0])):
        extra.append(("BACKGROUND", (c, r), (c, r), fills[cap[r][c]]))
extra.append(("BOX", (0, len(cap) - 1), (-1, len(cap) - 1), 1.4, ACCENT))
cap_rows = [row if r == 0 else [Paragraph(row[0], CELL_B)] + row[1:] for r, row in enumerate(cap)]
captab = table(cap_rows, [3.0 * cm] + [1.95 * cm] * 7, zebra=False, extra=extra, font_cells=CELL_C)
story.append(KeepTogether(cap_intro + [captab, Paragraph(
    "Table 3. Capability comparison. <i>Partial</i>: reference-based tools leave unmatched sequences unassigned but "
    "cannot say whether they are novel or noisy; Stalder et al. use ecological co-occurrence rather than physical "
    "sampling conditions; field validation is partial where results are compared with an existing pipeline or a single "
    "eDNA dataset rather than a multi-year time series.", CAPTION)]))

# ============================================================================ 4. research gaps
H("4. Research gaps")
P("Reading the studies together reveals six gaps between what the literature provides and what the problem in "
  "Section 1.3 requires.")
gaps = [
    ("G1. No calibrated novelty decision in eDNA classifiers.",
     "DeepBarcoding, BarcodeBERT, Waggoner et al. and Villon et al. are closed-set: every input receives a known label. "
     "Open-set methods such as the energy score (Liu et al., 2020) are established in machine learning, but none of "
     "the reviewed studies trains such a score jointly with an eDNA classifier to produce a calibrated "
     "“possibly novel” decision."),
    ("G2. Novel taxa are not separated from low-quality reads.",
     "Reference-based tools leave unmatched sequences unassigned without saying why. None of the reviewed studies uses "
     "temporal evidence, such as whether an unfamiliar sequence recurs across years, to distinguish a real but unknown "
     "organism from a one-off sequencing artefact."),
    ("G3. Physical sampling context is ignored.",
     "All reviewed deep learning models classify from the sequence alone. Stalder et al. add ecological co-occurrence, "
     "but none of the reviewed studies fuses physical sampling conditions such as season, temperature, salinity, "
     "chlorophyll or depth with the sequence representation."),
    ("G4. Novelty thresholds are not exploration-aware.",
     "Where a cut-off exists, it is a single fixed value. None of the reviewed studies adapts the novelty threshold to "
     "how thoroughly the relevant environmental conditions have been sampled, although unknown taxa are more likely to "
     "appear in rarely sampled conditions."),
    ("G5. Evaluation rarely matches the real problem.",
     "Evaluation is often in silico or on curated benchmarks, and focuses on fish 12S or invertebrate COI. Marine "
     "micro-eukaryotes and the 18S V9 marker, where the dark-taxa problem is most severe, are under-represented. Few "
     "studies test novelty by hiding whole taxa, repeat experiments across random seeds, or report hypothesis tests."),
    ("G6. Inference depends on database search.",
     "BLAST and k-mer tools search a reference database for every query sequence, which is slow for large surveys, "
     "whereas a trained network classifies with a single forward pass.")]
for title, text in gaps:
    story.append(KeepTogether([Paragraph(f"<b>{title}</b> {text}", BODY)]))

HH("Summary of the research gap")
P("<b>None of the reviewed approaches combines accurate genus-level classification, a learned and exploration-aware "
  "novelty decision, environmental context and persistence-based separation of novel taxa from noise, evaluated with "
  "an open-set protocol on a real, multi-year marine eDNA time series.</b>")

# ============================================================================ 5. proposed direction
H("5. Proposed direction")
P("To address these gaps, this project proposes a context-aware, exploration-aware open-set deep learning model, "
  "evaluated on the MBARI Monterey Bay 18S V9 time series. Table 4 maps each gap to a design component.")
prop = [["Gap", "Proposed component", "How it addresses the gap"],
        ["G1", "Novelty head with an energy score and a hinge-style novelty loss",
         "Learns, together with the classifier, a threshold above which a sequence is flagged as a possible novel taxon"],
        ["G2", "Persistence-based ranking",
         "Ranks unassigned ASVs by novelty and by the share of years in which they recur, so one-off artefacts are demoted"],
        ["G3", "Environment branch and feature fusion",
         "Encodes month and CTD measurements as a context vector and fuses it with the DNA representation"],
        ["G4", "Exploration-aware adaptive threshold",
         "Makes the cut-off depend on sampling density: strict in well-sampled conditions, lenient in rarely sampled ones"],
        ["G5", "Open-set evaluation protocol",
         "Hides whole genera for training, calibration and testing; repeats experiments over several seeds; tests "
         "pre-stated hypotheses"],
        ["G6", "Neural network inference",
         "Classifies every ASV in a single forward pass, without a database search"]]
story.append(KeepTogether([table(prop, [1.2 * cm, 5.2 * cm, 10.3 * cm]), Paragraph(
    "Table 4. Mapping of research gaps to the components of the proposed approach.", CAPTION)]))

# ============================================================================ 6. conclusion
H("6. Conclusion")
P("Environmental DNA can reveal marine biodiversity at a scale that microscopy cannot, but its value is limited by "
  "incomplete reference databases: in the MBARI 18S V9 series, three-quarters of the sequences cannot be named to "
  "genus. The reviewed deep learning studies show that neural networks classify DNA barcodes accurately and quickly, "
  "and recent work has begun to address sequences missing from reference data. However, these models remain largely "
  "closed-set, use the sequence alone, cannot separate genuinely novel taxa from noisy reads, and are rarely evaluated "
  "for novelty on real marine time series. Closing these gaps requires a model that combines classification, a "
  "learned and exploration-aware novelty decision, environmental context and temporal persistence, tested with a "
  "rigorous open-set protocol. That is the direction this project takes.")

# ============================================================================ references
H("References")
refs = [
    "Altschul, S. F., Gish, W., Miller, W., Myers, E. W., &amp; Lipman, D. J. (1990). Basic local alignment search "
    "tool. <i>Journal of Molecular Biology, 215</i>(3), 403–410.",
    "Amaral-Zettler, L. A., McCliment, E. A., Ducklow, H. W., &amp; Huse, S. M. (2009). A method for studying protistan "
    "diversity using massively parallel sequencing of V9 hypervariable regions of small-subunit ribosomal RNA genes. "
    "<i>PLoS ONE, 4</i>(7), e6372.",
    "Boyer, F., Mercier, C., Bonin, A., Le Bras, Y., Taberlet, P., &amp; Coissac, E. (2016). obitools: A unix-inspired "
    "software package for DNA metabarcoding. <i>Molecular Ecology Resources, 16</i>(1), 176–182.",
    "Callahan, B. J., McMurdie, P. J., &amp; Holmes, S. P. (2017). Exact sequence variants should replace operational "
    "taxonomic units in marker-gene data analysis. <i>The ISME Journal, 11</i>, 2639–2643.",
    "de Vargas, C., Audic, S., Henry, N., et al. (2015). Eukaryotic plankton diversity in the sunlit ocean. "
    "<i>Science, 348</i>(6237), 1261605.",
    "Djurhuus, A., Closek, C. J., Kelly, R. P., Pitz, K. J., Michisaki, R. P., Starks, H. A., Walz, K. R., "
    "Andruszkiewicz, E. A., Olesin, E., Hubbard, K., Montes, E., et al. (2020). Environmental DNA reveals seasonal "
    "shifts and potential interactions in a marine community. <i>Nature Communications, 11</i>, 254.",
    "Field, C. B., Behrenfeld, M. J., Randerson, J. T., &amp; Falkowski, P. (1998). Primary production of the biosphere: "
    "Integrating terrestrial and oceanic components. <i>Science, 281</i>(5374), 237–240.",
    "Flück, B., et al. (2022). Applying convolutional neural networks to speed up environmental DNA annotation in a "
    "highly diverse ecosystem. <i>Scientific Reports, 12</i>.",
    "Guillou, L., Bachar, D., Audic, S., et al. (2013). The Protist Ribosomal Reference database (PR2): A catalog of "
    "unicellular eukaryote small sub-unit rRNA sequences with curated taxonomy. <i>Nucleic Acids Research, 41</i>(D1), "
    "D597–D604.",
    "Hendrycks, D., &amp; Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in "
    "neural networks. <i>International Conference on Learning Representations (ICLR)</i>.",
    "Huson, D. H., Beier, S., Flade, I., et al. (2016). MEGAN Community Edition: Interactive exploration and analysis of "
    "large-scale microbiome sequencing data. <i>PLoS Computational Biology, 12</i>(6), e1004957.",
    "Liu, W., Wang, X., Owens, J. D., &amp; Li, Y. (2020). Energy-based out-of-distribution detection. <i>Advances in "
    "Neural Information Processing Systems, 33</i>.",
    "MBARI (Chavez, F., &amp; Pitz, K.). 18S Monterey Bay Time Series: An eDNA data set from Monterey Bay, California, "
    "including years 2006, 2013–2016. Published through OBIS and GBIF (GBIF dataset "
    "e0b59ee7-19ae-4eb0-9217-33317fb50d47).",
    "Millan Arias, P., Sadjadi, N., Safari, M., Gong, Z., Wang, A. T., Haurum, J. B., Zarubiieva, I., Steinke, D., "
    "Kari, L., Chang, A. X., Lowe, S. C., &amp; Taylor, G. W. (2026). BarcodeBERT: Transformers for biodiversity "
    "analyses. <i>Bioinformatics Advances, 6</i>(1), vbag054. https://doi.org/10.1093/bioadv/vbag054",
    "Scheirer, W. J., de Rezende Rocha, A., Sapkota, A., &amp; Boult, T. E. (2013). Toward open set recognition. "
    "<i>IEEE Transactions on Pattern Analysis and Machine Intelligence, 35</i>(7), 1757–1772.",
    "Stalder et al. (2025). Zero-shot deep learning for the annotation of unknown eDNA sequences using co-occurrences "
    "and phylogenetic embeddings. <i>PLOS Computational Biology, 21</i>(12), e1013776.",
    "Taberlet, P., Coissac, E., Pompanon, F., Brochmann, C., &amp; Willerslev, E. (2012). Towards next-generation "
    "biodiversity assessment using DNA metabarcoding. <i>Molecular Ecology, 21</i>(8), 2045–2050.",
    "Villon, S., Mangeas, M., Berteaux-Lecellier, V., Vigliola, L., &amp; Lecellier, G. (2026). Fine-grained assignment "
    "of unknown marine eDNA sequences using neural networks. <i>Biology, 15</i>(3), 285. "
    "https://doi.org/10.3390/biology15030285",
    "Waggoner et al. (2025). Creating interpretable deep learning models to identify species using environmental DNA "
    "sequences. <i>Scientific Reports, 15</i>, 27436. https://doi.org/10.1038/s41598-025-09846-7",
    "Wang, Q., Garrity, G. M., Tiedje, J. M., &amp; Cole, J. R. (2007). Naive Bayesian classifier for rapid assignment "
    "of rRNA sequences into the new bacterial taxonomy. <i>Applied and Environmental Microbiology, 73</i>(16), "
    "5261–5267.",
    "Wood, D. E., Lu, J., &amp; Langmead, B. (2019). Improved metagenomic analysis with Kraken 2. <i>Genome Biology, "
    "20</i>, 257.",
    "Yang, C.-H., Wu, K.-C., Chuang, L.-Y., &amp; Chang, H.-W. (2022). DeepBarcoding: Deep learning for species "
    "classification using DNA barcoding. <i>IEEE/ACM Transactions on Computational Biology and Bioinformatics, 19</i>(4), "
    "2158–2165.",
]
for r in refs:
    story.append(Paragraph(r, REF))


def on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(GRID); canvas.setLineWidth(0.6)
    canvas.line(2.2 * cm, h - 1.5 * cm, w - 2.2 * cm, h - 1.5 * cm)
    canvas.setFont("Helvetica", 8.2); canvas.setFillColor(MUTED)
    canvas.drawString(2.2 * cm, h - 1.25 * cm, "Literature Review: Open-Set Deep Learning for Novel Taxon Discovery in Marine eDNA")
    canvas.line(2.2 * cm, 1.55 * cm, w - 2.2 * cm, 1.55 * cm)
    canvas.drawString(2.2 * cm, 1.1 * cm, "Mathias, Naidu, Kansara, Mekap")
    canvas.drawRightString(w - 2.2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.1 * cm,
                        bottomMargin=2.1 * cm, title="Literature Review: Open-Set Deep Learning for Novel Taxon Discovery in Marine eDNA",
                        author="Alisha Mathias, Deep Naidu, Krish Kansara, Shreyas Mekap",
                        subject="Real-world problem, literature survey and research gap")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("saved", OUT)
