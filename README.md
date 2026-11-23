# Causal Mechanistic Dissection of StripedHyena 2 (Evo 2 20B)

**Author:** Morgan Patterson (University of Missouri–Kansas City)  
**Contributor:** Gemini Neon (Google Antigravity)  
**Full Technical Report (PDF):** [`docs/report.pdf`](docs/report.pdf)  
**Companion Study:** [BrightMine35/evo2-mito-pathogenicity-atlas](https://github.com/BrightMine35/evo2-mito-pathogenicity-atlas)

A causal mechanistic dissection of all **24 hybrid blocks** (`7 HCS`, `7 HCM`, `7 HCL`, `3 Sparse MHA`) and **192 post-RoPE attention heads** inside **Evo 2 20B** using validated `mmaemu_all` FP8 emulation across 5× NVIDIA RTX 5060 Ti GPUs (80 GB VRAM total).

---

## Executive Summary of Findings

1. **Residual-Stream Probing Creates an Illusion; Pure Mixer Updates Expose Operator Specialization (Experiment 1)**
   - Probing full residual streams (`x_l`) creates an accumulation artifact where late attention blocks appear to dominate local TF motifs (`0.9509` AUROC vs. `HCS 0.9089`). Isolating pure mixer updates (`Δh_l`) inverts this ranking: **`HCS`** leads on local TF motifs (mean `0.9093`, peak `L0 = 0.9953` vs. `MHA 0.6724`), **`HCM`** peaks on canonical GT/AG splice sites (`L19 = 0.9983`, class mean `0.9557`) and exon/intron identity (`L22 = 0.9567`), and causal 1 kb upstream GC% scales monotonically with operator receptive field (`HCS -0.0293 < HCM +0.0388 < HCL +0.1357 < MHA +0.3619`, peaking at `L17 = 0.4029`).

2. **Causal Organelle, Splice & AlphaGenome Modality Dissociation (Experiments 2A–2B)**
   - Single-layer mean ablation across `16,569 bp` rCRS mtDNA reveals essential input/output trunk blocks (`L00–L02`, `L21`, `L23`), near-complete redundancy at `L22 (HCM)` (`ΔNLL = +0.0004`, ClinVar all `0.8318` vs. intact `0.8209`), and selective folded mt-tRNA stem vulnerability under attention blocks **`L10`** (`tRNA_stem ΔNLL = +0.3673`, `4.2×` `CDS +0.0876`, ClinVar tRNA `−0.1394`) and **`L17`** (`2.6×` stem/CDS ratio, ClinVar tRNA `−0.1145`).
   - Stratifying *BRCA1* Exons 15 and 18 by AlphaGenome LOF modality reveals double dissociation: `L15 (HCM)` collapses Exon 15 Splicing (`−0.4008`) with minimal effect on AlphaMissense (`−0.0591`), whereas `L14 (HCS)` collapses Exon 18 AlphaMissense (`−0.5812`) with virtually zero effect on Splicing (`−0.0002`).

3. **Sparse Attention Is Causally Required for 55–65 bp Compensatory RNA Epistasis (Experiment 2C)**
   - Equal-count 3-layer ablation (`Ablate_MHA [3,10,17]`) across 24 canonical Watson–Crick stems (`bpp ≥ 0.952`, `L = 768 bp`) abolishes medium-range (55–65 bp) acceptor coupling (`E_full +8.316 → −0.498` nats, paired Wilcoxon two-sided `p = 0.0156`, one-sided `p = 0.0078`; `E_site j +8.655 → +0.035` nats, two-sided `p = 0.0078`) while retaining `47.2%` of short-range (10–23 bp) coupling (`+2.505` nats). Equal-count convolution sets (`-HCL [2,9,16]`, `-HCM [1,8,15]`, `-HCS [0,7,14]`) include essential early trunk blocks `L2/L1/L0` and collapse baseline likelihood globally.

4. **Unsupervised RNA Base-Pair Contact Discovery in Deep Attention Head `L17_H5` (Experiment 3)**
   - Without structural supervision, post-RoPE attention head **`L17_H5`** recovers ViennaRNA `bpp ≥ 0.50` contacts across all `22` human mt-tRNAs at mean APC `AUROC = 0.7604` (`raw_dir = 0.7821`, `≥40 bp = 0.8547`, `AUPRC = 0.0706`), peaking on *MT-TI* (`0.9748`), *MT-TS2* (`0.9469`), *MT-TV* (`0.9270`), *MT-TD* (`0.8907`), *MT-TY* (`0.8903`), and *MT-TL1* (`0.8121` / `≥40 bp = 0.9085`).
   - Contact recovery scales to kilobase `12S`/`16S` rRNAs (`MT-RNR1`: `L10_H53 = 0.8337`, `L17_H5 = 0.8041` with `≥130 bp = 0.7757`; `MT-RNR2`: `L17_H7 = 0.8616`, `L17_H2 = 0.8552` with `≥130 bp = 0.7284`). Long-range (`≥40 bp`) mt-tRNA specialist hits (>0.70 AUROC) expand `4.3×` across MHA depth (`L3 = 42 → L10 = 64 → L17 = 180`).

---

## Repository Structure

| Path | Description |
|---|---|
| `docs/report.pdf` | Publication-style 8-page technical report generated from verified CSV data |
| `figures/fig1_operator_probing_division_of_labor.png` | 24-layer pure mixer vs. residual-stream linear probing across 6 tasks |
| `figures/fig2_causal_ablation_24layers.png` | 24-layer causal mean-ablation on rCRS mtDNA & *BRCA1* AlphaGenome modalities |
| `figures/fig3_rna_epistasis_operator_ablation.png` | Equal-count 3-layer operator ablation across 24 RNA stem epistasis pairs |
| `figures/fig4_mha_rna_contact_maps.png` | Unsupervised RNA base-pair contact recovery across all 192 MHA heads |
| `data/` | Processed CSV deliverables (`6` tables), `top_rna_contact_maps.npz`, `rna_stem_epistasis_pairs.csv`, and `rna_fold_structures.json` |
| `scripts/` | Reproducible preparation, probing, ablation, contact-map, figure, and report scripts |
| `SHA256SUMS.txt` | SHA-256 digests for all files in `data/` |
| `CONTRIBUTORS.md` & `LICENSE` | Authorship / contributor attribution and MIT License |

---

## Reproducing Figures, Report & Checksums

```bash
python3 scripts/plot_direction_c.py
python3 scripts/build_direction_c_report.py
shasum -a 256 -c SHA256SUMS.txt
```

---

## Acknowledgments & Citations

See [`CONTRIBUTORS.md`](CONTRIBUTORS.md) and [`docs/report.pdf`](docs/report.pdf) for complete attribution and verified primary literature references (Brixi et al., *Nature* 2026; Ding & Qin, [arXiv:2610.39415](https://arxiv.org/abs/2610.39415) 2026 / [mmaemu](https://github.com/astding/mmaemu); Findlay et al., *Nature* 2018; Avsec et al., *Nature* 2026; Lorenz et al., *Algorithms Mol Biol* 2011; Rao et al., *ICLR* 2021; Andrews et al., *Nature Genetics* 1999; Landrum et al., *Nucleic Acids Res* 2025).
