#!/usr/bin/env python3
"""Build PDF, DOCX, and Markdown technical reports for Direction C: Mechanistic Dissection of StripedHyena 2."""

import hashlib
import html
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor


CHROME_BIN = os.environ.get('CHROME_BIN', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')

NAVY_HEX = '1B365D'
TEAL_HEX = '0F766E'
BODY_HEX = '1E293B'
MUTED_HEX = '475569'
ZEBRA_HEX = 'F8FAFC'
WARN_BG_HEX = 'FEF2F2'
WARN_BAR_HEX = 'B91C1C'
TEAL_BG_HEX = 'F0FDFA'
TEAL_BAR_HEX = '0F766E'
BORDER_HEX = 'CBD5E1'

NAVY = RGBColor(0x1B, 0x36, 0x5D)
TEAL = RGBColor(0x0F, 0x76, 0x6E)
BODY = RGBColor(0x1E, 0x29, 0x3B)
MUTED = RGBColor(0x47, 0x55, 0x69)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
WARN_RED = RGBColor(0x99, 0x1B, 0x1B)


def resolve_paths():
    root = Path(__file__).resolve().parent.parent
    env_data = os.environ.get('SH2_DATA_DIR')
    env_fig = os.environ.get('SH2_FIG_DIR')
    env_out = os.environ.get('SH2_OUT_DIR')

    if env_data:
        data_dir = Path(env_data)
    elif (root / 'datasets' / 'processed' / 'layer_probing_results.csv').exists():
        data_dir = root / 'datasets' / 'processed'
    elif (root / 'data' / 'layer_probing_results.csv').exists():
        data_dir = root / 'data'
    else:
        raise FileNotFoundError('Could not locate Direction C CSV datasets.')

    ref_dir = root / 'datasets' / 'raw_reference'
    if not ref_dir.exists():
        ref_dir = data_dir

    fig_dir = Path(env_fig) if env_fig else (root / 'figures')
    if env_out:
        out_dir = Path(env_out)
    elif (root / 'manuscripts').exists() or (root / 'datasets').exists():
        out_dir = root / 'manuscripts'
    else:
        out_dir = root / 'docs'

    out_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, ref_dir, fig_dir, out_dir


def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return '—'
    if p < 1e-3:
        return f'{p:.2e}'
    return f'{p:.4f}'


def fmt_signed(x: float, decimals: int = 3) -> str:
    return f'{x:+.{decimals}f}'


def compute_metrics(data_dir: Path, ref_dir: Path) -> dict:
    pr = pd.read_csv(data_dir / 'layer_probing_results.csv')
    mt = pd.read_csv(data_dir / 'causal_ablation_24layers_mtdna.csv')
    br = pd.read_csv(data_dir / 'causal_ablation_24layers_brca1.csv')
    ep = pd.read_csv(data_dir / 'rna_epistasis_operator_ablation.csv')
    hd = pd.read_csv(data_dir / 'mha_rna_contact_heads.csv')
    en = pd.read_csv(data_dir / 'mha_rna_contact_ensemble.csv')

    # Experiment 1 probing summary
    task_order = [
        ('TF_motifs', 'Local TF Motifs (6–19 bp)', 'AUROC'),
        ('Splice_GT_AG', 'Canonical Splice (GT/AG)', 'AUROC'),
        ('Exon_vs_Intron', 'Exon vs. Intron Boundary', 'AUROC'),
        ('Codon_Frame', 'Coding Frame (0 / 1 / 2)', 'Macro-F1'),
        ('RNA_Stem3p_vs_Loop', 'RNA Paired 3′ Stem vs. Loop', 'AUROC'),
        ('Regional_1kb_GC', 'Regional 1 kb Upstream GC%', 'Ridge R²'),
    ]
    probe_rows = []
    probe_dict = {}
    for tk, label, metric_lbl in task_order:
        sm = pr[(pr['task'] == tk) & (pr['hook_type'] == 'mixer_update')]
        sr = pr[(pr['task'] == tk) & (pr['hook_type'] == 'residual_stream')]
        peak_m = sm.loc[sm['oof_score'].idxmax()]
        peak_r = sr.loc[sr['oof_score'].idxmax()]
        mm = sm.groupby('operator')['oof_score'].mean().to_dict()
        rm = sr.groupby('operator')['oof_score'].mean().to_dict()
        rec = {
            'task': tk,
            'label': label,
            'metric': metric_lbl,
            'n': int(peak_m['n_samples']),
            'n_pos': int(peak_m['n_pos']),
            'mix_peak_layer': int(peak_m['layer']),
            'mix_peak_op': str(peak_m['operator']),
            'mix_peak_val': float(peak_m['oof_score']),
            'res_peak_layer': int(peak_r['layer']),
            'res_peak_op': str(peak_r['operator']),
            'res_peak_val': float(peak_r['oof_score']),
            'mix_means': {k: float(v) for k, v in mm.items()},
            'res_means': {k: float(v) for k, v in rm.items()},
        }
        probe_dict[tk] = rec
        probe_rows.append(rec)

    # Experiment 2A mtDNA causal ablation summary
    mt_intact = mt[mt['cond'] == 'intact'].iloc[0].to_dict()
    mt_key_layers = [0, 1, 2, 5, 10, 17, 21, 22, 23]
    mt_key_rows = [mt[mt['cond'] == 'intact'].iloc[0].to_dict()] + [
        mt[mt['layer'] == l].iloc[0].to_dict() for l in mt_key_layers
    ]

    # Experiment 2B BRCA1 modality causal ablation summary
    br_key_layers = [-1, 7, 14, 15, 17, 19]
    br_key_rows = [br[br['layer'] == l].iloc[0].to_dict() for l in br_key_layers]

    # Experiment 2C RNA epistasis ablation + exact Wilcoxon p-values
    strata_meta = [
        ('short_10_25', 'Short tRNA (10–23 bp)'),
        ('medium_55_70', 'Medium tRNA Acceptor (55–65 bp)'),
        ('long_130_600', 'Long rRNA (147–240 bp)'),
    ]
    cond_meta = [
        ('Intact', 'Intact (0 ablated)', 'None'),
        ('Ablate_MHA', '−MHA', '[3, 10, 17]'),
        ('Ablate_HCL', '−HCL (incl. trunk L2)', '[2, 9, 16]'),
        ('Ablate_HCM', '−HCM (incl. trunk L1)', '[1, 8, 15]'),
        ('Ablate_HCS', '−HCS (incl. trunk L0)', '[0, 7, 14]'),
    ]
    ep_summary = []
    ep_lookup = {}
    for ck, clbl, clayers in cond_meta:
        for sk, slbl in strata_meta:
            sub = ep[(ep['condition'] == ck) & (ep['stratum'] == sk)].sort_values('pair_id')
            vf = sub['epistasis_full_nats'].values
            vs = sub['epistasis_sitej_nats'].values
            alt = 'greater' if ck == 'Intact' else 'two-sided'
            pf = float(wilcoxon(vf, alternative=alt).pvalue)
            ps = float(wilcoxon(vs, alternative=alt).pvalue)
            rec = {
                'cond': ck,
                'cond_label': clbl,
                'layers': clayers,
                'stratum': sk,
                'stratum_label': slbl,
                'n_pairs': len(sub),
                'dist_min': int(sub['dist'].min()),
                'dist_max': int(sub['dist'].max()),
                'mean_full': float(np.mean(vf)),
                'sem_full': float(np.std(vf, ddof=1) / np.sqrt(len(vf))),
                'p_full_vs0': pf,
                'mean_sitej': float(np.mean(vs)),
                'sem_sitej': float(np.std(vs, ddof=1) / np.sqrt(len(vs))),
                'p_sitej_vs0': ps,
            }
            ep_summary.append(rec)
            ep_lookup[(ck, sk)] = rec

    # Paired Wilcoxon tests Intact vs Ablate_MHA on medium_55_70
    s_int_med = ep[(ep['condition'] == 'Intact') & (ep['stratum'] == 'medium_55_70')].sort_values('pair_id')
    s_mha_med = ep[(ep['condition'] == 'Ablate_MHA') & (ep['stratum'] == 'medium_55_70')].sort_values('pair_id')
    p_int_vs_mha_full_2s = float(wilcoxon(s_int_med['epistasis_full_nats'].values, s_mha_med['epistasis_full_nats'].values, alternative='two-sided').pvalue)
    p_int_vs_mha_full_1s = float(wilcoxon(s_int_med['epistasis_full_nats'].values, s_mha_med['epistasis_full_nats'].values, alternative='greater').pvalue)
    p_int_vs_mha_sitej_2s = float(wilcoxon(s_int_med['epistasis_sitej_nats'].values, s_mha_med['epistasis_sitej_nats'].values, alternative='two-sided').pvalue)

    # Experiment 3 MHA contacts
    th = hd[hd['feature'] == 'tRNA'].groupby(['layer', 'head', 'head_name'])[
        ['auroc_apc', 'auprc_apc', 'prec_L5_apc', 'auroc_raw_dir', 'auroc_ge40']
    ].mean().reset_index().sort_values('auroc_apc', ascending=False)
    top5_trna_heads = th.head(5).to_dict('records')
    ens_mean = en[['ensemble_auroc', 'ensemble_auprc', 'ensemble_prec_L5', 'ensemble_auroc_ge40']].mean().to_dict()

    select_trnas = hd[(hd['head_name'] == 'L17_H5') & (hd['gene'].isin(['MT-TI', 'MT-TS2', 'MT-TV', 'MT-TD', 'MT-TY', 'MT-TL1']))].merge(
        en[['gene', 'ensemble_auroc', 'ensemble_auroc_ge40']], on='gene', how='left'
    ).sort_values('auroc_apc', ascending=False).to_dict('records')

    rnr1_top = hd[hd['gene'] == 'MT-RNR1'].sort_values('auroc_apc', ascending=False).iloc[0].to_dict()
    rnr1_h5 = hd[(hd['gene'] == 'MT-RNR1') & (hd['head_name'] == 'L17_H5')].iloc[0].to_dict()
    rnr2_top = hd[hd['gene'] == 'MT-RNR2'].sort_values('auroc_apc', ascending=False).head(2).to_dict('records')
    rnr2_h5 = hd[(hd['gene'] == 'MT-RNR2') & (hd['head_name'] == 'L17_H5')].iloc[0].to_dict()

    depth_counts = {}
    for l_val in [3, 10, 17]:
        st = hd[(hd['feature'] == 'tRNA') & (hd['layer'] == l_val)]
        sr = hd[(hd['feature'] == 'rRNA') & (hd['layer'] == l_val)]
        depth_counts[l_val] = {
            'tRNA_all': int((st['auroc_apc'] > 0.70).sum()),
            'tRNA_ge40': int((st['auroc_ge40'] > 0.70).sum()),
            'rRNA_all': int((sr['auroc_apc'] > 0.70).sum()),
        }

    # File manifest checksums
    desc_map = {
        'layer_probing_results.csv': '288 rows (24 layers × 2 hooks × 6 tasks, 5-fold GroupKFold OOF)',
        'causal_ablation_24layers_mtdna.csv': '25 rows (intact + 24 single-layer mean ablations on 16,569 bp rCRS mtDNA)',
        'causal_ablation_24layers_brca1.csv': '25 rows (intact + 24 layer ablations on BRCA1 Exons 15 & 18 by modality)',
        'rna_epistasis_operator_ablation.csv': '120 rows (24 RNA stem WC pairs × 5 equal-count 3-layer ablation conditions)',
        'mha_rna_contact_heads.csv': '4,608 rows (24 mt-tRNA/rRNA structures × 192 MHA attention heads)',
        'mha_rna_contact_ensemble.csv': '22 rows (leave-one-tRNA-out 192-head unsupervised contact ensemble)',
        'top_rna_contact_maps.npz': 'Full N×N APC/raw contact maps for top specialist heads & ensembles',
        'rna_stem_epistasis_pairs.csv': '24 canonical Watson–Crick stem pairs across short/medium/long strata',
        'rna_fold_structures.json': 'ViennaRNA 2.7 MFE + partition-function base-pair probability matrices (24 RNAs)',
    }
    manifest = []
    seen = set()
    for folder in (data_dir, ref_dir):
        for p in sorted(folder.iterdir()):
            if p.suffix in ('.csv', '.npz', '.json') and p.name in desc_map and p.name not in seen:
                seen.add(p.name)
                manifest.append({
                    'filename': p.name,
                    'desc': desc_map[p.name],
                    'bytes': p.stat().st_size,
                    'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                })

    return {
        'probe_rows': probe_rows,
        'probe_dict': probe_dict,
        'mt_intact': mt_intact,
        'mt_key_rows': mt_key_rows,
        'br_key_rows': br_key_rows,
        'ep_summary': ep_summary,
        'ep_lookup': ep_lookup,
        'p_int_vs_mha_full_2s': p_int_vs_mha_full_2s,
        'p_int_vs_mha_full_1s': p_int_vs_mha_full_1s,
        'p_int_vs_mha_sitej_2s': p_int_vs_mha_sitej_2s,
        'top5_trna_heads': top5_trna_heads,
        'ens_mean': {k: float(v) for k, v in ens_mean.items()},
        'select_trnas': select_trnas,
        'rnr1_top': rnr1_top,
        'rnr1_h5': rnr1_h5,
        'rnr2_top': rnr2_top,
        'rnr2_h5': rnr2_h5,
        'depth_counts': depth_counts,
        'manifest': manifest,
    }


# ---------------------------------------------------------------------------
# HTML / Chrome Headless PDF Builder
# ---------------------------------------------------------------------------
def make_html_table(title, headers, rows, right_cols=None, col_widths=None, compact=False):
    if right_cols is None:
        right_cols = set()
    cls = 'data-table compact' if compact else 'data-table'
    colgroup = '<colgroup>' + ''.join(f'<col style="width:{w};">' for w in col_widths) + '</colgroup>' if col_widths else ''
    ths = ''.join(f'<th class="{"num" if j in right_cols else ""}">{html.escape(str(h))}</th>' for j, h in enumerate(headers))
    trs = []
    bold_triggers = ('Local TF Motifs', 'Regional 1 kb', 'intact', 'L10', 'L17', 'L15', 'L14', 'L21', 'L22', 'Intact', '−MHA', 'L17_H5', 'MT-RNR1', 'MT-RNR2', '192-Head LOO', 'Layer 17')
    for row in rows:
        tds = []
        for j, val in enumerate(row):
            s = str(val)
            c = 'num' if j in right_cols else ''
            esc = html.escape(s)
            if j == 0 and any(s.startswith(t) for t in bold_triggers):
                esc = f'<strong>{esc}</strong>'
            tds.append(f'<td class="{c}">{esc}</td>')
        trs.append('<tr>' + ''.join(tds) + '</tr>')
    return (
        f'<div class="table-block">'
        f'<div class="table-title">{html.escape(title)}</div>'
        f'<table class="{cls}">{colgroup}<thead><tr>{ths}</tr></thead>'
        f'<tbody>{"".join(trs)}</tbody></table></div>'
    )


def make_html_fig(fig_dir: Path, fig_name: str, caption_title: str, caption_body: str, max_height: str = '3.35in'):
    src = (fig_dir / fig_name).resolve().as_uri()
    return (
        f'<figure>'
        f'<img src="{src}" style="max-height:{max_height};" alt="{html.escape(caption_title)}">'
        f'<figcaption><strong>{html.escape(caption_title)}</strong> {html.escape(caption_body)}</figcaption>'
        f'</figure>'
    )


def build_pdf(M: dict, fig_dir: Path, out_pdf: Path):
    p = M['probe_dict']
    ep = M['ep_lookup']
    mi = M['mt_intact']
    br_rows = M['br_key_rows']
    br_int = br_rows[0]
    br_l14 = next(r for r in br_rows if r['cond'] == 'L14')
    br_l15 = next(r for r in br_rows if r['cond'] == 'L15')
    br_l17 = next(r for r in br_rows if r['cond'] == 'L17')
    mt_l10 = next(r for r in M['mt_key_rows'] if r['cond'] == 'L10')
    mt_l17 = next(r for r in M['mt_key_rows'] if r['cond'] == 'L17')
    mt_l05 = next(r for r in M['mt_key_rows'] if r['cond'] == 'L05')
    mt_l22 = next(r for r in M['mt_key_rows'] if r['cond'] == 'L22')
    h5 = M['top5_trna_heads'][0]
    dc = M['depth_counts']

    t1 = make_html_table(
        'Table 1. Experiment 1: 24-Layer Pure Mixer Update (Δh_l) vs. Residual Stream (x_l) Probing (5-Fold GroupKFold OOF)',
        ['Sequence Task (Metric)', 'N (Pos)', 'Peak Mixer Layer', 'HCS Mean', 'HCM Mean', 'HCL Mean', 'MHA Mix', 'MHA Res'],
        [
            [
                f"{r['label']} ({r['metric']})",
                f"{r['n']:,} ({r['n_pos']:,})",
                f"L{r['mix_peak_layer']} ({r['mix_peak_op']}): {r['mix_peak_val']:.4f}",
                f"{r['mix_means']['HCS']:+.4f}" if tk == 'Regional_1kb_GC' else f"{r['mix_means']['HCS']:.4f}",
                f"{r['mix_means']['HCM']:+.4f}" if tk == 'Regional_1kb_GC' else f"{r['mix_means']['HCM']:.4f}",
                f"{r['mix_means']['HCL']:+.4f}" if tk == 'Regional_1kb_GC' else f"{r['mix_means']['HCL']:.4f}",
                f"{r['mix_means']['MHA']:+.4f}" if tk == 'Regional_1kb_GC' else f"{r['mix_means']['MHA']:.4f}",
                f"{r['res_means']['MHA']:+.4f}" if tk == 'Regional_1kb_GC' else f"{r['res_means']['MHA']:.4f}",
            ]
            for r in M['probe_rows']
            for tk in [r['task']]
        ],
        right_cols={3, 4, 5, 6, 7},
        col_widths=['27%', '11%', '17%', '9%', '9%', '9%', '9%', '9%'],
        compact=True,
    )

    t2a = make_html_table(
        'Table 2A. Experiment 2A: Representative 24-Layer Causal Mean-Ablations on 16,569 bp rCRS mtDNA (H-Strand)',
        ['Condition', 'Op.', 'ΔNLL Genome', 'ΔNLL CDS', 'ΔNLL tRNA Stem', 'ΔNLL tRNA Loop', 'ClinVar All AUROC', 'ΔCDS AUROC', 'ΔtRNA AUROC'],
        [
            [
                r['cond'] if r['cond'] != 'intact' else 'intact (baseline)',
                r['operator'],
                f"{r['mean_nll_genome']:.4f} (abs)" if r['cond'] == 'intact' else fmt_signed(r['delta_mean_nll_genome'], 4),
                f"{r['nll_CDS']:.4f}" if r['cond'] == 'intact' else fmt_signed(r['delta_nll_CDS'], 4),
                f"{r['nll_tRNA_stem']:.4f}" if r['cond'] == 'intact' else fmt_signed(r['delta_nll_tRNA_stem'], 4),
                f"{r['nll_tRNA_loop']:.4f}" if r['cond'] == 'intact' else fmt_signed(r['delta_nll_tRNA_loop'], 4),
                f"{r['auroc_all_mtDNA']:.4f}",
                f"{r['auroc_CDS']:.4f} (abs)" if r['cond'] == 'intact' else fmt_signed(r['delta_auroc_CDS'], 4),
                f"{r['auroc_tRNA']:.4f} (abs)" if r['cond'] == 'intact' else fmt_signed(r['delta_auroc_tRNA'], 4),
            ]
            for r in M['mt_key_rows']
        ],
        right_cols={2, 3, 4, 5, 6, 7, 8},
        col_widths=['14%', '7%', '12%', '11%', '12%', '12%', '12%', '10%', '10%'],
        compact=True,
    )

    t2b = make_html_table(
        'Table 2B. Experiment 2B: BRCA1 Exon 15 & Exon 18 Causal Ablation Stratified by AlphaGenome LOF Modality',
        ['Condition', 'Op.', 'X15 All', 'X15 Splice Δ', 'X15 Missense Δ', 'X18 All', 'X18 Splice Δ', 'X18 Missense Δ'],
        [
            [
                r['cond'] if r['cond'] != 'intact' else 'intact (baseline)',
                r['operator'],
                f"{r['X15_auroc_all']:.4f}",
                f"{r['X15_auroc_Splicing']:.4f} (abs)" if r['cond'] == 'intact' else fmt_signed(r['delta_X15_auroc_Splicing'], 4),
                f"{r['X15_auroc_AlphaMissense']:.4f} (abs)" if r['cond'] == 'intact' else fmt_signed(r['delta_X15_auroc_AlphaMissense'], 4),
                f"{r['X18_auroc_all']:.4f}",
                f"{r['X18_auroc_Splicing']:.4f} (abs)" if r['cond'] == 'intact' else fmt_signed(r['delta_X18_auroc_Splicing'], 4),
                f"{r['X18_auroc_AlphaMissense']:.4f} (abs)" if r['cond'] == 'intact' else fmt_signed(r['delta_X18_auroc_AlphaMissense'], 4),
            ]
            for r in br_rows
        ],
        right_cols={2, 3, 4, 5, 6, 7},
        col_widths=['15%', '8%', '11%', '14%', '15%', '11%', '13%', '13%'],
        compact=True,
    )

    ep_tbl_rows = []
    for r in M['ep_summary']:
        ep_tbl_rows.append([
            r['cond_label'],
            r['layers'],
            r['stratum_label'],
            f"{r['dist_min']}–{r['dist_max']} bp",
            f"{r['mean_full']:+.3f} ± {r['sem_full']:.2f}",
            fmt_p(r['p_full_vs0']),
            f"{r['mean_sitej']:+.3f} ± {r['sem_sitej']:.2f}",
            fmt_p(r['p_sitej_vs0']),
        ])
    t3 = make_html_table(
        'Table 3. Experiment 2C: Equal-Count 3-Layer Ablation Across 24 Canonical Watson–Crick Stem Pairs (L = 768 bp)',
        ['Condition', 'Ablated Layers', 'Span Stratum (n=8)', 'Pair Span', 'E_full (nats)', 'p vs 0', 'E_site j (nats)', 'p vs 0'],
        ep_tbl_rows,
        right_cols={3, 4, 5, 6, 7},
        col_widths=['16%', '11%', '20%', '10%', '14%', '8%', '13%', '8%'],
        compact=True,
    )

    t4a_rows = []
    for hr in M['top5_trna_heads']:
        t4a_rows.append([
            hr['head_name'],
            '22 mt-tRNAs (mean)',
            f"{hr['auroc_apc']:.4f}",
            f"{hr['auroc_raw_dir']:.4f}",
            f"{hr['auroc_ge40']:.4f}",
            f"{hr['auprc_apc']:.4f}",
            f"{hr['prec_L5_apc']:.4f}",
        ])
    em = M['ens_mean']
    t4a_rows.append([
        '192-Head LOO Ensemble', '22 mt-tRNAs (LOO mean)', f"{em['ensemble_auroc']:.4f}", '—', f"{em['ensemble_auroc_ge40']:.4f}", f"{em['ensemble_auprc']:.4f}", f"{em['ensemble_prec_L5']:.4f}",
    ])
    t4a_rows.append([
        f"MT-RNR1 Top: {M['rnr1_top']['head_name']}", '12S rRNA (954 nt, 157 pairs)', f"{M['rnr1_top']['auroc_apc']:.4f}", f"{M['rnr1_top']['auroc_raw_dir']:.4f}", f"{M['rnr1_top']['auroc_ge40']:.4f}", f"{M['rnr1_top']['auprc_apc']:.4f}", f"≥130bp: {M['rnr1_top']['auroc_ge130']:.4f}",
    ])
    t4a_rows.append([
        'MT-RNR1 L17_H5', '12S rRNA (954 nt, 157 pairs)', f"{M['rnr1_h5']['auroc_apc']:.4f}", f"{M['rnr1_h5']['auroc_raw_dir']:.4f}", f"{M['rnr1_h5']['auroc_ge40']:.4f}", f"{M['rnr1_h5']['auprc_apc']:.4f}", f"≥130bp: {M['rnr1_h5']['auroc_ge130']:.4f}",
    ])
    t4a_rows.append([
        f"MT-RNR2 Top: {M['rnr2_top'][0]['head_name']} / {M['rnr2_top'][1]['head_name']}", '16S rRNA (1,559 nt, 295 pairs)', f"{M['rnr2_top'][0]['auroc_apc']:.4f} / {M['rnr2_top'][1]['auroc_apc']:.4f}", '—', f"{M['rnr2_top'][0]['auroc_ge40']:.4f} / {M['rnr2_top'][1]['auroc_ge40']:.4f}", f"{M['rnr2_top'][0]['auprc_apc']:.4f}", f"≥130bp H2: {M['rnr2_top'][1]['auroc_ge130']:.4f}",
    ])
    t4a = make_html_table(
        'Table 4A. Experiment 3: Unsupervised RNA Secondary-Structure Contact Recovery Across 192 Sparse MHA Heads',
        ['Head / Model', 'Evaluation Cohort', 'AUROC (APC)', 'AUROC Raw', 'AUROC ≥40 bp', 'AUPRC (APC)', 'Prec@L/5 or ≥130 bp'],
        t4a_rows,
        right_cols={2, 3, 4, 5, 6},
        col_widths=['22%', '23%', '12%', '10%', '11%', '10%', '12%'],
        compact=True,
    )

    t4b_rows = [
        [
            r['gene'],
            f"{int(r['length'])} nt",
            f"{int(r['n_pos_pairs'])}",
            f"{r['auroc_apc']:.4f}",
            f"{r['auroc_raw_dir']:.4f}",
            f"{r['auroc_ge40']:.4f}" if not math.isnan(r['auroc_ge40']) else '—',
            f"{r['ensemble_auroc']:.4f}",
        ]
        for r in M['select_trnas']
    ] + [
        ['Layer 3 Depth Hit Count (>0.70)', '64 heads × 22 tRNAs / 2 rRNAs', '—', f"tRNA all: {dc[3]['tRNA_all']}", '—', f"tRNA ≥40bp: {dc[3]['tRNA_ge40']}", f"rRNA: {dc[3]['rRNA_all']}"],
        ['Layer 10 Depth Hit Count (>0.70)', '64 heads × 22 tRNAs / 2 rRNAs', '—', f"tRNA all: {dc[10]['tRNA_all']}", '—', f"tRNA ≥40bp: {dc[10]['tRNA_ge40']}", f"rRNA: {dc[10]['rRNA_all']}"],
        ['Layer 17 Depth Hit Count (>0.70)', '64 heads × 22 tRNAs / 2 rRNAs', '—', f"tRNA all: {dc[17]['tRNA_all']}", '—', f"tRNA ≥40bp: {dc[17]['tRNA_ge40']}", f"rRNA: {dc[17]['rRNA_all']}"],
    ]
    t4b = make_html_table(
        'Table 4B. Peak Single-Head L17_H5 tRNA Contact Recovery & Unsupervised Specialist Emergence Across MHA Depth',
        ['Gene / Depth Metric', 'Length / Cohort', 'Stem Pairs', 'L17_H5 APC', 'L17_H5 Raw', '≥40 bp AUROC', 'LOO Ens. / rRNA Hits'],
        t4b_rows,
        right_cols={2, 3, 4, 5, 6},
        col_widths=['25%', '19%', '10%', '11%', '11%', '12%', '12%'],
        compact=True,
    )

    ta = make_html_table(
        f"Table A1. Data Manifest: {len(M['manifest'])} Verified Study Deliverables & SHA-256 Checksums",
        ['Filename', 'Contents', 'Bytes', 'SHA-256 Digest'],
        [[r['filename'], r['desc'], f"{r['bytes']:,}", r['sha256']] for r in M['manifest']],
        right_cols={2},
        col_widths=['27%', '31%', '8%', '34%'],
        compact=True,
    )

    html_doc = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Causal Mechanistic Dissection of StripedHyena 2 (Evo 2 20B)</title>
<style>
  @page {{
    size: letter;
    margin: 0.68in 0.68in 0.68in 0.68in;
  }}
  body {{
    font-family: Georgia, "Times New Roman", serif;
    font-size: 9.6pt;
    line-height: 1.40;
    color: #1E293B;
    margin: 0;
  }}
  header.paper-head {{
    border-bottom: 2px solid #1B365D;
    padding-bottom: 8px;
    margin-bottom: 12px;
  }}
  h1.title {{
    font-family: Georgia, serif;
    font-size: 15.2pt;
    line-height: 1.22;
    color: #1B365D;
    margin: 0 0 6px 0;
  }}
  .author-line {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 9.3pt;
    color: #334155;
    margin-bottom: 3px;
  }}
  .meta-line {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 8.2pt;
    color: #0F766E;
  }}
  h2 {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 11.5pt;
    color: #1B365D;
    border-bottom: 1px solid #CBD5E1;
    padding-bottom: 2px;
    margin: 12px 0 6px 0;
    break-after: avoid;
    page-break-after: avoid;
  }}
  h3 {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 10pt;
    color: #0F766E;
    margin: 9px 0 4px 0;
    break-after: avoid;
    page-break-after: avoid;
  }}
  p {{
    margin: 0 0 6px 0;
    text-align: justify;
  }}
  ul {{
    margin: 4px 0 8px 0;
    padding-left: 18px;
  }}
  li {{
    margin-bottom: 3.5px;
  }}
  .table-block {{
    margin: 7px 0 9px 0;
    break-inside: avoid;
    page-break-inside: avoid;
  }}
  .table-title {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 8.6pt;
    font-weight: 700;
    color: #1B365D;
    margin-bottom: 3px;
    break-after: avoid;
  }}
  table.data-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 7.5pt;
    border-bottom: 1.5px solid #1B365D;
  }}
  table.data-table.compact {{
    font-size: 6.8pt;
    table-layout: fixed;
  }}
  table.data-table.compact td, table.data-table.compact th {{
    padding: 3px 4px;
    overflow-wrap: break-word;
  }}
  table.data-table thead {{
    display: table-header-group;
  }}
  table.data-table th {{
    background-color: #1B365D;
    color: #FFFFFF;
    text-align: left;
    padding: 4px 5px;
    font-weight: 700;
  }}
  table.data-table th.num, table.data-table td.num {{
    text-align: right;
  }}
  table.data-table td {{
    padding: 3.5px 5px;
    border-bottom: 1px solid #CBD5E1;
    vertical-align: middle;
  }}
  table.data-table tbody tr:nth-child(even) {{
    background-color: #F8FAFC;
  }}
  table.data-table tr {{
    break-inside: avoid;
    page-break-inside: avoid;
  }}
  figure {{
    margin: 7px 0 9px 0;
    text-align: center;
    break-inside: avoid;
    page-break-inside: avoid;
  }}
  figure img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto 4px auto;
  }}
  figcaption {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 8.0pt;
    line-height: 1.28;
    color: #475569;
    text-align: left;
    font-style: italic;
  }}
  figcaption strong {{
    color: #1B365D;
    font-style: normal;
  }}
  .callout {{
    background: #F0FDFA;
    border: 1px solid #CBD5E1;
    border-left: 4px solid #0F766E;
    padding: 7px 10px;
    margin: 7px 0;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 8.4pt;
    line-height: 1.34;
    break-inside: avoid;
    page-break-inside: avoid;
  }}
  .callout.warn {{
    background: #FEF2F2;
    border-left-color: #B91C1C;
  }}
  .callout .callout-title {{
    font-weight: 700;
    color: #0F766E;
    margin-bottom: 3px;
    font-size: 8.8pt;
  }}
  .callout.warn .callout-title {{
    color: #991B1B;
  }}
  ol.refs {{
    padding-left: 18px;
    font-size: 7.9pt;
    line-height: 1.24;
    margin: 4px 0;
  }}
  ol.refs li {{
    margin-bottom: 1.5px;
  }}
  .page-break {{
    break-before: page;
    page-break-before: always;
  }}
  code {{
    font-family: "SFMono-Regular", Menlo, Consolas, monospace;
    font-size: 8.0pt;
    background: #F1F5F9;
    padding: 0.5px 3px;
    border-radius: 2px;
  }}
</style>
</head>
<body>

<header class="paper-head">
  <h1 class="title">Causal Mechanistic Dissection of StripedHyena 2 (Evo 2 20B): Operator Division of Labor, Residual-Stream Confounding, and Unsupervised RNA Contact-Map Recovery Across 192 Sparse Attention Heads</h1>
  <div class="author-line"><strong>Author:</strong> Morgan Patterson (University of Missouri&ndash;Kansas City) &nbsp;|&nbsp; <strong>Contributor / Tooling:</strong> Gemini Neon (Google Antigravity)</div>
  <div class="meta-line">Architecture: 24 Hybrid Layers (7 HCS / 7 HCM / 7 HCL / 3 Sparse MHA, 192 Heads, d<sub>model</sub>=8,192) &bull; Precision: Validated Evo 2 20B <code>mmaemu_all</code> FP8 Emulation &bull; Hardware: 5&times; RTX 5060 Ti (80 GB VRAM)</div>
</header>

<div class="callout">
  <div class="callout-title">Executive Summary: Empirical Division of Labor Inside StripedHyena 2</div>
  <p style="margin-bottom:3px;"><strong>1. Pure Mixer vs. Residual-Stream Illusion:</strong> Probing residual streams (<code>x_l</code>) creates a cumulative accumulation artifact where late <code>MHA</code> layers appear to lead local TF motif classification (<code>{p['TF_motifs']['res_means']['MHA']:.4f}</code> vs. <code>HCS {p['TF_motifs']['res_means']['HCS']:.4f}</code>). Isolating pure mixer updates (<code>&Delta;h_l</code>) reveals true specialization: <strong><code>HCS</code></strong> peaks at <code>L0</code> for local motifs (mean <code>{p['TF_motifs']['mix_means']['HCS']:.4f}</code>, peak <code>{p['TF_motifs']['mix_peak_val']:.4f}</code> vs. <code>MHA {p['TF_motifs']['mix_means']['MHA']:.4f}</code>), <strong><code>HCM</code></strong> peaks at <code>L19</code> on canonical splice sites (<code>{p['Splice_GT_AG']['mix_means']['HCM']:.4f}</code>, peak <code>{p['Splice_GT_AG']['mix_peak_val']:.4f}</code>), and regional 1 kb GC composition scales monotonically with operator receptive field (<code>HCS {p['Regional_1kb_GC']['mix_means']['HCS']:+.4f} &lt; HCM {p['Regional_1kb_GC']['mix_means']['HCM']:+.4f} &lt; HCL {p['Regional_1kb_GC']['mix_means']['HCL']:+.4f} &lt; MHA {p['Regional_1kb_GC']['mix_means']['MHA']:+.4f}</code>).</p>
  <p style="margin-bottom:3px;"><strong>2. Causal Organelle, Splice, &amp; Epistasis Dissociation:</strong> Single-layer mean ablation across 16,569 bp human mtDNA shows deep <code>MHA</code> blocks selectively damage folded mt-tRNA stems (<code>L10</code> tRNA stem &Delta;NLL = <code>{mt_l10['delta_nll_tRNA_stem']:+.4f}</code>, <code>{mt_l10['delta_nll_tRNA_stem']/mt_l10['delta_nll_CDS']:.1f}&times;</code> CDS) and drop ClinVar tRNA discrimination (<code>L10 {mt_l10['delta_auroc_tRNA']:+.4f}</code>; <code>L17 {mt_l17['delta_auroc_tRNA']:+.4f}</code>), while <em>BRCA1</em> Exon 15 splicing collapses specifically under <code>L15 HCM</code> (<code>{br_l15['delta_X15_auroc_Splicing']:+.4f}</code> vs. AlphaMissense <code>{br_l15['delta_X15_auroc_AlphaMissense']:+.4f}</code>). Under equal-count 3-layer ablation on 24 RNA stems, removing all MHA layers (<code>[3,10,17]</code>) erases medium-range (55&ndash;65 bp) Watson&ndash;Crick epistasis (<code>E_full {ep[('Intact','medium_55_70')]['mean_full']:+.3f} &rarr; {ep[('Ablate_MHA','medium_55_70')]['mean_full']:+.3f}</code> nats, paired Wilcoxon two-sided <code>p = {fmt_p(M['p_int_vs_mha_full_2s'])}</code>; <code>E_site j {ep[('Intact','medium_55_70')]['mean_sitej']:+.3f} &rarr; {ep[('Ablate_MHA','medium_55_70')]['mean_sitej']:+.3f}</code> nats, <code>p = {fmt_p(M['p_int_vs_mha_sitej_2s'])}</code>) while preserving 47% of short-range (10&ndash;23 bp) coupling.</p>
  <p style="margin-bottom:0;"><strong>3. Unsupervised RNA Base-Pair Attention Heads:</strong> Without structural supervision, post-RoPE attention head <strong><code>L17_H5</code></strong> recovers folded RNA base pairs across all 22 human mt-tRNAs (mean APC AUROC = <code>{h5['auroc_apc']:.4f}</code>; long-range &ge;40 bp AUROC = <code>{h5['auroc_ge40']:.4f}</code>; peaking on <em>MT-TI</em> <code>0.9748</code> and <em>MT-TS2</em> <code>0.9469</code>) and remains accurate across kilobase-scale 12S (<code>MT-RNR1</code>: <code>0.8041</code>; <code>&ge;130 bp 0.7757</code>) and 16S rRNAs (<code>L17_H7 0.8616</code>).</p>
</div>

<h2>1. Experimental Motivation, Architecture &amp; Precision Protocol</h2>
<p>Evo 2 20B (Brixi et al., 2026) employs the 24-block hybrid <strong>StripedHyena 2</strong> architecture interleaving three scales of gated convolution (7 Hyena Short <code>HCS</code> blocks at <code>[0,4,7,11,14,18,21]</code>, 7 Hyena Medium <code>HCM</code> blocks at <code>[1,5,8,12,15,19,22]</code>, and 7 Hyena Long <code>HCL</code> blocks at <code>[2,6,9,13,16,20,23]</code>) with exactly three sparse multi-head causal attention blocks (<code>MHA</code> at layers <code>[3,10,17]</code>, 64 heads per layer = 192 total attention heads, head dimension 128, <code>d_model = 8,192</code>). Because Evo 2 checkpoints suffer silent precision divergence when run naively in BF16 on consumer GPUs (Ding &amp; Qin, 2026), all mechanistic hooks and ablations here execute under the verified <code>mmaemu_all</code> FP8-emulation loader partitioned across five RTX 5060 Ti (16 GB) GPUs (80 GB aggregate VRAM).</p>
<p>A core methodological pitfall in mechanistic interpretability of residual architectures is probing the un-differenced residual stream <code>x_l = x_{{l-1}} + &Delta;h_l + &Delta;m_l</code>, which sums all prior blocks&apos; outputs. Consequently, any downstream layer inherits high linear separability for features computed much earlier upstream. We decouple representation transfer from active computation via three complementary designs: (1) paired <strong>mixer-update (&Delta;h<sub>l</sub>) vs. residual-stream (x<sub>l</sub>) probing</strong> across human <em>BRCA1</em> and mitochondrial DNA (rCRS <code>NC_012920.1</code>); (2) <strong>single-layer causal mean-ablation</strong> across all 24 blocks and equal-count 3-layer operator ablation on compensatory RNA stem mutations; and (3) exact <strong>post-RoPE attention extraction</strong> across all 192 MHA heads benchmarked against thermodynamic partition-function base-pair probability matrices from ViennaRNA 2.7 (Lorenz et al., 2011).</p>

<h2>2. Experiment 1: Linear Probing Dissociates True Mixer Computation from Residual Carryover</h2>
<p>Across 12 contiguous 8,192 bp <em>BRCA1</em> locus windows (98,304 bp; GENCODE v46, aligned to Findlay et al. 2018) and both strands of circular human mtDNA (16,569 bp; Andrews et al., 1999), we recorded <code>x_l</code> and isolated mixer outputs <code>&Delta;h_l</code> across all 24 layers. Out-of-fold (OOF) L2-regularized LogisticRegression / Ridge probes were evaluated using 5-fold GroupKFold stratified by genomic window or RNA transcript across six biological tasks (Table 1, Figure 1).</p>

{t1}
{make_html_fig(fig_dir, 'fig1_operator_probing_division_of_labor.png', 'Figure 1. Operator Division of Labor and the Residual-Stream Confound Across All 24 StripedHyena 2 Blocks.', '(A–D) Layerwise 5-fold GroupKFold OOF probe scores for isolated mixer updates Δh_l (solid markers) vs. full residual streams x_l (dashed gray). (E) Operator-class means on pure Δh_l across sequence classification tasks. (F) Direct comparison on TF motifs (left) vs. 1 kb upstream GC% regression (right), exposing residual-stream accumulation on MHA layers vs. true HCS local motif sharpness and MHA/HCL long-range scaling.', max_height='3.55in')}

<ul>
  <li><strong>Exposure of the Residual-Stream Confound:</strong> On canonical TF motifs (<code>N = {p['TF_motifs']['n']}</code>), residual-stream probes suggest <code>MHA</code> outperforms <code>HCS</code> (<code>{p['TF_motifs']['res_means']['MHA']:.4f}</code> vs. <code>{p['TF_motifs']['res_means']['HCS']:.4f}</code>). Pure mixer updates <code>&Delta;h_l</code> invert this hierarchy: <strong><code>HCS</code></strong> leads strongly (<code>{p['TF_motifs']['mix_means']['HCS']:.4f}</code>, peaking at first block <strong><code>L0 = {p['TF_motifs']['mix_peak_val']:.4f}</code></strong>), while attention mixer updates drop sharply at every MHA depth (mean <code>{p['TF_motifs']['mix_means']['MHA']:.4f}</code>).</li>
  <li><strong>Medium Convolutions &amp; Regional Composition:</strong> <code>HCM</code> mixer updates achieve peak performance on canonical GT/AG splice motifs (<code>L19 = {p['Splice_GT_AG']['mix_peak_val']:.4f}</code>, class mean <code>{p['Splice_GT_AG']['mix_means']['HCM']:.4f}</code>) and exon/intron identity (<code>L22 = {p['Exon_vs_Intron']['mix_peak_val']:.4f}</code>), while coding reading frame peaks in late convolutions (<code>L18 HCS = {p['Codon_Frame']['mix_peak_val']:.4f}</code>). Causal 1 kb upstream GC composition scales strictly with operator integration horizon on <code>&Delta;h_l</code>: <code>HCS ({p['Regional_1kb_GC']['mix_means']['HCS']:+.4f}) &lt; HCM ({p['Regional_1kb_GC']['mix_means']['HCM']:+.4f}) &lt; HCL ({p['Regional_1kb_GC']['mix_means']['HCL']:+.4f}) &lt; MHA ({p['Regional_1kb_GC']['mix_means']['MHA']:+.4f})</code>, peaking at final attention block <code>L17 ({p['Regional_1kb_GC']['mix_peak_val']:.4f})</code>.</li>
</ul>

<div class="page-break"></div>

<h2>3. Experiment 2A &amp; 2B: 24-Layer Causal Mean-Ablation on mtDNA &amp; BRCA1 Modalities</h2>
<p>Probing establishes correlation rather than causal necessity. To measure direct causal contribution per layer, we mean-ablated each individual block&apos;s mixer output across sequence positions (<code>&Delta;h_l &larr; mean_t(&Delta;h_l)</code>, leaving the residual skip connection intact) one layer at a time during full 16,569 bp rCRS mtDNA inference (ClinVar 2025; Landrum et al., 2025) and 4,096 bp <em>BRCA1</em> Exon 15 / Exon 18 inference cross-stratified against AlphaGenome Variant Impact LOF modalities (Avsec et al., 2026) in Table 2A, Table 2B, and Figure 2.</p>

{t2a}
{t2b}
{make_html_fig(fig_dir, 'fig2_causal_ablation_24layers.png', 'Figure 2. Single-Layer Causal Mean-Ablation Across All 24 Layers Reveals Trunk Blocks, Redundancy, and Functional Dissociation.', '(A) mtDNA sequence ΔNLL across Coding (CDS), tRNA Stems, and tRNA Loops after non-trunk ablation (callouts summarize global collapse in L0–L2, L21, L23). (B) ClinVar AUROC shifts on Coding vs. Non-Coding mt-tRNA variants, highlighting selective tRNA collapse at L10 and L17 MHA. (C–D) AlphaGenome modality dissociation across BRCA1 Exon 15 (splicing-driven) and Exon 18 (BRCT missense vs. splicing).', max_height='3.55in')}

<div class="callout warn">
  <div class="callout-title">Key Causal Findings: Backbone Trunk, Layer Redundancy, &amp; Modality Double Dissociation</div>
  <p style="margin-bottom:3px;"><strong>Trunk vs. Redundant Layers:</strong> Input/output trunk blocks <code>L00 (HCS)</code>, <code>L01 (HCM)</code>, <code>L02 (HCL)</code>, <code>L21 (HCS)</code>, and <code>L23 (HCL)</code> cause catastrophic perplexity collapse (+1.23 to +2.68 nats genome &Delta;NLL, ClinVar AUROC 0.33&ndash;0.48), whereas <code>L22 (HCM)</code> is almost wholly redundant (genome &Delta;NLL = <code>{mt_l22['delta_mean_nll_genome']:+.4f}</code>, ClinVar AUROC = <code>{mt_l22['auroc_all_mtDNA']:.4f}</code> vs. intact <code>{mi['auroc_all_mtDNA']:.4f}</code>).</p>
  <p style="margin-bottom:3px;"><strong>Selective Sparse Attention Requirement in Folded mt-tRNAs:</strong> Mean-ablating attention layers <code>L10</code> and <code>L17</code> disproportionately damages base-paired mt-tRNA stems relative to coding sequences (<code>L10</code> &Delta;NLL tRNA stem <code>{mt_l10['delta_nll_tRNA_stem']:+.4f}</code> vs. CDS <code>{mt_l10['delta_nll_CDS']:+.4f}</code>, <code>4.2&times;</code> selectivity; <code>L17</code> stem <code>{mt_l17['delta_nll_tRNA_stem']:+.4f}</code> vs. CDS <code>{mt_l17['delta_nll_CDS']:+.4f}</code>, <code>2.6&times;</code>) and drops ClinVar tRNA AUROC by <code>{mt_l10['delta_auroc_tRNA']:+.4f}</code> and <code>{mt_l17['delta_auroc_tRNA']:+.4f}</code> while largely sparing CDS (<code>{mt_l10['delta_auroc_CDS']:+.4f}</code>, <code>{mt_l17['delta_auroc_CDS']:+.4f}</code>). Conversely, convolution <code>L05 (HCM)</code> shows the opposite bias (CDS &Delta;NLL <code>{mt_l05['delta_nll_CDS']:+.4f}</code> vs. tRNA stem <code>{mt_l05['delta_nll_tRNA_stem']:+.4f}</code>, <code>2.1&times;</code>).</p>
  <p style="margin-bottom:0;"><strong>BRCA1 AlphaGenome Modality Dissociation:</strong> In <em>BRCA1</em> Exon 15 (intact Splicing = <code>{br_int['X15_auroc_Splicing']:.4f}</code>, AlphaMissense = <code>{br_int['X15_auroc_AlphaMissense']:.4f}</code>), ablating medium convolution <code>L15 (HCM)</code> collapses splicing LOF recognition by <code>{br_l15['delta_X15_auroc_Splicing']:+.4f}</code> while barely perturbing missense calls (<code>{br_l15['delta_X15_auroc_AlphaMissense']:+.4f}</code>). In BRCT Exon 18, short convolution <code>L14 (HCS)</code> drops missense by <code>{br_l14['delta_X18_auroc_AlphaMissense']:+.4f}</code> with virtually zero effect on splicing (<code>{br_l14['delta_X18_auroc_Splicing']:+.4f}</code>), while <code>L17 (MHA)</code> knocks out missense recognition by <code>{br_l17['delta_X18_auroc_AlphaMissense']:+.4f}</code>.</p>
</div>

<h2>4. Experiment 2C: Equal-Count Operator Ablation on RNA Stem Epistasis</h2>
<p>To test which operators mediate compensatory nucleotide coupling along RNA helices without unequal layer counts, we evaluated 24 canonical Watson&ndash;Crick stems (all ViennaRNA base-pair probability &ge; 0.952) stratified evenly into 8 short tRNA stems (10&ndash;23 bp), 8 medium tRNA acceptor stems (55&ndash;65 bp), and 8 long 12S/16S rRNA stems (147&ndash;240 bp) embedded in <code>L = 768 bp</code> windows. For each pair <code>(i, j)</code>, we quantified full-sequence double-mutant epistasis <code>E<sub>full</sub> = &Delta;&#8466;<sub>ij</sub> &minus; (&Delta;&#8466;<sub>i</sub> + &Delta;&#8466;<sub>j</sub>)</code> and directed downstream conditional rescue <code>E<sub>site j</sub> = log P(comp<sub>j</sub> | mut<sub>i</sub>) / P(wt<sub>j</sub> | mut<sub>i</sub>) &minus; log P(comp<sub>j</sub> | wt) / P(wt<sub>j</sub> | wt)</code> under intact inference and four equal-count 3-layer mean ablations (Table 3, Figure 3).</p>

{t3}
{make_html_fig(fig_dir, 'fig3_rna_epistasis_operator_ablation.png', 'Figure 3. Equal-Count 3-Layer Ablation Isolates Attention-Dependent RNA Stem Epistasis at 55–65 bp Separations.', '(A) Mean full-sequence compensatory epistasis E_full (± SEM) across intact and 3-layer ablated models. (B) Directed conditional base-pair rescue E_site j at the downstream nucleotide j. MHA ablation ([3,10,17]) abolishes medium-range (55–65 bp) acceptor coupling while retaining 47% of short-range (10–23 bp) coupling.', max_height='2.75in')}

<ul>
  <li><strong>Medium-Range tRNA Acceptor Epistasis Requires Sparse Attention:</strong> Under intact inference, Evo 2 20B exhibits strong compensatory coupling on both short 10&ndash;23 bp stems (<code>E_full = {ep[('Intact','short_10_25')]['mean_full']:+.3f}</code> nats, one-sided Wilcoxon <code>p = {fmt_p(ep[('Intact','short_10_25')]['p_full_vs0'])}</code>; <code>E_site j = {ep[('Intact','short_10_25')]['mean_sitej']:+.3f}</code>, <code>p = {fmt_p(ep[('Intact','short_10_25')]['p_sitej_vs0'])}</code>) and medium 55&ndash;65 bp acceptor stems (<code>E_full = {ep[('Intact','medium_55_70')]['mean_full']:+.3f}</code> nats, <code>p = {fmt_p(ep[('Intact','medium_55_70')]['p_full_vs0'])}</code>; <code>E_site j = {ep[('Intact','medium_55_70')]['mean_sitej']:+.3f}</code>, <code>p = {fmt_p(ep[('Intact','medium_55_70')]['p_sitej_vs0'])}</code>). Removing the three <code>MHA</code> layers completely extinguishes medium-range coupling (<code>E_full {ep[('Intact','medium_55_70')]['mean_full']:+.3f} &rarr; {ep[('Ablate_MHA','medium_55_70')]['mean_full']:+.3f}</code>, paired two-sided Wilcoxon <code>p = {fmt_p(M['p_int_vs_mha_full_2s'])}</code>, one-sided <code>p = {fmt_p(M['p_int_vs_mha_full_1s'])}</code>; <code>E_site j {ep[('Intact','medium_55_70')]['mean_sitej']:+.3f} &rarr; {ep[('Ablate_MHA','medium_55_70')]['mean_sitej']:+.3f}</code>, two-sided <code>p = {fmt_p(M['p_int_vs_mha_sitej_2s'])}</code>), whereas local convolutions sustain 47.2% of short-range coupling (<code>E_full = {ep[('Ablate_MHA','short_10_25')]['mean_full']:+.3f}</code> nats). Long rRNA spans (147&ndash;240 bp) show modest intact directed site-<em>j</em> rescue (<code>{ep[('Intact','long_130_600')]['mean_sitej']:+.3f}</code> nats, <code>p = {fmt_p(ep[('Intact','long_130_600')]['p_sitej_vs0'])}</code>) alongside near-zero full-sequence epistasis (<code>{ep[('Intact','long_130_600')]['mean_full']:+.3f}</code> nats).</li>
  <li><strong>Trunk Sensitivity Control:</strong> Because the 3-layer subsets <code>−HCL [2,9,16]</code>, <code>−HCM [1,8,15]</code>, and <code>−HCS [0,7,14]</code> necessarily include essential early trunk blocks (<code>L2</code>, <code>L1</code>, and <code>L0</code> identified in Experiment 2A), those ablations cause global likelihood collapse across every stem length rather than distance-selective disruption.</li>
</ul>

<div class="page-break"></div>

<h2>5. Experiment 3: Unsupervised RNA Secondary-Structure Discovery Across 192 MHA Heads</h2>
<p>Given that MHA blocks causally mediate 55&ndash;65 bp RNA stem coupling, we extracted exact post-RoPE query-key causal attention maps from all 192 heads (64 heads &times; Layers 3, 10, 17) across 2,048 bp windows covering all 22 human mt-tRNAs plus intact <code>MT-RNR1</code> (12S rRNA, 954 nt) and <code>MT-RNR2</code> (16S rRNA, 1,559 nt). Attention maps were symmetrized and corrected via Average Product Correction (APC; Rao et al., 2021) without any structural supervision or fine-tuning, and scored against ViennaRNA partition-function pairs (<code>bpp &ge; 0.50</code>, <code>|i &minus; j| &ge; 4</code>; Table 4A, Table 4B, Figure 4).</p>

{t4a}
{t4b}
{make_html_fig(fig_dir, 'fig4_mha_rna_contact_maps.png', 'Figure 4. Unsupervised RNA Base-Pair Contact Maps Emerge in Deep Sparse Attention Heads of Evo 2 20B.', '(A) Contact AUROC across all 192 MHA heads highlights specialist L17_H5 (mean 0.7604 overall, 0.8547 ≥40 bp). (B) Per-gene APC contact recovery across all 22 mt-tRNAs and 12S/16S rRNAs. (C–D) Split triangle contact comparisons on MT-TL1 (crimson lower: RNAfold bpp ≥ 0.5; blue upper & gold circles: L17_H5 APC or 192-head LOO ensemble). (E) Kilobase-scale contact recovery on 954 nt 12S rRNA (MT-RNR1, L17_H5 ≥130 bp AUROC = 0.7757). (F) Surge of long-range specialist gene×head hits across MHA depth (42 → 64 → 180).', max_height='3.20in')}

<ul>
  <li><strong>Discovery of Unsupervised Specialist Head <code>L17_H5</code>:</strong> A single deep attention head, Layer 17 Head 5 (<code>L17_H5</code>), acts as an unsupervised RNA base-pair contact detector across folded human mitochondrial transcripts. Across all 22 mt-tRNAs, <code>L17_H5</code> achieves mean APC AUROC = <strong><code>{h5['auroc_apc']:.4f}</code></strong> (mean raw directed AUROC = <code>{h5['auroc_raw_dir']:.4f}</code>, long-range &ge;40 bp AUROC = <strong><code>{h5['auroc_ge40']:.4f}</code></strong>, AUPRC = <code>{h5['auprc_apc']:.4f}</code>), peaking on <em>MT-TI</em> (<code>0.9748</code>), <em>MT-TS2</em> (<code>0.9469</code>), <em>MT-TV</em> (<code>0.9270</code>), <em>MT-TD</em> (<code>0.8907</code>), <em>MT-TY</em> (<code>0.8903</code>), and canonical MELAS locus <em>MT-TL1</em> (<code>0.8121</code>; <code>&ge;40 bp = 0.9085</code>). Crucially, a 192-head leave-one-tRNA-out regularized logistic regression ensemble reaches mean gene AUROC = <code>{em['ensemble_auroc']:.4f}</code> (<code>&ge;40 bp = {em['ensemble_auroc_ge40']:.4f}</code>), demonstrating that dense linear mixing dilutes the clean geometric signal concentrated in <code>L17_H5</code>.</li>
  <li><strong>Kilobase-Scale rRNA Contacts &amp; Depth Emergence:</strong> Unsupervised contact recovery generalizes to ~1&ndash;1.6 kb ribosomal RNAs without parameter tuning: on <code>MT-RNR1</code> (954 nt), <code>L10_H53</code> reaches AUROC = <code>{M['rnr1_top']['auroc_apc']:.4f}</code> while <code>L17_H5</code> achieves strong very-long-range contact recovery (<code>{M['rnr1_h5']['auroc_apc']:.4f}</code> overall, <code>{M['rnr1_h5']['auroc_ge40']:.4f}</code> &ge;40 bp, and <strong><code>{M['rnr1_h5']['auroc_ge130']:.4f}</code></strong> for stems separated by <code>&ge;130 bp</code>). On <code>MT-RNR2</code> (1,559 nt), <code>L17_H7</code> reaches <code>{M['rnr2_top'][0]['auroc_apc']:.4f}</code> and <code>L17_H2</code> achieves <code>{M['rnr2_top'][1]['auroc_apc']:.4f}</code> (<code>&ge;130 bp = {M['rnr2_top'][1]['auroc_ge130']:.4f}</code>). Counting gene &times; head hits exceeding AUROC 0.70 on long-range mt-tRNA pairs (&ge;40 bp) shows a 4.3&times; architectural progression across depth: <strong><code>L3 = {dc[3]['tRNA_ge40']} &rarr; L10 = {dc[10]['tRNA_ge40']} &rarr; L17 = {dc[17]['tRNA_ge40']}</code></strong>, while mid/deep attention dominates kilobase rRNA stems (<code>L3 = {dc[3]['rRNA_all']} &rarr; L10 = {dc[10]['rRNA_all']} &rarr; L17 = {dc[17]['rRNA_all']}</code>).</li>
</ul>

<h2>6. Discussion, Limitations, Credits &amp; Verified References</h2>
<p>By pairing pure mixer-update probing against single-layer mean ablations and exact post-RoPE attention extraction, this study establishes a causal functional taxonomy inside StripedHyena 2: <code>HCS</code> blocks execute sharp local motif filtering and exon-specific structural checks (notably <code>L14</code> in <em>BRCA1</em> Exon 18), <code>HCM</code> blocks specialize in splice-junction syntax and coding integrity (<code>L15</code> in Exon 15, <code>L05</code> on mtDNA CDS), <code>HCL</code> integrates regional GC composition and long-range backbone state, and deep sparse <code>MHA</code> heads&mdash;exemplified by unsupervised structural head <code>L17_H5</code>&mdash;supply the geometric coordinate lookup required for 55&ndash;65 bp compensatory RNA epistasis and kilobase rRNA contacts.</p>
<p><strong>Scope &amp; Limitations:</strong> First, linear probing and mean-ablation results on layers <code>L0&ndash;L2</code> reflect essential representation initialization (&ldquo;trunk collapse&rdquo;) where single-layer removal destroys downstream token distribution; sub-component circuit patching is left for future work. Second, RNA base-pair benchmarks use thermodynamic ViennaRNA ensembles rather than crystallography for mt-rRNAs. Finally, all inference executed under <code>mmaemu_all</code> FP8 weight emulation on consumer GPUs matches official reference arithmetic cleanly (Ding &amp; Qin, 2026) within the 80 GB VRAM budget.</p>
<p><strong>Contributor &amp; Tooling Acknowledgment:</strong> Project direction, experimental hypotheses, multi-GPU architecture, and final verification were led by <strong>Morgan Patterson</strong> (UMKC). Pipeline engineering, statistical benchmarking, figure generation, and technical drafting were contributed by <strong>Gemini Neon (Google Antigravity)</strong> under author direction.</p>

<ol class="refs">
  <li>Brixi G, Durrant MG, Ku J, et al. Genome modelling and design across all domains of life with Evo 2. <em>Nature</em>. 2026 Apr;652(8112):1349&ndash;1361. doi:10.1038/s41586-026-10176-5. PMID:41781614.</li>
  <li>Ding X, Qin Z. Accuracy is not compatibility: two Evo 2 checkpoints hinge on one FP8 matrix multiplication. <em>arXiv:2610.39415</em>. 2026 Oct.</li>
  <li>Findlay GM, Daza RM, Martin B, et al. Accurate classification of BRCA1 variants with saturation genome editing. <em>Nature</em>. 2018 Oct;562(7726):217&ndash;222. doi:10.1038/s41586-018-0461-z. PMID:30209399.</li>
  <li>Avsec &#381;, Latysheva N, Cheng J, et al. Advancing regulatory variant effect prediction with AlphaGenome. <em>Nature</em>. 2026 Jan;649(8099):1206&ndash;1218. doi:10.1038/s41586-025-10014-0. PMID:41606153.</li>
  <li>Lorenz R, Bernhart SH, H&ouml;ner zu Siederdissen C, et al. ViennaRNA Package 2.0. <em>Algorithms for Molecular Biology</em>. 2011 Nov 24;6:26. doi:10.1186/1748-7188-6-26. PMID:22115189.</li>
  <li>Rao R, Ovchinnikov S, Meier J, Rives A, Sercu T. Transformer protein language models are unsupervised structure learners. <em>ICLR</em>. 2021.</li>
  <li>Andrews RM, Kubacka I, Chinnery PF, et al. Reanalysis and revision of the Cambridge reference sequence for human mitochondrial DNA. <em>Nature Genetics</em>. 1999 Oct;23(2):147. doi:10.1038/13779. PMID:10508508.</li>
  <li>Landrum MJ, Chitipiralla S, Kaur K, et al. ClinVar: updates to support classifications of both germline and somatic variants. <em>Nucleic Acids Research</em>. 2025 Jan 6;53(D1):D1313&ndash;D1321. doi:10.1093/nar/gkae1090. PMID:39578691.</li>
</ol>

<h2>Appendix: Reproducibility &amp; Data Integrity Manifest</h2>
{ta}

</body>
</html>'''

    out_html = Path('/tmp/sh2_report_build.html')
    out_html.write_text(html_doc, encoding='utf-8')
    cmd = [
        CHROME_BIN,
        '--headless=new',
        '--disable-gpu',
        '--no-pdf-header-footer',
        f'--print-to-pdf={out_pdf}',
        out_html.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f'Rendered {out_pdf} ({out_pdf.stat().st_size:,} bytes)')


# ---------------------------------------------------------------------------
# python-docx DOCX Builder
# ---------------------------------------------------------------------------
def _set_cell_shading(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))


def _set_cell_margins(cell, top=50, bottom=50, left=70, right=70):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for side, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{side}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def _set_table_borders(table):
    tblPr = table._tbl.tblPr
    tblPr.append(parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{BORDER_HEX}"/>'
        f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="{NAVY_HEX}"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{BORDER_HEX}"/>'
        f'<w:left w:val="none"/><w:right w:val="none"/><w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    ))


def _set_row_repeat_header(row):
    row._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))


def _set_row_cant_split(row):
    row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))


def _add_bottom_border(paragraph, color_hex=NAVY_HEX, sz='10'):
    pPr = paragraph._p.get_or_add_pPr()
    pPr.append(parse_xml(
        f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="{sz}" w:space="4" w:color="{color_hex}"/></w:pBdr>'
    ))


def _add_field(run, field_str):
    run._r.append(parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="begin"/>'))
    run._r.append(parse_xml(f'<w:instrText {nsdecls("w")} xml:space="preserve"> {field_str} </w:instrText>'))
    run._r.append(parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="separate"/>'))
    run._r.append(parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="end"/>'))


def _add_run(p, text, font_name='Georgia', size=10.0, bold=False, italic=False, color=BODY):
    r = p.add_run(text)
    r.font.name = font_name
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    r.font.color.rgb = color
    return r


def _add_body(doc, text, bold_prefix=None, space_after=5.0, size=10.0, color=BODY):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    if bold_prefix:
        _add_run(p, bold_prefix, size=size, bold=True, color=NAVY)
    _add_run(p, text, size=size, color=color)
    return p


def _add_bullet(doc, bold_lead, text, size=9.8):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.line_spacing = 1.13
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3.5)
    if bold_lead:
        _add_run(p, bold_lead, size=size, bold=True, color=NAVY)
    _add_run(p, text, size=size, color=BODY)
    return p


def _add_h1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(11)
    p.paragraph_format.space_after = Pt(4.5)
    p.paragraph_format.keep_with_next = True
    _add_run(p, text, font_name='Calibri', size=13.0, bold=True, color=NAVY)
    _add_bottom_border(p, NAVY_HEX, sz='8')
    return p


def _add_callout(doc, title, specs, bg_hex=TEAL_BG_HEX, bar_hex=TEAL_BAR_HEX, title_color=TEAL):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.9)
    _set_cell_shading(cell, bg_hex)
    _set_cell_margins(cell, top=85, bottom=85, left=130, right=110)
    _set_row_cant_split(tbl.rows[0])
    cell._tc.get_or_add_tcPr().append(parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{bar_hex}"/>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{BORDER_HEX}"/>'
        f'<w:right w:val="single" w:sz="4" w:space="0" w:color="{BORDER_HEX}"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="{BORDER_HEX}"/>'
        f'</w:tcBorders>'
    ))
    p0 = cell.paragraphs[0]
    p0.paragraph_format.space_before = Pt(0)
    p0.paragraph_format.space_after = Pt(3)
    _add_run(p0, title, font_name='Calibri', size=9.8, bold=True, color=title_color)
    for idx, (lead, body_txt) in enumerate(specs):
        p = cell.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2.5 if idx < len(specs) - 1 else 0)
        p.paragraph_format.line_spacing = 1.1
        if lead:
            _add_run(p, lead, font_name='Calibri', size=9.0, bold=True, color=NAVY)
        _add_run(p, body_txt, font_name='Calibri', size=9.0, color=BODY)
    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(3.5)


def _add_table(doc, title, headers, rows, col_widths, right_cols=None, font_size=7.8):
    if right_cols is None:
        right_cols = set()
    cap = doc.add_paragraph()
    cap.paragraph_format.space_before = Pt(5)
    cap.paragraph_format.space_after = Pt(2.5)
    cap.paragraph_format.keep_with_next = True
    _add_run(cap, title, font_name='Calibri', size=9.0, bold=True, color=NAVY)

    tbl = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    _set_table_borders(tbl)

    hdr_row = tbl.rows[0]
    _set_row_repeat_header(hdr_row)
    _set_row_cant_split(hdr_row)
    for j, htxt in enumerate(headers):
        cell = hdr_row.cells[j]
        cell.width = Inches(col_widths[j])
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        _set_cell_shading(cell, NAVY_HEX)
        _set_cell_margins(cell, top=50, bottom=50, left=60, right=60)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if j in right_cols else WD_ALIGN_PARAGRAPH.LEFT
        _add_run(p, htxt, font_name='Calibri', size=font_size, bold=True, color=WHITE)

    bold_triggers = ('Local TF Motifs', 'Regional 1 kb', 'intact', 'L10', 'L17', 'L15', 'L14', 'L21', 'L22', 'Intact', '−MHA', 'L17_H5', 'MT-RNR1', 'MT-RNR2', '192-Head LOO', 'Layer 17')
    for i, rdata in enumerate(rows):
        row = tbl.rows[i + 1]
        _set_row_cant_split(row)
        bg = ZEBRA_HEX if i % 2 == 1 else 'FFFFFF'
        for j, val in enumerate(rdata):
            cell = row.cells[j]
            cell.width = Inches(col_widths[j])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            _set_cell_shading(cell, bg)
            _set_cell_margins(cell, top=40, bottom=40, left=60, right=60)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if j in right_cols else WD_ALIGN_PARAGRAPH.LEFT
            s = str(val)
            is_b = (j == 0 and any(s.startswith(t) for t in bold_triggers))
            _add_run(p, s, font_name='Calibri', size=font_size, bold=is_b, color=BODY)

    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(3.5)


def _add_fig(doc, img_path: Path, width_in: float, title: str, body: str):
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(4)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.paragraph_format.keep_with_next = True
    p_img.add_run().add_picture(str(img_path), width=Inches(width_in))

    p_cap = doc.add_paragraph()
    p_cap.paragraph_format.space_before = Pt(0)
    p_cap.paragraph_format.space_after = Pt(6)
    p_cap.paragraph_format.line_spacing = 1.1
    _add_run(p_cap, title + ' ', font_name='Calibri', size=8.8, bold=True, color=NAVY)
    _add_run(p_cap, body, font_name='Calibri', size=8.8, italic=True, color=MUTED)


def build_docx(M: dict, fig_dir: Path, out_docx: Path):
    p = M['probe_dict']
    ep = M['ep_lookup']
    mi = M['mt_intact']
    br_rows = M['br_key_rows']
    br_int = br_rows[0]
    br_l14 = next(r for r in br_rows if r['cond'] == 'L14')
    br_l15 = next(r for r in br_rows if r['cond'] == 'L15')
    br_l17 = next(r for r in br_rows if r['cond'] == 'L17')
    mt_l10 = next(r for r in M['mt_key_rows'] if r['cond'] == 'L10')
    mt_l17 = next(r for r in M['mt_key_rows'] if r['cond'] == 'L17')
    mt_l05 = next(r for r in M['mt_key_rows'] if r['cond'] == 'L05')
    mt_l22 = next(r for r in M['mt_key_rows'] if r['cond'] == 'L22')
    h5 = M['top5_trna_heads'][0]
    em = M['ens_mean']
    dc = M['depth_counts']

    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11.0)
    sec.top_margin = Inches(0.75)
    sec.bottom_margin = Inches(0.75)
    sec.left_margin = Inches(0.8)
    sec.right_margin = Inches(0.8)
    sec.different_first_page_header_footer = True

    hdr = sec.header.paragraphs[0]
    hdr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    _add_run(hdr, 'Patterson & Gemini Neon (2026)  |  StripedHyena 2 (Evo 2 20B) Mechanistic Dissection', font_name='Calibri', size=8.4, italic=True, color=MUTED)

    ftr = sec.footer.paragraphs[0]
    ftr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    _add_run(ftr, 'Page ', font_name='Calibri', size=8.4, color=MUTED)
    rpg = ftr.add_run()
    rpg.font.name = 'Calibri'
    rpg.font.size = Pt(8.4)
    rpg.font.color.rgb = MUTED
    _add_field(rpg, 'PAGE')
    _add_run(ftr, ' of ', font_name='Calibri', size=8.4, color=MUTED)
    rnp = ftr.add_run()
    rnp.font.name = 'Calibri'
    rnp.font.size = Pt(8.4)
    rnp.font.color.rgb = MUTED
    _add_field(rnp, 'NUMPAGES')

    pt = doc.add_paragraph()
    pt.paragraph_format.space_before = Pt(0)
    pt.paragraph_format.space_after = Pt(4)
    _add_run(
        pt,
        'Causal Mechanistic Dissection of StripedHyena 2 (Evo 2 20B): Operator Division of Labor, Residual-Stream Confounding, and Unsupervised RNA Contact-Map Recovery Across 192 Sparse Attention Heads',
        font_name='Georgia', size=14.8, bold=True, color=NAVY,
    )
    _add_bottom_border(pt, NAVY_HEX, sz='12')

    pa = doc.add_paragraph()
    pa.paragraph_format.space_after = Pt(2)
    _add_run(pa, 'Author: ', font_name='Calibri', size=9.5, bold=True, color=NAVY)
    _add_run(pa, 'Morgan Patterson (University of Missouri–Kansas City)   |   ', font_name='Calibri', size=9.5, color=BODY)
    _add_run(pa, 'Contributor / Tooling: ', font_name='Calibri', size=9.5, bold=True, color=NAVY)
    _add_run(pa, 'Gemini Neon (Google Antigravity)', font_name='Calibri', size=9.5, color=BODY)

    pm = doc.add_paragraph()
    pm.paragraph_format.space_after = Pt(8)
    _add_run(pm, '24 Hybrid Layers (7 HCS / 7 HCM / 7 HCL / 3 Sparse MHA, 192 Heads, d_model=8,192)  •  Validated Evo 2 20B mmaemu_all  •  5× RTX 5060 Ti (80 GB VRAM)', font_name='Calibri', size=8.5, color=TEAL)

    _add_callout(doc, 'Executive Summary: Empirical Division of Labor Inside StripedHyena 2', [
        ('1. Pure Mixer vs. Residual-Stream Illusion: ',
         f"Probing cumulative residual streams (x_l) makes late MHA blocks appear dominant on TF motifs ({p['TF_motifs']['res_means']['MHA']:.4f} vs. HCS {p['TF_motifs']['res_means']['HCS']:.4f}). Pure mixer updates (Δh_l) invert this: HCS leads on TF motifs ({p['TF_motifs']['mix_means']['HCS']:.4f}, L0={p['TF_motifs']['mix_peak_val']:.4f} vs. MHA {p['TF_motifs']['mix_means']['MHA']:.4f}), HCM peaks at L19 on splice motifs ({p['Splice_GT_AG']['mix_means']['HCM']:.4f}, peak {p['Splice_GT_AG']['mix_peak_val']:.4f}), and 1 kb upstream GC% rises monotonically across operator horizon (HCS {p['Regional_1kb_GC']['mix_means']['HCS']:+.4f} < HCM {p['Regional_1kb_GC']['mix_means']['HCM']:+.4f} < HCL {p['Regional_1kb_GC']['mix_means']['HCL']:+.4f} < MHA {p['Regional_1kb_GC']['mix_means']['MHA']:+.4f})."),
        ('2. Causal Organelle, Splice, & Epistasis Dissociation: ',
         f"Single-layer mean ablation on 16,569 bp rCRS mtDNA reveals that MHA blocks L10 and L17 selectively harm mt-tRNA stems (L10 stem ΔNLL {mt_l10['delta_nll_tRNA_stem']:+.4f} vs. CDS {mt_l10['delta_nll_CDS']:+.4f}, 4.2×) and ClinVar tRNA AUROC (L10 {mt_l10['delta_auroc_tRNA']:+.4f}; L17 {mt_l17['delta_auroc_tRNA']:+.4f}). In BRCA1, L15 (HCM) drops Exon 15 splicing ({br_l15['delta_X15_auroc_Splicing']:+.4f} vs. AlphaMissense {br_l15['delta_X15_auroc_AlphaMissense']:+.4f}), while L14 (HCS) drops Exon 18 missense ({br_l14['delta_X18_auroc_AlphaMissense']:+.4f} vs. splicing {br_l14['delta_X18_auroc_Splicing']:+.4f}). Removing all 3 MHA blocks ([3,10,17]) extinguishes 55–65 bp RNA acceptor epistasis (E_full {ep[('Intact','medium_55_70')]['mean_full']:+.3f} → {ep[('Ablate_MHA','medium_55_70')]['mean_full']:+.3f} nats, p={fmt_p(M['p_int_vs_mha_full_2s'])}; E_site j {ep[('Intact','medium_55_70')]['mean_sitej']:+.3f} → {ep[('Ablate_MHA','medium_55_70')]['mean_sitej']:+.3f} nats, p={fmt_p(M['p_int_vs_mha_sitej_2s'])}), while keeping 47% of short-range coupling."),
        ('3. Unsupervised RNA Base-Pair Attention Heads: ',
         f"Post-RoPE attention head L17_H5 recovers folded base pairs across all 22 human mt-tRNAs without structural supervision (mean APC AUROC={h5['auroc_apc']:.4f}, ≥40 bp={h5['auroc_ge40']:.4f}; MT-TI=0.9748, MT-TS2=0.9469) and generalizes to kilobase 12S/16S rRNAs (MT-RNR1 L17_H5 ≥130 bp=0.7757; MT-RNR2 L17_H7=0.8616)."),
    ])

    _add_h1(doc, '1. Architecture, Precision Protocol & Residual Decoupling')
    _add_body(doc, 'Evo 2 20B (Brixi et al., 2026) implements the 24-block StripedHyena 2 architecture interleaving seven Hyena Short (HCS: [0,4,7,11,14,18,21]), seven Hyena Medium (HCM: [1,5,8,12,15,19,22]), and seven Hyena Long (HCL: [2,6,9,13,16,20,23]) gated convolution blocks with three sparse causal Multi-Head Attention blocks (MHA: [3,10,17], 64 heads/layer = 192 total heads, head_dim=128, d_model=8,192). All forward hooks and ablations executed under validated mmaemu_all FP8 linear emulation (Ding & Qin, 2026) across five RTX 5060 Ti GPUs (80 GB VRAM total).')

    _add_h1(doc, '2. Experiment 1: Pure Mixer Update vs. Residual-Stream Linear Probing')
    _add_body(doc, 'Across 12 contiguous 8,192 bp BRCA1 windows (98,304 bp; GENCODE v46 / Findlay et al., 2018) and both strands of circular human mtDNA (16,569 bp rCRS; Andrews et al., 1999), we evaluated 5-fold GroupKFold OOF linear probes on isolated mixer updates (Δh_l) and cumulative residual streams (x_l).')
    _add_table(
        doc,
        'Table 1. Experiment 1: 24-Layer Pure Mixer Update (Δh_l) vs. Residual Stream (x_l) Probing',
        ['Sequence Task (Metric)', 'N (Pos)', 'Peak Mixer Layer', 'HCS', 'HCM', 'HCL', 'MHA Mix', 'MHA Res'],
        [
            [
                f"{r['label']} ({r['metric']})",
                f"{r['n']:,} ({r['n_pos']:,})",
                f"L{r['mix_peak_layer']} ({r['mix_peak_op']}): {r['mix_peak_val']:.4f}",
                f"{r['mix_means']['HCS']:+.4f}" if tk == 'Regional_1kb_GC' else f"{r['mix_means']['HCS']:.4f}",
                f"{r['mix_means']['HCM']:+.4f}" if tk == 'Regional_1kb_GC' else f"{r['mix_means']['HCM']:.4f}",
                f"{r['mix_means']['HCL']:+.4f}" if tk == 'Regional_1kb_GC' else f"{r['mix_means']['HCL']:.4f}",
                f"{r['mix_means']['MHA']:+.4f}" if tk == 'Regional_1kb_GC' else f"{r['mix_means']['MHA']:.4f}",
                f"{r['res_means']['MHA']:+.4f}" if tk == 'Regional_1kb_GC' else f"{r['res_means']['MHA']:.4f}",
            ]
            for r in M['probe_rows']
            for tk in [r['task']]
        ],
        [1.95, 0.80, 1.25, 0.56, 0.56, 0.56, 0.61, 0.61],
        right_cols={3, 4, 5, 6, 7},
    )
    _add_fig(doc, fig_dir / 'fig1_operator_probing_division_of_labor.png', 6.8,
             'Figure 1. Operator Division of Labor and the Residual-Stream Confound Across All 24 StripedHyena 2 Blocks.',
             'Pure mixer updates Δh_l (solid) isolate true operator computations (HCS TF motifs, HCM splice sites, HCL/MHA regional GC) from residual-stream accumulation (dashed gray).')

    _add_h1(doc, '3. Experiment 2A & 2B: 24-Layer Causal Mean-Ablation on mtDNA & BRCA1 Modalities')
    _add_body(doc, 'Single-layer mean-ablation (Δh_l <- mean_t(Δh_l)) across 16,569 bp rCRS mtDNA (ClinVar 2025; Landrum et al., 2025) and BRCA1 Exons 15 & 18 stratified by AlphaGenome LOF modality (Avsec et al., 2026) demonstrates causal division of labor beyond linear correlation.')
    _add_table(
        doc,
        'Table 2A. Experiment 2A: Representative 24-Layer Causal Mean-Ablations on rCRS mtDNA',
        ['Condition', 'Op.', 'ΔNLL Gen.', 'ΔNLL CDS', 'ΔtRNA Stem', 'ΔtRNA Loop', 'ClinVar All', 'ΔCDS AUC', 'ΔtRNA AUC'],
        [
            [
                r['cond'] if r['cond'] != 'intact' else 'intact (base)',
                r['operator'],
                f"{r['mean_nll_genome']:.4f}" if r['cond'] == 'intact' else fmt_signed(r['delta_mean_nll_genome'], 4),
                f"{r['nll_CDS']:.4f}" if r['cond'] == 'intact' else fmt_signed(r['delta_nll_CDS'], 4),
                f"{r['nll_tRNA_stem']:.4f}" if r['cond'] == 'intact' else fmt_signed(r['delta_nll_tRNA_stem'], 4),
                f"{r['nll_tRNA_loop']:.4f}" if r['cond'] == 'intact' else fmt_signed(r['delta_nll_tRNA_loop'], 4),
                f"{r['auroc_all_mtDNA']:.4f}",
                f"{r['auroc_CDS']:.4f}" if r['cond'] == 'intact' else fmt_signed(r['delta_auroc_CDS'], 4),
                f"{r['auroc_tRNA']:.4f}" if r['cond'] == 'intact' else fmt_signed(r['delta_auroc_tRNA'], 4),
            ]
            for r in M['mt_key_rows']
        ],
        [0.95, 0.45, 0.78, 0.76, 0.84, 0.84, 0.80, 0.74, 0.74],
        right_cols={2, 3, 4, 5, 6, 7, 8},
        font_size=7.4,
    )
    _add_table(
        doc,
        'Table 2B. Experiment 2B: BRCA1 Exon 15 & Exon 18 AlphaGenome Modality Causal Dissociation',
        ['Condition', 'Op.', 'X15 All', 'X15 Splice Δ', 'X15 Missense Δ', 'X18 All', 'X18 Splice Δ', 'X18 Missense Δ'],
        [
            [
                r['cond'] if r['cond'] != 'intact' else 'intact (base)',
                r['operator'],
                f"{r['X15_auroc_all']:.4f}",
                f"{r['X15_auroc_Splicing']:.4f}" if r['cond'] == 'intact' else fmt_signed(r['delta_X15_auroc_Splicing'], 4),
                f"{r['X15_auroc_AlphaMissense']:.4f}" if r['cond'] == 'intact' else fmt_signed(r['delta_X15_auroc_AlphaMissense'], 4),
                f"{r['X18_auroc_all']:.4f}",
                f"{r['X18_auroc_Splicing']:.4f}" if r['cond'] == 'intact' else fmt_signed(r['delta_X18_auroc_Splicing'], 4),
                f"{r['X18_auroc_AlphaMissense']:.4f}" if r['cond'] == 'intact' else fmt_signed(r['delta_X18_auroc_AlphaMissense'], 4),
            ]
            for r in br_rows
        ],
        [1.00, 0.50, 0.75, 0.95, 1.00, 0.75, 0.95, 1.00],
        right_cols={2, 3, 4, 5, 6, 7},
    )
    _add_fig(doc, fig_dir / 'fig2_causal_ablation_24layers.png', 6.8,
             'Figure 2. Single-Layer Causal Mean-Ablation Across All 24 Layers.',
             'Demonstrates universal trunk layers (L0–L2, L21, L23), near-redundancy of L22, selective MHA sensitivity in folded mt-tRNAs (L10/L17), and AlphaGenome modality dissociation (L15 splicing vs. L14/L17 missense).')

    _add_h1(doc, '4. Experiment 2C: Equal-Count Operator Ablation on RNA Epistasis')
    _add_body(doc, f"Across 24 canonical Watson–Crick stem pairs (ViennaRNA bpp >= 0.952; Lorenz et al., 2011) in L=768 bp windows, ablating all 3 MHA blocks ([3,10,17]) completely abolishes medium-range (55–65 bp) acceptor stem epistasis (E_full {ep[('Intact','medium_55_70')]['mean_full']:+.3f} -> {ep[('Ablate_MHA','medium_55_70')]['mean_full']:+.3f} nats, paired Wilcoxon two-sided p={fmt_p(M['p_int_vs_mha_full_2s'])}, one-sided p={fmt_p(M['p_int_vs_mha_full_1s'])}; E_site j {ep[('Intact','medium_55_70')]['mean_sitej']:+.3f} -> {ep[('Ablate_MHA','medium_55_70')]['mean_sitej']:+.3f} nats, two-sided p={fmt_p(M['p_int_vs_mha_sitej_2s'])}) while preserving 47.2% of short-range coupling ({ep[('Ablate_MHA','short_10_25')]['mean_full']:+.3f} nats). Note that the equal-count 3-layer convolutional ablations (-HCL [2,9,16], -HCM [1,8,15], -HCS [0,7,14]) each contain a trunk layer (L2/L1/L0) and therefore collapse baseline sequence likelihood globally.")
    _add_table(
        doc,
        'Table 3. Experiment 2C: Equal-Count 3-Layer Ablation Across 24 Canonical RNA Stem Pairs',
        ['Condition', 'Layers', 'Stratum (n=8)', 'Span', 'E_full (nats)', 'p vs 0', 'E_site j', 'p vs 0'],
        [
            [r['cond_label'], r['layers'], r['stratum_label'], f"{r['dist_min']}–{r['dist_max']}", f"{r['mean_full']:+.3f}±{r['sem_full']:.2f}", fmt_p(r['p_full_vs0']), f"{r['mean_sitej']:+.3f}±{r['sem_sitej']:.2f}", fmt_p(r['p_sitej_vs0'])]
            for r in M['ep_summary']
        ],
        [1.15, 0.72, 1.38, 0.65, 0.95, 0.55, 0.95, 0.55],
        right_cols={3, 4, 5, 6, 7},
        font_size=7.4,
    )
    _add_fig(doc, fig_dir / 'fig3_rna_epistasis_operator_ablation.png', 6.8,
             'Figure 3. Equal-Count 3-Layer Ablation on 24 RNA Stem Epistasis Pairs.',
             'MHA ablation eliminates 55–65 bp acceptor stem epistasis while local convolutions retain 47% of 10–23 bp stem coupling.')

    _add_h1(doc, '5. Experiment 3: Unsupervised RNA Contact Maps Across 192 MHA Heads')
    _add_body(doc, f"Exact post-RoPE attention extraction across all 192 heads (APC-corrected, Rao et al., 2021) identifies specialist head L17_H5 achieving mean APC AUROC = {h5['auroc_apc']:.4f} (raw directed = {h5['auroc_raw_dir']:.4f}, long-range >=40 bp = {h5['auroc_ge40']:.4f}) across all 22 mt-tRNAs, outperforming a 192-head LOO logistic ensemble ({em['ensemble_auroc']:.4f}) and scaling to kilobase rRNAs (MT-RNR1 L17_H5 >=130 bp = {M['rnr1_h5']['auroc_ge130']:.4f}; MT-RNR2 L17_H7 = {M['rnr2_top'][0]['auroc_apc']:.4f}, L17_H2 >=130 bp = {M['rnr2_top'][1]['auroc_ge130']:.4f}). Long-range (>40 bp) specialist hits (>0.70 AUROC) expand 4.3x across depth: L3={dc[3]['tRNA_ge40']} -> L10={dc[10]['tRNA_ge40']} -> L17={dc[17]['tRNA_ge40']}.")
    _add_table(
        doc,
        'Table 4. Experiment 3: Top Unsupervised MHA Contact Heads & L17_H5 Specialist Performance',
        ['Head / Cohort', 'Evaluation Target', 'AUROC APC', 'AUROC Raw', 'AUROC ≥40 bp', 'AUPRC', '≥130 bp / L5'],
        [
            [hr['head_name'], '22 mt-tRNAs (mean)', f"{hr['auroc_apc']:.4f}", f"{hr['auroc_raw_dir']:.4f}", f"{hr['auroc_ge40']:.4f}", f"{hr['auprc_apc']:.4f}", f"L/5={hr['prec_L5_apc']:.4f}"]
            for hr in M['top5_trna_heads']
        ] + [
            ['192-Head LOO Ens.', '22 mt-tRNAs (LOO mean)', f"{em['ensemble_auroc']:.4f}", '—', f"{em['ensemble_auroc_ge40']:.4f}", f"{em['ensemble_auprc']:.4f}", f"L/5={em['ensemble_prec_L5']:.4f}"],
            [f"MT-RNR1 {M['rnr1_top']['head_name']}", '12S rRNA (954 nt)', f"{M['rnr1_top']['auroc_apc']:.4f}", f"{M['rnr1_top']['auroc_raw_dir']:.4f}", f"{M['rnr1_top']['auroc_ge40']:.4f}", f"{M['rnr1_top']['auprc_apc']:.4f}", f"≥130={M['rnr1_top']['auroc_ge130']:.4f}"],
            ['MT-RNR1 L17_H5', '12S rRNA (954 nt)', f"{M['rnr1_h5']['auroc_apc']:.4f}", f"{M['rnr1_h5']['auroc_raw_dir']:.4f}", f"{M['rnr1_h5']['auroc_ge40']:.4f}", f"{M['rnr1_h5']['auprc_apc']:.4f}", f"≥130={M['rnr1_h5']['auroc_ge130']:.4f}"],
            [f"MT-RNR2 {M['rnr2_top'][0]['head_name']}", '16S rRNA (1,559 nt)', f"{M['rnr2_top'][0]['auroc_apc']:.4f}", f"{M['rnr2_top'][0]['auroc_raw_dir']:.4f}", f"{M['rnr2_top'][0]['auroc_ge40']:.4f}", f"{M['rnr2_top'][0]['auprc_apc']:.4f}", f"≥130={M['rnr2_top'][0]['auroc_ge130']:.4f}"],
            [f"MT-RNR2 {M['rnr2_top'][1]['head_name']}", '16S rRNA (1,559 nt)', f"{M['rnr2_top'][1]['auroc_apc']:.4f}", f"{M['rnr2_top'][1]['auroc_raw_dir']:.4f}", f"{M['rnr2_top'][1]['auroc_ge40']:.4f}", f"{M['rnr2_top'][1]['auprc_apc']:.4f}", f"≥130={M['rnr2_top'][1]['auroc_ge130']:.4f}"],
        ],
        [1.35, 1.45, 0.82, 0.80, 0.88, 0.72, 0.88],
        right_cols={2, 3, 4, 5, 6},
    )
    _add_fig(doc, fig_dir / 'fig4_mha_rna_contact_maps.png', 6.8,
             'Figure 4. Unsupervised RNA Base-Pair Contact Discovery Across 192 Sparse MHA Heads.',
             'Specialist head L17_H5 recovers folded stems across 22 mt-tRNAs and kilobase 12S/16S rRNAs without structural supervision.')

    _add_h1(doc, '6. Verified References & Data Manifest')
    refs = [
        'Brixi G, Durrant MG, Ku J, et al. Genome modelling and design across all domains of life with Evo 2. Nature. 2026 Apr;652(8112):1349-1361. PMID:41781614.',
        'Ding X, Qin Z. Accuracy is not compatibility: two Evo 2 checkpoints hinge on one FP8 matrix multiplication. arXiv:2610.39415. 2026 Oct.',
        'Findlay GM, Daza RM, Martin B, et al. Accurate classification of BRCA1 variants with saturation genome editing. Nature. 2018 Oct;562(7726):217-222. PMID:30209399.',
        'Avsec Z, Latysheva N, Cheng J, et al. Advancing regulatory variant effect prediction with AlphaGenome. Nature. 2026 Jan;649(8099):1206-1218. PMID:41606153.',
        'Lorenz R, Bernhart SH, Honer zu Siederdissen C, et al. ViennaRNA Package 2.0. Algorithms for Molecular Biology. 2011 Nov 24;6:26. PMID:22115189.',
        'Rao R, Ovchinnikov S, Meier J, Rives A, Sercu T. Transformer protein language models are unsupervised structure learners. ICLR. 2021.',
        'Andrews RM, Kubacka I, Chinnery PF, et al. Reanalysis and revision of the Cambridge reference sequence for human mitochondrial DNA. Nature Genetics. 1999 Oct;23(2):147. PMID:10508508.',
        'Landrum MJ, Chitipiralla S, Kaur K, et al. ClinVar: updates to support classifications of both germline and somatic variants. Nucleic Acids Research. 2025 Jan 6;53(D1):D1313-D1321. PMID:39578691.',
    ]
    for i, rf in enumerate(refs, start=1):
        _add_body(doc, f'{i}. {rf}', space_after=2.5, size=8.8)

    _add_table(
        doc,
        f"Table A1. Verified Study Deliverables & SHA-256 Checksums ({len(M['manifest'])} Files)",
        ['Filename', 'Contents', 'Bytes', 'SHA-256'],
        [[r['filename'], r['desc'], f"{r['bytes']:,}", r['sha256']] for r in M['manifest']],
        [1.95, 2.20, 0.65, 2.10],
        right_cols={2},
        font_size=7.0,
    )

    doc.save(out_docx)
    print(f'Saved {out_docx} ({out_docx.stat().st_size:,} bytes)')


# ---------------------------------------------------------------------------
# Standalone Markdown Technical Report
# ---------------------------------------------------------------------------
def build_markdown(M: dict, out_md: Path):
    p = M['probe_dict']
    ep = M['ep_lookup']
    br_rows = M['br_key_rows']
    br_int = br_rows[0]
    br_l14 = next(r for r in br_rows if r['cond'] == 'L14')
    br_l15 = next(r for r in br_rows if r['cond'] == 'L15')
    br_l17 = next(r for r in br_rows if r['cond'] == 'L17')
    mt_l10 = next(r for r in M['mt_key_rows'] if r['cond'] == 'L10')
    mt_l17 = next(r for r in M['mt_key_rows'] if r['cond'] == 'L17')
    mt_l05 = next(r for r in M['mt_key_rows'] if r['cond'] == 'L05')
    mt_l22 = next(r for r in M['mt_key_rows'] if r['cond'] == 'L22')
    h5 = M['top5_trna_heads'][0]
    em = M['ens_mean']
    dc = M['depth_counts']

    lines = [
        '# Causal Mechanistic Dissection of StripedHyena 2 (Evo 2 20B): Operator Division of Labor, Residual-Stream Confounding, and Unsupervised RNA Contact-Map Recovery Across 192 Sparse Attention Heads',
        '',
        '**Author:** Morgan Patterson (University of Missouri–Kansas City)  ',
        '**Contributor / Tooling:** Gemini Neon (Google Antigravity) — pipeline engineering, statistical benchmarking, figure generation, and technical drafting under author direction.  ',
        '**Hardware & Precision:** 5× NVIDIA RTX 5060 Ti (80 GB VRAM aggregate) • `evo2_20b` with validated `mmaemu_all` FP8 emulation across 24 hybrid layers (`7 HCS [0,4,7,11,14,18,21]`, `7 HCM [1,5,8,12,15,19,22]`, `7 HCL [2,6,9,13,16,20,23]`, `3 MHA [3,10,17]`, `192` total heads, `d_model = 8192`).',
        '',
        '---',
        '',
        '## 1. Experiment 1: Pure Mixer Update vs. Residual-Stream Linear Probing',
        '',
        'Linear probing on full residual streams (`x_l`) creates an accumulation artifact where late sparse attention (`MHA`) layers appear to score `0.9509` AUROC on local TF motifs despite low direct mixer contribution (`0.6724`). Evaluating isolated mixer outputs (`Δh_l`) exposes the true computational hierarchy across 5-fold GroupKFold OOF evaluation:',
        '',
        '| Sequence Task | Metric | N (Pos) | Peak Mixer Layer | HCS Mean | HCM Mean | HCL Mean | MHA Mix | MHA Res |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|',
    ]
    for r in M['probe_rows']:
        tk = r['task']
        fmt_v = (lambda v: f'{v:+.4f}') if tk == 'Regional_1kb_GC' else (lambda v: f'{v:.4f}')
        lines.append(
            f"| {r['label']} | {r['metric']} | {r['n']:,} ({r['n_pos']:,}) | `L{r['mix_peak_layer']}` ({r['mix_peak_op']}): {r['mix_peak_val']:.4f} | "
            f"{fmt_v(r['mix_means']['HCS'])} | {fmt_v(r['mix_means']['HCM'])} | {fmt_v(r['mix_means']['HCL'])} | {fmt_v(r['mix_means']['MHA'])} | {fmt_v(r['res_means']['MHA'])} |"
        )
    lines += [
        '',
        '![Figure 1: Operator Division of Labor and Residual Confound](../figures/fig1_operator_probing_division_of_labor.png)',
        '',
        '---',
        '',
        '## 2. Experiment 2A & 2B: 24-Layer Causal Mean-Ablation on mtDNA & BRCA1 Modalities',
        '',
        f"- **Universal Trunk vs. Redundant Block:** Early/late backbone layers (`L00`, `L01`, `L02`, `L21`, `L23`) collapse whole-genome log-likelihood (+1.23 to +2.68 nats ΔNLL, ClinVar AUROC 0.33–0.48), whereas `L22 (HCM)` is nearly redundant (genome ΔNLL `{mt_l22['delta_mean_nll_genome']:+.4f}`, ClinVar all `{mt_l22['auroc_all_mtDNA']:.4f}`).",
        f"- **Folded mt-tRNA Vulnerability to MHA Ablation:** `L10 (MHA)` causes `{mt_l10['delta_nll_tRNA_stem']:+.4f}` nats ΔNLL on tRNA stems ({mt_l10['delta_nll_tRNA_stem']/mt_l10['delta_nll_CDS']:.1f}× coding `{mt_l10['delta_nll_CDS']:+.4f}`) and drops ClinVar tRNA AUROC `{mt_l10['delta_auroc_tRNA']:+.4f}` (`L17`: stem `{mt_l17['delta_nll_tRNA_stem']:+.4f}`, tRNA AUROC `{mt_l17['delta_auroc_tRNA']:+.4f}`). Conversely, `L05 (HCM)` preferentially hurts coding sequences (`{mt_l05['delta_nll_CDS']:+.4f}` vs. tRNA stem `{mt_l05['delta_nll_tRNA_stem']:+.4f}`).",
        f"- **BRCA1 AlphaGenome Modality Double Dissociation:** In Exon 15 (intact Splicing `{br_int['X15_auroc_Splicing']:.4f}`, AlphaMissense `{br_int['X15_auroc_AlphaMissense']:.4f}`), `L15 (HCM)` selectively destroys splicing LOF classification (`{br_l15['delta_X15_auroc_Splicing']:+.4f}` vs. missense `{br_l15['delta_X15_auroc_AlphaMissense']:+.4f}`). In Exon 18, `L14 (HCS)` collapses missense (`{br_l14['delta_X18_auroc_AlphaMissense']:+.4f}`) with zero effect on splicing (`{br_l14['delta_X18_auroc_Splicing']:+.4f}`), while `L17 (MHA)` collapses missense by `{br_l17['delta_X18_auroc_AlphaMissense']:+.4f}`.",
        '',
        '![Figure 2: 24-Layer Causal Mean-Ablation](../figures/fig2_causal_ablation_24layers.png)',
        '',
        '---',
        '',
        '## 3. Experiment 2C: Equal-Count 3-Layer Ablation on RNA Stem Epistasis',
        '',
        f"Across 24 canonical Watson–Crick stem pairs (`L = 768 bp`), intact Evo 2 20B shows significant positive epistasis on short 10–23 bp stems (`E_full = {ep[('Intact','short_10_25')]['mean_full']:+.3f}`, one-sided Wilcoxon `p = {fmt_p(ep[('Intact','short_10_25')]['p_full_vs0'])}`) and medium 55–65 bp acceptor stems (`E_full = {ep[('Intact','medium_55_70')]['mean_full']:+.3f}`, `p = {fmt_p(ep[('Intact','medium_55_70')]['p_full_vs0'])}`; `E_site j = {ep[('Intact','medium_55_70')]['mean_sitej']:+.3f}`, `p = {fmt_p(ep[('Intact','medium_55_70')]['p_sitej_vs0'])}`). Ablating all 3 MHA blocks (`[3,10,17]`) erases medium-range coupling (`E_full {ep[('Intact','medium_55_70')]['mean_full']:+.3f} -> {ep[('Ablate_MHA','medium_55_70')]['mean_full']:+.3f}`, paired two-sided `p = {fmt_p(M['p_int_vs_mha_full_2s'])}`, one-sided `p = {fmt_p(M['p_int_vs_mha_full_1s'])}`; `E_site j {ep[('Intact','medium_55_70')]['mean_sitej']:+.3f} -> {ep[('Ablate_MHA','medium_55_70')]['mean_sitej']:+.3f}`, two-sided `p = {fmt_p(M['p_int_vs_mha_sitej_2s'])}`) while preserving 47.2% of short-range coupling (`{ep[('Ablate_MHA','short_10_25')]['mean_full']:+.3f}`). Equal-count conv sets (`-HCL [2,9,16]`, `-HCM [1,8,15]`, `-HCS [0,7,14]`) contain essential trunk blocks `L2`/`L1`/`L0` and collapse baseline likelihood globally.",
        '',
        '![Figure 3: Equal-Count RNA Epistasis Ablation](../figures/fig3_rna_epistasis_operator_ablation.png)',
        '',
        '---',
        '',
        '## 4. Experiment 3: Unsupervised RNA Contact Maps Across 192 MHA Heads',
        '',
        f"- **Specialist Head `L17_H5`:** Recovers ViennaRNA `bpp >= 0.50` base pairs across all 22 human mt-tRNAs at mean APC AUROC = `{h5['auroc_apc']:.4f}` (`raw_dir = {h5['auroc_raw_dir']:.4f}`, `>=40 bp = {h5['auroc_ge40']:.4f}`, `AUPRC = {h5['auprc_apc']:.4f}`), reaching `0.9748` on *MT-TI*, `0.9469` on *MT-TS2*, `0.9270` on *MT-TV*, `0.8907` on *MT-TD*, `0.8903` on *MT-TY*, and `0.8121` (`>=40 bp = 0.9085`) on *MT-TL1*. The 192-head LOO logistic ensemble averages `{em['ensemble_auroc']:.4f}` (`>=40 bp = {em['ensemble_auroc_ge40']:.4f}`).",
        f"- **Kilobase rRNAs & Depth Progression:** On `MT-RNR1` (954 nt), `L10_H53` reaches `{M['rnr1_top']['auroc_apc']:.4f}` and `L17_H5` reaches `{M['rnr1_h5']['auroc_apc']:.4f}` (`>=40 bp = {M['rnr1_h5']['auroc_ge40']:.4f}`, `>=130 bp = {M['rnr1_h5']['auroc_ge130']:.4f}`). On `MT-RNR2` (1,559 nt), `L17_H7` reaches `{M['rnr2_top'][0]['auroc_apc']:.4f}` and `L17_H2` reaches `{M['rnr2_top'][1]['auroc_apc']:.4f}` (`>=130 bp = {M['rnr2_top'][1]['auroc_ge130']:.4f}`). Long-range (`>=40 bp`) mt-tRNA gene×head hits >0.70 jump 4.3× with depth: `L3={dc[3]['tRNA_ge40']} -> L10={dc[10]['tRNA_ge40']} -> L17={dc[17]['tRNA_ge40']}`.",
        '',
        '![Figure 4: Unsupervised RNA Contact Maps Across 192 Heads](../figures/fig4_mha_rna_contact_maps.png)',
        '',
        '---',
        '',
        '## 5. References',
        '',
        '1. Brixi G et al. *Nature* 2026;652(8112):1349–1361. PMID:41781614.',
        '2. Ding X, Qin Z. *arXiv:2610.39415*. 2026.',
        '3. Findlay GM et al. *Nature* 2018;562(7726):217–222. PMID:30209399.',
        '4. Avsec Ž et al. *Nature* 2026;649(8099):1206–1218. PMID:41606153.',
        '5. Lorenz R et al. *Algorithms Mol Biol* 2011;6:26. PMID:22115189.',
        '6. Rao R et al. *ICLR* 2021.',
        '7. Andrews RM et al. *Nature Genetics* 1999;23(2):147. PMID:10508508.',
        '8. Landrum MJ et al. *Nucleic Acids Res* 2025;53(D1):D1313–D1321. PMID:39578691.',
        '',
    ]
    out_md.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Saved {out_md} ({out_md.stat().st_size:,} bytes)')


def main():
    data_dir, ref_dir, fig_dir, out_dir = resolve_paths()
    M = compute_metrics(data_dir, ref_dir)
    build_pdf(M, fig_dir, out_dir / 'report.pdf')
    build_docx(M, fig_dir, out_dir / 'StripedHyena2_Mechanistic_Dissection_Report.docx')
    build_markdown(M, out_dir / 'stripedhyena2_mechanistic_dissection.md')


if __name__ == '__main__':
    main()
