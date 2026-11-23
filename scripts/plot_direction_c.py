#!/usr/bin/env python3
"""Generate Figures 1-4 for Direction C: Mechanistic Dissection of StripedHyena 2 (Evo 2 20B)."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def resolve_dirs():
    script_dir = Path(__file__).resolve().parent
    env_data = os.environ.get('SH2_DATA_DIR')
    env_fig = os.environ.get('SH2_FIG_DIR')
    if env_data and env_fig:
        return Path(env_data), Path(env_fig)
    if (script_dir.parent / 'datasets' / 'processed' / 'layer_probing_results.csv').exists():
        return script_dir.parent / 'datasets' / 'processed', script_dir.parent / 'figures'
    if (script_dir.parent / 'data' / 'layer_probing_results.csv').exists():
        return script_dir.parent / 'data', script_dir.parent / 'figures'
    raise FileNotFoundError('Could not locate Direction C processed CSVs relative to script.')


OP_COLORS = {'HCS': '#0f766e', 'HCM': '#2563eb', 'HCL': '#d97706', 'MHA': '#dc2626'}
OP_MARKERS = {'HCS': 'o', 'HCM': 's', 'HCL': '^', 'MHA': 'D'}
OP_ORDER = ['HCS', 'HCM', 'HCL', 'MHA']


def setup_rc():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'axes.edgecolor': '#334155',
        'axes.linewidth': 0.9,
        'axes.titlesize': 10.5,
        'axes.titleweight': 'bold',
        'axes.labelsize': 9.2,
        'xtick.labelsize': 8.0,
        'ytick.labelsize': 8.0,
        'legend.fontsize': 7.5,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_fig1(data_dir: Path, fig_dir: Path):
    df = pd.read_csv(data_dir / 'layer_probing_results.csv')
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2), gridspec_kw={'hspace': 0.36, 'wspace': 0.25})
    ax_list = axes.ravel()

    traj_specs = [
        ('TF_motifs', 'AUROC (5-Fold GroupKFold)', 'A. Local TF Motifs (6–19 bp, N=372)', (0.45, 1.03)),
        ('Splice_GT_AG', 'AUROC (5-Fold GroupKFold)', 'B. Canonical GT/AG vs. Decoy (N=924)', (0.48, 1.03)),
        ('Codon_Frame', 'Macro-F1 (5-Fold GroupKFold)', 'C. Coding Reading Frame 0/1/2 (N=565)', (0.30, 1.03)),
        ('Regional_1kb_GC', 'Ridge R² (Causal 1 kb Upstream GC)', 'D. Regional 1 kb GC% Regression (N=1,824)', (-0.15, 0.72)),
    ]

    for idx, (task, ylabel, title, ylim) in enumerate(traj_specs):
        ax = ax_list[idx]
        sub_r = df[(df['task'] == task) & (df['hook_type'] == 'residual_stream')].sort_values('layer')
        sub_m = df[(df['task'] == task) & (df['hook_type'] == 'mixer_update')].sort_values('layer')

        ax.plot(sub_r['layer'], sub_r['oof_score'], color='#475569', ls='--', lw=1.4, alpha=0.85, label='Residual stream (x_l)', zorder=2)
        ax.plot(sub_m['layer'], sub_m['oof_score'], color='#0f172a', ls='-', lw=1.8, alpha=0.90, label='Mixer update (Δh_l)', zorder=3)
        for op in OP_ORDER:
            mo = sub_m[sub_m['operator'] == op]
            ax.scatter(mo['layer'], mo['oof_score'], c=OP_COLORS[op], marker=OP_MARKERS[op], s=42, edgecolors='white', lw=0.5, zorder=5, label=f'{op} (Δh_l)')
        for al in (3, 10, 17):
            ax.axvline(al, color='#dc2626', ls=':', lw=0.8, alpha=0.35, zorder=1)
        ax.set_xticks([0, 3, 6, 10, 13, 17, 20, 23])
        ax.set_ylim(*ylim)
        ax.set_xlabel('StripedHyena 2 Block Index (0–23)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.22)
        if idx == 0:
            ax.legend(loc='lower right', fontsize=6.9, ncol=2, frameon=True, facecolor='white', framealpha=0.94)

    # Panel E: Operator Means on Pure Mixer Update across Local vs Splicing vs RNA Structure Tasks
    ax_e = ax_list[4]
    tasks_e = [
        ('TF_motifs', 'TF Motifs\n(AUROC)'),
        ('Splice_GT_AG', 'Splice GT/AG\n(AUROC)'),
        ('Exon_vs_Intron', 'Exon/Intron\n(AUROC)'),
        ('Codon_Frame', 'Codon Frame\n(Macro-F1)'),
        ('RNA_Stem3p_vs_Loop', 'RNA Stem 3′\n(AUROC)'),
    ]
    xi = np.arange(len(tasks_e))
    w = 0.19
    for j, op in enumerate(OP_ORDER):
        vals = [
            float(df[(df['task'] == tk) & (df['hook_type'] == 'mixer_update') & (df['operator'] == op)]['oof_score'].mean())
            for tk, _ in tasks_e
        ]
        bars = ax_e.bar(xi + (j - 1.5) * w, vals, width=w, color=OP_COLORS[op], edgecolor='#1e293b', lw=0.5, label=op)
        for rect in bars:
            h = rect.get_height()
            ax_e.text(rect.get_x() + rect.get_width() / 2.0, h + 0.010, f'{h:.2f}', ha='center', va='bottom', fontsize=6.2, rotation=90, color='#1e293b', fontweight='bold')
    ax_e.set_xticks(xi)
    ax_e.set_xticklabels([lbl for _, lbl in tasks_e], fontsize=7.5)
    ax_e.set_ylim(0.45, 1.12)
    ax_e.set_ylabel('Mean Probe Score (Pure Mixer Δh_l)')
    ax_e.set_title('E. Operator-Class Means Across Sequence Tasks')
    ax_e.legend(loc='upper right', ncol=2, fontsize=7.0, frameon=True, facecolor='white')
    ax_e.grid(axis='y', alpha=0.22)

    # Panel F: Residual-Stream Illusion vs Pure Mixer Reality (TF Motifs & Regional 1 kb GC)
    ax_f = ax_list[5]
    # Dual comparison: Left group TF_motifs Res vs Mix for HCS vs MHA; Right group Regional_1kb_GC Mix across HCS, HCM, HCL, MHA
    cats = ['TF: HCS\n(Res vs Mix)', 'TF: MHA\n(Res vs Mix)', '1kb GC:\nHCS (Mix)', '1kb GC:\nHCM (Mix)', '1kb GC:\nHCL (Mix)', '1kb GC:\nMHA (Mix)']
    x_f = np.arange(len(cats))
    tf_hcs_res = float(df[(df['task'] == 'TF_motifs') & (df['hook_type'] == 'residual_stream') & (df['operator'] == 'HCS')]['oof_score'].mean())
    tf_hcs_mix = float(df[(df['task'] == 'TF_motifs') & (df['hook_type'] == 'mixer_update') & (df['operator'] == 'HCS')]['oof_score'].mean())
    tf_mha_res = float(df[(df['task'] == 'TF_motifs') & (df['hook_type'] == 'residual_stream') & (df['operator'] == 'MHA')]['oof_score'].mean())
    tf_mha_mix = float(df[(df['task'] == 'TF_motifs') & (df['hook_type'] == 'mixer_update') & (df['operator'] == 'MHA')]['oof_score'].mean())
    gc_vals = [
        float(df[(df['task'] == 'Regional_1kb_GC') & (df['hook_type'] == 'mixer_update') & (df['operator'] == op)]['oof_score'].mean())
        for op in OP_ORDER
    ]

    ax_f.bar(0 - 0.20, tf_hcs_res, width=0.33, color='#99f6e4', edgecolor='#0f766e', lw=0.8, hatch='//', label='Residual (x_l)')
    ax_f.bar(0 + 0.20, tf_hcs_mix, width=0.33, color=OP_COLORS['HCS'], edgecolor='#0f172a', lw=0.8, label='Mixer Update (Δh_l)')
    ax_f.bar(1 - 0.20, tf_mha_res, width=0.33, color='#fecaca', edgecolor='#dc2626', lw=0.8, hatch='//')
    ax_f.bar(1 + 0.20, tf_mha_mix, width=0.33, color=OP_COLORS['MHA'], edgecolor='#0f172a', lw=0.8)

    ax_f.axvline(1.5, color='#64748b', ls='--', lw=1.0)
    ax_f.set_ylim(0.0, 1.16)
    ax_f.set_ylabel('TF Motif AUROC (Left)', color='#0f172a')
    for xv, val, align in [
        (0 - 0.05, tf_hcs_res, 'right'),
        (0 + 0.05, tf_hcs_mix, 'left'),
        (1 - 0.05, tf_mha_res, 'right'),
        (1 + 0.05, tf_mha_mix, 'left'),
    ]:
        ax_f.text(xv, val + 0.02, f'{val:.3f}', ha=align, va='bottom', fontsize=6.6, fontweight='bold')

    ax_fr = ax_f.twinx()
    for idx_g, (op, gv) in enumerate(zip(OP_ORDER, gc_vals)):
        xp = 2 + idx_g
        ax_fr.bar(xp, gv, width=0.56, color=OP_COLORS[op], edgecolor='#0f172a', lw=0.8, alpha=0.88)
        ax_fr.text(xp, max(gv, 0) + 0.015, f'{gv:+.3f}', ha='center', va='bottom', fontsize=6.9, fontweight='bold', color='#1e293b')
    ax_fr.axhline(0.0, color='#334155', lw=0.7)
    ax_fr.set_ylim(-0.08, 0.52)
    ax_fr.set_ylabel('1 kb GC% Ridge R² (Right)', color='#1e293b')
    ax_fr.spines['top'].set_visible(False)

    ax_f.set_xticks(x_f)
    ax_f.set_xticklabels(cats, fontsize=6.9)
    ax_f.set_title('F. Residual Confound vs. Operator Receptive Field')
    ax_f.legend(loc='upper right', fontsize=7.0, frameon=True, facecolor='white')
    ax_f.grid(axis='y', alpha=0.18)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.08)
    fig.savefig(fig_dir / 'fig1_operator_probing_division_of_labor.png')
    plt.close(fig)


def plot_fig2(data_dir: Path, fig_dir: Path):
    mt = pd.read_csv(data_dir / 'causal_ablation_24layers_mtdna.csv')
    br = pd.read_csv(data_dir / 'causal_ablation_24layers_brca1.csv')

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.2), gridspec_kw={'hspace': 0.36, 'wspace': 0.22})
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # Panel A: mtDNA non-trunk ΔNLL compartment comparison (L03-L20, L22; callout for trunk L0-2, L21, L23)
    mt_l = mt[mt['layer'] >= 0].copy()
    trunk_ids = {0, 1, 2, 21, 23}
    nontrunk = mt_l[~mt_l['layer'].isin(trunk_ids)].copy()
    xi = np.arange(len(nontrunk))
    w = 0.27
    ax_a.bar(xi - w, nontrunk['delta_nll_CDS'], width=w, color='#64748b', edgecolor='#1e293b', lw=0.5, label='CDS ΔNLL')
    ax_a.bar(xi, nontrunk['delta_nll_tRNA_loop'], width=w, color='#f59e0b', edgecolor='#78350f', lw=0.5, label='tRNA Loop ΔNLL')
    ax_a.bar(xi + w, nontrunk['delta_nll_tRNA_stem'], width=w, color='#0f766e', edgecolor='#064e3b', lw=0.5, label='tRNA Stem ΔNLL')

    for i_pos, (_, row) in enumerate(nontrunk.iterrows()):
        if row['operator'] == 'MHA':
            ax_a.axvspan(i_pos - 0.45, i_pos + 0.45, color='#fee2e2', alpha=0.45, zorder=0)
    l10_idx = list(nontrunk['layer']).index(10)
    l17_idx = list(nontrunk['layer']).index(17)
    l5_idx = list(nontrunk['layer']).index(5)
    ax_a.annotate('L10 MHA\nStem 4.2× CDS', xy=(l10_idx + w, float(nontrunk.iloc[l10_idx]['delta_nll_tRNA_stem'])),
                  xytext=(l10_idx + 1.1, 0.43), arrowprops=dict(arrowstyle='->', color='#991b1b', lw=1.0),
                  fontsize=7.2, fontweight='bold', bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='#dc2626'))
    ax_a.annotate('L17 MHA\nStem 2.6× CDS', xy=(l17_idx + w, float(nontrunk.iloc[l17_idx]['delta_nll_tRNA_stem'])),
                  xytext=(l17_idx - 2.6, 0.34), arrowprops=dict(arrowstyle='->', color='#991b1b', lw=1.0),
                  fontsize=7.2, fontweight='bold', bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='#dc2626'))
    ax_a.annotate('L5 HCM\nCDS 2.1× Stem', xy=(l5_idx - w, float(nontrunk.iloc[l5_idx]['delta_nll_CDS'])),
                  xytext=(l5_idx + 0.7, 0.50), arrowprops=dict(arrowstyle='->', color='#334155', lw=1.0),
                  fontsize=7.0, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#475569'))

    ax_a.set_xticks(xi)
    ax_a.set_xticklabels([f'L{int(r.layer)}\n{r.operator}' for _, r in nontrunk.iterrows()], fontsize=6.8)
    ax_a.set_ylim(0, 0.58)
    ax_a.set_ylabel('Next-Base Cross-Entropy Increase (ΔNLL, nats)')
    ax_a.set_title('A. mtDNA Compartment NLL Sensitivity (Non-Trunk Layers)')
    ax_a.legend(loc='upper right', fontsize=7.2, frameon=True, facecolor='white')
    ax_a.text(0.02, 0.96, 'Trunk layers (off-scale): L0–L2 (1.2–1.5), L21 (2.55–3.45), L23 (1.23–1.58)',
              transform=ax_a.transAxes, va='top', fontsize=6.9, color='#475569',
              bbox=dict(boxstyle='round,pad=0.2', fc='#f8fafc', ec='#cbd5e1'))
    ax_a.grid(axis='y', alpha=0.22)

    # Panel B: 24-layer mtDNA ClinVar ΔAUROC (CDS vs tRNA)
    layers = mt_l['layer'].values
    ax_b.axhline(0.0, color='#334155', lw=0.9)
    ax_b.plot(layers, mt_l['delta_auroc_CDS'], '-o', color='#475569', lw=1.5, ms=4.5, label='ClinVar CDS ΔAUROC (intact=0.910)')
    ax_b.plot(layers, mt_l['delta_auroc_tRNA'], '-s', color='#0f766e', lw=1.8, ms=5.0, label='ClinVar mt-tRNA ΔAUROC (intact=0.802)')
    for al in (3, 10, 17):
        ax_b.scatter([al], [mt_l.loc[mt_l['layer'] == al, 'delta_auroc_tRNA'].values[0]], c='#dc2626', s=65, zorder=6, edgecolors='#0f172a', lw=0.7)
    ax_b.annotate('L10 MHA: tRNA −0.139\n(CDS −0.052)', xy=(10, float(mt_l.loc[mt_l['layer'] == 10, 'delta_auroc_tRNA'].values[0])),
                  xytext=(5.2, -0.26), arrowprops=dict(arrowstyle='->', color='#991b1b', lw=1.0),
                  fontsize=7.2, fontweight='bold', bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='#dc2626'))
    ax_b.annotate('L17 MHA: tRNA −0.115\n(CDS −0.010)', xy=(17, float(mt_l.loc[mt_l['layer'] == 17, 'delta_auroc_tRNA'].values[0])),
                  xytext=(12.2, -0.25), arrowprops=dict(arrowstyle='->', color='#991b1b', lw=1.0),
                  fontsize=7.2, fontweight='bold', bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='#dc2626'))
    ax_b.set_xticks(range(24))
    ax_b.set_xticklabels([str(i) for i in range(24)], fontsize=7.2)
    ax_b.set_ylim(-0.66, 0.08)
    ax_b.set_xlabel('Mean-Ablated Block Index (0–23)')
    ax_b.set_ylabel('ClinVar Pathogenic vs. Benign ΔAUROC')
    ax_b.set_title('B. 24-Layer Causal Impact on ClinVar mtDNA Discrimination')
    ax_b.legend(loc='lower center', fontsize=7.2, frameon=True, facecolor='white')
    ax_b.grid(alpha=0.22)

    # Panel C & D: BRCA1 X15 & X18 non-trunk Modality Causal Ablation (Splicing vs AlphaMissense)
    br_l = br[(br['layer'] >= 0) & (~br['layer'].isin(trunk_ids))].copy()
    x_br = np.arange(len(br_l))
    for ax_sub, ex, spl_intact, mis_intact, title in [
        (ax_c, 'X15', float(br.loc[br['cond'] == 'intact', 'X15_auroc_Splicing'].values[0]),
         float(br.loc[br['cond'] == 'intact', 'X15_auroc_AlphaMissense'].values[0]),
         f'C. BRCA1 Exon 15 Modality Ablation (Spl={0.912:.3f}, Mis={0.931:.3f})'),
        (ax_d, 'X18', float(br.loc[br['cond'] == 'intact', 'X18_auroc_Splicing'].values[0]),
         float(br.loc[br['cond'] == 'intact', 'X18_auroc_AlphaMissense'].values[0]),
         f'D. BRCA1 Exon 18 (BRCT) Modality Ablation (Spl={0.833:.3f}, Mis={0.980:.3f})'),
    ]:
        ax_sub.axhline(0.0, color='#334155', lw=0.8)
        ax_sub.bar(x_br - 0.19, br_l[f'delta_{ex}_auroc_Splicing'], width=0.38, color='#059669', edgecolor='#064e3b', lw=0.5, label=f'AlphaGenome Splicing LOFs (intact={spl_intact:.3f})')
        ax_sub.bar(x_br + 0.19, br_l[f'delta_{ex}_auroc_AlphaMissense'], width=0.38, color='#2563eb', edgecolor='#1e3a8a', lw=0.5, label=f'AlphaMissense LOFs (intact={mis_intact:.3f})')
        for i_pos, (_, row) in enumerate(br_l.iterrows()):
            if row['operator'] == 'MHA':
                ax_sub.axvspan(i_pos - 0.45, i_pos + 0.45, color='#fee2e2', alpha=0.4, zorder=0)
        ax_sub.set_xticks(x_br)
        ax_sub.set_xticklabels([f'L{int(r.layer)}\n{r.operator}' for _, r in br_l.iterrows()], fontsize=6.8)
        ax_sub.set_ylabel('Modality LOF vs. FUNC ΔAUROC')
        ax_sub.set_title(title)
        ax_sub.legend(loc='lower left', fontsize=7.0, frameon=True, facecolor='white')
        ax_sub.grid(axis='y', alpha=0.22)

    ax_c.set_ylim(-0.48, 0.08)
    l15_i = list(br_l['layer']).index(15)
    l14_i = list(br_l['layer']).index(14)
    ax_c.annotate('L15 HCM:\nSplicing −0.401\nMissense −0.059', xy=(l15_i - 0.19, float(br_l.iloc[l15_i]['delta_X15_auroc_Splicing'])),
                  xytext=(l15_i - 4.8, -0.35), arrowprops=dict(arrowstyle='->', color='#064e3b', lw=1.0),
                  fontsize=7.2, fontweight='bold', bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='#059669'))

    ax_d.set_ylim(-0.82, 0.08)
    l17_i = list(br_l['layer']).index(17)
    ax_d.annotate('L17 MHA:\nMissense −0.722\nSplicing −0.266', xy=(l17_i + 0.19, float(br_l.iloc[l17_i]['delta_X18_auroc_AlphaMissense'])),
                  xytext=(l17_i - 5.2, -0.73), arrowprops=dict(arrowstyle='->', color='#1e3a8a', lw=1.0),
                  fontsize=7.2, fontweight='bold', bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='#dc2626'))
    ax_d.annotate('L14 HCS:\nMissense −0.581\nSplicing −0.000', xy=(l14_i + 0.19, float(br_l.iloc[l14_i]['delta_X18_auroc_AlphaMissense'])),
                  xytext=(l14_i - 5.2, -0.52), arrowprops=dict(arrowstyle='->', color='#1e3a8a', lw=1.0),
                  fontsize=7.2, fontweight='bold', bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='#2563eb'))

    fig.subplots_adjust(left=0.055, right=0.97, top=0.94, bottom=0.08)
    fig.savefig(fig_dir / 'fig2_causal_ablation_24layers.png')
    plt.close(fig)


def plot_fig3(data_dir: Path, fig_dir: Path):
    ep = pd.read_csv(data_dir / 'rna_epistasis_operator_ablation.csv')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.2), gridspec_kw={'wspace': 0.25})

    cond_order = ['Intact', 'Ablate_MHA', 'Ablate_HCL', 'Ablate_HCM', 'Ablate_HCS']
    cond_labels = ['Intact\n(0 ablated)', '−MHA\n[3,10,17]', '−HCL\n[2,9,16]', '−HCM\n[1,8,15]', '−HCS\n[0,7,14]']
    strata = [
        ('short_10_25', 'Short tRNA (10–23 bp)', '#0f766e'),
        ('medium_55_70', 'Medium tRNA Acceptor (55–65 bp)', '#2563eb'),
        ('long_130_600', 'Long rRNA (147–240 bp)', '#d97706'),
    ]

    for ax, metric, ylabel, title, ylim in [
        (ax1, 'epistasis_full_nats', 'Full-Sequence Epistatic Coupling E_full (nats)', 'A. Compensatory Watson–Crick Coupling (Δℒ_ij − [Δℒ_i + Δℒ_j])', (-4.8, 11.4)),
        (ax2, 'epistasis_sitej_nats', 'Causal Site-j Conditional Rescue E_site j (nats)', 'B. Directed Conditional Rescue at Downstream Base j', (-1.5, 11.4)),
    ]:
        xi = np.arange(len(cond_order))
        w = 0.24
        ax.axhline(0.0, color='#334155', lw=0.9)
        for j, (st_key, st_lbl, col) in enumerate(strata):
            means, sems = [], []
            for cnd in cond_order:
                vals = ep[(ep['condition'] == cnd) & (ep['stratum'] == st_key)][metric].values
                means.append(float(np.mean(vals)))
                sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
            bars = ax.bar(xi + (j - 1) * w, means, yerr=sems, width=w, color=col, edgecolor='#0f172a', lw=0.5, capsize=3, label=st_lbl)
            for idx_b, rect in enumerate(bars):
                if idx_b in (0, 1) and j in (0, 1):
                    h = means[idx_b]
                    off = (0.58 + sems[idx_b]) if (0.0 <= h < 1.0) else ((0.35 + sems[idx_b]) if h >= 0 else -(0.65 + sems[idx_b]))
                    ax.text(rect.get_x() + rect.get_width() / 2.0, h + off, f'{h:+.2f}', ha='center', va='center', fontsize=6.8, fontweight='bold', color='#0f172a')
        ax.set_xticks(xi)
        ax.set_xticklabels(cond_labels, fontsize=7.8)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=7.3, frameon=True, facecolor='white')
        ax.grid(axis='y', alpha=0.22)

    ax1.annotate('MHA ablation keeps 47% short (+2.51)\nbut erases 55–65 bp coupling (−0.50, p=0.0078)',
                 xy=(1.0, 2.51), xytext=(1.45, 5.9), arrowprops=dict(arrowstyle='->', color='#0f766e', lw=1.0),
                 fontsize=7.2, fontweight='bold', bbox=dict(boxstyle='round,pad=0.24', fc='white', ec='#0f766e'))
    ax2.annotate('Medium acceptor rescue:\nIntact +8.66 → −MHA +0.04 nats',
                 xy=(1.05, 0.12), xytext=(1.60, 5.4), arrowprops=dict(arrowstyle='->', color='#2563eb', lw=1.0),
                 fontsize=7.2, fontweight='bold', bbox=dict(boxstyle='round,pad=0.24', fc='white', ec='#2563eb'))

    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.14)
    fig.savefig(fig_dir / 'fig3_rna_epistasis_operator_ablation.png')
    plt.close(fig)


def plot_fig4(data_dir: Path, fig_dir: Path):
    hd = pd.read_csv(data_dir / 'mha_rna_contact_heads.csv')
    en = pd.read_csv(data_dir / 'mha_rna_contact_ensemble.csv')
    npz = np.load(data_dir / 'top_rna_contact_maps.npz')

    fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.5), gridspec_kw={'hspace': 0.34, 'wspace': 0.26})
    (ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axes

    # Panel A: All 192 heads unsupervised tRNA contact AUROC (overall vs >=40bp)
    trna_h = hd[hd['feature'] == 'tRNA'].groupby(['layer', 'head', 'head_name'])[['auroc_apc', 'auroc_ge40']].mean().reset_index()
    layer_x = {3: 0, 10: 1, 17: 2}
    rng = np.random.default_rng(7)
    for l_val, col in [(3, '#475569'), (10, '#2563eb'), (17, '#dc2626')]:
        sub = trna_h[trna_h['layer'] == l_val]
        jit = rng.uniform(-0.14, 0.14, size=len(sub))
        ax_a.scatter(layer_x[l_val] - 0.17 + jit, sub['auroc_apc'], c=col, alpha=0.55, s=18, marker='o')
        ax_a.scatter(layer_x[l_val] + 0.17 + jit, sub['auroc_ge40'], c=col, alpha=0.75, s=22, marker='^', edgecolors='#0f172a', lw=0.35)

    h5 = trna_h[trna_h['head_name'] == 'L17_H5'].iloc[0]
    ax_a.scatter([2 - 0.17, 2 + 0.17], [h5['auroc_apc'], h5['auroc_ge40']], c='#facc15', s=95, marker='*', edgecolors='#0f172a', lw=0.7, zorder=10)
    ax_a.annotate(f'L17_H5 Specialist\nAll: {h5["auroc_apc"]:.3f} | ≥40bp: {h5["auroc_ge40"]:.3f}',
                  xy=(2 + 0.17, h5['auroc_ge40']), xytext=(0.45, 0.875),
                  arrowprops=dict(arrowstyle='->', color='#0f172a', lw=1.0),
                  fontsize=7.3, fontweight='bold', bbox=dict(boxstyle='round,pad=0.25', fc='#fef9c3', ec='#ca8a04'))
    ax_a.axhline(0.5, color='#94a3b8', ls='--', lw=0.9)
    ax_a.set_xticks([0, 1, 2])
    ax_a.set_xticklabels(['Layer 3\n(64 Heads)', 'Layer 10\n(64 Heads)', 'Layer 17\n(64 Heads)'])
    ax_a.set_ylim(0.28, 0.96)
    ax_a.set_ylabel('Mean 22 mt-tRNA Contact AUROC')
    ax_a.set_title('A. All 192 MHA Heads: Base-Pair Contact Discovery')
    handles_a = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#475569', markersize=6, label='All pairs |i−j| ≥ 4'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#475569', markeredgecolor='#0f172a', markersize=6, label='Long-range ≥ 40 bp'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#facc15', markeredgecolor='#0f172a', markersize=9, label='L17_H5 specialist'),
    ]
    ax_a.legend(handles=handles_a, loc='lower left', fontsize=6.9, frameon=True, facecolor='white')
    ax_a.grid(axis='y', alpha=0.22)

    # Panel B: Per-gene AUROC L17_H5 vs 192-head LOO Ensemble (sorted by L17_H5)
    h5_genes = hd[hd['head_name'] == 'L17_H5'][['gene', 'feature', 'auroc_apc', 'auroc_ge40']].merge(
        en[['gene', 'ensemble_auroc']], on='gene', how='left'
    ).sort_values('auroc_apc', ascending=True)
    yg = np.arange(len(h5_genes))
    ax_b.barh(yg - 0.18, h5_genes['auroc_apc'], height=0.35, color='#dc2626', alpha=0.85, label='Single Head L17_H5')
    ax_b.barh(yg + 0.18, h5_genes['ensemble_auroc'].fillna(np.nan), height=0.35, color='#2563eb', alpha=0.80, label='192-Head LOO Ens.')
    ax_b.axvline(0.5, color='#64748b', ls=':', lw=0.9)
    ax_b.set_yticks(yg)
    ax_b.set_yticklabels([f'{r.gene}{"*" if r.feature=="rRNA" else ""}' for _, r in h5_genes.iterrows()], fontsize=6.4)
    ax_b.set_xlim(0.42, 1.03)
    ax_b.set_xlabel('Unsupervised Contact AUROC (|i−j| ≥ 4)')
    ax_b.set_title('B. Per-Gene Contact Recovery (* = 12S/16S rRNA)')
    ax_b.legend(loc='lower right', fontsize=6.9, frameon=True, facecolor='white')
    ax_b.grid(axis='x', alpha=0.22)

    # Helper for split triangle contact maps (lower = RNAfold bpp>=0.5, upper = Pred)
    def plot_split_contact(ax, bpp_mat, pred_mat, title, sub_label, thresh_pct=96):
        bpp_mat = np.asarray(bpp_mat, dtype=np.float32)
        pred_mat = np.asarray(pred_mat, dtype=np.float32)
        N = bpp_mat.shape[0]
        # Background heatmap: upper triangle = clipped z-scored pred_mat
        up_vals = pred_mat[np.triu_indices(N, k=4)]
        lo_q, hi_q = np.percentile(up_vals, [5, 99.5])
        norm_pred = np.clip((pred_mat - lo_q) / max(1e-6, hi_q - lo_q), 0, 1)
        canvas = np.full((N, N), np.nan)
        iu4, ju4 = np.triu_indices(N, k=4)
        canvas[iu4, ju4] = norm_pred[iu4, ju4]
        ax.imshow(canvas, cmap='Blues', origin='upper', vmin=0.0, vmax=1.0, aspect='equal')
        # Lower triangle true RNAfold stems in crisp crimson
        ti, tj = np.where(np.tril(bpp_mat >= 0.50, k=-4))
        ax.scatter(tj, ti, s=max(1.5, 420.0 / N), c='#dc2626', marker='s', alpha=0.92, label='RNAfold stem (bpp≥0.5)')
        # Overlay top predicted contacts in upper triangle as gold stars/dots
        cut = np.percentile(up_vals, thresh_pct)
        pi, pj = np.where((np.triu(pred_mat >= cut, k=4)) & (np.triu(bpp_mat >= 0.50, k=4)))
        ax.scatter(pj, pi, s=max(2.0, 520.0 / N), c='#f59e0b', marker='o', edgecolors='#0f172a', lw=0.35, label='Matched top pred')
        ax.plot([0, N - 1], [0, N - 1], color='#334155', lw=0.8, ls='--')
        ax.set_xlim(-0.5, N - 0.5)
        ax.set_ylim(N - 0.5, -0.5)
        ax.set_xlabel('3′ Nucleotide Index j')
        ax.set_ylabel('5′ Nucleotide Index i')
        ax.set_title(title)
        ax.text(0.03, 0.05, sub_label, transform=ax.transAxes, fontsize=7.0, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#64748b', alpha=0.94))

    plot_split_contact(ax_c, npz['MT_TL1_bpp'], npz['MT_TL1_top1_L17_H5'],
                       'C. MT-TL1 (75 nt): L17_H5 APC vs. Truth',
                       'AUROC=0.812 | ≥40bp=0.909', thresh_pct=93)
    ax_c.legend(loc='upper right', fontsize=6.5, frameon=True, facecolor='white')

    plot_split_contact(ax_d, npz['MT_TL1_bpp'], npz['MT_TL1_loo_ensemble'],
                       'D. MT-TL1: 192-Head LOO Ensemble',
                       'Unsupervised LR ensemble AUROC=0.734 avg', thresh_pct=93)

    plot_split_contact(ax_e, npz['MT_RNR1_bpp'], npz['MT_RNR1_top1_L17_H5'],
                       'E. 12S rRNA (954 nt): L17_H5 vs. Truth',
                       'AUROC=0.804 | ≥40bp=0.833 | ≥130bp=0.776', thresh_pct=98)

    # Panel F: Long-range specialist head emergence across Layers 3 -> 10 -> 17
    trna_lr_hits = [int((hd[(hd['feature'] == 'tRNA') & (hd['layer'] == l)]['auroc_ge40'] > 0.70).sum()) for l in (3, 10, 17)]
    trna_all_hits = [int((hd[(hd['feature'] == 'tRNA') & (hd['layer'] == l)]['auroc_apc'] > 0.70).sum()) for l in (3, 10, 17)]
    rrna_all_hits = [int((hd[(hd['feature'] == 'rRNA') & (hd['layer'] == l)]['auroc_apc'] > 0.70).sum()) for l in (3, 10, 17)]
    xf = np.arange(3)
    wf = 0.25
    b_f1 = ax_f.bar(xf - wf, trna_all_hits, width=wf, color='#0f766e', edgecolor='#0f172a', lw=0.6, label='tRNA All Pairs >0.70 AUROC')
    b_f2 = ax_f.bar(xf, trna_lr_hits, width=wf, color='#dc2626', edgecolor='#0f172a', lw=0.6, label='tRNA ≥40 bp Pairs >0.70 AUROC')
    b_f3 = ax_f.bar(xf + wf, rrna_all_hits, width=wf, color='#2563eb', edgecolor='#0f172a', lw=0.6, label='12S/16S rRNA >0.70 AUROC')
    for bars in (b_f1, b_f2, b_f3):
        for rect in bars:
            h = int(rect.get_height())
            ax_f.text(rect.get_x() + rect.get_width() / 2.0, h + 3.5, f'{h}', ha='center', va='bottom', fontsize=7.2, fontweight='bold')
    ax_f.set_xticks(xf)
    ax_f.set_xticklabels(['Layer 3\n(Early MHA)', 'Layer 10\n(Mid MHA)', 'Layer 17\n(Deep MHA)'])
    ax_f.set_ylim(0, 212)
    ax_f.set_ylabel('Count of Gene × Head Pairs (AUROC > 0.70)')
    ax_f.set_title('F. Long-Range Contact Emergence Across MHA Depth')
    ax_f.legend(loc='upper left', fontsize=6.8, frameon=True, facecolor='white')
    ax_f.grid(axis='y', alpha=0.22)

    fig.subplots_adjust(left=0.05, right=0.97, top=0.94, bottom=0.08)
    fig.savefig(fig_dir / 'fig4_mha_rna_contact_maps.png')
    plt.close(fig)


def main():
    setup_rc()
    data_dir, fig_dir = resolve_dirs()
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_fig1(data_dir, fig_dir)
    print('Saved fig1_operator_probing_division_of_labor.png')
    plot_fig2(data_dir, fig_dir)
    print('Saved fig2_causal_ablation_24layers.png')
    plot_fig3(data_dir, fig_dir)
    print('Saved fig3_rna_epistasis_operator_ablation.png')
    plot_fig4(data_dir, fig_dir)
    print('Saved fig4_mha_rna_contact_maps.png')


if __name__ == '__main__':
    main()
