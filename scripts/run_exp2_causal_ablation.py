#!/usr/bin/env python3
"""Experiment 2: 24-layer causal mean-ablation (mtDNA & BRCA1) + equal-count 3-layer RNA epistasis ablation."""

import argparse
import gzip
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import gc
import os as _os
_os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
from Bio import SeqIO
from sklearn.metrics import roc_auc_score, average_precision_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if os.environ.get('SH2_LOADER_DIR'):
    sys.path.insert(0, os.environ['SH2_LOADER_DIR'])
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import evo2_mmaemu_loader as loader  # noqa: E402

DATA_DIR = os.environ.get('SH2_REF_DATA_DIR', os.path.join(BASE_DIR, 'datasets', 'raw_reference'))
if not os.path.exists(DATA_DIR) and os.path.isdir(os.path.join(BASE_DIR, 'data')):
    DATA_DIR = os.path.join(BASE_DIR, 'data')
RES_DIR = os.environ.get('SH2_ATLAS_RES_DIR', DATA_DIR)
OUT_DIR = os.environ.get('SH2_OUT_DIR', os.path.join(BASE_DIR, 'datasets', 'processed'))
if not os.path.exists(OUT_DIR) and os.path.isdir(os.path.join(BASE_DIR, 'data')):
    OUT_DIR = os.path.join(BASE_DIR, 'data')
RC_DNA = str.maketrans('ACGTN', 'TGCAN')
COMP = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
ATTN_LAYERS = {3, 10, 17}
OP_MAP = {
    **{i: 'HCS' for i in [0, 4, 7, 11, 14, 18, 21]},
    **{i: 'HCM' for i in [1, 5, 8, 12, 15, 19, 22]},
    **{i: 'HCL' for i in [2, 6, 9, 13, 16, 20, 23]},
    **{i: 'MHA' for i in [3, 10, 17]},
}


def revcomp(s: str) -> str:
    return s.translate(RC_DNA)[::-1]


def run_pass_with_ablation(evo, seq_str: str, ablate_layers):
    """Run 1 forward pass with optional mean-ablation on mixer outputs, returning log_softmax float32 (L, 512)."""
    blocks = evo.model.blocks
    handles = []

    def ablate_fn(mod, inp, out):
        if isinstance(out, tuple):
            t = out[0]
            m = t.mean(dim=1, keepdim=True).expand_as(t)
            return (m, *out[1:])
        return out.mean(dim=1, keepdim=True).expand_as(out)

    for l in ablate_layers:
        mixer = blocks[l].inner_mha_cls if l in ATTN_LAYERS else blocks[l].out_filter_dense
        handles.append(mixer.register_forward_hook(ablate_fn))

    dev0 = evo.model.block_idx_to_device[0]
    ids = torch.tensor(evo.tokenizer.tokenize(seq_str), dtype=torch.long, device=dev0).unsqueeze(0)
    try:
        with torch.inference_mode():
            logits, _ = evo.model(ids)
            lp = torch.log_softmax(logits[0].float(), dim=-1).cpu().numpy()
            del logits
    finally:
        for h in handles:
            h.remove()
        gc.collect()
        for dev_i in range(torch.cuda.device_count()):
            with torch.cuda.device(dev_i):
                torch.cuda.empty_cache()
    return lp, ids[0].cpu().numpy()


def run_part_mtdna(evo):
    print('=== Exp 2A: 24-Layer Causal Mean-Ablation on 16,569 bp rCRS mtDNA ===', flush=True)
    rec = SeqIO.read(f'{DATA_DIR}/NC_012920.1.gb', 'genbank')
    rcrs_h = str(rec.seq).upper()
    L = len(rcrs_h)

    atlas = pd.read_csv(f'{RES_DIR}/mitochondrial_organelle_pathogenicity_atlas.csv')
    rna_path = os.path.join(OUT_DIR, 'rna_fold_structures.json')
    if not os.path.exists(rna_path):
        rna_path = os.path.join(DATA_DIR, 'rna_fold_structures.json')
    with open(rna_path, 'r', encoding='utf-8') as f:
        rnas = json.load(f)

    # Map every rCRS 1-based pos -> compartment label
    pos_feat = atlas[['pos', 'feature']].drop_duplicates('pos').set_index('pos')['feature'].to_dict()
    paired_pos = set()
    for gr in rnas:
        paired_pos.update(gr['fivep_rcrs_pos'])
        paired_pos.update(gr['threep_rcrs_pos'])

    comp_indices = {c: [] for c in ['CDS', 'tRNA_stem', 'tRNA_loop', 'rRNA_stem', 'rRNA_loop', 'D-loop']}
    for p in range(2, L + 1):
        if p == 3107:
            continue
        feat = pos_feat.get(p, 'Intergenic')
        idx = p - 1
        if feat == 'CDS':
            comp_indices['CDS'].append(idx)
        elif feat == 'tRNA':
            comp_indices['tRNA_stem' if p in paired_pos else 'tRNA_loop'].append(idx)
        elif feat == 'rRNA':
            comp_indices['rRNA_stem' if p in paired_pos else 'rRNA_loop'].append(idx)
        elif feat == 'D-loop':
            comp_indices['D-loop'].append(idx)
    comp_indices = {k: np.array(v, dtype=int) for k, v in comp_indices.items()}

    path_l = {'Pathogenic', 'Likely pathogenic', 'Pathogenic/Likely pathogenic', 'Likely pathogenic; drug response', 'Pathogenic; drug response'}
    ben_l = {'Benign', 'Likely benign', 'Benign/Likely benign'}
    ev = atlas[atlas['clinsig'].isin(path_l | ben_l) & (atlas['pos'] >= 2)].copy()
    ev['y'] = ev['clinsig'].isin(path_l).astype(int)
    tok = evo.tokenizer
    tok_id = {b: int(tok.tokenize(b)[0]) for b in 'ACGT'}
    ev_idx_m1 = (ev['pos'].values.astype(int) - 2)
    ev_ref_tids = np.array([tok_id[b] for b in ev['ref'].values], dtype=int)
    ev_alt_tids = np.array([tok_id[b] for b in ev['alt'].values], dtype=int)
    ev_y = ev['y'].values
    ev_feat = ev['feature'].values

    ckpt_path = f'{OUT_DIR}/ckpt_exp2_mtdna.npz'
    cache = {}
    if os.path.exists(ckpt_path):
        z = np.load(ckpt_path, allow_pickle=True)
        cache = dict(z['results'].item())
        print(f'Resumed {len(cache)}/25 mtDNA ablation conditions.', flush=True)

    conds = [('intact', [])] + [(f'L{l:02d}', [l]) for l in range(24)]
    for cname, ab_list in conds:
        if cname in cache:
            continue
        t0 = time.time()
        lp, ids_np = run_pass_with_ablation(evo, rcrs_h, ab_list)
        # Next-base NLL per position: nll[idx] = -lp[idx-1, ids_np[idx]]
        nll_all = np.zeros(L, dtype=np.float64)
        nll_all[1:] = -lp[np.arange(L - 1), ids_np[1:]]
        rec_out = {'cond': cname, 'layer': -1 if cname == 'intact' else int(cname[1:]), 'operator': 'Intact' if cname == 'intact' else OP_MAP[int(cname[1:])]}
        valid_mask = np.ones(L, dtype=bool)
        valid_mask[0] = False
        valid_mask[3106] = False
        rec_out['mean_nll_genome'] = float(nll_all[valid_mask].mean())
        for ck, idxs in comp_indices.items():
            rec_out[f'nll_{ck}'] = float(nll_all[idxs].mean())

        dfwd = lp[ev_idx_m1, ev_alt_tids] - lp[ev_idx_m1, ev_ref_tids]
        rec_out['auroc_all_mtDNA'] = float(roc_auc_score(ev_y, -dfwd))
        rec_out['auprc_all_mtDNA'] = float(average_precision_score(ev_y, -dfwd))
        for feat in ['CDS', 'tRNA', 'rRNA']:
            m = ev_feat == feat
            rec_out[f'auroc_{feat}'] = float(roc_auc_score(ev_y[m], -dfwd[m]))
        cache[cname] = rec_out
        np.savez(ckpt_path, results=cache)
        print(f'  mtDNA {cname:6s} ({rec_out["operator"]:6s}): {time.time() - t0:.1f}s | AUROC all={rec_out["auroc_all_mtDNA"]:.4f} CDS={rec_out["auroc_CDS"]:.4f} tRNA={rec_out["auroc_tRNA"]:.4f} | NLL_genome={rec_out["mean_nll_genome"]:.4f}', flush=True)

    df = pd.DataFrame([cache[c[0]] for c in conds])
    base = df[df['cond'] == 'intact'].iloc[0]
    for col in list(df.columns):
        if col.startswith('nll_') or col == 'mean_nll_genome':
            df['delta_' + col] = df[col] - base[col]
        elif col.startswith('auroc_') or col.startswith('auprc_'):
            df['delta_' + col] = df[col] - base[col]
    out_csv = f'{OUT_DIR}/causal_ablation_24layers_mtdna.csv'
    df.to_csv(out_csv, index=False)
    print(f'Saved {out_csv}', flush=True)


def run_part_brca1(evo):
    print('=== Exp 2B: 24-Layer Causal Mean-Ablation on BRCA1 X15 & X18 (4,096 bp) ===', flush=True)
    rec = SeqIO.read(gzip.open(f'{DATA_DIR}/GRCh37.p13_chr17.fna.gz', 'rt'), 'fasta')
    chr17 = str(rec.seq).upper()
    ag = pd.read_csv(f'{DATA_DIR}/brca1_all3893_evo2_vs_alphagenome_avi.csv')
    tok_id = {b: int(evo.tokenizer.tokenize(b)[0]) for b in 'ACGT'}

    win_len = 4096
    half = win_len // 2
    exon_defs = []
    for ex in ['X15', 'X18']:
        sub = ag[(ag['experiment'] == ex) & ag['class'].isin(['FUNC', 'LOF'])].copy()
        mid = int((sub['pos'].min() + sub['pos'].max()) // 2)
        g_top_1 = mid + half
        g_bot_1 = g_top_1 - win_len + 1
        wseq = revcomp(chr17[g_bot_1 - 1 : g_top_1])
        assert len(wseq) == win_len
        idx_c = g_top_1 - sub['pos'].values.astype(int)
        assert np.all((idx_c >= 100) & (idx_c < win_len - 100))
        for p, r, ic in zip(sub['pos'].values, sub['ref'].values, idx_c):
            assert wseq[ic] == COMP[r], f'Ref mismatch {ex} {p}: {wseq[ic]} vs {COMP[r]}'
        sub['idx_c'] = idx_c
        sub['ref_c_tid'] = [tok_id[COMP[r]] for r in sub['ref'].values]
        sub['alt_c_tid'] = [tok_id[COMP[a]] for a in sub['alt'].values]
        sub['y'] = (sub['class'] == 'LOF').astype(int).values
        exon_defs.append((ex, wseq, sub))

    ckpt_path = f'{OUT_DIR}/ckpt_exp2_brca1.npz'
    cache = {}
    if os.path.exists(ckpt_path):
        z = np.load(ckpt_path, allow_pickle=True)
        cache = dict(z['results'].item())
        print(f'Resumed {len(cache)}/25 BRCA1 ablation conditions.', flush=True)

    conds = [('intact', [])] + [(f'L{l:02d}', [l]) for l in range(24)]
    for cname, ab_list in conds:
        if cname in cache:
            continue
        t0 = time.time()
        rec_out = {'cond': cname, 'layer': -1 if cname == 'intact' else int(cname[1:]), 'operator': 'Intact' if cname == 'intact' else OP_MAP[int(cname[1:])]}
        for ex, wseq, sub in exon_defs:
            lp, _ = run_pass_with_ablation(evo, wseq, ab_list)
            idx_m1 = sub['idx_c'].values - 1
            drev = lp[idx_m1, sub['alt_c_tid'].values] - lp[idx_m1, sub['ref_c_tid'].values]
            y = sub['y'].values
            rec_out[f'{ex}_auroc_all'] = float(roc_auc_score(y, -drev))
            # AlphaGenome LOF modality vs all exon FUNC
            is_func = sub['class'].values == 'FUNC'
            for mod_lbl, mod_val in [('Splicing', 'Splicing'), ('AlphaMissense', 'AlphaMissense'), ('Stop', 'Protein Termination')]:
                m = is_func | ((sub['class'].values == 'LOF') & (sub['top_modality'].values == mod_val))
                if m.sum() > 0 and y[m].sum() >= 3:
                    rec_out[f'{ex}_auroc_{mod_lbl}'] = float(roc_auc_score(y[m], -drev[m]))
        cache[cname] = rec_out
        np.savez(ckpt_path, results=cache)
        print(
            f'  BRCA1 {cname:6s} ({rec_out["operator"]:6s}): {time.time() - t0:.1f}s | '
            f'X15={rec_out["X15_auroc_all"]:.4f} (Spl={rec_out["X15_auroc_Splicing"]:.4f}, Mis={rec_out["X15_auroc_AlphaMissense"]:.4f}) | '
            f'X18={rec_out["X18_auroc_all"]:.4f} (Spl={rec_out["X18_auroc_Splicing"]:.4f}, Mis={rec_out["X18_auroc_AlphaMissense"]:.4f})',
            flush=True,
        )

    df = pd.DataFrame([cache[c[0]] for c in conds])
    base = df[df['cond'] == 'intact'].iloc[0]
    for col in list(df.columns):
        if '_auroc_' in col:
            df['delta_' + col] = df[col] - base[col]
    out_csv = f'{OUT_DIR}/causal_ablation_24layers_brca1.csv'
    df.to_csv(out_csv, index=False)
    print(f'Saved {out_csv}', flush=True)


def run_part_epistasis(evo):
    print('=== Exp 2C: Equal-Count (3-Layer) Operator Ablation on 24 RNA Epistasis Stem Pairs (L=768 bp) ===', flush=True)
    ep_path = os.path.join(OUT_DIR, 'rna_stem_epistasis_pairs.csv')
    if not os.path.exists(ep_path):
        ep_path = os.path.join(DATA_DIR, 'rna_stem_epistasis_pairs.csv')
    ep = pd.read_csv(ep_path)
    tok_id = {b: int(evo.tokenizer.tokenize(b)[0]) for b in 'ACGT'}

    conditions = [
        ('Intact', []),
        ('Ablate_MHA', [3, 10, 17]),
        ('Ablate_HCL', [2, 9, 16]),
        ('Ablate_HCM', [1, 8, 15]),
        ('Ablate_HCS', [0, 7, 14]),
    ]

    # Pre-construct 768 bp trimmed sequences [128:896] and verify exact nucleotide alignment
    items = []
    for _, row in ep.iterrows():
        seq1024 = row['wt_seq']
        seq768 = seq1024[128:896]
        wi = int(row['wi']) - 128
        wj = int(row['wj']) - 128
        assert 200 <= wi < wj < 768 - 200
        wt_i, wt_j = row['wt_pair'].split('-')
        alt_i, alt_j = row['alt_i'], row['alt_j']
        assert seq768[wi] == wt_i and seq768[wj] == wt_j
        s_wt = seq768
        s_i = seq768[:wi] + alt_i + seq768[wi + 1:]
        s_j = seq768[:wj] + alt_j + seq768[wj + 1:]
        s_ij = seq768[:wi] + alt_i + seq768[wi + 1:wj] + alt_j + seq768[wj + 1:]
        items.append({
            'pair_id': row['pair_id'], 'stratum': row['stratum'], 'gene': row['gene'],
            'dist': int(row['dist']), 'bpp': float(row['bpp']), 'wi': wi, 'wj': wj,
            'wt_i': wt_i, 'wt_j': wt_j, 'alt_i': alt_i, 'alt_j': alt_j,
            'seqs': {'wt': s_wt, 'mut_i': s_i, 'mut_j': s_j, 'double': s_ij},
        })

    ckpt_path = f'{OUT_DIR}/ckpt_exp2_epistasis.npz'
    cache = {}
    if os.path.exists(ckpt_path):
        z = np.load(ckpt_path, allow_pickle=True)
        cache = dict(z['results'].item())
        print(f'Resumed {len(cache)}/{len(conditions)} conditions.', flush=True)

    rows_out = []
    for cond_name, ab_layers in conditions:
        if cond_name in cache:
            rows_out.extend(cache[cond_name])
            continue
        t_cond = time.time()
        cond_rows = []
        for idx_it, it in enumerate(items):
            ll_map = {}
            sitej_ratio = {}
            wi, wj = it['wi'], it['wj']
            ti_wt, ti_alt = tok_id[it['wt_i']], tok_id[it['alt_i']]
            tj_wt, tj_alt = tok_id[it['wt_j']], tok_id[it['alt_j']]
            for gname, sq in it['seqs'].items():
                lp, ids_np = run_pass_with_ablation(evo, sq, ab_layers)
                L = len(sq)
                ll_sum = float(np.sum(lp[np.arange(L - 1), ids_np[1:]]))
                ll_map[gname] = ll_sum
                if gname in ('wt', 'mut_i'):
                    # log P(alt_j | prefix) - log P(wt_j | prefix) at causal position wj
                    sitej_ratio[gname] = float(lp[wj - 1, tj_alt] - lp[wj - 1, tj_wt])
            d_i = ll_map['mut_i'] - ll_map['wt']
            d_j = ll_map['mut_j'] - ll_map['wt']
            d_ij = ll_map['double'] - ll_map['wt']
            epistasis_full = float(d_ij - (d_i + d_j))
            epistasis_sitej = float(sitej_ratio['mut_i'] - sitej_ratio['wt'])
            cond_rows.append({
                'condition': cond_name, 'n_ablated': len(ab_layers),
                'pair_id': it['pair_id'], 'stratum': it['stratum'], 'gene': it['gene'],
                'dist': it['dist'], 'bpp': it['bpp'],
                'delta_ll_i': d_i, 'delta_ll_j': d_j, 'delta_ll_double': d_ij,
                'epistasis_full_nats': epistasis_full,
                'sitej_logratio_wt': sitej_ratio['wt'], 'sitej_logratio_muti': sitej_ratio['mut_i'],
                'epistasis_sitej_nats': epistasis_sitej,
            })
        cache[cond_name] = cond_rows
        np.savez(ckpt_path, results=cache)
        rows_out.extend(cond_rows)
        cdf = pd.DataFrame(cond_rows)
        means_e = cdf.groupby('stratum')['epistasis_full_nats'].mean().round(3).to_dict()
        means_j = cdf.groupby('stratum')['epistasis_sitej_nats'].mean().round(3).to_dict()
        print(f'  {cond_name:11s} done in {(time.time()-t_cond)/60:.1f}m | E_full={means_e} | E_sitej={means_j}', flush=True)

    df_ep = pd.DataFrame(rows_out)
    out_csv = f'{OUT_DIR}/rna_epistasis_operator_ablation.csv'
    df_ep.to_csv(out_csv, index=False)
    print(f'Saved {out_csv}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=['all', 'mtdna_brca1', 'epistasis'], default='all')
    args = parser.parse_args()
    evo, _, _ = loader.load_evo2_20b(mode_spec='mmaemu_all', profile='hopper-te')
    if args.part in ('all', 'mtdna_brca1'):
        run_part_mtdna(evo)
        run_part_brca1(evo)
    if args.part in ('all', 'epistasis'):
        run_part_epistasis(evo)


if __name__ == '__main__':
    main()
