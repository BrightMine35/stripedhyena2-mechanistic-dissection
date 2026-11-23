#!/usr/bin/env python3
"""Experiment 1: 24-layer residual-stream vs. pure-mixer linear probing on Evo 2 20B (mmaemu_all)."""

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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

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
OUT_DIR = os.environ.get('SH2_OUT_DIR', os.path.join(BASE_DIR, 'datasets', 'processed'))
if not os.path.exists(OUT_DIR) and os.path.isdir(os.path.join(BASE_DIR, 'data')):
    OUT_DIR = os.path.join(BASE_DIR, 'data')
RC_DNA = str.maketrans('ACGTN', 'TGCAN')
ATTN_LAYERS = {3, 10, 17}
OP_MAP = {
    **{i: 'HCS' for i in [0, 4, 7, 11, 14, 18, 21]},
    **{i: 'HCM' for i in [1, 5, 8, 12, 15, 19, 22]},
    **{i: 'HCL' for i in [2, 6, 9, 13, 16, 20, 23]},
    **{i: 'MHA' for i in [3, 10, 17]},
}


def run_hooked_windows(evo, sequences, target_indices_per_seq):
    """Run forward passes and capture (N_tokens, 8192) float32 CPU arrays for all 24 res + mix hooks."""
    blocks = evo.model.blocks
    n_layers = len(blocks)
    res_buf = {l: [] for l in range(n_layers)}
    mix_buf = {l: [] for l in range(n_layers)}
    dev0 = evo.model.block_idx_to_device[0]

    for seq_idx, (seq_str, tok_indices) in enumerate(zip(sequences, target_indices_per_seq)):
        t0 = time.time()
        tok_np = np.array(tok_indices, dtype=np.int64)
        idx_cache = {}

        def make_hook(store_dict, layer_id):
            def fn(mod, inp, out):
                ten = out[0] if isinstance(out, tuple) else out
                dev = ten.device
                if dev not in idx_cache:
                    idx_cache[dev] = torch.from_numpy(tok_np).to(dev)
                sl = ten[0, idx_cache[dev], :].detach().float().cpu().numpy()
                store_dict[layer_id].append(sl)
            return fn

        handles = []
        for l in range(n_layers):
            handles.append(blocks[l].register_forward_hook(make_hook(res_buf, l)))
            mixer = blocks[l].inner_mha_cls if l in ATTN_LAYERS else blocks[l].out_filter_dense
            handles.append(mixer.register_forward_hook(make_hook(mix_buf, l)))

        ids = torch.tensor(evo.tokenizer.tokenize(seq_str), dtype=torch.long, device=dev0).unsqueeze(0)
        try:
            with torch.inference_mode():
                logits, _ = evo.model(ids)
                del logits
        finally:
            for h in handles:
                h.remove()
            idx_cache.clear()
            gc.collect()
            for dev_i in range(torch.cuda.device_count()):
                with torch.cuda.device(dev_i):
                    torch.cuda.empty_cache()
        print(f'  Pass {seq_idx + 1}/{len(sequences)} (L={len(seq_str)}, captured={len(tok_indices)} toks) in {time.time() - t0:.2f}s', flush=True)

    res_mats = {l: np.concatenate(res_buf[l], axis=0) for l in range(n_layers)}
    mix_mats = {l: np.concatenate(mix_buf[l], axis=0) for l in range(n_layers)}
    return res_mats, mix_mats


def probe_binary_groupcv(X, y, groups, C=0.01, n_splits=5):
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_aucs = []
    for tr, va in gkf.split(X, y, groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_va = scaler.transform(X[va])
        clf = LogisticRegression(C=C, max_iter=400, solver='lbfgs', class_weight='balanced')
        clf.fit(X_tr, y[tr])
        prob = clf.predict_proba(X_va)[:, 1]
        oof[va] = prob
        if len(np.unique(y[va])) == 2:
            fold_aucs.append(float(roc_auc_score(y[va], prob)))
    mask = np.isfinite(oof)
    oof_auc = float(roc_auc_score(y[mask], oof[mask]))
    oof_auprc = float(average_precision_score(y[mask], oof[mask]))
    return oof_auc, oof_auprc, float(np.mean(fold_aucs)), float(np.std(fold_aucs))


def probe_multiclass_groupcv(X, y, groups, C=0.01, n_splits=5):
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    oof = np.full(len(y), -1, dtype=np.int64)
    fold_f1 = []
    for tr, va in gkf.split(X, y, groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_va = scaler.transform(X[va])
        clf = LogisticRegression(C=C, max_iter=400, solver='lbfgs', class_weight='balanced')
        clf.fit(X_tr, y[tr])
        pred = clf.predict(X_va)
        oof[va] = pred
        fold_f1.append(float(f1_score(y[va], pred, average='macro')))
    oof_f1 = float(f1_score(y, oof, average='macro'))
    return oof_f1, float('nan'), float(np.mean(fold_f1)), float(np.std(fold_f1))


def probe_ridge_groupcv(X, y, groups, alpha=5000.0, n_splits=5):
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_r2 = []
    for tr, va in gkf.split(X, y, groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_va = scaler.transform(X[va])
        reg = Ridge(alpha=alpha)
        reg.fit(X_tr, y[tr])
        pred = reg.predict(X_va)
        oof[va] = pred
        fold_r2.append(float(r2_score(y[va], pred)))
    oof_r2 = float(r2_score(y, oof))
    return oof_r2, float('nan'), float(np.mean(fold_r2)), float(np.std(fold_r2))


def main():
    rna_path = os.path.join(OUT_DIR, 'rna_fold_structures.json')
    if not os.path.exists(rna_path):
        rna_path = os.path.join(DATA_DIR, 'rna_fold_structures.json')
    brca_path = os.path.join(OUT_DIR, 'brca1_12windows_annotations.json')
    if not os.path.exists(brca_path):
        brca_path = os.path.join(DATA_DIR, 'brca1_12windows_annotations.json')
    with open(brca_path, 'r', encoding='utf-8') as f:
        brca = json.load(f)
    with open(rna_path, 'r', encoding='utf-8') as f:
        rnas = json.load(f)

    rec = SeqIO.read(os.path.join(DATA_DIR, 'NC_012920.1.gb'), 'genbank')
    rcrs_h = str(rec.seq).upper()
    rcrs_l = rcrs_h.translate(RC_DNA)[::-1]
    L_rcrs = len(rcrs_h)

    win_size = brca['win_size']
    n_win = brca['n_win']

    per_win_set = [set() for _ in range(n_win)]
    for lst_key in ['tf_points', 'splice_points', 'exon_points', 'frame_points', 'gc_points']:
        for item in brca[lst_key]:
            p = item['idx']
            w = item['win_idx']
            assert p // win_size == w
            per_win_set[w].add(p % win_size)

    per_win_sorted = [sorted(s) for s in per_win_set]
    brca_gidx_to_row = {}
    cursor = 0
    for w, locs in enumerate(per_win_sorted):
        for loc in locs:
            brca_gidx_to_row[w * win_size + loc] = cursor
            cursor += 1
    print(f'Total BRCA1 captured tokens across 12 windows: {cursor}', flush=True)

    rna_rows_meta = []
    h_toks, l_toks = set(), set()
    for gr in rnas:
        strand = gr['strand']
        gene = gr['gene']
        is_trna = 1 if gr['feature'] == 'tRNA' else 0
        for p1 in gr['threep_rcrs_pos']:
            idx0 = (p1 - 1) if strand == 1 else (L_rcrs - p1)
            (h_toks if strand == 1 else l_toks).add(idx0)
            rna_rows_meta.append((strand, idx0, 1, gene, is_trna))
        for p1 in gr['unpaired_rcrs_pos']:
            if p1 == 3107:
                continue
            idx0 = (p1 - 1) if strand == 1 else (L_rcrs - p1)
            (h_toks if strand == 1 else l_toks).add(idx0)
            rna_rows_meta.append((strand, idx0, 0, gene, is_trna))

    h_sorted = sorted(h_toks)
    l_sorted = sorted(l_toks)
    h_idx_map = {k: i for i, k in enumerate(h_sorted)}
    l_idx_map = {k: i for i, k in enumerate(l_sorted)}
    print(f'Total RNA captured tokens: H={len(h_sorted)}, L={len(l_sorted)}, rows={len(rna_rows_meta)}', flush=True)

    print('=== Loading Evo 2 20B (mmaemu_all) ===', flush=True)
    evo, _, _ = loader.load_evo2_20b(mode_spec='mmaemu_all', profile='hopper-te')

    print('=== Running 12 BRCA1 windows + 2 rCRS passes ===', flush=True)
    brca_ckpt = f'{OUT_DIR}/ckpt_exp1_brca1.npz'
    if os.path.exists(brca_ckpt):
        z = np.load(brca_ckpt, allow_pickle=True)
        res_brca = z['res_brca'].item()
        mix_brca = z['mix_brca'].item()
        print('Loaded cached BRCA1 embeddings from checkpoint.', flush=True)
    else:
        brca_seqs = [w['seq'] for w in brca['windows']]
        res_brca, mix_brca = run_hooked_windows(evo, brca_seqs, per_win_sorted)
        np.savez(brca_ckpt, res_brca=res_brca, mix_brca=mix_brca)
    res_rna, mix_rna = run_hooked_windows(evo, [rcrs_h, rcrs_l], [h_sorted, l_sorted])

    tf_rows = np.array([brca_gidx_to_row[d['idx']] for d in brca['tf_points']], dtype=int)
    tf_y = np.array([d['label'] for d in brca['tf_points']], dtype=int)
    tf_grp = np.array([d['win_idx'] for d in brca['tf_points']], dtype=int)

    sp_rows = np.array([brca_gidx_to_row[d['idx']] for d in brca['splice_points']], dtype=int)
    sp_y = np.array([d['label'] for d in brca['splice_points']], dtype=int)
    sp_grp = np.array([d['win_idx'] for d in brca['splice_points']], dtype=int)

    ex_rows = np.array([brca_gidx_to_row[d['idx']] for d in brca['exon_points']], dtype=int)
    ex_y = np.array([d['label'] for d in brca['exon_points']], dtype=int)
    ex_grp = np.array([d['win_idx'] for d in brca['exon_points']], dtype=int)

    fr_rows = np.array([brca_gidx_to_row[d['idx']] for d in brca['frame_points']], dtype=int)
    fr_y = np.array([d['label'] for d in brca['frame_points']], dtype=int)
    fr_grp = np.array([d['win_idx'] for d in brca['frame_points']], dtype=int)

    gc_rows = np.array([brca_gidx_to_row[d['idx']] for d in brca['gc_points']], dtype=int)
    gc_y = np.array([d['gc_1kb'] for d in brca['gc_points']], dtype=np.float64)
    gc_grp = np.array([d['win_idx'] for d in brca['gc_points']], dtype=int)

    n_h = len(h_sorted)
    rna_y = np.array([r[2] for r in rna_rows_meta], dtype=int)
    rna_grp = np.array([r[3] for r in rna_rows_meta])
    rna_combined_idx = np.array(
        [h_idx_map[r[1]] if r[0] == 1 else (n_h + l_idx_map[r[1]]) for r in rna_rows_meta],
        dtype=int,
    )

    records = []
    for l in range(24):
        op = OP_MAP[l]
        for hook_type, mat_brca, mat_rna in [
            ('residual_stream', res_brca[l], res_rna[l]),
            ('mixer_update', mix_brca[l], mix_rna[l]),
        ]:
            m1 = probe_binary_groupcv(mat_brca[tf_rows], tf_y, tf_grp, C=0.01)
            records.append({'layer': l, 'operator': op, 'hook_type': hook_type, 'task': 'TF_motifs', 'metric_name': 'AUROC',
                            'oof_score': m1[0], 'oof_auprc': m1[1], 'fold_mean': m1[2], 'fold_std': m1[3], 'n_samples': len(tf_y), 'n_pos': int(tf_y.sum())})
            m2 = probe_binary_groupcv(mat_brca[sp_rows], sp_y, sp_grp, C=0.01)
            records.append({'layer': l, 'operator': op, 'hook_type': hook_type, 'task': 'Splice_GT_AG', 'metric_name': 'AUROC',
                            'oof_score': m2[0], 'oof_auprc': m2[1], 'fold_mean': m2[2], 'fold_std': m2[3], 'n_samples': len(sp_y), 'n_pos': int(sp_y.sum())})
            m3 = probe_binary_groupcv(mat_brca[ex_rows], ex_y, ex_grp, C=0.01)
            records.append({'layer': l, 'operator': op, 'hook_type': hook_type, 'task': 'Exon_vs_Intron', 'metric_name': 'AUROC',
                            'oof_score': m3[0], 'oof_auprc': m3[1], 'fold_mean': m3[2], 'fold_std': m3[3], 'n_samples': len(ex_y), 'n_pos': int(ex_y.sum())})
            m4 = probe_multiclass_groupcv(mat_brca[fr_rows], fr_y, fr_grp, C=0.01)
            records.append({'layer': l, 'operator': op, 'hook_type': hook_type, 'task': 'Codon_Frame', 'metric_name': 'Macro_F1',
                            'oof_score': m4[0], 'oof_auprc': m4[1], 'fold_mean': m4[2], 'fold_std': m4[3], 'n_samples': len(fr_y), 'n_pos': int(len(fr_y))})
            m5 = probe_binary_groupcv(mat_rna[rna_combined_idx], rna_y, rna_grp, C=0.01)
            records.append({'layer': l, 'operator': op, 'hook_type': hook_type, 'task': 'RNA_Stem3p_vs_Loop', 'metric_name': 'AUROC',
                            'oof_score': m5[0], 'oof_auprc': m5[1], 'fold_mean': m5[2], 'fold_std': m5[3], 'n_samples': len(rna_y), 'n_pos': int(rna_y.sum())})
            m6 = probe_ridge_groupcv(mat_brca[gc_rows], gc_y, gc_grp, alpha=5000.0)
            records.append({'layer': l, 'operator': op, 'hook_type': hook_type, 'task': 'Regional_1kb_GC', 'metric_name': 'R2',
                            'oof_score': m6[0], 'oof_auprc': m6[1], 'fold_mean': m6[2], 'fold_std': m6[3], 'n_samples': len(gc_y), 'n_pos': int(len(gc_y))})
        print(f'  Probed Layer {l:2d} ({op})', flush=True)

    df_out = pd.DataFrame(records)
    out_csv = f'{OUT_DIR}/layer_probing_results.csv'
    df_out.to_csv(out_csv, index=False)
    print(f'Saved {out_csv}', flush=True)
    for task in ['TF_motifs', 'Splice_GT_AG', 'Exon_vs_Intron', 'Codon_Frame', 'RNA_Stem3p_vs_Loop', 'Regional_1kb_GC']:
        sub = df_out[(df_out['task'] == task) & (df_out['hook_type'] == 'mixer_update')]
        means = sub.groupby('operator')['oof_score'].mean().round(4).to_dict()
        best = sub.sort_values('oof_score', ascending=False).iloc[0]
        print(f'  mixer_update {task:20s}: op_means={means} | best=L{int(best["layer"])} ({best["operator"]}) {best["oof_score"]:.4f}')


if __name__ == '__main__':
    main()
