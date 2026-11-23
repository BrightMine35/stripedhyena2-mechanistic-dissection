#!/usr/bin/env python3
"""Experiment 3: Unsupervised RNA secondary-structure contact map extraction across all 192 MHA heads."""

import json
import math
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import LeaveOneGroupOut
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
ATTN_LAYERS = [3, 10, 17]


def revcomp(s: str) -> str:
    return s.translate(RC_DNA)[::-1]


def extract_gene_mha_maps(evo, win_seq: str, g_start: int, g_end: int):
    """Run one 2,048 bp forward pass catching post-RoPE qkv at layers 3, 10, 17 and computing exact causal attention."""
    N = g_end - g_start
    blocks = evo.model.blocks
    attn_mats = {}
    handles = []

    def make_pre_hook(layer_id):
        def fn(mod, args):
            qkv = args[0]  # (1, L, 3, 64, 128) post-RoPE on layer device
            with torch.inference_mode():
                q = qkv[0, g_start:g_end, 0].permute(1, 0, 2).float()  # (64, N, 128)
                k = qkv[0, :g_end, 1].permute(1, 0, 2).float()         # (64, g_end, 128)
                scale = 1.0 / math.sqrt(q.shape[-1])
                chunks = []
                for qs in range(0, N, 64):
                    qe = min(qs + 64, N)
                    qc = q[:, qs:qe, :]
                    scores = torch.matmul(qc, k.transpose(-1, -2)) * scale  # (64, C, g_end)
                    q_pos = torch.arange(g_start + qs, g_start + qe, device=q.device)[:, None]
                    k_pos = torch.arange(g_end, device=q.device)[None, :]
                    scores = scores.masked_fill(k_pos > q_pos, float('-inf'))
                    probs = torch.softmax(scores, dim=-1)
                    chunks.append(probs[:, :, g_start:g_end].cpu().numpy())
                attn_mats[layer_id] = np.concatenate(chunks, axis=1)  # (64, N, N)
        return fn

    for l in ATTN_LAYERS:
        handles.append(blocks[l].inner_mha_cls.inner_attn.register_forward_pre_hook(make_pre_hook(l)))

    dev0 = evo.model.block_idx_to_device[0]
    ids = torch.tensor(evo.tokenizer.tokenize(win_seq), dtype=torch.long, device=dev0).unsqueeze(0)
    try:
        with torch.inference_mode():
            logits, _ = evo.model(ids)
            del logits
    finally:
        for h in handles:
            h.remove()
        gc.collect()
        for dev_i in range(torch.cuda.device_count()):
            with torch.cuda.device(dev_i):
                torch.cuda.empty_cache()
    return attn_mats


def causal_apc(A_h: np.ndarray) -> np.ndarray:
    """Reflect strictly causal lower triangle (j > i attending back to i) and apply Average Product Correction."""
    L = np.tril(A_h, k=-1)
    M = L + L.T
    rm = M.mean(axis=1, keepdims=True)
    cm = M.mean(axis=0, keepdims=True)
    gm = float(M.mean()) + 1e-12
    M_apc = M - (rm * cm) / gm
    return M_apc


def main():
    rec = SeqIO.read(f'{DATA_DIR}/NC_012920.1.gb', 'genbank')
    rcrs_h = str(rec.seq).upper()
    L_rcrs = len(rcrs_h)
    rna_path = os.path.join(OUT_DIR, 'rna_fold_structures.json')
    if not os.path.exists(rna_path):
        rna_path = os.path.join(DATA_DIR, 'rna_fold_structures.json')
    with open(rna_path, 'r', encoding='utf-8') as f:
        rnas = json.load(f)

    evo, _, _ = loader.load_evo2_20b(mode_spec='mmaemu_all', profile='hopper-te')

    win_len = 2048
    head_records = []
    trna_feat_X, trna_feat_y, trna_feat_grp, trna_feat_dist = [], [], [], []
    saved_maps = {}

    for idx_g, gr in enumerate(rnas):
        gene = gr['gene']
        strand = int(gr['strand'])
        pmin, pmax = int(gr['pos_min']), int(gr['pos_max'])
        glen = pmax - pmin + 1
        pad_left = (win_len - glen) // 2
        w_start_1 = pmin - pad_left
        h_chars = [rcrs_h[(p1 - 1) % L_rcrs] for p1 in range(w_start_1, w_start_1 + win_len)]
        h_seq = ''.join(h_chars)
        if strand == 1:
            win_seq = h_seq
            gs = pmin - w_start_1
            ge = gs + glen
        else:
            win_seq = revcomp(h_seq)
            ge = (w_start_1 + win_len - 1) - pmin + 1
            gs = ge - glen
        assert win_seq[gs:ge] == gr['dna_trans'], f'Transcribed RNA slice mismatch {gene}'

        t0 = time.time()
        attn_mats = extract_gene_mha_maps(evo, win_seq, gs, ge)

        # True contact upper-triangle pairs with |j - i| >= 4
        pair_bpp = np.zeros((glen, glen), dtype=np.float32)
        for (i, j, p) in gr['bpp_pairs_ge50']:
            pair_bpp[i, j] = float(p)
            pair_bpp[j, i] = float(p)

        iu, ju = np.triu_indices(glen, k=4)
        y_pairs = (pair_bpp[iu, ju] >= 0.50).astype(int)
        dists = ju - iu
        n_pos = int(y_pairs.sum())
        if n_pos == 0:
            continue

        # Build (n_pairs, 192) APC scores & raw causal scores
        all_192_feats = np.zeros((len(y_pairs), 192), dtype=np.float32)
        h_cursor = 0
        for l in ATTN_LAYERS:
            A_l = attn_mats[l]  # (64, N, N)
            for h_idx in range(64):
                M_apc = causal_apc(A_l[h_idx])
                scores = M_apc[iu, ju]
                raw_dir = A_l[h_idx][ju, iu]
                all_192_feats[:, h_cursor] = scores
                h_cursor += 1

                auc_all = float(roc_auc_score(y_pairs, scores))
                auprc_all = float(average_precision_score(y_pairs, scores))
                k_top = max(1, glen // 5)
                top_idx = np.argsort(-scores)[:k_top]
                prec_l5 = float(y_pairs[top_idx].mean())

                auc_lr = float('nan')
                m_lr = dists >= 40
                if y_pairs[m_lr].sum() >= 2 and (1 - y_pairs[m_lr]).sum() >= 2:
                    auc_lr = float(roc_auc_score(y_pairs[m_lr], scores[m_lr]))

                auc_r130 = float('nan')
                m_130 = dists >= 130
                if y_pairs[m_130].sum() >= 2 and (1 - y_pairs[m_130]).sum() >= 2:
                    auc_r130 = float(roc_auc_score(y_pairs[m_130], scores[m_130]))

                head_records.append({
                    'gene': gene, 'feature': gr['feature'], 'strand': strand, 'length': glen,
                    'layer': l, 'head': h_idx, 'head_name': f'L{l}_H{h_idx}',
                    'auroc_apc': auc_all, 'auprc_apc': auprc_all, 'prec_L5_apc': prec_l5,
                    'auroc_raw_dir': float(roc_auc_score(y_pairs, raw_dir)),
                    'auroc_ge40': auc_lr, 'auroc_ge130': auc_r130,
                    'n_pos_pairs': n_pos, 'n_total_pairs': len(y_pairs),
                })

        if gr['feature'] == 'tRNA':
            trna_feat_X.append(all_192_feats)
            trna_feat_y.append(y_pairs)
            trna_feat_grp.append(np.full(len(y_pairs), gene))
            trna_feat_dist.append(dists)

        if gene in ('MT-TL1', 'MT-RNR1'):
            # Compute head means across layer maps for compact storage + keep top heads
            saved_maps[f'{gene}_bpp'] = pair_bpp
            for l in ATTN_LAYERS:
                saved_maps[f'{gene}_L{l}_allheads'] = np.stack([causal_apc(attn_mats[l][hi]) for hi in range(64)], axis=0).astype(np.float16)

        best_g = max([r for r in head_records if r['gene'] == gene], key=lambda x: x['auroc_apc'])
        print(f'  [{idx_g+1:2d}/{len(rnas)}] {gene:8s} (N={glen:4d}, pos={n_pos:3d}) in {time.time()-t0:.2f}s | best head {best_g["head_name"]:8s} AUROC={best_g["auroc_apc"]:.4f} Prec@L/5={best_g["prec_L5_apc"]:.3f}', flush=True)

    df_heads = pd.DataFrame(head_records)
    df_heads.to_csv(f'{OUT_DIR}/mha_rna_contact_heads.csv', index=False)

    # Rao et al. (ICLR 2021) Leave-One-tRNA-Out logistic regression across 192 heads
    X_t = np.concatenate(trna_feat_X, axis=0)
    y_t = np.concatenate(trna_feat_y, axis=0)
    g_t = np.concatenate(trna_feat_grp, axis=0)
    d_t = np.concatenate(trna_feat_dist, axis=0)

    logo = LeaveOneGroupOut()
    oof_ens = np.zeros(len(y_t), dtype=np.float64)
    ens_rows = []
    # Keep last fit coefs normalized mean for head weights
    coef_accum = np.zeros(192, dtype=np.float64)
    n_folds = 0
    for tr, va in logo.split(X_t, y_t, g_t):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_t[tr])
        Xva = scaler.transform(X_t[va])
        clf = LogisticRegression(C=0.05, max_iter=500, solver='lbfgs', class_weight='balanced')
        clf.fit(Xtr, y_t[tr])
        probs = clf.predict_proba(Xva)[:, 1]
        oof_ens[va] = probs
        coef_accum += clf.coef_[0]
        n_folds += 1

        gene_va = str(g_t[va[0]])
        glen_va = int(next(r['length'] for r in rnas if r['gene'] == gene_va))
        k_top = max(1, glen_va // 5)
        top_i = np.argsort(-probs)[:k_top]
        m40 = d_t[va] >= 40
        ens_rows.append({
            'gene': gene_va, 'length': glen_va, 'n_pos': int(y_t[va].sum()),
            'ensemble_auroc': float(roc_auc_score(y_t[va], probs)),
            'ensemble_auprc': float(average_precision_score(y_t[va], probs)),
            'ensemble_prec_L5': float(y_t[va][top_i].mean()),
            'ensemble_auroc_ge40': float(roc_auc_score(y_t[va][m40], probs[m40])) if y_t[va][m40].sum() >= 1 else float('nan'),
        })

    df_ens = pd.DataFrame(ens_rows)
    df_ens.to_csv(f'{OUT_DIR}/mha_rna_contact_ensemble.csv', index=False)

    # Save MT-TL1 ensemble predicted matrix + top 5 weights
    coef_mean = coef_accum / max(1, n_folds)
    head_names = [f'L{l}_H{h}' for l in ATTN_LAYERS for h in range(64)]
    head_rank = sorted(zip(head_names, coef_mean.tolist()), key=lambda x: -abs(x[1]))

    # Keep only top 3 specialist heads + mean ensemble for MT-TL1 and MT-RNR1 to keep .npz compact
    mean_trna = df_heads[df_heads['feature'] == 'tRNA'].groupby(['layer', 'head', 'head_name'])[['auroc_apc', 'auprc_apc', 'prec_L5_apc']].mean().reset_index().sort_values('auroc_apc', ascending=False)
    top3 = mean_trna.head(3).to_dict('records')
    compact_npz = {
        'MT_TL1_bpp': saved_maps['MT-TL1_bpp'],
        'MT_RNR1_bpp': saved_maps['MT-RNR1_bpp'],
        'coef_mean_192': coef_mean,
    }
    for rank_i, th in enumerate(top3):
        l_id, h_id = int(th['layer']), int(th['head'])
        compact_npz[f'MT_TL1_top{rank_i+1}_L{l_id}_H{h_id}'] = saved_maps[f'MT-TL1_L{l_id}_allheads'][h_id]
        compact_npz[f'MT_RNR1_top{rank_i+1}_L{l_id}_H{h_id}'] = saved_maps[f'MT-RNR1_L{l_id}_allheads'][h_id]
    # Also compute MT-TL1 LOO-CV predicted contact map!
    m_tl1 = (g_t == 'MT-TL1')
    tl1_len = next(r['length'] for r in rnas if r['gene'] == 'MT-TL1')
    iu_tl1, ju_tl1 = np.triu_indices(tl1_len, k=4)
    tl1_mat = np.zeros((tl1_len, tl1_len), dtype=np.float32)
    tl1_mat[iu_tl1, ju_tl1] = oof_ens[m_tl1]
    tl1_mat[ju_tl1, iu_tl1] = oof_ens[m_tl1]
    compact_npz['MT_TL1_loo_ensemble'] = tl1_mat
    np.savez(f'{OUT_DIR}/top_rna_contact_maps.npz', **compact_npz)

    print(f'LOO-tRNA 192-head Ensemble pooled OOF AUROC={roc_auc_score(y_t, oof_ens):.4f}, mean gene AUROC={df_ens["ensemble_auroc"].mean():.4f}, mean Prec@L/5={df_ens["ensemble_prec_L5"].mean():.3f}')
    print('Top 5 mean unsupervised heads across 22 mt-tRNAs:', mean_trna.head(5)[['head_name', 'auroc_apc', 'auprc_apc', 'prec_L5_apc']].to_string(index=False))
    print('Top 5 regression weight heads:', head_rank[:5])


if __name__ == '__main__':
    main()
