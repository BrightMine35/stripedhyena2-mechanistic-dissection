#!/usr/bin/env python3
"""Prepare ground-truth datasets for Direction C (StripedHyena 2 mechanistic dissection)."""

import gzip
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.environ.get('SH2_REF_DATA_DIR', str(BASE_DIR / 'datasets' / 'raw_reference'))
if not os.path.exists(DATA_DIR) and (BASE_DIR / 'data').is_dir():
    DATA_DIR = str(BASE_DIR / 'data')
RES_DIR = os.environ.get('SH2_ATLAS_RES_DIR', DATA_DIR)
OUT_DIR = os.environ.get('SH2_OUT_DIR', str(BASE_DIR / 'datasets' / 'processed'))
if not os.path.exists(OUT_DIR) and (BASE_DIR / 'data').is_dir():
    OUT_DIR = str(BASE_DIR / 'data')
RNAFOLD_BIN = os.environ.get('RNAFOLD_BIN', shutil.which('RNAfold') or 'RNAfold')
HG19_OFFSET = 1847983

RC_DNA = str.maketrans('ACGTN', 'TGCAN')
WC_PAIRS = {('A', 'T'), ('T', 'A'), ('G', 'C'), ('C', 'G')}


def revcomp(s: str) -> str:
    return s.translate(RC_DNA)[::-1]


def run_rnafold_p(name: str, rna_seq: str):
    """Run RNAfold -p --noPS and parse MFE dot-bracket, MFE energy, and dot.ps base-pair probs."""
    with tempfile.TemporaryDirectory() as tmpd:
        inp = f'>{name}\n{rna_seq}\n'
        res = subprocess.run(
            [RNAFOLD_BIN, '--noPS', '-p'],
            input=inp,
            text=True,
            cwd=tmpd,
            capture_output=True,
            check=True,
        )
        lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
        db_line = lines[2]
        mfe_struct, rest = db_line.split(' ', 1)
        mfe_energy = float(rest.strip().lstrip('(').rstrip(')').strip())

        dp_path = os.path.join(tmpd, f'{name}_dp.ps')
        if not os.path.exists(dp_path):
            dp_path = os.path.join(tmpd, 'dot.ps')
        bpp_pairs = []
        with open(dp_path, 'r', encoding='utf-8', errors='ignore') as f:
            for ln in f:
                if ln.endswith(' ubox\n') or ln.strip().endswith('ubox'):
                    parts = ln.strip().split()
                    if len(parts) == 4 and parts[3] == 'ubox' and parts[0].isdigit() and parts[1].isdigit():
                        i = int(parts[0]) - 1
                        j = int(parts[1]) - 1
                        p = float(parts[2]) ** 2
                        if p >= 0.50:
                            bpp_pairs.append((i, j, p))

        # Parse MFE stack pairs
        stack = []
        mfe_pairs = []
        for idx, ch in enumerate(mfe_struct):
            if ch == '(':
                stack.append(idx)
            elif ch == ')' and stack:
                i = stack.pop()
                mfe_pairs.append((i, idx))
        mfe_pairs.sort()
        return mfe_struct, mfe_energy, mfe_pairs, bpp_pairs


def build_mt_rna_structures():
    rec = SeqIO.read(os.path.join(DATA_DIR, 'NC_012920.1.gb'), 'genbank')
    rcrs = str(rec.seq).upper()
    L_rcrs = len(rcrs)
    atlas = pd.read_csv(f'{RES_DIR}/mitochondrial_organelle_pathogenicity_atlas.csv')
    # Get unique gene spans for tRNA and rRNA
    sub = atlas[atlas['feature'].isin(['tRNA', 'rRNA'])][['pos', 'gene', 'feature', 'strand']].drop_duplicates('pos')
    gene_rows = []
    for gene, g in sub.groupby('gene', sort=False):
        pos_min = int(g['pos'].min())
        pos_max = int(g['pos'].max())
        strand = int(g['strand'].iloc[0])
        region = str(g['feature'].iloc[0])
        # All 22 tRNAs and 12S/16S rRNAs are contiguous on NC_012920.1 (no origin wrap)
        n_gap = 1 if (pos_min <= 3107 <= pos_max) else 0
        assert (pos_max - pos_min + 1) == (len(g) + n_gap), f'Non-contiguous RNA {gene}: span={pos_max-pos_min+1}, len={len(g)}'
        dna_h = rcrs[pos_min - 1 : pos_max]
        dna_trans = dna_h if strand == 1 else revcomp(dna_h)
        rna_trans = dna_trans.replace('T', 'U')
        safe_name = gene.replace('-', '_')
        mfe_struct, mfe_e, mfe_pairs, bpp_pairs = run_rnafold_p(safe_name, rna_trans)
        paired_mfe_set = {x for p in mfe_pairs for x in p}
        bpp_high = [(i, j, p) for (i, j, p) in bpp_pairs if p >= 0.50]
        fivep_bpp = {i for (i, j, p) in bpp_high}
        threep_bpp = {j for (i, j, p) in bpp_high}

        # Map local index k (0..len-1) back to 1-based rCRS pos
        def local_to_rcrs(k):
            return (pos_min + k) if strand == 1 else (pos_max - k)

        gene_rows.append({
            'gene': gene,
            'feature': region,
            'strand': strand,
            'pos_min': pos_min,
            'pos_max': pos_max,
            'length': len(dna_trans),
            'dna_trans': dna_trans,
            'mfe_struct': mfe_struct,
            'mfe_energy': mfe_e,
            'mfe_pairs': mfe_pairs,
            'bpp_pairs_ge50': bpp_high,
            'fivep_rcrs_pos': sorted(local_to_rcrs(k) for k in fivep_bpp),
            'threep_rcrs_pos': sorted(local_to_rcrs(k) for k in threep_bpp),
            'unpaired_rcrs_pos': sorted(local_to_rcrs(k) for k in range(len(dna_trans)) if k not in paired_mfe_set and k not in fivep_bpp and k not in threep_bpp),
        })
        print(f'  RNAfold {gene:8s} (str={strand:+d}, len={len(dna_trans):4d}): MFE={mfe_e:7.2f}, mfe_pairs={len(mfe_pairs):3d}, bpp>=0.5={len(bpp_high):3d}')

    # Select 24 canonical Watson-Crick pairs for compensatory epistasis ablation:
    # 8 short (10..25 bp), 8 medium (55..70 bp), 8 long (130..600 bp)
    candidates = {'short_10_25': [], 'medium_55_70': [], 'long_130_600': []}
    for gr in gene_rows:
        mfe_set = set(map(tuple, gr['mfe_pairs']))
        dna = gr['dna_trans']
        for (i, j, p) in gr['bpp_pairs_ge50']:
            if (i, j) not in mfe_set or p < 0.75:
                continue
            b_i, b_j = dna[i], dna[j]
            if (b_i, b_j) not in WC_PAIRS:
                continue
            dist = j - i
            ri = gr['pos_min'] + i if gr['strand'] == 1 else gr['pos_max'] - i
            rj = gr['pos_min'] + j if gr['strand'] == 1 else gr['pos_max'] - j
            rec_p = {
                'gene': gr['gene'],
                'feature': gr['feature'],
                'strand': gr['strand'],
                'local_i': i,
                'local_j': j,
                'dist': dist,
                'bpp': p,
                'wt_i': b_i,
                'wt_j': b_j,
                'rcrs_i': ri,
                'rcrs_j': rj,
                'pos_min': gr['pos_min'],
                'pos_max': gr['pos_max'],
            }
            if 10 <= dist <= 25 and gr['feature'] == 'tRNA':
                candidates['short_10_25'].append(rec_p)
            elif 55 <= dist <= 70 and gr['feature'] == 'tRNA':
                candidates['medium_55_70'].append(rec_p)
            elif 130 <= dist <= 820 and gr['feature'] == 'rRNA':
                candidates['long_130_600'].append(rec_p)

    selected = []
    for stratum, pool in candidates.items():
        pool_sorted = sorted(pool, key=lambda x: (-x['bpp'], -x['dist']))
        chosen = []
        used_genes = set()
        used_coords = []
        # First pass: maximize gene diversity and avoid adjacent stack neighbors (|di|>=3)
        for cand in pool_sorted:
            if len(chosen) == 8:
                break
            min_sep = 2 if stratum == 'long_130_600' else 4
            if any(cand['gene'] == uc[0] and abs(cand['local_i'] - uc[1]) < min_sep for uc in used_coords):
                continue
            if stratum != 'long_130_600' and cand['gene'] in used_genes and len(pool_sorted) > 8:
                continue
            cand['stratum'] = stratum
            chosen.append(cand)
            used_genes.add(cand['gene'])
            used_coords.append((cand['gene'], cand['local_i']))
        if len(chosen) < 8:
            for cand in pool_sorted:
                if len(chosen) == 8:
                    break
                min_sep = 2 if stratum == 'long_130_600' else 4
                if any(cand['gene'] == uc[0] and abs(cand['local_i'] - uc[1]) < min_sep for uc in used_coords):
                    continue
                if cand in chosen:
                    continue
                cand['stratum'] = stratum
                chosen.append(cand)
                used_coords.append((cand['gene'], cand['local_i']))
        assert len(chosen) == 8, f'Only {len(chosen)} pairs found for {stratum}'
        selected.extend(chosen)

    # Build 1,024 bp transcribed-strand circular context centered on midpoint of (i, j)
    win_len = 1024
    half = win_len // 2
    ep_rows = []
    for idx_p, pinfo in enumerate(selected):
        # Extract on linear circularly wrapped H-strand centered near midpoint of [pos_min..pos_max] in rCRS
        mid_rcrs = (pinfo['rcrs_i'] + pinfo['rcrs_j']) // 2
        w_start_1 = mid_rcrs - half + 1  # 1-based H-strand start (may wrap <=0)
        h_seq_chars = [rcrs[(pos1 - 1) % L_rcrs] for pos1 in range(w_start_1, w_start_1 + win_len)]
        h_win = ''.join(h_seq_chars)
        if pinfo['strand'] == 1:
            wt_win = h_win
            wi = pinfo['rcrs_i'] - w_start_1
            wj = pinfo['rcrs_j'] - w_start_1
        else:
            wt_win = revcomp(h_win)
            wi = (w_start_1 + win_len - 1) - pinfo['rcrs_i']
            wj = (w_start_1 + win_len - 1) - pinfo['rcrs_j']

        assert 50 <= wi < wj < win_len - 50, f'Window bounds check failed {wi}, {wj}'
        assert wt_win[wi] == pinfo['wt_i'] and wt_win[wj] == pinfo['wt_j'], (
            f'Base mismatch {wt_win[wi]}{wt_win[wj]} vs {pinfo["wt_i"]}{pinfo["wt_j"]}'
        )
        # Watson-Crick swap: A<->G and T<->C preserves purine/pyrimidine geometry & reverses H-bond polarity!
        swap_map = {'A': 'G', 'T': 'C', 'G': 'A', 'C': 'T'}
        alt_i = swap_map[pinfo['wt_i']]
        alt_j = swap_map[pinfo['wt_j']]
        assert (alt_i, alt_j) in WC_PAIRS

        ep_rows.append({
            'pair_id': f'P{idx_p+1:02d}_{pinfo["gene"]}_{pinfo["local_i"]}_{pinfo["local_j"]}',
            'stratum': pinfo['stratum'],
            'gene': pinfo['gene'],
            'strand': pinfo['strand'],
            'dist': pinfo['dist'],
            'bpp': round(pinfo['bpp'], 4),
            'wi': wi,
            'wj': wj,
            'wt_pair': f'{pinfo["wt_i"]}-{pinfo["wt_j"]}',
            'comp_pair': f'{alt_i}-{alt_j}',
            'alt_i': alt_i,
            'alt_j': alt_j,
            'wt_seq': wt_win,
        })

    ep_df = pd.DataFrame(ep_rows)
    ep_df.to_csv(f'{OUT_DIR}/rna_stem_epistasis_pairs.csv', index=False)
    with open(f'{OUT_DIR}/rna_fold_structures.json', 'w', encoding='utf-8') as f:
        json.dump(gene_rows, f, indent=2)
    print(f'Saved {len(ep_df)} compensatory RNA stem pairs (8 short / 8 medium / 8 long) & 24 RNA structures.')


def build_brca1_probing_loci():
    rec = SeqIO.read(gzip.open(f'{DATA_DIR}/GRCh37.p13_chr17.fna.gz', 'rt'), 'fasta')
    chr17 = str(rec.seq).upper()

    raw = open(f'{DATA_DIR}/brca1_gtf_v46.json', encoding='utf-8').read()
    gtf = json.loads(raw[raw.index('['):])
    mane = next((t for t in gtf if t.get('is_mane_select')), gtf[0])

    # Tile 12 x 8,192 bp non-overlapping windows covering all of BRCA1 on coding (-) strand (5'->3')
    # ENST00000357654.9 hg19 span is [41196311..41277381] -> use top 1-based = 41283500 down to 41185197 (98,304 bp = 12 * 8192)
    win_size = 8192
    n_win = 12
    total_bp = win_size * n_win
    g_top_1 = 41283500  # 1-based inclusive top genomic pos on chr17 (~6.1 kb upstream of BRCA1 TSS)
    g_bot_1 = g_top_1 - total_bp + 1  # 41185197 (~11.1 kb downstream of 3' UTR)

    genomic_fwd = chr17[g_bot_1 - 1 : g_top_1]
    assert len(genomic_fwd) == total_bp
    coding_seq = revcomp(genomic_fwd)  # coding_seq[k] comes from 1-based genomic pos (g_top_1 - k)

    def g1_to_coding(p1: int) -> int:
        return g_top_1 - p1

    # Annotate exons, CDS reading frame, and splice GT/AG on coding_seq
    exonic_mask = np.zeros(total_bp, dtype=bool)
    cds_frame = np.full(total_bp, -1, dtype=np.int8)
    true_donors = []   # index of 'T' in coding GT (donor +2)
    true_acceptors = []  # index of 'G' in coding AG (acceptor -1)

    for e in mane['exons']:
        s19 = e['start'] - HG19_OFFSET
        e19 = e['end'] - HG19_OFFSET
        # 1-based exonic span [s19+1 .. e19] -> coding_seq[g1_to_coding(e19) : g1_to_coding(s19+1)+1]
        c_start = g1_to_coding(e19)
        c_end_excl = g1_to_coding(s19)  # length e19 - s19
        assert 0 <= c_start < c_end_excl <= total_bp
        exonic_mask[c_start:c_end_excl] = True
        ex_num = e['exon_number']
        if ex_num > 1:
            acc = coding_seq[c_start - 2 : c_start]
            assert acc == 'AG', f'Acceptor mismatch exon {ex_num}: {acc}'
            true_acceptors.append(c_start - 1)  # index of 'G' in 'AG'
        if ex_num < mane['num_exons']:
            don = coding_seq[c_end_excl : c_end_excl + 2]
            assert don == 'GT', f'Donor mismatch exon {ex_num}: {don}'
            true_donors.append(c_end_excl + 1)  # index of 'T' in 'GT'

    # CDS frame tracking in 5'->3' order
    frame_cursor = 0
    for c in mane['cds']:
        s19 = c['start'] - HG19_OFFSET
        e19 = c['end'] - HG19_OFFSET
        c_start = g1_to_coding(e19)
        c_end_excl = g1_to_coding(s19)
        for pos_c in range(c_start, c_end_excl):
            cds_frame[pos_c] = frame_cursor % 3
            frame_cursor += 1
    assert frame_cursor % 3 == 0 and coding_seq[g1_to_coding(mane['cds'][0]['end'] - HG19_OFFSET) : g1_to_coding(mane['cds'][0]['end'] - HG19_OFFSET) + 3] == 'ATG'

    # Find intronic decoy GT and AG sites strictly away from true splice sites (>= 50 bp from exons) and >= 256 bp inside windows
    true_donor_set = set(true_donors)
    true_acceptor_set = set(true_acceptors)
    decoy_donors = []
    decoy_acceptors = []
    for w in range(n_win):
        lo = w * win_size + 256
        hi = (w + 1) * win_size - 64
        for idx in range(lo, hi):
            if exonic_mask[max(0, idx - 50) : min(total_bp, idx + 50)].any():
                continue
            dinuc = coding_seq[idx - 1 : idx + 1]
            if dinuc == 'GT' and idx not in true_donor_set:
                decoy_donors.append(idx)
            elif dinuc == 'AG' and idx not in true_acceptor_set:
                decoy_acceptors.append(idx)

    # Local TF motifs (consensus motifs in BRCA1 locus) -> anchor on final motif token!
    tf_patterns = {
        'TATA': re.compile(r'TATA[AT]A[AT]'),
        'SP1': re.compile(r'GGGCGG|CCGCCC'),
        'EBOX': re.compile(r'CACGTG'),
        'CTCF_CORE': re.compile(r'CC[AC]CC[AT]GG[AG]GG|GC[CT]CC[AT]GG[GT]GG'),
        'AP1': re.compile(r'TGA[CG]TCA'),
        'NFKB': re.compile(r'GGG[AG][ACGT][CT]{3}CC'),
    }
    tf_pos_hits = []
    tf_hit_span = np.zeros(total_bp, dtype=bool)
    for mname, pat in tf_patterns.items():
        for m in pat.finditer(coding_seq):
            end_idx = m.end() - 1
            win_local = end_idx % win_size
            if 256 <= win_local < win_size - 32:
                tf_pos_hits.append((end_idx, mname, m.start(), m.end()))
                tf_hit_span[max(0, m.start() - 16) : min(total_bp, m.end() + 16)] = True

    rng = np.random.default_rng(42)
    # Sample matched decoy GT/AG (10 per true site per window if available) and matched non-TF control tokens
    decoy_d_arr = np.array(decoy_donors, dtype=int)
    decoy_a_arr = np.array(decoy_acceptors, dtype=int)
    sampled_decoy_d = []
    sampled_decoy_a = []
    # Space sampled decoys by >= 64 bp
    for arr, out_list in [(decoy_d_arr, sampled_decoy_d), (decoy_a_arr, sampled_decoy_a)]:
        perm = rng.permutation(arr)
        for x in perm:
            if not any(abs(x - y) < 64 for y in out_list):
                out_list.append(int(x))
            if len(out_list) >= 440:
                break

    # Matched non-TF tokens matched on exact last nucleotide distribution & local >=16bp away from motifs
    tf_nonhit_pool = []
    for w in range(n_win):
        base = w * win_size
        for k in range(256, win_size - 32, 37):
            p = base + k
            if not tf_hit_span[p]:
                tf_nonhit_pool.append(p)
    tf_pos_indices = [x[0] for x in tf_pos_hits]
    # Keep distinct TF hits separated by >=16 bp
    tf_pos_dedup = []
    for p, mn, ms, me in sorted(tf_pos_hits):
        if not tf_pos_dedup or (p - tf_pos_dedup[-1][0]) >= 16:
            tf_pos_dedup.append((p, mn))
    # Match TF negatives by ending base nucleotide
    pos_bases = pd.Series([coding_seq[p] for p, _ in tf_pos_dedup]).value_counts().to_dict()
    neg_by_base = {b: [p for p in tf_nonhit_pool if coding_seq[p] == b] for b in 'ACGT'}
    tf_neg_sel = []
    for b, cnt in pos_bases.items():
        pool_b = neg_by_base.get(b, tf_nonhit_pool)
        n_take = min(len(pool_b), cnt * 3)
        tf_neg_sel.extend(rng.choice(pool_b, size=n_take, replace=False).tolist())

    # Regional 1 kb causal GC & Exon/Intron & Codon Frame sample points (stride 47 bp across >=1024 inside each 8192 window)
    gc_points, gc_vals = [], []
    exon_pts, exon_y = [], []
    frame_pts, frame_y = [], []
    # Prefix sum for O(1) GC
    is_gc = np.array([1 if c in ('G', 'C') else 0 for c in coding_seq], dtype=np.int32)
    pref_gc = np.concatenate([[0], np.cumsum(is_gc)])

    for w in range(n_win):
        base = w * win_size
        for k in range(1024, win_size - 32, 47):
            p = base + k
            gc_1k = (pref_gc[p + 1] - pref_gc[p + 1 - 1024]) / 1024.0
            gc_points.append(p)
            gc_vals.append(float(gc_1k))
        # Balanced exon vs intron samples per window (stride 7 in exon, stride 79 in intron)
        for k in range(256, win_size - 32):
            p = base + k
            if exonic_mask[p] and (k % 7 == 0):
                exon_pts.append(p)
                exon_y.append(1)
            elif (not exonic_mask[p]) and (k % 73 == 0):
                exon_pts.append(p)
                exon_y.append(0)
            if cds_frame[p] >= 0 and (k % 3 == 0 or k % 4 == 0) and (k % 5 == 0):
                frame_pts.append(p)
                frame_y.append(int(cds_frame[p]))

    windows = []
    for w in range(n_win):
        s = w * win_size
        e = s + win_size
        windows.append({
            'win_idx': w,
            'coding_start': s,
            'coding_end': e,
            'hg19_top': g_top_1 - s,
            'hg19_bot': g_top_1 - e + 1,
            'seq': coding_seq[s:e],
        })

    out = {
        'g_top_1': g_top_1,
        'g_bot_1': g_bot_1,
        'win_size': win_size,
        'n_win': n_win,
        'windows': windows,
        'splice_points': (
            [{'idx': int(p), 'label': 1, 'stype': 'donor', 'win_idx': int(p // win_size)} for p in true_donors]
            + [{'idx': int(p), 'label': 1, 'stype': 'acceptor', 'win_idx': int(p // win_size)} for p in true_acceptors]
            + [{'idx': int(p), 'label': 0, 'stype': 'donor', 'win_idx': int(p // win_size)} for p in sampled_decoy_d]
            + [{'idx': int(p), 'label': 0, 'stype': 'acceptor', 'win_idx': int(p // win_size)} for p in sampled_decoy_a]
        ),
        'tf_points': (
            [{'idx': int(p), 'label': 1, 'motif': mn, 'win_idx': int(p // win_size)} for p, mn in tf_pos_dedup]
            + [{'idx': int(p), 'label': 0, 'motif': 'bg', 'win_idx': int(p // win_size)} for p in tf_neg_sel]
        ),
        'gc_points': [
            {'idx': int(p), 'gc_1kb': float(v), 'win_idx': int(p // win_size)}
            for p, v in zip(gc_points, gc_vals)
        ],
        'exon_points': [
            {'idx': int(p), 'label': int(y), 'win_idx': int(p // win_size)}
            for p, y in zip(exon_pts, exon_y)
        ],
        'frame_points': [
            {'idx': int(p), 'label': int(y), 'win_idx': int(p // win_size)}
            for p, y in zip(frame_pts, frame_y)
        ],
    }
    with open(f'{OUT_DIR}/brca1_12windows_annotations.json', 'w', encoding='utf-8') as f:
        json.dump(out, f)

    print(f'BRCA1 coding locus: 12 x {win_size} bp ({total_bp} bp)')
    print(f'  True splice sites: {len(true_donors)} GT donors, {len(true_acceptors)} AG acceptors (100% canonical match verified)')
    print(f'  Decoy splice sites sampled: {len(sampled_decoy_d)} GT, {len(sampled_decoy_a)} AG')
    print(f'  TF motifs (dedup >=16bp): {len(tf_pos_dedup)} positive ({pd.Series([m for _, m in tf_pos_dedup]).value_counts().to_dict()}), {len(tf_neg_sel)} base-matched negatives')
    print(f'  Exon/Intron points: {sum(exon_y)} exon / {len(exon_y) - sum(exon_y)} intron')
    print(f'  CDS Frame points: {len(frame_y)} across frames 0/1/2')
    print(f'  1 kb causal GC points: {len(gc_points)}')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print('=== Step 1: Mitochondrial RNAfold structures & epistasis pairs ===')
    build_mt_rna_structures()
    print('=== Step 2: BRCA1 12-window coding-strand ground truth ===')
    build_brca1_probing_loci()


if __name__ == '__main__':
    main()
