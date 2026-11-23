import io
import os
import pkgutil
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmaemu

from evo2 import Evo2
from evo2.utils import CONFIG_MAP
from vortex.model.model import StripedHyena
from vortex.model.tokenizer import CharLevelTokenizer
from vortex.model.utils import dotdict, load_checkpoint

E4M3_MAX = 448.0
DEFAULT_CKPT = os.environ.get('EVO2_20B_CKPT', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'evo2_20b', 'evo2_20b.pt'))


def read_te_scales(checkpoint_path: str = DEFAULT_CKPT):
    """Extract Transformer Engine FP8 forward scales [input, weight] per block index."""
    torch.serialization.add_safe_globals([io.BytesIO])
    state = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=True)
    scales = {}
    for key, extra in state.items():
        if not key.endswith('.projections._extra_state') or extra is None:
            continue
        if isinstance(extra, io.BytesIO):
            extra.seek(0)
            try:
                meta = torch.load(extra, map_location='cpu', weights_only=True)
            except Exception:
                extra.seek(0)
                meta = torch.load(extra, map_location='cpu', weights_only=False)
        elif isinstance(extra, torch.Tensor):
            buf = io.BytesIO(extra.numpy().tobytes())
            try:
                meta = torch.load(buf, map_location='cpu', weights_only=True)
            except Exception:
                buf.seek(0)
                meta = torch.load(buf, map_location='cpu', weights_only=False)
        else:
            meta = extra
        block_idx = int(key.split('.')[1])
        scales[block_idx] = meta['scale_fwd'][:2].float().clone()
    return scales


class FP8Projection(nn.Module):
    """Hyena block input projection supporting Hopper FP8 emulation via mmaemu, native sm_120 FP8, or BF16."""

    def __init__(
        self,
        weight: nn.Parameter,
        scale: torch.Tensor,
        mode: str = 'mmaemu',
        profile: str = 'hopper-te',
        chunk_rows: int = 4096,
    ):
        super().__init__()
        self.weight = weight  # [N, K] bf16 parameter (not duplicated in VRAM)
        self.register_buffer('scale', scale.to(device=weight.device, dtype=torch.float32), persistent=False)
        self.register_buffer('scale_inv', torch.reciprocal(scale).to(device=weight.device, dtype=torch.float32), persistent=False)
        self.mode = mode
        self.profile = profile
        self.chunk_rows = chunk_rows

    def _quantize_input(self, x: torch.Tensor):
        orig_shape = x.shape
        orig_len = orig_shape[-2]
        pad = (16 - orig_len % 16) % 16
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        xq = (x.float() * self.scale[0]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        xq = xq.reshape(-1, orig_shape[-1])
        return xq, orig_shape, orig_len

    def _quantize_weight_t(self):
        # Compute E4M3 weight on demand so we do not add 4+ GiB persistent VRAM across 21 layers
        wq = (self.weight.detach().float() * self.scale[1]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        return wq.t()

    def _run_scaled_mm(self, xq_chunk: torch.Tensor, wq_t: torch.Tensor):
        if self.mode == 'mmaemu':
            return mmaemu.scaled_mm(
                xq_chunk,
                wq_t,
                self.scale_inv[0],
                self.scale_inv[1],
                out_dtype=torch.bfloat16,
                use_fast_accum=True,
                profile=self.profile,
            )
        elif self.mode == 'native':
            return torch._scaled_mm(
                xq_chunk,
                wq_t,
                scale_a=self.scale_inv[0],
                scale_b=self.scale_inv[1],
                out_dtype=torch.bfloat16,
                use_fast_accum=True,
            )
        else:
            raise ValueError(f'Invalid FP8Projection mode: {self.mode}')

    def forward(self, x: torch.Tensor):
        if self.mode == 'bf16':
            return F.linear(x.to(self.weight.dtype), self.weight, None), None

        with torch.cuda.device(x.device):
            xq, orig_shape, orig_len = self._quantize_input(x)
            wq_t = self._quantize_weight_t()
            m_total = xq.shape[0]
            if self.chunk_rows and m_total > self.chunk_rows:
                outs = []
                for start in range(0, m_total, self.chunk_rows):
                    end = min(start + self.chunk_rows, m_total)
                    outs.append(self._run_scaled_mm(xq[start:end], wq_t))
                y = torch.cat(outs, dim=0)
            else:
                y = self._run_scaled_mm(xq, wq_t)

            y = y.reshape(*orig_shape[:-2], -1, self.weight.shape[0])[..., :orig_len, :]
            return y, None


def set_evo2_fp8_mode(evo: Evo2, scales: dict, mode_spec: str, profile: str = 'hopper-te', chunk_rows: int = 4096):
    """
    Configure projection modes across all 21 FP8 Hyena blocks.
    mode_spec options:
      - 'mmaemu_all': mmaemu on all 21 Hyena blocks
      - 'native_all': native sm_120 torch._scaled_mm on all 21 Hyena blocks
      - 'bf16_all': bf16 F.linear on all 21 Hyena blocks
      - 'mmaemu_b0_native': mmaemu on Block 0, native sm_120 on Blocks 1..23
      - 'mmaemu_b0_bf16': mmaemu on Block 0, bf16 on Blocks 1..23
    """
    blocks = evo.model.blocks
    if not hasattr(evo, '_bf16_projections'):
        evo._bf16_projections = {i: blocks[i].projections for i in sorted(scales.keys())}

    for i, orig in evo._bf16_projections.items():
        if mode_spec == 'mmaemu_all':
            blk_mode = 'mmaemu'
        elif mode_spec == 'native_all':
            blk_mode = 'native'
        elif mode_spec == 'bf16_all':
            blk_mode = 'bf16'
        elif mode_spec == 'mmaemu_b0_native':
            blk_mode = 'mmaemu' if i == 0 else 'native'
        elif mode_spec == 'mmaemu_b0_bf16':
            blk_mode = 'mmaemu' if i == 0 else 'bf16'
        else:
            raise ValueError(f'Unknown mode_spec: {mode_spec}')

        if blk_mode == 'bf16':
            blocks[i].projections = orig
        elif isinstance(blocks[i].projections, FP8Projection):
            blocks[i].projections.mode = blk_mode
            blocks[i].projections.profile = profile
            blocks[i].projections.chunk_rows = chunk_rows
        else:
            blocks[i].projections = FP8Projection(
                orig.weight,
                scales[i],
                mode=blk_mode,
                profile=profile,
                chunk_rows=chunk_rows,
            )
    return len(evo._bf16_projections)


def load_evo2_20b(
    mode_spec: str = 'mmaemu_all',
    checkpoint_path: str = DEFAULT_CKPT,
    profile: str = 'hopper-te',
    chunk_rows: int = 4096,
):
    """Load Evo 2 20B cleanly without TE using our patched multi-GPU Vortex backend and FP8 projection mode."""
    config = dotdict(yaml.safe_load(pkgutil.get_data('evo2', CONFIG_MAP['evo2_20b'])))
    config.use_fp8_input_projections = False
    evo = Evo2.__new__(Evo2)
    evo.model = StripedHyena(config)
    load_checkpoint(evo.model, checkpoint_path)
    evo.tokenizer = CharLevelTokenizer(512)
    scales = read_te_scales(checkpoint_path)
    n_proj = set_evo2_fp8_mode(evo, scales, mode_spec=mode_spec, profile=profile, chunk_rows=chunk_rows)
    return evo, scales, n_proj
