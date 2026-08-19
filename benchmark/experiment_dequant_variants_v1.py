#!/usr/bin/env python3
"""P5b: compare full FP32, full BF16, and row-tiled BF16 dequantization."""
import argparse, ctypes, gc, json, os, platform, threading, time
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import psutil
import torch
import torch.nn.functional as F
from little_fig.engine.figquant import FigQuantTensor, figquant_dequantize

SHAPES = {"q_proj": (2048, 2048), "mlp_proj": (5632, 2048)}
GROUP_SIZE = 128

def rss_mib(): return psutil.Process(os.getpid()).memory_info().rss / 2**20
def log(s): print(f"[P5b] {s}", flush=True)

def make_q(out_features, in_features):
    numel = out_features * in_features
    ng = (numel + GROUP_SIZE - 1) // GROUP_SIZE
    padded = ng * GROUP_SIZE
    g = torch.Generator().manual_seed(out_features * 10000 + in_features)
    lo = torch.randint(0, 16, ((padded + 1)//2,), dtype=torch.uint8, generator=g)
    hi = torch.randint(0, 16, lo.shape, dtype=torch.uint8, generator=g)
    return FigQuantTensor(lo | (hi << 4), torch.linspace(-1, 1, 16),
                          torch.full((ng,), .02), torch.Size((out_features, in_features)),
                          ng, GROUP_SIZE, numel)

def deq_bf16(q):
    low = (q.indices & 15).long(); high = (q.indices >> 4).long()
    idx = torch.stack((low, high), 1).reshape(-1)[:q.n_groups*q.group_size].reshape(q.n_groups, q.group_size)
    cb = q.codebook.to(torch.bfloat16).unsqueeze(0).expand(q.n_groups, -1)
    return (torch.gather(cb, 1, idx) * q.scales.to(torch.bfloat16).unsqueeze(1)).reshape(-1)[:q.numel].reshape(q.shape)

def deq_tiled(q, x, tile_rows):
    out, inp = q.shape; result = torch.empty((out, inp), dtype=torch.bfloat16)
    groups_per_row = inp // q.group_size
    for r0 in range(0, out, tile_rows):
        r1 = min(out, r0 + tile_rows); g0 = r0 * groups_per_row; g1 = r1 * groups_per_row
        packed0 = (g0 * q.group_size) // 2; packed1 = (g1 * q.group_size + 1) // 2
        p = q.indices[packed0:packed1]; lo=(p&15).long(); hi=(p>>4).long()
        idx=torch.stack((lo,hi),1).reshape(-1)[:(g1-g0)*q.group_size].reshape(g1-g0,q.group_size)
        cb=q.codebook.to(torch.bfloat16).unsqueeze(0).expand(g1-g0,-1)
        result[r0:r1] = (torch.gather(cb,1,idx)*q.scales[g0:g1].to(torch.bfloat16).unsqueeze(1)).reshape(r1-r0, inp)
        del p, lo, hi, idx, cb
    return result

def run_variant(q, x, name, tile_rows=128):
    gc.collect(); start=rss_mib(); peak=start; stop=threading.Event()
    def sample():
        nonlocal peak
        while not stop.wait(.01): peak=max(peak,rss_mib())
    t=threading.Thread(target=sample,daemon=True); t.start()
    deq_start=rss_mib()
    if name == 'v1_fp32_full': w=figquant_dequantize(q); y=F.linear(x,w.to(x.dtype))
    elif name == 'v2_bf16_full': w=deq_bf16(q); y=F.linear(x,w.to(x.dtype))
    else:
        y_parts=[]; out, inp=q.shape; gpr=inp//q.group_size
        for r0 in range(0,out,tile_rows):
            r1=min(out,r0+tile_rows); g0=r0*gpr; g1=r1*gpr
            p=q.indices[(g0*q.group_size)//2:(g1*q.group_size+1)//2]
            lo=(p&15).long(); hi=(p>>4).long(); idx=torch.stack((lo,hi),1).reshape(-1)[:(g1-g0)*q.group_size].reshape(g1-g0,q.group_size)
            cb=q.codebook.to(torch.bfloat16).unsqueeze(0).expand(g1-g0,-1)
            wt=(torch.gather(cb,1,idx)*q.scales[g0:g1].to(torch.bfloat16).unsqueeze(1)).reshape(r1-r0,inp)
            y_parts.append(F.linear(x.to(wt.dtype),wt)); del p,lo,hi,idx,cb,wt
        y=torch.cat(y_parts,-1); w=None
    deq_end=rss_mib(); y.float().mean().backward(); stop.set(); t.join(); peak=max(peak,rss_mib())
    del y, w; gc.collect()
    return {'variant':name,'rss_start_mib':start,'rss_peak_mib':peak,'rss_after_dequant_mib':deq_end,
            'dequant_peak_delta_mib':deq_end-deq_start}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--iterations',type=int,default=3); ap.add_argument('--batch-size',type=int,default=2); ap.add_argument('--sequence-length',type=int,default=16); ap.add_argument('--tile-rows',type=int,default=128); ap.add_argument('--results-path',default=os.path.join(os.path.dirname(__file__),'dequant_variants_v1_results.json')); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    shapes={'smoke':(256,256)} if a.smoke else SHAPES; results={'scope':'P5b dequant variants; isolated synthetic FigQuant weights','iterations':a.iterations,'batch_size':a.batch_size,'sequence_length':a.sequence_length,'tile_rows':a.tile_rows,'cases':[]}
    for n,shape in shapes.items():
        q=make_q(*shape); x=torch.randn(a.batch_size,a.sequence_length,shape[1],requires_grad=True)
        for v in ('v1_fp32_full','v2_bf16_full','v3_bf16_tiled'):
            for i in range(a.iterations): results['cases'].append(dict(layer=n,iteration=i+1,shape=list(shape),**run_variant(q,x,v,a.tile_rows)))
        del q,x; gc.collect()
    os.makedirs(os.path.dirname(os.path.abspath(a.results_path)),exist_ok=True); tmp=a.results_path+'.tmp'; json.dump(results,open(tmp,'w'),indent=2); os.replace(tmp,a.results_path); log(f'Saved -> {a.results_path}')
if __name__=='__main__': main()
