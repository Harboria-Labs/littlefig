#!/usr/bin/env python3
"""P5c gate: isolated correctness, timing, RSS, and allocator measurements."""
import argparse, ctypes, gc, json, multiprocessing as mp, os, sys, threading, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import psutil
import torch
import torch.nn.functional as F
from little_fig.engine.figquant import FigQuantTensor, figquant_dequantize, figquant_quantize

SHAPES = {"q_proj": (2048, 2048), "k_proj": (2048, 2048), "mlp_proj": (5632, 2048)}
GROUP_SIZE = 128

def rss(): return psutil.Process(os.getpid()).memory_info().rss / 2**20
def trim():
    if sys.platform != "linux": return None
    try: return int(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except (OSError, AttributeError): return None

def make_case(shape):
    out, inp = shape; g = torch.Generator().manual_seed(out * 10000 + inp)
    original = torch.randn(shape, generator=g, dtype=torch.float32)
    return original, figquant_quantize(original, group_size=GROUP_SIZE, n_iters=1, double_quant=False)

def tiled(q, x, tile):
    out, inp = q.shape; parts=[]; gpr=inp//q.group_size
    for r0 in range(0, out, tile):
        r1=min(out,r0+tile); g0=r0*gpr; g1=r1*gpr
        p=q.indices[(g0*q.group_size)//2:(g1*q.group_size+1)//2]
        idx=torch.stack((p&15,(p>>4)&15),1).reshape(-1)[:(g1-g0)*q.group_size].reshape(g1-g0,q.group_size)
        cb=q.codebook.to(torch.bfloat16).unsqueeze(0).expand(g1-g0,-1)
        w=(torch.gather(cb,1,idx)*q.scales[g0:g1].to(torch.bfloat16).unsqueeze(1)).reshape(r1-r0,inp)
        parts.append(F.linear(x.to(w.dtype), w)); del p,idx,cb,w
    return torch.cat(parts, -1)

def worker(cfg, conn):
    print(f"[P5c worker] start {cfg['layer']} {cfg['variant']} iteration {cfg['iteration']}", flush=True)
    torch.set_num_threads(1); shape=tuple(cfg["shape"]); original,q=make_case(shape)
    x=torch.randn(cfg["batch"],cfg["seq"],shape[1]); ref=F.linear(x, original)
    gc.collect(); start=rss(); peak=start; stop=threading.Event()
    def sample():
        nonlocal peak
        while not stop.wait(.005): peak=max(peak,rss())
    t=threading.Thread(target=sample,daemon=True); t.start(); t0=time.perf_counter()
    if cfg["variant"] == "v1_fp32_full": y=F.linear(x, figquant_dequantize(q))
    elif cfg["variant"] == "v2_bf16_full":
        w=figquant_dequantize(q).to(torch.bfloat16); y=F.linear(x.to(w.dtype),w)
    else: y=tiled(q,x,cfg["tile"])
    elapsed=(time.perf_counter()-t0)*1000; stop.set(); t.join(); peak=max(peak,rss())
    err=(y.float()-ref.float()).abs(); relative=err/ref.float().abs().clamp_min(1e-8); after=rss(); del y,ref,original,q,x; gc.collect(); after_gc=rss(); trim_ret=trim(); after_trim=rss()
    conn.send({"variant":cfg["variant"],"shape":list(shape),"rss_start_mib":start,"rss_peak_mib":peak,"rss_after_gc_mib":after_gc,"rss_after_trim_mib":after_trim,"malloc_trim_return":trim_ret,"wall_ms":elapsed,"max_abs_error":err.max().item(),"rmse":err.pow(2).mean().sqrt().item(),"max_relative_error":relative.max().item()}); print(f"[P5c worker] done {cfg['layer']} {cfg['variant']} iteration {cfg['iteration']} ({elapsed:.1f} ms)", flush=True); conn.close()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--iterations",type=int,default=3); ap.add_argument("--batch",type=int,default=2); ap.add_argument("--seq",type=int,default=256); ap.add_argument("--tile",type=int,default=128); ap.add_argument("--results-path",default="benchmark/p5c_dequant_variants_results.json"); ap.add_argument("--error-tol",type=float,default=0.08)
    a=ap.parse_args(); rows=[]; ctx=mp.get_context("spawn")
    for layer,shape in SHAPES.items():
        for v in ("v1_fp32_full","v2_bf16_full","v3_bf16_tiled"):
            for i in range(a.iterations):
                print(f"[P5c] running {layer} {v} iteration {i+1}/{a.iterations}", flush=True)
                parent,child=ctx.Pipe(False); p=ctx.Process(target=worker,args=({"shape":shape,"variant":v,"batch":a.batch,"seq":a.seq,"tile":a.tile,"iteration":i+1,"layer":layer},child)); p.start(); row=parent.recv(); p.join(); row.update(layer=layer,iteration=i+1,correctness_pass=row["max_abs_error"]<=a.error_tol); rows.append(row); print(f"[P5c] collected {len(rows)} case(s)", flush=True)
    out={"scope":"P5c isolated synthetic FigQuant variants","correctness_reference":"unquantized FP32 weights","error_tolerance_max_abs":a.error_tol,"iterations":a.iterations,"layer_shapes":{k:list(v) for k,v in SHAPES.items()},"cases":rows}; os.makedirs(os.path.dirname(os.path.abspath(a.results_path)),exist_ok=True); json.dump(out,open(a.results_path,"w"),indent=2); print(json.dumps(out,indent=2))
if __name__ == "__main__": main()
