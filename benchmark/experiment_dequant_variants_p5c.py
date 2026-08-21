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
def append_jsonl(path, row):
    if not path: return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n"); f.flush(); os.fsync(f.fileno())
def trim():
    if sys.platform != "linux": return None
    try: return int(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except (OSError, AttributeError): return None

def make_case(shape):
    out, inp = shape; g = torch.Generator().manual_seed(out * 10000 + inp)
    print(f"[P5c worker] allocating source weights {out}x{inp}", flush=True)
    # Match transformer weight scale; std=1 creates an artificial output-error regime.
    original = torch.randn(shape, generator=g, dtype=torch.float32) * 0.02
    print(f"[P5c worker] quantizing source weights {out}x{inp}", flush=True)
    return original, figquant_quantize(original, group_size=GROUP_SIZE, n_iters=8, double_quant=False)

def tiled_weight(q, tile):
    out, inp = q.shape; parts=[]; gpr=inp//q.group_size
    for r0 in range(0, out, tile):
        r1=min(out,r0+tile); g0=r0*gpr; g1=r1*gpr
        p=q.indices[(g0*q.group_size)//2:(g1*q.group_size+1)//2].to(torch.int64)
        idx=torch.stack((p&15,(p>>4)&15),1).reshape(-1)[:(g1-g0)*q.group_size].reshape(g1-g0,q.group_size).to(torch.int64)
        cb=q.codebook.to(torch.bfloat16).unsqueeze(0).expand(g1-g0,-1)
        w=(torch.gather(cb,1,idx)*q.scales[g0:g1].to(torch.bfloat16).unsqueeze(1)).reshape(r1-r0,inp)
        parts.append(w); del p,idx,cb
    return torch.cat(parts, 0)

def worker(cfg, conn):
    print(f"[P5c worker] start {cfg['layer']} {cfg['variant']} iteration {cfg['iteration']}", flush=True)
    torch.set_num_threads(1); shape=tuple(cfg["shape"]); original,q=make_case(shape)
    x=torch.randn(cfg["batch"],cfg["seq"],shape[1]); ref=F.linear(x, original)
    gc.collect(); start=rss(); peak=start; stop=threading.Event()
    def sample():
        nonlocal peak
        while not stop.wait(.005): peak=max(peak,rss())
    t=threading.Thread(target=sample,daemon=True); t.start(); t0=time.perf_counter()
    if cfg["variant"] == "v1_fp32_full": w=figquant_dequantize(q); y=F.linear(x, w)
    elif cfg["variant"] == "v2_bf16_full":
        w=figquant_dequantize(q).to(torch.bfloat16); y=F.linear(x.to(w.dtype),w)
    else:
        w=tiled_weight(q,cfg["tile"]); y=F.linear(x.to(w.dtype),w)
    elapsed=(time.perf_counter()-t0)*1000; stop.set(); t.join(); peak=max(peak,rss())
    output_err=(y.float()-ref.float()).abs(); output_relative=output_err/ref.float().abs().clamp_min(1e-8)
    weight_err=(w.float()-original).abs(); weight_rmse=weight_err.pow(2).mean().sqrt().item()
    after=rss(); del y,w,ref,original,q,x; gc.collect(); after_gc=rss(); trim_ret=trim(); after_trim=rss()
    conn.send({"variant":cfg["variant"],"shape":list(shape),"rss_start_mib":start,"rss_peak_mib":peak,"rss_after_gc_mib":after_gc,"rss_after_trim_mib":after_trim,"malloc_trim_return":trim_ret,"wall_ms":elapsed,"weight_rmse":weight_rmse,"weight_mse":weight_err.pow(2).mean().item(),"output_level_max_abs_error_informational":output_err.max().item(),"output_level_rmse_informational":output_err.pow(2).mean().sqrt().item(),"output_level_max_relative_error_informational":output_relative.max().item()}); print(f"[P5c worker] done {cfg['layer']} {cfg['variant']} iteration {cfg['iteration']} ({elapsed:.1f} ms)", flush=True); conn.close()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--iterations",type=int,default=3); ap.add_argument("--batch",type=int,default=2); ap.add_argument("--seq",type=int,default=256); ap.add_argument("--tile",type=int,default=128); ap.add_argument("--results-path",default="benchmark/p5c_dequant_variants_results.json"); ap.add_argument("--progress-path",default=None); ap.add_argument("--error-tol",type=float,default=0.08); ap.add_argument("--case-timeout",type=float,default=900.0, help="maximum seconds per isolated worker")
    a=ap.parse_args(); progress=a.progress_path or a.results_path + ".jsonl"; rows=[]; ctx=mp.get_context("spawn")
    os.makedirs(os.path.dirname(os.path.abspath(progress)),exist_ok=True)
    if os.path.exists(progress):
        with open(progress,encoding="utf-8") as f:
            for line in f:
                try: event=json.loads(line)
                except json.JSONDecodeError: continue
                if event.get("event") == "case_complete":
                    event.pop("event",None); rows.append(event)
        print(f"[P5c] resumed {len(rows)} completed case(s) from {progress}",flush=True)
    for layer,shape in SHAPES.items():
        for v in ("v1_fp32_full","v2_bf16_full","v3_bf16_tiled"):
            for i in range(a.iterations):
                print(f"[P5c] running {layer} {v} iteration {i+1}/{a.iterations}", flush=True)
                key=(layer,v,i+1)
                if any((r.get("layer"),r.get("variant"),r.get("iteration")) == key for r in rows):
                    print(f"[P5c] skip completed {layer} {v} iteration {i+1}",flush=True); continue
                append_jsonl(progress,{"event":"case_started","unix_s":time.time(),"layer":layer,"variant":v,"iteration":i+1,"shape":list(shape)})
                parent,child=ctx.Pipe(False); p=ctx.Process(target=worker,args=({"shape":shape,"variant":v,"batch":a.batch,"seq":a.seq,"tile":a.tile,"iteration":i+1,"layer":layer},child)); p.start(); child.close()
                if not parent.poll(a.case_timeout):
                    if p.is_alive(): p.terminate()
                    p.join(30)
                    raise TimeoutError(f"P5c worker timed out after {a.case_timeout:.0f}s: {layer} {v} iteration {i+1}")
                row=parent.recv(); p.join(30)
                if p.exitcode != 0: raise RuntimeError(f"P5c worker failed with exit code {p.exitcode}: {layer} {v} iteration {i+1}")
                row.update(layer=layer,iteration=i+1,correctness_pass=row["weight_rmse"]<=a.error_tol); rows.append(row); print(f"[P5c] collected {len(rows)} case(s)", flush=True)
                append_jsonl(progress, {"event":"case_complete", **row})
    out={"scope":"P5c isolated synthetic FigQuant variants","correctness_reference":"unquantized FP32 weights","error_tolerance_max_abs":a.error_tol,"iterations":a.iterations,"layer_shapes":{k:list(v) for k,v in SHAPES.items()},"cases":rows}; os.makedirs(os.path.dirname(os.path.abspath(a.results_path)),exist_ok=True); json.dump(out,open(a.results_path,"w"),indent=2); print(json.dumps(out,indent=2))
if __name__ == "__main__": main()
