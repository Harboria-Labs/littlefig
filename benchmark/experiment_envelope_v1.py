#!/usr/bin/env python3
"""P6a: project storage/dequant/GEMM envelopes from measured local rates."""
import argparse, json, os, tempfile, time

MODELS = {
    "gemma_class_4b": {"hidden": 2560, "intermediate": 10240, "layers": 34},
    "llama_class_8b": {"hidden": 4096, "intermediate": 14336, "layers": 32},
    "class_26b": {"hidden": 5120, "intermediate": 20480, "layers": 48},
}

# Committed P5c slowdown profile, MLP 5632x2048, batch=2, seq=256.
P5C = {"out": 5632, "inp": 2048, "dequant_s": 0.2842, "gemm_s": 3.2535,
       "batch": 2, "seq": 256, "tile_rows": 128}

def measure_read_bandwidth(size_mib, block_mib, repeats):
    size = size_mib * 1024 * 1024; block = block_mib * 1024 * 1024
    fd, path = tempfile.mkstemp(prefix="littlefig_p6a_", suffix=".bin", dir=os.getcwd())
    os.close(fd)
    try:
        chunk = bytes(min(block, 8 * 1024 * 1024))
        with open(path, "wb", buffering=0) as f:
            left = size
            while left:
                part = chunk[:min(len(chunk), left)]; f.write(part); left -= len(part)
            f.flush(); os.fsync(f.fileno())
        samples=[]
        for _ in range(repeats):
            read=0; started=time.perf_counter()
            with open(path, "rb", buffering=0) as f:
                while True:
                    data=f.read(block)
                    if not data: break
                    read += len(data)
            elapsed=time.perf_counter()-started
            samples.append({"bytes":read,"seconds":elapsed,"gib_s":read/elapsed/2**30})
        return samples
    finally:
        try: os.remove(path)
        except OSError: pass

def layer_matrices(cfg):
    h=cfg["hidden"]; m=cfg["intermediate"]
    return [("q",h,h),("k",h,h),("v",h,h),("o",h,h),
            ("gate",m,h),("up",m,h),("down",h,m)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results-path",default="benchmark/envelope_v1_results.json")
    ap.add_argument("--read-size-mib",type=int,default=512)
    ap.add_argument("--read-block-mib",type=int,default=8)
    ap.add_argument("--read-repeats",type=int,default=3)
    a=ap.parse_args()
    reads=measure_read_bandwidth(a.read_size_mib,a.read_block_mib,a.read_repeats)
    read_gib_s=min(x["gib_s"] for x in reads)  # conservative measured sample
    p5_numel=P5C["out"]*P5C["inp"]
    dequant_weights_s=p5_numel/P5C["dequant_s"]
    p5_flops=2*P5C["batch"]*P5C["seq"]*P5C["out"]*P5C["inp"]
    gemm_flops_s=p5_flops/P5C["gemm_s"]
    results={}
    for name,cfg in MODELS.items():
        io=deq=gemm=packed=weights=0.0
        rows=[]
        for matrix,out,inp in layer_matrices(cfg):
            n=out*inp; packed_bytes=n/2 + (n/128)*4 + 16*4
            io_s=packed_bytes/(read_gib_s*2**30)
            deq_s=n/dequant_weights_s
            flops=2*P5C["batch"]*P5C["seq"]*out*inp
            gemm_s=flops/gemm_flops_s
            rows.append({"matrix":matrix,"shape":[out,inp],"packed_mib":packed_bytes/2**20,
                         "io_s":io_s,"dequant_s":deq_s,"gemm_s":gemm_s})
            io+=io_s; deq+=deq_s; gemm+=gemm_s; packed+=packed_bytes; weights+=n
        io*=cfg["layers"]; deq*=cfg["layers"]; gemm*=cfg["layers"]; packed*=cfg["layers"]
        totals={"io_s":io,"dequant_s":deq,"gemm_s":gemm}
        results[name]={"config":cfg,"per_layer_matrices":rows,"model_packed_gib":packed/2**30,
                       "projected_forward_totals_s":totals,"bottleneck":max(totals,key=totals.get)}
    payload={"scope":"P6a forward-pass envelope; target hardware sequential read plus P5c CPU rates",
             "architecture_assumptions":"Reference decoder-only shapes; projections, not measured model runs.",
             "workload":{"batch":P5C["batch"],"sequence_length":P5C["seq"]},
             "read_bandwidth":{"file_size_mib":a.read_size_mib,"block_mib":a.read_block_mib,
                               "samples":reads,"conservative_gib_s":read_gib_s},
             "measured_rate_sources":{"p5c_profile":P5C,"dequant_weights_per_s":dequant_weights_s,
                                      "gemm_flops_per_s":gemm_flops_s},"models":results}
    os.makedirs(os.path.dirname(os.path.abspath(a.results_path)),exist_ok=True)
    with open(a.results_path,"w",encoding="utf-8") as f: json.dump(payload,f,indent=2)
    print(json.dumps(payload,indent=2))

if __name__ == "__main__": main()
