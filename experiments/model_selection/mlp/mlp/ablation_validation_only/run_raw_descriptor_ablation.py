#!/usr/bin/env python3
"""Validation-only ablation using the complete, non-deduplicated 35-feature set."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from collections import deque
from pathlib import Path

SEEDS = (202601, 202602, 202603, 202604, 202605)
RAW_FEATURES = ("Nnodes", "Nedges", "density", "lambda_alg", "r_deg", "d_nb_avg", "C_bet", "phi_diam", "phi_rad", "wiener", "branch_count", "Gini_f", "f_A", "J_AB", "M_AB", "IR", "ISC", "CAF", "tau_mean", "CV_arm", "D_f", "SLE", "shortest_path_std", "global_efficiency", "n_leaves", "max_degree", "degree_std", "degree_entropy", "laplacian_largest", "adjacency_spectral_radius", "adjacency_spectral_gap", "C_bet_max", "C_bet_std", "C_close_mean", "C_close_std")
CFG = {"hidden_dim": 256, "depth": 3, "dropout": 0.0, "lr": 0.0005, "batch_size": 256, "weight_decay": 1e-5}

def main():
    here = Path(__file__).resolve().parent; project = here.parents[3]; data = here / "data"; out_root = here / "raw_descriptor_ablation"; out = out_root / "runs"; logs = out_root / "logs"; out.mkdir(parents=True, exist_ok=True); logs.mkdir(exist_ok=True)
    tasks=[]
    settings=[("full_35", RAW_FEATURES)] + [(f"without_{name}", tuple(x for x in RAW_FEATURES if x != name)) for name in RAW_FEATURES]
    for name, features in settings:
        for seed in SEEDS: tasks.append({"id":f"rawdesc__{name}__seed{seed}","label":name,"features":features,"seed":seed})
    (out_root / "manifest.json").write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    pending=deque(t for t in tasks if not (out/f"{t['id']}.json").exists()); completed=len(tasks)-len(pending); active=[]; failed=[]; slots=[gpu for gpu in (0, 1, 2, 3) for _ in range(3)]; worker=here/"validation_ablation_worker.py"
    while pending or active:
        available=list(slots)
        for item in active:
            if item["proc"].poll() is None and item["gpu"] in available: available.remove(item["gpu"])
        while pending and available:
            t=pending.popleft(); gpu=available.pop(0); log=(logs/f"{t['id']}.log").open("a"); out_json=out/f"{t['id']}.json"
            cmd=[sys.executable,str(worker),"--train-csv",str(data/"train_100k_descriptor_matrix.csv"),"--split-indices",str(data/"train_100k_split_indices.npz"),"--output",str(out_json),"--validation-predictions-output",str(out_root/"predictions"/f"{t['id']}.npz"),"--checkpoint-output",str(out_root/"checkpoints"/f"{t['id']}.pt"),"--experiment-id",t["id"],"--dimension","leave_one_raw_feature_out","--value-label",t["label"],"--features",",".join(t["features"]),"--seed",str(t["seed"]),"--hidden-dim",str(CFG["hidden_dim"]),"--depth",str(CFG["depth"]),"--dropout",str(CFG["dropout"]),"--lr",str(CFG["lr"]),"--batch-size",str(CFG["batch_size"]),"--weight-decay",str(CFG["weight_decay"]),"--epochs","200","--patience","30","--device","cuda:0"]
            env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu); active.append({"task":t,"gpu":gpu,"proc":subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=project),"log":log})
        time.sleep(3); still=[]
        for item in active:
            code=item["proc"].poll()
            if code is None: still.append(item); continue
            item["log"].close(); (failed if code != 0 or not (out/f"{item['task']['id']}.json").exists() else []).append(item["task"]["id"])
            if code == 0 and (out/f"{item['task']['id']}.json").exists(): completed += 1
        active=still; (out_root/"status.json").write_text(json.dumps({"total":len(tasks),"completed":completed,"pending":len(pending),"active":len(active),"failed":failed,"updated_at":time.strftime("%F %T")},indent=2),encoding="utf-8"); print(f"completed={completed}/{len(tasks)} active={len(active)} pending={len(pending)} failed={len(failed)}",flush=True)
    if failed: raise SystemExit(f"failures: {failed}")

if __name__ == "__main__": main()
