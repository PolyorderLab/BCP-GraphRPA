#!/usr/bin/env python3
"""Validation-only descriptor and training-pool ablations.

Every task selects its checkpoint by validation MSE and reports validation
MAPE. The external 2k test set is deliberately absent from this runner.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEEDS = (202601, 202602, 202603, 202604, 202605)
FEATURES = ("SLE", "r_deg", "CV_arm", "n_leaves", "M_AB", "IR", "D_f", "ISC", "CAF", "tau_mean", "f_A")
GROUPS = {
    "composition_length": ("f_A", "SLE"),
    "interface": ("ISC", "M_AB", "IR", "CAF"),
    "topology_asymmetry": ("D_f", "CV_arm", "tau_mean"),
    "graph_topology": ("r_deg", "n_leaves"),
}
CFG = {"hidden_dim": 256, "depth": 3, "dropout": 0.0, "lr": 0.0005, "batch_size": 256, "weight_decay": 1e-5}
POOL_SIZES = (1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000)

def split_for(n, seed_path):
    idx = np.arange(n)
    tr, va = train_test_split(idx, test_size=0.2, random_state=42)
    target = Path(seed_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        old = np.load(target)
        if not np.array_equal(old["train_index"], tr) or not np.array_equal(old["val_index"], va):
            raise ValueError(f"Existing split differs: {target}")
    else:
        np.savez(target, train_index=tr, val_index=va, random_state=np.array(42))
    return target

def descriptor_tasks():
    settings = [("full_11", "reference", FEATURES)]
    for group, members in GROUPS.items():
        settings += [(f"only_{group}", "only_group", members),
                     (f"without_{group}", "leave_group_out", tuple(x for x in FEATURES if x not in members))]
    for name in FEATURES:
        settings.append((f"without_{name}", "leave_feature_out", tuple(x for x in FEATURES if x != name)))
    return [{"id":f"desc__{name}__seed{seed}", "family":"descriptor", "dimension":kind,
             "label":name, "features":list(features), "seed":seed, "csv":"train_100k_descriptor_matrix.csv",
             "split":"train_100k_split_indices.npz", "max_train":None} for name,kind,features in settings for seed in SEEDS]

def pool_tasks():
    return [{"id":f"train_pool__{n//1000}k__seed{seed}", "family":"training_pool", "dimension":"train_pool",
             "label":f"{n//1000}k", "features":list(FEATURES), "seed":seed, "csv":f"train_{n//1000}k_descriptor_matrix.csv",
             "split":f"train_{n//1000}k_split_indices.npz", "max_train":None} for n in POOL_SIZES for seed in SEEDS]

def prepare_matrix(project, n):
    sys.path.insert(0, str(project / "baselinemodels/mlp_pub"))
    import analyze_descriptors as mod
    d = project / "baselinemodels/pub0818/mlp/ablation_validation_only/data"; csv = d / f"train_{n//1000}k_descriptor_matrix.csv"
    if not csv.exists():
        frame = mod.extract_split(project / "TrainData/data/chiNlimited" / f"n_train_{n//1000}k", "train", workers=8, limit=None)
        frame.to_csv(csv, index=False)
    split_for(n, d / f"train_{n//1000}k_split_indices.npz")

def main():
    here = Path(__file__).resolve().parent
    root = here.parents[3]
    ap = argparse.ArgumentParser(); ap.add_argument("--gpus", default="0,1,2,3"); ap.add_argument("--slots-per-gpu", type=int, default=3); ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--patience", type=int, default=30); args = ap.parse_args()
    data = here / "data"; data.mkdir(exist_ok=True)
    # Prepare every pool matrix from its own YML/MAT directory; no external test data is touched.
    for n in POOL_SIZES: prepare_matrix(root, n)
    split_for(100000, data / "train_100k_split_indices.npz")
    tasks = descriptor_tasks() + pool_tasks(); out = here / "runs"; logs = here / "logs"; out.mkdir(exist_ok=True); logs.mkdir(exist_ok=True)
    (here / "manifest.json").write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    pending = deque(t for t in tasks if not (out / f"{t['id']}.json").exists()); completed = len(tasks)-len(pending); active=[]; failed=[]
    slots=[int(x) for x in args.gpus.split(",") for _ in range(args.slots_per_gpu)]; worker=here/"validation_ablation_worker.py"; status=here/"status.json"
    while pending or active:
        available=list(slots)
        for item in active:
            if item["proc"].poll() is None and item["gpu"] in available: available.remove(item["gpu"])
        while pending and available:
            t=pending.popleft(); gpu=available.pop(0); out_json=out/f"{t['id']}.json"; log=(logs/f"{t['id']}.log").open("a")
            cmd=[sys.executable,str(worker),"--train-csv",str(data/t["csv"]),"--split-indices",str(data/t["split"]),"--output",str(out_json),"--validation-predictions-output",str(here/"predictions"/f"{t['id']}.npz"),"--checkpoint-output",str(here/"checkpoints"/f"{t['id']}.pt"),"--experiment-id",t["id"],"--dimension",t["dimension"],"--value-label",t["label"],"--features",",".join(t["features"]),"--seed",str(t["seed"]),"--hidden-dim",str(CFG["hidden_dim"]),"--depth",str(CFG["depth"]),"--dropout",str(CFG["dropout"]),"--lr",str(CFG["lr"]),"--batch-size",str(CFG["batch_size"]),"--weight-decay",str(CFG["weight_decay"]),"--epochs",str(args.epochs),"--patience",str(args.patience),"--device","cuda:0"]
            env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu); active.append({"task":t,"gpu":gpu,"proc":subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=root),"log":log})
        time.sleep(3); still=[]
        for item in active:
            code=item["proc"].poll()
            if code is None: still.append(item); continue
            item["log"].close()
            if code != 0 or not (out/f"{item['task']['id']}.json").exists(): failed.append({"id":item['task']['id'],"exit_code":code})
            else: completed += 1
        active=still; status.write_text(json.dumps({"total":len(tasks),"completed":completed,"pending":len(pending),"active":len(active),"failed":failed,"updated_at":time.strftime("%F %T")},indent=2),encoding="utf-8"); print(f"completed={completed}/{len(tasks)} active={len(active)} pending={len(pending)} failed={len(failed)}",flush=True)
    if failed: raise SystemExit(f"failures: {failed}")

if __name__ == "__main__": main()
