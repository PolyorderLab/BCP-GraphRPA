#!/usr/bin/env python3
"""Train the frozen joint OFAT-selected configuration for five seeds."""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

SEEDS = (202601, 202602, 202603, 202604, 202605)
FEATURES = "SLE,r_deg,CV_arm,n_leaves,M_AB,IR,D_f,ISC,CAF,tau_mean,f_A"
CONFIG = {"batch_size": 256, "lr": 0.0005, "hidden_dim": 256, "depth": 3,
          "dropout": 0.0, "weight_decay": 1e-5, "epochs": 200, "patience": 30}

def main():
    here = Path(__file__).resolve().parent
    mlp = here.parent
    search = mlp / "hyperparameter_selection_80k"
    source_run = search / "runs_100k_fixed80_20_validation_only"
    worker = search / "mlp_validation_worker.py"
    runs, checkpoints = here / "runs", here / "checkpoints"
    predictions, logs = here / "validation_predictions", here / "logs"
    for path in (runs, checkpoints, predictions, logs): path.mkdir(parents=True, exist_ok=True)
    protocol = {
        "role": "joint confirmation after OFAT values frozen",
        "configuration": CONFIG, "features": FEATURES.split(","), "seeds": list(SEEDS),
        "development_split": "n_train_100k fixed 80k/20k, random_state=42",
        "checkpoint_selection": "validation MSE in log10 chiN*",
        "final_test_policy": "evaluate only after all five checkpoints finish",
    }
    (here / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    processes=[]
    for index, seed in enumerate(SEEDS):
        output=runs/f"joint_selected__seed{seed}.json"
        if output.exists(): continue
        gpu=index if index < 4 else 0
        cmd=[sys.executable,str(worker),"--train-csv",str(search/"data/n_train_100k_descriptor_matrix.csv"),
             "--split-indices",str(source_run/"fixed_80k_20k_split_indices.npz"),
             "--output",str(output),"--checkpoint-output",str(checkpoints/f"joint_selected__seed{seed}.pt"),
             "--validation-predictions-output",str(predictions/f"joint_selected__seed{seed}.npz"),
             "--experiment-id",f"joint_selected__seed{seed}","--dimension","joint_selected_configuration",
             "--value-label","batch256_lr0.0005_hidden256_depth3_dropout0", "--features",FEATURES,
             "--seed",str(seed),"--hidden-dim","256","--depth","3","--dropout","0.0",
             "--lr","0.0005","--batch-size","256","--weight-decay","1e-5",
             "--epochs","200","--patience","30","--device","cuda:0"]
        env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu)
        handle=(logs/f"joint_selected__seed{seed}.log").open("a",encoding="utf-8")
        processes.append((seed,subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT,env=env,cwd=here.parents[3]),handle))
    failures=[]
    for seed,proc,handle in processes:
        code=proc.wait(); handle.close()
        if code or not (runs/f"joint_selected__seed{seed}.json").exists(): failures.append({"seed":seed,"exit_code":code})
    status={"total":5,"completed":len(list(runs.glob("joint_selected__seed*.json"))),"failed":failures}
    (here/"training_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    if failures or status["completed"] != 5: raise SystemExit(f"joint training incomplete: {status}")
    print(json.dumps(status),flush=True)

if __name__ == "__main__": main()
