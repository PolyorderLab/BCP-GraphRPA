#!/usr/bin/env python3
"""Evaluate frozen joint-selected checkpoints on the sealed 2k final test."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE=Path(__file__).resolve().parent; MLP=HERE.parent; SEARCH=MLP/"hyperparameter_selection_80k"
PROJECT=HERE.parents[3]; TEST_DIR=PROJECT/"TrainData/data/chiNlimited/n_test_2k_leakage_free_seed42"
TEST_CSV=MLP/"final_test_evaluation/test_descriptor_matrix.csv"
FEATURES=["SLE","r_deg","CV_arm","n_leaves","M_AB","IR","D_f","ISC","CAF","tau_mean","f_A"]
SEEDS=(202601,202602,202603,202604,202605)
sys.path.insert(0,str(SEARCH)); from mlp_validation_worker import MLP as Network
sys.path.insert(0,str(MLP/"descriptor_selection_80k")); from analyze_descriptors import extract_split

def metrics(t,p):
    tr,pr=10.0**t,10.0**p; rel=np.abs(pr-tr)/np.clip(np.abs(tr),1e-12,None)*100
    return {"test_mape_percent":float(rel.mean()),"test_relative_error_p95_percent":float(np.percentile(rel,95)),
            "test_rmse_log10":float(np.sqrt(mean_squared_error(t,p))),"test_mae_log10":float(mean_absolute_error(t,p)),
            "test_r2_log10":float(r2_score(t,p)),"test_rmse_raw":float(np.sqrt(mean_squared_error(tr,pr))),
            "test_mae_raw":float(mean_absolute_error(tr,pr)),"test_r2_raw":float(r2_score(tr,pr))}

def main():
    status=json.loads((HERE/"training_status.json").read_text())
    if status["completed"] != 5 or status["failed"]: raise RuntimeError("Five joint checkpoints are not frozen.")
    if not TEST_CSV.exists(): extract_split(TEST_DIR,"test",workers=8,limit=None).to_csv(TEST_CSV,index=False)
    test=pd.read_csv(TEST_CSV,usecols=[*FEATURES,"log10_chiN_star"])
    if len(test)!=2000: raise ValueError(f"Expected 2000 test rows, got {len(test)}")
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); rows=[]
    for seed in SEEDS:
        cp=HERE/"checkpoints"/f"joint_selected__seed{seed}.pt"; payload=torch.load(cp,map_location="cpu",weights_only=False)
        x=(test[FEATURES].to_numpy(np.float32)-np.asarray(payload["scaler_mean"],np.float32))/np.asarray(payload["scaler_scale"],np.float32)
        y=test["log10_chiN_star"].to_numpy(np.float32); model=Network(len(FEATURES),256,3,0.0)
        model.load_state_dict(payload["model_state_dict"]); model.to(device).eval()
        with torch.no_grad(): pred=model(torch.from_numpy(x.astype(np.float32)).to(device)).cpu().numpy().ravel()
        rows.append({"seed":seed,"n_test":len(test),**metrics(y,pred)}); np.savez(HERE/f"test_predictions_seed{seed}.npz",y_true_log=y,y_pred_log=pred)
    frame=pd.DataFrame(rows); frame.to_csv(HERE/"joint_selected_test_metrics.csv",index=False)
    summary={"configuration":{"batch_size":256,"lr":0.0005,"hidden_dim":256,"depth":3,"dropout":0.0,"weight_decay":1e-5},
             "test_source_dir":str(TEST_DIR),"test_size":2000,"n_seeds":5,
             "test_mape_mean_percent":float(frame.test_mape_percent.mean()),"test_mape_sd_percent":float(frame.test_mape_percent.std(ddof=1)),
             "test_mape_per_seed_percent":frame.test_mape_percent.tolist(),"selection_status":"joint configuration and checkpoints frozen before test access"}
    (HERE/"joint_selected_test_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2),flush=True)

if __name__ == "__main__": main()
