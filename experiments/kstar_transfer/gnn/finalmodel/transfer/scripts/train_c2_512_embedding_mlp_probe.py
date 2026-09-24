#!/usr/bin/env python3
"""Frozen C2-512 embedding probe for leakage-free k* transfer evaluation."""
from __future__ import annotations

import argparse, csv, json, os, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader as TensorLoader, Subset, TensorDataset
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))
from src.ira_common import LearnableNodeGINEEncoder
from src.ira_graph_common import SingleTaskRegressor, VirtualNodeWrapper
from src.plotting import compute_metrics

TRANSFER = Path(__file__).resolve().parent.parent
FINAL = ROOT / 'checkpoints/gnn'
TRAIN_CACHE = ROOT / 'results/cache/kstar_train_100k.pt'
TEST_CACHE = ROOT / 'results/cache/kstar_seed43_independent_2k.pt'
SEEDS = (202601, 202602, 202603, 202604, 202605)

class ProbeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512,256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(.1),
                                 nn.Linear(256,128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(.1),
                                 nn.Linear(128,1))
    def forward(self,x): return self.net(x)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

def wrap(data): return VirtualNodeWrapper(data, node_mode='zero', edge_mode='full', graph_mode='original')

def model_for(seed, device):
    m = SingleTaskRegressor(LearnableNodeGINEEncoder(max_nodes=80, edge_dim=2, hidden_dim=512, num_layers=5), hidden_dim=512, num_dec_layers=1).to(device)
    ckpt = FINAL / f'seed{seed}' / 'best_model.pth'
    state = torch.load(ckpt, map_location=device, weights_only=True)
    enc = {k.removeprefix('encoder.'):v for k,v in state.items() if k.startswith('encoder.')}
    m.encoder.load_state_dict(enc, strict=True); m.encoder.eval()
    for p in m.encoder.parameters(): p.requires_grad=False
    return m.encoder, ckpt

def embed(encoder, dataset, device):
    loader=DataLoader(wrap(dataset), batch_size=32, shuffle=False, pin_memory=True)
    xs=[]; ys=[]
    with torch.no_grad():
        for b in loader:
            b=b.to(device); xs.append(encoder(b).cpu()); ys.append(b.y.cpu())
    return torch.cat(xs).numpy(), torch.cat(ys).numpy().reshape(-1,1)

def standardize(a,b,c):
    mean=a.mean(0,keepdims=True); std=np.where(a.std(0,keepdims=True)<1e-8,1.,a.std(0,keepdims=True))
    return (a-mean)/std,(b-mean)/std,(c-mean)/std,mean,std

def train(x,y,vx,vy,seed,device):
    seed_all(seed); m=ProbeMLP().to(device); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4); loss=nn.MSELoss()
    gen=torch.Generator().manual_seed(seed); loader=TensorLoader(TensorDataset(torch.tensor(x).float(),torch.tensor(y).float()),batch_size=512,shuffle=True,generator=gen)
    vx=torch.tensor(vx).float().to(device); vy=torch.tensor(vy).float().to(device)
    best=float('inf'); best_state=None; bad=0; hist=[]
    for epoch in range(1,501):
        m.train(); total=0.
        for bx,by in loader:
            bx=bx.to(device); by=by.to(device); opt.zero_grad(); l=loss(m(bx),by); l.backward(); opt.step(); total+=l.item()*len(bx)
        m.eval()
        with torch.no_grad(): val=float(loss(m(vx),vy))
        hist.append({'epoch':epoch,'train_loss':total/len(x),'val_loss':val})
        if val<best: best=val; best_state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=40: break
    m.load_state_dict(best_state); return m,hist,best

def predict(m,x,device):
    m.eval(); out=[]
    with torch.no_grad():
        for bx in TensorLoader(TensorDataset(torch.tensor(x).float()),batch_size=512): out.append(m(bx[0].to(device)).cpu())
    return torch.cat(out).numpy().reshape(-1)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,choices=SEEDS,required=True); p.add_argument('--run-id',type=int,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--device',default='cuda:0'); args=p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True,exist_ok=True); seed_all(args.seed); device=torch.device(args.device)
    raw=torch.load(TRAIN_CACHE,map_location='cpu',weights_only=False); test=torch.load(TEST_CACHE,map_location='cpu',weights_only=False)
    ti,vi=train_test_split(np.arange(len(raw)),test_size=.2,random_state=42); np.savez(args.output_dir/'split_indices.npz',train_idx=ti,val_idx=vi)
    enc,ckpt=model_for(args.seed,device)
    tx,ty=embed(enc,Subset(raw,ti),device); vx,vy=embed(enc,Subset(raw,vi),device); sx,sy=embed(enc,test,device)
    tx,vx,sx,mean,std=standardize(tx,vx,sx)
    np.savez_compressed(args.output_dir/'c2_512_embeddings.npz',train_x=tx,train_y_log=ty,val_x=vx,val_y_log=vy,test_x=sx,test_y_log=sy,feature_mean=mean,feature_std=std)
    mlp,hist,best=train(tx,ty,vx,vy,args.seed,device); torch.save(mlp.state_dict(),args.output_dir/'best_probe_mlp.pth')
    with (args.output_dir/'probe_loss_history.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['epoch','train_loss','val_loss']); w.writeheader(); w.writerows(hist)
    pred=10**predict(mlp,sx,device); truth=10**sy.reshape(-1); met=compute_metrics(truth,pred); ape=np.abs(pred-truth)/np.clip(np.abs(truth),1e-12,None)*100
    np.savez(args.output_dir/'seed43_predictions.npz',y_true=truth,y_pred=pred,abs_error=np.abs(pred-truth),ape_percent=ape)
    result={'experiment':'C2_512_fixed_embedding_MLP_probe','seed':args.seed,'run_id':args.run_id,'checkpoint':str(ckpt),'embedding_dim':512,'mlp':'512-256-128-1; LayerNorm/ReLU/dropout=0.1','best_val_loss_log_space':best,'epochs_ran':len(hist),'test_data_dir':str(ROOT/'data/datasets/seed43_2k'),'test_size':len(test),'MAPE':float(met['MAPE']),'RMSE':float(met['RMSE']),'MAE':float(met['MAE']),'R2':float(met['R²']),'MaxAE':float(met['MaxAE'])}
    (args.output_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n'); (args.output_dir/'run_config.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
