import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, mean_squared_error
from constants import TOPO_MAP_REV, COLOR_MAP_TOPO

def save_loss_curve(train_hist, val_hist, save_path):
    plt.figure(figsize=(7, 5))
    plt.plot(train_hist, label='Train Loss', color='royalblue')
    plt.plot(val_hist, label='Val Loss', color='orange')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(False) 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_parity_plot(model, loader, save_path, device, title=r"Parity Plot", y_attr='y'):
    model.eval()
    all_preds_raw, all_trues_raw, all_topos = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            # Handle MTL or special outputs if necessary, but here we assume single regressor output
            if isinstance(out, (tuple, list)):
                preds = out[0].cpu().numpy()
            else:
                preds = out.cpu().numpy()
                
            trues = getattr(batch, y_attr).cpu().numpy()
            all_preds_raw.append(preds.reshape(-1, 1))
            all_trues_raw.append(trues.reshape(-1, 1))
            
            if hasattr(batch, 'topo_type'):
                all_topos.extend(batch.topo_type.cpu().numpy().flatten())

    y_pred = 10 ** np.vstack(all_preds_raw).flatten()
    y_true = 10 ** np.vstack(all_trues_raw).flatten()
    all_topos = np.array(all_topos)
    
    mape_data = []
    fig, ax = plt.subplots(figsize=(10, 7), dpi=100)
    
    unique_topos = np.unique(all_topos)
    for topo_idx in unique_topos:
        mask = (all_topos == topo_idx)
        if not np.any(mask): continue
        
        t_true = y_true[mask]
        t_pred = y_pred[mask]
        
        t_mape = np.mean(np.abs((t_pred - t_true) / t_true)) * 100
        t_r2 = r2_score(t_true, t_pred)
        t_rmse = np.sqrt(mean_squared_error(t_true, t_pred))
        
        topo_name = TOPO_MAP_REV.get(int(topo_idx), f"Unknown-{topo_idx}")
        
        mape_data.append({
            "Topology": topo_name,
            "Samples": len(t_true),
            "MAPE(%)": round(t_mape, 4),
            "R2": round(t_r2, 4),
            "RMSE": round(t_rmse, 4)
        })
        
        MAX_PTS = 15000 
        if len(t_true) > MAX_PTS:
            idx = np.random.choice(len(t_true), MAX_PTS, replace=False)
            p_true, p_pred = t_true[idx], t_pred[idx]
        else:
            p_true, p_pred = t_true, t_pred

        ax.scatter(p_true, p_pred, c=COLOR_MAP_TOPO.get(int(topo_idx), 'black'), 
                   label=f"{topo_name} (MAPE: {t_mape:.1f}%, R²: {t_r2:.2f})", 
                   alpha=0.6, edgecolors='black', linewidths=0.5, s=25)

    df_mape = pd.DataFrame(mape_data)
    csv_out_path = os.path.join(os.path.dirname(save_path), "topology_mape_stats.csv")
    df_mape.to_csv(csv_out_path, index=False)

    total_mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    total_r2 = r2_score(y_true, y_pred)
    total_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', alpha=0.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('True Value')
    ax.set_ylabel('Predicted Value')
    ax.set_title(f"{title}\nTotal Test MAPE: {total_mape:.2f}% | R²: {total_r2:.4f} | RMSE: {total_rmse:.4f}")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    return total_mape

def save_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6), dpi=150)
    labels = [TOPO_MAP_REV.get(i, str(i)) for i in range(len(np.unique(y_true)))]
    # Use standard labels if 5 classes
    if len(labels) == 5:
        labels = [TOPO_MAP_REV[i] for i in range(5)]
        
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Topology')
    plt.ylabel('True Topology')
    plt.title('Topology Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def evaluate_and_plot_mtl(model, loader, device, result_root):
    model.eval()
    preds_chiN, trues_chiN = [], []
    preds_kstar, trues_kstar = [], []
    preds_cls, trues_cls = [], []
    all_topos = []
    
    total_rec_A_acc = 0.0
    total_rec_F_mse = 0.0
    total_rec_E_mse = 0.0
    total_graphs = 0
    
    criterion_reg = torch.nn.MSELoss()
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            out_chiN, out_kstar, out_cls, (out_A, out_F, out_E) = out
            
            preds_chiN.append(out_chiN.cpu().numpy().reshape(-1, 1))
            trues_chiN.append(batch.chiN_star.cpu().numpy().reshape(-1, 1))
            preds_kstar.append(out_kstar.cpu().numpy().reshape(-1, 1))
            trues_kstar.append(batch.k_star.cpu().numpy().reshape(-1, 1))

            pred_classes = torch.argmax(out_cls, dim=1).cpu().numpy()
            true_classes = batch.topo_type.flatten().cpu().numpy()
            preds_cls.extend(pred_classes)
            trues_cls.extend(true_classes)
            all_topos.extend(true_classes)
                
            # Reconstruction evaluation
            target_adj = batch.dense_adj.view(-1, 32, 32)
            target_F = batch.dense_F.view(-1, 32, 3)
            target_E = batch.dense_edge_attr.view(-1, 32, 32, 2)
            
            pred_adj_bin = (out_A > 0.5).float()
            total_rec_A_acc += (pred_adj_bin == target_adj).float().mean().item() * batch.num_graphs
            total_rec_F_mse += criterion_reg(out_F, target_F).item() * batch.num_graphs
            total_rec_E_mse += criterion_reg(out_E, target_E).item() * batch.num_graphs
            total_graphs += batch.num_graphs

    # Plot Regressions
    y_pred_chiN = 10 ** np.vstack(preds_chiN).flatten()
    y_true_chiN = 10 ** np.vstack(trues_chiN).flatten()
    save_parity_plot_data(y_true_chiN, y_pred_chiN, np.array(all_topos), 
                         os.path.join(result_root, "parity_plot_chiN.png"), "chiN*")
    
    y_pred_kstar = 10 ** np.vstack(preds_kstar).flatten()
    y_true_kstar = 10 ** np.vstack(trues_kstar).flatten()
    save_parity_plot_data(y_true_kstar, y_pred_kstar, np.array(all_topos), 
                         os.path.join(result_root, "parity_plot_kstar.png"), "k*")
    
    # Plot Classification
    cls_acc = accuracy_score(trues_cls, preds_cls) * 100
    save_confusion_matrix(trues_cls, preds_cls, os.path.join(result_root, "confusion_matrix_cls.png"))
    
    # Report
    rec_A_acc = (total_rec_A_acc / total_graphs) * 100
    rec_F_mse = total_rec_F_mse / total_graphs
    rec_E_mse = total_rec_E_mse / total_graphs
    
    return {
        "chiN_mape": np.mean(np.abs((y_pred_chiN - y_true_chiN) / y_true_chiN)) * 100,
        "kstar_mape": np.mean(np.abs((y_pred_kstar - y_true_kstar) / y_true_kstar)) * 100,
        "cls_acc": cls_acc,
        "rec_A_acc": rec_A_acc,
        "rec_F_mse": rec_F_mse,
        "rec_E_mse": rec_E_mse
    }

def evaluate_and_plot_dual(model, loader, device, result_root):
    model.eval()
    preds_chiN, trues_chiN = [], []
    preds_kstar, trues_kstar = [], []
    all_topos = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out_chiN, out_kstar = model(batch)
            
            preds_chiN.append(out_chiN.cpu().numpy().reshape(-1, 1))
            trues_chiN.append(batch.chiN_star.cpu().numpy().reshape(-1, 1))
            
            preds_kstar.append(out_kstar.cpu().numpy().reshape(-1, 1))
            trues_kstar.append(batch.k_star.cpu().numpy().reshape(-1, 1))
            
            if hasattr(batch, 'topo_type'):
                all_topos.extend(batch.topo_type.flatten().cpu().numpy())

    y_pred_chiN = 10 ** np.vstack(preds_chiN).flatten()
    y_true_chiN = 10 ** np.vstack(trues_chiN).flatten()
    save_parity_plot_data(y_true_chiN, y_pred_chiN, np.array(all_topos), 
                         os.path.join(result_root, "parity_plot_chiN.png"), "chiN*")
    
    y_pred_kstar = 10 ** np.vstack(preds_kstar).flatten()
    y_true_kstar = 10 ** np.vstack(trues_kstar).flatten()
    save_parity_plot_data(y_true_kstar, y_pred_kstar, np.array(all_topos), 
                         os.path.join(result_root, "parity_plot_kstar.png"), "k*")
    
    return {
        "chiN_mape": np.mean(np.abs((y_pred_chiN - y_true_chiN) / y_true_chiN)) * 100,
        "kstar_mape": np.mean(np.abs((y_pred_kstar - y_true_kstar) / y_true_kstar)) * 100
    }

def evaluate_and_plot_chiN_recon(model, loader, device, result_root):
    model.eval()
    preds_chiN, trues_chiN = [], []
    all_topos = []
    
    total_rec_A_acc, total_rec_F_mse, total_rec_E_mse, total_graphs = 0.0, 0.0, 0.0, 0
    criterion_reg = torch.nn.MSELoss()
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out_chiN, (out_A, out_F, out_E) = model(batch)
            
            preds_chiN.append(out_chiN.cpu().numpy().reshape(-1, 1))
            trues_chiN.append(batch.chiN_star.cpu().numpy().reshape(-1, 1))
            
            if hasattr(batch, 'topo_type'):
                all_topos.extend(batch.topo_type.flatten().cpu().numpy())
                
            target_adj = batch.dense_adj.view(-1, 32, 32)
            target_F = batch.dense_F.view(-1, 32, 3)
            target_E = batch.dense_edge_attr.view(-1, 32, 32, 2)
            
            pred_adj_bin = (out_A > 0.5).float()
            total_rec_A_acc += (pred_adj_bin == target_adj).float().mean().item() * batch.num_graphs
            total_rec_F_mse += criterion_reg(out_F, target_F).item() * batch.num_graphs
            total_rec_E_mse += criterion_reg(out_E, target_E).item() * batch.num_graphs
            total_graphs += batch.num_graphs

    y_pred_chiN = 10 ** np.vstack(preds_chiN).flatten()
    y_true_chiN = 10 ** np.vstack(trues_chiN).flatten()
    save_parity_plot_data(y_true_chiN, y_pred_chiN, np.array(all_topos), 
                         os.path.join(result_root, "parity_plot_chiN.png"), "chiN*")
    
    return {
        "chiN_mape": np.mean(np.abs((y_pred_chiN - y_true_chiN) / y_true_chiN)) * 100,
        "rec_A_acc": (total_rec_A_acc / total_graphs) * 100,
        "rec_F_mse": total_rec_F_mse / total_graphs,
        "rec_E_mse": total_rec_E_mse / total_graphs
    }

def evaluate_and_plot_k_recon(model, loader, device, result_root):
    model.eval()
    preds_kstar, trues_kstar = [], []
    all_topos = []
    
    total_rec_A_acc, total_rec_F_mse, total_rec_E_mse, total_graphs = 0.0, 0.0, 0.0, 0
    criterion_reg = torch.nn.MSELoss()
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out_kstar, (out_A, out_F, out_E) = model(batch)
            
            preds_kstar.append(out_kstar.cpu().numpy().reshape(-1, 1))
            trues_kstar.append(batch.k_star.cpu().numpy().reshape(-1, 1))
            
            if hasattr(batch, 'topo_type'):
                all_topos.extend(batch.topo_type.flatten().cpu().numpy())
                
            target_adj = batch.dense_adj.view(-1, 32, 32)
            target_F = batch.dense_F.view(-1, 32, 3)
            target_E = batch.dense_edge_attr.view(-1, 32, 32, 2)
            
            pred_adj_bin = (out_A > 0.5).float()
            total_rec_A_acc += (pred_adj_bin == target_adj).float().mean().item() * batch.num_graphs
            total_rec_F_mse += criterion_reg(out_F, target_F).item() * batch.num_graphs
            total_rec_E_mse += criterion_reg(out_E, target_E).item() * batch.num_graphs
            total_graphs += batch.num_graphs

    y_pred_kstar = 10 ** np.vstack(preds_kstar).flatten()
    y_true_kstar = 10 ** np.vstack(trues_kstar).flatten()
    save_parity_plot_data(y_true_kstar, y_pred_kstar, np.array(all_topos), 
                         os.path.join(result_root, "parity_plot_kstar.png"), "k*")
    
    return {
        "kstar_mape": np.mean(np.abs((y_pred_kstar - y_true_kstar) / y_true_kstar)) * 100,
        "rec_A_acc": (total_rec_A_acc / total_graphs) * 100,
        "rec_F_mse": total_rec_F_mse / total_graphs,
        "rec_E_mse": total_rec_E_mse / total_graphs
    }

def evaluate_and_plot_chiN_cls(model, loader, device, result_root):
    model.eval()
    preds_chiN, trues_chiN = [], []
    preds_cls, trues_cls = [], []
    all_topos = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out_chiN, out_cls = model(batch)
            
            preds_chiN.append(out_chiN.cpu().numpy().reshape(-1, 1))
            trues_chiN.append(batch.chiN_star.cpu().numpy().reshape(-1, 1))
            
            pred_classes = torch.argmax(out_cls, dim=1).cpu().numpy()
            true_classes = batch.topo_type.flatten().cpu().numpy()
            preds_cls.extend(pred_classes)
            trues_cls.extend(true_classes)
            all_topos.extend(true_classes)

    y_pred_chiN = 10 ** np.vstack(preds_chiN).flatten()
    y_true_chiN = 10 ** np.vstack(trues_chiN).flatten()
    save_parity_plot_data(y_true_chiN, y_pred_chiN, np.array(all_topos), 
                         os.path.join(result_root, "parity_plot_chiN.png"), "chiN*")
                         
    from sklearn.metrics import accuracy_score
    cls_acc = accuracy_score(trues_cls, preds_cls) * 100
    from .viz_utils import save_confusion_matrix # or since this is viz_utils, just call it directly
    save_confusion_matrix(trues_cls, preds_cls, os.path.join(result_root, "confusion_matrix_cls.png"))
    
    return {
        "chiN_mape": np.mean(np.abs((y_pred_chiN - y_true_chiN) / y_true_chiN)) * 100,
        "cls_acc": cls_acc
    }

def evaluate_and_plot_k_cls(model, loader, device, result_root):
    model.eval()
    preds_kstar, trues_kstar = [], []
    preds_cls, trues_cls = [], []
    all_topos = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out_kstar, out_cls = model(batch)
            
            preds_kstar.append(out_kstar.cpu().numpy().reshape(-1, 1))
            trues_kstar.append(batch.k_star.cpu().numpy().reshape(-1, 1))
            
            pred_classes = torch.argmax(out_cls, dim=1).cpu().numpy()
            true_classes = batch.topo_type.flatten().cpu().numpy()
            preds_cls.extend(pred_classes)
            trues_cls.extend(true_classes)
            all_topos.extend(true_classes)

    y_pred_kstar = 10 ** np.vstack(preds_kstar).flatten()
    y_true_kstar = 10 ** np.vstack(trues_kstar).flatten()
    save_parity_plot_data(y_true_kstar, y_pred_kstar, np.array(all_topos), 
                         os.path.join(result_root, "parity_plot_kstar.png"), "k*")
                         
    from sklearn.metrics import accuracy_score
    cls_acc = accuracy_score(trues_cls, preds_cls) * 100
    save_confusion_matrix(trues_cls, preds_cls, os.path.join(result_root, "confusion_matrix_cls.png"))
    
    return {
        "kstar_mape": np.mean(np.abs((y_pred_kstar - y_true_kstar) / y_true_kstar)) * 100,
        "cls_acc": cls_acc
    }

def evaluate_and_plot_dual_recon(model, loader, device, result_root):
    model.eval()
    preds_chiN, trues_chiN = [], []
    preds_kstar, trues_kstar = [], []
    all_topos = []
    
    total_rec_A_acc, total_rec_F_mse, total_rec_E_mse, total_graphs = 0.0, 0.0, 0.0, 0
    criterion_reg = torch.nn.MSELoss()
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out_chiN, out_kstar, (out_A, out_F, out_E) = model(batch)
            
            preds_chiN.append(out_chiN.cpu().numpy().reshape(-1, 1))
            trues_chiN.append(batch.chiN_star.cpu().numpy().reshape(-1, 1))
            
            preds_kstar.append(out_kstar.cpu().numpy().reshape(-1, 1))
            trues_kstar.append(batch.k_star.cpu().numpy().reshape(-1, 1))
            
            if hasattr(batch, 'topo_type'):
                all_topos.extend(batch.topo_type.flatten().cpu().numpy())
                
            target_adj = batch.dense_adj.view(-1, 32, 32)
            target_F = batch.dense_F.view(-1, 32, 3)
            target_E = batch.dense_edge_attr.view(-1, 32, 32, 2)
            
            pred_adj_bin = (out_A > 0.5).float()
            total_rec_A_acc += (pred_adj_bin == target_adj).float().mean().item() * batch.num_graphs
            total_rec_F_mse += criterion_reg(out_F, target_F).item() * batch.num_graphs
            total_rec_E_mse += criterion_reg(out_E, target_E).item() * batch.num_graphs
            total_graphs += batch.num_graphs

    y_pred_chiN = 10 ** np.vstack(preds_chiN).flatten()
    y_true_chiN = 10 ** np.vstack(trues_chiN).flatten()
    save_parity_plot_data(y_true_chiN, y_pred_chiN, np.array(all_topos), 
                         os.path.join(result_root, "parity_plot_chiN.png"), "chiN*")
                         
    y_pred_kstar = 10 ** np.vstack(preds_kstar).flatten()
    y_true_kstar = 10 ** np.vstack(trues_kstar).flatten()
    save_parity_plot_data(y_true_kstar, y_pred_kstar, np.array(all_topos), 
                         os.path.join(result_root, "parity_plot_kstar.png"), "k*")
    
    return {
        "chiN_mape": np.mean(np.abs((y_pred_chiN - y_true_chiN) / y_true_chiN)) * 100,
        "kstar_mape": np.mean(np.abs((y_pred_kstar - y_true_kstar) / y_true_kstar)) * 100,
        "rec_A_acc": (total_rec_A_acc / total_graphs) * 100,
        "rec_F_mse": total_rec_F_mse / total_graphs,
        "rec_E_mse": total_rec_E_mse / total_graphs
    }

def evaluate_and_plot_dual_cls(model, loader, device, result_root):
    model.eval()
    preds_chiN, trues_chiN = [], []
    preds_kstar, trues_kstar = [], []
    preds_cls, trues_cls = [], []
    all_topos = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out_chiN, out_kstar, out_cls = model(batch)
            
            preds_chiN.append(out_chiN.cpu().numpy().reshape(-1, 1))
            trues_chiN.append(batch.chiN_star.cpu().numpy().reshape(-1, 1))
            
            preds_kstar.append(out_kstar.cpu().numpy().reshape(-1, 1))
            trues_kstar.append(batch.k_star.cpu().numpy().reshape(-1, 1))
            
            pred_classes = torch.argmax(out_cls, dim=1).cpu().numpy()
            true_classes = batch.topo_type.flatten().cpu().numpy()
            preds_cls.extend(pred_classes)
            trues_cls.extend(true_classes)
            all_topos.extend(true_classes)

    y_pred_chiN = 10 ** np.vstack(preds_chiN).flatten()
    y_true_chiN = 10 ** np.vstack(trues_chiN).flatten()
    save_parity_plot_data(y_true_chiN, y_pred_chiN, np.array(all_topos), 
                         os.path.join(result_root, "parity_plot_chiN.png"), "chiN*")
                         
    y_pred_kstar = 10 ** np.vstack(preds_kstar).flatten()
    y_true_kstar = 10 ** np.vstack(trues_kstar).flatten()
    save_parity_plot_data(y_true_kstar, y_pred_kstar, np.array(all_topos), 
                         os.path.join(result_root, "parity_plot_kstar.png"), "k*")
                         
    from sklearn.metrics import accuracy_score
    cls_acc = accuracy_score(trues_cls, preds_cls) * 100
    save_confusion_matrix(trues_cls, preds_cls, os.path.join(result_root, "confusion_matrix_cls.png"))
    
    return {
        "chiN_mape": np.mean(np.abs((y_pred_chiN - y_true_chiN) / y_true_chiN)) * 100,
        "kstar_mape": np.mean(np.abs((y_pred_kstar - y_true_kstar) / y_true_kstar)) * 100,
        "cls_acc": cls_acc
    }

def save_parity_plot_data(y_true, y_pred, all_topos, save_path, label_name):
    return _plot_generic_parity(y_true, y_pred, all_topos, save_path, title=f"Parity Plot for {label_name}")

def _plot_generic_parity(y_true, y_pred, all_topos, save_path, title):
    mape_data = []
    fig, ax = plt.subplots(figsize=(10, 7), dpi=100)
    unique_topos = np.unique(all_topos)
    for topo_idx in unique_topos:
        mask = (all_topos == topo_idx)
        if not np.any(mask): continue
        t_true, t_pred = y_true[mask], y_pred[mask]
        t_mape = np.mean(np.abs((t_pred - t_true) / t_true)) * 100
        t_r2 = r2_score(t_true, t_pred)
        t_rmse = np.sqrt(mean_squared_error(t_true, t_pred))
        topo_name = TOPO_MAP_REV.get(int(topo_idx), f"Unknown-{topo_idx}")
        mape_data.append({"Topology": topo_name, "Samples": len(t_true), "MAPE(%)": round(t_mape, 4), "R2": round(t_r2, 4), "RMSE": round(t_rmse, 4)})
        MAX_PTS = 15000 
        p_true, p_pred = (t_true[np.random.choice(len(t_true), MAX_PTS, replace=False)], t_pred[np.random.choice(len(t_true), MAX_PTS, replace=False)]) if len(t_true) > MAX_PTS else (t_true, t_pred)
        ax.scatter(p_true, p_pred, c=COLOR_MAP_TOPO.get(int(topo_idx), 'black'), label=f"{topo_name} (MAPE: {t_mape:.1f}%, R²: {t_r2:.2f})", alpha=0.6, edgecolors='black', linewidths=0.5, s=25)
    
    pd.DataFrame(mape_data).to_csv(save_path.replace('.png', '_stats.csv'), index=False)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', alpha=0.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title(f"{title}\nTotal Test MAPE: {np.mean(np.abs((y_pred - y_true) / y_true))*100:.2f}%")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight', dpi=200); plt.close()
