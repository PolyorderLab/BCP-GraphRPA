import torch
import torch.nn as nn

def freeze_encoder(model, freeze=True):
    """
    Freeze or unfreeze the entire encoder.
    """
    for param in model.encoder.parameters():
        param.requires_grad = not freeze

def freeze_encoder_layers(model, num_layers_to_freeze):
    """
    Freeze the first `num_layers_to_freeze` of the GNNEncoder.
    Assuming model.encoder.convs is a ModuleList and model.encoder.edge_lin is a ModuleList.
    """
    # First freeze everything in encoder
    freeze_encoder(model, freeze=True)
    total_layers = model.encoder.num_layers
    
    # Ensure num_layers_to_freeze is within bounds
    num_layers_to_freeze = min(max(num_layers_to_freeze, 0), total_layers)
    
    # Unfreeze layers from num_layers_to_freeze to total_layers - 1
    for i in range(num_layers_to_freeze, total_layers):
        for param in model.encoder.convs[i].parameters():
            param.requires_grad = True
        for param in model.encoder.edge_lin[i].parameters():
            param.requires_grad = True
            
def get_fine_tune_optimizer(model, encoder_lr=1e-4, head_lr=1e-3, weight_decay=0):
    """
    Returns an Adam optimizer with different learning rates for the encoder and the heads.
    """
    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('encoder.'):
            encoder_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = torch.optim.Adam([
        {'params': encoder_params, 'lr': encoder_lr},
        {'params': head_params, 'lr': head_lr}
    ], weight_decay=weight_decay)
    return optimizer

def load_pretrained_encoder(model, checkpoint_path, device, load_heads=True):
    """
    Loads pretrained weights into the model. 
    It loads the 'encoder.' keys and optionally matching heads keys.
    """
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_state = model.state_dict()
    
    loaded_keys = []
    for name, param in state_dict.items():
        if name.startswith('encoder.') and name in model_state:
            if model_state[name].shape == param.shape:
                model_state[name].copy_(param)
                loaded_keys.append(name)
            else:
                print(f"Shape mismatch for {name}: {model_state[name].shape} vs {param.shape}")
                
    if load_heads:
        for name, param in state_dict.items():
            if not name.startswith('encoder.') and name in model_state:
                if model_state[name].shape == param.shape:
                    model_state[name].copy_(param)
                    loaded_keys.append(name)
                    
    print(f"Loaded {len(loaded_keys)} keys from pretrained model {checkpoint_path}.")
    return model
