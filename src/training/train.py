import pickle
from tqdm import tqdm
import torch
from torch.nn import GaussianNLLLoss
from torch.utils.data import DataLoader
from config import BASE_DIR, REPORT_FIGS, DATA_DIR, EXPERIMENTS_DIR, DEVICE
import matplotlib.pyplot as plt
from src.training import WeatherForecastModel
from src.core import WeatherDataset, split_dataset
from torch.optim.lr_scheduler import StepLR


def train_model(model_params, gtk_params):

    (data_folder,
    variables,
    locations,
    context_window,
    rollout_steps,
    stride,
    eps_start,
    eps_end,
    batch_size,
    n_epochs,
    patience,
    run_folder,
    learning_rate,
    weight_decay) = gtk_params

    (d_model,
    num_heads, 
    target_node_idx, 
    max_len,
    num_nodes,
    n_features,
    batch_first,                
    n_graph_layers, 
    out_dim,
    dropout_attn,
    dropout_gcn) = model_params

    model = WeatherForecastModel(model_params)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=patience//2, gamma=0.7)
    criterion = GaussianNLLLoss()

    train_losses = []
    val_losses_1 = []
    val_losses_5 = []
    val_losses_20 = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    dataset = WeatherDataset(data_folder, locations, context_window,rollout_steps,stride,variables)
    train_set, val_set, test_set = split_dataset(dataset, train_frac=0.7, val_frac=0.15, test_frac=0.15)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    
    cache_path = BASE_DIR / "src" / "cache"
    model_save_path = run_folder /  "model.pt"
    plot_save_path = run_folder / "loss_plot.png"
    params_save_path = run_folder / "model_params.pkl"
    gtk_save_path = run_folder / "gtk_params.pkl"

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0.0 

        epsilon = max(eps_end, eps_start - (epoch * (eps_start - eps_end) / 20))

        for context, targets, targets_fourier in tqdm(train_loader, desc=f"Epoch: {epoch+1}"):
            context = context.to(DEVICE)
            targets = targets.to(DEVICE)
            targets_fourier = targets_fourier.to(DEVICE)

            optimizer.zero_grad()
            current_context = context 
            batch_rollout_loss = 0 

            for t in range(rollout_steps):
                
                preds = model(current_context) 

                mid = preds.shape[-1]//2
                mu, log_var = preds[:,:mid], preds[:,mid:]        
                var = torch.exp(log_var)

                step_target = targets[:, t, target_node_idx, :]
                step_loss = criterion(mu, step_target, var)
                batch_rollout_loss += step_loss

                use_teacher_forcing = torch.rand(1).item() < epsilon
                if use_teacher_forcing:
                    next_weather = targets[:, t, :, :] 
                else:
                    next_weather = targets[:, t, :, :].detach().clone() 
                    next_weather[:, target_node_idx, :] = mu.detach()
                
                next_input = torch.cat([next_weather, targets_fourier[:, t, :, :]], dim=-1)
                current_context = torch.cat([current_context[:, 1:], next_input.unsqueeze(1)], dim=1)

                # next_input = torch.cat([next_weather, targets_fourier[:, t, :, :]], dim=-1)
                # current_context = torch.cat([current_context[:, 1:], next_input.unsqueeze(1)], dim=1)
            
            avg_step_loss = batch_rollout_loss / rollout_steps
            avg_step_loss.backward()
            optimizer.step()
            
            total_train_loss += avg_step_loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        scheduler.step()

        model.eval()
        # total_val_loss = 0.0
        val_loss_1 = 0.0
        val_loss_5 = 0.0
        val_loss_20 = 0.0

        with torch.no_grad():

            for context, targets, targets_fourier in val_loader:
                context = context.to(DEVICE)
                targets = targets.to(DEVICE)
                targets_fourier = targets_fourier.to(DEVICE)
                current_context = context
                # val_rollout_loss = 0
                step_losses = []
                
                for t in range(rollout_steps):
                    
                    preds = model(current_context)
                    mid = preds.shape[-1]//2
                    mu, log_var = preds[:,:mid], preds[:,mid:]
                    var = torch.exp(log_var)
                    
                    step_target = targets[:, t, target_node_idx, :]
                    step_loss = criterion(mu, step_target, var)
                    step_losses.append(step_loss.item())

                    next_weather = targets[:,t,:,:].clone()
                    next_weather[:, target_node_idx, :] = mu 

                    next_input = torch.cat([next_weather, targets_fourier[:, t, :, :]], dim=-1)

                    current_context = torch.cat([current_context[:, 1:], next_input.unsqueeze(1)], dim=1)
                
                val_loss_1 += step_losses[0]
                val_loss_5 += sum(step_losses[:5]) / 5
                val_loss_20 += sum(step_losses[:20]) / 20

        val_loss_1 /= len(val_loader)
        val_loss_5 /= len(val_loader)
        val_loss_20 /= len(val_loader)
        # val_losses.append(avg_val_loss)
        
        val_losses_1.append(val_loss_1)
        val_losses_5.append(val_loss_5)
        val_losses_20.append(val_loss_20)

        print(
            f"Epoch {epoch+1} — "
            f"Train: {avg_train_loss:.4f} — "
            f"Val@1: {val_loss_1:.4f} — "
            f"Val@5: {val_loss_5:.4f} — "
            f"Val@20: {val_loss_20:.4f} "
            f"(eps: {epsilon:.2f})"
)
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        # plt.plot(val_losses, label='Val Loss')
        plt.plot(val_losses_1,label='Val Loss (1)')
        plt.plot(val_losses_5,label='Val Loss (5)')
        plt.plot(val_losses_20,label='Val Loss (20)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.savefig(plot_save_path)
        plt.close()
        
        avg_val_loss = val_loss_5

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), cache_path)
            opt_epoch = epoch
            print(f"  --> Best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  --> No improvement for {epochs_no_improve}/{patience} epochs")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered! No imporvement since epoch {opt_epoch}.")
                model.load_state_dict(torch.load(cache_path,weights_only=True))
                break

    torch.save(model.state_dict(), model_save_path)

    with open(params_save_path, "wb") as f:
        pickle.dump(model_params, f)

    with open(gtk_save_path, "wb") as f:
        pickle.dump(gtk_params, f)

    print(f"""Best model saved to {model_save_path}
Parameters saved to {params_save_path}""")
    return best_val_loss





# variables = [
#     't2m',
#     'tp', 
#     'u10', 
#     'v10', 
#     # 'u100', 
#     # 'v100', 
#      'deg0l', 
#      'sp', 
#     'i10fg', 
#     'd2m',  
#     # 'msl',
#     'lcc', 
#     # 'mcc', 
#     'hcc', 
#     'ishf', 
#     'skt',
#     ]

# pred_dim = len(variables)

# data_folder = "DataCombined"
# context_window = 30
# rollout_steps = 15
# stride=15
# eps_start = 1.0  # Start with 100% Teacher Forcing
# eps_end = 0.5    # End with 50% Teacher Forcing (or lower)
# batch_size = 16
# n_epochs = 60
# patience = 5

# d_model=16 
# num_heads=16 
# target_node_idx=loc_name2ind("Stockholm")
# max_len=500
# num_nodes=9
# n_features=pred_dim+4   
# batch_first=True                
# n_graph_layers=1
# out_dim=2 * pred_dim
# dropout_rate = 0.2

# model_params = (d_model,
#     num_heads, 
#     target_node_idx, 
#     max_len,
#     num_nodes,
#     n_features,
#     batch_first,                
#     n_graph_layers, 
#     out_dim,
#     dropout_rate)