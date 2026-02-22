import os
import torch
from torch.utils.data import Dataset, DataLoader
from src.training.model import WeatherForecastModel
from src.core.data_processing import WeatherDataset,split_dataset
import pandas as pd
from config import BASE_DIR, REPORT_FIGS, DATA_DIR, EXPERIMENTS_DIR
import matplotlib.pyplot as plt
from src.core.helper import loc_name2ind, loc_ind2name, var_ind2name, var_name2ind
import pickle

def evaluate(model_params, gtk_params):

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
    dropout_gcn)= model_params

    num_variables = len(variables)

    dataset = WeatherDataset(data_folder, locations, context_window,rollout_steps,stride,variables)
    mean = dataset.mean
    std = dataset.std
    weather_mean = mean[..., :num_variables]
    weather_std  = std[..., :num_variables] 
    _, _, test_data = split_dataset(dataset,train_frac=0.7,val_frac=0.15,test_frac=0.15)
    test_loader = DataLoader(test_data,batch_size=40, shuffle=False)
    model = WeatherForecastModel(model_params)

    model_save_path = run_folder / "model.pt"
    plot_folder_path = run_folder / "Var_plot"
    os.makedirs(plot_folder_path, exist_ok=True)

    model.load_state_dict(torch.load(model_save_path,weights_only=True))
    model.eval()

    context, targets, targets_fourier = next(iter(test_loader))
    target_node_idx = loc_name2ind("Stockholm")
    sample_idx = 0
    x_sample = context[sample_idx:sample_idx+1]
    rollout_len = targets.shape[1]

    prediction_list = []

    with torch.no_grad():
        current_context = x_sample
        for t in range(rollout_len):
            pred = model(current_context) # [1, 32]
            mu_pred = pred[:, :num_variables]
            prediction_list.append(pred)

            next_weather = targets[sample_idx, t, :, :].clone().unsqueeze(0) # [1, 9, 16]
            next_weather[:, target_node_idx, :] = mu_pred
            
            next_fourier = targets_fourier[sample_idx, t, :, :].unsqueeze(0) # [1, 9, 4]
            feature_next = torch.cat([next_weather, next_fourier], dim=-1) # [1, 9, 20]
            current_context = torch.cat([current_context[:, 1:, :, :], feature_next.unsqueeze(1)],dim=1)

    y_pred = torch.cat(prediction_list, dim=0)

    mu_pred  = y_pred[:, :num_variables]
    std_pred = y_pred[:, num_variables:]

    mu_pred_denorm = (mu_pred * weather_std.view(1, num_variables) + weather_mean.view(1, num_variables))

    std_pred_denorm = std_pred * weather_std.view(1, num_variables)

    weather_std  = weather_std.view(-1)
    weather_mean = weather_mean.view(-1)

    y_target_denorm = (
        targets * weather_std.view(1, 1, num_variables) +
        weather_mean.view(1, 1, num_variables)
    )

    rollout_range = range(rollout_len)
    # 1. Denormalize the context for Stockholm
    # context is [1, 120, 9, 20], we want [120, 16] weather vars
    context_stockholm = context[sample_idx, :, target_node_idx, :num_variables]
    context_denorm = (context_stockholm * weather_std.view(1, num_variables) + weather_mean.view(1, num_variables)).detach().cpu().numpy()

    # 2. Setup X-axis ranges
    context_range = range(-context_window, 0)
    rollout_range = range(0, rollout_len)


    # 1. Get Normalized Context for Stockholm
    # Context shape: [1, 120, 9, 20] -> slice to [120, 16] (weather only)
    context_stockholm_norm = context[sample_idx, :, target_node_idx, :num_variables].detach().cpu().numpy()

    # 2. Setup X-axis ranges
    context_range = range(-context_window, 0)
    rollout_range = range(0, rollout_len)

    # 3. Normalized Plotting Loop
    for q,var in enumerate(variables):
        # Past (Normalized)
        past_y = context_stockholm_norm[:, q]
        
        # Future Prediction (Normalized)
        mu = mu_pred[:, q].detach().cpu().numpy()
        sigma = std_pred[:, q].detach().cpu().numpy()
        
        # Future Actual (Normalized)
        # targets shape is already normalized from the Dataset
        true_y = targets[sample_idx, :rollout_len, target_node_idx, q].detach().cpu().numpy()

        plt.figure(figsize=(14, 6))
        plt.plot(context_range, past_y, color='black', label='Normalized History', linewidth=1.5)
        plt.fill_between(rollout_range, mu - sigma, mu + sigma, color='orange', alpha=0.2, label='Uncertainty (±1σ)')
        plt.plot(rollout_range, mu, color='darkorange', linewidth=2, label='Normalized Forecast')
        plt.plot(rollout_range, true_y, color='red', linestyle='--', alpha=0.8, label='Normalized Actual')

        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.ylim(-4, 4) # Standardized data usually stays within +/- 3 or 4 std devs
        plt.axhline(y=0, color='black', linewidth=0.5, alpha=0.3) # The Mean line
        
        plt.xlabel('Hours (Relative to Prediction Start)')
        plt.ylabel('Standard Deviations (z-score)')
        plt.title(f'Normalized Space: {var} in Stockholm')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(plot_folder_path,f"plot_{var}.png"))
        #plt.show()
        plt.close()
    print(f"Evaluation complete, plots found in {plot_folder_path}")



    
# BASE_PATH = r"MainFolder/Autoregressive Model Windows/Saved Models/model_16_30_2026-01-08_7361"
# locations = ["Berlin", "Copenhagen", "Gdansk", "Helsinki", "Oslo", "Stockholm", "Sundsvall", "Trondheim", "Visby"]

# data_folder = "DataCombined"

# params_save_path = os.path.join(BASE_PATH,"model_params.pkl")
# gtk_save_path = os.path.join(BASE_PATH,"gtk_params.pkl")
# with open(params_save_path, "rb") as f:
#     model_params = pickle.load(f)

# with open(gtk_save_path, "rb") as g:
#     gtk_params = pickle.load(g)

# (variables,
# context_window,
# rollout_steps,
# stride,
# eps_start,
# eps_end,
# batch_size,
# n_epochs,
# patience) = gtk_params

# num_variables = len(variables)