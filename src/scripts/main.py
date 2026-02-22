import os, random
from datetime import datetime
from src.training import train_model
from src.core import loc_name2ind,locations
from src.inference import evaluate
import csv
from config import BASE_DIR, REPORT_FIGS, DATA_DIR, EXPERIMENTS_DIR

data_folder = DATA_DIR / "raw"
csv_path = EXPERIMENTS_DIR /  "parameter_search.csv"

locations = [
        "Berlin",
        "Copenhagen",
        "Gdansk",
        "Helsinki",
        "Oslo",
        "Stockholm",
        "Sundsvall",
        "Trondheim",
        "Visby",
        ]

variables = [
        't2m',
        'tp', 
        'u10', 
        'v10', 
        # 'u100', 
        # 'v100', 
        'deg0l', 
        'sp', 
        'i10fg', 
        'd2m',  
        # 'msl',
        'lcc', 
        # 'mcc', 
        'hcc', 
        'ishf', 
        'skt',
        ]


# context_window = 40
rollout_steps = 20
stride=15
eps_start = 1.0
eps_end = 0.5
batch_size = 16
n_epochs = 60
patience = 7

pred_dim = len(variables)

# d_model = 16 
# num_heads = 8 
target_node_idx = loc_name2ind("Stockholm")
max_len = 500
num_nodes = 9
n_features = pred_dim + 4   
batch_first = True                
# n_graph_layers = 1
out_dim = 2 * pred_dim
# dropout_attn = 0.2
# dropout_gcn = 0.1

cache_path = BASE_DIR / "src" / "cache"
os.makedirs(os.path.dirname(cache_path), exist_ok=True)

if not os.path.exists(csv_path):
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "context_window",
            "d_model",
            "num_heads",
            "n_graph_layers",
            "dropout_attn",
            "dropout_gcn",
            "val_loss (5)",
            "run_folder"
        ])


for N in range(50):
        SEARCH_SPACE = {
        "learning_rate":[3e-5, 1e-4, 3e-4,7e-5],
        "weight_decay": [0.0, 1e-5, 1e-4,3e-5],
        "context_window": [30, 40, 50, 60],
        "d_model": [16, 32, 64, 128],
        "num_heads": [1, 2, 4, 8, 16],
        "n_graph_layers": [1, 2],
        "dropout_attn": [0.0, 0.1, 0.2],
        "dropout_gcn": [0.1, 0.2, 0.3]
        }
        learning_rate = random.choice(SEARCH_SPACE["learning_rate"])
        weight_decay = random.choice(SEARCH_SPACE["weight_decay"])
        context_window = random.choice(SEARCH_SPACE["context_window"])
        d_model = random.choice(SEARCH_SPACE["d_model"])
        num_heads = random.choice(SEARCH_SPACE["num_heads"])
        n_graph_layers = random.choice(SEARCH_SPACE["n_graph_layers"])
        dropout_attn = random.choice(SEARCH_SPACE["dropout_attn"])
        dropout_gcn = random.choice(SEARCH_SPACE["dropout_gcn"])

        model_params = (
        d_model,
        num_heads,
        target_node_idx,
        max_len,
        num_nodes,
        n_features,
        batch_first,
        n_graph_layers,
        out_dim,
        dropout_attn,
        dropout_gcn)

        date_str = datetime.now().strftime("%Y-%m-%d")
        random_id = f"{random.randint(0,9)}{random.randint(0,9)}{random.randint(0,9)}{random.randint(0,9)}"
        run_folder = EXPERIMENTS_DIR /  f"model_{d_model}_{context_window}_{date_str}_{random_id}"
        os.makedirs(run_folder, exist_ok=True)        
        
        gtk_params = (
        data_folder,
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
        weight_decay)


        val_loss = train_model(model_params, gtk_params)
        evaluate(model_params, gtk_params)

        with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                        context_window,
                        d_model,
                        num_heads,
                        n_graph_layers,
                        dropout_attn,
                        dropout_gcn,
                        val_loss,
                        run_folder
                ])

