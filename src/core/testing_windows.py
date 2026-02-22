import torch
from config import BASE_DIR, REPORT_FIGS, DATA_DIR, EXPERIMENTS_DIR

torch.cuda.is_available()  # Should return True if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)