import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import BASE_DIR, REPORT_FIGS, DATA_DIR, EXPERIMENTS_DIR


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [B, T, d_model]
        returns: [B, T, d_model] with positional encodings added
        """
        T = x.size(1)
        x = x + self.pe[:T, :].unsqueeze(0)
        return x

class LearnedGCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_nodes):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.A = nn.Parameter(torch.randn(num_nodes, num_nodes))

    def forward(self, X): 
        B, N, F_dim = X.shape
        A_sym = F.softmax(self.A, dim=1)
        A_batch = A_sym.unsqueeze(0).expand(B, N, N)
        X = torch.bmm(A_batch, X)
        return self.linear(X)

class GraphEmbedding(nn.Module):
    def __init__(self, n_features, d_model, num_nodes, n_graph_layers,dropout_gcn):
        super().__init__()
        self.n_graph_layers = n_graph_layers
        self.GCN_in = LearnedGCN(
            n_features,
            d_model,
            num_nodes
            )
        self.dropout = nn.Dropout(p=dropout_gcn)
        self.bn = nn.BatchNorm1d(d_model)
        self.GCN_layers = nn.ModuleList([LearnedGCN(d_model,d_model,num_nodes)for _ in range(n_graph_layers-1)])

        

    def forward(self, X):
        """
        X: [B, T, N, F_dim]
        returns: [B, T, N, d_model]
        """
        B, T, N, F_dim = X.shape
        X_f = X.view(B * T, N, F_dim)
        X_f = self.dropout(F.gelu(self.GCN_in(X_f)))
        for layer in self.GCN_layers:
            X_f = self.dropout(F.gelu(layer(X_f)))
        X_f = X_f.view(B * T * N, -1)                # [B*T*N, d_model]
        X_f = self.bn(X_f)                           # normalize features
        h = X_f.view(B, T, N, -1)
        return h

class TemporalTransformer(nn.Module):
    def __init__(self, d_model, target_node_idx, max_len, num_heads,n_features, num_nodes, batch_first,n_graph_layers,dropout_attn,dropout_gcn):
        super().__init__()
        self.graph_embed = GraphEmbedding(
            n_features=n_features,
            d_model=d_model, 
            num_nodes=num_nodes,
            n_graph_layers=n_graph_layers,
            dropout_gcn=dropout_gcn
            )
        
        self.pos_enc = PositionalEncoding(
            d_model, 
            max_len=max_len,
            # add_fourier=True
            )
        
        self.transformer = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            batch_first=batch_first,
            )
        self.target_node_idx = target_node_idx

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout_attn)

    def forward(self, X):
        """
        X: [B, T, N, F_dim]
        returns: [B, T, d_model] for the target node
        """
        h = self.graph_embed(X)  # [B, T, N, d_model]
        h = h[:, :, self.target_node_idx, :]  # [B, T, d_model]
        h = self.pos_enc(h)
        h = self.ln1(h)
        out, _ = self.transformer(h, h, h)
        out = self.dropout(out)
        out = self.ln2(h+out)
        return out  # [B, T, d_model]

class HorizonDecoder(nn.Module):
    def __init__(self, d_model, out_dim=16):
        super().__init__()
        self.output_head = nn.Linear(d_model,out_dim) 
        

    def forward(self, x):
        """
        x: [B, T, d_model] – output from transformer
        returns: [B, len(horizons), out_dim]
        """
        last = x[:, -1, :] 
        return self.output_head(last)


class WeatherForecastModel(nn.Module):
    def __init__(self, model_params):
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
        super().__init__()
        
        self.encoder = TemporalTransformer(
            d_model=d_model, 
            num_heads=num_heads,
            target_node_idx=target_node_idx, 
            max_len=max_len,
            num_nodes=num_nodes,
            n_features=n_features,
            batch_first=batch_first,
            n_graph_layers=n_graph_layers,
            dropout_attn=dropout_attn,
            dropout_gcn=dropout_gcn
            )
        self.decoder = HorizonDecoder(d_model,out_dim)

    def forward(self, X):
        """
        X: [B, T, N, F_dim]
        returns: [B, H, F] (H = len(horizons), F = number of features)
        """
        z = self.encoder(X)             # [B, T, d_model]
        out = self.decoder(z)           # [B, H, F]
        return out


if __name__ == "__main__":

    d_model=128 
    num_heads=8 
    target_node_idx=0
    max_len=500
    num_nodes=9
    n_features=16
    batch_first=True
    n_graph_layers=1
    out_dim = 16
    dropout_attn = 0.2
    dropout_gcn = 0.1
    model_params = (d_model,
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
    
    model = WeatherForecastModel(model_params)
    X = torch.rand(32, 40, 9, 16)  # [B=16, T=40, N=9, F=18]
    y = model(X)
    print(y.shape)  # [16, 3, 18]
