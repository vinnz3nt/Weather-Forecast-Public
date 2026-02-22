import torch
import torch.nn as nn
from config import BASE_DIR, REPORT_FIGS, DATA_DIR, EXPERIMENTS_DIR

def get_vars():
    variables = [
        '2m_temperature',
        'total_precipitation',
        '10m_u_component_of_wind', 
        '10m_v_component_of_wind',
        '100m_u_component_of_wind', 
        '100m_v_component_of_wind',
        'zero_degree_level',
        'surface_pressure',
        'instantaneous_10m_wind_gust',
        '2m_dewpoint_temperature',
        'mean_sea_level_pressure',
        'low_cloud_cover',
        'medium_cloud_cover',
        'high_cloud_cover',
        'instantaneous_surface_sensible_heat_flux',
        'skin_temperature',
    ]
    return variables

def get_short_vars():
    short_vars = [
        't2m','tp','10u', '10v','100u','100v','zdl','sp',
        'i10g','dt2m','mslp','lcc','mcc','hcc','isf','st',]
    return short_vars

def get_locs():
    locations = [
        "Berlin",
        "Copenhagen",
        "Gdansk",
        "Helsinki",
        "Oslo",
        "Stockholm",
        "Sundsvall",
        "Trondheim",
        "Visby"
        ]
    return locations

variables = get_vars()
locations = get_locs()

def loc_name2ind(name: str):
    return locations.index(name)

def loc_ind2name(ind:int):
    return locations[ind]

def var_name2ind(name:int):
    return variables.index(name)

def var_ind2name(ind:int):
    return variables[ind]

def masked_mse_loss(pred, target):
    mask = ~torch.isnan(target)
    loss = (pred - target) ** 2
    loss = loss[mask]
    return loss.mean()


class QuantileLoss(nn.Module):
    def __init__(self, quantiles,out_dim):
        super().__init__()
        
        self.quantiles = quantiles
        self.out_dim = int(out_dim/len(quantiles))
    def forward(self, preds, target):
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:,i*self.out_dim:(i+1)*self.out_dim]
            loss = torch.max((q - 1) * errors, q * errors)
            losses.append(loss.unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


if __name__ == "__main__":
    quantiles = [0.1, 0.5, 0.9]
    criterion = QuantileLoss(quantiles)