from utils import load_yaml
from model.train import train
import torch
from utils import get_logger
import logging
import time
from data.data_loder import calculate_parements_stock

if __name__ == "__main__":
    cfg = load_yaml('config.yaml')

    get_logger(f"pinn", log_file=f"logs/loss_{time.strftime('%d')}.log")
    logger = logging.getLogger("pinn")
    
    if cfg.needing_calculate:
        from data.data_loder import calculate_parements_stock
        paraments = calculate_parements_stock(cfg.csv_path, cfg.result_path, save_csv=True)
    else :
        paraments = calculate_parements_stock(cfg.csv_path)

    if cfg.train:
        train(paraments)
    else:
        torch.load("./models/pinn_model.pth")

    

    