import torch

BATCH_SIZE = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR = 1e-5
EPOCHS = 20
SAVE_PATH = "../models/"
LOG_PATH = "../logs/"