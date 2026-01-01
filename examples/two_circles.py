import sys
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.datasets import make_circles

from src.model.flow_matching import FlowMatching
from src.model.base_models.mlp import MLP

from .utils import *


def main(num_samples=1024, device='cpu'):
    num_samples = 1024
    points, _ = make_circles(n_samples=num_samples, noise=0.06, factor=0.5)
    train_data  = torch.from_numpy(points).to(torch.float)
    batch_size = 256
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    n_epochs = 1000
    lr = 0.0005
    obs_dim = (2,)
    fm = FlowMatching(MLP(2, 512, 5), obs_dim).to(device)
    optimizer = torch.optim.AdamW(fm.parameters(), lr=lr)

    total_loss = []
    for epoch in tqdm(range(n_epochs)):
        epoch_loss = 0
        for i, x1 in enumerate(train_loader):
            x1 = x1.to(device)
            t = torch.rand(x1.shape[0]).to(device)
            x0 = torch.randn_like(x1).to(device)
            loss = fm.criterion(t, x0, x1)
            epoch_loss += loss.detach().cpu().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss.append(epoch_loss/(i+1))
        if epoch %100 == 0:
            print(f"Total Loss Epoch {epoch+1}: ", total_loss[-1])
    
    samples_fm = fm.sample(num_samples)

    df = pd.DataFrame(train_data)
    df['sample'] = 'ground truth'
    df_fm = pd.DataFrame(samples_fm,)
    df_fm['sample'] = 'flow matching'
    title = "Comparison: True Data vs Flow Matching"
    fig = px.scatter(pd.concat([df, df_fm], axis=0), x=0, y=1,
                            color='sample',
                            size_max=1, opacity=1,
                            title=title)
    fig.update_traces(marker_size=5)
    fig.show()

if __name__ == '__main__':
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    device = str(sys.argv[2]) if len(sys.argv) > 2 else 'cpu'
    main(num_samples, device)
