import pandas as pd
import plotly.express as px
import re
import numpy as np
import torch
import wandb


def log_dif(model_cur_sd, model_prev_sd):
    # Initialize a new run

    # Create lists to store data for the plot
    layer_names = []
    std_devs = []
    l1_norms = []
    param_counts = []
    colors = []
    markers = []

    # Iterate over the parameters and compute necessary metrics
    for name, param in model_cur_sd.items():
        if name in model_prev_sd:
            prev_param = model_prev_sd[name]
            std_dev = param.std().item()
            l1_norm = torch.abs(param - prev_param).mean().item()
            param_count = param.numel()

            # Determine color based on the criteria using regex
            layer_match = re.match(r"layers\.(\d+)(?:\..*)?$", name)
            if layer_match:
                layer_num = int(layer_match.group(1))
                colors.append(layer_num)
            else:
                colors.append(-1)

            # Determine marker type
            if param.ndim == 1:
                markers.append("x")
            else:
                markers.append("circle")

            layer_names.append(name)
            std_devs.append(std_dev)
            l1_norms.append(np.log1p(l1_norm))  # log(1 + x) transformation
            param_counts.append(np.log(param_count))

    # Create a DataFrame for the plot
    df = pd.DataFrame(
        {
            "Layer Name": layer_names,
            "Standard Deviation": std_devs,
            "L1 Norm of Changes (log scale)": l1_norms,
            "Parameter Count (log)": param_counts,
            "Color": colors,
            "Marker": markers,
        }
    )

    # Determine the number of layers
    max_layer_num = df[df["Color"] != -1]["Color"].max()

    # Create a color scale for the layers (yellow to red)
    color_scale = px.colors.sequential.YlOrRd
    color_discrete_map = {
        i: color_scale[int(i * (len(color_scale) - 1) / max_layer_num)]
        for i in range(int(max_layer_num) + 1)
    }
    color_discrete_map[-1] = "blue"  # Blue for non-layer parameters

    # Create Plotly figure
    fig = px.scatter(
        df,
        x="Standard Deviation",
        y="L1 Norm of Changes (log scale)",
        size="Parameter Count (log)",
        color="Color",
        hover_name="Layer Name",
        title="Model Weight Distribution and Changes",
        symbol="Marker",
        color_discrete_map=color_discrete_map,
        opacity=0.7,
    )

    #

    table = wandb.Table(columns=["plotly_figure"])

    # Create path for Plotly figure
    path_to_plotly_html = "./plotly_figure.html"

    # Write Plotly figure to HTML
    fig.write_html(path_to_plotly_html, auto_play=False)

    # Add Plotly figure as HTML file into Table
    table.add_data(wandb.Html(path_to_plotly_html))

    # Log Table
    wandb.log({"weight_distribution_changes": table})


# Test script
import torch
import torch.nn as nn
import torch.optim as optim


# Define a simple neural network with many layers
class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )

    def forward(self, x):
        return self.layers(x)


# Initialize the network and optimizer
model = TestNet()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Dummy input and target
input = torch.randn(5, 10)
target = torch.randn(5, 10)

# Forward pass
output = model(input)
loss = criterion(output, target)

# Backward pass and optimization
loss.backward()
optimizer.step()

# Save current and previous state dicts
model_prev_sd = model.state_dict()
model_prev_sd = {k: v.clone() for k, v in model_prev_sd.items()}
optimizer.zero_grad()
output = model(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
model_cur_sd = model.state_dict()

# Log differences
wandb.init(project="test")
log_dif(model_cur_sd, model_prev_sd)
