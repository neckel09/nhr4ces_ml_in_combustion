import numpy as np
import torch
import matplotlib.pyplot as plt
import cantera as ct


def plot_Y_over_t(t, Y, species_names):

    gas = ct.Solution("h2o2.yaml")

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12,4), sharex=True)
    axes = axes.flat

    for i, sp in enumerate(species_names):
        axes[i].plot(t, Y[:, gas.species_index(sp)])
        axes[i].set_title(sp)
        if i > 3:
            axes[i].set_xlabel("t [ms]")

    axes[0].set_ylabel("Y [-]")
    axes[4].set_ylabel("Y [-]")

    plt.tight_layout()

    return fig

def plot_Y_over_PV(pv, Y, species_names):

    gas = ct.Solution("h2o2.yaml")

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12,4), sharex=True)
    axes = axes.flat

    for i, sp in enumerate(species_names):
        axes[i].plot(pv, Y[:, gas.species_index(sp)])
        axes[i].set_title(sp)
        if i > 3:
            axes[i].set_xlabel("PV [-]")

    axes[0].set_ylabel("Y [-]")
    axes[4].set_ylabel("Y [-]")

    plt.tight_layout()

    return fig

def plot_dY_over_t(t, dY, species_names):
    
    gas = ct.Solution("h2o2.yaml")

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12,4), sharex=True)
    axes = axes.flat

    for i, sp in enumerate(species_names):
        axes[i].plot(t, dY[:, gas.species_index(sp)])
        axes[i].set_title(sp)
        if i > 3:
            axes[i].set_xlabel("t [ms]")

    axes[0].set_ylabel("dY ")
    axes[4].set_ylabel("dY ")

    plt.tight_layout()
    
    return fig

def plot_dY_over_PV(pv, dY, species_names):
    
    gas = ct.Solution("h2o2.yaml")

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12,4), sharex=True)
    axes = axes.flat

    for i, sp in enumerate(species_names):
        axes[i].plot(pv, dY[:, gas.species_index(sp)])
        axes[i].set_title(sp)
        if i > 3:
            axes[i].set_xlabel("PV [-]")

    axes[0].set_ylabel("dY ")
    axes[4].set_ylabel("dY ")

    plt.tight_layout()
    
    return fig

def plot_loss(losses):
    fig, ax = plt.subplots(figsize=(9, 4))
    epochs = list(range(1, len(losses) + 1))
    ax.plot(epochs, losses, label="Train", linewidth=1.8)
    
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_parity(y_norm, y_pred, sp):
    
    gas = ct.Solution("h2o2.yaml")

    fig = plt.figure(figsize=(4,4))
    plt.scatter(y_norm[:,gas.species_index(sp)], y_pred[:, gas.species_index(sp)])
    ax = plt.gca()

    max_value = max(torch.max(torch.abs(y_norm[:,gas.species_index(sp)])), np.max(np.abs(y_pred[:, gas.species_index(sp)])))
    min_value = min(torch.min(y_norm[:,gas.species_index(sp)]), np.min(y_pred[:, gas.species_index(sp)]))
    print(min_value)
    ax.set_xlim((min_value, max_value))
    ax.set_ylim((min_value, max_value))

    ax.set_xlabel("target")
    ax.set_ylabel("prediction")

    plt.plot([min_value, max_value], [min_value, max_value], color="tab:red", linestyle="--")

    return fig

def plot_comparison(t: list, Y: list, species_names):

    assert len(t) == len(Y)

    gas = ct.Solution("h2o2.yaml")

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12,4), sharex=True)
    axes = axes.flat

    labels = ["DC", "ML"]
    for i, sp in enumerate(species_names):
        lines = []
        for j in range(len(Y)):
            l, = axes[i].plot(t[j], Y[j][:, gas.species_index(sp)], label=labels[j])
            lines.append(l)

        axes[i].set_title(sp)
        if i > 3:
            axes[i].set_xlabel("t [ms]")

    axes[0].set_ylabel("Y [-]")
    axes[4].set_ylabel("Y [-]")

    plt.legend(handles=lines)
    plt.tight_layout()

    return fig