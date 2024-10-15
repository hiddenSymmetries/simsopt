#!/usr/bin/env python

"""
Example script for the force metric in a stage-two coil optimization
"""
from analysis_tools import *
from optimization_tools import *
import imageio
import matplotlib
import matplotlib.pyplot as plt
import os

# plt.ioff()
initial_optimizations(N=10, with_force=False, MAXITER=400)
df, df_filtered, df_pareto = get_dfs()
success_plt(df, df_filtered).show()

df, df_filtered, df_pareto = get_dfs()

y_axes = ["max_max_force", "mean_RMS_force"]
labels = ["max force [N/m]", "mean force [N/m]"]
y_lims = [(8500, 25000), (6000, 10000)]
for y_axis, label, y_lim in zip(y_axes, labels, y_lims):

    fig = plt.figure(figsize=(6.5, 6.5))
    # plt.rc("font", size=13)
    plt.rc("font", size=17)
    markersize = 5
    n_pareto = df_pareto.shape[0]
    n_filtered = df_filtered.shape[0] - n_pareto
    color="coil_surface_distance"
    color_label="coil-surface distance [m]"
    norm = plt.Normalize(min(df_filtered[color]), max(df_filtered[color]))
    plt.scatter(
        df_filtered["normalized_BdotN"],
        df_filtered[y_axis],
        c=df_filtered[color],
        s=markersize,
        label=f'all optimizations, N={n_filtered}',
        norm=norm
    )
    plt.scatter(
        df_pareto["normalized_BdotN"], 
        df_pareto[y_axis], 
        c=df_pareto[color], 
        marker="+",
        label=f'Pareto front, N={n_pareto}',
        norm=norm,
    )
    plt.xlabel(r'$\langle|\mathbf{B}\cdot\mathbf{n}|\rangle/\langle B \rangle$ [unitless]')
    plt.ylabel(label)
    plt.xlim(0.7 * min(df_filtered["normalized_BdotN"]), max(df_filtered["normalized_BdotN"]))
    plt.ylim(y_lim)
    plt.xscale("log")
    plt.colorbar(label=color_label)
    plt.clim(0.17, 0.31)
    plt.legend(loc='upper right', fontsize='11')
    # plt.title('Pareto Front')
    plt.savefig(f"./output/QA/with-force-penalty/4/pareto_{y_axis}.pdf", bbox_inches='tight')
    plt.show()