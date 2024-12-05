#!/usr/bin/env python

"""
Example script for the force metric in a stage-two coil optimization
"""
from analysis_tools import *
from optimization_tools import *
# import imageio
import matplotlib
import matplotlib.pyplot as plt
import os

# initial_optimizations(N=2000, with_force=True, MAXITER=2000, FORCE_OBJ=LpCurveForce, OUTPUT_DIR="./output/QA/LPF/optimizations/")

df, df_filtered, df_pareto = get_dfs(INPUT_DIR="./output/QA/LPT/optimizations/")
success_plt(df, df_filtered).show()
df, df_filtered, df_pareto = get_dfs(INPUT_DIR="./output/QA/LPT/optimizations/")

y_axes = ["max_max_force", "mean_RMS_force", "max_max_torque", "mean_RMS_torque", "net_forces", "net_torques"]
labels = ["max force [N/m]", "mean force [N/m]", "max torque [N]", "mean torque [N]", "net force [N]", "net torque [N-m]"]
y_lims = [(8500, 25000), (6000, 10000), (1000, 30000), (100, 10000), (2000, 15000), (1e-3, 1e4)]
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
    # print(np.array(df_filtered[y_axis][0]), np.array(df_filtered[y_axis][0]).shape, len(np.array([df_filtered[y_axis][0]])))
    try: 
        # print(df_filtered[y_axis], np.array(df_filtered[y_axis])[0].shape)
        np.array(df_filtered[y_axis])[0].shape[0]
        N = np.array(df_filtered[y_axis]).shape[0]
        N2 = np.array(df_pareto[y_axis]).shape[0]
        new_array = np.zeros(N)
        new_array2 = np.zeros(N2)
        q = 0
        for i in df_filtered[y_axis].keys():
            new_array[q] = np.mean(df_filtered[y_axis][i])
            q = q + 1
        q = 0
        for i in df_pareto[y_axis].keys():
            new_array[q] = np.mean(df_pareto[y_axis][i])
            q = q + 1
    except IndexError:
        new_array = df_filtered[y_axis] 
        new_array2 = df_pareto[y_axis]
    except AttributeError:
        N = np.array(df_filtered[y_axis]).shape[0]
        N2 = np.array(df_pareto[y_axis]).shape[0]
        new_array = np.zeros(N)
        new_array2 = np.zeros(N2)
        q = 0
        for i in df_filtered[y_axis].keys():
            new_array[q] = np.mean(df_filtered[y_axis][i])
            q = q + 1
        q = 0
        for i in df_pareto[y_axis].keys():
            new_array[q] = np.mean(df_pareto[y_axis][i])
            q = q + 1
    # except KeyError:
    #     new_array = df_filtered[y_axis] 
    #     new_array2 = df_pareto[y_axis]
    print(new_array)
    plt.scatter(
        df_filtered["normalized_BdotN"],
        new_array,
        c=df_filtered[color],
        s=markersize,
        label=f'all optimizations, N={n_filtered}',
        norm=norm
    )
    plt.scatter(
        df_pareto["normalized_BdotN"], 
        new_array2, 
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
    if y_lim[0] < 1:
        plt.yscale("log")
    plt.colorbar(label=color_label)
    plt.clim(0.17, 0.31)
    plt.legend(loc='upper right', fontsize='11')
    # plt.title('Pareto Front')
    # plt.savefig(f"./output/QA/with-force-penalty/4/pareto_{y_axis}.pdf", bbox_inches='tight')
plt.show()