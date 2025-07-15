#!/usr/bin/env python

"""
coil_force_pareto_scans.py
--------------------------

This script performs analysis and visualization for stage-two coil optimization results using force metrics. 
It processes optimization results stored in JSON files, applies engineering and quality filters, 
computes Pareto fronts for selected objectives, and generates summary plots. 
The script is intended for use in advanced coil design studies, although current functionality has been
limited to the Landreman-Paul QA and QH configurations. There is no reason why it cannot be used for other
configurations, but the parameters used for the scans defined in optimization_tools.py would need to be changed.

The assumption is that a large number of optimizations
have already been performed with "python initialization.py" (or sbatch cold_starts.sh to run batch jobs on 
a supercomputer) and the results are stored in the ./output/QA/B2Energy/ or a similar directory. This pareto
scan script was used to generate the results in the papers:

    Hurwitz, S., Landreman, M., Huslage, P. and Kaptanoglu, A., 2025. 
    Electromagnetic coil optimization for reduced Lorentz forces.
    Nuclear Fusion, 65(5), p.056044.
    https://iopscience.iop.org/article/10.1088/1741-4326/adc9bf/meta

    Kaptanoglu, A.A., Wiedman, A., Halpern, J., Hurwitz, S., Paul, E.J. and Landreman, M., 2025. 
    Reactor-scale stellarators with force and torque minimized dipole coils. 
    Nuclear Fusion, 65(4), p.046029.
    https://iopscience.iop.org/article/10.1088/1741-4326/adc318/meta

Main steps:
- Load and concatenate optimization results from a specified directory.
- Filter results based on engineering and quality constraints.
- Compute Pareto-optimal sets for force and field error objectives.
- Optionally copy Pareto-optimal result folders to a new output directory.
- Generate and save histograms and scatter plots for key metrics, colored by various parameters.

Usage:
    python coil_force_pareto_scans.py

"""
from simsopt.util.coil_optimization_helper_functions import *
import matplotlib.pyplot as plt
import numpy as np

INPUT_DIR = "./output/QA/B2Energy/"

df, df_filtered, df_pareto = get_dfs(INPUT_DIR=INPUT_DIR)

success_plt(df, df_filtered).show()

df, df_filtered, df_pareto = get_dfs(INPUT_DIR=INPUT_DIR)

# If you do not have a lot of runs, you can just plot the unfiltered results
df_filtered = df

# ---
# Main plotting loop: For each key parameter, generate a scatter plot of mean_RMS_force vs normalized_BdotN,
# colored by the parameter, for both all filtered results and the Pareto front.
# ---

y_axes = ["mean_RMS_force"]
labels = ["mean force [N/m]"]
y_lims = [(20000, 60000)]
keys = ['length_target', 'length_weight', 'max_κ_threshold', 'max_κ_weight',
        'msc_threshold', 'msc_weight', 'cc_threshold', 'cc_weight',
        'cs_threshold', 'cs_weight', 'force_threshold', 'force_weight',
        'arclength_weight', 'JF', 'Jf', 'gradient_norm',
        'max_length', 'max_max_κ', 'max_MSC',
        'max_max_force', 'min_min_force',
        'mean_RMS_force',
        'max_max_torque', 'min_min_torque',
        'mean_RMS_torque', 'max_arclength_variance',
        'BdotN', 'mean_AbsB', 'normalized_BdotN', 'coil_coil_distance',
        'coil_surface_distance']

for color in keys:
    for y_axis, label, y_lim in zip(y_axes, labels, y_lims):

        fig = plt.figure(figsize=(6.5, 6.5))
        plt.rc("font", size=17)
        markersize = 5
        n_pareto = df_pareto.shape[0]
        n_filtered = df_filtered.shape[0] - n_pareto
        color_label = color
        norm = plt.Normalize(min(df_filtered[color]), max(df_filtered[color]))
        try:
            # Handle case where y_axis column contains arrays
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
            # y_axis column is scalar
            new_array = df_filtered[y_axis]
            new_array2 = df_pareto[y_axis]
        except AttributeError:
            # y_axis column is likely a Series
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
        # Scatter plot for all filtered results
        plt.scatter(
            df_filtered["normalized_BdotN"],
            new_array,
            c=df_filtered[color],
            s=markersize,
            label=f'all optimizations, N={n_filtered}',
            norm=norm
        )
        # Scatter plot for Pareto front
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
        # plt.ylim(y_lim)
        plt.xscale("log")
        plt.colorbar(label=color_label)
        plt.clim(3.0, 6.0)
        plt.legend(loc='upper right', fontsize='11')
        plt.savefig(INPUT_DIR + f"pareto_colorbar_{color}_yaxis_{y_axis}.pdf", bbox_inches='tight')
plt.show()
