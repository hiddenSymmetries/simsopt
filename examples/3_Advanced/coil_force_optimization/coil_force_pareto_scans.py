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

Dependencies:
    - optimization_tools (local module)
    - matplotlib
    - pandas
    - numpy
    - paretoset
    - glob, json, shutil, os

"""
from optimization_tools import *
import matplotlib.pyplot as plt
import shutil
import glob
import json
import pandas as pd
import numpy as np
from paretoset import paretoset
import os

def success_plt(df, df_filtered):
    """
    Generate and save histograms comparing distributions of key metrics before and after filtering.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing all raw optimization results.
    df_filtered : pandas.DataFrame
        DataFrame containing filtered optimization results.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the histograms.
    """
    fig = plt.figure(1, figsize=(14.5, 11))
    nrows = 5
    ncols = 5

    def plot_2d_hist(field, log=False, subplot_index=0):
        """
        Plot a histogram for a given field, comparing before and after filtering.

        Parameters
        ----------
        field : str
            The column name to plot.
        log : bool, optional
            Whether to use a logarithmic x-axis (default: False).
        subplot_index : int
            The subplot index in the figure.
        """
        plt.subplot(nrows, ncols, subplot_index)
        nbins = 20
        if log:
            data = df[field]
            bins = np.logspace(np.log10(data.min()), np.log10(data.max()), nbins)
        else:
            bins = nbins
        n, bins, patchs = plt.hist(df[field], bins=bins, label="before filtering")
        plt.hist(df_filtered[field], bins=bins, alpha=1, label="after filtering")
        plt.xlabel(field)
        plt.legend(loc=0, fontsize=6)
        plt.xlim(0, np.mean(df[field]) * 2)
        if log:
            plt.xscale("log")
        plt.savefig('hist.pdf')

    # 2nd entry of each tuple is True if the field should be plotted on a log x-scale.
    fields = (
        ("R1", False),
        ("order", False),
        ("max_max_force", False),
        ("max_length", False),
        ("max_max_κ", False),
        ("max_MSC", False),
        ("coil_coil_distance", False),
        ("coil_surface_distance", False),
        ("length_target", False),
        ("force_threshold", False),
        ("max_κ_threshold", False),
        ("msc_threshold", False),
        ("cc_threshold", False),
        ("cs_threshold", False),
        ("length_weight", True),
        ("max_κ_weight", True),
        ("msc_weight", True),
        ("cc_weight", True),
        ("cs_weight", True),
        ("force_weight", True),
        ('ncoils', False)
    )

    i = 1
    for field, log in fields:
        plot_2d_hist(field, log, i)
        i += 1

    plt.tight_layout()
    return fig

def get_dfs(INPUT_DIR='./output/QA/B2Energy/', OUTPUT_DIR=None):
    """
    Load, filter, and compute Pareto front for coil optimization results. Filtering is 
    done based on the following engineering constraints. Max denotes the maximum over all coils,
    max denotes the maximum over a single coil, mean denotes an average over the plasma surface. 
    These numbers will need to be adjusted for other stellarator configurations and rescaled if 
    the major radius is scaled from the 1 m baseline:

    - Max(coil lengths) < 5 * margin_up
    - Max(max(coil curvatures)) < 12.00 * margin_up
    - Max(mean-squared-curvature) < 6.00 * margin_up
    - Max(coil-coil distance) > 0.083 * margin_low
    - Max(coil-surface distance) > 0.166 * margin_low
    - mean(Abs(B)) > 0.22
    - Max(arclength variance) < 1e-2
    - Coil-surface distance < 0.375
    - Coil-coil distance < 0.15
    - Max(coil length) > 3.0
    - Max(normalized BdotN) < 4e-2
    - Max(max(force)) < 50000

    Parameters
    ----------
    INPUT_DIR : str, optional
        Directory containing optimization result folders with results.json files.
    OUTPUT_DIR : str or None, optional
        If provided, Pareto-optimal result folders will be copied to this directory.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing all raw optimization results.
    df_filtered : pandas.DataFrame
        DataFrame containing filtered optimization results.
    df_pareto : pandas.DataFrame
        DataFrame containing Pareto-optimal results (minimizing normalized_BdotN and max_max_force).
    """
    ### STEP 1: Import raw data
    inputs = f"{INPUT_DIR}**/results.json"
    results = glob.glob(inputs, recursive=True)
    dfs = []
    for results_file in results:
        with open(results_file, "r") as f:
            data = json.load(f)
        # Wrap lists in another list
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = [value]
        dfs.append(pd.DataFrame(data))
    df = pd.concat(dfs, ignore_index=True)

    ### STEP 2: Filter the data
    margin_up = 1.5  # the upper bound margin to consider for tolerable engineering constraints
    margin_low = 0.5  # the lower bound margin to consider for tolerable engineering constraints

    df_filtered = df.query(
        # ENGINEERING CONSTRAINTS:
        f"max_length < {5 * margin_up}"
        f"and max_max_κ < {12.00 * margin_up}"
        f"and max_MSC < {6.00 * margin_up}"
        f"and coil_coil_distance > {0.083 * margin_low}"
        f"and coil_surface_distance > {0.166 * margin_low}"
        f"and mean_AbsB > 0.22"  # prevent coils from becoming detached from LCFS
        # FILTERING OUT BAD/UNNECESSARY DATA:
        f"and max_arclength_variance < 1e-2"
        f"and coil_surface_distance < 0.375"
        f"and coil_coil_distance < 0.15"
        f"and max_length > 3.0"
        f"and normalized_BdotN < {4e-2}"
        f"and max_max_force<50000"
    )
    print(df_filtered.keys(), df_filtered[["normalized_BdotN", "max_max_force"]])
    print(paretoset(df_filtered[["normalized_BdotN", "max_max_force"]]))

    ### STEP 3: Generate Pareto front and export UUIDs as .txt
    pareto_mask = paretoset(df_filtered[["normalized_BdotN", "max_max_force"]], sense=[min, min])
    df_pareto = df_filtered[pareto_mask]

    # Copy pareto fronts to a separate folder
    if OUTPUT_DIR is not None:
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for UUID in df_pareto['UUID']:
            SOURCE_DIR = glob.glob(f"/**/{UUID}/", recursive=True)[0]
            DEST_DIR = f"{OUTPUT_DIR}{UUID}/"
            shutil.copytree(SOURCE_DIR, DEST_DIR)

    ### Return statement
    return df, df_filtered, df_pareto

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
