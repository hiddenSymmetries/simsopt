#!/usr/bin/env python

"""
Example script for the force metric in a stage-two coil optimization
"""
from analysis_tools import *
from optimization_tools import *
# import imageio
import matplotlib.pyplot as plt

INPUT_DIR = "./output/QA/TVE/"
df, df_filtered, df_pareto = get_dfs(INPUT_DIR=INPUT_DIR)
success_plt(df, df_filtered).show()
df, df_filtered, df_pareto = get_dfs(INPUT_DIR=INPUT_DIR)

y_axes = ["mean_RMS_force"]  #, "mean_RMS_force", "max_max_torque", "mean_RMS_torque", "net_forces", "net_torques"]
labels = ["mean RMS force [N/m]"]  #, "mean force [N/m]", "max torque [N]", "mean torque [N]", "net force [N]", "net torque [N-m]"]
y_lims = [(12000, 35000)]  #, (5500, 10000), (5000, 30000), (2000, 6200), (4000, 13000), (5e2, 7000)]
#keys = ["coil_coil_distance"]
#keys = ["max_length"]
#keys = ["max_max_κ"]
#keys = ["force_threshold"]
#['max_max_torque']
keys = ['length_target', 'length_weight', 'max_κ_threshold', 'max_κ_weight',
       'msc_threshold', 'msc_weight', 'cc_threshold', 'cc_weight',
       'cs_threshold', 'cs_weight', 'force_threshold', 'force_weight',
       'arclength_weight', 'JF', 'Jf', 'gradient_norm', 
       'max_length', 'max_max_κ','max_MSC', 
       'max_max_force', 'min_min_force',
       'mean_RMS_force', 
       'max_max_torque', 'min_min_torque', 
       'mean_RMS_torque', 'max_arclength_variance',
       'BdotN', 'mean_AbsB', 'normalized_BdotN', 'coil_coil_distance',
       'coil_surface_distance']
for color in keys:
    print(color)
    for y_axis, label, y_lim in zip(y_axes, labels, y_lims):

        fig = plt.figure(figsize=(6.5, 6.5))
        # plt.rc("font", size=13)
        plt.rc("font", size=17)
        markersize = 5
        n_pareto = df_pareto.shape[0]
        n_filtered = df_filtered.shape[0] - n_pareto
        #color="max_max_κ"
        # color="coil_surface_distance"
        color_label=color  #"coil-surface distance [m]"
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
        # print(new_array)
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
        #plt.ylim(y_lim)
        plt.xscale("log")
        #if y_lim[0] < 1:
        #    plt.yscale("log")
        plt.colorbar(label=color_label)
        #plt.clim(0.17, 0.31)
        # plt.clim(3.5, 6.5)
        # plt.clim(6e3, 2e4)
        #plt.clim(3.5, 4.8)
        #plt.clim(0.07, 0.14)
        plt.legend(loc='upper right', fontsize='11')
        # plt.title('Pareto Front')
        plt.savefig(INPUT_DIR + f"pareto_colorbar_{color}_yaxis_{y_axis}.pdf", bbox_inches='tight')
plt.show()
