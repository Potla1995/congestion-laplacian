import numpy
import pandas
import seaborn
from matplotlib import pyplot as plt

labels_dict = {
    'Lower Bound': 'Lower bound (Thm. 1.1 Eq. (4))',
    'Lower Bound (normalized)': 'Lower bound (Thm. 3.1 Eq. (18))',
    'Generic Upper Bound': 'Upper bound (Thm. 1.1 Eq. (5))',
    'Generic Upper Bound (normalized)': 'Upper bound (Thm. 3.1 Eq. (19))',
    'Main Thm Upper Bound': 'Upper bound (Thm. 1.1 Eq. (6))',
    'Main Thm Upper Bound (normalized)': 'Upper bound (Thm. 3.1 Eq. (20))',
    'Largest HSC Rank': 'Hier. spectral clustering',
    'Greedy Upper Bound': 'Hyper-Greedy',
    'Cotengra Upper Bound': 'Cotengra-Auto',
    'HyperOpt Upper Bound': 'Hyper-Opt'
}


def plot_csv(filename: str,
             x_column: str,
             y_columns: list[str],
             x_values: list = None,
             fig_size: tuple[float, float] = (5, 3),
             is_random_expt: bool = True,
             x_label: str = None,
             title: str = None,
             x_lim: tuple = None,
             y_lim: tuple = None,
             h_width: float = 0.3,
             has_lap: bool = True):
    # Update plt params
    plot_colors = seaborn.color_palette("Set2")
    plt.figure(1, figsize=fig_size)
    params = {"ytick.color": "black",
              "xtick.color": "black",
              "axes.labelcolor": "black",
              "axes.edgecolor": "black",
              "axes.labelpad": 10,
              "axes.labelsize": 16,
              "text.usetex": False,
              "font.family": "serif",
              # "font.serif": ["Computer Modern Roman"],
              'xtick.major.pad': 8,
              'ytick.major.pad': 8}
    plt.rcParams.update(params)
    if title is not None:
        plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label)
    if x_lim is not None:
        plt.xlim(x_lim[0], x_lim[1])
    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1])
    # Read in dataframe
    df = pandas.read_csv(filename)
    if x_values is None:
        x_values = df[x_column].unique()
    print(x_values)
    y_values = dict()
    if is_random_expt:
        for x in x_values:
            indices_x = numpy.where(df[x_column] == x)
            for y_name in y_columns:
                this_name_y_values = [df[y_name][i] for i in indices_x]
                y_mean = numpy.mean(this_name_y_values)
                y_std = numpy.std(this_name_y_values, ddof=1)/numpy.sqrt(len(this_name_y_values[0]))  # Unbiased variance estimator
                # print(f'DBG: {this_name_y_values}, {numpy.std(this_name_y_values, ddof=1)}, {len(this_name_y_values[0])}', {y_std})
                y_values[x, y_name] = (y_mean, y_std)  # Add to dict for plotting
    # Plot each line
    for i in range(len(y_columns)):
        if is_random_expt:
            plt.plot(x_values,
                     [y_values[(x, y_columns[i])][0] for x in x_values],
                     marker='.',
                     markersize=10,
                     label=labels_dict[y_columns[i]],
                     color=plot_colors[i]
                     )
            for x in x_values:
                plt.vlines(x=x,
                           ymin=y_values[(x, y_columns[i])][0] - y_values[(x, y_columns[i])][1],
                           ymax=y_values[(x, y_columns[i])][0] + y_values[(x, y_columns[i])][1],
                           color=plot_colors[i]
                           )
                plt.hlines(y=y_values[(x, y_columns[i])][0] - y_values[(x, y_columns[i])][1],
                           xmin=x - h_width,
                           xmax=x + h_width,
                           color=plot_colors[i]
                           )
                plt.hlines(y=y_values[(x, y_columns[i])][0] + y_values[(x, y_columns[i])][1],
                           xmin=x - h_width,
                           xmax=x + h_width,
                           color=plot_colors[i]
                           )
        else:
            plt.plot(x_values,
                     [df[x][y_columns[i]] for x in x_values],
                     marker='.',
                     markersize=10,
                     label=labels_dict[y_columns[i]],
                     color=plot_colors[i]
                     )

    plt.legend(loc='upper left')
    if not has_lap:
        filename += '.no_lap'
    plt.savefig(filename + '.pdf', format='pdf', bbox_inches='tight')
    plt.show()


## RANDOM REGULAR GRAPHS
## 3-regular
plot_csv('output/RRG[3].csv',
         'Number of Nodes',
         [
             'Largest HSC Rank', 'Greedy Upper Bound', 'Cotengra Upper Bound', 'HyperOpt Upper Bound',
             'Lower Bound', 'Generic Upper Bound', 'Main Thm Upper Bound'],
         x_label='$n$',
         title='cng$(G(n,3))$',
         y_lim=(-5,70)
         #fig_size=(4, 3)
         )
plot_csv('output/RRG[3].csv',
         'Number of Nodes',
         [
             'Largest HSC Rank', 'Greedy Upper Bound', 'Cotengra Upper Bound', 'HyperOpt Upper Bound'
         ],
         x_label='$n$',
         title='cng$(G(n,3))$',
         has_lap=False
         #fig_size=(4, 3)
         )
# 4-regular
plot_csv('output/RRG[4].csv',
         'Number of Nodes',
         [
             'Largest HSC Rank', 'Greedy Upper Bound', 'Cotengra Upper Bound', 'HyperOpt Upper Bound',
             'Lower Bound', 'Generic Upper Bound', 'Main Thm Upper Bound'],
         x_label='$n$',
         title='cng$(G(n,4))$',
         y_lim=(0,100)
         #fig_size=(4, 3)
         )
plot_csv('output/RRG[4].csv',
         'Number of Nodes',
         [
             'Largest HSC Rank', 'Greedy Upper Bound', 'Cotengra Upper Bound', 'HyperOpt Upper Bound'
         ],
         x_label='$n$',
         title='cng$(G(n,4))$',
         has_lap=False
         #fig_size=(4, 3)
         )


## ERDOS RENYI GRAPHS
# Comparison of methods
plot_csv('output/ERG[0.12, 0.125, 0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18, 0.185]_[15].csv',
         'Probability',
         [
          'Largest HSC Rank', 'Greedy Upper Bound', 'Cotengra Upper Bound', 'HyperOpt Upper Bound'],
         x_label='$p$',
         x_values=[0.12, 0.125, 0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18, 0.185],
         title='cng$(G(15,p))$',
         x_lim=(0.115, 0.19),
         h_width=0.001,
         y_lim=(3.5,7)
         #fig_size=(4, 3)
         )
# p=0.15
plot_csv('output/ERG[0.15]_[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24].csv',
         'Number of Nodes',
         [
          'Largest HSC Rank', 'Greedy Upper Bound', 'Cotengra Upper Bound', 'HyperOpt Upper Bound',
         'Lower Bound (normalized)', 'Main Thm Upper Bound (normalized)'
         ],
         x_label='$n$',
         title='cng$(G_{n, 0.15})$',
         y_lim=(-5,40)
         #fig_size=(4, 3)
         )
plot_csv('output/ERG[0.15]_[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24].csv',
         'Number of Nodes',
         [
          'Largest HSC Rank', 'Greedy Upper Bound', 'Cotengra Upper Bound', 'HyperOpt Upper Bound'
         ],
         x_label='$n$',
         title='cng$(G_{n, 0.15})$',
         has_lap=False
         #fig_size=(4, 3)
         )
# p=0.2
plot_csv('output/ERG[0.2]_[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24].csv',
         'Number of Nodes',
         [
          'Largest HSC Rank', 'Greedy Upper Bound', 'Cotengra Upper Bound', 'HyperOpt Upper Bound',
         'Lower Bound (normalized)', 'Main Thm Upper Bound (normalized)'
         ],
         x_label='$n$',
         title='cng$(G_{n, 0.2})$',
         y_lim=(-5,50)
         #fig_size=(4, 3)
         )
plot_csv('output/ERG[0.2]_[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24].csv',
         'Number of Nodes',
         [
          'Largest HSC Rank', 'Greedy Upper Bound', 'Cotengra Upper Bound', 'HyperOpt Upper Bound'
         ],
         x_label='$n$',
         title='cng$(G_{n, 0.2})$',
         has_lap=False
         #fig_size=(4, 3)
         )

# Random circuit of fixed qubit count and increasing depth
# 2-qubit gates
plot_csv('output/RC[5]-[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]-dict_values([0, 1, 0, 0]).csv',
         'Depth',
         [
          'Largest HSC Rank', 'Greedy Upper Bound', 'Cotengra Upper Bound', 'HyperOpt Upper Bound',
             'Lower Bound (normalized)', 'Main Thm Upper Bound (normalized)',  'Generic Upper Bound (normalized)',
         ],
         x_label='Circuit depth $d$',
         title='Congestion of $RQC(q=5, d, k=2)$',
         y_lim=(-5,100)
         #fig_size=(4, 3)
         )
plot_csv('output/RC[5]-[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]-dict_values([0, 1, 0, 0]).csv',
         'Depth',
         [
          'Largest HSC Rank', 'Greedy Upper Bound', 'Cotengra Upper Bound', 'HyperOpt Upper Bound',
         ],
         x_label='Circuit depth $d$',
         title='Congestion of $RQC(q=5, d, k=2)$',
         y_lim=(4.5,14),
         has_lap=False,
         #fig_size=(4, 3)
         )
# 3-qubit gates
plot_csv('output/RC[5]-[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]-dict_values([0, 0, 1, 0]).csv',
         'Depth',
         [
          'Largest HSC Rank', 'Greedy Upper Bound', 'Cotengra Upper Bound', 'HyperOpt Upper Bound',
          'Lower Bound (normalized)', 'Main Thm Upper Bound (normalized)', 'Generic Upper Bound (normalized)'
         ],
         x_label='Circuit depth $d$',
         title='Congestion of $RQC(q=5, d, k=3)$',
         y_lim=(-5, 70)
         #fig_size=(4, 3)
         )
plot_csv('output/RC[5]-[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]-dict_values([0, 0, 1, 0]).csv',
         'Depth',
         [
          'Largest HSC Rank', 'Greedy Upper Bound', 'Cotengra Upper Bound', 'HyperOpt Upper Bound'
         ],
         x_label='Circuit depth $d$',
         title='Congestion of $RQC(q=5, d, k=3)$',
         has_lap=False,
         y_lim=(4, 13)
         #fig_size=(4, 3)
         )

# Random circuit of fixed depth and increasing qbit count
# 2-qubit gates
plot_csv('output/RC[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]-[10]-dict_values([0, 1, 0, 0]).csv',
         'Number of Qubits',
         [
             'Largest HSC Rank', 'Greedy Upper Bound', 'Cotengra Upper Bound', 'HyperOpt Upper Bound',
             'Lower Bound (normalized)', 'Main Thm Upper Bound (normalized)', 'Generic Upper Bound (normalized)'
          ],
         x_label='Qubit count $q$',
         title='Congestion of $RQC(q, d=10, k=2)$',
         y_lim=(-10, 240)
         #fig_size=(4, 3)
         )
plot_csv('output/RC[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]-[10]-dict_values([0, 1, 0, 0]).csv',
         'Number of Qubits',
         [
             'Largest HSC Rank', 'Greedy Upper Bound', 'Cotengra Upper Bound', 'HyperOpt Upper Bound',
          ],
         x_label='Qubit count $q$',
         title='Congestion of $RQC(q, d=10, k=2)$',
         has_lap=False
         #fig_size=(4, 3)
         )
# 3-qubit gates
plot_csv('output/RC[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]-[10]-dict_values([0, 0, 1, 0]).csv',
         'Number of Qubits',
         [
             'Largest HSC Rank', 'Greedy Upper Bound', 'Cotengra Upper Bound', 'HyperOpt Upper Bound',
              'Lower Bound (normalized)', 'Main Thm Upper Bound (normalized)', 'Generic Upper Bound (normalized)'
          ],
         x_label='Qubit count $q$',
         title='Congestion of $RQC(q, d=10, k=3)$',
         y_lim=(-10,240)
         #fig_size=(4, 3)
         )
plot_csv('output/RC[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]-[10]-dict_values([0, 0, 1, 0]).csv',
         'Number of Qubits',
         [
             'Largest HSC Rank', 'Greedy Upper Bound', 'Cotengra Upper Bound', 'HyperOpt Upper Bound'
          ],
         x_label='Qubit count $q$',
         title='Congestion of $RQC(q, d=10, k=3)$',
         has_lap=False,
         y_lim=(5, 42)
         #fig_size=(4, 3)
         )
