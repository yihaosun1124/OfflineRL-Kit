import csv
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import argparse


COLORS = (
    [   
        '#318DE9', # blue
        '#FF7D00', # orange
        '#E52B50', # red
        '#7B68EE', # purple
        '#00CD66', # green
        '#FFD700', # yellow
    ]
)


def merge_csv(root_dir, query_file, query_x, query_y):
    """Merge result in csv_files into a single csv file."""
    csv_files = []
    for dirname, _, files in os.walk(root_dir):
        for f in files:
            if f == query_file:
                csv_files.append(os.path.join(dirname, f))
    results = {}
    for csv_file in csv_files:
        content = [[query_x, query_y]]
        df = pd.read_csv(csv_file)
        values = df[[query_x, query_y]].values
        for line in values:
            if np.isnan(line[1]): continue
            content.append(line)
        results[csv_file] = content
    assert len(results) > 0
    sorted_keys = sorted(results.keys())
    sorted_values = [results[k][1:] for k in sorted_keys]
    content = [
        [query_x, query_y+'_mean', query_y+'_std']
    ]
    for rows in zip(*sorted_values):
        array = np.array(rows)
        assert len(set(array[:, 0])) == 1, (set(array[:, 0]), array[:, 0])
        line = [rows[0][0], round(array[:, 1].mean(), 4), round(array[:, 1].std(), 4)]
        content.append(line)
    output_path = os.path.join(root_dir, query_y.replace('/', '_')+".csv")
    print(f"Output merged csv file to {output_path} with {len(content[1:])} lines.")
    csv.writer(open(output_path, "w")).writerows(content)
    return output_path


def csv2numpy(file_path):
    df = pd.read_csv(file_path)
    step = df.iloc[:,0].to_numpy()
    mean = df.iloc[:,1].to_numpy()
    std = df.iloc[:,2].to_numpy()
    return step, mean, std


def smooth(y, radius=0):
    convkernel = np.ones(2 * radius + 1)
    out = np.convolve(y, convkernel, mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
    return out


def plot_figure(
    results,
    x_label,
    y_label,
    title=None,
    smooth_radius=10,
    figsize=None,
    dpi=None,
    color_list=None
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if color_list == None:
        color_list = [COLORS[i] for i in range(len(results))]
    else:
        assert len(color_list) == len(results)
    for i, (algo_name, csv_file) in enumerate(results.items()):
        x, y, shaded = csv2numpy(csv_file)
        y = smooth(y, smooth_radius)
        shaded = smooth(shaded, smooth_radius)
        ax.plot(x, y, color=color_list[i], label=algo_name)
        ax.fill_between(x, y-shaded, y+shaded, color=color_list[i], alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plotter")
    parser.add_argument("--root-dir", default="log")
    parser.add_argument("--task", default="walker2d-medium-v2")
    parser.add_argument("--algos", type=str, nargs='*', default=["mopo&penalty_coef=0.5&rollout_length=5"])
    parser.add_argument("--query-file", default="policy_training_progress.csv")
    parser.add_argument("--query-x", default="timestep")
    parser.add_argument("--query-y", default="eval/normalized_episode_reward")
    parser.add_argument("--title", default=None)
    parser.add_argument("--xlabel", default="timestep")
    parser.add_argument("--ylabel", default="normalized_episode_reward")
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument("--colors", type=str, nargs='*', default=None)
    parser.add_argument("--show", action='store_true')
    parser.add_argument("--output-path", default="./figure.png")
    parser.add_argument("--figsize", type=float, nargs=2, default=(5, 5))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    results = {}
    for algo in args.algos:
        path = os.path.join(args.root_dir, args.task, algo)
        csv_file = merge_csv(path, args.query_file, args.query_x, args.query_y)
        results[algo] = csv_file

    plt.style.use('seaborn')
    plot_figure(
        results=results,
        x_label=args.xlabel,
        y_label=args.ylabel,
        title=args.title,
        smooth_radius=args.smooth,
        figsize=args.figsize,
        dpi=args.dpi,
        color_list=args.colors
    )
    if args.output_path:
        plt.savefig(args.output_path, bbox_inches='tight')
    if args.show:
        plt.show()