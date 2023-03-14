#!/bin/python

"""
    viewer.py : Result visualizer script.

    Plots up to 6 data series, optionally following live changes to the input.

    This file is a part of fedrl.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys

parser = argparse.ArgumentParser(description='Experiment result viewer')

parser.add_argument('--live', default=False, action='store_true',
                    help='Enable live updates')

parser.add_argument('--rate', type=int, default=2,
                    help='Live update frames per second')

parser.add_argument('--window', type=int, default=15,
                    help='Rolling average window size')

parser.add_argument('--no_window', default=False, action='store_true',
                    help='Plot data directly, without rolling averages')

parser.add_argument('--xmax', default=0, type=int, help='Truncate data series to <xmax> entries')
parser.add_argument('--x_label', default='training episodes', help='X axis label')
parser.add_argument('--y_label', default='reward', help='Y axis label')
parser.add_argument('--title', default=None, help='Figure title')
parser.add_argument('--save', default=None, help='Figure file output')
parser.add_argument('--width', default=6,
                    help='Figure width for file export (inches)')
parser.add_argument('--height', default=4,
                    help='Figure height for file export (inches)')
parser.add_argument('--dpi', default=100,
                    help='Figure DPI for file export')
parser.add_argument('sources', nargs='+', help='Data series to plot')

args = parser.parse_args()

def read_source(source):
    with open(source, 'r') as f:
        fdata = eval(f.read())

        if type(fdata[0]) == list:
            data_Y = np.array(fdata[1])
            data_X = np.array(fdata[0])
        else:
            data_Y = np.array(fdata)
            data_X = np.arange(len(data_Y))

    mean_Y = None

    if args.xmax > 0:
        for i, x in enumerate(data_X):
            if x > args.xmax:
                data_X = data_X[:i]
                data_Y = data_Y[:i]
                break

    if args.window % 2 == 0:
        print('WARNING: rolling average window should be odd')

    if not args.no_window:
        pfx_Y = [data_Y[0]] * (args.window // 2)
        sfx_Y = [data_Y[-1]] * (args.window // 2)

        cmb_Y = np.concatenate((pfx_Y, data_Y, sfx_Y))
        mean_Y = []

        for i in range(len(data_Y)):
            lb = data_Y[0]
            rb = data_Y[-1]

            start = i - args.window // 2
            total = 0
            count = 0

            for j in range(start, start + args.window):
                if j < 0:
                    total += lb
                elif j >= len(data_Y):
                    total += rb
                else:
                    total += data_Y[j]

                count += 1

            mean_Y.append(total / count)

    return data_X, data_Y, mean_Y

fig, ax = plt.subplots()

colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']

ax.set_title(args.title)
ax.set_ylabel(args.y_label)
ax.set_xlabel(args.x_label)

lines = {}
mean_lines = {}

for source in args.sources:
    data_X, data_Y, mean_Y = read_source(source)
    data_color = colors.pop(0)

    if args.no_window:
        lines[source], = ax.plot(data_X,
                data_Y, 
                data_color + '-',
                alpha=0.7,
                label=(source))
    else:
        lines[source], = ax.plot(data_X,
                data_Y, 
                data_color + '-',
                alpha=0.25)

        mean_lines[source], = ax.plot(data_X,
                mean_Y,
                data_color + '-', alpha=0.7,
                label=(source))

ax.legend()

while True:
    for source in args.sources:
        data_X, data_Y, mean_Y = read_source(source)
        
        lines[source].set_data(data_X, data_Y)

        plt.ylim((min(data_Y), max(plt.ylim()[1], max(data_Y))))
        plt.xlim((min(data_X), max(plt.xlim()[1], max(data_X))))

        if not args.no_window:
            mean_lines[source].set_data(data_X, mean_Y)

    if args.live:
        #plt.draw()
        plt.pause(1 / args.rate)

        if not plt.get_fignums():
            break
    else:
        plt.show()
        break

if args.save:
    #ax.set_size_inches(args.width, args.height)
    plt.savefig(args.save, dpi=args.dpi)
