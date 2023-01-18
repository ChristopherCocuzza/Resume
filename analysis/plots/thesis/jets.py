#!/usr/bin/env python
import os, sys
import subprocess
import numpy as np
import pandas as pd
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

## matplotlib
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
# matplotlib.rcParams['text.latex.preview'] = True
matplotlib.rc('text', usetex = True)
from matplotlib import pyplot
from matplotlib import gridspec

from analysis.corelib import core

from tools.tools import load, save, checkdir, lprint
from tools.config import conf, load_config

from fitlib.resman import RESMAN

import kmeanconf as kc

cwd = 'plots/thesis'


def find_eta_bins(table):
    ## find the indices of eta bins
    eta_headers = [_ for _ in table if 'eta' in _]
    absolute_eta = False
    for _ in eta_headers:
        if 'abs' in _:
            absolute_eta = True
            break

    eta_bin_indices = []
    eta_bin_indices.append([0])
    if absolute_eta:
        eta_key = 'eta-abs-min'
    else:
        eta_key = 'eta-min'
    for i in range(len(table[eta_key]) - 1):
        if table[eta_key][i] != table[eta_key][i + 1]:
            eta_bin_indices[-1].append(i + 1)
            eta_bin_indices.append([i + 1])
    eta_bin_indices[-1].append(len(table[eta_key]))

    return eta_headers, absolute_eta, eta_bin_indices

def get_eta_label(data, absolute_eta, eta_bins, i_bin, plot_factor_base, combined = True):
    if plot_factor_base == 1.0:
        if absolute_eta:
            eta_min = str(data['eta-abs-min'][eta_bins[i_bin][0]])
            eta_max = str(data['eta-abs-max'][eta_bins[i_bin][0]])
            if combined:
                eta_label = r'|\eta_{\rm jet}| \in [%s, %s]' % (eta_min, eta_max)
            else:
                eta_label = (r'|\eta_{\rm jet}|~\textrm{range}', r'[%s, %s]' % (eta_min, eta_max))
        else:
            eta_min = str(data['eta-min'][eta_bins[i_bin][0]])
            eta_max = str(data['eta-max'][eta_bins[i_bin][0]])
            if combined:
                eta_label = r'\eta_{\rm jet} \in [%s, %s]' % (eta_min, eta_max)
            else:
                eta_label = (r'\eta_{\rm jet}~\textrm{range}', r'[%s, %s]' % (eta_min, eta_max))
    else:
        if absolute_eta:
            eta_min = str(data['eta-abs-min'][eta_bins[i_bin][0]])
            eta_max = str(data['eta-abs-max'][eta_bins[i_bin][0]])
            plot_factor_power = round(np.log(data['plot-factor'][eta_bins[i_bin][0]]) / np.log(plot_factor_base))
            if combined:
                eta_label = r'|\eta_{\rm jet}| \in [%s, %s] \, (\times %i^{%i})' % \
                            (eta_min, eta_max, plot_factor_base, plot_factor_power)
            else:
                eta_label = (r'|\eta_{\rm jet}|~\textrm{range}', r'[%s, %s]' % (eta_min, eta_max), \
                             r'\times %i^{%i}' % (plot_factor_base, plot_factor_power))
        else:
            eta_min = str(data['eta-min'][eta_bins[i_bin][0]])
            eta_max = str(data['eta-max'][eta_bins[i_bin][0]])
            plot_factor_power = round(np.log(data['plot-factor'][eta_bins[i_bin][0]]) / np.log(plot_factor_base))
            if combined:
                eta_label = r'\eta_{\rm jet} \in [%s, %s] \, (\times %i^{%i})' % \
                            (eta_min, eta_max, plot_factor_base, plot_factor_power)
            else:
                eta_label = (r'\eta_{\rm jet}~\textrm{range}', r'[%s, %s]' % (eta_min, eta_max), \
                             r'\times %i^{%i}' % (plot_factor_base, plot_factor_power))
    return eta_label

def get_extra_label(dataset):
    ## return the additional information like run number or trigger condition
    if dataset == 10001:
        text = ''
    elif dataset == 10002:
        text = ''
    elif dataset == 10003:
        text = '2003'
    elif dataset == 10004:
        text = '2004'
    return text

def find_plot_factor_base(plot_factors):
    ## find the base of the plot factors
    if all(_ == 1.0 for _ in plot_factors):
        return False, 1.0
    for _ in plot_factors:
        if _ != 1.0:
            plot_factor = _
            break
    bases = [2, 3, 5, 6, 7, 10]
    power_differences = {}
    for base in bases:
        plot_factor_power = np.log(plot_factor) / np.log(base)
        power_differences[base] = abs(round(plot_factor_power) - plot_factor_power)
    difference_min = np.inf
    for base in bases:
        if power_differences[base] < difference_min:
            difference_min = power_differences[base]
            true_base = base
    return True, true_base

def plot_replicas(ax, pt, data, eta_bins, i_bin, factor, z_order = 1):
    cluster = list(data)[0]
    theory_replicas = data[cluster]
    ## plot replicas
    thy = [theory_replicas[j][eta_bins[i_bin][0] : eta_bins[i_bin][1]] for j in range(len(theory_replicas))]
    ys = np.mean(thy, axis = 0)
    ys_d = np.std(thy, axis = 0)
    ax.fill_between(pt, (ys - ys_d) * factor, (ys + ys_d) * factor, color = 'gold', alpha = 0.8, zorder = z_order)

    return ax

def plot_data(wdir, data):
    datasets = sorted(list(data))
    if datasets != [10001, 10002, 10003, 10004]:
        print('this combination of datasets is not implemented in function %s' % sys._getframe().f_code.co_name)
        return

    n_rows, n_columns = 2, 2
    figure = pyplot.figure(figsize = (n_columns * 7.5, n_rows * 5.0), constrained_layout = True)
    gs = figure.add_gridspec(4, 2)
    axs = {}
    axs[0] = figure.add_subplot(gs[0 : 2, 0])
    axs[1] = figure.add_subplot(gs[0 : 2, 1])
    axs[2] = figure.add_subplot(gs[2 : 4, 0])
    axs[3] = figure.add_subplot(gs[2, 1])
    axs[4] = figure.add_subplot(gs[3, 1])

    i_last_row = range(len(axs) - 1, len(axs) - 1 - n_columns, -1)

    eta_size = 20
    collaboration_size = 30
    styles = ['-']
    colors = ['b', 'g', 'm', 'darkorange', 'c', 'r']
    rhic_colors = {10003: 'b', 10004: 'g'}

    handle_left, label_left = [], []
    handle_right, label_right = [], []

    ## plot data
    for i in range(len(datasets)):
        dataset = datasets[i]
        if dataset == 10004:
            i -= 1
        ## get eta bin indices and setup figure
        _, absolute_eta, eta_bins = find_eta_bins(data[dataset])

        if 'plot-factor' not in data[dataset]:
            data[dataset]['plot-factor'] = np.ones(len(data[dataset]['value']))
        plot_with_factor, plot_factor_base = find_plot_factor_base(data[dataset]['plot-factor'])

        for j in range(len(eta_bins)):
            eta_range = get_eta_label(data[dataset], absolute_eta, eta_bins, j, plot_factor_base)

            ## plot errorbars with mean theory values and experimental values
            pt = np.array(data[dataset]['pT'][eta_bins[j][0] : eta_bins[j][1]])
            plot_factor = np.array(data[dataset]['plot-factor'][eta_bins[j][0] : eta_bins[j][1]])
            value  = np.array(data[dataset]['value'][eta_bins[j][0] : eta_bins[j][1]]) * plot_factor
            alpha  = np.array(data[dataset]['alpha'][eta_bins[j][0] : eta_bins[j][1]]) * plot_factor
            theory = np.array(data[dataset]['thy'][eta_bins[j][0] : eta_bins[j][1]]) * plot_factor
            if dataset in list(rhic_colors):
                if dataset == 10003:
                    z_order = 4
                else:
                    z_order = 3
                axs[i].errorbar(pt, value, alpha, fmt = '.', color = rhic_colors[dataset], ms = 5.0, capsize = 0.0, zorder = z_order)
            else:
                axs[i].errorbar(pt, value, alpha, fmt = '.', color = colors[j], ms = 5.0, capsize = 0.0, zorder = 3)
            if dataset in list(rhic_colors):
                ## make sure STAR 2003 data is plotted above those of STAR 2004
                if dataset == 10003:
                    z_order = 2.5
                    style = 'dashed'
                else:
                    z_order = 2
                    style = 'solid'
                ## label RHIC data by their years
                label = r'%s' % data[dataset]['col'][0].upper()
                label += r'~' + get_extra_label(dataset).replace(' ', r'~')
                axs[i].plot(pt, theory, color = rhic_colors[dataset], linestyle = style, zorder = z_order, label = r'\boldmath{$\mathrm{%s}$}' % label)
            elif dataset == 10002:
                ## show legends of CDF dataset separately
                _, eta_label, _ = get_eta_label(data[dataset], absolute_eta, eta_bins, j, plot_factor_base, combined = False)
                if j in [0, 1]:
                    handle, = axs[i].plot(pt, theory, color = colors[j], zorder = 2)
                    handle_left.append(handle)
                    label_left.append(r'$%s$' % eta_label)
                else:
                    handle, = axs[i].plot(pt, theory, color = colors[j], zorder = 2)
                    handle_right.append(handle)
                    label_right.append(r'$%s$' % eta_label)
            else:
                _, eta_label, _ = get_eta_label(data[dataset], absolute_eta, eta_bins, j, plot_factor_base, combined = False)
                axs[i].plot(pt, theory, color = colors[j], zorder = 2, label = r'$%s$' % eta_label)

            if dataset == 10003:
                axs[i] = plot_replicas(axs[i], pt, data[dataset]['thy-rep'], eta_bins, j, plot_factor, z_order = 2)
            else:
                axs[i] = plot_replicas(axs[i], pt, data[dataset]['thy-rep'], eta_bins, j, plot_factor)

        if dataset == 10002:
            ## show legends of CDF dataset separately
            legends_left = axs[i].legend(handle_left, label_left, fontsize = eta_size, frameon = False, \
                                         handlelength = 0.7, loc = 'upper center', bbox_to_anchor = (0.6, 1.0))
            legends_right = axs[i].legend(handle_right, label_right, fontsize = eta_size, frameon = False, \
                                        handlelength = 0.7, loc = 'upper right')
            axs[i].add_artist(legends_left)
            for line in legends_left.get_lines():
                line.set_linewidth(3.0)
            for line in legends_right.get_lines():
                line.set_linewidth(3.0)
        elif dataset == 10001:
            legends = axs[i].legend(fontsize = eta_size, frameon = False, handlelength = 0.7, \
                                    loc = 'upper right')
            for line in legends.get_lines():
                line.set_linewidth(3.0)
        else:
            legends = axs[i].legend(fontsize = collaboration_size, loc = 'best', frameon = False, handlelength = 1.5)
            for line in legends.get_lines():
                line.set_linewidth(3.0)

        axs[i].set_yscale('log')

        ## setup y axis label
        unit = data[dataset]['units'][0]
        if data[dataset]['obs'][0].replace('<', '').replace('>', '') == 'd2_sigma_over_d_y_d_pt':
            # y_label = r'\boldmath{$\frac{\mathrm{d}^2\sigma}{\mathrm{d}\eta \, \mathrm{d} p_T}~{\scriptstyle \left(\mathrm{%s} / (\mathrm{GeV} / c) \right)}$}' % unit
            y_label = r'\boldmath{$\frac{\mathrm{d}^2\sigma^{\rm jet}}{\mathrm{d}\eta_{\rm jet} \, \mathrm{d} p_T^{\rm jet}}~{\scriptstyle \left(\frac{\mathrm{%s}}{\mathrm{GeV}} \right)}$}' % unit
        elif data[dataset]['obs'][0].replace('<', '').replace('>', '') == 'd2_sigma_over_2_pi_d_y_d_pt':
            # y_label = r'\boldmath{$\frac{\mathrm{d}^2\sigma}{2 \pi \, \mathrm{d}\eta \, \mathrm{d} p_T}~{\scriptstyle \left(\mathrm{%s} / (\mathrm{GeV} / c) \right)}$}' % unit
            y_label = r'\boldmath{$\frac{\mathrm{d}^2\sigma^{\rm jet}}{2 \pi \, \mathrm{d}\eta_{\rm jet} \, \mathrm{d} p_T^{\rm jet}}~{\scriptstyle \left(\frac{\mathrm{%s}}{\mathrm{GeV}} \right)}$}' % unit
        if (i % n_columns) == 0:
            axs[i].set_ylabel(y_label, size = 40)

        if dataset == 10003:
            axs[i].text(0.05, 0.1, r'$%s$' % eta_range, color = 'k', transform = axs[i].transAxes, size = 23)
        elif dataset == 10004:
            pass
        else:
            text = r'%s' % data[dataset]['col'][0].upper()
            text += r'~' + get_extra_label(dataset).replace(' ', r'~')
            if dataset == 10001:
                x_, y_ = 0.2, 0.85
            elif dataset == 10002:
                x_, y_ = 0.8, 0.1
            axs[i].text(x_, y_, r'\boldmath{$\mathrm{%s}$}' % text, color = 'k', transform = axs[i].transAxes, size = collaboration_size)

    ## plot data over theory for RHIC data
    for i in range(len(rhic_colors)):
        dataset = list(rhic_colors)[i]
        for j in range(len(eta_bins)):
            pt = data[dataset]['pT'][eta_bins[j][0] : eta_bins[j][1]]
            plot_factor_base = 1.0
            eta_range = get_eta_label(data[dataset], absolute_eta, eta_bins, j, plot_factor_base)

            theory = data[dataset]['thy'][eta_bins[j][0] : eta_bins[j][1]]
            value = data[dataset]['value'][eta_bins[j][0] : eta_bins[j][1]]
            alpha = data[dataset]['alpha'][eta_bins[j][0] : eta_bins[j][1]]

            axs[i + 3].errorbar(pt, value / theory, alpha / theory, fmt = '.', color = rhic_colors[dataset], ms = 5.0, capsize = 3.5, zorder = 2)
            axs[i + 3].axhline(1.0, color = 'black', ls = 'dashdot', linewidth = 0.5, zorder = 0)

            axs[i + 3] = plot_replicas(axs[i + 3], pt, data[dataset]['thy-rep'], eta_bins, j, 1.0 / theory)

    for i in range(len(axs)):
        axs[i].tick_params(axis = 'both', which = 'both', right = True, top = True, direction = 'in', labelsize = 21)
        axs[i].tick_params(axis = 'x', pad = 10)
        ## set tick length
        axs[i].yaxis.set_tick_params(which = 'major', length = 7.5)
        axs[i].xaxis.set_tick_params(which = 'major', length = 7.5)
        axs[i].minorticks_off()
    for i in [2, 4]:
        axs[i].set_xlabel(r'\boldmath$p_T^{\rm jet}~(\mathrm{GeV})$', size = 30)
    axs[3].set_ylim(0.6, 1.2)
    axs[4].set_ylim(0.8, 1.2)
    axs[3].text(0.05, 0.20, r'\boldmath{$\mathrm{data/theory}$}', color = 'k', transform = axs[3].transAxes, size = 23)
    axs[4].text(0.30, 0.14, r'\boldmath{$\mathrm{data/theory}$}', color = 'k', transform = axs[4].transAxes, size = 23)
    axs[3].text(0.52, 0.25, r'\boldmath{$\mathrm{STAR~2003}$}', color = 'k', transform = axs[3].transAxes, size = collaboration_size)
    axs[4].text(0.25, 0.74, r'\boldmath{$\mathrm{STAR~2004}$}', color = 'k', transform = axs[4].transAxes, size = collaboration_size)

    ## add factor labels for D0
    dataset = 10001
    _, absolute_eta, eta_bins = find_eta_bins(data[dataset])
    plot_with_factor, plot_factor_base = find_plot_factor_base(data[dataset]['plot-factor'])
    eta_name, _, factor = get_eta_label(data[dataset], absolute_eta, eta_bins, 5, plot_factor_base, combined = False)
    axs[0].text(0.70, 0.875, r'$%s$' % eta_name, color = 'k', transform = axs[0].transAxes, \
                size = 23, horizontalalignment = 'right', verticalalignment = 'bottom')
    axs[0].text(0.29, 0.03, r'$%s$' % factor, color = colors[5], transform = axs[0].transAxes, \
                size = 23, horizontalalignment = 'right', verticalalignment = 'bottom')
    _, _, factor = get_eta_label(data[dataset], absolute_eta, eta_bins, 4, plot_factor_base, combined = False)
    axs[0].text(0.40, 0.015, r'$%s$' % factor, color = colors[4], transform = axs[0].transAxes, \
                size = 23, horizontalalignment = 'left', verticalalignment = 'bottom')
    _, _, factor = get_eta_label(data[dataset], absolute_eta, eta_bins, 3, plot_factor_base, combined = False)
    axs[0].text(0.54, 0.03, r'$%s$' % factor, color = colors[3], transform = axs[0].transAxes, \
                size = 23, horizontalalignment = 'left', verticalalignment = 'bottom')
    _, _, factor = get_eta_label(data[dataset], absolute_eta, eta_bins, 2, plot_factor_base, combined = False)
    axs[0].text(0.70, 0.03, r'$%s$' % factor, color = colors[2], transform = axs[0].transAxes, \
                size = 23, horizontalalignment = 'left', verticalalignment = 'bottom')
    _, _, factor = get_eta_label(data[dataset], absolute_eta, eta_bins, 1, plot_factor_base, combined = False)
    axs[0].text(0.86, 0.01, r'$%s$' % factor, color = colors[1], transform = axs[0].transAxes, \
                size = 23, horizontalalignment = 'left', verticalalignment = 'bottom')
    _, _, factor = get_eta_label(data[dataset], absolute_eta, eta_bins, 0, plot_factor_base, combined = False)
    axs[0].text(0.99, 0.18, r'$%s$' % factor, color = colors[0], transform = axs[0].transAxes, \
                size = 23, horizontalalignment = 'right', verticalalignment = 'bottom')

    ## add factor labels for CDF
    dataset = 10002
    _, absolute_eta, eta_bins = find_eta_bins(data[dataset])
    plot_with_factor, plot_factor_base = find_plot_factor_base(data[dataset]['plot-factor'])
    eta_name, _, factor = get_eta_label(data[dataset], absolute_eta, eta_bins, 4, plot_factor_base, combined = False)
    axs[1].text(0.45, 0.875, r'$%s$' % eta_name, color = 'k', transform = axs[1].transAxes, \
                size = 23, horizontalalignment = 'right', verticalalignment = 'bottom')
    axs[1].text(0.32, 0.03, r'$%s$' % factor, color = colors[4], transform = axs[1].transAxes, \
                size = 23, horizontalalignment = 'right', verticalalignment = 'bottom')
    _, _, factor = get_eta_label(data[dataset], absolute_eta, eta_bins, 3, plot_factor_base, combined = False)
    axs[1].text(0.41, 0.12, r'$%s$' % factor, color = colors[3], transform = axs[1].transAxes, \
                size = 23, horizontalalignment = 'left', verticalalignment = 'bottom')
    _, _, factor = get_eta_label(data[dataset], absolute_eta, eta_bins, 2, plot_factor_base, combined = False)
    axs[1].text(0.62, 0.23, r'$%s$' % factor, color = colors[2], transform = axs[1].transAxes, \
                size = 23, horizontalalignment = 'left', verticalalignment = 'bottom')
    _, _, factor = get_eta_label(data[dataset], absolute_eta, eta_bins, 1, plot_factor_base, combined = False)
    axs[1].text(0.80, 0.34, r'$%s$' % factor, color = colors[1], transform = axs[1].transAxes, \
                size = 23, horizontalalignment = 'left', verticalalignment = 'bottom')
    _, _, factor = get_eta_label(data[dataset], absolute_eta, eta_bins, 0, plot_factor_base, combined = False)
    axs[1].text(0.80, 0.50, r'$%s$' % factor, color = colors[0], transform = axs[1].transAxes, \
                size = 23, horizontalalignment = 'left', verticalalignment = 'bottom')

    pyplot.tight_layout()
    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/jets'%cwd
    filename += '.png'
    #filename += '.pdf'
    axs[0].set_rasterized(True)
    axs[1].set_rasterized(True)
    axs[2].set_rasterized(True)
    axs[3].set_rasterized(True)
    axs[4].set_rasterized(True)
    pyplot.savefig(filename)
    print('Saving figure to %s'%filename)

if __name__ == "__main__":

    wdir = 'results/star/final'

    print('Plotting jet data...')    
    load_config('%s/input.py' % wdir)
    istep = core.get_istep()
    replicas = core.get_replicas(wdir)
    core.mod_conf(istep, replicas[0]) ## set conf as specified in istep
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))

    data = predictions['reactions']['jet']
    del predictions

    for dataset in data:
        predictions = copy.copy(data[dataset]['prediction-rep'])
        del data[dataset]['prediction-rep']
        del data[dataset]['residuals-rep']
        del data[dataset]['shift-rep']
        data[dataset]['thy'] = np.mean(predictions, axis = 0)
        data[dataset]['dthy'] = np.std(predictions, axis = 0)
        data[dataset]['thy-rep'] = {'all': predictions}

    plot_data(wdir, data)






