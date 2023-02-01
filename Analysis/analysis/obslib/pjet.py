#!/usr/bin/env python
import sys, os
import math
import numpy as np
import copy
from subprocess import Popen, PIPE, STDOUT
import pandas as pd
import scipy as sp

## matplotlib
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
from matplotlib.ticker import MultipleLocator, FormatStrFormatter ## for minor ticks in x label
matplotlib.rc('text', usetex = True)
import pylab as py
#from adjustText import adjust_text

## from fitpack tools
from tools.tools     import load, save, checkdir, lprint
from tools.config    import conf, load_config

## from fitpack
from fitlib.resman import RESMAN
from qcdlib import aux, mellin

## from fitpack obslib
from obslib.jets.jet_tools import find_eta_bins
from obslib.jets.jet_mellin import JETMELLIN
from obslib.pjets.pjet_mellin import PJETMELLIN
import obslib.pjets.reader

## from fitpack analysis
from analysis.corelib import core
from analysis.corelib import classifier
from analysis.obslib.jet import count_eta_bins

def plot_data(wdir, istep, data, style, plot_with_factor, dpi):
    if len(data) == 6:
        nrows, ncols = 3, 2
    else:
        print('number of dataset is different from 6')
        print('you need to specify number of rows and columns for subplots in "plot_data"')
        return
    figure, ax = py.subplots(nrows, ncols)

    plot_styles = ['-']
    plot_colors = ['b', 'g', 'm', 'y', 'c', 'r']
    # plot_styles = ['-', '-.', '--', ':', '-', '--']
    plot_with_factor_input = plot_with_factor

    count = 0
    for idx in data:
        row, col = count / ncols, count % ncols
        data_idx = data[idx]

        plot_with_factor = plot_with_factor_input
        if not plot_with_factor:
            data_idx['plot-factor'] = np.ones(len(data_idx['value']))
        if all(_ == 1.0 for _ in data_idx['plot-factor']):
            ## this counts for the case where there is no plot factor from the experimental group
            ## value of plot factor is all set to 1.0 in such case
            plot_with_factor = False

        if plot_with_factor:
            ## find the base of the plot factors
            plot_factor = data_idx['plot-factor'][0]
            for i in [2, 3, 5, 6, 7, 10]:
                plot_factor_power = np.log(plot_factor) / np.log(i)
                if abs(round(plot_factor_power) - plot_factor_power) < 1e-10:
                    plot_factor_base = i
                    break

        _, absolute_eta, eta_bins = find_eta_bins(data_idx)

        for i in xrange(len(eta_bins)):
            eta_range = ''
            if plot_with_factor:
                if absolute_eta:
                    eta_min, eta_max = str(data_idx['eta-abs-min'][eta_bins[i][0] - 1]), str(data_idx['eta-abs-max'][eta_bins[i][0] - 1])
                    plot_factor_power = round(np.log(data_idx['plot-factor'][eta_bins[i][0] - 1]) / np.log(plot_factor_base))
                    eta_range += r'$|\eta|~\in~[%s,~%s]~(\times %i^{%i})$' % (eta_min, eta_max, plot_factor_base, plot_factor_power)
                else:
                    eta_min, eta_max = str(data_idx['eta-min'][eta_bins[i][0] - 1]), str(data_idx['eta-max'][eta_bins[i][0] - 1])
                    plot_factor_power = round(np.log(data_idx['plot-factor'][eta_bins[i][0] - 1]) / np.log(plot_factor_base))
                    eta_range += r'$\eta~\in~[%s,~%s]~(\times %i^{%i})$' % (eta_min, eta_max, plot_factor_base, plot_factor_power)
            else:
                if absolute_eta:
                    eta_min, eta_max = str(data_idx['eta-abs-min'][eta_bins[i][0] - 1]), str(data_idx['eta-abs-max'][eta_bins[i][0] - 1])
                    eta_range += r'$|\eta|~\in~[%s,~%s]$' % (eta_min, eta_max)
                else:
                    eta_min, eta_max = str(data_idx['eta-min'][eta_bins[i][0] - 1]), str(data_idx['eta-max'][eta_bins[i][0] - 1])
                    eta_range += r'$\eta~\in~[%s,~%s]$' % (eta_min, eta_max)

            if i==0: color = 'firebrick'
            if i==1: color = 'darkgreen'

            pt = np.array(data_idx['pT'][eta_bins[i][0] - 1 : eta_bins[i][1]])
            value = np.array(data_idx['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]]) * \
                np.array(data_idx['value'][eta_bins[i][0] - 1 : eta_bins[i][1]])
            alpha = np.array(data_idx['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]]) * \
                np.array(data_idx['alpha'][eta_bins[i][0] - 1 : eta_bins[i][1]])
            ax[row, col].errorbar(pt, value, alpha, color = color, fmt = '.', zorder = 10, capsize=2.0, label=eta_range)

            theory  = np.array(data_idx['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]]) * \
                np.array(data_idx['thy'][eta_bins[i][0] - 1 : eta_bins[i][1]])
            dtheory = np.array(data_idx['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]]) * \
                np.array(data_idx['dthy'][eta_bins[i][0] - 1 : eta_bins[i][1]])
            up   = theory + dtheory
            down = theory - dtheory
            ax[row, col].plot(pt, theory, color='black', zorder = 5)
            ax[row, col].fill_between(pt, down, up, color='gold', zorder = 5)

            if (style == 'best_lines') or (style == 'all_lines') or (style == 'cluster_lines') or (style == 'cluster_bands'):
                line_width = min(0.1, 10.0 / float(len(data_idx['thy-rep'])))
                if (style == 'cluster_lines') or (style == 'cluster_bands'):
                    count_clusters = [0 for _ in labels['cluster']]
                    for j in range(len(data_idx['thy-rep'])):
                        count_clusters[labels['cluster'][j]] += 1
                        thy = np.array(data_idx['thy-rep'][j][eta_bins[i][0] - 1 : eta_bins[i][1]]) * \
                                       np.array(data_idx['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]])
                        if (i == 0) and (count_clusters[labels['cluster'][j]] == 1):
                            label = r'$\overline{\chi _{%d}^2} = %.2f$' % (labels['cluster'][j], labels['chi_2'][labels['cluster'][j]])
                            ax[row, col].plot(pt, thy, color = labels['color'][labels['cluster'][j]], \
                                              linewidth = line_width, zorder = 0, label = label)
                        else:
                            ax[row, col].plot(pt, thy, color = labels['color'][labels['cluster'][j]], \
                                                     linewidth = line_width, zorder = 0)
                else:
                    for j in range(len(data_idx['thy-rep'])):
                        thy  = np.array(data_idx['thy-rep'][j][eta_bins[i][0] - 1 : eta_bins[i][1]]) * \
                                       np.array(data_idx['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]])
                        dthy = np.array(data_idx['dthy'][j][eta_bins[i][0] - 1 : eta_bins[i][1]]) * \
                                       np.array(data_idx['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]])
                        ax[row, col].plot(pt, thy, color = 'gray', linewidth = line_width, zorder = 0)

        collab = data_idx['col'][0].upper()
        if idx == 20001:   year,loc = 2003,'lower right'
        elif idx == 20002: year,loc = 2005,'lower right'
        elif idx == 20003: year,loc = 2006,'lower left'
        elif idx == 20004: year,loc = 2009,'lower right'
        elif idx == 20005: year,loc = 2005,'lower right'
        elif idx == 20006: year,loc = 2012,'lower right'
        handles, labels = ax[row, col].get_legend_handles_labels()
        legends = ax[row, col].legend(handles, labels, fontsize = 10, loc = loc, frameon=False, handletextpad=0.5, handlelength = 1.0)
        ax[row, col].text(0.05,0.80,r'\textrm{\textbf{%s~%s}}'%(collab,year),transform=ax[row,col].transAxes)
        ax[row, col].tick_params(axis='both',which='both',top=True,right=True,direction='in')
        ax[row, col].axhline(0.0, color = 'black', ls = '--', zorder = 0, alpha=0.5)
        count += 1

    ax[0, 0].set_xlim(8,38)
    ax[1, 0].set_xlim(8,38)
    ax[2, 0].set_xlim(8,38)
    ax[0, 1].set_xlim(8,55)
    ax[1, 1].set_xlim(8,55)
    ax[2, 1].set_xlim(8,55)

    ax[0, 0].set_ylim(-0.12,0.30)
    ax[0, 1].set_ylim(-0.12,0.30)
    ax[1, 0].set_ylim(-0.08,0.09)
    ax[1, 1].set_ylim(-0.08,0.09)
    ax[2, 0].set_ylim(-0.05,0.03)
    ax[2, 1].set_ylim(-0.05,0.03)

    ax[2, 0].set_xlabel(r'\boldmath$p_T~(\mathrm{GeV})$', size = 15)
    ax[2, 1].set_xlabel(r'\boldmath$p_T~(\mathrm{GeV})$', size = 15)
    # plot_temp.xaxis.set_label_coords(1.05, 0.2)
    ax[0, 0].text(0.80,0.80,r'\boldmath$A_{LL}$',transform=ax[0,0].transAxes,size=15)

    ax[0, 0].tick_params(labelbottom=False)
    ax[0, 1].tick_params(labelbottom=False)
    ax[1, 0].tick_params(labelbottom=False)
    ax[1, 1].tick_params(labelbottom=False)

    ax[0, 1].tick_params(labelleft=False)
    ax[1, 1].tick_params(labelleft=False)
    ax[2, 1].tick_params(labelleft=False)


    py.tight_layout()
    py.subplots_adjust(hspace=0,wspace=0)
    filename = '%s/gallery/pjets.png'%wdir
    figure.savefig(filename, dpi = dpi)
    print('Saving figure to %s'%filename)

def plot_data_separate(wdir, istep, data_idx, idx, style, plot_with_factor, dpi):
    if not plot_with_factor:
        data_idx['plot-factor'] = np.ones(len(data_idx['value']))

    _, _, eta_bins = find_eta_bins(data_idx)

    figure = py.figure(figsize = [10.0, 5.0], dpi = 500)
    plot_temp = figure.add_subplot(1, 1, 1)

    plot_colors = ['r']
    # plot_colors = ['b', 'g', 'r', 'c', 'm', 'y']
    plot_styles = ['-']
    # plot_styles = ['-', '-.', '--', ':', '-', '--']
    for i in xrange(len(eta_bins)):
        pt = np.array(data_idx['pT'][eta_bins[i][0] - 1 : eta_bins[i][1]])
        value = np.array(data_idx['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]]) * \
            np.array(data_idx['value'][eta_bins[i][0] - 1 : eta_bins[i][1]])
        alpha = np.array(data_idx['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]]) * \
            np.array(data_idx['alpha'][eta_bins[i][0] - 1 : eta_bins[i][1]])
        # plot_temp.errorbar(pt, value, alpha, fmt = '%s.' % plot_colors[i])
        plot_temp.errorbar(pt, value, alpha, fmt = '%s.' % plot_colors[0], zorder = 10)
        theory = np.array(data_idx['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]]) * \
            np.array(data_idx['thy'][eta_bins[i][0] - 1 : eta_bins[i][1]])
        # plot_temp.plot(pt, theory, '%s%s' % (plot_colors[i], plot_styles[i]))
        plot_temp.plot(pt, theory, '%s%s' % (plot_colors[0], plot_styles[0]), zorder = 5)
        plot_temp.axhline(0.0, color = 'black', ls = ':', zorder = 0)

        if 'thy-rep' in data_idx:
            line_width = min(0.1, 10.0 / float(len(data_idx['thy-rep'])))
            for j in range(len(data_idx['thy-rep'])):
                thy = np.array(data_idx['thy-rep'][j][eta_bins[i][0] - 1 : eta_bins[i][1]]) * \
                               np.array(data_idx['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]])
                plot_temp.plot(pt, thy, color = 'gray', linewidth = line_width, zorder = 0)

    # legend = []
    # for i in range(len(pdf_sets)):
    #     legend.append(Line2D([0], [0], linewidth = 1, color = 'k', linestyle = '%s' % plot_styles[i], label = pdf_sets[i]))
    # plot_temp.legend(handles = legend)

    plot_temp.set_xlabel(r'$p_T$', size = 15)
    plot_temp.set_ylabel(r'$A_{LL}$', size = 15)
    # plot_temp.set_yscale('log')
    if data_idx['col'][0] == 'star':
        if idx == 20001:
            figure.savefig('%s/gallery/pjet-%s-2003-%d.pdf' % (wdir, data_idx['col'][0], istep), dpi = dpi)
        elif idx == 20002:
            figure.savefig('%s/gallery/pjet-%s-2005-%d.pdf' % (wdir, data_idx['col'][0], istep), dpi = dpi)
        elif idx == 20003:
            figure.savefig('%s/gallery/pjet-%s-2006-%d.pdf' % (wdir, data_idx['col'][0], istep), dpi = dpi)
        elif idx == 20004:
            figure.savefig('%s/gallery/pjet-%s-2009-%d.pdf' % (wdir, data_idx['col'][0], istep), dpi = dpi)
        elif idx == 20006:
            figure.savefig('%s/gallery/pjet-%s-2012-%d.pdf' % (wdir, data_idx['col'][0], istep), dpi = dpi)
    else:
        figure.savefig('%s/gallery/pjet-%s-%d.pdf' % (wdir, data_idx['col'][0], istep), dpi = dpi)

def plot_data_on_theory(wdir, istep, data, labels, style, dpi):
    skip_single_point_bin = True
    count = count_eta_bins(data, skip_single_point_bin)

    if count == 6:
        n_rows, n_columns = 3, 2
        figure = py.figure(figsize = (n_columns * 5.0, n_rows * 3.0))
        axs = [py.subplot(n_rows, n_columns, i) for i in range(1, 7)]
    else:
        print('number of datasets (eta bins) is different from 6')
        print('you need to specify number of rows and columns for subplots in "plot_data"')
        return

    count = 0
    for dataset in data:
        _, absolute_eta, eta_bins = find_eta_bins(data[dataset])

        for i in xrange(len(eta_bins)):
            pt = np.array(data[dataset]['pT'][eta_bins[i][0] - 1 : eta_bins[i][1]])
            if len(pt) == 1:
                if skip_single_point_bin:
                    print('dataset %d, eta bin %d is skipped for having only one point (could be caused by cuts)' % (dataset, i))
                    continue
            if absolute_eta:
                eta_min, eta_max = str(data[dataset]['eta-abs-min'][eta_bins[i][0] - 1]), str(data[dataset]['eta-abs-max'][eta_bins[i][0] - 1])
                eta_range = r'|\eta|~\in~[%s,~%s]' % (eta_min, eta_max)
            else:
                eta_min, eta_max = str(data[dataset]['eta-min'][eta_bins[i][0] - 1]), str(data[dataset]['eta-max'][eta_bins[i][0] - 1])
                eta_range = r'\eta~\in~[%s,~%s]' % (eta_min, eta_max)

            value  = np.array(data[dataset]['value'][eta_bins[i][0] - 1 : eta_bins[i][1]])
            alpha  = np.array(data[dataset]['alpha'][eta_bins[i][0] - 1 : eta_bins[i][1]])
            theory = np.array(data[dataset]['thy'][eta_bins[i][0] - 1 : eta_bins[i][1]])
            axs[count].errorbar(pt, value / theory, alpha / theory, fmt = 'x', color = 'k', ms = 5.0, capsize = 5.0, zorder = 0)
            axs[count].axhline(1.0, color = 'black', ls = 'dashdot', linewidth = 0.5)
            # axs[count].set_ylim(-1.0, 3.0)

            if (style == 'best_lines') or (style == 'all_lines') or (style == 'cluster_dots') or (style == 'cluster_bands'):
                line_width = min(0.1, 10.0 / float(len(data[dataset]['thy-rep'])))
                if style == 'cluster_dots':
                    pt_offset = (max(pt) - min(pt)) / 80.0
                    sizes = np.arange(10.0, 0.0, -3.0)
                    count_clusters = [0 for _ in list(set(labels['cluster']))]
                    for j in range(len(data[dataset]['thy-rep'])):
                        count_clusters[labels['cluster'][j]] += 1
                        thy = np.array(data[dataset]['thy-rep'][j][eta_bins[i][0] - 1 : eta_bins[i][1]])
                        size = sizes[labels['cluster'][j]]
                        z_order = int(sizes[len(sizes) - labels['cluster'][j] - 1])
                        color = labels['color'][labels['cluster'][j]]
                        offset = float(labels['cluster'][j] + 1) * pt_offset
                        if count_clusters[labels['cluster'][j]] == 1:
                            label = r'$\overline{\chi _{%d}^2} = %.2f$' % (labels['cluster'][j], labels['chi_2'][labels['cluster'][j]])
                            axs[count].scatter(pt + offset, thy / theory, color = color, s = size, zorder = z_order, label = label)
                        else:
                            axs[count].scatter(pt + offset, thy / theory, color = color, s = size, zorder = z_order)
                elif style == 'cluster_bands':
                    pt_offset = (max(pt) - min(pt)) / 80.0
                    thy = {_: [] for _ in list(set(labels['cluster']))}
                    for j in range(len(labels['cluster'])):
                        thy[labels['cluster'][j]].append(np.array(data[dataset]['thy-rep'][j][eta_bins[i][0] - 1 : eta_bins[i][1]]))
                    for j in thy:
                        offset = float(labels['cluster'][j] + 1) * pt_offset
                        color = labels['color'][labels['cluster'][j]]
                        label = r'$\overline{\chi _{%d}^2} = %.2f$' % (labels['cluster'][j], labels['chi_2'][labels['cluster'][j]])
                        ys = np.mean(thy[j], axis = 0)
                        ys_d = np.std(thy[j], axis = 0)
                        axs[count].vlines(pt + offset, (ys - ys_d) / theory, (ys + ys_d) / theory, color = color, linestyles = 'solid', linewidth = 3.0, label = label)
                else:
                    for j in range(len(data[dataset]['thy-rep'])):
                        thy = np.array(data[dataset]['thy-rep'][j][eta_bins[i][0] - 1 : eta_bins[i][1]])
                        axs[count].plot(pt, thy / theory, color = 'gray', linewidth = line_width, zorder = 1)
            text = r'\mathrm{%s}' % data[dataset]['col'][0].upper()
            if dataset == 20001:
                text += '~2003'
            elif dataset == 20002:
                text += '~2005'
            elif dataset == 20003:
                text += '~2006'
            elif dataset == 20004:
                text += '~2009'
            elif dataset == 20005:
                text += '~2005'
            elif dataset == 20006:
                text += '~2012'
            text += ',~%s' % eta_range
            text = r'$%s$' % text
            axs[count].text(0.1, 0.85, text, color = 'k', transform = axs[count].transAxes, size = 11)

            count += 1

        ## this part changes label ordering and sets width of legends
        # handles, labels = axs[count].get_legend_handles_labels()
        # for _ in range(2):
        #     handles.append(handles[1])
        #     del handles[1]
        #     labels.append(labels[1])
        #     del labels[1]
        # legends = axs[count].legend(handles, labels, fontsize = 4, loc = 'best')
        # for line in legends.get_lines():
        #     line.set_linewidth(1.0)

    i_last_row = range(len(axs) - 1, len(axs) - 1 - n_columns, -1)
    for i in range(len(axs)):
        # axs[i].semilogx()
        if i in i_last_row:
            axs[i].set_xlabel(r'\boldmath$p_T~(\mathrm{GeV})$', size = 13)
            # axs[i].xaxis.set_label_coords(0.97, -0.025)
        if (i % n_columns) == 0:
            axs[i].set_ylabel(r'$\mathrm{data} / \mathrm{theory}$', size = 13)
        axs[i].tick_params(axis = 'both', which = 'both', right = True, top = True, direction = 'in', labelsize = 20)

    legends = axs[0].legend(frameon = 0, loc = 'best')
    # for line in legends.get_lines():
    #     line.set_linewidth(1.0)

    # axs[0, 0].set_ylim(- 5.0, 0.0)
    # axs[0, 1].set_ylim(- 5.0, 5.0)
    # axs[1, 0].set_ylim(- 2.0, 4.0)
    # axs[1, 1].set_ylim(- 2.0, 3.0)
    # axs[2, 0].set_ylim(- 7.0, -2.0)
    # axs[2, 1].set_ylim(- 2.0, 4.0)

    py.tight_layout()
    py.savefig('%s/gallery/pjet-ratio-%d.pdf' % (wdir, istep), dpi = dpi, bbox_inches = 'tight')
    py.close()

def plot_data_by_channel(wdir, istep, data, labels, style, dpi):
    skip_single_point_bin = True
    count = count_eta_bins(data, skip_single_point_bin)

    if count == 6:
        n_rows, n_columns = 2, 3
        figure = py.figure(figsize = (n_columns * 5.0, n_rows * 3.0))
        axs = [py.subplot(n_rows, n_columns, i) for i in range(1, 7)]
    else:
        print('number of datasets (eta bins) is different from 6')
        print('you need to specify number of rows and columns for subplots in "plot_data"')
        return

    delta_sigma_by_channels = load('%s/data/pjet_by_channel.dat' % wdir)

    count = 0
    datasets = sorted(delta_sigma_by_channels.keys())
    channels = sorted(delta_sigma_by_channels[datasets[0]].keys())
    if 'total' in channels:
        del channels[channels.index('total')]
    n_channel_per_count = 2
    n_sub_channel = int(math.ceil(len(channels) / n_channel_per_count))
    channel_labels = {}
    for i in range(n_sub_channel):
        channel_labels[i + 1] = {}
        sub_channels = channels[i * n_channel_per_count : (i + 1) * n_channel_per_count]
        for sub_channel in sub_channels:
            channel_labels[i + 1][sub_channel] = gen_channel_labels(sub_channel)
    for dataset in datasets:
        _, absolute_eta, eta_bins = find_eta_bins(data[dataset])

        for i in xrange(len(eta_bins)):
            pt = np.array(data[dataset]['pT'][eta_bins[i][0] - 1 : eta_bins[i][1]])
            if len(pt) == 1:
                if skip_single_point_bin:
                    print('dataset %d, eta bin %d is skipped for having only one point (could be caused by cuts)' % (dataset, i))
                    continue
            if absolute_eta:
                eta_min, eta_max = str(data[dataset]['eta-abs-min'][eta_bins[i][0] - 1]), str(data[dataset]['eta-abs-max'][eta_bins[i][0] - 1])
                eta_range = r'|\eta|~\in~[%s,~%s]' % (eta_min, eta_max)
            else:
                eta_min, eta_max = str(data[dataset]['eta-min'][eta_bins[i][0] - 1]), str(data[dataset]['eta-max'][eta_bins[i][0] - 1])
                eta_range = r'\eta~\in~[%s,~%s]' % (eta_min, eta_max)

            if (style == 'best_lines') or (style == 'all_lines') or (style == 'cluster_lines'):
                line_width = min(0.1, 10.0 / float(len(data[dataset]['thy-rep'])))
                if style == 'cluster_lines':
                    colors = ['r', 'b', 'g', 'm', 'y', 'cyan']
                    line_styles = ['solid', 'dashed', 'dotted']
                    count_clusters = [0 for _ in list(set(labels['cluster']))]
                    theories = {}
                    theory_total = {}
                    for channel in channels:
                        theories[channel] = {}
                        for cluster in list(set(labels['cluster'])):
                            theories[channel][cluster] = []
                            theory_total[cluster] = []
                    for j in range(len(delta_sigma_by_channels[dataset][channels[0]])):
                        cluster = labels['cluster'][j]
                        # count_clusters[labels['cluster'][j]] += 1
                        theory_total[cluster].append(np.abs(delta_sigma_by_channels[dataset]['total'][j])[eta_bins[i][0] - 1 : eta_bins[i][1]])
                        for k in range(len(channels)):
                            theories[channels[k]][cluster].append(np.array(delta_sigma_by_channels[dataset][channels[k]][j])[eta_bins[i][0] - 1 : eta_bins[i][1]])
                            # z_order = int(sizes[len(sizes) - labels['cluster'][j] - 1])
                            # color = labels['color'][labels['cluster'][j]]
                    for cluster in list(set(labels['cluster'])):
                        line_style = line_styles[cluster]
                        denominator = np.mean(theory_total[cluster], axis = 0)
                        for k in range(len(channels)):
                            color = colors[k]
                            numerator = np.mean(theories[channels[k]][cluster], axis = 0)
                            if (count == 0) and (k == 0):
                                label = r'$\overline{\chi _{%d}^2} = %.2f$' % (cluster, labels['chi_2'][cluster])
                                axs[count].plot(pt, numerator / denominator, color = color, linestyle = line_style, label = label)
                            elif count in channel_labels:
                                if (cluster == 0) and (channels[k] in channel_labels[count]):
                                    label = r'$%s$' % channel_labels[count][channels[k]]
                                    axs[count].plot(pt, numerator / denominator, color = color, linestyle = line_style, label = label)
                                else:
                                    axs[count].plot(pt, numerator / denominator, color = color, linestyle = line_style)
                            else:
                                axs[count].plot(pt, numerator / denominator, color = color, linestyle = line_style)
                else:
                    pass
            text = r'\mathrm{%s}' % data[dataset]['col'][0].upper()
            if dataset == 20001:
                text += '~2003'
            elif dataset == 20002:
                text += '~2005'
            elif dataset == 20003:
                text += '~2006'
            elif dataset == 20004:
                text += '~2009'
            elif dataset == 20005:
                text += '~2005'
            elif dataset == 20006:
                text += '~2012'
            text += ',~%s' % eta_range
            text = r'$%s$' % text
            axs[count].text(0.025, 0.75, text, color = 'k', transform = axs[count].transAxes, size = 11)
            ## adjust_text is not working...
            # texts = [axs[count].text(0.025, 0.7, text, color = 'k', transform = axs[count].transAxes, size = 11)]
            # texts = [axs[count].text(0.025, 0.7, text, color = 'k', size = 11)]
            # adjust_text(texts, only_move = {'points': 'y', 'text': 'y'})
            # adjust_text(texts)

            count += 1

    i_last_row = range(len(axs) - 1, len(axs) - 1 - n_columns, -1)
    for i in range(len(axs)):
        # axs[i].semilogy()
        # axs[i].set_ylim(0.0, 1.0)
        axs[i].axhline(0.0, color = 'black', ls = 'dashdot', linewidth = 0.5)
        if i in i_last_row:
            axs[i].set_xlabel(r'\boldmath$p_T~(\mathrm{GeV})$', size = 13)
            # axs[i].xaxis.set_label_coords(0.97, -0.025)
        if (i % n_columns) == 0:
            axs[i].set_ylabel(r'$\sigma _{\mathrm{channel}} / \left| \sigma _{\mathrm{total}} \right|$', size = 13)
        axs[i].tick_params(axis = 'both', which = 'both', right = True, top = True, direction = 'in', labelsize = 20)

    label_indices = list(channel_labels.keys())
    for _ in label_indices:
        legends = axs[_].legend(frameon = 0, loc = 'lower left')
        # for line in legends.get_lines():
        #     line.set_linewidth(1.0)
    legends = axs[0].legend(frameon = 0, loc = 'upper right')
    for _ in range(len(legends.legendHandles)):
        legends.legendHandles[_].set_color('k')

    py.tight_layout()
    py.savefig('%s/gallery/pjet-channels-%d.pdf' % (wdir, istep), dpi = dpi, bbox_inches = 'tight')
    py.close()

def plot_data_by_helicity(wdir, istep, data, labels, style, dpi):
    skip_single_point_bin = True
    sigma_by_channels       = load('%s/data/jet_by_channel.dat'  % wdir)
    delta_sigma_by_channels = load('%s/data/pjet_by_channel.dat' % wdir)
    datasets = sorted(sigma_by_channels.keys())
    channels = sorted(sigma_by_channels[datasets[0]].keys())
    if 'total' in channels:
        del channels[channels.index('total')]

    if len(channels) == 6:
        n_rows, n_columns = 6, 2
    else:
        print('number of channels is different from 6')
        print('you need to specify number of rows and columns for subplots in "plot_data"')
        return

    channel_labels = {}
    for channel in channels:
        channel_labels[channel] = gen_channel_labels(channel)
    for dataset in datasets:
        _, absolute_eta, eta_bins = find_eta_bins(data[dataset])
        print('\tplotting dataset %d...' % dataset)

        for i in xrange(len(eta_bins)):
            pt = np.array(data[dataset]['pT'][eta_bins[i][0] - 1 : eta_bins[i][1]])
            if len(pt) == 1:
                if skip_single_point_bin:
                    print('\tdataset %d, eta bin %d is skipped for having only one point (could be caused by cuts)' % (dataset, i))
                    continue
            if absolute_eta:
                eta_min, eta_max = str(data[dataset]['eta-abs-min'][eta_bins[i][0] - 1]), str(data[dataset]['eta-abs-max'][eta_bins[i][0] - 1])
                eta_range = r'|\eta|~\in~[%s,~%s]' % (eta_min, eta_max)
            else:
                eta_min, eta_max = str(data[dataset]['eta-min'][eta_bins[i][0] - 1]), str(data[dataset]['eta-max'][eta_bins[i][0] - 1])
                eta_range = r'\eta~\in~[%s,~%s]' % (eta_min, eta_max)

            figure = py.figure(figsize = (n_columns * 10.0, n_rows * 3.0))
            axs = [py.subplot(n_rows, n_columns, _) for _ in range(1, 13)]
            if (style == 'best_lines') or (style == 'all_lines') or (style == 'cluster_lines') or (style == 'cluster_dots'):
                line_width = min(0.1, 10.0 / float(len(data[dataset]['thy-rep'])))
                if style == 'cluster_dots':
                    colors = labels['color']
                    sizes = np.arange(10.0, 0.0, -3.0)
                    # line_styles = ['solid', 'dashed', 'dotted']
                    count_clusters = [0 for _ in list(set(labels['cluster']))]
                    pt_offset = (max(pt) - min(pt)) / 80.0
                    for j in range(len(sigma_by_channels[dataset][channels[0]])):
                        cluster = labels['cluster'][j]
                        count_clusters[cluster] += 1
                        size = sizes[labels['cluster'][j]]
                        z_order = int(sizes[len(sizes) - labels['cluster'][j] - 1])
                        offset = float(cluster) * pt_offset
                        for k in range(len(channels)):
                            sigma       = np.array(sigma_by_channels[dataset][channels[k]][j])[eta_bins[i][0] - 1 : eta_bins[i][1]]
                            delta_sigma = np.array(delta_sigma_by_channels[dataset][channels[k]][j])[eta_bins[i][0] - 1 : eta_bins[i][1]]
                            numerator   = (sigma + delta_sigma) / 2.0
                            # pt_positive, pt_negative = [], []
                            # numerator_positive, numerator_negative = [], []
                            # sigma_positive, sigma_negative = [], []
                            # for _ in range(len(numerator)):
                            #     if numerator[_] >= 0.0:
                            #         pt_positive.append(pt[_])
                            #         numerator_positive.append(numerator[_])
                            #         sigma_positive.append(sigma[_])
                            #     else:
                            #         pt_negative.append(pt[_])
                            #         numerator_negative.append(numerator[_])
                            #         sigma_negative.append(sigma[_])
                            if count_clusters[cluster] == 1:
                                label = r'$\overline{\chi _{%d}^2} = %.2f$' % (cluster, labels['chi_2'][cluster])
                                axs[k * 2].scatter(pt + offset, numerator / sigma, color = colors[cluster], s = size, zorder = z_order, marker = '.', label = label)
                            else:
                                axs[k * 2].scatter(pt + offset, numerator / sigma, color = colors[cluster], s = size, zorder = z_order, marker = '.')
                            numerator = (sigma - delta_sigma) / 2.0
                            if count_clusters[cluster] == 1:
                                label = r'$\overline{\chi _{%d}^2} = %.2f$' % (cluster, labels['chi_2'][cluster])
                                axs[(k * 2) + 1].scatter(pt + offset, numerator / sigma, color = colors[cluster], s = size, zorder = z_order, marker = '.', label = label)
                            else:
                                axs[(k * 2) + 1].scatter(pt + offset, numerator / sigma, color = colors[cluster], s = size, zorder = z_order, marker = '.')
                    for k in range(len(channels)):
                        text = channel_labels[channels[k]]
                        text = r'$%s$' % text
                        axs[k * 2].text(0.1, 0.8, text, color = 'k', transform = axs[k * 2].transAxes, size = 17)
                        axs[(k * 2) + 1].text(0.1, 0.8, text, color = 'k', transform = axs[(k * 2) + 1].transAxes, size = 17)
                        # y_limits = axs[k].get_ylim()
                else:
                    pass
            # text = r'\mathrm{%s}' % data[dataset]['col'][0].upper()
            # if dataset == 20001:
            #     text += '~2003'
            # elif dataset == 20002:
            #     text += '~2005'
            # elif dataset == 20003:
            #     text += '~2006'
            # elif dataset == 20004:
            #     text += '~2009'
            # elif dataset == 20005:
            #     text += '~2005'
            # elif dataset == 20006:
            #     text += '~2012'
            # text += ',~%s' % eta_range
            # text = r'$%s$' % text
            # axs[count].text(0.025, 0.75, text, color = 'k', transform = axs[count].transAxes, size = 11)

            i_last_row = range(len(axs) - 1, len(axs) - 1 - n_columns, -1)
            axs[0].title.set_text(r'$\sigma _{\mathrm{channel}}^+ / \sigma _{\mathrm{channel}}$')
            axs[1].title.set_text(r'$\sigma _{\mathrm{channel}}^- / \sigma _{\mathrm{channel}}$')
            for _ in range(len(axs)):
                # axs[_].semilogy()
                axs[_].set_ylim(-0.5, 1.5)
                axs[_].axhline(0.0, color = 'black', ls = 'dashdot', linewidth = 0.5)
                axs[_].axhline(1.0, color = 'black', ls = 'dashdot', linewidth = 0.5)
                if _ in i_last_row:
                    axs[_].set_xlabel(r'\boldmath$p_T~(\mathrm{GeV})$', size = 13)
                    # axs[_].xaxis.set_label_coords(0.97, -0.025)
                if (_ % n_columns) == 0:
                    # axs[_].set_ylabel(r'$\sigma _{\mathrm{channel}}^+ / \sigma _{\mathrm{channel}}$', size = 13)
                    pass
                axs[_].tick_params(axis = 'both', which = 'both', right = True, top = True, direction = 'in', labelsize = 20)

            legends = axs[0].legend(frameon = 0, loc = 'upper right')
            for line in legends.get_lines():
                line.set_linewidth(1.0)
            # for _ in range(len(legends.legendHandles)):
            #     legends.legendHandles[_].set_color('k')

            py.subplots_adjust(left = 0.12, bottom = 0.08, right = 0.99, top = 0.97, wspace = None, hspace = 0.2)
            py.tight_layout()
            py.savefig('%s/gallery/pjet_%d_%d-helicity-%d.pdf' % (wdir, dataset, i, istep), dpi = dpi, bbox_inches = 'tight')
            py.close()

def compute_sigma_by_channel(wdir):
    load_config('%s/input.py' % wdir)
    istep = core.get_istep()
    print('\ncomputing individual channels of polarized and unpolarized jets from %s' % (wdir))

    conf['aux'] = aux.AUX()
    conf['mellin']  = mellin.MELLIN(npts = 4)
    conf['path2jettab']  = '%s/grids/grids-jets'  % os.environ['FITPACK']
    conf['path2pjettab'] = '%s/grids/grids-pjets' % os.environ['FITPACK']
    conf['pjet tabs'] = obslib.pjets.reader.READER().load_data_sets('pjet')

    jetmellin  = JETMELLIN()
    pjetmellin = PJETMELLIN()
    jetmellin.load_factab(conf['pjet_qr_fit']['method'], conf['pjet_qr_fit']['f_scale'], conf['pjet_qr_fit']['r_scale'], data_tables = conf['pjet tabs'])
    pjetmellin.load_factab(conf['pjet_qr_fit']['method'], conf['pjet_qr_fit']['f_scale'], conf['pjet_qr_fit']['r_scale'])

    replicas = core.get_replicas(wdir)
    core.mod_conf(istep, replicas[0]) ## set conf as specified in istep

    conf['predict'] = True
    resman = RESMAN(nworkers = 1, parallel = False, datasets = False)
    parman = resman.parman
    parman.order = replicas[0]['order'][istep]
    ppdf = conf['ppdf']

    sigma_by_channels = {}
    for i in conf['pjet tabs'].keys():
        sigma_by_channels[i] = {}
        for channel in pjetmellin.channels:
            sigma_by_channels[i][channel] = []
        sigma_by_channels[i]['total'] = []

    delta_sigma_by_channels = {}
    for i in conf['pjet tabs'].keys():
        delta_sigma_by_channels[i] = {}
        for channel in pjetmellin.channels:
            delta_sigma_by_channels[i][channel] = []
        delta_sigma_by_channels[i]['total'] = []

    for i in range(len(replicas)):
        lprint('%d/%d' % (i + 1, len(replicas)))
        parman.order = copy.copy(replicas[i]['order'][istep])
        parman.set_new_params(replicas[i]['params'][istep], initial = True)

        ## this calculation has been verified to be correct by printing out task['jet_theory'] and task['pjet_theory'] in PJETMELLIN.update()
        sigma_by_channels_temp = jetmellin.compute_cr_fac_channel(data_tables = conf['pjet tabs'])
        delta_sigma_by_channels_temp = pjetmellin.compute_cr_fac_channel()
        for j in sigma_by_channels_temp.keys():
            indices = sorted(sigma_by_channels_temp[j].keys())
            for channel in sigma_by_channels_temp[j][indices[0]]['sigma_channel'].keys():
                temp = []
                delta_temp = []
                for k in indices:
                    temp.append(sigma_by_channels_temp[j][k]['sigma_channel'][channel] * 0.38942957e9)
                    delta_temp.append(delta_sigma_by_channels_temp[j][k]['sigma_channel'][channel])
                sigma_by_channels[j][channel].append(temp)
                delta_sigma_by_channels[j][channel].append(delta_temp)
    print()
    save(sigma_by_channels      , '%s/data/jet_by_channel.dat'  % wdir)
    save(delta_sigma_by_channels, '%s/data/pjet_by_channel.dat' % wdir)

def check_sigma_positivity(wdir):
    load_config('%s/input.py' % wdir)
    istep   = core.get_istep()
    labels  = load('%s/data/labels-%d.dat' % (wdir, istep))
    labels['cluster'] = [0 for _ in labels['cluster']]
    print('\nclassifying replicas according to positivity of individual channels of polarized jets from %s' % (wdir))

    conf['aux'] = aux.AUX()
    conf['mellin']  = mellin.MELLIN(npts = 4)
    conf['pjet tabs'] = obslib.pjets.reader.READER().load_data_sets('pjet')

    sigma_by_channels       = load('%s/data/jet_by_channel.dat'  % wdir)
    delta_sigma_by_channels = load('%s/data/pjet_by_channel.dat' % wdir)

    datasets = sorted(sigma_by_channels.keys())
    channels = sigma_by_channels[datasets[0]].keys()
    class next_replica(Exception): pass
    for i in range(len(sigma_by_channels[datasets[0]][channels[0]])):
        try:
            for dataset in datasets:
                for channel in channels:
                    for index in range(len(sigma_by_channels[dataset][channel][i])):
                        sigma       = sigma_by_channels[dataset][channel][i][index]
                        delta_sigma = delta_sigma_by_channels[dataset][channel][i][index]
                        if ((sigma + delta_sigma) < 0.0) or ((sigma - delta_sigma) < 0.0):
                            labels['cluster'][i] = 1
                            raise next_replica()
        except next_replica:
            pass

    labels = classifier.get_cluster_chi_2(wdir, istep, labels)
    save(labels, '%s/data/labels-%d.dat' % (wdir, istep))

def gen_channel_labels(channel):
    if (channel == 'q_q') or (channel == 'q q'):
        label = r'q q'
    elif (channel == 'q_qb') or (channel == 'q qb'):
        label = r'q \overline{q}'
    elif (channel == 'q_qp') or (channel == 'q qp'):
        label = r'q q^\prime'
    elif (channel == 'q_qbp') or (channel == 'q qbp'):
        label = r'q \overline{q}^\prime'
    elif (channel == 'q_g') or (channel == 'q g'):
        label = r'q g'
    elif (channel == 'g_g') or (channel == 'g g'):
        label = r'g g'
    return label

def magnitude(value):
    if (value == 0.0):
        order = 0
    else:
        order = int(math.floor(math.log10(abs(value))))
    return order

def plot_obs(wdir, kc, task = 1, plot_with_factor = False, style = 'best', dpi = 200):
    ## attribute 'plot_with_factor' is inherited from jet and not yet being used here
    ## it is kept there because someday it might be useful
    ## availables styles are
    ## 'best'         : plot using only the best cluster
    ## 'best_lines'   : plot using only best cluster, also show replicas by lines from the best cluster
    ## 'all'          : plot using all clusters
    ## 'all_lines'    : plot using all cluster, also show replicas by lines from all clusters without distinguishing them
    ## 'cluster_lines': plot using all cluster, also show replicas by lines from all clusters distinctively
    ## 'cluster_dots' : plot using all cluster, also show replicas by dots from all clusters distinctively
    ## 'cluster_bands': plot using all cluster, also show replicas by bands from all clusters distinctively

    if task == 1:
        print('\nplotting PJET data from %s' % (wdir))
    elif task == 2:
        print('\nplotting PJET data over theory from %s' % (wdir))
    elif task == 3:
        print('\nplotting PJET data by channel from %s' % (wdir))
    elif task == 4:
        print('\nplotting PJET data by helicity from %s' % (wdir))

    load_config('%s/input.py' % wdir)
    istep = core.get_istep()
    replicas = core.get_replicas(wdir)
    core.mod_conf(istep, replicas[0]) #--set conf as specified in istep

    predictions = load('%s/data/predictions-%d.dat' % (wdir, istep))
    if ('pjet' not in predictions['reactions']) or (len(predictions['reactions']['pjet']) == 0):
        print('PJET is not in data file')
        return
    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)
    best_cluster = cluster[0]

    data = predictions['reactions']['pjet']

    count = 0
    for idx in data:
        predictions = copy.copy(data[idx]['prediction-rep'])
        del data[idx]['prediction-rep']
        del data[idx]['residuals-rep']
        del data[idx]['shift-rep']
        if (style == 'best') or (style == 'best_lines'):
            best_predictions = [predictions[i] for i in range(len(predictions)) if cluster[i] == best_cluster]
            data[idx]['thy'] = np.mean(best_predictions, axis = 0)
            data[idx]['dthy'] = np.std(best_predictions, axis = 0)
            if style == 'best_lines':
                data[idx]['thy-rep'] = best_predictions
        elif (style == 'all') or (style == 'all_lines') or (style == 'cluster_lines') or (style == 'cluster_dots') or (style == 'cluster_bands'):
            all_predictions = [predictions[i] for i in range(len(predictions))]
            data[idx]['thy'] = np.mean(all_predictions, axis = 0)
            data[idx]['dthy'] = np.std(all_predictions, axis = 0)
            if (style == 'all_lines') or (style == 'cluster_lines') or (style == 'cluster_dots') or (style == 'cluster_bands'):
                count += 1
                data[idx]['thy-rep'] = all_predictions
                if count == 1:
                    colors = labels['cluster_colors']
                    labels = {'cluster': cluster, 'color': colors, 'chi_2': cluster_average_chi2}

    if task == 1:
        plot_data(wdir, istep, data,style, plot_with_factor, dpi)
        # for idx in data:
        #     plot_data_separate(wdir, istep, data[idx], idx, style, plot_with_factor, dpi)
    elif task == 2:
        plot_data_on_theory(wdir, istep, data, labels, style, dpi)
    elif task == 3:
        plot_data_by_channel(wdir, istep, data, labels, style, dpi)
    elif task == 4:
        plot_data_by_helicity(wdir, istep, data, labels, style, dpi)

    return

if __name__ == "__main__":
    pass
