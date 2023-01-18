#!/usr/bin/env python
import sys, os
import numpy as np
import copy
from subprocess import Popen, PIPE, STDOUT
import pandas as pd
import scipy as sp
# import time
# from glob import glob

## matplotlib
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
from matplotlib.ticker import MultipleLocator, FormatStrFormatter ## for minor ticks in x label
matplotlib.rc('text', usetex = True)
import pylab as py

## from fitpack tools
from tools.tools     import load, save, checkdir, lprint
from tools.config    import conf, load_config

## from fitpack analysis
from analysis.corelib import core
from analysis.corelib import classifier

## from fitpack obslib
from obslib.jets.jet_tools import find_eta_bins

## < general functions for polarized and unpolarized jets

def count_eta_bins(data, skip_single_point_bin):
    count = 0
    for i in data:
        _, absolute_eta, eta_bins = find_eta_bins(data[i])

        for j in xrange(len(eta_bins)):
            if eta_bins[j][0] == eta_bins[j][1]:
                if skip_single_point_bin:
                    continue
                else:
                    count += 1
            else:
                count += 1
    return count

##   general functions for polarized and unpolarized jets >

def plot_data(wdir, data, istep, dpi, plot_with_factor):
    if len(data) == 4:
        n_rows, n_columns = 2, 2
    else:
        print('number of dataset is different from 4')
        print('you need to specify number of rows and columns for subplots in "plot_data"')
        return
    figure, ax = py.subplots(n_rows, n_columns)

    plot_styles = ['-']
    plot_colors = ['b', 'g', 'm', 'y', 'c', 'r']
    # plot_styles = ['-', '-.', '--', ':', '-', '--']
    plot_with_factor_input = plot_with_factor

    count = 0
    for idx in data:
        i_row, i_column = int(count / n_columns) , int(count % n_columns)
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

        for i in range(len(eta_bins)):
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

            pt = np.array(data_idx['pT'][eta_bins[i][0] - 1 : eta_bins[i][1]])
            value = np.array(data_idx['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]]) * \
                np.array(data_idx['value'][eta_bins[i][0] - 1 : eta_bins[i][1]])
            alpha = np.array(data_idx['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]]) * \
                np.array(data_idx['alpha'][eta_bins[i][0] - 1 : eta_bins[i][1]])
            ax[i_row, i_column].errorbar(pt, value, alpha, fmt = '%s.' % plot_colors[i], zorder = 10, capsize=2.0)

            theory = np.array(data_idx['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]]) * \
                np.array(data_idx['thy'][eta_bins[i][0] - 1 : eta_bins[i][1]])
            ax[i_row, i_column].plot(pt, theory, '%s%s' % (plot_colors[i], plot_styles[0]), label = eta_range, zorder = 5)
            ax[i_row, i_column].axhline(0.0, color = 'black', ls = ':', zorder = 0)

            if 'thy-rep' in data_idx:
                line_width = min(0.1, 10.0 / float(len(data_idx['thy-rep'])))
                for j in range(len(data_idx['thy-rep'])):
                    thy = np.array(data_idx['thy-rep'][j][eta_bins[i][0] - 1 : eta_bins[i][1]]) * \
                                   np.array(data_idx['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]])
                    ax[i_row, i_column].plot(pt, thy, color = 'gray', linewidth = line_width, zorder = 0)

        ax[i_row, i_column].set_yscale('log')
        ax[i_row, i_column].relim()
        ax[i_row, i_column].autoscale_view()
        if (idx != 10001) and (idx != 10002):
            ax[i_row, i_column].legend(fontsize = 10, loc = 'lower left',frameon=False,handletextpad=0.5,handlelength=0.0)
        if idx == 10001:
            year = 'run II'
        elif idx == 10002:
            year = 'run II'
        elif idx == 10003:
            year = 'MB'
        elif idx == 10004:
            year = 'HT'
        #ax[i_row, i_column].title.set_text(r'$\mathrm{%s~%s}$' % (data_idx['col'][0].upper(), year.replace(' ', '~')))
        ax[i_row, i_column].text(0.45,0.85,r'\boldmath$\mathrm{%s~%s}$' % (data_idx['col'][0].upper(), year.replace(' ', '~')),transform=ax[i_row,i_column].transAxes,size=15)
        ax[i_row, i_column].tick_params(axis='both',which='both',top=True,right=True,direction='in')
        count += 1

    ax[1, 0].set_xlabel(r'$p_T~(\mathrm{GeV})$', size = 15)
    ax[1, 1].set_xlabel(r'$p_T~(\mathrm{GeV})$', size = 15)
    # plot_temp.xaxis.set_label_coords(1.05, 0.2)
    ax[0, 0].set_ylabel(r'$\frac{\mathrm{d}^2\sigma}{\mathrm{d}y \mathrm{d} p_T}~{\scriptstyle [\mathrm{nb} / (\mathrm{GeV} / c)]}$', size = 17)
    ## the units of D0 and CDF can be different
    ax[1, 0].set_ylabel(r'$\frac{\mathrm{d}^2\sigma}{2 \pi \mathrm{d}y \mathrm{d} p_T}~{\scriptstyle [\mathrm{nb} / (\mathrm{GeV} / c)]}$', size = 17)
    ## y labeling has to be more automatic

    py.tight_layout()
    filename = '%s/gallery/jets.png'%wdir
    figure.savefig(filename, dpi = dpi)
    print('Saving figure to %s'%filename)

def plot_data_separate(wdir, data_idx, show_cluster, istep, idx, dpi, plot_with_factor):
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

    figure = py.figure(figsize = [10.0, 5.0], dpi = 500)
    plot_temp = figure.add_subplot(1, 1, 1)

    plot_styles = ['-']
    plot_colors = ['b', 'g', 'm', 'y', 'c', 'r']
    # plot_styles = ['-', '-.', '--', ':', '-', '--']
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

        pt = np.array(data_idx['pT'][eta_bins[i][0] - 1 : eta_bins[i][1]])
        value = np.array(data_idx['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]]) * \
            np.array(data_idx['value'][eta_bins[i][0] - 1 : eta_bins[i][1]])
        alpha = np.array(data_idx['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]]) * \
            np.array(data_idx['alpha'][eta_bins[i][0] - 1 : eta_bins[i][1]])
        plot_temp.errorbar(pt, value, alpha, fmt = '%s.' % plot_colors[i], zorder = 10)
        theory = np.array(data_idx['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]]) * \
            np.array(data_idx['thy'][eta_bins[i][0] - 1 : eta_bins[i][1]])
        plot_temp.plot(pt, theory, '%s%s' % (plot_colors[i], plot_styles[0]), label = eta_range, zorder = 5)

        # theory_deviation = np.array(data_idx['dthy'][eta_bins[i][0] - 1 : eta_bins[i][1]]) * np.array(data_idx['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]])
        # plot_temp.fill_between(pt, theory - (0.5 * theory_deviation), theory + (0.5 * theory_deviation), color = 'r', alpha = 0.5)
        if 'thy-rep' in data_idx:
            line_width = min(0.1, 10.0 / float(len(data_idx['thy-rep'])))
            if show_cluster != {}:
                count_clusters = [0 for _ in show_cluster['cluster']]
                for j in range(len(data_idx['thy-rep'])):
                    count_clusters[show_cluster['cluster'][j]] += 1
                    thy = np.array(data_idx['thy-rep'][j][eta_bins[i][0] - 1 : eta_bins[i][1]]) * \
                                   np.array(data_idx['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]])
                    if count_clusters[show_cluster['cluster'][j]] == 1:
                        label = r'$\overline{\chi _{%d}^2} = %.2f$' % (show_cluster['cluster'][j], show_cluster['chi2'][show_cluster['cluster'][j]])
                        plot_temp.plot(pt, thy, color = show_cluster['color'][show_cluster['cluster'][j]], \
                                       linewidth = line_width, zorder = 0, label = label)
                    else:
                        plot_temp.plot(pt, thy, color = show_cluster['color'][show_cluster['cluster'][j]], \
                                       linewidth = line_width, zorder = 0)
            else:
                for j in range(len(data_idx['thy-rep'])):
                    thy = np.array(data_idx['thy-rep'][j][eta_bins[i][0] - 1 : eta_bins[i][1]]) * np.array(data_idx['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]])
                    plot_temp.plot(pt, thy, color = 'gray', linewidth = line_width, zorder = 0)

    # plot_temp.legend(fontsize = 10, loc = 'best')
    legends = plot_temp.legend(fontsize = 10, loc = 'best')
    for line in legends.get_lines():
        line.set_linewidth(1.0)

    plot_temp.set_xlabel(r'$p_T~(\mathrm{GeV})$', size = 15)
    # plot_temp.xaxis.set_label_coords(1.05, 0.2)
    if data_idx['obs'][0].replace('<', '').replace('>', '') == 'd2_sigma_over_d_y_d_pt':
        if data_idx['units'][0] == 'nb':
            plot_temp.set_ylabel(r'$\frac{\mathrm{d}^2\sigma}{\mathrm{d}y \mathrm{d} p_T}~{\scriptstyle [\mathrm{nb} / (\mathrm{GeV} / c)]}$', size = 17)
        elif data_idx['units'][0] == 'pb':
            plot_temp.set_ylabel(r'$\frac{\mathrm{d}^2\sigma}{\mathrm{d}y \mathrm{d} p_T}~{\scriptstyle [\mathrm{pb} / (\mathrm{GeV} / c)]}$', size = 17)
    elif data_idx['obs'][0].replace('<', '').replace('>', '') == 'd2_sigma_over_2_pi_d_y_d_pt':
        if data_idx['units'][0] == 'nb':
            plot_temp.set_ylabel(r'$\frac{\mathrm{d}^2\sigma}{2 \pi \mathrm{d}y \mathrm{d} p_T}~{\scriptstyle [\mathrm{nb} / (\mathrm{GeV} / c)]}$', size = 17)
        elif data_idx['units'][0] == 'pb':
            plot_temp.set_ylabel(r'$\frac{\mathrm{d}^2\sigma}{2 \pi \mathrm{d}y \mathrm{d} p_T}~{\scriptstyle [\mathrm{pb} / (\mathrm{GeV} / c)]}$', size = 17)
    plot_temp.set_yscale('log')

    if data_idx['col'][0] == 'star':
        if idx == 10003:
            plot_temp.set_title(r'$\mathrm{STAR~MB}$')
            py.tight_layout()
            figure.savefig('%s/gallery/jet-%s-MB-%d.pdf' % (wdir, data_idx['col'][0], istep), dpi = dpi)
        elif idx == 10004:
            plot_temp.set_title(r'$\mathrm{STAR~HT}$')
            py.tight_layout()
            figure.savefig('%s/gallery/jet-%s-HT-%d.pdf' % (wdir, data_idx['col'][0], istep), dpi = dpi)
    else:
        plot_temp.set_title(r'$\mathrm{%s}$' % data_idx['col'][0].upper())
        py.tight_layout()
        figure.savefig('%s/gallery/jet-%s-%d.png' % (wdir, data_idx['col'][0], istep), dpi = dpi)

def plot_data_on_theory(wdir, data, show_cluster, istep, dpi):
    skip_single_point_bin = False
    count = count_eta_bins(data, skip_single_point_bin)
    if count == 13:
        n_rows, n_columns = 5, 3
        figure = py.figure(figsize = (n_columns * 4.5, n_rows * 3.0))
        axs = [py.subplot(n_rows, n_columns, i) for i in range(1, 12)]
        axs.extend([py.subplot(n_rows, n_columns, i) for i in range(14, 16)])
        i_last_row = [8, 11, 12]
        i_first_column = [0, 3, 6, 9, 11]
    elif count == 12:
        n_rows, n_columns = 4, 3
        figure = py.figure(figsize = (n_columns * 4.5, n_rows * 3.0))
        axs = [py.subplot(n_rows, n_columns, i) for i in range(1, 13)]
        i_last_row = [9, 10, 11]
        i_first_column = [0, 3, 6, 9]
    elif count == 11:
        n_rows, n_columns = 4, 3
        figure = py.figure(figsize = (n_columns * 4.5, n_rows * 3.0))
        axs = [py.subplot(n_rows, n_columns, i) for i in range(1, 12)]
        i_last_row = [8, 9, 10]
        i_first_column = [0, 3, 6, 9]

    count = 0

    for key in sorted(data.keys()):
        _, absolute_eta, eta_bins = find_eta_bins(data[key])
        data_idx = data[key]
        for i in range(len(eta_bins)):
            if absolute_eta:
                eta_min, eta_max = data[key]['eta-abs-min'][eta_bins[i][0] - 1], data[key]['eta-abs-max'][eta_bins[i][0] - 1]
            else:
                eta_min, eta_max = data[key]['eta-min'][eta_bins[i][0] - 1], data[key]['eta-max'][eta_bins[i][0] - 1]

            theory = data[key]['thy'][eta_bins[i][0] - 1 : eta_bins[i][1]]
            pt = data[key]['pT'][eta_bins[i][0] - 1 : eta_bins[i][1]]
            value = data[key]['value'][eta_bins[i][0] - 1 : eta_bins[i][1]]
            alpha = data[key]['alpha'][eta_bins[i][0] - 1 : eta_bins[i][1]]
            # theory_deviation = np.array(data[key]['dthy'][eta_bins[i][0] - 1 : eta_bins[i][1]]) * np.array(data[key]['plot-factor'][eta_bins[i][0] - 1 : eta_bins[i][1]])

            if absolute_eta:
                eta_range = r'$|\eta|~\in~[%s,~%s]$' % (str(eta_min), str(eta_max))
            else:
                eta_range = r'$\eta~\in~[%s,~%s]$' % (str(eta_min), str(eta_max))

            axs[count].errorbar(pt, value / theory, alpha / theory, color = 'firebrick', marker = '.', \
                                markersize = 3.0, capsize = 2.0, linestyle = 'none', zorder = 10)
            if 'thy-rep' in data[key]:
                line_width = min(0.1, 10.0 / float(len(data[key]['thy-rep'])))
                if show_cluster != {}:
                    count_clusters = [0 for _ in show_cluster['cluster']]
                    for j in range(len(data_idx['thy-rep'])):
                        count_clusters[show_cluster['cluster'][j]] += 1
                        thy = np.array(data_idx['thy-rep'][j][eta_bins[i][0] - 1 : eta_bins[i][1]])
                        if count_clusters[show_cluster['cluster'][j]] == 1:
                            label = r'$\overline{\chi _{%d}^2} = %.2f$' % (show_cluster['cluster'][j], show_cluster['chi2'][show_cluster['cluster'][j]])
                            axs[count].plot(pt, thy / theory, color = show_cluster['color'][show_cluster['cluster'][j]], \
                                            linewidth = line_width, zorder = 0, label = label)
                        else:
                            axs[count].plot(pt, thy / theory, color = show_cluster['color'][show_cluster['cluster'][j]], \
                                           linewidth = line_width, zorder = 0)
                else:
                    for j in range(len(data[key]['thy-rep'])):
                        thy = data[key]['thy-rep'][j][eta_bins[i][0] - 1 : eta_bins[i][1]]
                        axs[count].plot(pt, thy / theory, color = 'gray', linewidth = line_width, zorder = 0)
            # axs[count].fill_between(pt, value / (theory - (0.5 * theory_deviation)), value / (theory + (0.5 * theory_deviation)), color = 'r', alpha = 0.5)
            axs[count].text(0.1, 0.15, eta_range, color = 'k', transform = axs[count].transAxes, size = 23)
            axs[count].text(0.1, 0.85, r'$\mathrm{%s}$' % data[key]['col'][0].upper(), color = 'k', transform = axs[count].transAxes, size = 23)
            count += 1

    legends = axs[1].legend(fontsize = 10, loc = 'best')
    for line in legends.get_lines():
        line.set_linewidth(1.0)

    # minorLocator = MultipleLocator(0.05)
    # majorLocator = MultipleLocator(0.2)
    for i in range(len(axs)):
        axs[i].axhline(1.0, color = 'black', ls = 'dashdot', linewidth = 1.0)
        axs[i].set_ylim(0.8, 1.2)
        # axs[i].yaxis.set_minor_locator(minorLocator)
        # axs[i].yaxis.set_major_locator(majorLocator)
        axs[i].tick_params(axis = 'both', which = 'both', right = True, top = True, direction = 'in', labelsize = 20)

    for i in range(len(axs)):
        if i in i_first_column:
            axs[i].set_ylabel(r'\boldmath$\mathrm{data/theory}$', size = 24)
        if i in i_last_row:
            axs[i].set_xlabel(r'\boldmath$p_{T}~(\mathrm{GeV})$', size = 24)
            # axs[i].xaxis.set_label_coords(0.80, -0.12)

    py.tight_layout()
    py.savefig('%s/gallery/jet-ratio-%d.png' % (wdir, istep), dpi = dpi)
    py.close()

def plot_obs(wdir, kc, task=1, plot_with_factor = True, only_best_cluster = True, replica_lines = False, dpi = 200):

    if task == 1:
        print('\nplotting JET data from %s' % (wdir))
    elif task == 2:
        print('\nplotting JET data over theory from %s' % (wdir))

    load_config('%s/input.py' % wdir)
    istep = core.get_istep()
    replicas = core.get_replicas(wdir)
    core.mod_conf(istep, replicas[0]) #--set conf as specified in istep

    predictions = load('%s/data/predictions-%d.dat' % (wdir, istep))
    if ('jet' not in predictions['reactions']) or (len(predictions['reactions']['jet']) == 0):
        print('JET is not in data file')
        return
    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)
    best_cluster = cluster[0]
    show_cluster = {}

    data = predictions['reactions']['jet']

    for idx in data:
        predictions = copy.copy(data[idx]['prediction-rep'])
        del data[idx]['prediction-rep']
        del data[idx]['residuals-rep']
        del data[idx]['shift-rep']
        if only_best_cluster:
            best_predictions = [predictions[i] for i in range(len(predictions)) if cluster[i] == best_cluster]
            data[idx]['thy'] = np.mean(best_predictions, axis = 0)
            data[idx]['dthy'] = np.std(best_predictions, axis = 0)
            if replica_lines: data[idx]['thy-rep'] = best_predictions
        else:
            all_predictions = [predictions[i] for i in range(len(predictions))]
            data[idx]['thy'] = np.mean(all_predictions, axis = 0)
            data[idx]['dthy'] = np.std(all_predictions, axis = 0)
            if replica_lines:
                data[idx]['thy-rep'] = all_predictions
                cluster_colors = labels['cluster_colors']
                show_cluster = {'cluster': cluster, 'color': cluster_colors, 'chi2': cluster_average_chi2}

    if task == 1:
        plot_data(wdir, data, istep, dpi, plot_with_factor)
        #for idx in data:
        #    plot_data_separate(wdir, data[idx], show_cluster, istep, idx, dpi, plot_with_factor)
    elif task == 2:
        plot_data_on_theory(wdir, data, show_cluster, istep, dpi)

    return

if __name__ == "__main__":
    pass






