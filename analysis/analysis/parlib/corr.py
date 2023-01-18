import sys,os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import pylab as py


#--from tools
from tools.tools     import checkdir,save,load
import tools.config
from tools.config    import load_config, conf, options
from tools.inputmod  import INPUTMOD

#--from local
from analysis.corelib import core
from analysis.corelib import classifier

#--plot NxN correlation matrix for N parameters

def heatmap(data, row_labels, col_labels, ax=None, reformat = False, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",size=30)
    cbar.ax.tick_params(labelsize=30,size=10)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    if reformat: size = 20
    else:        size = 10
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False, labelsize = size)

    # Rotate the tick labels and set their alignment.
    py.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",textcolors=["black", "white"],threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    #for i in range(data.shape[0]):
    #    for j in range(data.shape[1]):
    #        kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
    #        text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
    #        texts.append(text)

    return texts
   
def func(x, pos):
    return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "")

def formatter(names,samples,ax,norms=False):
    NEW = []

    #--reorganize names
    flavs = ['g1','uv1','dv1','s1','ub1','db1','sb1','sea1']
    for i in range(len(flavs)):
        for j in range(len(names)):
            flav = flavs[i]
            name = names[j]
            if name.split()[1] == flav: NEW.append(name)
         
    for dist in ['ht4','off']:
        for _ in ['F2p','F2n']:
            for j in range(len(names)):
                name = names[j]
                if name.split()[0] == dist and name.split()[1] == _ : NEW.append(name)

    if norms:
        for exp in ['idis','dy']:
            for name in names:
                if name.split()[0] == exp: NEW.append(name)
  
    #--get parameters to match names
    SAMP = []
    samples = np.array(samples)
    for i in range(len(NEW)):
        new = NEW[i]
        for j in range(len(names)):
            name = names[j]
            if name == new: SAMP.append(samples.T[j])

    SAMP = np.array(SAMP).T

    #--demark separate distributions
    demark(ax,NEW)

    #--add labels
    labeler(ax,NEW)
    #--reformat names
    reform = []
    for new in NEW:
        ref = '{}'.format(new).replace('pdf ','')\
                              .replace('g1 '  , '_')\
                              .replace('uv1 ' , '_')\
                              .replace('dv1 ' , '_')\
                              .replace('s1 '  , '_')\
                              .replace('ub1 ' , '_')\
                              .replace('db1 ' , '_')\
                              .replace('sb1 ' , '_')\
                              .replace('sea1 ', '_')\
                              .replace('mix ' , '_')\
                              .replace('ht4 F2p N' , '$b_0$')\
                              .replace('ht4 F2p a' , '$b_1$')\
                              .replace('ht4 F2p d' , '$b_3$')\
                              .replace('ht4 F2n N' , '$b_0$')\
                              .replace('ht4 F2n a' , '$b_1$')\
                              .replace('ht4 F2n d' , '$b_3$')\
                              .replace('off F2p N' , '$c_0$')\
                              .replace('off F2p x0', '$c_1$')\
                              .replace('_a'  , r'$a_1$')\
                              .replace('_N'  , r'$a_0$')\
                              .replace('_b'  , r'$a_2$')\
                              .replace('_c'  , r'$a_3$')\
                              .replace('_d'  , r'$a_4$')\
                              .replace('_x0' , r'$x_0$')

        reform.append(ref)
 

    return reform, SAMP

def demark(ax,names):
    dists = []
    for name in names:
        if len(name.split()) == 4:
            dists.append(name.split()[0] + ' ' + name.split()[1] + ' ' + name.split()[2])
        if len(name.split()) == 3:
            dists.append(name.split()[0] + ' ' + name.split()[1])
        if len(name.split()) == 2:
            dists.append(name.split()[0])

    for i in range(len(dists)):
        if i==0: dist = dists[i] 
        elif dists[i] != dist:
            ax.axhline(i-0.5,0,1,color='black')
            ax.axvline(i-0.5,0,1,color='black')
            dist = dists[i]

    ax.axhline(-0.5,0,1,color='black') 
    ax.axhline(len(names)-0.5,0,1,color='black') 
    ax.axvline(-0.5,0,1,color='black') 
    ax.axvline(len(names)-0.5,0,1,color='black')

def labeler(ax,names):
    
    dists = []
    N = float(len(names))
    for name in names:
        if len(name.split()) == 3:
            dists.append(name.split()[0] + ' ' + name.split()[1])
        if len(name.split()) == 2:
            dists.append(name.split()[0])

    #--get lengths of distributions
    n = []
    l = 1
    for i in range(len(names)):
        if i==0: 
            dist = dists[i]
        elif dists[i] != dist:
            dist = dists[i]
            n.append(l)
            l = 1
        elif i==(N-1): n.append(l+1)
        else: l += 1

    
    labels = [r'$g$',r'$u_v$',r'$d_v$',r'$s$',r'$\bar{u}$',r'$\bar{d}$',r'$\bar{s}$',r'$S$',r'$H^p$',r'$H^n$',r'$\delta f_0$']
    m = 0
    for i in range(len(n)):
        label = labels[i]
        x = 0.184 + m/N/1.540
        y = 0.853 - m/N/1.155
        width = 0.315*n[i]
        #--label top axis
        ax.annotate(label ,xy=(x,0.94),xytext=(x,0.95),xycoords='figure fraction',fontsize=30,\
                    ha='center',va='bottom',arrowprops=dict(arrowstyle='-[, widthB=%s, lengthB=0.2'%width, lw=1.0))
        #--label left axis
        ax.annotate(label ,xy=(0.12,y),xytext=(0.08,y),xycoords='figure fraction',fontsize=30,\
                    ha='left',va='center',arrowprops=dict(arrowstyle='-[, widthB=%s, lengthB=0.2'%width, lw=1.0))
        
        if i < (len(n)-1): m += (n[i] + n[i+1])/2.0

def plot_corr(wdir,kc,norms=False,reformat=False, title=None):
    # We can nicely plot a correlation matrix. Since this is bound by -1 and 1,
    # we use those as vmin and vmax. We may also remove leading zeros and hide
    # the diagonal elements (which are all 1) by using a
    # :class:`matplotlib.ticker.FuncFormatter`.

    fig, ax = py.subplots(1,1,figsize=(16,12))

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas = core.get_replicas(wdir) 
    core.mod_conf(istep,replicas[0])

    clusters,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)
    best_cluster = cluster_order[0]

    #--get data
    step=conf['steps'][istep]
    samples=[]
    for i in range(len(replicas)):

        if clusters[i]!=best_cluster: continue

        replica=replicas[i]
        order=replica['order'][istep]
        params = []
        for j in range(len(order)):
            if norms == False:
                if order[j][0] == 2: continue
            params.append(replica['params'][istep][j])
        #--sort alphabetically
        z = sorted(zip(order,params))
        order  = [z[k][0] for k in range(len(z))]
        params = [z[k][1] for k in range(len(z))]
        samples.append(params)

    #--get names of parameters
    if norms: _names = [(order[i][1]+ ' ' + str(order[i][2])) for i in range(len(order))]
    else:     _names = [(order[i][1]+ ' ' + str(order[i][2])) for i in range(len(order)) if order[i][0]!=2]
   
    #--reformat names by hand (needs to be done on case by case basis)
    if reformat: names, samples = formatter(_names,samples,ax,norms)
    else:        
        names = [r'\textrm{%s}'%_names[i] for i in range(len(_names))]
        demark(ax,_names)

    samples=np.array(samples)
    pmin =[np.amin(p) for p in samples.T]
    pmax =[np.amax(p) for p in samples.T]
    mean =[np.mean(p) for p in samples.T]
    std  =[np.std(p)  for p in samples.T]
    corr = np.corrcoef(samples.T)


    cbarlabel = r'\textbf{\textrm{correlation coeff.}}'
    im, _ = heatmap(corr, names, names, ax=ax, reformat = reformat, cmap="PuOr", vmin=-1, vmax=1, cbarlabel=cbarlabel)
    
    annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=7)
 
    if title!=None: ax.set_ylabel(title,size=50,labelpad=60.0) 

    py.tight_layout()

    if reformat: py.subplots_adjust(top=0.90,left=0.10)

    filename = '%s/gallery/params-corr.png'%wdir
    py.savefig(filename)
    print('Saving correlation coefficient heatmap to %s'%filename)

#--plot 2D scatter plot for parameters from two distributions

def format_scatter(order):

    #--reformat names
    reform = []
    for name in order:
        ref = '{}'.format(name).replace('g1 '  , r'$g$ _')\
                               .replace('uv1 ' , r'$u_v$ _')\
                               .replace('dv1 ' , r'$d_v$ _')\
                               .replace('s1 '  , r'$s$ _')\
                               .replace('ub1 ' , r'$\bar{u}$ _')\
                               .replace('db1 ' , r'$\bar{d}$ _')\
                               .replace('sb1 ' , r'$\bar{s}$ _')\
                               .replace('sea1 ', r'$S$ _')\
                               .replace('mix ' , r'$mix$ _')\
                               .replace('_a'  , r'$a_1$')\
                               .replace('_N'  , r'$a_0$')\
                               .replace('_b'  , r'$a_2$')\
                               .replace('_c'  , r'$a_3$')\
                               .replace('_d'  , r'$a_4$')\
                               .replace('_x0' , r'$x_0$')

        reform.append(ref)
 

    return reform

def plot_scatter(wdir,kc,dist1,dist2,reformat=False):

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas = core.get_replicas(wdir) 
    core.mod_conf(istep,replicas[0])

    clusters,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)
    best_cluster = cluster_order[0]

    #--get appropriate order
    step=conf['steps'][istep]
    order1  = []
    order2  = []

    for i in range(len(replicas)):

        if clusters[i]!=best_cluster: continue

        replica=replicas[i]
        order=replica['order'][istep]
        for j in range(len(order)):
            if order[j][0] == 2: continue
            dist = order[j][2].split()[0]
            if dist == dist1: 
                if replica['order'][istep][j] not in order1: order1 .append(replica['order'][istep][j])
            if dist == dist2:
                if replica['order'][istep][j] not in order2: order2 .append(replica['order'][istep][j])


    order1  = sorted([order1[i][2] for i in range(len(order1))])
    order2  = sorted([order2[i][2] for i in range(len(order2))])

    #--get parameters corresponding to order
    params1 = np.zeros((len(order1),len(replicas)))
    params2 = np.zeros((len(order1),len(replicas)))

    for i in range(len(replicas)):

        if clusters[i]!=best_cluster: continue

        replica=replicas[i]
        order=replica['order'][istep]
        for j in range(len(order)):
            if order[j][0] == 2: continue
            dist = order[j][2].split()[0]
            par = order[j][2]
            if dist == dist1:
                idx = order1.index(par)
                params1[idx][i] = replica['params'][istep][j]
            if dist == dist2:
                idx = order2.index(par)
                params2[idx][i] = replica['params'][istep][j]

    #--create plot with enough space for # of parameters
    nrows,ncols = len(order1),len(order2)
    fig = py.figure(figsize=(ncols*7,nrows*4))

    if reformat:
        order1 = format_scatter(order1)
        order2 = format_scatter(order2)

    #--plot all parameter combinations
    idx = 1
    for i in range(len(order1)):
        for j in range(len(order2)):
            ax = py.subplot(nrows,ncols,idx)
            idx += 1
            ax.scatter(params1[i],params2[j])
            ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=20)
            ax.set_title(r'%s \textrm{vs.} %s'%(order1[i],order2[j]), size=30)
            corr = np.corrcoef(params1[i],params2[j])[0][1]
            ax.text(0.85, 0.85, '%3.2f'%corr, transform=ax.transAxes,size=30)
            #ax.legend(loc='best',frameon=False,fontsize=15)

    py.tight_layout()

    #py.subplots_adjust(top=0.90,left=0.10)

    filename = '%s/gallery/params-scatter-%s-%s.png'%(wdir,dist1,dist2)
    py.savefig(filename)
    print('Saving correlation coefficient heatmap to %s'%filename)





#--plot correlation as a function of x


