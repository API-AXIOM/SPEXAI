import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import  gridspec
import matplotlib.colors as colors
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.colors import ListedColormap


font = {'size'   : 16}
matplotlib.rc('font', **font)

def loss(train, test, yscale='log', lossfn='MSE'):
    '''
    Plot the loss of training
    Parameters
    ----------
    train, test: list, list
        loss values per epoch
    yscale: str default 'log'
        scale y-axis
    lossfn: str default 'MSE'
        name loss function    
    '''
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(15,5))
    plt.rcParams['text.usetex']=True

    line0, = plt.plot(test,'b')
    line1, = plt.plot(train, 'r', alpha=0.8)
    plt.legend((line0, line1), ('Loss validation', 'Loss training'), loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel(lossfn+' loss')
    # plt.title('Average loss per epoch')
    plt.yscale(yscale)
    plt.show()


def fraction(list_labels, labels, pred, test, mask, x, crit_frac=-3,loss_fn=nn.MSELoss(),
             name_x='Energy', name_label='Temperature', yscale=['log','log']):
    '''
    Plots the predicated data against the original data and show the fraction
    between them in a subplot
    Parameters
    ----------
    list_labels: list
        labels of the plotted data.
    labels: array or Tensor
        labels of the data
    pred: array or Tensor
        predicted data
    test: array or Tensor
        origanal validation data
    mask: array or Tensor
        array of boolian where data meets criteria
    x: array or Tensor
        x-axis
    crit_frac: float
        log_10 of critical fraction
    loss_fn
        pytorch lossfunction
    name_x, name_label: str
        name of x-as and label
    yscale: list
        list of str to indicate y-axis
    '''
    plt.style.use('seaborn-v0_8-whitegrid')
    
    #make sure there all arrays
    if torch.is_tensor(labels) == True:
        labels = labels.cpu().detach().numpy()
    if torch.is_tensor(pred) == True:
        pred = pred.cpu().detach().numpy()
    if torch.is_tensor(test) == True:
        test = test.cpu().detach().numpy()
    if torch.is_tensor(mask) == True:
        mask = mask.cpu().detach().numpy()
    if torch.is_tensor(x) == True:
        x = x.cpu().detach().numpy()

    for i in list_labels:
        #find label closest to i
        j = np.argmin(abs(labels-i))
        
        #calculete indivdiual loss
        indv_loss = loss_fn(torch.from_numpy(pred[j][mask[j]]),
                            torch.from_numpy(test[j][mask[j]])).item()
        #calculate fraction between orignal spectrum and predicted spectrum
        frac = abs(pred[j][mask[j]]-test[j][mask[j]])/test[j][mask[j]]
        
        #plot
        fig = plt.figure(figsize=(10,5))
        plt.rcParams['text.usetex']=True

        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
        #subplot spectrums
        ax0 = plt.subplot(gs[0])
        line0, = ax0.plot(x[mask[j]], test[j][mask[j]], linewidth=2)
        line1, = ax0.plot(x[mask[j]], pred[j][mask[j]], linewidth=.8)

        #subplot fraction
        ax1 = plt.subplot(gs[1], sharex = ax0)
        ax1.hlines(1e-3, min(x[mask[j]]), max(x[mask[j]]), colors='black', linestyles='dashed')
        line2 = ax1.scatter(x[mask[j]][frac > 10**crit_frac], frac[frac > 10**crit_frac], 
                            s=10, color='r')
        line3 = ax1.scatter(x[mask[j]][frac <= 10**crit_frac], frac[frac <= 10**crit_frac],
                             s=10, color='b')
        
        #remove gap between subplots
        plt.setp(ax0.get_xticklabels(), visible=False)
        yticks = ax1.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)
        # plt.subplots_adjust(hspace=.0)
        #scale
        ax0.set_yscale(yscale[0])
        ax1.set_yscale(yscale[1])
        #legend & labels
        ax0.legend((line0, line1), ('SPEX', 'NN pred.'), loc='upper right')
        ax0.set_ylabel(r'Flux $\mathrm{[Arb.]}$')
        ax0.set_xlim(min(x[mask[j]]), max(x[mask[j]]))
        ax1.set_ylabel(r'Fraction $\mathrm{[|F_{NN}-F_{SPEX}|/F_{spex}]}$')
        ax1.set_xlabel(name_x+r' $\mathrm{[KeV]}$')
        if yscale[1] == 'log':
            ax1.set_ylim(10**(crit_frac-3),10**(crit_frac+3))
            ax1.set_yticks([10**(crit_frac-3), 10**(crit_frac-2), 10**(crit_frac-1), 10**(crit_frac),10**(crit_frac+1), 10**(crit_frac+2),10**(crit_frac+3)])
        ax1.set_xlim(min(x[mask[j]]), max(x[mask[j]]))



        print('{} is {:.3f} KeV with MSE-loss of {:.3e}'.format(name_label, labels[j], 
                                                                   indv_loss))
        plt.tick_params(which='both', bottom=True, direction='in', top=True, right=True, left=True)
        plt.setp(ax0.get_xticklabels(), visible=False)
        fig.tight_layout()
        plt.show()
        
        print('{} of the maximum is {:.4f} [KeV]'
        .format(name_x, x[mask[j]][np.argmax(test[j][mask[j]])]))

def ind_loss(pred, test, mask, x, crit_frac=-3, loss_fn=nn.MSELoss(), xlabel=r'Temperature $\mathrm{[KeV]}$',
              ylabel=['MSE-loss', 'Max fraction'], yscale=['log','log']):
    '''
    Parameters
    ----------
    pred: array or Tensor
        predicted data
    test: array or Tensor
        origanal validation data
    mask: array or Tensor
        array of boolian where data meets criteria
    x: array or Tensor
        x-axis
    crit_frac: float
        log_10 of critical fraction
    loss_fn
        pytorch lossfunction
    '''
    plt.style.use('seaborn-v0_8-whitegrid')
    
    #make sure there all arrays
    if torch.is_tensor(x) == True:
        x = x.cpu().detach().numpy()
    if torch.is_tensor(pred) == True:
        pred = pred.cpu().detach().numpy()
    if torch.is_tensor(test) == True:
        test = test.cpu().detach().numpy()
    if torch.is_tensor(mask) == True:
        mask = mask.cpu().detach().numpy()

    #calculate individual loss on each row
    indv_loss = np.array([loss_fn(torch.from_numpy(pred[i][mask[i]]), 
                                  torch.from_numpy(test[i][mask[i]])).item() 
                                  for i in range(len(x))])
    #calculate fraction 
    fraction = abs(pred-test)/test
    fraction = np.where(mask == True, fraction, 0)
    
    fig = plt.figure(figsize=(10,5))
    plt.rcParams['text.usetex']=True

    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1]) 
    ax0 = plt.subplot(gs[0])
    ax0.plot(x, indv_loss)
    ax0.set_yscale(yscale[0])
    ax0.set_ylabel(ylabel[0])
    
    ax1 = plt.subplot(gs[1], sharex = ax0)
    ax1.hlines(10**crit_frac, min(x), max(x), colors='black', linestyles='dashed')
    #calculate if maximum fraction is above critical fraction
    for i, values in enumerate(fraction):
        maximum = np.max(values)
        if len(values[values > 10**crit_frac]) >= 3:
            ax1.scatter(x[i], maximum, s=10, color='darkred')
        elif len(values[values > 10**crit_frac]) >= 1 and len(values[values > 10**crit_frac])< 3:
            ax1.scatter(x[i], maximum, s=10, color='r')
        else:
            ax1.scatter(x[i], maximum, s=10, color='b')
    
    
    ax1.set_yscale(yscale[1])
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel[1])
    # remove vertical gap between subplots
    plt.tick_params(which='both', bottom=True, direction='in', top=True, right=True, left=True)
    plt.setp(ax0.get_xticklabels(), visible=False)
    fig.tight_layout()
    plt.show()
    
def heatmap(X, Y, Z, mask, Z_center=-3., color='seismic', xlabel=r'Energy $\mathrm{[KeV]}$',
            ylabel=r'Temperature $\mathrm{[KeV]}$', zlabel=r'Fraction $\mathrm{[|F_{NN}-F_{SPEX}|/F_{spex}]}$', 
            title=None):
    '''
    Parameters
    ----------
    X: array or Tensor
        list of grid points in x-axis
    Y: array or Tensor
        list of grid points in y-axis
    Z: array or Tensor
        z-values on each gridpoint
    mask: array or Tensor
        Boolians where False will stay black
    Z_center = float default -3
        center of colorbar in logspace
    '''
    plt.style.use('seaborn-ticks')

    #make sure there all arrays
    if torch.is_tensor(X) == True:
        X = X.cpu().detach().numpy()
    if torch.is_tensor(Y) == True:
        Y = Y.cpu().detach().numpy()
    if torch.is_tensor(Z) == True:
        Z = Z.cpu().detach().numpy()
    if torch.is_tensor(mask) == True:
        mask = mask.cpu().detach().numpy()

    #colormap
    top = cm.get_cmap('autumn', 128)
    bottom = cm.get_cmap('winter', 224)

    newcolors = np.vstack((bottom(np.linspace(0, 1/1.75, 128)),
                        top(np.linspace(0, 1, 128))))
    newcmp = ListedColormap(newcolors, name='winter_autumn')

    X_axis = np.linspace(min(X), max(X), 10)

    fig, ax = plt.subplots(figsize=(10,5))
    plt.rcParams['text.usetex']=True

    x,y = np.meshgrid(X, Y)
    ax.set_facecolor('black')
    z = abs(np.ma.masked_where(mask == False, Z))
    norm = colors.LogNorm(vmin = 10**(Z_center-3), vmax = 10**(Z_center+3))
    plt.pcolormesh(x, y, z, cmap=newcmp, norm=norm)
    ticks = [10**(Z_center-3), 10**(Z_center-2), 10**(Z_center-1), 10**(Z_center),
                10**(Z_center+1),10**(Z_center+2),10**(Z_center+3)]
    cbar = plt.colorbar(ticks=ticks, format='%.0e', norm=norm)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    cbar.set_label(zlabel, rotation=270, labelpad=15)
    plt.tick_params(which='both', bottom=True, direction='out', top=True, right=True, left=True)
    fig.tight_layout()
    plt.show()


    if title is not None:
        fig.suptitle(title)    
    plt.show()

    #======================print_Npoint above fraction===================================
    z_masked = Z[mask].flatten()
    N_total = len(z_masked)
    N_limit = len(z_masked[z_masked >= 10**Z_center])
    N_limitH = len(z_masked[z_masked >= 10**(Z_center+1)])
    print('N_points above f_(10^({})) = {}; This is {:.03g}% of N_total'.format(Z_center, N_limit, (N_limit/N_total)*100))
    print('N_points above f_(10^({})) = {}; This is {:.03g}% of N_total'.format(Z_center+1, N_limitH, (N_limitH/N_total)*100))

def plot_errors(X, Z, mask, Z_center=-3., color='seismic', xlabel='Energy [KeV]'):
    plt.style.use('seaborn-v0_8-whitegrid')

    #make sure there all arrays
    if torch.is_tensor(X) == True:
        X = X.cpu().detach().numpy()
    if torch.is_tensor(Z) == True:
        Z = Z.cpu().detach().numpy()
    if torch.is_tensor(mask) == True:
        mask = mask.cpu().detach().numpy()

    #=============================plot means+stdev========================================
    Znan = np.where(mask, Z, np.nan)
    mean_x = np.nanmean(Znan, axis=0)

def plot_errors(X, Z, mask, Z_center=-3., color='seismic', xlabel=r'Energy $\mathrm{[KeV]}$'):
    plt.style.use('seaborn-v0_8-whitegrid')

    #make sure there all arrays
    if torch.is_tensor(X) == True:
        X = X.cpu().detach().numpy()
    if torch.is_tensor(Z) == True:
        Z = Z.cpu().detach().numpy()
    if torch.is_tensor(mask) == True:
        mask = mask.cpu().detach().numpy()

    #=============================plot means+stdev========================================
    Znan = np.where(mask, Z, np.nan)
    mean_x = np.nanmean(Znan, axis=0)
    p68 = np.nanpercentile(abs(Znan), 68,axis=0)
    p95 = np.nanpercentile(abs(Znan), 95,axis=0)
    p997 = np.nanpercentile(abs(Znan), 99.7,axis=0)


    alpha = .1
    #plot 1-sigma interval
    fig = plt.subplots(figsize=(10,5))
    plt.rcParams['text.usetex']=True

    # Change the color of the line between +/- Z_center
    plt.fill_between(X, 0, p997 , color='red', alpha=.1, label='99.7%')
    plt.fill_between(X, 0, p95 , color='red', alpha=0.4, label='95%')
    plt.fill_between(X, 0, p68 , color='red', alpha=.8, label='68%')
    
    plt.scatter(X, abs(mean_x), color='darkred', s=4, label="mean")

    #vlines
    plt.hlines(10**(Z_center), min(X), max(X), colors='black', linestyles='dashed')

    #set axis
    plt.yscale('log')
    plt.ylim(10**(Z_center-3),10**(Z_center+3))
    plt.yticks([10**(Z_center-3), 10**(Z_center-2), 10**(Z_center-1), 10**(Z_center),10**(Z_center+1), 10**(Z_center+2),10**(Z_center+3)])
    plt.xlim(min(X), max(X))
    plt.tick_params(which='both', bottom=True, direction='inout', top=True, right=True, left=True)


    plt.ylabel('Error $\mathrm{[|F_{NN}-F_{SPEX}|/F_{spex}]}$')
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()

def show_data(list_labels,  labels, data, mask, x, name_x='Energy', name_label='Temperature',
              title='', yscale='log', xlimit=None, ylimit=None):
    '''
    Parameters
    ----------
    list_labels: list
        labels of the plotted data.
    labels: array or Tensor
        labels of the data
    data: array or Tensor

    mask: array or Tensor
        array of boolian where data meets criteria
    x: array or Tensor
        x-axis
    xlimit: list default None
        list of min-max value of x-axis
    ylimint: list default None
        list of min-max value of y-axis
    '''
    plt.style.use('seaborn-v0_8-whitegrid')

    #make sure there all arrays
    if torch.is_tensor(labels) == True:
        labels = labels.cpu().detach().numpy()
    if torch.is_tensor(data) == True:
        data = data.cpu().detach().numpy()
    if torch.is_tensor(mask) == True:
        mask = mask.cpu().detach().numpy()
    if torch.is_tensor(x) == True:
        x = x.cpu().detach().numpy()

    fig = plt.figure(figsize=(10,5))
    plt.rcParams['text.usetex']=True

    for i in list_labels:
        #find energy bin closest to i
        j = np.argmin(abs(labels-i))
        plt.plot(x[mask[j]], data[j][mask[j]], alpha=0.8, label=f'{name_label} is {labels[j]:.2f} KeV')
    plt.yscale(yscale)
    if xlimit is not None:
        plt.xlim(xlimit[0], xlimit[1])
    if ylimit is not None:
        plt.ylim(ylimit[0], ylimit[1])
    plt.ylabel(r'Flux $\mathrm{[Arb.]}$')
    plt.xlabel(name_x+r' $\mathrm{[KeV]}$')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()
