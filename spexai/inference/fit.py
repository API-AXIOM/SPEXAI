# DEPRECATED: legacy CNN inference stack. Use the operator stack instead
# (spexai.inference.fitting for emcee/ultranest fits over the operator model).
import warnings as _warnings
_warnings.warn(
    "spexai.inference.fit is deprecated (legacy CNN stack); use "
    "spexai.inference.fitting instead.",
    DeprecationWarning, stacklevel=2)

import numpy as np
import pandas as pd
import torch
import emcee
#from scipy.stats import poisson
from torch.distributions.poisson import Poisson

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
font = {'size'   : 16}
matplotlib.rc('font', **font)
plt.style.use('seaborn-v0_8-whitegrid')

from spexai.inference import write_tensors

class SingleTemp(object):
    def __init__(self, counts, model, prior,
                 e_min=None, e_max=None, fdir_nn='neuralnetworks/'):
        '''
        Class to be able fit a spectrum with a temperature distribution in the form of a gaussion
        Parameters
        ----------
        counts : numpy.ndarray
            The counts in the spectrum to fit. The energy grid needs to match the 
            grid in the RMF loaded in `model`

        model : spexai.CombinedModel object
            An object defining the emulator model

        prior: dir
            Intial guess/prior of the variables in the form of {'var_name': {'mu':float, 'sigma':float}}

        fdir_NN: str default:'neuralnetworks/'
            directory of torch object to restucture spectra and best nn

        e_min, e_max: float default:None
            minimum or maximum energy

        '''

        self.counts = counts

        self.e_min = e_min
        self.e_max = e_max

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using', self.device)
        self.combined_model = model
        self.combined_model.to(self.device)

        #cut off spectra outside of e_min and e_max
        if self.e_min is None:
            self.e_min= min(model.chan_cent)
        if self.e_max is None:
            self.e_max = max(model.chan_cent)

        intv = np.where(model.chan_cent < self.e_min, False, True)
        self.intv = np.where(model.chan_cent > self.e_max, False, intv)

        self.counts = counts[self.intv]
        self.chan_diff = model.chan_diff[self.intv]
        self.intensity = self.counts/self.chan_diff/model.exp_time
        self.energy = model.chan_cent[self.intv]

        #prior
        self.prior = prior

        self.bounds = {}

        #bounds
        self.bounds['temp'] = [0.2, 10]
        self.bounds['logz'] = [-10, 1]
        self.bounds['norm'] = [1e2, 1e15]
        self.bounds['vel'] = [0, 600]

        self.param_names = list(self.prior.keys())


#    def load_data(self, filepath):
#        '''reads in the fits file of the data'''
#        counts, channels, exp_time  = write_tensors.read_data(filepath)
#        chan_diff = self.combined_model.chan_diff.numpy()
#        chan_cent = self.combined_model.chan_cent.numpy()
#        self.exp_time = exp_time
#
#        #cut off spectra outside of e_min and e_max
#        if self.e_min is None:
#            self.e_min= min(chan_cent)
#        if self.e_max is None:
#            self.e_max = max(chan_cent)
#
#        intv = np.where(chan_cent < self.e_min, False, True)
#        self.intv = np.where(chan_cent > self.e_max, False, intv)
#
#        self.counts = (counts[self.intv]).astype(int)
#        self.intensity = counts[self.intv]/chan_diff[self.intv]/self.exp_time
#        self.energy = chan_cent[self.intv]
#        self.chan_diff = chan_diff[self.intv]
#
#    def sim_data(self, params, exp_time=10000):
#        '''creates simulated data from the neural network model'''
#        dict_abund = {}
#        for i in torch.arange(6,31):
#            dict_abund[f'Z{i}'] = params['met']
#            for key in params.keys():
#                if key == f'Z{i}':
#                    dict_abund[f'Z{i}'] = params['met']*params[f'Z{i}']
#
#        temp = torch.tensor([params['temp']], dtype=torch.float32, device=self.device)
#
#        self.combined_model.to(self.device)
#        with torch.no_grad():
#            spectra = self.combined_model(temp, dict_abund, params['logz'], 
#                                          params['norm'], params['vel']).cpu().detach().numpy()
#        
#        chan_diff = self.combined_model.chan_diff.numpy()
#        chan_cent = self.combined_model.chan_cent.numpy()
#        self.exp_time = exp_time
#
#        #cut off spectra outside of e_min and e_max
#        if self.e_min is None:
#            self.e_min= min(chan_cent)
#        if self.e_max is None:
#            self.e_max = max(chan_cent)
#
#        intv = np.where(chan_cent < self.e_min, False, True)
#        self.intv = np.where(chan_cent > self.e_max, False, intv)
#        self.counts = np.random.poisson(spectra*chan_diff*exp_time)[self.intv]
#
#        self.intensity = self.counts/(chan_diff[self.intv]*exp_time)
#        self.energy = chan_cent[self.intv]
#        self.chan_diff = chan_diff[self.intv]
#

    def add_prior(self, priors_to_add):
        """
        Add additional parameters to be estimated, and their priors. 
        Updates the dictionary in `self.prior`

        Parameters
        ----------
        priors_to_add : dict
            A dictionary containing additional parameter names and 
            the `mu` and `sigma` values for their gaussian priors

        """ 
        for i in priors_to_add.keys():
            self.param_names.append(i)

        self.prior.update(priors_to_add)

    def fit_spectrum(self, nwalkers, nsteps):
        ''' 
        Fit the spectrum with emcee method and prints the autocorrolation time

        Paramaters
        ----------
        nwalkers: int
            number walkers for the EMCEE fit
        nsteps: int
            number of steps

        '''

        self.nwalkers = nwalkers
        self.nsteps = nsteps

        self.sampler = emcee.EnsembleSampler(
            nwalkers = self.nwalkers,
            ndim = len(self.param_names),
            log_prob_fn = self.log_prob,
            kwargs = {
                'param_names': self.param_names,
            }
            #    'data': self.counts,
            #    'model': self.combined_model.to(self.device)
            #    }
            )
        
        initialpos = None
        for i in self.prior.values():
            if initialpos is None:
                initialpos = np.random.normal(i['mu'],i['sigma'], size=(1, self.sampler.nwalkers)).T
            else:
                initialpos = np.concatenate((initialpos, np.random.normal(i['mu'],i['sigma'], size=(1, self.sampler.nwalkers)).T),
                                             axis=1)

        print(initialpos.shape)
        self.sampler.run_mcmc(initialpos, nsteps=self.nsteps,  progress=True, store=True)
        print(self.sampler.get_autocorr_time())
    
    def log_prior(self, params):
        ''' 
        calculates the log prior for the parameters everthing is uniform except the metalicty.
        Parameters
        ----------
        params: dict
            A dictionary of the fit parameters with the parater names and there current values
        '''

        # check if there are parameters in `params` that are not in `prior`:
        if len(set(self.prior).symmetric_difference(set(params))) > 0:
            raise ValueError("You must set priors for all parameters you wish to evaluate!") 

        #check if the parameter fall in bounded intervals        
        for key in self.bounds.keys():
            if (self.bounds[key][0] > params[key] or self.bounds[key][1] < params[key]):
                return -np.inf
        if 0 > params['met']:
            return -np.inf
        for key in params.keys():
            if key.startswith('Z'):
                if 0 > params[key]:
                    return -np.inf
        #caluculate priors on parameters
        prior = 0
        for key in self.prior.keys():
            prior+= np.log(1.0/(np.sqrt(2*np.pi)*self.prior[key]['sigma']))-0.5*(params[key]-self.prior[key]['mu'])**2/self.prior[key]['sigma']**2
        return prior

    def log_likelihood(self, params):
        ''' 
        Calculate the log_likelhood between the model and the data

        Parameters
        ----------
        data: array
        model: torch nn.Model() object
            NN model
        params: dir
            dictionary of parameters names and values
        '''
        #intialize list of abunadace
        dict_abund = {}
        for i in torch.arange(6,31):
            dict_abund[f'Z{i}'] = params['met']
            for key in params.keys():
                if key == f'Z{i}':
                    dict_abund[f'Z{i}'] = params['met']*params[f'Z{i}']
        
        temp = torch.tensor([params['temp']], dtype=torch.float32, device=self.device)
        #calculate spectra with NN

        with torch.no_grad():
            ymodel = self.combined_model(temp, dict_abund, params['logz'],  params['norm'], params['vel'])

        mu = ymodel[self.intv]*self.chan_diff*self.combined_model.exp_time+1e-30

        ps = Poisson(mu)
        return torch.sum(ps.log_prob(self.counts)).detach().numpy()
        #return np.sum(poisson.logpmf(self.counts, mu=mu))
    

    def log_posterior(self, params):
        """
        Calculate the log-posterior for the model parameters,
        given the data and the priors

        Parameters
        ----------
        data: array
        model: torch nn.Model() object
            NN model
        params: dir
            dictionary of parameters names and values
        """
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        else:
            lpost = lp + self.log_likelihood(params)
            return lpost
        

    def plot_spectrum(self, nsample=20):
        ''' 
        plots the spectra data against the distribution of best fits
        Parmeters
        ---------
        nsample: int, default: 20
            number of times sampeling from posterior
        '''
        fig = plt.figure(figsize=(20,10))
        plt.plot(self.energy, self.intensity, alpha=0.5, label='Simulated Data')

        for i in range(nsample):
            sample = self.df.sample().squeeze().to_dict()
            dict_abund = {}
            for i in torch.arange(6,31):
                dict_abund[f'Z{i}'] = sample['met']
                for key in sample.keys():
                    if key == f'Z{i}':
                        dict_abund[f'Z{i}'] = sample['met']*sample[f'Z{i}']

            temp = torch.tensor([sample['temp']], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                sample_spectra = self.combined_model(temp, dict_abund, sample['logz'], 
                           sample['norm'], sample['vel'])
            plt.plot(self.energy, sample_spectra[self.intv].cpu().detach().numpy(), color='green', alpha = 0.05)
        plt.plot(self.energy[0], sample_spectra[self.intv].cpu().detach().numpy()[0], color='green',  label='Models drawn from posterior')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('Energy [KeV]')
        plt.ylabel('Counts/KeV/s')
        plt.ylim(1e-2, max(self.intensity)*2)
        plt.xlim(self.e_min, self.e_max)
        plt.show()

        
        for i in range(len(self.param_names)):
            mcmc = np.percentile(self.fit_distribution[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            txt = f"{self.param_names[i]} = {mcmc[1]}_(-{q[0]})^(+{q[1]})"
            print(txt)
            # The below only works in Visual Studio Code
            #txt = "\mathrm{{{3}}} = {0:.5f}_{{-{1:.5f}}}^{{{2:.5f}}}"
            #txt = txt.format(mcmc[1], q[0], q[1], self.param_names[i])
            #display(Math(txt))

    def plot_timeseries(self):
        '''
        This function plots the timeseries of the walkers of all the parameters from the emcee fit
        '''
        fig, axes = plt.subplots(len(self.param_names), figsize=(20, 10), sharex=True)
        samples = self.sampler.get_chain()
        for i in range(len(self.param_names)):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(self.param_names[i], fontsize='12', rotation=45)
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
    
    def cornerplot(self, discard, fitdir=None, thin=15, true_values=None):
        '''
        shows a corner plot of the sampled posterior for all the parameters,
        can also save the sampled posterior as .csv file and overplot the true values
        Parameters
        ----------
        discard: int
            number of the steps from the front of the chain to discard
        fitdir: str, default:None
            file directotory to save .csv of the parameters from the sampled posterior
        thin: int, default:15
             takes the chain and returns every nth sample
        true_values: dict, default:None
            Dictonary of the trua values of the parameters with keys is names of the parameters
        '''
        self.df = None
        self.fit_distribution = self.sampler.get_chain(discard=discard, thin=thin, flat=True)
        self.df = pd.DataFrame(self.fit_distribution, columns=self.param_names)
        fig = plt.figure()
        g = sns.PairGrid(self.df, diag_sharey=False, corner=True, layout_pad=0.25, despine=False)
        # g.map_lower(sns.scatterplot,  size=.2)
        g.map_lower(sns.kdeplot, cmap="Blues")
        g.map_diag(sns.histplot, stat='density')
        g.map_diag(sns.kdeplot,  color="k")

        #save dataframe
        if fitdir is not None:
            self.df.to_csv(fitdir+'.csv')

        #if true_values are given overplot them
        if true_values is not None:
            j = len(true_values)
            # Loop through each subplot in the pairplot
            for i, ax in enumerate(g.axes.flat):
                if ax is not None:
                    ax.axvline(true_values[i%j], ls='--', color='red')
                    if i%j!=i//j:
                        ax.axhline(true_values[i//j], ls='--', color='red')
                        ax.scatter(true_values[i%j],true_values[i//j], marker=",",color='red')

    def log_prob(self, param_values, param_names, derived=None):
        ''' 
        Update the base model with all the parameters that are being constrained.
        '''
        params = self.flat_to_nested_dict(dict(zip(param_names, param_values)))

        if derived is not None:
            derived = [getattr(self.combined_model, d) for d in derived]
        else:
            derived = [] 
        return self.log_posterior(params), derived
    
    def flat_to_nested_dict(self, dct: dict) -> dict:
        """Convert a dct of key: value pairs into a nested dict.
        Keys that have dots in them indicate nested structure.
        """
        def key_to_dct(key, val, dct):
            if '.' in key:
                key, parts = key.split('.', maxsplit=1)

                if key not in dct:
                    dct[key] = {}

                key_to_dct(parts, val, dct[key])
            else:
                dct[key] = val
        out = {}
        for k, v in dct.items():
            key_to_dct(k, v, out)
        return out
    


class TwoTemp(SingleTemp):
    def __init__(self, nwalkers, nsteps, prior,
                 e_min=None, e_max=None, fdir_nn='neuralnetworks/'):
        '''
        Class to be able fit a spectrum with a Two Temperature model in the form of a gaussion
        Parameters
        ----------
        nwalkers: int
            number walkers for the EMCEE fit
        nsteps: int
            number of steps
        prior: dir
            Intial guess/prior of the variables in the form of {'var_name': {'mu':float, 'sigma':float}}
        fdir_NN: str default:'neuralnetworks/'
            directory of torch object to restucture spectra and best nn
        e_min, e_max: float default:None
            minium or maximum energy

        '''

        self.e_min = e_min
        self.e_max = e_max

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using', self.device)
        self.combined_model = model.TwoTemp(Luminosity_Distance=Luminosity_Distance, fdir_nn=fdir_nn, device=self.device)

        #prior
        self.prior = prior

        self.bounds = {}

        # parameter bounds
        self.bounds['temp1'] = [0.2, 10]
        self.bounds['temp2'] = [0.2, 10]
        self.bounds['logz'] = [-10, 1]
        self.bounds['vel'] = [0, 600]
        self.bounds['norm1'] = [1e5, 1e15]
        self.bounds['norm2'] = [1e5, 1e15]

        #fit parameters
        self.nwalkers = nwalkers
        self.nsteps = nsteps

        self.param_names = list(self.prior.keys())


    def sim_data(self, param, exp_time=10000):
        '''creates simulated data from the neural network model'''
        dict_abund = {}
        for i in torch.arange(6,31):
            dict_abund[f'Z{i}'] = param['met']
            for key in param.keys():
                if key == f'Z{i}':
                    dict_abund[f'Z{i}'] = param['met']*param[f'Z{i}']

        temp1 = torch.tensor([param['temp1']], dtype=torch.float32, device=self.device)
        temp2 = torch.tensor([param['temp2']], dtype=torch.float32, device=self.device)
        self.combined_model.to(self.device)
        with torch.no_grad():
            spectra = self.combined_model(temp1, temp2, dict_abund, param['logz'], 
                                         param['vel'], param['norm1'], param['norm2']).cpu().detach().numpy()
        
        chan_diff = self.combined_model.chan_diff.numpy()
        chan_cent = self.combined_model.chan_cent.numpy()
        self.exp_time = exp_time

        #cut off spectra outside of e_min and e_max
        if self.e_min is None:
            self.e_min= min(chan_cent)
        if self.e_max is None:
            self.e_max = max(chan_cent)

        intv = np.where(chan_cent < self.e_min, False, True)
        self.intv = np.where(chan_cent > self.e_max, False, intv)

        self.counts = np.random.poisson(spectra*chan_diff*exp_time)[self.intv]
        self.intensity = self.counts/chan_diff[self.intv]/exp_time
        self.energy = chan_cent[self.intv]
        self.chan_diff = chan_diff[self.intv]

    def plot_spectrum(self, nsample=20):
        ''' 
        plots the spectra data against the distribution of best fits
        Parmeters
        ---------
        nsample: int, default: 20
            number of times sampeling from posterior
        '''
        fig = plt.figure(figsize=(20,10))
        plt.plot(self.energy, self.intensity, alpha=0.5, label='Simulated Data')

        for i in range(nsample):
            sample = self.df.sample().squeeze().to_dict()
            dict_abund = {}
            for i in torch.arange(6,31):
                dict_abund[f'Z{i}'] = sample['met']
                for key in sample.keys():
                    if key == f'Z{i}':
                        dict_abund[f'Z{i}'] = sample['met']*sample[key]

            temp1 = torch.tensor([sample['temp1']], dtype=torch.float32, device=self.device)
            temp2 = torch.tensor([sample['temp2']], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                sample_spectra = self.combined_model(temp1, temp2, dict_abund, sample['logz'], 
                                                     sample['vel'], sample['norm1'], sample['norm2'])
            plt.plot(self.energy, sample_spectra[self.intv].cpu().detach().numpy(), color='green', alpha = 0.05)
        plt.plot(self.energy[0], sample_spectra[self.intv].cpu().detach().numpy()[0], color='green',  label='Models drawn from posterior')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('Energy [KeV]')
        plt.ylabel('Counts/KeV/s')
        plt.ylim(1e-2, max(self.intensity)*2)
        plt.xlim(self.e_min, self.e_max)
        plt.show()

        
        for i in range(len(self.param_names)):
            mcmc = np.percentile(self.fit_distribution[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            txt = f"{self.param_names[i]} = {mcmc[1]}_(-{q[0]})^(+{q[1]})"
            print(txt)
            # The below only works in Visual Studio Code
            #txt = "\mathrm{{{3}}} = {0:.5f}_{{-{1:.5f}}}^{{{2:.5f}}}"
            #txt = txt.format(mcmc[1], q[0], q[1], self.param_names[i])
            #display(Math(txt))

    def log_likelihood(self, data, model, params):
        ''' 
        Calculaters the log_likelhood between the model and the data
        Parameters
        ----------
        data: array
        model: torch nn.Model() object
            NN model
        params: dir
            directory of parameters names and values
        '''
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        
        #intialize list of abunadace
        dict_abund = {}
        for i in torch.arange(6,31):
            dict_abund[f'Z{i}'] = params['met']
            for key in params.keys():
                if key == f'Z{i}':
                    dict_abund[f'Z{i}'] = params['met']*params[key]
        
        temp1 = torch.tensor([params['temp1']], dtype=torch.float32, device=self.device)
        temp2 = torch.tensor([params['temp2']], dtype=torch.float32, device=self.device)
        #calculate spectra with NN
        with torch.no_grad():
            ymodel = model(temp1, temp2, dict_abund, params['logz'], 
                           params['vel'], params['norm1'],params['norm2'])

        if np.any(np.isnan(lp + np.sum(poisson.logpmf(data, mu=(ymodel[self.intv].cpu().detach().numpy()*self.chan_diff*self.exp_time+1e-30))))):
            print('error')

        return lp + np.sum(poisson.logpmf(data, mu=(ymodel[self.intv].cpu().detach().numpy()*self.chan_diff*self.exp_time+1e-30)))



class TempDist(SingleTemp):
    def __init__(self, nwalkers, nsteps, prior, dist_func, bounds, Luminosity_Distance = None,
                 e_min=None, e_max=None, fdir_nn='neuralnetworks/'):
        '''
        Class to be able fit a spectrum with a temperature distribution in the form of a gaussion
        Parameters
        ----------
        nwalkers: int
            number walkers for the EMCEE fit
        nsteps: int
            number of steps
        prior: dir
            Intial guess/prior of the variables in the form of {'var_name': {'mu':float, 'sigma':float}}
        dist_func: object
            function that returns takes in a dictinary with name 'params' with the distribution parameters
            and returns the temperutre grid and temperature distribution
        bounds: dict
            Hard boundaries for the distribution parameters in form of {'var_name': [float, float]}
        Luminosity_Distance: float default:None
            Luminosity Distance of the source in [m]
        fdir_nn: str default:'neuralnetworks/'
            directory of torch object to restucture spectra and best nn
        e_min, e_max: float default:None
            minium or maximum energy

        '''

        self.e_min = e_min
        self.e_max = e_max

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using', self.device)
        self.combined_model = model.TempDist(Luminosity_Distance=Luminosity_Distance, fdir_nn=fdir_nn, device=self.device)

        #prior
        self.prior = prior
        self.dist_func = dist_func
        self.bounds = bounds

        #bounds
        self.bounds['logz'] = [-10, 1]
        self.bounds['norm'] = [1e5, 1e15]
        self.bounds['vel'] = [0, 600]

        #fit parameters
        self.nwalkers = nwalkers
        self.nsteps = nsteps


        self.param_names = list(prior.keys())

    # **TODO**: This method should go into the `CombinedModel` class!
    def sim_data(self, params, exp_time=10000):
        '''creates simulated data from the neural network model'''
        dict_abund = {}
        for i in torch.arange(6,31):
            dict_abund[f'Z{i}'] = params['met']
            for key in params.keys():
                if key == f'Z{i}':
                    dict_abund[f'Z{i}'] = params['met']*params[key]

        temp_grid, temp_dist = self.dist_func(params)
        temp_grid = torch.tensor(temp_grid, dtype=torch.float32, device=self.device)
        temp_dist = torch.tensor(temp_dist, dtype=torch.float32, device=self.device)
        self.combined_model.to(self.device)
        with torch.no_grad():
            spectra = self.combined_model(temp_grid, temp_dist, dict_abund, params['logz'], 
                                          params['norm'], params['vel']).cpu().detach().numpy()
        
        chan_diff = self.combined_model.chan_diff.numpy()
        chan_cent = self.combined_model.chan_cent.numpy()
        self.exp_time = exp_time

        #cut off spectra outside of e_min and e_max
        if self.e_min is None:
            self.e_min= min(chan_cent)
        if self.e_max is None:
            self.e_max = max(chan_cent)

        intv = np.where(chan_cent < self.e_min, False, True)
        self.intv = np.where(chan_cent > self.e_max, False, intv)

        self.counts = np.random.poisson(spectra*chan_diff*exp_time)[self.intv]
        self.intensity = self.counts/chan_diff[self.intv]/exp_time
        self.energy = chan_cent[self.intv]
        self.chan_diff = chan_diff[self.intv]


    def plot_spectrum(self, nsample=20):
        ''' 
        plots the spectra data against the distribution of best fits
        Parmeters
        ---------
        nsample: int, default: 20
            number of times sampeling from posterior
        '''
        fig = plt.figure(figsize=(20,10))
        plt.plot(self.energy, self.intensity, alpha=0.5, label='Data')

        for i in range(nsample):
            params = self.df.sample().squeeze().to_dict()
            dict_abund = {}
            for i in torch.arange(6,31):
                dict_abund[f'Z{i}'] = params['met']
                for key in params.keys():
                    if key == f'Z{i}':
                        dict_abund[f'Z{i}'] = params['met']*params[key]

            temp_grid, temp_dist = self.dist_func(params)
            temp_grid = torch.tensor(temp_grid, dtype=torch.float32, device=self.device)
            temp_dist = torch.tensor(temp_dist, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                sample_spectra = self.combined_model(temp_grid, temp_dist, dict_abund, params['logz'], 
                           params['norm'], params['vel'])
            plt.plot(self.energy, sample_spectra[self.intv].cpu().detach().numpy()/self.chan_diff/self.exp_time, color='green', alpha = 0.05)
        plt.plot(self.energy[0], (sample_spectra[self.intv].cpu().detach().numpy()/self.chan_diff/self.exp_time)[0], color='green',  label='Models drawn from posterior')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('Energy [KeV]')
        plt.ylabel('Counts/KeV/s')
        plt.ylim(1e-2, max(self.intensity)*2)
        plt.xlim(self.e_min, self.e_max)
        plt.show()

        
        for i in range(len(self.param_names)):
            mcmc = np.percentile(self.fit_distribution[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            txt = f"{self.param_names[i]} = {mcmc[1]}_(-{q[0]})^(+{q[1]})"
            print(txt)
            # The below only works in Visual Studio Code
            #txt = "\mathrm{{{3}}} = {0:.5f}_{{-{1:.5f}}}^{{{2:.5f}}}"
            #txt = txt.format(mcmc[1], q[0], q[1], self.param_names[i])
            #display(Math(txt))
        
    
    def log_likelihood(self, data, model, params):
        ''' 
        Calculaters the log_likelhood between the model and the data
        Parameters
        ----------
        data: array
        model: torch nn.Model() object
            NN model
        params: dir
            directory of parameters names and values
        '''
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        
        #intialize list of abunadace
        dict_abund = {}
        for i in torch.arange(6,31):
            dict_abund[f'Z{i}'] = params['met']
            for key in params.keys():
                if key == f'Z{i}':
                    dict_abund[f'Z{i}'] = params['met']*params[key]
        

        temp_grid, temp_dist = self.dist_func(params)
        temp_grid = torch.tensor(temp_grid, dtype=torch.float32, device=self.device)
        temp_dist = torch.tensor(temp_dist, dtype=torch.float32, device=self.device)
        #calculate spectra with NN
        with torch.no_grad():
            ymodel = model(temp_grid, temp_dist, dict_abund, params['logz'], 
                           params['norm'], params['vel'])

        return lp + np.sum(poisson.logpmf(data, mu=(ymodel[self.intv].cpu().detach().numpy()*self.chan_diff*self.exp_time+1e-30)))
