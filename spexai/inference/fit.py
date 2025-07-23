import numpy as np
import pandas as pd
import torch
import emcee
from scipy.stats import poisson

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
font = {'size'   : 16}
matplotlib.rc('font', **font)

import spexai.inference.write_tensors as write_tensors

import spexai.inference.model as model

class FitTempDist(object):
    def __init__(self, nwalkers, nsteps, Luminosity_Distance = None,
                 prior= {'temp': {'mu': 5, 'sigma': 2}, 'stdevtemp': {'mu': -5, 'sigma': 2}, 'met':  {'mu': 1, 'sigma': .3}, 'Z_':{'mu': 1, 'sigma': .3},
                 'vel':  {'mu': 100, 'sigma': 50},  'norm': {'mu': 1e10, 'sigma': 1e10}, 'logz': {'mu': -5,  'sigma': 2}},
                 e_min=None, e_max=None):
        '''
        Class to be able fit a spectrum with a temperature distribution in the form of a gaussion
        Parameters
        ----------
        spectra: array
            spectra of data in counts/sec/kev
        energy_data: array
            center energy bins of data
        dx: array
            width energy bins of data
        exp_time: float
            exposure time of the data
        nwalkers: int
            number walkers for the EMCEE fit
        nsteps: int
            number of steps
        Luminosity_Distance: float default:None
            Luminosity Distance of the source in [m]
        prior: dir
            Intial guess/prior of the variables in the form of {'var_name': {'mu':float, 'sigma':float}}
        fdir: str default:'restructure_spectra'
            directory of torch object to restucture spectra
        e_min, e_max: float default:None
            minium or maximum energy

        '''

        self.e_min = e_min
        self.e_max = e_max

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.combined_model = model.TempDist(Luminosity_Distance=Luminosity_Distance).to(self.device)

        #prior
        self.prior = prior

        #interval
        self.int_temp = [0.2,10]
        self.int_logz = [-10, .3]
        self.int_stdvt = [-5, 1]
        self.int_norm = [1e5, 1e15]
        self.int_vel = [0, 600]

        #fit parameters
        self.nwalkers = nwalkers
        self.nsteps = nsteps

        self.param_names = ['Temperature [KeV]', 'Temp Stdev log[KeV]','Redshift [log(z)]', 'Velocity [km/sec]', 'Metalicity [Fe/H]', 'Normalisation']


    def load_data(self, filepath):
        '''reads in the fits file of the data'''
        counts, channels, exp_time  = write_tensors.read_data(filepath)
        dx = self.combined_model.new_dx
        chan_ecent = self.combined_model.x
        
        self.exp_time = exp_time

        #cut off spectra outside of e_min and e_max
        if self.e_min is None:
            self.e_min= min(channels)
        if self.e_max is None:
            self.e_max = max(channels)

        interval = np.where(chan_ecent < self.e_min, False, True)
        self.interval = np.where(chan_ecent > self.e_max, False, interval)
        interval_data = np.where(channels < self.e_min, False, True)
        interval_data = np.where(channels > self.e_max, False, interval_data)

        self.counts = (counts[interval_data]).astype(int)
        self.intensity = counts[interval_data]/dx[interval_data]/self.exp_time
        self.energy = chan_ecent[self.interval]
        self.dx = dx[interval_data]

    def fit_spectra(self, add_params=None, add_position=None):
        ''' 
        Fit the spectrum with emcee method and prints the autocorrolation time
        Paramaters
        ----------
        add_params: list of str
            list of element names to fit to the ratio of iron in the form 'Z. [Z./Fe]' with '.' the atom number.
        add_position: array
            array of initial position with lenght of nwalkers
        '''
        if add_params is not None:
            for i in add_params:
                self.param_names.append(i)

        self.sampler = emcee.EnsembleSampler(
            nwalkers = self.nwalkers,
            ndim = len(self.param_names),
            log_prob_fn = self.log_prob,
            kwargs = {
                'param_names': self.param_names,
                'data': self.counts,
                'model': self.combined_model
                }
            )
        
        initialpos = np.concatenate((np.random.normal(self.prior['temp']['mu'],self.prior['temp']['sigma'], size=(1, self.sampler.nwalkers)),
                                     np.random.normal(self.prior['stdevtemp']['mu'], self.prior['stdevtemp']['sigma'], size=(1, self.sampler.nwalkers)),
                                     np.random.normal(self.prior['logz']['mu'], self.prior['logz']['sigma'], size=(1, self.sampler.nwalkers)),
                                     np.random.normal(self.prior['vel']['mu'], self.prior['vel']['sigma'], size=(1, self.sampler.nwalkers)),
                                     np.random.normal(self.prior['met']['mu'], self.prior['met']['sigma'], size=(1, self.sampler.nwalkers)),
                                     np.random.normal(self.prior['norm']['mu'], self.prior['norm']['sigma'],  size=(1, self.sampler.nwalkers))), axis=0).T
        if add_position is not None:
            initialpos = np.concatenate((initialpos, add_position.T), axis=1)
        self.sampler.run_mcmc(initialpos, nsteps=self.nsteps,  progress=True, store=True)
        print(self.sampler.get_autocorr_time())

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
        self.df['method']= 'EMCEE'
        

        g = sns.pairplot(self.df, hue="method", plot_kws={"s": 12})
        if fitdir is not None:
            self.df.to_csv(fitdir)

        
        #if true_values are given overplot them
        if true_values is not None:
            # Loop through each subplot in the pairplot
            for i, ax in enumerate(g.axes.flat):
                if i%len(self.param_names)!=i//len(self.param_names):
                    ax.axhline(true_values[self.param_names[i//len(self.param_names)]], ls='--', color='k')
                ax.axvline(true_values[self.param_names[i%len(self.param_names)]], ls='--', color='k')


    def plot_spectrum(self):
        ''' 
        plots the spectra data against the distribution of best fits
        '''
        fig = plt.figure(figsize=(20,10))
        plt.plot(self.energy.detach().numpy(), self.intensity
                 , alpha=0.5, label='Simulated Data')

        for i in range(100):
            sample = self.df.sample().squeeze()[:-1].to_dict()
            dict_abund = {}
            for i in torch.arange(6,31):
                dict_abund[f'Z{i}'] = sample['Metalicity [Fe/H]']
                for key in sample.keys():
                    if key == f'Z{i} [Z{i}/Fe]':
                        dict_abund[f'Z{i}'] = sample['Metalicity [Fe/H]']*sample[f'Z{i} [Z{i}/Fe]']

            low = max(sample['Temperature [KeV]']-5*10**sample['Temp Stdev log[KeV]'], 0.2)
            high = min(sample['Temperature [KeV]']+5*10**sample['Temp Stdev log[KeV]'],10)
            temp_grid = torch.linspace(low,
                                        high, 
                                        500, dtype=torch.float32, device='cuda')
            temp_dist = self.normal_dist(temp_grid, sample['Temperature [KeV]'], 10**sample['Temp Stdev log[KeV]'])
            temp_dist = temp_dist/torch.sum(temp_dist*torch.mean(torch.diff(temp_grid)))
            with torch.no_grad():
                sample_spectra = self.combined_model(temp_grid, temp_dist, dict_abund, sample['Redshift [log(z)]'], 
                           sample['Normalisation'], sample['Velocity [km/sec]'])
            plt.plot(self.energy.detach().numpy(), sample_spectra[self.interval].cpu().detach().numpy(), color='green', alpha = 0.05)
        plt.plot(self.energy.detach().numpy()[0], sample_spectra[self.interval].cpu().detach().numpy()[0], color='green',  label='Models drawn from posterior')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('Energy [KeV]')
        plt.ylabel('Counts/KeV/s')
        plt.ylim(1e-2, max(self.intensity)*2)
        plt.xlim(0.1, 15)
        plt.show()

        
        for i in range(len(self.param_names)):
            mcmc = np.percentile(self.fit_distribution[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            txt = f"{self.param_names[i]} = {mcmc[1]}_(-{q[0]})^({+{q[1]}})"
            # The below only works in Visual Studio Code
            #txt = "\mathrm{{{3}}} = {0:.5f}_{{-{1:.5f}}}^{{{2:.5f}}}"
            #txt = txt.format(mcmc[1], q[0], q[1], self.param_names[i])
            #display(Math(txt))
        
    
    def log_prior(self, params):
        ''' 
        calculates the log prior for the parameters everthing is uniform except the metalicty.
        '''
        if (self.int_temp[0] > params['Temperature [KeV]'] or self.int_temp[1] < params['Temperature [KeV]']
            or self.int_stdvt[0] > params['Temp Stdev log[KeV]'] or self.int_stdvt[1] < params['Temp Stdev log[KeV]']
            or self.int_logz[0] > params['Redshift [log(z)]'] or self.int_logz[1] < params['Redshift [log(z)]']
            or self.int_norm[0] > params['Normalisation'] or self.int_norm[1] < params['Normalisation']
            or self.int_vel[0] > params['Velocity [km/sec]'] or self.int_vel[1] < params['Velocity [km/sec]']
            or 0 > params['Metalicity [Fe/H]']
        ):
            return -np.inf
        
        prior_met = np.log(1.0/(np.sqrt(2*np.pi)*self.prior['met']['sigma']))-0.5*(params['Metalicity [Fe/H]']-self.prior['met']['mu'])**2/self.prior['met']['sigma']**2
        prior_stdevtemp = np.log(1.0/(np.sqrt(2*np.pi)*self.prior['stdevtemp']['sigma']))-0.5*(params['Temp Stdev log[KeV]']-self.prior['stdevtemp']['mu'])**2/self.prior['stdevtemp']['sigma']**2
        prior_temp =  np.log(1.0/(np.sqrt(2*np.pi)*self.prior['temp']['sigma']))-0.5*(params['Temperature [KeV]']-self.prior['temp']['mu'])**2/self.prior['temp']['sigma']**2
        prior_vel =  np.log(1.0/(np.sqrt(2*np.pi)*self.prior['vel']['sigma']))-0.5*(params['Velocity [km/sec]']-self.prior['vel']['mu'])**2/self.prior['vel']['sigma']**2
        
        prior_Z = 0
        for key in params.keys():
            if key.startswith('Z'):
                prior_Z += np.log(1.0/(np.sqrt(2*np.pi)*self.prior['Z_']['sigma']))-0.5*(params[key]-self.prior['Z_']['mu'])**2/self.prior['Z_']['sigma']**2
                if params[key] < 0:
                    return -np.inf
        return prior_met + prior_Z + prior_temp + prior_vel + prior_stdevtemp

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
            dict_abund[f'Z{i}'] = params['Metalicity [Fe/H]']
            for key in params.keys():
                if key == f'Z{i} [Z{i}/Fe]':
                    dict_abund[f'Z{i}'] = params['Metalicity [Fe/H]']*params[f'Z{i} [Z{i}/Fe]']
        
        #intialize temperature grid
        low = max(params['Temperature [KeV]']-4*10**params['Temp Stdev log[KeV]'], 0.1)
        high = min(params['Temperature [KeV]']+4*10**params['Temp Stdev log[KeV]'],10)
        temp_grid = torch.linspace(low,
                                    high, 
                                    400, dtype=torch.float32, device='cuda')
        temp_dist = self.normal_dist(temp_grid, params['Temperature [KeV]'], 10**params['Temp Stdev log[KeV]'])
        #calculate spectra with NN
        with torch.no_grad():
            ymodel = model(temp_grid, temp_dist, dict_abund, params['Redshift [log(z)]'], 
                           params['Normalisation'], params['Velocity [km/sec]'])
            
        if np.any(np.isnan(lp + np.sum(poisson.logpmf(data, mu=(ymodel[self.interval].cpu().detach().numpy()*self.dx*self.exp_time+1e-30))))):
            print('error')

        return lp + np.sum(poisson.logpmf(data, mu=(ymodel[self.interval].cpu().detach().numpy()*self.dx*self.exp_time+1e-30)))
    
    
    def log_prob(self, param_values, param_names, data, model, derived=None):
        ''' 
        Update the base model with all the parameters that are being constrained.
        '''
        params = self.flat_to_nested_dict(dict(zip(param_names, param_values)))

        if derived is not None:
            derived = [getattr(model, d) for d in derived]
        else:
            derived = [] 
        return self.log_likelihood(data, model, params), derived
    
    def normal_dist(self, x, mean, sd):
        '''
        Normal Distribution
        '''
        sd = torch.tensor(sd, dtype=torch.float32, device=x.device)
        prob_density = (torch.sqrt(torch.tensor(2*np.pi, dtype=torch.float32, device=x.device)*sd**2)) * torch.exp(-0.5*((x-torch.tensor(mean, dtype=torch.float32, device=x.device))/sd)**2)
        return prob_density
    
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
    
