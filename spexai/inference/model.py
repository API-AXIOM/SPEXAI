import pickle
import numpy as np
import torch
import torch.nn as nn
from torchinterp1d import interp1d as interp1d_torch
import scipy.constants as constants

from spexai.inference import write_tensors
# TODO(operator-port): port CombinedModel off the per-element CNN/FFN nets
# onto SpectralOperator. Bridged to spexai.deprecated until then.
from spexai.deprecated import FFN, CNN
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)


class CombinedModel(nn.Module):
    def __init__(self, lumdist=None,  fdir_nn= 'neuralnetworks/',
                   shape=(50125,), list_elements=np.arange(1, 31), possible_modelnames=[],
                     possible_models=[], device=torch.device('cpu')):
        '''
        torch nn.Modele object to calulate a full observed X-ray spectra with NN-emulators of the individual elements
        Parameters
        ----------
        lumdist: float, default:None
            Luminosity Distantace of the source in [m]
        fdir: str, default:'restructure_spectra'
            directory of torch object to restucture spectra
        fdir_bestmodels: str, default:'Best_NN/'
            directory of NN models of the elements
        shape: tuple, default:(50124,)
            shape of the output of the model
        list_elements: array of int, default:np.arange(1, 31)
            array of the atom number of all the elements in the combined model
        possible_modelnames: list of str, default:[]
            add str names of the NN emulators of the individual elements.
        possible_modelnames: list of torch.nn.Module, default:[]
            add the model class of the NN emulators of the individual elements.
        device: torch.device(), default:torch.device('cpu')
            device used to evalute the model
        '''
        super(CombinedModel, self).__init__()

        #add most common NN models
        possible_modelnames.append(['FF_out(50125)_nL(3|150)_Act(tanh)_p(0.0)', 
                                    'FF_out(50125)_nL(3|150)_Act(nonlin)_p(0.0)',
                                    'FF_out(50125)_nL(3|250)_Act(nonlin)_p(0.0)', 
                                    'CNN_out(50125)_nFF(2|150)_nCNN(1|75|100)_Act(tanh)_p(0.0)'])
        self.models = write_tensors.load_models(list_elements, fdir_nn,  possible_modelnames, device)
        self.device = device

        #read in mean and stdev for inverse standard scaling
        self.means = nn.ParameterDict({})
        self.scales = nn.ParameterDict({})

        for key in self.models.keys():
            dir_mean = str(fdir_nn+str(key)+'/'+str(key)+'_mean.txt')
            mean = torch.from_numpy(np.loadtxt(dir_mean)).type(torch.float32)
            self.means[key] = mean

            dir_scale  =  str(fdir_nn+str(key)+'/'+str(key)+'_scale.txt')
            scale =  torch.from_numpy(np.loadtxt(dir_scale)).type(torch.float32)
            self.scales[key] = scale

        self.e_cent = torch.from_numpy(np.loadtxt(fdir_nn+'energy/spex_e_cent.txt')).type(torch.float32)
        e_hi = torch.from_numpy(np.loadtxt(fdir_nn+'energy/spex_e_hi.txt')).type(torch.float32)
        e_lo = torch.from_numpy(np.loadtxt(fdir_nn+'energy/spex_e_lo.txt')).type(torch.float32)
        self.e_diff = e_hi-e_lo
        self.diag_index = torch.arange(len(self.e_cent)).view(1,-1).repeat(2,1)

        self.lumdist = lumdist
        self.shape = shape

    def broadening_old(self, spectra, velocity):
        '''
        This function applies the velocity broadening on the spectra with a convulation with a gaussion
        Parameters
        ----------
        spectra: tensor
            spectra before velocity broadening
        velocity: float
           velocity broading of turbulance [km/sec]
        '''
        if velocity < 1e-30:
            velocity = 1e-30
        shape = (len(self.e_cent), len(self.e_cent))
        #tensor function to calculate pdf-value of the normal distribution
        normal = lambda x, stdev : torch.exp(torch.tensor(-0.5, dtype=torch.float32, device=x.device)*torch.pow(torch.div(x,stdev.to(x.device)), torch.tensor(2, dtype=torch.float32, device=x.device)))

        #use x_values to construct sparse matrix of normaldistributions 
        prev_values = self.sm_x._values().type(torch.float32)
        values = normal(prev_values.to(spectra.device), torch.tensor(velocity*1e3/constants.c, dtype=torch.float32)).cpu()

        normal_matrix = torch.sparse_coo_tensor(self.sm_x._indices(), values,  shape)

        #normalise the normal distribution
        values = (torch.tensor(1, dtype=torch.float32, device=spectra.device)/torch.sparse.mm(normal_matrix.to(spectra.device), self.e_diff.view(-1,1).to(spectra.device)).flatten()).cpu()
        normilisation = torch.sparse_coo_tensor(self.diag_index, values, shape)
        normal_matrix = torch.sparse.mm(normilisation.to(spectra.device), normal_matrix.to(spectra.device)).cpu()

        #convolve the normaldistributions with the spectrum
        spectra_dx = spectra.view(-1,1)*self.e_diff.view(-1,1).to(spectra.device)
        return torch.sparse.mm(normal_matrix.to(spectra.device), spectra_dx).flatten()

    def broadening(self, spectrum, velocity):
        '''
        New version of line broadening, with better sparse matrix casting rules
        This function applies the velocity broadening on the spectra with a 
        convolution with a gaussion
        
        Parameters
        ----------
        spectra: tensor
            spectra before velocity broadening
        velocity: float
           velocity broading of turbulance [km/sec]
        '''
        if velocity < 1e-30:
            return spectrum
    #        velocity = 1e-30
    
        shape = (len(self.e_cent), len(self.e_cent))
        #tensor function to calculate pdf-value of the normal distribution
        normal = lambda x, stdev : torch.exp(torch.tensor(-0.5, dtype=torch.float32, device=x.device)*torch.pow(torch.div(x,stdev.to(x.device)), torch.tensor(2, dtype=torch.float32, device=x.device)))
    
        prev_values = self.sm_x_csr.values().type(torch.float32)
        stdev = torch.tensor(velocity*1e3/constants.c, dtype=torch.float32)
        
        gaussian_values = normal(prev_values.to(spectrum.device), stdev).cpu()
        
        normal_matrix = torch.sparse_csr_tensor(self.sm_x_csr.crow_indices(), self.sm_x_csr.col_indices(), gaussian_values, shape, dtype=torch.float32)
        normalisation_values = (torch.tensor(1, dtype=torch.float32, device=spectrum.device)/torch.mv(normal_matrix.to(spectrum.device), self.e_diff.to(spectrum.device)).flatten()).cpu()
        #normalisation = torch.sparse_coo_tensor(combined_model.diag_index, normalisation_values_csr, shape).to_sparse_csr()
        #normal_matrix_new = torch.sparse.mm(normalisation.to(spectrum.device), normal_matrix.to(spectrum.device)).cpu()
        normal_matrix_new = normal_matrix.to_sparse_coo() * normalisation_values[:,None]
        normal_matrix_new = normal_matrix_new.to_sparse_csr()
        
        #convolve the normal distributions with the spectrum
        spectrum_dx = spectrum.view(-1,1)*self.e_diff.view(-1,1).to(spectrum.device)
        spectrum_broad = torch.sparse.mm(normal_matrix_new.to(spectrum.device), spectrum_dx).flatten()
        
        return spectrum_broad

    def emulator_model(self, temp, abundances, norm):
        """
        Compute a spectrum generated by the emulator, without redshift or 
        velocity broadening. 

        Parameters
        ----------
        temp: tesor dtype:torch.float32 device:CombinedModel.device
            Temperature in [kev]
        abundances: torch.nn.ModuleDict
            dictonary of the element abundances in solar abundances. With dictonary key the Z__ element number.
        norm: float
            Normalisation [1e64 m^{-3}] in units of SPEX

        """
        if self.lumdist is not None:
            norm = norm*(1e22/self.lumdist)**2
        y = torch.zeros(self.shape, dtype=torch.float32, device=temp.device)
        for i in np.arange(1,6):
            abundances[f'Z{i}'] = 1
        for key, value in abundances.items():
            value = torch.tensor(value, dtype=torch.float32, device=temp.device)
            model = self.models[key].to(temp.device)
            y += torch.multiply(torch.pow(torch.tensor(10, dtype=torch.float32, device=temp.device), torch.add(torch.multiply(model(temp).flatten(), self.scales[key]), self.means[key])), value)

        return y 

    def forward(self, temp, abundances, logz, norm, velocity):
        '''
        Calculates the simulated spectra with the NN emulator of the individual elements

        Parameters
        ----------
        temp: tesor dtype:torch.float32 device:CombinedModel.device
            Temperature in [kev]
        abundances: torch.nn.ModuleDict
            dictonary of the element abundances in solar abundances. With dictonary key the Z__ element number.
        logz: float
            log_10 of redshift (z) between -10 and 1
        norm: float
            Normalisation [1e64 m^{-3}] in units of SPEX
        velocity: float
            velocity broading of turbulance [km/sec]
        '''

        y = self.emulator_model(temp, abundances, norm)
        output = self.broadening(y.flatten(),  velocity).flatten()
        output = self.rebin_interp(output.flatten(),  10**logz).flatten()*norm
        output = torch.mul(output, self.arf)
        output = torch.sparse.mm(self.rm, output.view(-1,1)).flatten()
        return output

    def load_data(self, filepath):
        """
        Load an observation. Currently only supports XRISM observations.

        **Note**: response files need to be loaded separately.

        Parameters
        ----------
        filepath: str
            Path and filename for the data to load.


        """
        self.counts, self.channels, self.exp_time  = write_tensors.read_data(filepath)

       
    def load_rm(self, filepath):
        '''
        load the response matrix and energy bins/channels from the  RMF FITS file
        '''
        rm, e, chan =  write_tensors.rmf_to_torchmatrix(filepath)
        self.rm = nn.Parameter(rm)
        #energy bins
        self.new_e_cent = e[0]
        self.new_e_diff = (e[2]-e[1]).type(torch.float32)
        #energy channels
        self.chan_cent = chan[0]
        self.chan_diff = (chan[2]-chan[1]).type(torch.float32)
        #define variables for  redshift
        self.diag_index = torch.arange(len(self.e_cent)).view(1,-1).repeat(2,1)
        self.rebin_interp = RebinSpectra_interpolate(self.e_cent, self.new_e_cent).to(self.device)

    def load_arf(self, filepath):
           '''
           load in the effective area response from the FITS file
           '''

           arf, spec_e = write_tensors.arf_to_tensor(filepath)
           self.arf = nn.Parameter(arf)

    def load_sparsematrix_x(self, n=300):
        '''
        load in the sparse matrix used for the convolution that implements line broadening
        '''
        self.sm_x = write_tensors.make_sparsex(self.e_cent, n=n)
        
    def simulate_data(self, params, ntemp=1, dist_func=None, exp_time=10000, e_min=None, e_max=None):
        '''
        Simulate a data set from the combined model. `ntemp` sets the number of 
        temperature components. If `dist_func` is not `None`, then this parameter 
        will be ignored and the temperature will be parametrized as a distribution
        instead.

        Parameters
        ----------
        params : dict
            A dictionary with parameters to use to generate the spectrum

        ntemp : int, default 1
            The number of temperature components, currently can be 1 or 2
            Will be ignored if `dist_func` is set to something other than
            `None`

        dist_func : function
            A distribution function to use for a temperature distribution. 
            If this is not `None`, then `ntemp` will be ignored.

        exp_time : float
            The exposure time of the simulated observation

        e_min, e_max : float, float
            Minimum and maximum energy bounds for the spectrum
        '''
        dict_abund = {}
        for i in torch.arange(6,31):
            dict_abund[f'Z{i}'] = params['met']
            for key in params.keys():
                if key == f'Z{i}':
                    dict_abund[f'Z{i}'] = params['met']*params[key]
        
        if dist_func is not None:
            temp_grid, temp_dist = self.dist_func(params)
            temp_grid = torch.tensor(temp_grid, dtype=torch.float32, device=self.device)
            temp_dist = torch.tensor(temp_dist, dtype=torch.float32, device=self.device)
            #self.combined_model.to(self.device)
            with torch.no_grad():
                spectrum = self.forward(temp_grid, temp_dist, dict_abund, params['logz'],
                                              params['norm'], params['vel']).cpu().detach().numpy()
        
        else:
            if ntemp == 1:
                temp = torch.tensor([params['temp']], dtype=torch.float32, device=self.device)

                #combined_model.to(self.device)
                with torch.no_grad():
                    spectrum = self.forward(temp, dict_abund, params['logz'],
                                                  params['norm'], params['vel']).cpu().detach().numpy()
           
            elif ntemp == 2:
                temp1 = torch.tensor([param['temp1']], dtype=torch.float32, device=self.device)
                temp2 = torch.tensor([param['temp2']], dtype=torch.float32, device=self.device)
                #self.combined_model.to(self.device)
                with torch.no_grad():
                    spectrum = self.forward(temp1, temp2, dict_abund, param['logz'],
                                           param['vel'], param['norm1'], param['norm2']).cpu().detach().numpy()

            else:
                raise ValueError("Currently can only do 1 or 2 temperatures or a distribution.")


        chan_diff = self.chan_diff.numpy()
        chan_cent = self.chan_cent.numpy()

        #cut off spectra outside of e_min and e_max
        if e_min is None:
            e_min= min(chan_cent)
        if e_max is None:
            e_max = max(chan_cent)

        intv = np.where(chan_cent < e_min, False, True)
        intv = np.where(chan_cent > e_max, False, intv)

        counts = np.random.poisson(spectrum*chan_diff*exp_time)[intv]
        intensity = counts/chan_diff[intv]/exp_time
        energy = chan_cent[intv]

        return energy, counts, intensity

    def save_matrices(self, path):
        """
        Store expensive matrices to pickle files for faster loading.
        **Note**: This is meant as short-term storage for fast loading 
        when the same responses are used multiple times. Do not 
        use this method for long-term storage!


        Parameters
        ----------
        path : str
           Path and any file prefixes to use for saving the data
        """

        with open(f"{path}rmf.pkl", "wb") as f:
            pickle.dump(rm, f)
            
        with open(f"{path}new_e_cent.pkl", "wb") as f:
            pickle.dump(new_e_cent, f)
        
        with open(f"{path}chan_cent.pkl", "wb") as f:
            pickle.dump(chan_cent, f)
            
        with open(f"{path}chan_diff.pkl", "wb") as f:
            pickle.dump(chan_diff, f)
        
        with open(f"{path}diag_index.pkl", "wb") as f:
            pickle.dump(diag_index, f)
            
        with open(f"{path}rebin_interp.pkl", "wb") as f:
            pickle.dump(rebin_interp, f)
        
        with open(f"{path}sm_x.pkl", "wb") as f:
            pickle.dump(sm_x, f)
        
        return

    def load_matrices(self, path):
        """
        Load expensive matrices from pickle files saved using the 
        `save_matrices` method.

        **Note**: This is meant as short-term storage for fast loading 
        when the same responses are used multiple times. Do not 
        use this method for long-term storage!

        Parameters
        ----------
        path : str
           Path and any file prefixes to use for saving the data
        """

        with open(f"{path}rmf.pkl", "rb") as f:
            self.rm = pickle.load(f)
            
        with open(f"{path}new_e_cent.pkl", "rb") as f:
            self.new_e_cent = pickle.load(f)
        
        with open(f"{path}chan_cent.pkl", "rb") as f:
            self.chan_cent = pickle.load(f)
        
        with open(f"{path}chan_diff.pkl", "rb") as f:
            self.chan_diff = pickle.load(f)
        
        with open(f"{path}diag_index.pkl", "rb") as f:
            self.diag_index = pickle.load(f)
            
        with open(f"{path}rebin_interp.pkl", "rb") as f:
            self.rebin_interp = pickle.load(f)
                    
        with open(f"{path}sm_x.pkl", "rb") as f:
            self.sm_x = pickle.load(f)
        
class TwoTemp(CombinedModel):
    def __init__(self, **kwargs):
        '''
        Subclass of CombinedModel for a Two Temperature Model. Inherits parameters 
        from its `CombinedModel` superclass. 
        Parameters
        ----------
        Luminosity_Distance: float, default:None
            Luminosity Distantace of the source in [m]
        fdir: str, default:'restructure_spectra'
            directory of torch object to restucture spectra
        fdir_bestmodels: str, default:'Best_NN/'
            directory of NN models of the elements
        shape: tuple, default:(50124,)
            shape of the output of the model
        list_elements: array of int, default:np.arange(1, 31)
            array of the atom number of all the elements in the combined model
        possible_modelnames: list of str, default:[]
            add str names of the NN emulators of the individual elements.
        possible_modelnames: list of torch.nn.Module, default:[]
            add the model class of the NN emulators of the individual elements.
        '''
        super(TwoTemp, self).__init__(**kwargs)

    def forward(self, temp1, temp2, abundances, logz, velocity, norm1, norm2):
        '''
        Calculates the simulated spectra with the NN emulator of the individual elements
        Parameters
        ----------
        temp1: tesor dtype:torch.float32 device:CombinedModel.device
            First Temperature in [kev]
        temp2: tesor
            Second Temperature in [kev]
        abundances: torch.nn.ModuleDict
            Dictonary of the element abundances in solar abundances. With dictonary key the Z__ element number.
        logz: float
            log_10 of redshift (z) between -10 and 1
        velocity: float
            velocity broading of turbulance [km/sec]
        norm1: float
            Normalisation [1e64 m^{-3}] in units of SPEX for the first temperature
        norm2: float
            Normalisation [1e64 m^{-3}] in units of SPEX for the second temperature    
        '''
        
        if self.lumdist is not None:
            norm1 = norm1*(1e22/self.lumdist)**2
            norm2 = norm2*(1e22/self.lumdist)**2
        y = torch.zeros(self.shape, dtype=torch.float32, device=temp1.device)
        for i in np.arange(1,6):
            abundances[f'Z{i}'] = 1
        for key, value in abundances.items():
            value = torch.tensor(value, dtype=torch.float32, device=temp1.device)
            model = self.models[key].to(temp1.device)
            y += torch.multiply(torch.pow(torch.tensor(10, dtype=torch.float32, device=temp1.device), torch.add(torch.multiply(model(temp1).flatten(), self.scales[key]), self.means[key])), value)*norm1
            y += torch.multiply(torch.pow(torch.tensor(10, dtype=torch.float32, device=temp1.device), torch.add(torch.multiply(model(temp2).flatten(), self.scales[key]), self.means[key])), value)*norm2

        output = self.broadening(y.flatten(), velocity).flatten()
        output = self.rebin_interp(output.flatten(), 10**logz).flatten()
        output = torch.mul(output, self.arf)
        output = torch.sparse.mm(self.rm, output.view(-1,1)).flatten()
        return output
    

class TempDist(CombinedModel):
    def __init__(self, **kwargs):
        '''
        Subclass of CombinedModel for Temperature Distributions. Inherits parameters 
        from its `CombinedModel` superclass. 
        Parameters
        ----------
        Luminosity_Distance: float, default:None
            Luminosity Distantace of the source in [m]
        fdir: str, default:'restructure_spectra'
            directory of torch object to restucture spectra
        fdir_bestmodels: str, default:'Best_NN/'
            directory of NN models of the elements
        shape: tuple, default:(50124,)
            shape of the output of the model
        list_elements: array of int, default:np.arange(1, 31)
            array of the atom number of all the elements in the combined model
        possible_modelnames: list of str, default:[]
            add str names of the NN emulators of the individual elements.
        possible_modelnames: list of torch.nn.Module, default:[]
            add the model class of the NN emulators of the individual elements.
        '''
        super(TempDist, self).__init__(**kwargs)

    def forward(self, temp_grid, temp_dist, abundances, logz, norm, velocity):
        '''
        Calculates the simulated spectra with the NN emulator of the individual elements
        Parameters
        ----------
        temp_grid: tesor dtype:torch.float32 device:CombinedModel.device
            tensor of all the temperatures to calculate the spectra for
        temp_dist: tesor
            tensor of all the values to multiply the temperatures of the grid with
        abundances: torch.nn.ModuleDict
            dictonary of the element aboundancies in solar abundances. With dictonary key the Z__ element number.
        logz: float
            log_10 of redshift (z) between -10 and 1
        norm: float
            Normalisation [1e64 m^{-3}] in units of SPEX
        velocity: float
            velocity broading of turbulance [km/sec]
        '''
        dx = torch.mean(torch.diff(temp_grid))
        if self.lumdist is not None:
            norm = norm*(1e22/self.lumdist)**2
        y = torch.zeros(self.shape, dtype=torch.float32, device=temp_grid.device)
        for i in np.arange(1,6):
            abundances[f'Z{i}'] = 1
        for key, value in abundances.items():
            value = torch.tensor(value, dtype=torch.float32, device=temp_grid.device)
            model = self.models[key].to(temp_grid.device)
            y += torch.sum(torch.multiply(
                torch.multiply(torch.pow(torch.tensor(10, dtype=torch.float32, device=temp_grid.device),
                                         torch.add(torch.multiply(model(temp_grid), self.scales[key]), self.means[key])),value),
                dx*temp_dist.type(torch.float32).to(temp_grid.device).view(-1,1,)), axis=0)

        output = self.broadening(y.flatten(), velocity).flatten()
        output = self.rebin_interp(output.flatten(),  10**logz).flatten()*norm
        output = torch.mul(output, self.arf)
        output = torch.sparse.mm(self.rm, output.view(-1,1)).flatten()
        return output


class RebinSpectra_interpolate(nn.Module):
    def __init__(self, ecent, new_ecent):
        super(RebinSpectra_interpolate, self).__init__()
        '''
        Class to rebin the data from energy bounds belonging to the spectra, to a new energy grid.

        Parameters
        ----------
        ecent, new_ecent: tensor
            old and new energy grid.
        '''
        
        self.ecent = nn.Parameter(ecent)
        self.new_ecent = nn.Parameter(new_ecent)


    def forward(self, spectra, z):
        '''
        calcule the spectra in the new enegy grid
        Parameters
        ----------
        spectra: tensor
            spectra before rebinning
        z: float
            The redshift to shift the spectra with
        '''

        #add redshift
        ecent = self.ecent/(1+torch.tensor(z, dtype=torch.float32, device=spectra.device))
        #if needed add padding below
        if ecent[0] > self.new_ecent[0]:
            ecent = torch.cat([torch.tensor([self.new_ecent[0]], dtype=torch.float32, device=ecent.device), ecent])
            spectra = torch.cat([torch.tensor([spectra[0]], dtype=torch.float32, device=spectra.device), spectra])

        #if needed add padding above
        if ecent[-1] < self.new_ecent[-1]:
            ecent = torch.cat([ecent, torch.tensor([self.new_ecent[-1]], dtype=torch.float32, device=ecent.device)])
            spectra = torch.cat([spectra, torch.tensor([spectra[-1]], dtype=torch.float32, device=spectra.device)])
        y = interp1d_torch(ecent, spectra*torch.pow((1+torch.tensor(z, dtype=torch.float32, device=spectra.device)),2), self.new_ecent).type(torch.float32)
        return y
