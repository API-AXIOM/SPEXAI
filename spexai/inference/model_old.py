import numpy as np
import torch
import torch.nn as nn
from torchinterp1d import interp1d as interp1d_torch
import scipy.constants as constants

from spexai.inference import write_tensors
from spexai.train import FFN, CNN
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)


class CombinedModel(nn.Module):
    def __init__(self, Luminosity_Distance=None,  fdir_nn= 'neuralnetworks/',
                   shape=(50125,), list_elements=np.arange(1, 31), possible_modelnames=[],
                     possible_models=[], device=torch.device('cpu')):
        '''
        torch nn.Modele object to calulate a full observed X-ray spectra with NN-emulators of the individual elements
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
        device: torch.device(), default:torch.device('cpu')
            device used to evalute the model
        '''
        super(CombinedModel, self).__init__()

        #add most comen NN models
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

        self.LD = Luminosity_Distance
        self.shape = shape

    def broadening(self, spectra, velocity):
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
        if self.LD is not None:
            norm = norm*(1e22/self.LD)**2
        y = torch.zeros(self.shape, dtype=torch.float32, device=temp.device)
        for i in np.arange(1,6):
            abundances[f'Z{i}'] = 1
        for key, value in abundances.items():
            value = torch.tensor(value, dtype=torch.float32, device=temp.device)
            model = self.models[key].to(temp.device)
            y += torch.multiply(torch.pow(torch.tensor(10, dtype=torch.float32, device=temp.device), torch.add(torch.multiply(model(temp).flatten(), self.scales[key]), self.means[key])), value)

        output = self.broadening(y.flatten(), velocity).flatten()
        output = self.rebin_interp(output.flatten(),  10**logz).flatten()*norm
        output = torch.mul(output, self.arf)
        output = torch.sparse.mm(self.rm, output.view(-1,1)).flatten()
        return output

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
        
        if self.LD is not None:
            norm1 = norm1*(1e22/self.LD)**2
            norm2 = norm2*(1e22/self.LD)**2
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
        if self.LD is not None:
            norm = norm*(1e22/self.LD)**2
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