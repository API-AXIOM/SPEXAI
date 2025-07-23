import numpy as np
import torch
import torch.nn as nn
from torchinterp1d import interp1d as interp1d_torch
import scipy.constants as constants

import spexai.inference.write_tensors as write_tensors
from spexai.train.neuralnetwork import FFN, CNN


torch.set_default_dtype(torch.float32)


class CombinedModel(nn.Module):
    def __init__(self, Luminosity_Distance=None,  fdir_nn= 'neuralnetworks/',
                   shape=(50125,), list_elements=np.arange(1, 31), possible_modelnames=[], possible_models=[]):
        super(CombinedModel, self).__init__()
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
        '''

        #add most comen NN models
        possible_modelnames.append(['FF_out(50125)_nL(3|150)_Act(tanh)_p(0.0)', 'FF_out(50125)_nL(3|150)_Act(nonlin)_p(0.0)',
                                    'FF_out(50125)_nL(3|250)_Act(nonlin)_p(0.0)', 'CNN_out(50125)_nFF(2|150)_nCNN(1|75|100)_Act(tanh)_p(0.0)'])
        possible_models.append([FFN(1,50125,3,150,'tanh'), FFN(1,50125,3,150,'nonlin'), FFN(1,50125,3,250,'nonlin'), CNN(1,50125, 2, 150, 1, 100, 75, 'tanh')])
        self.models = write_tensors.load_models(list_elements, fdir_nn, possible_modelnames, possible_models)

        #read in mean and stdev for inverse standard scaling
        self.means = nn.ParameterDict({})
        self.scales = nn.ParameterDict({})
        for key in self.models.keys():
            dir_mean = str(fdir_nn+str(key)+'/'+str(key)+'_mean.txt')
            mean = torch.from_numpy(np.loadtxt(dir_mean))
            self.means[key] = mean

            dir_scale  =  str(fdir_nn+str(key)+'/'+str(key)+'_scale.txt')
            scale =  torch.from_numpy(np.loadtxt(dir_scale))
            self.scales[key] = scale

        self.diag_index = torch.arange(len(self.x)).view(1,-1).repeat(2,1)
        self.LD = Luminosity_Distance
        self.shape = shape
        
        self.load_data()
        self.rebin_interp = RebinSpectra_interpolate(self.x, self.new_x)


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
        shape = (len(self.x), len(self.x))
        #tensor function to calculate pdf-value of the normal distribution
        normal = lambda x, stdev : torch.exp(torch.tensor(-0.5, dtype=torch.float32, device=x.device)*torch.pow(torch.div(x,stdev.to(x.device)), torch.tensor(2, dtype=torch.float32, device=x.device)))

        #use x_values to construct sparse matrix of normaldistributions 
        prev_values = self.sm_x._values().type(torch.float32)
        values = normal(prev_values.to(spectra.device), torch.tensor(velocity*1e3/constants.c, dtype=torch.float32)).cpu()

        normal_matrix = torch.sparse_coo_tensor(self.sm_x._indices(), values,  shape)

        #normalise the normal distribution
        values = (torch.tensor(1, dtype=torch.float32, device=spectra.device)/torch.sparse.mm(normal_matrix.to(spectra.device), self.dx.view(-1,1).to(spectra.device)).flatten()).cpu()
        normilisation = torch.sparse_coo_tensor(self.diag_index, values, shape)
        normal_matrix = torch.sparse.mm(normilisation.to(spectra.device), normal_matrix.to(spectra.device)).cpu()

        #convolve the normaldistributions with the spectrum
        spectra_dx = spectra.view(-1,1)*self.dx.view(-1,1).to(spectra.device)
        return torch.sparse.mm(normal_matrix.to(spectra.device), spectra_dx).flatten()


    def forward(self, temp, abundancies, logz, norm, velocity):
        '''
        Calculates the simulated spectra with the NN emulator of the individual elements
        Parameters
        ----------
        temp: tesor dtype:torch.float32 device:CombinedModel.device
            Temperature in [kev]
        abundancies: torch.nn.ModuleDict
            dictonary of the element aboundancies in solar abundancies. With dictonary key the Z__ element number.
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
            abundancies[f'Z{i}'] = 1
        for key, value in abundancies.items():
            value = torch.tensor(value, dtype=torch.float32, device=temp.device)
            model = self.models[key].to(temp.device)
            y += torch.multiply(torch.pow(torch.tensor(10, dtype=torch.float32, device=temp.device), torch.add(torch.multiply(model(temp).flatten(), self.scales[key]), self.means[key])), value)

        output = self.broadening(y.flatten(), velocity).flatten()
        output = self.rebin_interp(output.flatten(), 10**logz).flatten()*norm
        output = torch.mul(output, self.spec_resp)
        output = torch.sparse.mm(self.rm, output.view(-1,1)).flatten()
        return output
    

    def load_rm(self, filepath):
        '''
        load the response matrix and energy bins/channels from the  RMF FITS file
        '''
        rm, spec_e, chan_e =  write_tensors.rmf_to_torchmatrix(filepath)
        self.rm = nn.Parameter(rm)
        #energy bins
        self.x = spec_e[0]
        self.dx = (spec_e[2]-spec_e[1]).type(torch.float32)
        #energy channels
        self.new_x = chan_e[0]
        self.new_dx = (chan_e[2]-chan_e[1]).type(torch.float32)


    def load_arf(self, filepath):
           '''
           load in the effective area response from the FITS file
           '''

           arf = write_tensors.arf_to_tensor(filepath)
           self.spec_resp = nn.Parameter(arf)


    def load_sparsematrix_x(self, n=300):
        '''
        load in the sparse matrix used for the convolution that implements line broadening
        '''
        sm_x = write_tensors.make_sparsex(self.x, n=n)
        self.sm_x = torch.load(sm_x)


    
class TempDist(CombinedModel):
    def __init__(self):
        super(CombinedModel, self).__init__()
        '''
        Subclass of CombinedModel for Temperature Distributions
        '''

    def forward(self, temp_grid, temp_dist, abundancies, logz, norm, velocity):
        '''
        Calculates the simulated spectra with the NN emulator of the individual elements
        Parameters
        ----------
        temp_grid: tesor dtype:torch.float32 device:CombinedModel.device
            tensor of all the temperatures to calculate the spectra for
        temp_dist: tesor
            tensor of all the values to multiply the temperatures of the grid with
        abundancies: torch.nn.ModuleDict
            dictonary of the element aboundancies in solar abundancies. With dictonary key the Z__ element number.
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
            abundancies[f'Z{i}'] = 1
        for key, value in abundancies.items():
            value = torch.tensor(value, dtype=torch.float32, device=temp_grid.device)
            model = self.models[key].to(temp_grid.device)
            y += torch.sum(torch.multiply(
                torch.multiply(torch.pow(torch.tensor(10, dtype=torch.float32, device=temp_grid.device),
                                         torch.add(torch.multiply(model(temp_grid), self.scales[key]), self.means[key])),value),
                dx*temp_dist.type(torch.float32).to(temp_grid.device).view(-1,1,)), axis=0)

        output = self.broadening(y.flatten(), velocity).flatten()
        output = self.rebin_interp(output.flatten(),  10**logz).flatten()*norm
        output = torch.mul(output, self.spec_resp)
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
