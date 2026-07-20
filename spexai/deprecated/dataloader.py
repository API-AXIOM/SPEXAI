import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


torch.set_default_dtype(torch.float32)

class SpexAIMemoryDataset(Dataset):
    def __init__(self, annotations_file, datadir, indexdir, flux=None, min_flux=None,  size="all",
                 min_energy=None, max_energy=None, min_temp=None, max_temp=None, element=None, pca=None, 
                 flux_pca=None, ncomp=10, apply_pca=False):
        
        """
        Helper class to load the SPEX spectra for ML. 
        Assumes spectra are for a single element, and that each spectrum is in 
        its own text file. There are two summary files, one for the training data 
        and one for the test data. Each of these files contains pairs of [filename, temperature]
        to link the temperature to the spectrum generated using that temperature.
        
        Parameters
        ----------
        annotations_file : str 
            Filename with the annotations, either for the training or test data
        
        datadir : str
            The path to the data and the annotations file
                        
        min_energy, max_energy: float, float, default None
            If None, set the minimum and maximum energy to be considered to 
            the first and last element of the spectrum. Otherwise, use these
            values to determine which segment to read out from each spectrum

        min_temp, max_tem: float, float, default None
            temperature range of data
        
        min_flux: float, default None
            The minimum flux of the spectrum in logspace
        """
        
        self.labels = pd.read_csv(indexdir + annotations_file, sep=" ")
        self.datadir = datadir
        self.element = element

        self.min_energy = min_energy
        self.max_energy = max_energy
        self.min_temp = min_temp
        self.max_temp = max_temp
        
        self.minidx = None
        self.maxidx = None

        self.scaler_flux = None
        self.scaler_temp = None
        self.scaler_pca = None
        self.pca = pca
        
        self.flux = []
        self.temp = []
         
        #self.flux_pca = None
        self.flux_scaled = None
        self.flux_scaled_pca= None
        self.temp_scaled = None
        self.energy = None

        # if flux is set, create internal array 
        # from that flux array
        if flux is not None:
            self.temp = []
            self.flux = torch.Tensor(np.array(flux))
            # loop over all the elements in flux 
            # to get out the respective temperatures
            # NOTE THAT THIS ASSUMES THAT THE 
            # TEMPERATURES AND FLUXES HAVE BEEN 
            # LOADED FROM THE SAME FILE!
            for i in range(len(self.flux)):
                idx = self.labels.index[i]
                self.temp.append(self.labels.iloc[idx, 0])
                # for one index, actually run read_data
                # so that the energy array gets set
                if i == 0:
                    t,f = self.read_data(idx)
            self.temp = torch.tensor(self.temp).type(torch.float)
            # check whether the energy and flux arrays have the same 
            # length, because if not, the values for min_energy and 
            # max_energy don't match those that generated the fluxes
            if len(self.energy) != self.flux.shape[1]:
                raise ValueError("min_energy and max_energy in class call not the same " + \
                                 "as those that generated the flux array!")
        else:
            if size == "all":
                for idx in self.labels.index:
                    temp, flux = self.read_data(idx)
                    self.temp.append(temp)
                    self.flux.append(flux)
            else:
                for idx in self.labels.index[:size]:
                    temp, flux = self.read_data(idx)
                    self.temp.append(temp)
                    self.flux.append(flux)

            self.temp = torch.Tensor(self.temp)
            self.flux = torch.Tensor(np.array(self.flux))
        #temperature bound
        if self.min_temp is not None:
            self.flux = self.flux[self.temp >= self.min_temp]
            self.temp = self.temp[self.temp >= self.min_temp]
        if self.max_temp is not None:
            self.flux = self.flux[self.temp <= self.max_temp]
            self.temp = self.temp[self.temp <= self.max_temp]

        #create lower flux bound    
        if min_flux is not None:
            #calculate mask
            self.mask = torch.Tensor(self.flux > min_flux)
            self.flux = torch.where(self.flux < (min_flux-5), (min_flux-5), self.flux)
            
        
        if apply_pca is True and flux_pca is None:
             if self.pca is None:
                 self.pca = PCA(n_components=ncomp)
             flux_pca = torch.Tensor(self.pca.fit_transform(self.flux))

        self.flux_pca = flux_pca

        #order the flux on temperature
        sort = torch.argsort(self.temp)
        self.temp = self.temp[sort]
        self.flux = self.flux[sort]
        if self.flux_pca is not None:
            self.flux_pca = self.flux_pca[sort]
        self.mask = self.mask[sort]

        
    def __len__(self):
        return len(self.labels)
    
    def read_data(self, idx):
        data_path = self.labels.iloc[idx,1]
        #data_path = os.path.join(self.datadir + self.labels.iloc[idx, 2])
        data = pd.read_csv(data_path, sep=" ", names=["energy", "flux"])

        if self.minidx is None and self.maxidx is None:
            if self.min_energy is None:
                self.minidx = 0
            else:
                self.minidx = data["energy"].searchsorted(self.min_energy)
            if self.max_energy is None:
                self.maxidx = len(data)-1
            else:
                self.maxidx = data["energy"].searchsorted(self.max_energy)
        flux = data.loc[self.minidx:self.maxidx, "flux"].to_numpy()
        #if len(flux) != 50125:
            #print(data_path)
            
        if self.energy is None:
            self.energy = data.loc[self.minidx:self.maxidx, "energy"].to_numpy()
        temp = float(self.labels.iloc[idx, 0])
        return temp, flux
    
    def __getitem__(self, idx):
        label = self.temp[idx]
        flux = self.flux[idx,:]
        return label, flux
    
   
    def scale_data(self):
        '''
        Scales the temperature and the flux to a mean=0 and stdev=1.
        '''
        self.scaler_flux = StandardScaler()
        self.scaler_temp = StandardScaler()
        self.flux_scaled = torch.Tensor(self.scaler_flux.fit_transform(self.flux))
        self.temp_scaled = self.temp #torch.from_numpy(self.scaler_temp.fit_transform(self.temp.view(len(self.temp), 1))).flatten()
        if self.pca is not None:
            self.scaler_pca = StandardScaler()
            self.flux_scaled_pca = torch.Tensor(self.flux_pca)

    def split_data(self, split_ratio=0.8, seed=42):
        '''
        Split the data randomly in a training and validation subset, 
        returns the resulting torch arrays in a dictionary
        Parameters
        ----------
        split_ratio: float, default 0.8
            ratio between number of traing and validation data
        seed: ind, default 42
            integer used as ranomd seed to shuffle the data.
        '''
        
        #shuffle data before splitting
        length = len(self.temp)
        np.random.seed(seed)
        shuffle = np.random.permutation(length)
        temp = self.temp[shuffle]
        flux = self.flux[shuffle]
        mask = self.mask[shuffle]

        #order the tensor after splitting wit sort
        sort_train = np.argsort(temp[0:int(length*split_ratio)])
        sort_test = np.argsort(temp[int(length*split_ratio):length])
        #split mask
        self.mask_train = mask[0:int(length*split_ratio)][sort_train]
        self.mask_test = mask[int(length*split_ratio):length][sort_test]
        #split temp and spectra
        self.x_train = temp[0:int(length*split_ratio)][sort_train]
        self.x_test = temp[int(length*split_ratio):length][sort_test]
        self.y_train = flux[0:int(length*split_ratio)][sort_train]
        self.y_test = flux[int(length*split_ratio):length][sort_test]
        #if pca data is available
        if self.flux_pca is not None:
            flux_pca = self.flux_pca[shuffle]
            self.y_pca_train = flux_pca[0:int(length*split_ratio)][sort_train]
            self.y_pca_test = flux_pca[int(length*split_ratio):length][sort_test]
        #if scaled data is available
        if self.scaler_flux is not None:
            temp_scaled = self.temp_scaled[shuffle]
            flux_scaled = self.flux_scaled[shuffle]
            self.x_scaled_train = temp_scaled[0:int(length*split_ratio)][sort_train]
            self.x_scaled_test = temp_scaled[int(length*split_ratio):length][sort_test]
            self.y_scaled_train = flux_scaled[0:int(length*split_ratio)][sort_train]
            self.y_scaled_test = flux_scaled[int(length*split_ratio):length][sort_test]
            #if scaled and pca data is available
            if self.flux_scaled_pca is not None:
                flux_scaled_pca = self.flux_scaled_pca[shuffle]
                self.y_scaled_pca_train = flux_scaled_pca[0:int(length*split_ratio)][sort_train]
                self.y_scaled_pca_test = flux_scaled_pca[int(length*split_ratio):length][sort_test]
    
    def power(self, x, scaler=None):
        '''
        Takes power of 10 on array and makes sure data is not scaled or a pca
        '''
        
        if torch.is_tensor(x) == True:
            x = x.cpu().detach().numpy()

        if scaler is not False:
            if self.scaler_flux is not None:
                if self.pca is not None:
                    #scaled and PCA, the PCA of data was scaled.
                    if self.scaler_pca is not None:
                        x = self.pca.inverse_transform(self.scaler_pca.inverse_transform(x))
                    #scaled and PCA, pca is taken from scaled data
                    else:
                        x = self.scaler_flux.inverse_transform(self.pca.inverse_transform(x))
                #Scaled but not PCA
                else:
                    x = self.scaler_flux.inverse_transform(x)
            #not scaled but is PCA
            elif self.pca is not None:
                x = self.pca.inverse_transform(x)
        return np.power(10, x)
    
    def save_scaler(self):
        '''
        Save the scaler mean and scale as a tensor.
        '''
        np.savetxt(np.array(self.scaler_flux.scale_, requires_grad=True, dtype=torch.float32), 'Best_NN/Z'+str(self.element)+'_scale.txt')
        np.savetxt(np.array(self.scaler_flux.mean_, requires_grad=True, dtype=torch.float32), 'Best_NN/Z'+str(self.element)+'_mean.txt')

def fraction(pred, test, absolute=True):
    if absolute is True:
        return abs(pred-test)/test
    else:
        return (pred-test)/test
    
