# DEPRECATED: legacy CNN-era I/O. The operator stack uses
# spexai.inference.response (RMF/ARF) and self-contained operator checkpoints.
import warnings as _warnings
_warnings.warn(
    "spexai.inference.write_tensors is deprecated (legacy CNN stack); use "
    "spexai.inference.response instead.",
    DeprecationWarning, stacklevel=2)

from astropy.io import fits
import numpy as np
import pandas as pd
import torch
import re
torch.set_default_dtype(torch.float32)

# TODO(operator-port): load_models still builds CNN/FFN nets; replace with
# SpectralOperator checkpoint loading. Bridged to spexai.deprecated so the
# CNN path keeps working until the port lands.
from spexai.deprecated import FFN, CNN



def rmf_to_torchmatrix(filepath):
    """
    Read in an OGIP-compliant response matrix using astropy,
    and turn into a sparse PyTorch tensor.

    Parameters
    ----------
    filepath: str
        The path of the response matrix file

    Returns
    -------
    rm: sparse_coo_tensor
        response matrix

    (spec_e_cent, spec_e_lo, spec_e_hi): (tensor, tensor, tensor)
        tensors of center energy bins and upper and lower energy bounds of the spectrum

    (chan_e_cent, chan_e_lo, chan_e_hi): (tensor, tensor, tensor)
        tensors of center energy bins and upper and lower energy bounds of the channels
    """ 
    with fits.open(filepath) as rmf_file:
        rmf_data = rmf_file['MATRIX'].data
        rmf_channel = rmf_file['EBOUNDS'].data
        #assumes that the fchan, and nchan are array and that index stars at 0.
        fchan = rmf_data["F_CHAN"]
        nchan = rmf_data["N_CHAN"]
        matrix = rmf_data['MATRIX']

        naxis = rmf_file["MATRIX"].header["NAXIS2"]
        detchans = rmf_file["MATRIX"].header["DETCHANS"]
        
    spec_e_lo = torch.tensor(rmf_data["ENERG_LO"].astype(np.float32))
    spec_e_hi = torch.tensor(rmf_data["ENERG_HI"].astype(np.float32))
    spec_e_cent = torch.div(torch.add(spec_e_lo, spec_e_hi), torch.tensor(2))

    chan_e_lo = torch.tensor(rmf_channel['E_MIN'].astype(np.float32))
    chan_e_hi = torch.tensor(rmf_channel['E_MAX'].astype(np.float32))
    chan_e_cent = torch.div(torch.add(chan_e_lo, chan_e_hi), torch.tensor(2))

    #list
    chan = np.array([], dtype=int) #column index
    spec = np.array([], dtype=int) #row index
    values = np.array([], dtype=np.float32) #value

    #looping over all energy spectra bins
    for i, v in enumerate(nchan):
        #reading in matrix of response for given spectra energy bin
        values = np.append(values, matrix[i][matrix[i] != 0])
        #looping over number of groups per spectra energy
        for j, length in enumerate(v):
            # columb index start Fchan to Fchan+Nchan
            chan = np.append(chan, np.arange(fchan[i][j], fchan[i][j]+length, dtype=int))
            # row index row beloning to 
            spec = np.append(spec, np.ones(length, dtype=int)*i)

    shape = (naxis, detchans)
    index = np.append([chan], [spec], axis=0)
    rm = torch.sparse_coo_tensor(index, values, shape, requires_grad=True)

    return rm, (spec_e_cent, spec_e_lo, spec_e_hi), (chan_e_cent, chan_e_lo, chan_e_hi)


def arf_to_tensor(filepath):
    """
    Read in an OGIP-compliant Auxiliary Response File using astropy,
    and turn into a PyTorch tensor.

    Parameters
    ----------
    filepath: str
        The path of the response matrix file

    Returns
    -------
    rm: sparse_coo_tensor
        response matrix
    spec_e: tuples
        tensors of center energy bins and upper and lower energy bounds of the spectrum
    """ 
    with fits.open(filepath) as amf_file:
        amf_file.verify('fix')
        data = amf_file['SPECRESP'].data
        
    spec_resp = torch.tensor(data['SPECRESP'].astype(np.float32))

    spec_e_lo = torch.tensor(data['ENERG_LO'].astype(np.float32))
    spec_e_hi = torch.tensor(data['ENERG_HI'].astype(np.float32))
    spec_e_cent = torch.div(torch.add(spec_e_lo, spec_e_hi), torch.tensor(2))
    spec_resp *= 1e-4 #take into acount per square cm instead of m
    return spec_resp, (spec_e_cent, spec_e_lo, spec_e_hi)


def read_energy_file(filepath, min_energy=None, max_energy=None):
    '''
    Read in file with energy and errors

    Parameters
    ----------
    filepath: str
        The filepath of enery rows
    
    min_energy, max_energy: float, float, default None
        If None, set the minimum and maximum energy to be considered to 
        the first and last element of the spectrum. Otherwise, use these
        values to determine which segment to read out from each spectrum
    
    Returns
    -------
    spec_e_cent: tensor
        tensor of the center of energy bins

    spec_e_lo: tensor
        tensor of the lower bound of energy bins
        
    spec_e_hi: tensor
        tensor of the higher bound of energy bins
    '''
    
    data = pd.read_csv(filepath,  header=None, skiprows=1, delim_whitespace=True, index_col=False, names=["energy", "pos_err", 'neg_err', 'flux'])
    data['energy_low'] = data.energy+data.neg_err
    data['energy_high'] = data.energy+data.pos_err

    if min_energy is None:
        minidx = 0
    else:
        minidx = data["energy"].searchsorted(min_energy)
    if max_energy is None:
        maxidx = len(data)-1
    else:
        maxidx = data["energy"].searchsorted(max_energy)

    spec_e_cent = torch.tensor(data.loc[minidx:maxidx, "energy"].to_numpy())
    spec_e_lo = torch.tensor(data.loc[minidx:maxidx, 'energy_low'].to_numpy())
    spec_e_hi = torch.tensor(data.loc[minidx:maxidx, 'energy_high'].to_numpy())

    return spec_e_cent, spec_e_lo, spec_e_hi


def load_models(elements, file_dir, model_names, device):
    ''' 
    Read all the trained models of all the elements and put them in ModuleDict with the elements name as key (Z__).
    Parameters
    ----------
    elements: array of int between 1 and 30
        atom number of elements
    file_dir: str
        directory name of the best NN models
    model_names: list of str
        list of the model names
    models: list of torch.nn.Module() objects
        The model class with the same NN architecture as the model names

    '''
    dic = torch.nn.ModuleDict({})
    for i in elements:
        added = False
        for name in np.array(model_names).flatten():
            try:
                if name.startswith('FF'):
                    l_names = re.findall(r"[-+]?(?:\d*\.*\d+)", name)
                    act_func = re.findall(r'\(.*?\)', name)[2][1:-1]
                    model = FFN(1, int(l_names[0]), int(l_names[1]), int(l_names[2]), act_func, float(l_names[3]))
                elif name.startswith('CNN'):
                    l_names = re.findall(r"[-+]?(?:\d*\.*\d+)", name)
                    act_func = re.findall(r'\(.*?\)', name)[3][1:-1]
                    model = CNN(1, int(l_names[0]), int(l_names[1]), int(l_names[2]), int(l_names[3]),
                                 int(l_names[4]), int(l_names[5]), act_func, int(l_names[6]))

                model.load_state_dict(torch.load(file_dir+'Z'+str(i)+'/'+name+'.pt', map_location=device))
                model.eval()
                dic['Z'+str(i)] = model
                added = True
            except:
                pass
        if added is False:
            NameError(f'Error: the model for element {i} did not load')
    return dic


def make_sparsex(x, n=300):
    ''' '
    Will fill a sparse matrix with energies around a central energy and will transform the energies with a factor of itself
    Parameters
    ----------
    x: tensor
        energy grid spectra
    n: int, default:300
        kernel size of the convolution
    '''
    collumn_index = torch.tensor([])
    row_index = torch.tensor([])
    x_values = torch.tensor([])

    for i, x_i in enumerate(x):
        full_col = torch.arange(-int((n/25)*x_i), int((n/25)*x_i), dtype=torch.long)+i
        full_col = full_col[full_col >= 0]
        col = full_col[full_col <= len(x)-1]
        row = torch.ones_like(col)*i
        x_row = (x[col]-x_i)/x_i

        collumn_index = torch.cat((collumn_index, col)).type(torch.long)
        row_index = torch.cat((row_index, row))
        x_values = torch.cat((x_values, x_row))

    index = torch.cat((row_index.view(1,-1), collumn_index.view(1,-1)))
    sparse_x = torch.sparse_coo_tensor(index, x_values, (len(x), len(x))).type(torch.float32)
    return sparse_x

def read_data(filepath):
    '''
    Reads fits file of the observed data
    '''
    with fits.open(filepath) as data_file:
        data_file.verify('fix')
        data = data_file['SPECTRUM'].data
        exposure_time = data_file['SPECTRUM'].header['EXPOSURE']
    counts = data['COUNTS']
    channels = data['CHANNEL']
    return counts, channels, exposure_time