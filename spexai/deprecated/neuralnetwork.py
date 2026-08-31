import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float32)


class FFN(nn.Module):
    def __init__(self, input_size, output_size, n_hidden, hidden_size, act_name='relu', p=0.):
        super(FFN, self).__init__()
        '''
        Fast Forward Neural Network with flexibal number of layers and layer size.
        Parameters
        ----------
        input_size, output_size: int, int
            size of input and output of model
        n_nidden: int
            number of hidden layers
        hidden_size: int
            number of neurons in hidden layers
        act_name: str, default 'relu'
            name of the activation function used
        p: float, default 0.
            dropout rate between 0 and 1
        '''
        self.out = output_size
        self.act_name = act_name
        self.n_hidden = n_hidden
        self.hidden_size = hidden_size
        self.p = p
        self.act_fn = nn.ModuleDict({
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'softmax' : nn.Softmax(),
            'sofltpus' : nn.Softplus(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU()})

        self.layers = nn.Sequential()
        for i in range(n_hidden):
            self.layers.add_module(f'dropout{i}', nn.Dropout(p))
            self.layers.add_module(f'linear{i}', nn.Linear(input_size, hidden_size))
            #add trainible layer for custom activation function
            if act_name == 'nonlin':
                alpha = nn.Parameter(torch.normal(0., 1., size=(hidden_size,)), requires_grad=True)
                gamma = nn.Parameter(torch.normal(0., 1., size=(hidden_size,)), requires_grad=True)
                self.layers.add_module(f'activation{i}', NonLinear(alpha, gamma))
            else:
                self.layers.add_module(f'activation{i}', self.act_fn[act_name]) 
            input_size = hidden_size
        self.layers.add_module(f'dropout{i+1}', nn.Dropout(p))
        self.layers.add_module(f'linear{i+1}', nn.Linear(input_size, output_size))


    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 1)
        # ModuleList can act as an iterable, or be indexed using ints
        return self.layers(x)
    
    def __str__(self):
        return f'FF_out({self.out})_nL({self.n_hidden}|{self.hidden_size})_Act({self.act_name})_p({self.p})'

    

class CNN(torch.nn.Module):
    def __init__(self, input_size, output_size, n_FF, hidden_size, n_CNN,  kernal_size, n_channels, act_name='relu', p=0.):
        super(CNN, self).__init__()
        '''
        Inversed Convolutional Neural Network with flexibal number of layers and layer size and a Flexibal Fast Forward layer at the start..
        hidden_size must be divisible by n_channels and 
        (hidden_size/n_channels - ((kernal_size-1) * (n_CC-1))) must be <= then output_size.

        Parameters
        ----------
        input_size, output_size: int, int
            size of input and output of model
        n_FF: int
            number of linear layers, must be bigger then 0 and divisible by n_channels.
        hidden_size: int
            number of neurons in output linear layers
        n_CNN: int
            number of convolutional layers
        n_channels: int
            number of input channels and execept for last layer output channels
        act_name: str, default 'relu'
            name of the activation function used
        p: float, default 0.
            dropout rate between 0 and 1
        '''

        self.act_fn = nn.ModuleDict({
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'softmax' : nn.Softmax(),
            'sofltpus' : nn.Softplus(),
            'tanh': nn.Tanh(),
        })
        
        
        self.n_channels = n_channels

        self.out = output_size
        self.act_name = act_name
        self.nFF = n_FF
        self.nCNN = n_CNN
        self.nkernal = kernal_size
        self.hidden_size = hidden_size
        self.p = p

        #Fast Forward part
        self.FF_layers = nn.Sequential()
        for i in range(n_FF):
            self.FF_layers.add_module(f'dropout{i}', nn.Dropout(p))
            self.FF_layers.add_module(f'linear{i}', nn.Linear(input_size, hidden_size))
            #add trainible layer for custom activation function
            if act_name == 'nonlin':
                alpha = nn.Parameter(torch.normal(0., 1., size=(n_FF,)), requires_grad=True)
                gamma = nn.Parameter(torch.normal(0., 1., size=(n_FF,)), requires_grad=True)
                self.FF_layers.add_module(f'activation{i}', NonLinear(alpha, gamma))
            else:
                self.FF_layers.add_module(f'activation{i}', self.act_fn[act_name]) 
            input_size = hidden_size
        
        #Inversed Convolutinal Layers
        self.CNN_layers = nn.Sequential()
        if hidden_size % n_channels != 0:
            raise Exception('hidden_size must be divisible by n_channels')
        #calculate the layer size of CNN
        L_in = int(hidden_size/n_channels)
        for j in range(n_CNN-1):
            self.CNN_layers.add_module(f'dropout{n_FF+j}', nn.Dropout(p))
            self.CNN_layers.add_module(f'CCN{j}', nn.ConvTranspose1d(n_channels,n_channels, kernal_size))
            #calculate output size of Convlayer
            L_in = (L_in - 1) + kernal_size
            #add trainible layer for custom activation function
            if act_name == 'nonlin':
                alpha = nn.Parameter(torch.normal(0., 1., size=(n_channels, L_in,)), requires_grad=True)
                gamma = nn.Parameter(torch.normal(0., 1., size=(n_channels, L_in,)), requires_grad=True)
                self.CNN_layers.add_module(f'activation{n_FF+i}', NonLinear(alpha, gamma))
            else:
                self.CNN_layers.add_module(f'activation{n_FF+i}', self.act_fn[act_name]) 
        self.CNN_layers.add_module(f'dropout{n_FF+n_CNN}', nn.Dropout(p))

        #calculate kernal size to make sure that L_out = output_size
        kernal_size = output_size + 1 - L_in
        if kernal_size <= 0:
            raise Exception('(hidden_size/n_channels - ((kernal_size-1) * (n_CC-1))) must be <= then output_size')
        self.CNN_layers.add_module(f'CNN{n_CNN}', nn.ConvTranspose1d(n_channels, 1, kernal_size))

    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 1)
        # FF part
        x = self.FF_layers(x)
        #CNN part
        x = x.view(batch_size, self.n_channels, -1)
        x = self.CNN_layers(x)
        return x.view(batch_size, -1)
    
    def __str__(self):
        return f'CNN_out({self.out})_nFF({self.nFF}|{self.hidden_size})_nCNN({self.nCNN}|{self.n_channels}|{self.nkernal})_Act({self.act_name})_p({self.p})'
    
    
class NonLinear(nn.Module):
    '''
    Non-linear activation function
    '''
    def __init__(self, alpha=None, gamma=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    
    def forward(self, input):
        return  torch.multiply(torch.add(self.gamma, torch.multiply(torch.sigmoid(torch.multiply(self.alpha, input)), torch.subtract(1, self.gamma))), input)
