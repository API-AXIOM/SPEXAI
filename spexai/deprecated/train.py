import time
import os
import datetime
import numpy as np
import torch
import torch.optim as optim


torch.set_default_dtype(torch.float32)

class NeuralNetworkTrainer(object):
    def __init__(self, X, y, X_test, y_test, model, loss_fn=torch.nn.MSELoss(),
                 optimizer='adam', l_rate=1e-2, lr_factor=0.33, lr_patience=250,
                 lr_threshold=2e-2, lr_mode='rel', lr_cooldown=100, pca=None,
                 scaler_pca = None, 
                 scaler_flux=None, mask=None, mask_test=None, f_dir='log/', 
                 save_model=False, name=None,  element=00):
        '''
        Trainer class to train a given model on the training data. This class sets up 
        the training process for the neural network.

        Parameters
        ---------
        X: pytorch.Tensor
            training data (temperatures)

        y: pytorch.Tensor
            training data labels (spectra)

        X_test: pytorch.Tensor
            validation data (temperatures)

        y_test : pytorch.Tensor
            validation data (spectra)

        model: torch.Module
            the neural network model

        mask, mask_test: tensor
            this tensor equals one on position where the data is used for training
            (i.e. where the flux in the spectrum is > `min_flux`)
            and 0 for data positions that are excluded in the loss function

        scaler_flux : sklearn.StandardScaler object
            A standard scaler for per-feature scaling of the spectra

        save_model : bool, default False
            If True, save the trained model to disk

        f_dir : str
            The directory where to store the saved model (if `save_model = True`)      

        name : str
            An identifier for a model to be stored

        element : int
            A numerical identifier for the element being trained, used in constructing
            an informative name for the model if saved to disk

        Other Parameters
        ----------------
        loss_fn : torch.nn loss function
            The loss function used for training the neural network

        optimizer : str
            The optimizer to use for training the neural network
            Possible values can be `adam`, `nadam`, `adadelta` or `adagrad`

        l_rate : float
            Learning rate

        lr_factor, lr_patience, lr_treshold, lr_mode, lr_cooldown: 
            Parameters for setting and adjusting the learning rate
 
        '''
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        self.loss_fn = loss_fn
        self.mask = mask
        self.mask_test = mask_test
        self.scaler_flux = scaler_flux
        self.element = element

        self.X = X
        self.y = y
        self.mask = mask
        self.X_test = X_test
        self.y_test = y_test
        self.mask_test = mask_test

        self.pca = pca
        self.f_dir = f_dir
        self.save_model = save_model
        self.name = name

        self.model = model.to(self.device)
        self.optimizer = self._build_optimizer(optimizer, l_rate)
        self.schedular = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                factor=lr_factor, 
                                                                patience=lr_patience,
                                                                threshold=lr_threshold, 
                                                                threshold_mode=lr_mode,
                                                                cooldown=lr_cooldown)
        
        self.loss_test = []
        self.loss_train = []

        self.scale = torch.tensor(scaler_flux.scale_, dtype=torch.float32).to(self.device)
        self.mean = torch.tensor(scaler_flux.mean_, dtype=torch.float32).to(self.device)

    def train(self, nepochs, nbatch):
        '''
        Trains model over `nepochs`  with a batch size of `nbatch`

        Parameters
        ----------
        nepochs : int
            The number of epochs for which to train

        nbatch : int
            The number of training examples in each minibatch
        '''
        time_run = time.time()
        for epoch in range(nepochs):
            self.model.train()
            shuffle = np.random.permutation(len(self.X))
            X_shuffled = self.X[shuffle]
            y_shuffled = self.y[shuffle]
            if self.mask is not None:
                mask_shuffled = self.mask[shuffle]
            
            loss_batch = 0
            #train model from subset of trainingdata in minibatches
            time_epoch = time.time()
            for i in range(len(self.X)//nbatch):    
                X_batch = X_shuffled[i*nbatch:(i+1)*nbatch]
                y_batch = y_shuffled[i*nbatch:(i+1)*nbatch]
                


                # forward pass
                y_pred = self.model(X_batch.to(self.device)).squeeze(1)

                if self.mask is not None:
                    mask_batch = mask_shuffled[i*nbatch:(i+1)*nbatch]
                    #rescale back to original spectra and look if pred spectra is above minimum flux (-10)
                    if self.pca is not None:
                        y_pred_pca = self.scaler_pca.inverse_transform(y_pred)
                        y_pred_unscaled = self.pca.inverse_transform(y_pred_pca)
                        mask_batch = torch.where(y_pred_unscaled > -10, 1, mask_batch.to(self.device))
                    else:
                        mask_batch = torch.where(torch.add(torch.mul(y_pred, self.scale), self.mean) > -10, 1, mask_batch.to(self.device))
                    y_pred = torch.mul(y_pred, mask_batch.to(self.device))
                    y_batch = torch.mul(y_batch.to(self.device), mask_batch)

                # compute loss
                loss = self.loss_fn(y_pred, y_batch.to(self.device))

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_batch += loss.item()
                
            loss_batch /= len(self.X)//nbatch
            self.schedular.step(loss_batch)

            loss_test = self.test(nbatch)            

            self.loss_train.append(loss_batch)
            self.loss_test.append(loss_test)
            time_l = time.time()-time_epoch
            
            #print epoch time every 10% of the model training
            if i==len(self.X)//nbatch-1 and epoch%(0.1*nepochs)==0:
                self.model.eval()
                with torch.no_grad():
                    pred_test = self.model(self.X_test.to(self.device)).squeeze(1).cpu().detach()
                    pred_train = self.model(self.X.to(self.device)).squeeze(1).cpu().detach()
                print('Epoch ({}/{}): Runtime {:.2e} sec, Loss: Test={:.2e};Train={:2e}' .format(epoch, nepochs, time_l,
                                                                        self.original_loss(pred_test, self.y_test, mask = self.mask_test),
                                                                        self.original_loss(pred_train, self.y, mask = self.mask)))
                      

        self.model.eval()
        with torch.no_grad():
            pred_test = self.model(self.X_test.to(self.device)).squeeze(1).cpu().detach()
            pred_train = self.model(self.X.to(self.device)).squeeze(1).cpu().detach()
        print('Epoch ({}/{}): Runtime {:.2e} sec, Loss: Test={:.2e};Train={:2e}' .format(epoch, nepochs, time_l,
                                                                self.original_loss(pred_test, self.y_test, mask = self.mask_test),
                                                                self.original_loss(pred_train, self.y, mask = self.mask)))
        print('Time of the run is {:.2e} hrs \n'.format((time.time()-time_run)/3600))

        if self.save_model == True:
            self.save()


    def test(self, nbatch):
        '''
        Calculates the average loss given a current model on a validation 
        or test data set

        Parameters
        -----
        nbatch: int
            the size of the minibatch over which to compute the loss function        

        Returns
        -------
        loss_test: float
            Average validation/test loss

        '''
        #intit
        self.model.eval()
        loss_test = 0

        # shuffle the testdata (therefor minibatch are diff each time)
        shuffle = np.random.permutation(len(self.X_test))
        X_test_shuffled = self.X_test[shuffle]
        y_test_shuffled = self.y_test[shuffle]
        if self.mask_test is not None:
            mask_test_shuffle = self.mask_test[shuffle]
        
        
        with torch.no_grad():
            #calculate loss of testdata in minibatches
            for i in range(len(self.X_test)//nbatch):
                X_test_batch = X_test_shuffled[i*nbatch:(i+1)*nbatch].to(self.device)
                y_test_batch = y_test_shuffled[i*nbatch:(i+1)*nbatch]
            
                #forward
                y_test_pred = self.model(X_test_batch)

                if self.mask_test is not None:
                    mask_test_batch = mask_test_shuffle[i*nbatch:(i+1)*nbatch]
                    mask_test_batch = torch.where(torch.add(torch.mul(y_test_pred, self.scale), self.mean) > -10, 1, mask_test_batch.to(self.device))
                    y_test_pred = torch.mul(y_test_pred, mask_test_batch.to(self.device))
                    y_test_batch = torch.mul(y_test_batch.to(self.device), mask_test_batch)

                loss_test += self.loss_fn(y_test_pred, y_test_batch.to(self.device)).item()

        loss_test /= (len(self.X_test)//nbatch) #avg loss

        return loss_test
    
    def predict(self, X):
        '''
        Model prediction for a given temperature or array of 
        temperatures. 

        Paramters
        ---------
        X: Tensor
            Set of temperatures for which to predict
            the spectrum

        Returns
        -------
        ypred : pytorch.Tensor
             The predicted spectra for the temperatures stored 
             in `X`
        '''
        self.model.eval()
        X = X.to(self.device)
        return self.model(X)
    
    def save(self):
        '''
        Save the trained model to a pickle file with the name and date of the model
        '''
        #give_data
        
        date = datetime.datetime.now()
        date_name = str(date.day)+'|'+str(date.month)
        try:
            os.makedirs(os.path.join('log','Z'+str(self.element), date_name))
        except:
            pass
        
        if self.name is not None:
            torch.save(self.model.state_dict(), 'log/'+'Z'+str(self.element)+'/'+date_name+'/'+self.name+'.pt')
            np.savetxt('log/'+'Z'+str(self.element)+'/'+date_name+'/'+self.name+'_Loss_train.txt', self.loss_train)
            np.savetxt('log/'+'Z'+str(self.element)+'/'+date_name+'/'+self.name+'_Loss_test.txt', self.loss_test)
        else:
            torch.save(self.model.state_dict(), 'log/'+'Z'+str(self.element)+'/'+date_name+'/'+str(self.model)+'.pt')
            np.savetxt('log/'+'Z'+str(self.element)+'/'+date_name+'/'+str(self.model)+'_Loss_train.txt', self.loss_train)
            np.savetxt('log/'+'Z'+str(self.element)+'/'+date_name+'/'+str(self.model)+'_Loss_test.txt', self.loss_test)
    
    def load(self, model_dir):
        '''
        Load a trained model from file.

        Parameters
        ----------
        model_file : str
            Path to saved `state_dict` of model

        '''
        self.model = torch.load(model_dir).to(self.device)
        self.loss_test = np.loadtxt(model_dir+'_Loss_test')
        self.loss_train = np.loadtxt(model_dir+'_Loss_train')

    def _build_optimizer(self, optimizer, learning_rate):
        '''
        Initializes the optimzer
        Parameters
        ----------
        optimizer: str
            Name of the optimzer
        learning rate: float
        '''
        if optimizer == "adam":
            optimizer = optim.Adam(self.model.parameters(),
                                lr=learning_rate)
        elif optimizer == "nadam":
            optimizer = optim.NAdam(self.model.parameters(),
                                lr=(learning_rate*2))
        elif optimizer == "adadelta":
            optimizer = optim.Adadelta(self.model.parameters(),
                                lr=(learning_rate*1e3))
        elif optimizer == "adagrad":
            optimizer = optim.Adagrad(self.model.parameters(),
                                lr=(learning_rate*10))
        return optimizer
    
    def original_loss(self,  pred,  original, mask=None):
        '''
        This defines the custom loss function we use in training the emulator. 
        It calculates the loss function defined in the `loss_fn` attribute, but if 
        a `mask` is set, regions where the value of `mask` = 0 will only contribute 
        to the loss function if the emulated value is above the critical value, and 
        ignored otherwise (with the assumption that it will be small enough not to  
        contribute to the flux in a meaningful way. This helps the network focus on 
        training in those areas where the flux most contributes to the observed spectrum

        Parameters
        ----------
        pred : torch.Tensor
            the spectrum fluxes predicted by the neural network

        original : torch.Tensor
            the training data for the same temperature as the predicted spectrum

        mask : torch.Tensor
            a boolean mask to apply to the spectra, where the loss function will 
            only be calculated for energy bins where mask == 1 or where the predicted 
            flux exceeds the threshold. 

        '''
        #calculate loss for original spectra
        if mask is None:
            if self.scaler_flux is None:
                loss = self.loss_fn(torch.pow(10, pred), torch.pow(10, original)).item()
            else:
                pred = torch.from_numpy(self.scaler_flux.inverse_transform(pred))
                original = torch.from_numpy(self.scaler_flux.inverse_transform(original))
                loss = self.loss_fn(torch.pow(10, pred), torch.pow(10, original)).item()
        else:    
            #calculate loss on region where flux is above minimum
            if self.scaler_flux is None:
                mask = mask > 0
                loss = self.loss_fn(torch.pow(10, pred[mask]), torch.pow(10, original[mask])).item()
            else:
                #calculate loss over original spectrum
                mask = mask > 0
                pred = torch.from_numpy(self.scaler_flux.inverse_transform(pred))
                original = torch.from_numpy(self.scaler_flux.inverse_transform(original))
                loss = self.loss_fn(torch.pow(10, pred[mask]), torch.pow(10, original[mask])).item()
        return loss
        
