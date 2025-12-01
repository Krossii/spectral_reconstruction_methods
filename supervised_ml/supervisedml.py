#this needs is errorhandling, the datasets, then some testing and eventually an implementation of checkpointing and state dict saving
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

import numpy as np
import json
import argparse
import time
import pprint
import os
from typing import List, Tuple, Callable

def KL_kernel_Momentum(
        Momentum, 
        Omega
        ):
    Momentum = Momentum[:, np.newaxis]  # Reshape Momentum as column to allow broadcasting
    ker = Omega / (Omega**2 + Momentum**2)  # Element-wise division
    return ker / np.pi 

def KL_kernel_Position_Vacuum(
        Position, 
        Omega
        ):
    Position = Position[:, np.newaxis]  # Reshape Position as column to allow broadcasting
    ker = np.exp(-Omega * np.abs(Position)) + np.exp(-Omega*(len(Position)-Position))
    return ker

def KL_kernel_Position_FiniteT(
        Position, 
        Omega,
        T
        ):
    Position = Position[:, np.newaxis]  # Reshape Position as column to allow broadcasting
    with np.errstate(divide='ignore'):
        ker = np.cosh(Omega * (Position-1/(2*T))) / np.sinh(Omega/2/T)

        # set all entries in ker to 1 where Position is modulo 1/T and the entry is nan, because of numerical instability for large Omega
        ker[np.isnan(ker) & (Position % (1/T) == 0)] = 1
        #set all other nan entries to 0
        ker[np.isnan(ker)] = 0
    return ker

def KL_kernel_Omega(
        KL,
        x,
        Omega,
        args=[]
        ):
    ret=KL(x, Omega, *args)
    ret[:,Omega==0]=1
    ret=Omega * ret
    # set for all Omega=0 to 1
    if len(args)==0:
        ret[:,Omega==0]=0
    else:
        ret[:,Omega==0]=2*args[0]
    return ret

def Di(
        KL,
        rhoi,
        delomega
        ):
    # Ensure both tensors are of the same data type (float32)
    KL = tf.cast(KL, dtype=tf.float32)  # Cast KL to float32
    rhoi = tf.cast(rhoi, dtype=tf.float32)  # Cast rhoi to float32
    delomega = tf.cast(delomega, dtype=tf.float32)  # Cast delomega to float32

    # Perform matrix multiplication
    dis = tf.matmul(KL, rhoi, transpose_b=True)  # Shape will be [Nt, batch_size]
    dis = tf.transpose(dis) #transpose to [batch_size, Nt]
    dis = dis * delomega  # Multiply by delomega
    return dis

class KadesFC(
        tf.keras.Model
        ):
    def __init__(
            self, 
            num_output_nodes: int, 
            **kwargs
            ):
        super(KadesFC, self).__init__(**kwargs)
        self.num_output_nodes = num_output_nodes
        self.relu1 = tf.keras.layers.Activation('relu')
        self.fc1 = tf.keras.layers.Dense(6700)  
        self.relu2 = tf.keras.layers.Activation('relu')
        self.fc2 = tf.keras.layers.Dense(12168) 
        self.relu3 = tf.keras.layers.Activation('relu')
        self.fc3 = tf.keras.layers.Dense(1024) 
        self.relu4 = tf.keras.layers.Activation('relu')
        self.fc4 = tf.keras.layers.Dense(num_output_nodes)

    def call(
            self, 
            inputs: tf.Tensor
            ) -> tf.Tensor:
        x = self.relu1(inputs)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        x = self.relu3(x)
        x = self.fc3(x)
        x = self.relu4(x)
        return self.fc4(x)
    
    def get_config(
            self
            ):
        config = super(KadesFC, self).get_config()
        config.update({"num_output_nodes": self.num_output_nodes})
        return config
    
    @classmethod
    def from_config(
            cls, 
            config
            ):
        num_output_nodes = config.pop('num_output_nodes')
        return cls(num_output_nodes=num_output_nodes, **config)

class KadesConv(
        tf.keras.Model
        ):
    def __init__(
            self,
            num_output_nodes: int, 
            **kwargs
            ):
        super(KadesConv, self).__init__(**kwargs)
        self.num_output_nodes = num_output_nodes
        # Conv layer 1
        self.hidden_layer1 = tf.keras.layers.Conv1D(64, 10, padding = 'same', activation = 'relu')
        # Conv layer 2
        self.hidden_layer2 = tf.keras.layers.Conv1D(256, 5, padding = 'same', activation = 'relu')
        # Flatten to manage the dimensions
        self.flatten_layer = tf.keras.layers.Flatten()
        # Squared hidden layer 
        self.hidden_layer3 = tf.keras.layers.Dense(4096, activation = 'relu')
        # Layer 4
        self.hidden_layer4 = tf.keras.layers.Dense(1024, activation = 'relu')
        # Output layer
        self.output_layer = tf.keras.layers.Dense(num_output_nodes, activation = 'relu')

    def call(
            self, 
            inputs: tf.Tensor
            ) -> tf.Tensor:
        inputs = tf.expand_dims(inputs, axis =-1)
        x = self.hidden_layer1(inputs)
        x = self.hidden_layer2(x)
        x = self.flatten_layer(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        return self.output_layer(x)
    
    def get_config(
            self
            ):
        config = super(KadesConv, self).get_config()
        config.update({"num_output_nodes": self.num_output_nodes})
        return config
    
    @classmethod
    def from_config(
            cls, 
            config
            ):
        num_output_nodes = config.pop('num_output_nodes')
        return cls(num_output_nodes=num_output_nodes, **config)

class SupervisedNN(
    tf.keras.Model
    ):
    def __init__(
            self, 
            num_output_nodes: int, 
            **kwargs
            ):
        super(SupervisedNN, self).__init__(**kwargs)
        self.num_output_nodes = num_output_nodes
        # Layer 1
        self.hidden_layer1 = tf.keras.layers.Dense(6700, activation = 'elu')
        # Layer 2
        self.hidden_layer2 = tf.keras.layers.Dense(12168, activation = 'elu')
        # Layer 3
        self.hidden_layer3 = tf.keras.layers.Dense(1024, activation = 'elu')
        # Output layer
        self.output_layer = tf.keras.layers.Dense(num_output_nodes, activation = 'softplus')

    def call(
            self, 
            inputs: tf.Tensor
            ) -> tf.Tensor:
        x = self.hidden_layer1(inputs)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        return self.output_layer(x)
    
    def get_config(
            self
            ):
        config = super(SupervisedNN, self).get_config()
        config.update({"num_output_nodes": self.num_output_nodes})
        return config
    
    @classmethod
    def from_config(
            cls, 
            config
            ):
        num_output_nodes = config.pop('num_output_nodes')
        return cls(num_output_nodes=num_output_nodes, **config)
    
class LossCalculator:
    def __init__(
            self, 
            model: tf.keras.Model=None,
            std=None,
            kernel: tf.Tensor=None,
            delomega: tf.Tensor =None,
            lambda_s_func: Callable[[int], float]=lambda x:0.0,
            lambda_l2_func: Callable[[int], float]=lambda x:0.0
            ):
        self.model = model
        self.std = std
        if self.std is None:
            self.std = tf.constant(1.0, dtype=tf.float32)
        else:
            self.std = tf.cast(self.std, dtype=tf.float32)
        self.kernel = kernel
        self.delomega = delomega
        self.lambda_s_func = lambda_s_func
        self.lambda_l2_func = lambda_l2_func

    def get_lambda_s(
            self, 
            epoch: int
            ) -> float:
        return self.lambda_s_func(epoch)
    
    def get_lambda_l2(
            self, 
            epoch: int
            ) -> float:
        return self.lambda_l2_func(epoch)

    def l2_regularization(
            self, 
            weights: List[tf.Tensor] = None
            ) -> tf.Tensor:
        if weights is None:
            weights = self.model.trainable_weights
        return tf.reduce_sum([tf.nn.l2_loss(w) for w in weights])
    
    def smoothness_loss(
            self, 
            rho: tf.Tensor = None
            ) -> tf.Tensor:
        return tf.reduce_sum(tf.square(rho[:, 1:] - rho[:, :-1]))
    
    def custom_loss(
            self, 
            y_pred: tf.Tensor, 
            weighting: tf.Tensor, 
            y_true: tf.Tensor = None
            ) -> tf.Tensor:
        weighting = tf.cast(tf.squeeze(weighting), dtype=tf.float32)
        y_true = tf.cast(tf.squeeze(y_true), dtype=tf.float32)
        y_pred = tf.cast(tf.squeeze(y_pred), dtype=tf.float32)
        weighting /= weighting[0]
        assert y_pred.shape == y_true.shape == weighting.shape, "Shape mismatch in loss calculation"
        chi_squared = tf.square((y_true - y_pred)/ weighting)
        return tf.reduce_mean(chi_squared)
    
    def rho_loss(
            self, 
            rho: tf.Tensor, 
            rho_true: tf.Tensor
            ) -> tf.Tensor:
        rho_true = tf.cast(tf.squeeze(rho_true), dtype=tf.float32)
        rho = tf.cast(tf.squeeze(rho), dtype=tf.float32)
        assert rho.shape == rho_true.shape, "Shape mismatch in rho loss calculation"
        return tf.reduce_mean(tf.square(rho - rho_true))

    def total_loss(
            self, 
            epoch: int,
            rho: tf.Tensor, 
            y_true: tf.Tensor,
            err: tf.Tensor, 
            y_pred: tf.Tensor = None,
            rho_true: tf.Tensor = None
            ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        y_pred = Di(self.kernel, rho, self.delomega)
        main_loss = self.custom_loss(y_pred, err, y_true)
        smooth_loss = self.smoothness_loss(rho)
        l2_loss = self.l2_regularization()
        rho_loss = self.rho_loss(rho, rho_true) if rho_true is not None else 0.0
        total_loss_value = rho_loss
        #total_loss_value = main_loss + self.get_lambda_s(epoch) * smooth_loss + self.get_lambda_l2(epoch) * l2_loss + rho_loss
        return total_loss_value, [main_loss, self.get_lambda_s(epoch)*smooth_loss, self.get_lambda_l2(epoch)*l2_loss, rho_loss] ### maybe fix the passing here at some point

class networkTrainer:
    def __init__(
            self, 
            model: tf.keras.Model, 
            optimizer: tf.keras.optimizers.Optimizer, 
            loss_calculator: LossCalculator
            ):
       self.model = model
       self.optimizer = optimizer
       self.loss_calculator = loss_calculator

    @tf.function(reduce_retracing=True)
    def train_step(
            self, 
            epoch: int, 
            corr: tf.Tensor, 
            err: tf.Tensor, 
            rho_true: tf.Tensor = None
            ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        with tf.GradientTape() as tape:
            rho_pred = self.model(corr)
            total_loss_value, individual_losses = self.loss_calculator.total_loss(epoch, rho=rho_pred, y_true = corr, err=err, rho_true = rho_true)
        # Compute gradients and update weights
        gradients = tape.gradient(total_loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))    
        return total_loss_value, individual_losses
    
    def test_step(
            self, 
            epoch: int, 
            corr: tf.Tensor, 
            err: tf.Tensor, 
            rho_true: tf.Tensor = None
            ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        rho = self.model(corr)
        total_loss_value, individual_losses = self.loss_calculator.total_loss(epoch, rho=rho, y_true = corr, err= err, rho_true = rho_true)
        return total_loss_value, individual_losses

    def trainloop(
            self, 
            dat: tf.data.Dataset,
            epoch: int, 
            verbose: bool = False
            ):
        train_losses = []
        train_losses_ind = []
        step = 0

        for step, (X, y, z) in enumerate(dat):
            plt.clf() 
            plt.plot(X[0])
            plt.plot(self.model(y)[0])
            plt.ylim(-0.1,2.5)
            plt.savefig("debug.png")
            total_loss_value, individual_losses = self.train_step(epoch, corr=y, err=z, rho_true = X)
            train_losses.append(total_loss_value.numpy())
            for i in range(len(individual_losses)):
                individual_losses[i] = individual_losses[i].numpy()
            train_losses_ind.append(individual_losses)
            if verbose and step % 50 == 0:
                print(f'Batch {step}/{len(dat)}, Loss: {total_loss_value}, main: {individual_losses[0]}, smooth: {individual_losses[1]}, l2: {individual_losses[2]}, rho: {individual_losses[3]}')
        return train_losses, train_losses_ind

    def testloop(
            self, 
            dat: tf.data.Dataset,
            epoch: int, 
            verbose: bool = False
            ):
        test_losses = []
        test_losses_ind = []
        for X,y,z in dat:
            total_loss_value, individual_losses = self.test_step(epoch, corr=y, err=z, rho_true = X)
            test_losses.append(total_loss_value)
            test_losses_ind.append(individual_losses)
        if verbose:
            print(f'Validation loss: {total_loss_value}')
        return test_losses, test_losses_ind

    def train(
            self, 
            num_epochs: int, 
            train_dat: tf.data.Dataset,
            test_dat: tf.data.Dataset,
            verbose: bool = False,
            start_epoch: int = 0
            ) -> Tuple[List[tf.Tensor], List[List[tf.Tensor]],List[tf.Tensor], List[List[tf.Tensor]]]:
        t_losses, v_losses = [],[]
        t_individual_losses, v_individual_losses = [],[]
        net_num_epochs = num_epochs - start_epoch

        if verbose:
            print(f'Training for {net_num_epochs} epochs')
            start_time = time.time()
        for epoch in range(start_epoch, start_epoch + num_epochs):
            if verbose:
                print(f'\nStart of epoch %d' %(epoch,))
            train_losses, train_losses_ind = self.trainloop(train_dat, epoch, verbose)
            val_losses, val_losses_ind = self.testloop(test_dat, epoch, verbose)
            t_losses.append(tf.reduce_sum(train_losses))
            t_individual_losses.append(tf.reduce_sum(train_losses_ind, axis=0))
            v_losses.append(tf.reduce_sum(val_losses))
            v_individual_losses.append(tf.reduce_sum(val_losses_ind, axis=0))

        return t_losses, t_individual_losses, v_losses, v_individual_losses
    
#Interface and runner classes
    
@dataclass
class networkParameters:
    lambda_s: List = field(default_factory=list)
    lambda_l2: List = field(default_factory=list)
    epochs: List = field(default_factory=list)
    learning_rate: List = field(default_factory=list)
    batch_size: int = 128
    errorWeighting: bool = False
    networkStructure: str = ""

    def __post_init__(
            self
            ):
        # Ensure all entries in lambda_s, lambda_l2 and learning_rate are floats
        if not all(isinstance(item, float) for item in self.lambda_s):
            raise ValueError("All entries in lambda_s must be floats.")
        if not all(isinstance(item, float) for item in self.lambda_l2):
            raise ValueError("All entries in lambda_l2 must be floats.")
        
        # Ensure all entries in epochs are integers
        if not all(isinstance(item, int) for item in self.epochs):
            raise ValueError("All entries in epochs must be integers.")

class supervisedFit:
    def __init__(
            self,networkParameters:networkParameters
            ):
        self.lambda_s=networkParameters.lambda_s
        self.lambda_l2=networkParameters.lambda_l2
        self.epochs=networkParameters.epochs
        self.learning_rate=networkParameters.learning_rate
        self.batch_size=networkParameters.batch_size
        self.errorWeighting=networkParameters.errorWeighting
        self.networkStructure=networkParameters.networkStructure

    def initKernel(
            self,
            extractedQuantity: str,
            finiteT_kernel: bool,
            Nt: int,
            x: np.ndarray,
            omega: np.ndarray
            ):
        if extractedQuantity=="RhoOverOmega" and finiteT_kernel:
            kernel=KL_kernel_Omega(KL_kernel_Position_FiniteT,x,omega,args=(1/Nt,))
        elif extractedQuantity=="RhoOverOmega" and finiteT_kernel==False:
            kernel=KL_kernel_Omega(KL_kernel_Position_Vacuum,x,omega)
        elif extractedQuantity=="Rho" and finiteT_kernel:
            kernel=KL_kernel_Position_FiniteT(x,omega,1/Nt)
        elif extractedQuantity=="Rho" and finiteT_kernel==False:
            kernel=KL_kernel_Position_Vacuum(x,omega)
        else:
            raise ValueError("Invalid choice spectral function target")
        return kernel

    def get_data(
            self, 
            file_dir: str
            )-> dict:
        with open(file_dir, 'rb') as f:
            file = np.load(f, allow_pickle=True)
        return file
            
    def fit_known(
            self, 
            x: np.ndarray, 
            error: np.ndarray, 
            correlator: np.ndarray, 
            finiteT_kernel: bool,
            Nt: int, 
            omega: np.ndarray, 
            model_file: str,
            extractedQuantity: str = "RhoOverOmega", 
            ) -> Tuple[np.ndarray, np.ndarray]:
        kernel = self.initKernel(extractedQuantity, finiteT_kernel, Nt, x, omega)
        del_omega = omega[1] - omega[0]
        errorWeight = error if self.errorWeighting else np.ones(len(x))
        if self.networkStructure == "SupervisedNN":
            model = load_model(model_file, custom_objects = {'SupervisedNN': SupervisedNN, 'LossCalculator': LossCalculator})
        if self.networkStructure == "KadesFC":
            model = load_model(model_file, custom_objects = {'KadesFC': KadesFC, 'LossCalculator': LossCalculator})
        if self.networkStructure == "KadesConv":
            model = load_model(model_file, custom_objects = {'KadesConv': KadesConv, 'LossCalculator': LossCalculator})
        correlator = tf.reshape(correlator, (1,len(correlator)))

        lossCalc = LossCalculator(
            model=model,
            kernel=kernel,
            delomega=del_omega,
            std=errorWeight,
            lambda_s_func=lambda x: self.lambda_s[0],
            lambda_l2_func=lambda x: self.lambda_l2[0]
        )
        if len(self.learning_rate) == 1:
            lr = self.learning_rate[0]
        else:
            lr = tf.keras.optimizers.schedules.PolynomialDecay(self.learning_rate[0], self.learning_rate[2], self.learning_rate[1], power=0.5)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        trainer = networkTrainer(model, optimizer, lossCalc)
        total_loss, individual_loss = trainer.test_step(0, correlator, error)
        spectralFunction = model(correlator)
        return np.squeeze(spectralFunction), total_loss, individual_loss

    def fitCorrelator(
            self, 
            x: np.ndarray, 
            error: np.ndarray, 
            correlator: np.ndarray, 
            finiteT_kernel: bool, 
            Nt: int, 
            omega: np.ndarray, 
            train_file: str, 
            validation_file: str,
            extractedQuantity: str = "RhoOverOmega", 
            verbose: bool = True
            ) -> Tuple[np.ndarray, np.ndarray]:

        kernel = self.initKernel(extractedQuantity, finiteT_kernel, Nt, x, omega)
        del_omega = omega[1] - omega[0]
        errorWeight = error if self.errorWeighting else np.ones(len(x))
        
        if self.networkStructure == "SupervisedNN":
            model = SupervisedNN(num_output_nodes=len(omega))
        if self.networkStructure == "KadesFC":
            model = KadesFC(num_output_nodes=len(omega))
        if self.networkStructure == "KadesConv":
            model = KadesConv(num_output_nodes=len(omega))
        if self.networkStructure != "SupervisedNN" and self.networkStructure != "KadesFC" and self.networkStructure != "KadesConv":
            raise ValueError("Invalid choice of network")
        

        if len(self.learning_rate) == 1:
            lr = self.learning_rate[0]
        else:
            lr = tf.keras.optimizers.schedules.PolynomialDecay(self.learning_rate[0], self.learning_rate[2], self.learning_rate[1], power=0.5)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        training_total_loss_history = []
        training_loss_history = []
        validation_total_loss_history = []
        validation_loss_history = []
        
        lossCalc = LossCalculator(
            model=model,
            kernel=kernel,
            delomega=del_omega,
            std=errorWeight,
            lambda_s_func=lambda x: self.lambda_s[0],
            lambda_l2_func=lambda x: self.lambda_l2[0]
        )

        train_raw = self.get_data(train_file)
        validation_raw = self.get_data(validation_file)
        if verbose:
            print("Loaded the dataset")
        #assume one file for training and validation each
        train_fcts = tf.data.Dataset.from_tensor_slices(np.array([d['fct'] for d in train_raw]))
        train_corrs = tf.data.Dataset.from_tensor_slices(np.array([d['corr'] for d in train_raw]))
        train_errs = tf.data.Dataset.from_tensor_slices(np.array([d['noise'] for d in train_raw]))

        validation_fcts = tf.data.Dataset.from_tensor_slices(np.array([d['fct'] for d in validation_raw]))
        validation_corrs = tf.data.Dataset.from_tensor_slices(np.array([d['corr'] for d in validation_raw]))
        validation_errs = tf.data.Dataset.from_tensor_slices(np.array([d['noise'] for d in validation_raw]))

        train_dat = tf.data.Dataset.zip((train_fcts, train_corrs, train_errs))
        validation_dat = tf.data.Dataset.zip((validation_fcts, validation_corrs, validation_errs))
    
        train_dat = train_dat.shuffle(1000).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        validation_dat = validation_dat.shuffle(1000).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        #maybe as a to do: checkpointing
        trainer = networkTrainer(model, optimizer, lossCalc)
        for lambda_s, lambda_l2, learning_rate, epochs in zip(self.lambda_s, self.lambda_l2, self.learning_rate, self.epochs):
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            lossCalc.lambda_s_func = lambda x: lambda_s
            lossCalc.lambda_l2_func = lambda x: lambda_l2
            trainer.optimizer = optimizer
            t_total_loss_history_tmp, t_loss_history_tmp, v_total_loss_history_tmp, v_loss_history_tmp = trainer.train(
                epochs, train_dat, validation_dat, verbose=verbose
                )
            training_total_loss_history.extend(t_total_loss_history_tmp)
            training_loss_history.extend(t_loss_history_tmp)
            validation_total_loss_history.extend(v_total_loss_history_tmp)
            validation_loss_history.extend(v_loss_history_tmp)
            if verbose:
                print("-" * 40)
        #reshape the input data to respect batch_size preferences of the network
        correlator = tf.reshape(correlator, (1,len(correlator)))
        spectralFunction = model(correlator)
        modelname = '{}_Nt{}_{}.keras'.format(self.networkStructure, Nt, train_file[-10:-4])
        model.save(modelname)
        training_total_loss_history = np.expand_dims(training_total_loss_history, axis=-1)
        train_loss = np.concatenate((np.squeeze(training_loss_history), training_total_loss_history), axis = 1)
        validation_total_loss_history = np.expand_dims(validation_total_loss_history, axis=-1)
        val_loss = np.concatenate((np.squeeze(validation_loss_history),validation_total_loss_history), axis=1)
        return np.squeeze(spectralFunction), np.average(val_loss, axis=1), np.average(train_loss, axis=1), modelname
    
class ParameterHandler:
    def __init__(
            self, 
            paramsDefaultDict: dict
            ):
        self.allowed_params = paramsDefaultDict.keys()
        self.params = paramsDefaultDict
        
    def load_from_json(
            self, 
            config_path: str
            ) -> None:
        if config_path:
            with open(config_path, 'r') as f:
                data = json.load(f)
            for name in self.allowed_params:
                if name in data:
                    self.params[name] = data[name]

    def override_with_args(
            self, 
            args: argparse.Namespace
            ) -> None:
        for name in self.allowed_params:
            val = getattr(args, name, None)
            if val is not None:
                self.params[name] = val

    def check_parameters(
            self
            ) -> None:
        for name in self.allowed_params:
            if name == "outputFile" and self.params[name] is None:
                continue
            if name not in self.params or self.params[name] is None:
                raise ValueError(f"Parameter '{name}' is not set.")

    def load_params(
            self, 
            config_path: str, 
            args: argparse.Namespace
            ) -> None:
        self.load_from_json(config_path)
        self.override_with_args(args)
        self.check_parameters()

    def get_params(
            self
            ) -> dict:
        return self.params
    
    def getNetworkParams(
            self
            ) -> networkParameters:
        return networkParameters(
            lambda_s=self.params["lambda_s"],
            lambda_l2=self.params["lambda_l2"],
            epochs=self.params["epochs"],
            learning_rate=self.params["learning_rate"],
            batch_size=self.params["batch_size"],
            errorWeighting=self.params["errorWeighting"],
            networkStructure=self.params["networkStructure"]
        )
    
    def get_extractedQuantity(
            self
            ) -> str:
        return self.params["extractedQuantity"]
    
    def get_correlator_file(
            self
            ) -> str:
        return os.path.abspath(self.params["correlatorFile"])
    
    def get_training_validation_files(
            self
            )-> str:
        return os.path.abspath(self.params["trainingFile"]), os.path.abspath(self.params["validationFile"])

    def get_verbose(
            self
            ) -> bool:
        return self.params["verbose"]
    
    def get_correlator_cols(
            self
            ) -> List[int]:
        correlator_cols = self.params["correlatorCols"]
        if isinstance(correlator_cols, list):
            return correlator_cols
        if isinstance(correlator_cols, int):
            return [correlator_cols]
        if isinstance(correlator_cols, str) and ':' in correlator_cols:
            start_str, end_str = correlator_cols.split(':')
            start = int(start_str) if start_str else None
            end = int(end_str) if end_str else None
            return list(range(start if start is not None else 0, end + 1 if end is not None else len(np.loadtxt(self.params["correlatorFile"], max_rows=1))))
        if isinstance(correlator_cols, str) and correlator_cols.isdigit():
            return [int(correlator_cols)]
        if not correlator_cols:
            return []
        raise ValueError("correlator_cols must be an integer index, list of indices, or a string with a range (e.g., '6:10', '6:', ':10', ':').")
    
class FitRunner:
    def __init__(
            self, 
            parameterHandler: ParameterHandler
            ):
        self.parameterHandler = parameterHandler
        self.net_params = self.parameterHandler.getNetworkParams()
        self.fitter = supervisedFit(self.net_params)
        self.x, self.mean, self.error, self.correlators = self.extractColumns(
            self.parameterHandler.get_correlator_file(),
            self.parameterHandler.get_params()["xCol"],
            self.parameterHandler.get_params()["meanCol"],
            self.parameterHandler.get_params()["errorCol"],
            self.parameterHandler.get_correlator_cols()
        )
        self.omega = np.linspace(
            self.parameterHandler.get_params()["omega_min"],
            self.parameterHandler.get_params()["omega_max"],
            self.parameterHandler.get_params()["omega_points"]
        )
        self.trainfile, self.validationfile = self.parameterHandler.get_training_validation_files()
        self.finiteT_kernel = self.parameterHandler.get_params()["FiniteT_kernel"]
        self.verbose = self.parameterHandler.get_verbose()
        self.extractedQuantity = self.parameterHandler.get_extractedQuantity()
        self.Nt = self.parameterHandler.get_params()["Nt"] or len(self.x)
        self.outputDir = os.path.abspath(self.parameterHandler.get_params()["outputDir"])
        self.outputFile = self.parameterHandler.get_params()["outputFile"] or f"{self.extractedQuantity}_{os.path.basename(self.parameterHandler.get_correlator_file())}"

    def extractColumns(
            self, 
            file: str, 
            x_col: int, 
            mean_col: int, 
            error_col: int, 
            correlator_cols: List[int]
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data = np.loadtxt(file)
        x = data[:, x_col]
        mean = data[:, mean_col]
        error = data[:, error_col]
        correlator = data[:, correlator_cols]
        return x, mean, error, correlator    

    def run_fit(
            self, 
            fittedQuantity, 
            messageString,
            results: List[np.ndarray], 
            validation_loss_histories: List[np.ndarray],
            training_loss_histories: List[np.ndarray]
            ) -> None:
        start_time = time.time()
        print("=" * 40)
        print(messageString)
        print("=" * 40)
        sf, v_loss_history, t_loss_history, modelname = self.fitter.fitCorrelator(
            self.x,
            self.error,
            fittedQuantity,
            self.finiteT_kernel,
            self.Nt,
            self.omega,
            self.trainfile,
            self.validationfile,
            extractedQuantity=self.extractedQuantity,
            verbose=self.verbose
        )
        if self.verbose:
            print("-" * 40)
            print(f"Training time: {time.time() - start_time:.2f} seconds")
        results.append(sf)
        training_loss_histories.append(t_loss_history)
        validation_loss_histories.append(v_loss_history)
        return modelname

    def pred_res(
            self, 
            fittedQuantity, 
            messageString, 
            results: List[np.ndarray], 
            loss_histories: List[np.ndarray], 
            model_file: str
            ) -> None:
        print("=" * 40)
        print(messageString)
        print("=" * 40)
        spectralFunction, total_loss, individual_loss = self.fitter.fit_known(
            self.x,
            self.error,
            fittedQuantity,
            self.finiteT_kernel,
            self.Nt,
            self.omega,
            model_file,
            extractedQuantity=self.extractedQuantity,
        )
        if self.verbose:
            print("-" * 40)
        results.append(np.squeeze(spectralFunction))
        loss_histories.append(np.insert(np.array(individual_loss), 0, np.array(total_loss)))
    
    def run_fits(
            self
            ) -> Tuple[np.ndarray, np.ndarray]:
        results = []
        validation_loss_histories = []
        training_loss_histories = []
        pred_loss_histories = []
        if self.correlators.ndim == 1:
            self.correlators = np.array([self.correlators])
        else:
            self.correlators = self.correlators.T
        n_correlators = self.correlators.shape[0]
        if self.parameterHandler.get_params()["eval_model"]:
            self.pred_res(self.mean, "Fitting mean correlator", results, pred_loss_histories, self.parameterHandler.get_params()["model_file"])
            for i, corr in enumerate(self.correlators):
                self.pred_res(corr, f"Fitting correlator sample {i+1}/{n_correlators}", results, pred_loss_histories, self.parameterHandler.get_params()["model_file"])
        else:
            model_name = self.run_fit(self.mean, "Fitting mean correlator", results, validation_loss_histories, training_loss_histories)
            for i, corr in enumerate(self.correlators):
                self.pred_res(corr, f"Fitting correlator sample {i + 1}/{n_correlators}", results, pred_loss_histories, model_name)
        np.array(validation_loss_histories)
        np.array(training_loss_histories)
        np.array(pred_loss_histories)
        return np.array(results), np.squeeze(training_loss_histories), np.squeeze(validation_loss_histories), np.squeeze(pred_loss_histories)

    def calculate_mean_error(
            self, 
            mean: np.ndarray, 
            samples: np.ndarray, 
            errormethod: str = "jackknife"
            ) -> np.ndarray:
        N = len(samples)
        fac = N - 1 if errormethod == "jackknife" else 1
        if errormethod not in ["jackknife", "bootstrap"]:
            raise ValueError("Invalid choice of error estimation method")
        return np.sqrt(fac / N * np.sum((samples - mean) ** 2, axis=0))

    def save_results(
            self,
            mean: np.ndarray, 
            error: np.ndarray, 
            samples: np.ndarray, 
            training_loss_history: np.ndarray, 
            validation_loss_history: np.ndarray, 
            pred_loss_history: np.ndarray
            ) -> None:
        header = "Omega " + self.extractedQuantity + "_mean"
        if samples is not None and error is not None:
            header += f" {self.extractedQuantity}_error"
            for i in range(len(samples)):
                header += f" {self.extractedQuantity}_sample_{i}"
            writeData = np.column_stack((self.omega, mean, error, samples.T))
        else:
            writeData = np.column_stack((self.omega, mean))
        np.savetxt(os.path.join(self.outputDir, self.outputFile), writeData, header=header)
        if self.parameterHandler.get_params()["saveParams"]:
            self.save_params(self.parameterHandler.get_params(), os.path.join(self.outputDir, self.outputFile + ".params"))
        if self.parameterHandler.get_params()["saveLossHistory"]:
            self.save_loss_history(training_loss_history, os.path.join(self.outputDir, self.outputFile + ".trainloss.dat"))
            self.save_loss_history(validation_loss_history, os.path.join(self.outputDir, self.outputFile + ".valloss.dat"))
            self.save_loss_history(pred_loss_history, os.path.join(self.outputDir, self.outputFile + ".predloss.dat"))

    def save_loss_history(
            self, 
            loss_history: np.ndarray, 
            outputFile: str
            ) -> None:
        header = "mean_total_loss mean_main_loss mean_smoothness_loss mean_l2_loss"
        for i in range(len(loss_history[1:])):
            header += f" sample_{i}_total_loss sample_{i}_main_loss sample_{i}_smoothness_loss sample_{i}_l2_loss"
        reshaped_loss_history = np.squeeze(loss_history)
        np.savetxt(outputFile, reshaped_loss_history, header=header)

    def save_params(
            self, 
            params: dict, 
            outputFile: str
            ) -> None:
        with open(outputFile + '.json', 'w') as f:
            json.dump(params, f, indent=4)

def initializeArgumentParser(
        paramsDefaultDict: dict
        ) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="supervisedml",
        description="Train and fit spectral functions to provided correlators using a supervised neural network."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="",
        help="Path to JSON configuration file"
    )
    
    for name, default in paramsDefaultDict.items():
        typeArg=type(default)
        typeString=typeArg.__name__
        if typeArg==list:
            nargsArg='+'
            typeArg=type(default[0])
            typeString=f"List of {typeArg.__name__}"
        else:
            nargsArg=None
        parser.add_argument(
            f"--{name}",
            type=typeArg,
            nargs=nargsArg,
            # default=default,
            help=f"Value for parameter '{name}'"
        )
    return parser

def main(
        paramsDefaultDict
        ):
    parser=initializeArgumentParser(paramsDefaultDict)
    args = parser.parse_args()
    parameterHandler = ParameterHandler(paramsDefaultDict)
    parameterHandler.load_params(args.config,args)

    if parameterHandler.get_verbose():
        print("*"*40)
        print("Running fits with the following parameters:")
        pprint.pprint(parameterHandler.get_params())

    fitRunner = FitRunner(parameterHandler)
    results, training_loss_histories, validation_loss_histories, pred_loss_histories = fitRunner.run_fits()
    mean = results[0]
    if len(results)>1:
        samples = results[1:]
        error = fitRunner.calculate_mean_error(samples,mean,parameterHandler.get_params()["errormethod"])
    else:
        samples = None
        error = None
    fitRunner.save_results(mean,error,samples,training_loss_histories,validation_loss_histories, pred_loss_histories)



paramsDefaultDict = {
    #choice of SupervisedNN, KadesFC, UnsupervisedNN, Gaussian, MEM
    "networkStructure": "SupervisedNN",
    #NetworkParams (Ai specrec)
    "lambda_s": [1e-5],
    "lambda_l2": [1e-8],
    "epochs": [100],
    "learning_rate": [1e-4],
    "errorWeighting": True,
    #Supervised specific
    "batch_size": 128,
    "trainingFile": "",
    "validationFile": "",
    "eval_model": False,
    "model_file": "",
    #Correlator/Rho params
    "omega_min": 0,
    "omega_max": 10,
    "omega_points": 500,
    "Nt": 0,
    "extractedQuantity": "RhoOverOmega",
    "FiniteT_kernel": True,
    "correlatorFile": "",
    "xCol": 0,
    "meanCol": 1,
    "errorCol": 2,
    "correlatorCols": "3:",
    "errormethod": "jackknife",
    #General Params
    "saveParams": False,
    "saveLossHistory": False,
    "verbose": False,
    "outputFile": "",
    "outputDir": ""

}



if __name__ == "__main__":
    main(paramsDefaultDict)
