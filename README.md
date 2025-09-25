# spectral_reconstruction_methods

## Overview

This python program unifies different methods to extract the spectral function $\rho(\omega)$ (or $\rho(\omega)/\omega$) from a given correlator $D(x)$.
Namely these methods are:
- MEM (Maximum entropy method, as in: hep-lat/0011040 or https://doi.org/10.1007/BF02427376)
- GPR (Gaussian process regression, as implemented in: fredipy, see: https://github.com/JonasTurnwald/fredipy and references therein)
- Supervised machine learning (Novel machine learning technique, extended from the basis of: https://doi.org/10.1103/PhysRevD.102.096001)
- Unsupervised machine learning (Novel machine learning technique, extended from the basis of: https://doi.org/10.1103/PhysRevD.106.L051502)

The neural networks are trained to minimize the difference between the output of the neural network and the input correlator, while also penalizing a non-smooth spectral function and large weights in the network.

The program can be run from the command line using the following command:
```bash
python neuralFit.py --config config.json
```
where `config.json` is a JSON file containing the parameters for the training (see below). All parameters specified in the table below can also be passed as command line arguments. For example, the parameter `lambda_s` can be passed as `--lambda_s 1e-5`.


### Possible reconstruction methods

- `MEM`: An implementation of MEM.
- `Gaussian`: A pipeline to an application of GPR, linking to the library fredipy.
- `SupervisedNN`: A neural network that trains on certain input data and predicts the spectral function based on the learned connections.
- `UnsupervisedNN`: A neural network using unsupervised learning developed in large by Laurin and Simran.

All of the above take a constant input and and outputs all $\rho(\omega_i)$.


### Loss functions (specific for both AI methods)

The loss function is a sum of the following terms:


#### Correlator loss
This term penalizes the difference between the output of the neural network and the input correlator. The coupling for this term has a constant value of `1`.
The correlator loss is defined as
```math
\text{Correlator loss} = \sum_{i=1} \left( \frac{1}{\sigma_i} \left( D(x_i) - D_{\text{input}}(x_i) \right)^2 \right),
```
where $D(x_i)$ is the output of the neural network, $D_{\text{input}}(x_i)$ is the input correlator, and $\sigma_i$ is the error of the input correlator.
If the parameter `errorWeighting` is set to `False`, $\sigma_i = 1$.

#### Smoothness loss
This term penalizes a non-smooth spectral function. The coupling for this term can be specified using the `lambda_s` parameter.
The smoothness loss is defined as
```math
\text{Smoothness loss} = \lambda_s \sum_{i=1} (\rho(\omega_{i+1}) - \rho(\omega_i))^2,
```
where $\lambda_s$ is the coupling for the smoothness loss contribution.


#### L2 loss
This term penalizes large weights in the network and is meant to combat overfitting . The coupling for this term can be specified using the `lambda_l2` parameter.
The L2 loss is defined as
```math
\text{L2 loss} = \lambda_{l_2} \sum_{i} w_i^2,
```
where $\lambda_{l2}$ is the coupling for the L2 loss contribution and $w$ are the weights of the network.


### Training stages (specific to UnsupervisedNN)

The training can be done in multiple stages. The number of epochs, learning rate, and loss function parameters can be different for each stage. The training will be done in the order of the parameters provided.

### Correlator input file format

The input file should be a text file with columns separated by spaces. The first column should be the $x$ values, the second column should be the mean values, and the third column should be the error values. The rest of the columns should be the statistical samples of the correlator. The columns can be specified using the `xCol`, `meanCol`, `errorCol`, and `correlatorCols` parameters.

### Extracted quantities

The extracted quantity can be either $\rho(\omega)$ or $\rho(\omega)/\omega$. The extracted quantity can be specified using the `extractedQuantity` parameter.

### Error methods

The error method can be either jackknife or bootstrap. The error method can be specified using the `errormethod` parameter.
The errors are calculated from fitting the statistical samples of the correlator specified by the `correlatorCols` parameter.

### Output

The output of the training can be saved to a file. The output file can be specified using the `outputFile` parameter. The output directory can be specified using the `outputDir` parameter.
The path to the output file will be `outputDir/outputFile` and can be relative to the current working directory.
If the `outputFile` parameter is set to `null` or not specified, the output file will be named according to the correlator file and the extracted quantity.
The first column of the output file will be the frequency values $\omega$, the second column is the extracted quantity from the mean correlator, the third column is the error of the extracted quantity as calculated by the error method, and the rest of the columns are the extracted quantities from the statistical samples of the correlator.

If the `saveParams` parameter is set to `True`, the parameters used for the training will be saved to a file with the name `outputFile.params.json`, where `outputFile` is the name of the output file.

If the `saveLossHistory` parameter is set to `True`, the loss history will be saved to a file with the name `outputFile.loss.dat`. The columns of the file are the epoch number, the total loss, the correlator loss, the smoothness loss, and the L2 loss for the fit of the mean correlator. The following columns are these loss contributions for each of the fitted statistical samples.




## Parameters

The following parameters can be specified in the JSON file or as command line arguments.


| Parameter            | Default          | Possible Values | Purpose/Comment |
|----------------------|------------------|------------------------------------------------------|------|
| **Method**           | `"UnsupervisedNN"`   | `"UnsupervisedNN"`, `"SupervisedNN"`,`"MEM"`,`"Gaussian"`  | Method to use. |
| **lambda_s**         | `[1e-5]`         | Any float                | Coupling for the smoothness loss contribution |
| **lambda_l2**        | `[1e-8]`         | Any float                | Coupling for the L2 loss contribution |
| **epochs**           | `[100]`          | Any integer                 | Number of epochs |
| **learning_rate**    | `[1e-4]`         | Any float or list of floats               | Learning rate, if list of floats and `"SupervisedNN"`, a polynomial rearning rate scheduler is used. The syntax should then be `[lr_start, lr_end, epoch]` where epoch is an integer where the learning rate should freeze at the end value. |
| **errorWeighting**   | `True`           | `True`, `False`                   | Use error weighting for the correlator loss |
| **width**            | `[32, 32, 32]`   | List of integers                  | Specific to `"UnsupervisedNN"`: Structure of the neural network. The length of the list sets the number of layers, while the values set the widths of the layers.  |
| **batch_size**        | `128`              | Any integer                         | Specific to `"SupervisedNN"`: The batch size for the training and validation sets. |
| **create_data**        | `False`              | `True`,`False`                        | Specific to `"SupervisedNN"`: Wether to create new training and validation sets |
| **data_noise**        | `10e-5`              | Any float                         | Specific to `"SupervisedNN"`: The noise on the created data. |
| **trainingFile**        | `""`              | Any string                         | Specific to `"SupervisedNN"`: The file location of the training set |
| **validationFile**        | `""`              | Any string                         | Specific to `"SupervisedNN"`: The file location of the validation set |
| **eval_model**        | `False`              | `True`,`False`                         | Specific to `"SupervisedNN"`: If there is already a trained model available, which is to be evaluated on data. |
| **model_file**        | `""`              | Any string                         | Specific to `"SupervisedNN"`: The file location of the model to be evaluated. |
| **optimizer**        | `False`              | `True`, `False`                         | Specific to `"Gaussian"`: Wether the kernel parameters should be optimized with L-BFGS-B. |
| **variance**        | `0.3`              | Any float                         | Specific to `"Gaussian"`: The variance of the kernel |
| **lengthscale**        | `0.4`              | Any float                         | Specific to `"Gaussian"`: The lengthscale of the kernel |
| **alpha_min**        | `0`              | Any float                         | Specific to `"MEM"`:The lower bound of the alpha range |
| **alpha_max**        | `10`              | Any float                         | Specific to `"MEM"`: The upper bound of the alpha range |
| **alpha_points**        | `64`              | Any float                         | Specific to `"MEM"`: The number of points in the alpha range |
| **default_model**        | `"constant"`              | `"constant"`,`"quadratic"`,`"exact"`, `"file"`              | Specific to `"MEM"`: The default model to be used. The values should be in the second column. |
| **default_model_file**        | `""`              | Any string path              | Specific to `"MEM"`: The file path to the default model. |
| **omega_min**        | `0`              | Any float                         | The lower bound of the frequency range |
| **omega_max**        | `10`             | Any float                         | The upper bound of the frequency range |
| **omega_points**     | `500`            | Any integer                       | The number of points in the frequency range |
| **Nt**               | `0`              | Any integer                       | The temporal extent of the lattice. If `0`, the program will use the number of rows in the input file. Beware that this will cause problems if only a range of the correlator is fitted. |
| **extractedQuantity**| `"RhoOverOmega"` | `"RhoOverOmega"`, `"Rho"`         | The extracted quantity. Can be either $\rho(\omega)$ or $\rho(\omega)/\omega` |
| **FiniteT_kernel**   | `True`           | `True`, `False`                   | Use the finite temperature kernel for the correlator or the vaccum kernel |
| **multiFit**         | `False`          | `True`, `False`                   | If set to `True`, all statistical samples of the correlator will be fitted together. No error analysis is conducted! |
| **correlatorFile**   | `""`             | Any string path                   | Path to the input correlator file. Can be relative to the current working directory. |
| **xCol**             | `0`              | Any integer                       | Specifies the column of the input file that contains the $x$ values (or $\tau$ values). |
| **meanCol**          | `1`              | Any integer                       | Specifies the column of the input file that contains the mean correlator |
| **errorCol**         | `2`              | Any integer                       | Specifies the column of the input file that contains the error of the correlator |
| **correlatorCols**   | `"3:"`           | Integer, List of integers or range string | Specifies the columns of the input file that contain the statistical samples of the correlator. Several formats can be used. One integer specifies a single column, a list of integers specifies multiple columns, and a range string specifies a range of columns. The range string should be in the format `start:end` where `start` and `end` are integers. The range is inclusive. Optionally `start` or `end` can be empty to include all columns from the start or to the end. If set to `""`, no statistical samples are used to calculate the error and no error is column is given in the output. |
| **errormethod**      | `"jackknife"`    | `"jackknife"`, `"bootstrap"`      | The error method to use for the correlator. Can be either jackknife or bootstrap |
| **saveParams**       | `False`          | `True`, `False`                   | Save the parameters used for the training to a file |
| **saveLossHistory**  | `False`          | `True`, `False`                   | Save the loss history to a file |
| **verbose**          | `False`          | `True`, `False`                   | Print additional information during training |
| **outputFile**       | `""`             | Any string                        | Name of the output file. If set to `null`, the output file will be named according to the correlator file and the extracted quantity |
| **outputDir**        | `''`             | Any string                        | Directory where the output files will be saved. The path can be relative to the current working directory. |
| **cluster**        | `False`             | `True`, `False`                   | Flag for use on a cluster. |
| **cluster**        | `""`             | Any string                   | Directory of the filestructure on the cluster. This might be different from the working directory. |

## Passing Parameters

The parameters can be passed to the program using a JSON file or as command line arguments.
The parameters given as command line arguments will overwrite the parameters given in the JSON file.

### JSON
Create a JSON file (e.g., `params.json`):
```json
{
    "Method": "UnsupervisedNN",
    "lambda_s": [1e-6],
    "lambda_l2": [1e-4],
    "epochs": [500],
    "learning_rate": [1e-6],
    "errorWeighting": true,
    "width": [32,32,32],
    "batch_size": 128,
    "create_data": false,
    "data_noise": 10e-2,
    "trainingFile": "",
    "validationFile": "",
    "eval_model": false,
    "model_file": "",
    "optimizer": false,
    "variance": 0.3,
    "lengthscale": 0.4,
    "alpha_min": 1e-6,
    "alpha_max": 1000,
    "alpha_points": 64,
    "default_model": "exact",
    "default_model_file": "",
    "omega_min": 0,
    "omega_max": 5,
    "omega_points": 500,
    "Nt": 16,
    "extractedQuantity": "RhoOverOmega",
    "FiniteT_kernel": true,
    "multiFit": false,
    "correlatorFile": "",
    "xCol": 0,
    "meanCol": 1,
    "errorCol": 2,
    "correlatorCols": "3:",
    "errormethod": "jackknife",
    "saveParams": true,
    "saveLossHistory": true,
    "verbose": true,
    "outputFile": "",
    "outputDir": "./outputs/",
    "cluster": false,
    "clusterpath": ""
}
```

Run the program with the JSON file:
```bash
python reconstruction.py --config params.json
```
### Command Line

Run the program with command line arguments:
```bash
python reconstruction.py --config params.json --lambda_s 1e-6 --lambda_l2 1e-4 --epochs 2000 --learning_rate 1e-3 --errorWeighting true --networkStructure SupervisedNN --omega_min 0 --omega_max 10 --omega_points 500 --Nt 16 --extractedQuantity RhoOverOmega --FiniteT_kernel true --multiFit false --correlatorFile correlator.txt --xCol 0 --meanCol 1 --errorCol 2 --correlatorCols "" --errormethod jackknife --saveParams true --saveLossHistory true --verbose true --outputFile null --outputDir ""
```

## Known issues

The following issues are known and still need to be fixed/read through.
- supervised appears to be running now on a very ideal dataset. Still one should check a histogram of the weights to see if there are unused neurons 

- I experienced some issues with my compiler crashing while running supervised learning for about 3000-5000 epochs. It might be useful to check this on a different machine to see if this issue persists.
- I added a small wrapper for the raytune library - this is not finished though
- gpr fails due due to a matrix which is not positive definite (idk how to fix this)
- mem needs some work


## Needed Libraries

The following non-standard libraries are needed to run the program:
- `numpy` (general applications)
- `scipy` (general applications)
- `tensorflow` (for machine learning)
- `fredipy` (for the Gaussian method)
- `latqcdtools` (for parallelization in MEM)

### Hints on the installation of TensorFlow

https://www.tensorflow.org/install