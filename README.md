# TL;DR

This project consists of multiple modules:
- **neural_network** is the more important, and it contains what is needed to develop and run a Neural Network:
    - **neural_net.py** contains the core functionalities for the neural netowrk, i.e. fit, evaluate, backpropagation, gradient_computation, ... ;
    - **activation_func.py** contains the implementation of a number of activation functions that can be used on the Neural Network;
    - **loss_func.py** contains the implementation of a different loss functions that can be used to train the Neural Network;
    - **optimizer.py** contains the implementation of different optimization algorithms, i.e. SGD, Momentum, Adam, AdaMax;
    - **loss_func.py** contains the implementation of a L1 and L2 regularization function.

- **linear_model** to develop and run a linear model
    - **linear_least_square.py** contains the core functionalities for a least square regression model, i.e. predict, fit, normalization, ... ;
    - **optimizer.py** contains the implementation of different optimization algorithms, i.e. SGD, Momentum, Adam, AdaMax;
    - **QRdecomposition.py** contains the core functionalities to run a linear least square solver based on the QR decomposition;
    - **error_functions.py** contains all the loss functions used to create a level plot to show the paths followed by the different optimization alghorithms.  

- **model_selection** to perform a validation over the Neural Network:
    - **generate_folds.py** contains the functions to split the data into folds to perform k-fold cross validation or nested cross validation;
    - **model_selection.py** to perform the model selection or the nested cross validation.

- **preprocessing** to preprocess the data:
    - **task.py** defines the task for the Neural Network;
    - **hyperparameters.py** defines the hyperparameters to be validate with the model selection;
    - **experiment_settings.py** to define the performance function for the network evaluation
    - **parserExcel.py** to extract the data from a spreadsheet.

- **plot** is the module to generate plots
    - **net_plot.py** to plot the structure of the Neural Network;
    - **plot_graph.py** to plot different kind of graphs like the curvature of the loss function.
    
- **test** is the module to perform different tests over different datasets
    - **MLcup_test.py** to run the tests with the MLcup dataset;
    - **monk_test.py** to run the tests with the Monk dataset;
    - **test.py** to run test on handcrafted datasets and in particular to test the paths followed by the different optimizer algorithms.

- **data_set** contains the spreadshits with the dataset used for the experiments.

Moreover, there are also some external important file:
- **main.py** is the main file to run all the experiments related to the Neural Network;
- **main_linear_model.py** is the main file to run all the experiments related to the linear solvers;
- **keras_nn.py** is used to create an equivalent Neural Network to the one created with our framework, using the same starting weights; 
