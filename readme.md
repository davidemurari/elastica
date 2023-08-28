This repository contains the current implementation of the code for approximating the solutions to the BVP associated to the Euler Elastica.

The mathematical description of the implemented procedure is in the PDF file [Mathematical description](NeuralNetworkSolvingBVP.pdf).

The organisation of the repository is as follows:
- *testDerivative.ipynb* : this is a notebook showing how to implement the derivative of a function using forward automatic differentiation
- *generated_data.txt* : is the text file containing the training and test data points
- *main.ipynb* : is the notebook where all the code can be run.
- *scripts* : is the directory with python scripts including all the needed methods to run *main.ipynb*
- *savedResults* : is the directory where a new *.csv* file is created where the obtained results of the optuna tests are recorded.

The directory *script* is organised as follows:
- *createDataset.py* : 
  - *getData* : this generates the components of the data points by extracting them from the rows of the .txt file
  - *dataset* : is a class defining the Dataset object
  - *getDataLoaders* : is a method generating the dataloaders for the training and testing phases. This is done once the training and testig data points are provided, and also the batch_size for the training.
- *evaluateModel.py* :
  - *eval_model*: provides the numpy version of the approximated curve
  - *eval_derivative_model* : provides the numpy version of the derivative of the approximated curve (**this is not working as expected at the moment**)
  - *plotTestResults* : plots the comparison between the true and predicted beam configurations.
- *network.py* : is a method containing the class *approximate_curve* which defines the neural network to train. This can be flexibly changed in its architecture by providing suitable input parameters. Furtermore, this model is constrained to satisfy the BCs only if *correct_functional* is set to True.
- *trainig.py* : is a script including the trainig loop. This is based on a closure method, which allows to optionally use also LBFGS method as an optimizer. The loss function can be augmented using physical regularisation terms, and the tangents. This can be done by setting *train_with_tangents* and *pde_regularisation* to True. **However the performance with these set to true is still poor at the moment.**
- *utils.py* : is a script including some useful methods. For the moment the only one is *getBCs*, which extracts the components of the boundary conditions from a discretised trajectory.