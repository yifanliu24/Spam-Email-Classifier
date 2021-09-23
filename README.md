Before running this program, ensure that the numpy and str2bool packages are installed in your python interpreter.
If it is not, you may install them using the following commands from your command prompt:
python -m pip install numpy
python -m pip install str2bool

This program accepts six arguments in the following order:
1) boolean <flag_NB> specifies whether the Naive Bayes modeling is performed
2) boolean <flag_LR> specifies whether the MCAP logistic regression with L2 regulatrization is performed
3) boolean <flag_PTR> specifies whether the perceptron training rule modeling is performed
4) boolean <flag_HP> specifies whether the hyper-parameterization is performed. For the logistic regression, this will print values of lambda and the validation set accuracy. For the perceptron training rule, this will print values for the number of iterations and the validation set accuracy.
5) string <train-set> is the path to the training set directory (e.g. './dataset 1/train/')
6) string <test-set> is the path to the test set directory (e.g. './dataset 1/test/')

Dataset 1:
First make sure the datasets are in right directory: ./dataset 1/
This should be a subdirectory of where the .py file is stored.
Then run the following command in the command prompt: 
python main.py 1 1 1 1 './dataset 1/train/' './dataset 1/test/'

This should result in several lines being printed into the console. 
- The first line should state the test set accuracy of the Naive Bayes model.
- The next nine lines display a table showing the validation set accuracy of the Logistic Regression (LR) model for various values of lambda. The following line displays the chosen final value of lambda. The line after that displays the test set accuracy of the LR model.
- The next seven lines display a table showing the validation set accuracy of the Perceptron Training Rule (PTR) model for different numbers of iterations. The following line displays the selected ideal number of iterations. The final line displays the test set accuracy of the PTR model.

Adjusting the first four boolean parameters will reduce the amount of information generated. Replacing the first parameter with a 0 will not generate a Naive Bayes model. Replacing the second parameter with a 0 will not generate a Logistic Regression model. Replacing the third parameter with a 0 will not generate a Perceptron Training Rule model. Replacing the fourth parameter with a 0 will eliminate the lambda vs. accuracy and iterations vs. accuracy analysis for the LR and PTR models.

Dataset 2:
First make sure the datasets are in right directory: ./dataset 2/
Unlike Dataset 1, the files had to be unzipped in order to be processed in Dataset 2. This should be a subdirectory of where the .py file is stored.
Then run the following command in the command prompt: 
python main.py 1 1 1 1 './dataset 2/train/' './dataset 2/test/'

Dataset 3:
First make sure the datasets are in right directory: ./dataset 3/
Like Dataset 2, the files had to be unzipped in order to be processed in Dataset 3. This should be a subdirectory of where the .py file is stored.
Then run the following command in the command prompt: 
python main.py 1 1 1 1 './dataset 3/train/' './dataset 3/test/'
