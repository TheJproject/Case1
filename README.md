# Case 1

Hi Jonah, I will use this `README` as a place to share sources and ideas for each of the bullit points. 

* Describe your model and method (including model selection and valida-
tion)
  *

* Argue for your choices of model, model selection and validation.

* Describe how you handled missing data.

* Describe how you handled factors in the features (catergorical variables).
  * We can try both with the so called *integrer encoding* i.e. `K=1, H=2 ...` or use so called *One-Hot Encoding* i.e. extend the dataset with dummy varibales. A good quick guide can be seen at:
    * [source 1](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)

* Estimate the predictive performance of your model on xnew. We are interested in the root mean squared error 
As you do not know the true values ynew, you cannot just calculate the
error, you need to estimate it. Your estimate will be denoted RMSE.Describe what you did.
