# Case 1

Hi Jonah, I will use this `README` as a place to share sources and ideas for each of the bullit points. 



* Describe your model and method (including model selection and valida-
tion)
   * Regression tree with 'missing' as a new category: 
      * RMSE traning 49.26
      * RMSE test 46.32
   * XGBoost is very promising, with NaN replaced by "Missing".
      * RMSE test 35.04
   * Trying the stacking method to get better results but currently not as good as XGB
      * RMSE test 37.13 

* Argue for your choices of model, model selection and validation.

* Describe how you handled missing data.
    * 3 ways 
       * throw away input with missing features 
       * impute missing values with feature means 
       * If a categorical feature, let "missing" be a new category: 
       * [Source](https://davidrosenberg.github.io/mlcourse/Archive/2017Fall/Lectures/10a.trees.pdf), p 32

* Describe how you handled factors in the features (catergorical variables).
  * We can try both with the so called *integrer encoding* i.e. `K=1, H=2 ...` or use so called *One-Hot Encoding* i.e. extend the dataset with dummy varibales. A good quick guide can be seen at:
    * [source 1](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)

* Estimate the predictive performance of your model on xnew. We are interested in the root mean squared error 
As you do not know the true values ynew, you cannot just calculate the
error, you need to estimate it. Your estimate will be denoted RMSE.Describe what you did.


