# Things to do

- Outlier (Very very very important)
- Feature generation
   - [Check time variable](https://www.kaggle.com/denzo123/a-closer-look-at-date-variables)
- [Feature extraction](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73937)
- Missing value / NaN value
- **Para tuning**
- Ensemble
- Error analysis: find out data points with large error values and see why they perform badly
- [Check external data / Data leak](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/72958)
- Try new ideas
   - From [this discussion](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/78470), if you want to fit more closely to outliers: Change the metric you use for minimizing the sum of the residuals. That is, instead of minimizing the sum of the squared residuals, maybe minimize for the sum of the residuals to the power of 5.
   - From [this discussion](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/78470), if you want to fit less closely to outliers: Remove the outliers from your sample, or change the the optimization from minimizing the sum of the squared residuals, to minimizing the sum of the L1 residuals maybe. Or minimizing the sum of the Square Root Residuals
   - From [this discussion](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/78470), note the following manually crafted loss function:
      ```
      def custom_asymmetric_train(y_true, y_pred):
         residual = (y_true - y_pred).astype("float")
         grad = -5*residual**4
         hess = 20.0*residual**3
         return grad, hess

         def custom_asymmetric_valid(y_true, y_pred):
         residual = (y_true - y_pred).astype("float")
         loss = residual**5
         return "custom_asymmetric_eval", np.mean(loss), False

         model = LGBMClassifier(boosting_type="gbdt", objective=custom_asymmetric_train）

         model.fit( X_train, y_train, eval_set=eval_set, eval_metric=custom_asymmetric_valid）
      ```
      and <br/>
      
      ```
      params = {
               'boosting_type': 'gbdt',
               'objective': 'None',
               'metric': 'None',
               }

               model = lgb.train(params,
                         lgb_train, # a LGBM Dataset
                         num_boost_round=10,
                         fobj=custom_asymmetric_train,
                         feval=custom_asymmetric_valid,
                         valid_sets=eval_set # Also should be a LGBM Dataset I think
               )
      ```
   - find out what cardID means from [here](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/78732)
   - from [here](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/78614) and this [kernel](https://www.kaggle.com/roydatascience/recursive-feature-selection-using-sklearn-on-elo?scriptVersionId=9969948), try recursive feature selection
   - Varyinh SD limit for determining the set of outliers
