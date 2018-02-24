from process_eeg import get_features_and_labels
from sklearn.naive_bayes import GaussianNB
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV

train_x,test_x,train_y,test_y = get_features_and_labels(0.2)

gbm = lgb.LGBMClassifier(objective='binary',
                        num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=20)
gbm.fit(train_x, train_y[:,1],
        eval_set=[(test_x, test_y[:,1])],
        early_stopping_rounds=5)

print('Start predicting...')
# predict
y_pred = gbm.predict(test_x, num_iteration=gbm.best_iteration_)
# eval
print('The rmse of prediction is:', precision_recall_fscore_support(test_y[:,1], y_pred, average = 'binary'))

# feature importances
print('Feature importances:', list(gbm.feature_importances_))

# other scikit-learn modules
estimator = lgb.LGBMRegressor(num_leaves=31)

param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}

gbm = GridSearchCV(estimator, param_grid)

gbm.fit(train_x, train_y[:,1])

print('Best parameters found by grid search are:', gbm.best_params_)