from process_eeg import get_features_and_labels
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

train_x,test_x,train_y,test_y = get_features_and_labels(0.2)

est = GradientBoostingClassifier(n_estimators=200, max_depth=3)
est.fit(train_x,train_y[:,1])

est1 = RandomForestClassifier(n_estimators = 200, max_depth=3)
est1.fit(train_x,train_y[:,1])

est2 = GaussianNB()
est2.fit(train_x,train_y[:,1])

est_voting = VotingClassifier(estimators = [('gb', est), ('rf', est1), ('gnb', est2)], voting='soft')
est_voting.fit(train_x,train_y[:,1])

pred = est.predict(test_x)
pred1 = est1.predict(test_x)
pred2 = est2.predict(test_x)
pred3 = est_voting.predict(test_x)

score = precision_recall_fscore_support(test_y[:,1], pred, average = "binary")
score1 = precision_recall_fscore_support(test_y[:,1], pred1, average = "binary")
score2 = precision_recall_fscore_support(test_y[:,1], pred2, average = "binary")
score3 = precision_recall_fscore_support(test_y[:,1], pred3, average = "binary")


print(score)
print(score1)
print(score2)
print(score3)
