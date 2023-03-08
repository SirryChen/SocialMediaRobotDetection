import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, auc, roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv('./feature.csv')

X = df.drop(columns=["user_id","label"],axis=1)
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


rf = RandomForestClassifier(n_estimators=150)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
# y_pred_quant = rf.predict_proba(X_test)[:,1]
y_true = y_test

acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1score = f1_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
auc_ = auc(fpr, tpr)
confusion_matrix = confusion_matrix(y_test,y_pred)

print('acc:', acc)
print('precision:', precision)
print('recall:', recall)
print('f1score:', f1score)
print('mcc:', mcc)
print('auc:', auc_)
print('confusion_matrix:')
print(confusion_matrix)
