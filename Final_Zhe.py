## read data
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
data = pd.read_csv(r"C:\Users\Zhe\Desktop\Data-Mining\Toddler Autism dataset July 2018.csv")
pd.set_option('display.max_columns', None)
print(data.head())
print(data.info())
print(data.describe(include='all'))


## find missing value and ourlier
import seaborn as sns
import matplotlib.pyplot as plt

for i in data.columns[1:]:
    x = data[i].value_counts()
    print(i, x)

sns.boxplot(x=data['Age_Mons'], color="b")
plt.show()
sns.boxplot(x=data['Qchat-10-Score'], color="b")
plt.show()


## Split dataset by Class/ASD Traits

grouped = data.groupby(data.columns[-1])
data_asd = grouped.get_group("Yes")
data_non_asd = grouped.get_group("No")

## Age range
sns.distplot(data['Age_Mons'], kde=False, bins=36, color='b')
plt.title('All participated kids')
plt.show()
sns.distplot(data_asd['Age_Mons'], kde=False, bins=36, color='b')
plt.title('ASD positive kids')
plt.show()
sns.distplot(data_non_asd['Age_Mons'], kde=False, bins=36, color='b')
plt.title('ASD negative kids')
plt.show()


## Gender
plt.figure(figsize=(9,7))
ax_g = plt.gca()
sns.countplot(x='Sex', data=data, order=['f', 'm'], hue='Class/ASD Traits ')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax_g.xaxis.label.set_size(20)
ax_g.yaxis.label.set_size(20)
plt.show()


## Family member
plt.figure(figsize=(9,7))
ax_f = plt.gca()
sns.countplot(x='Family_mem_with_ASD', data=data, hue='Class/ASD Traits ')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax_f.xaxis.label.set_size(20)
ax_f.yaxis.label.set_size(20)
plt.show()


## Jaundice
plt.figure(figsize=(9,7))
ax_j = plt.gca()
sns.countplot(x='Jaundice', data=data, hue='Class/ASD Traits ')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax_j.xaxis.label.set_size(20)
ax_j.yaxis.label.set_size(20)
plt.show()

## Race
plt.figure(figsize=(13,10))
ax_e = plt.gca()
sns.countplot(x='Ethnicity', data=data, hue='Class/ASD Traits ')
plt.xticks(fontsize=16, rotation=45)
plt.yticks(fontsize=16, rotation=45)
ax_e.xaxis.label.set_size(20)
ax_e.yaxis.label.set_size(20)
plt.tight_layout()
plt.show()

## Who completed the test
plt.figure(figsize=(13,10))
ax_w = plt.gca()
sns.countplot(x='Who completed the test', data=data, hue='Class/ASD Traits ')
plt.xticks(fontsize=16, rotation=45)
plt.yticks(fontsize=16, rotation=45)
ax_w.xaxis.label.set_size(20)
ax_w.yaxis.label.set_size(20)
plt.tight_layout()
plt.show()

## Question most important
for ii in data_asd.columns[1:11]:
    plt.figure(figsize=(3, 6))
    ax = plt.gca()
    sns.countplot(x=ii, data=data_asd)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    plt.tight_layout()
    plt.show()

## Preprocessing--drop 0:11,split x,y and numeric convert
from sklearn.preprocessing import LabelEncoder
data.drop('Qchat-10-Score', inplace=True, axis=1)
X_data = pd.get_dummies(data.iloc[:, 11:-1])
X = X_data.values
y_data = data.values[:, -1]
y_le = LabelEncoder()
y = y_le.fit_transform(y_data)

## Preprocessing--split train/test and standardize the data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
stdsc.fit(X_train)
X_train_std = stdsc.transform(X_train)
X_test_std = stdsc.transform(X_test)

## grid search for DT, RF, LR and NB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

np.random.seed(42)
##grid search for DT
clf1 = DecisionTreeClassifier()
clf1.fit(X_train_std, y_train)
y_pred1 = clf1.predict(X_test_std)
print("Accuracy : ", accuracy_score(y_test, y_pred1) * 100)
print(classification_report(y_test, y_pred1))

para_DT = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'min_samples_leaf': [3, 4, 5, 6, 7, 8, 9]}
clf_DT = GridSearchCV(clf1, para_DT, cv=5, refit=True)
clf_DT.fit(X_train_std, y_train)
y_pred_DT = clf_DT.predict(X_test_std)
print("Accuracy : ", accuracy_score(y_test, y_pred_DT) * 100)
print(classification_report(y_test,y_pred_DT))
print(clf_DT.best_params_)
print(clf_DT.best_score_)

# grid search for RF
clf2 = RandomForestClassifier()
clf2.fit(X_train_std, y_train)
y_pred2 = clf2.predict(X_test_std)
print("Accuracy : ", accuracy_score(y_test, y_pred2) * 100)
print(classification_report(y_test, y_pred2))

para_RF = {
    'n_estimators': [10, 100, 200],
    'max_depth': [None, 3, 5, 10],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [3, 4, 5, 6, 7, 8]}
clf_RF = GridSearchCV(clf2, para_RF, cv=5)
clf_RF.fit(X_train_std, y_train)
y_pred_RF = clf_RF.predict(X_test_std)
print("Accuracy : ", accuracy_score(y_test, y_pred_RF) * 100)
print(classification_report(y_test,y_pred_RF))
print(clf_RF.best_params_)
print(clf_RF.best_score_)

# grid search for LR
clf3 = LogisticRegression()
clf3.fit(X_train_std, y_train)
y_pred3 = clf3.predict(X_test_std)
print("Accuracy : ", accuracy_score(y_test, y_pred3) * 100)
print(classification_report(y_test, y_pred3))

para_LR = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [50, 100, 200, 500, 1000]}
clf_LR = GridSearchCV(clf3, para_LR, cv=5)
clf_LR.fit(X_train_std, y_train)
y_pred_LR = clf_LR.predict(X_test_std)
print("Accuracy : ", accuracy_score(y_test, y_pred_LR) * 100)
print(classification_report(y_test,y_pred_LR))
print(clf_LR.best_params_)
print(clf_LR.best_score_)

# grid search for NBG
clf4 = GaussianNB()
clf4.fit(X_train_std, y_train)
y_pred4 = clf4.predict(X_test_std)
print("Accuracy : ", accuracy_score(y_test, y_pred4) * 100)
print(classification_report(y_test, y_pred4))

para_NBG = {'var_smoothing': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]}
clf_NBG = GridSearchCV(clf4, para_NBG, cv=5)
clf_NBG.fit(X_train_std, y_train)
y_pred_NBG = clf_NBG.predict(X_test_std)
print("Accuracy : ", accuracy_score(y_test, y_pred_NBG) * 100)
print(classification_report(y_test,y_pred_NBG))
print(clf_NBG.best_params_)
print(clf_NBG.best_score_)

# grid search for NBM
clf5 = MultinomialNB()
clf5.fit(X_train, y_train)
y_pred5 = clf5.predict(X_test)
print("Accuracy : ", accuracy_score(y_test, y_pred5) * 100)
print(classification_report(y_test, y_pred5))

para_NBM = {'alpha': [0.01, 0.1, 0.5, 1.0, 10.0, ]}
clf_NBM = GridSearchCV(clf5, para_NBM, cv=5)
clf_NBM.fit(X_train, y_train)
y_pred_NBM = clf_NBM.predict(X_test)
print("Accuracy : ", accuracy_score(y_test, y_pred_NBM) * 100)
print(classification_report(y_test,y_pred_NBM))
print(clf_NBM.best_params_)
print(clf_NBM.best_score_)


# grid search for KNN
clf6 = KNeighborsClassifier()
clf6.fit(X_train_std, y_train)
y_pred6 = clf6.predict(X_test_std)
print("Accuracy : ", accuracy_score(y_test, y_pred6) * 100)
print(classification_report(y_test, y_pred6))

para_KNN = {'n_neighbors': [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,]}
clf_KNN = GridSearchCV(clf6, para_KNN, cv=5)
clf_KNN.fit(X_train_std, y_train)
y_pred_KNN = clf_KNN.predict(X_test_std)
print("Accuracy : ", accuracy_score(y_test, y_pred_KNN) * 100)
print(classification_report(y_test,y_pred_KNN))
print(clf_KNN.best_params_)
print(clf_KNN.best_score_)




## ensemble and vote
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
Labels = ['DT', 'RF', 'LR', 'KNN', 'clf_vote_hard', 'clf_vote_soft']
clf_DT_final = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
clf_RF_final = RandomForestClassifier(max_depth=None, max_features='auto', min_samples_leaf=5, n_estimators=10)
clf_LR_final = LogisticRegression(C=0.001, max_iter=50, penalty='l1', solver='saga')
clf_KNN_final = KNeighborsClassifier(n_neighbors=19)
clf_vote_hard = VotingClassifier(estimators=[(Labels[0],clf_DT_final),
                                             (Labels[1],clf_RF_final),
                                             (Labels[2],clf_LR_final),
                                             (Labels[3],clf_KNN_final)], voting='hard')

clf_vote_soft = VotingClassifier(estimators=[(Labels[0],clf_DT_final),
                                             (Labels[1],clf_RF_final),
                                             (Labels[2],clf_LR_final),
                                             (Labels[3],clf_KNN_final)], voting='soft')
label2 = ['clf_vote_hard', 'clf_vote_soft']
for clf, labels in zip([clf_vote_hard, clf_vote_soft], label2):
    scores = cross_val_score(clf, X_train_std, y_train, cv=5, scoring='accuracy')  #Evaluate a score by cross-validation.
    print('Accuracy: %0.2f (+/- %0.2f) [%s]' %(scores.mean(), scores.std(), labels))

clf_vote_hard.fit(X_train_std, y_train)
y_pred_vhard = clf_vote_hard.predict(X_test_std)
print("Accuracy : ", accuracy_score(y_test, y_pred_vhard) * 100)
print(classification_report(y_test,y_pred_vhard))

clf_vote_soft.fit(X_train_std, y_train)
y_pred_vsoft = clf_vote_soft.predict(X_test_std)
print("Accuracy : ", accuracy_score(y_test, y_pred_vsoft) * 100)
print(classification_report(y_test,y_pred_vsoft))

## Best method: KNN
clf_KNN_final.fit(X_train_std, y_train)
y_pred_KNN_final = clf_KNN_final.predict(X_test_std)
print("Accuracy : ", accuracy_score(y_test, y_pred_KNN_final) * 100)
print(classification_report(y_test,y_pred_KNN_final))

from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred_KNN_final)
class_names = data['Class/ASD Traits '].unique()


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=25)
plt.xlabel('Predicted label',fontsize=25)
plt.tight_layout()
plt.show()