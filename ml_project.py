# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 19:52:00 2018

@author: Shreyas
"""

#%%
#Load libraries 
import pandas as pd
import numpy 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder



#%%
#Exploratory data analysis
seismic_df = pd.read_csv('dataset.txt')

seismic_df.shape
seismic_df.info()

#checking for Missing Values
seismic_df.isnull().sum()

seismic_df.describe()



#%%
#Preprocessing the data

#One hot encoding the categorical variables 'seismic','seismoacoustic','shift','ghazard'
#ordinality for seismic, seismoacoustic, ghazard is maintained since their categories are ordered

#integer encoding
label_encoder = LabelEncoder()
seismic_integer_encoded = label_encoder.fit_transform(seismic_df['seismic'])
seismic_df['seismic'] = seismic_integer_encoded

seismoacoustic_integer_encoded = label_encoder.fit_transform(seismic_df['seismoacoustic'])
seismic_df['seismoacoustic'] = seismoacoustic_integer_encoded

ghazard_integer_encoded = label_encoder.fit_transform(seismic_df['ghazard'])
seismic_df['ghazard'] = ghazard_integer_encoded


seismic_df = seismic_df.drop(['shift'],axis =1)

#One hot encoding 'shift'
# integer encode
#shift_integer_encoded = label_encoder.fit_transform(seismic_df['shift'])
#print(shift_integer_encoded)
# binary encode
#onehot_encoder = OneHotEncoder(sparse=False)
#shift_integer_encoded = shift_integer_encoded.reshape(len(shift_integer_encoded), 1)
#shift_onehot_encoded = onehot_encoder.fit_transform(shift_integer_encoded)

#type(shift_onehot_encoded)

#shift_onehot_encoded = pd.DataFrame(shift_onehot_encoded)
#seismic_df.append(shift_onehot_encoded)


#%%

#Model building

#Converting values of target attribute to categorical
#seismic_df["class"] = seismic_df["class"].astype('category')

X = seismic_df.iloc[:,0:17]
y = seismic_df.iloc[:,17]


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)



#%%
#___________________________Naive bayes model___________________________________________________

from sklearn.naive_bayes import GaussianNB
classifier1 = GaussianNB()

trained_model = classifier1.fit(X_train, y_train)
y_pred = trained_model.predict(X_test)


#Finding accuracy using 10 fold cross-validation

accuracies = cross_val_score(estimator = classifier1, X = X_train, y = y_train, cv = 10)
mn=accuracies.mean()
sd=accuracies.std()

print ("\n Gaussian naive Bayes :-  ") 
print ("\n mean Accuracy:  ")
print (mn)
print ("\n Standard Deviation:")
print (sd)

print ("\n Confusion Matrix : ")
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#print ("\n Classification Report : ")
#from sklearn.metrics import classification_report
#print (classification_report(y_test, y_pred) )

fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc=roc_auc_score(y_test, y_pred)
print ("\n AUC ROC score ",roc_auc)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


#%%
#____________________ KNN model_____________________________________________________

from sklearn.neighbors import KNeighborsClassifier
classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
trained_model = classifier2.fit(X_train, y_train)
y_pred = trained_model.predict(X_test)


accuracies = cross_val_score(estimator = classifier2, X = X_train, y = y_train, cv = 10)
mn=accuracies.mean()
sd=accuracies.std()

print (" \n KNN :-  ") 
print ("\n mean Accuracy: ")
print (mn)

print ("\n Standard Deviation:")
print (sd)

print ("\n Confusion Matrix : ")
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#print ("\n Classification Report : ")
#from sklearn.metrics import classification_report
#print (classification_report(y_test, y_pred) )

fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc=roc_auc_score(y_test, y_pred)
print (roc_auc)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



#%%

#_________________________________ Logistic regression model _____________________________________________


from sklearn.linear_model import LogisticRegression   #class
classifier3  = LogisticRegression(random_state = 0)

trained_model = classifier3.fit(X_train, y_train)
y_pred = trained_model.predict(X_test)

accuracies = cross_val_score(estimator = classifier3, X = X_train, y = y_train, cv = 10)
mn=accuracies.mean()
sd=accuracies.std()

print (" \n Logistic Regression :-  ") 
print ("\n mean Accuracy: ")
print (mn)

print ("\n Standard Deviation:")
print (sd)

print ("\n Confusion Matrix : ")
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#print ("\n Classification Report : ")
#from sklearn.metrics import classification_report
#print (classification_report(y_test, y_pred) )

fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc=roc_auc_score(y_test, y_pred)
print (roc_auc)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



#%%

#_______________________ Decision Tree Classification __________________________________________________
from sklearn.tree import DecisionTreeClassifier
classifier4 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
trained_model = classifier4.fit(X_train, y_train)
y_pred = trained_model.predict(X_test)



accuracies = cross_val_score(estimator = classifier4, X = X_train, y = y_train, cv = 10)
mn=accuracies.mean()
sd=accuracies.std()

print (" \n Decision Tree :-  ") 
print ("\n mean Accuracy: ")
print (mn)

print ("\n Standard Deviation:")
print (sd)

print ("\n Confusion Matrix : ")
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#print ("\n Classification Report : ")
#from sklearn.metrics import classification_report
#print (classification_report(y_test, y_pred) )

fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc=roc_auc_score(y_test, y_pred)
print (roc_auc)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



#%%

#________________________________ Support Vector Machines___________________________________________

from sklearn.svm import SVC
classifier5 = SVC()
trained_model = classifier5.fit(X_train, y_train)
y_pred = trained_model.predict(X_test)



accuracies = cross_val_score(estimator = classifier5, X = X_train, y = y_train, cv = 10)
mn=accuracies.mean()
sd=accuracies.std()

print (" \n Decision Tree :-  ") 
print ("\n mean Accuracy: ")
print (mn)

print ("\n Standard Deviation:")
print (sd)

print ("\n Confusion Matrix : ")
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#print ("\n Classification Report : ")
#from sklearn.metrics import classification_report
#print (classification_report(y_test, y_pred) )

fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc=roc_auc_score(y_test, y_pred)
print (roc_auc)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


#%%

#_________________________Random Forest Classification_________________________________________

from sklearn.ensemble import RandomForestClassifier

seed = 7
num_trees = 150
max_features = 10

classifier6 = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
trained_model = classifier6.fit(X_train, y_train)
y_pred = trained_model.predict(X_test)



accuracies = cross_val_score(estimator = classifier6, X = X_train, y = y_train, cv = 10)
mn=accuracies.mean()
sd=accuracies.std()

print (" \n Bagging using Random Forest : Randomly selected 10 features and 100 forest and then mean accuracy :-  ") 
print ("\n mean Accuracy: ")
print (mn)

print ("\n Standard Deviation:")
print (sd)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#print ("\n Classification Report : ")
#from sklearn.metrics import classification_report
#print (classification_report(y_test, y_pred) )

fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc=roc_auc_score(y_test, y_pred)
print (roc_auc)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



#%%

#_____________________ADA Boost classifier________________________________________________________

from sklearn.ensemble import AdaBoostClassifier

seed = 7
num_trees = 50
classifier7 = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
accuracies = cross_val_score(estimator = classifier7, X = X_train, y = y_train, cv = 10)
# demonstrates the construction of 50 decision trees in sequence using the AdaBoost algorithm.
mn=accuracies.mean()
sd=accuracies.std()

print (" \n Boosting using Adaboost : Randomly selected 50 tress and then mean accuracy :-  " )
print ("\n mean Accuracy: ")
print (mn)

print ("\n Standard Deviation:")
print (sd)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#print ("\n Classification Report : ")
#print (classification_report(y_test, y_pred) )

fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc=roc_auc_score(y_test, y_pred)
print (roc_auc)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



