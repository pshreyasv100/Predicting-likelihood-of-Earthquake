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
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
#from keras.utils import np_utils
from keras import optimizers
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score



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


#%%
#Splitting data into test and training sets
X = seismic_df.iloc[:,0:17]
y = seismic_df.iloc[:,17]

print(X.shape)
print(y.shape)

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

print ("\n________Gaussian naive Bayes _____________") 
print ("\n mean Accuracy:  ")
print (mn)
print ("\n Standard Deviation:")
print (sd)

print ("\n Confusion Matrix : ")
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


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
classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean', p = 2)
trained_model = classifier2.fit(X_train, y_train)
y_pred = trained_model.predict(X_test)


accuracies = cross_val_score(estimator = classifier2, X = X_train, y = y_train, cv = 10)
mn=accuracies.mean()
sd=accuracies.std()

print ("\n _____________KNN__________________") 
print ("\n mean Accuracy: ")
print (mn)

print ("\n Standard Deviation:")
print (sd)

print ("\n Confusion Matrix : ")
#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc=roc_auc_score(y_test, y_pred)
print("AUC ROC")
print(roc_auc)

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
#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

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

print (" \n Support Vector machines :-  ") 
print ("\n mean Accuracy: ")
print (mn)

print ("\n Standard Deviation:")
print (sd)

print ("\n Confusion Matrix : ")
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc=roc_auc_score(y_test, y_pred)
print("\nAUC ROC ",roc_auc)

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

print ("\n Bagging using Random Forest : Randomly selected 10 features and 100 forest and then mean accuracy :-  ") 
print ("\n mean Accuracy: ")
print (mn)

print ("\n Standard Deviation:")
print (sd)

cm = confusion_matrix(y_test, y_pred)
print(cm)

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

cm = confusion_matrix(y_test, y_pred)
print(cm)

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

#_____________________Neural Network classifier 1________________________________________________________
# create model
model = Sequential()
model.add(Dense(17, input_dim=17, kernel_initializer='normal', activation='relu'))
model.add(Dense(17, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model
sgd =optimizers.SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
model.compile(loss='binary_crossentropy',optimizer=sgd, metrics=['accuracy'])

#We manually provide the train and test partition
history = model.fit(X, y, validation_split=0.33, epochs=30, batch_size=16, verbose=2)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Neural Network model 1 accuracy (epoch = 30)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#_______________________Neural Network Classifier 2_______________________________
history2 = model.fit(X, y, validation_split=0.33, epochs=50, batch_size=16, verbose=2)

# summarize history for accuracy
plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.title('Neural Network model 2 accuracy (epoch = 50)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#_______________________Neural Network Classifier 3_______________________________
history3 = model.fit(X, y, validation_split=0.33, epochs=100, batch_size=16, verbose=2)

# summarize history for accuracy
plt.plot(history3.history['acc'])
plt.plot(history3.history['val_acc'])
plt.title('Neural Network model 3 accuracy (epoch = 100)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#_______________________Neural Network Classifier 4_______________________________
history4 = model.fit(X, y, validation_split=0.33, epochs=150, batch_size=16, verbose=2)
s
# summarize history for accuracy
plt.plot(history4.history['acc'])
plt.plot(history4.history['val_acc'])
plt.title('Neural Network model 4 accuracy (epoch = 150)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
