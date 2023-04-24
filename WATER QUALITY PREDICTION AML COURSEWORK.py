#!/usr/bin/env python
# coding: utf-8

# In[44]:


#Importing Libraries for Analysis
import numpy as np
import pandas as pd
#Importing Libraries for Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#Importing Libraries for Preprocessing
from sklearn.preprocessing import LabelEncoder,StandardScaler
#Importing Libraries for Model Selection
from sklearn.model_selection import train_test_split,cross_val_score,KFold,GridSearchCV
# Importing the Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[ ]:





# In[45]:


import warnings
warnings.filterwarnings('ignore')


# In[46]:


#Accessing the Dataset
data=pd.read_csv(r'C:\water_potability.csv')
data.head()


# In[47]:


#Shape of the Dataset
data.shape


# In[48]:


#Observation:

#The Following Dataset contains 3276 rows and 10 columns.


# In[49]:


data.info()


# In[50]:


#Statistical Analysis
data.describe()


# In[51]:


data.fillna(data.mean(), inplace=True)
data.head()


# In[52]:


#Checking the Null values of each column
data.isna().sum()


# In[53]:


#Observation

#Three columns are there that have Null values.
#The column names are ph,Sulfate and Trihalomethanes


# In[54]:


# EXXPLORATORY DATA ANALYSIS
#Analyzing the Potability column
data['Potability'].value_counts()


# In[55]:


#Visualizing the Potability column
fig=px.pie(data,values=data['Potability'].value_counts(),
          names=[0,1],
          title='<b>Potability Percentage for the raw data',
          hole=0.4,
          color_discrete_sequence=px.colors.qualitative.Pastel,
          template='plotly_dark')
fig.update_layout(title_font_size=30)
fig.show()


# In[56]:


#Potablity: If Water safe for Drinking

#Potable---> 1(Water safe for Drinking)
#Not Potable--->0(Water not safe for Drinking)
#Observation

#61% of Water is not safe for Drinking and 39% is safe for Drinking


# In[57]:


plt.figure(figsize=(6, 6))
heatmap = sns.heatmap(data.corr()[['Potability']].sort_values(by='Potability', ascending=False), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Features Correlating with Potability', fontdict={'fontsize':18}, pad=16)


# In[58]:


#Analyzing the ph feature
fig=px.histogram(data,x="ph",nbins=40,color_discrete_sequence=["#ffff00"],title="<b>PH Distribution",text_auto=True)
fig.update_layout(template="plotly_dark" )

fig.show()


# In[59]:


#Observation

#The PH is Normally Distributed.


# In[60]:


#Checking the Outliers in ph feature using 3-Standard Deviation method
upper=data['ph'].mean() + 3*data['ph'].std()
lower=data['ph'].mean() - 3*data['ph'].std()
print('Upper Limit is {} and Lower Limit is {}'.format(upper,lower))


# In[61]:


#Observation:

#The Upper Limit of ph distribution is 11.86
#The Lower Limit of ph distribution is 2.29
#Insight:

#This will help me to remove the Outliers.


# In[62]:


# Bivariate Analysis of ph with Potability

sns.displot(x=data['ph'],hue=data['Potability'],kind='kde',fill=True)


# In[ ]:





# In[63]:


#Observation:

#The Hardness is Normally Distributed.


# In[64]:


#Checking the Outliers in Hardness feature using 3-Standard Deviation method
upper_har=data['Hardness'].mean() + 3*data['Hardness'].std()
lower_har=data['Hardness'].mean() - 3*data['Hardness'].std()
print('Upper Limit is {} and Lower Limit is {}'.format(upper_har,lower_har))


# In[65]:


# Bivariate Analysis of Hardness with Potability

sns.displot(x=data['Hardness'],hue=data['Potability'],kind='kde',fill=True)


# In[66]:


#Observation:

#The Not Potable Water has more Hardness as compared to Potable Water.


# In[67]:


#Splitting of Train and Test Data


# In[68]:


X=data.drop('Potability',axis=1)
y=data['Potability']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[69]:


X_train.head()


# In[70]:


X_test.head()


# In[71]:


y_train.head()


# In[72]:


#Shape of the X_train and X_test
print(X_train.shape,X_test.shape)


# In[73]:


#Standardizing the Dataset in order to get more accuracy
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)


# In[74]:


#MODELLING


# In[75]:


models={
    'Logistic Regression':LogisticRegression(),
    'Decision Trees':DecisionTreeClassifier(),
    'Random Forest':RandomForestClassifier(),
    'SVM Classifier':SVC(),
    'GradientBoosting Classifier':GradientBoostingClassifier()
    
}


# In[76]:


#Training Different Models
for name,model in models.items():
    model.fit(X_train,y_train)
    print(f'{name} trained')


# In[77]:


#MODELS EVALUTAION FOR THE WATER POTABILITY COURSEWORK


# In[78]:


results={}
for name,model in models.items():
    result=np.mean(cross_val_score(model,X_train,y_train,scoring='accuracy',cv=10))*100
    results[name]=result


# In[79]:


for name,score in results.items():
    print(f"{name} : {round(score,3)}")


# In[80]:


#Applying Logistic Regression using GridSearchCV
param_grid={
    'penalty':['l1','l2','elasticnet','none'],
    'C':[0.1,0.5,1,2,3,4,5,10,100],
   'max_iter':[100,200,300,500,1000]      
           }
grid_lr=GridSearchCV(LogisticRegression(),param_grid,scoring='accuracy',cv=10)
grid_lr.fit(X_train,y_train)
print(grid_lr.best_params_,grid_lr.best_score_)
y_pred_lr=grid_lr.predict(X_test)
print('The Accuracy of Logistic Regression is :{}'.format(accuracy_score(y_test,y_pred_lr)))
print(classification_report(y_test,y_pred_lr))
sns.heatmap(confusion_matrix(y_test,y_pred_lr),annot=True,fmt='.2f')



# In[81]:


print(grid_lr.best_params_,
grid_lr.best_score_)


# In[82]:


#Applying Decision Tree using GridSearchCV

param_grid={
   'criterion':['gini','entropy','log_loss'],
    'max_depth':[1,5,6,7,8,9,10,20,50,'None']
}
grid_DT=GridSearchCV(DecisionTreeClassifier(),param_grid,cv=10,scoring='accuracy')
grid_DT.fit(X_train,y_train)
print(grid_DT.best_params_,grid_DT.best_score_)


# In[83]:


y_pred_DT=grid_DT.predict(X_test)
print('The Accuracy of Decision Tree is :{}'.format(accuracy_score(y_test,y_pred_DT)))
print(classification_report(y_test,y_pred_DT))
sns.heatmap(confusion_matrix(y_test,y_pred_DT),annot=True,fmt='.2f')


# In[84]:


#Applying Random Forest using GridSearchCV
param_grid={
    'n_estimators':[50,100,200,300],
    'criterion':['gini','entropy','log_loss'],
    'max_depth':[5,6,7,8,9,10,20,50,100],
    'oob_score':['True']
}
grid_RF=GridSearchCV(RandomForestClassifier(),param_grid,cv=10,scoring='accuracy')
grid_RF.fit(X_train,y_train)
print(grid_RF.best_params_,grid_RF.best_score_)


# In[85]:


y_pred_RF=grid_RF.predict(X_test)
print('The Accuracy of the Random Forest is :{}'.format(accuracy_score(y_test,y_pred_RF)))
print(classification_report(y_test,y_pred_RF))
sns.heatmap(confusion_matrix(y_test,y_pred_RF),annot=True,fmt='.2f')


# In[86]:


#Applying SVM Classifier using GridSearchCV
param_grid={
    'C':[0.1,0.5,1,2,3,4,5],
    'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
    'degree':[3,4,5,6,7]
    
}
grid_SVC=GridSearchCV(SVC(),param_grid,cv=10,scoring='accuracy')
grid_SVC.fit(X_train,y_train)
print(grid_SVC.best_params_,grid_SVC.best_score_)


# In[87]:


y_pred_SVC=grid_SVC.predict(X_test)
print('The Accuracy of SVC is :{}'.format(accuracy_score(y_test,y_pred_SVC)))
print(classification_report(y_test,y_pred_SVC))
sns.heatmap(confusion_matrix(y_test,y_pred_SVC),annot=True,fmt='.2f')
plt.show()


# In[88]:


#Applying GradientBoostingClassifier using GridSearchCV
param_grid={
    'n_estimators':[50,100,200,300],
    'learning_rate':[0.1,0.5,1,2,3,4,5]
}
grid_GDC=GridSearchCV(GradientBoostingClassifier(),param_grid,cv=10,scoring='accuracy')
grid_GDC.fit(X_train,y_train)
print(grid_GDC.best_params_,grid_GDC.best_score_)


# In[89]:


y_pred_GDC=grid_GDC.predict(X_test)
print('The Accuracy of GradientBoostingClassifier is :{}'.format(accuracy_score(y_test,y_pred_GDC)))
print(classification_report(y_test,y_pred_GDC))
sns.heatmap(confusion_matrix(y_test,y_pred_GDC),annot=True,fmt='.2f')


# In[90]:


# Plot a histogram of the data
plt.hist(data, bins=30)


# In[91]:


# Set x-axis and y-axis labels
plt.xlabel('Value')
plt.ylabel('Frequency')


# In[92]:


# Show the plot
plt.show()


# In[94]:


#from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming y_test and y_pred are the true and predicted labels, respectively
#fpr, tpr, thresholds = roc_curve(y_test, y_pred_RF,y_pred_lr,y_pred_DT,y_pred_SVC,y_pred_GDC)
#roc_auc = auc(fpr, tpr)

#plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic')
#plt.legend(loc="lower right")
#plt.show()


# In[95]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming y_test and y_pred are the true and predicted labels, respectively
fpr_RF, tpr_RF, thresholds_RF = roc_curve(y_test, y_pred_RF)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_pred_lr)
fpr_DT, tpr_DT, thresholds_DT = roc_curve(y_test, y_pred_DT)
fpr_SVC, tpr_SVC, thresholds_SVC = roc_curve(y_test, y_pred_SVC)
fpr_GDC, tpr_GDC, thresholds_GDC = roc_curve(y_test, y_pred_GDC)

# Compute ROC AUC score for each classifier
roc_auc_RF = auc(fpr_RF, tpr_RF)
roc_auc_lr = auc(fpr_lr, tpr_lr)
roc_auc_DT = auc(fpr_DT, tpr_DT)
roc_auc_SVC = auc(fpr_SVC, tpr_SVC)
roc_auc_GDC = auc(fpr_GDC, tpr_GDC)

# Plot the ROC curve for each classifier
plt.plot(fpr_RF, tpr_RF, label='Random Forest (AUC = %0.2f)' % roc_auc_RF)
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (AUC = %0.2f)' % roc_auc_lr)
plt.plot(fpr_DT, tpr_DT, label='Decision Tree (AUC = %0.2f)' % roc_auc_DT)
plt.plot(fpr_SVC, tpr_SVC, label='Support Vector Machine (AUC = %0.2f)' % roc_auc_SVC)
plt.plot(fpr_GDC, tpr_GDC, label='Gradient Boosting (AUC = %0.2f)' % roc_auc_GDC)

# Set plot labels and legend
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

plt.show()


# In[ ]:




