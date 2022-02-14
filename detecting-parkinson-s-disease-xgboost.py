#!/usr/bin/env python
# coding: utf-8

# # What is Parkinson’s Disease?
# > ### Parkinson’s disease is a progressive disorder of the central nervous system affecting movement and inducing tremors and stiffness. It has 5 stages to it and affects more than 1 million individuals every year in India. This is chronic and has no cure yet. It is a neurodegenerative disorder affecting dopamine-producing neurons in the brain.

# ![](https://www.drprempillay.org/wp-content/uploads/2015/08/ParkinsonsDisease.png)

# # **Symptoms:**
# 
# **Tremor:** A tremor, or shaking, usually begins in a limb, often your hand or fingers. You may rub your thumb and forefinger back and forth, 
# known as a pill-rolling tremor. Your hand may tremble when it's at rest.
# 
# **Slowed movement (bradykinesia):** Over time, Parkinson's disease may slow your movement, making simple tasks difficult and time-consuming. 
# Your steps may become shorter when you walk. It may be difficult to get out of a chair. You may drag your feet as you try to walk.
# 
# 
# **Rigid muscles:** Muscle stiffness may occur in any part of your body. The stiff muscles can be painful and limit your range of motion.
# Impaired posture and balance. Your posture may become stooped, or you may have balance problems as a result of Parkinson's disease.
# 
# **Loss of automatic movements:** You may have a decreased ability to perform unconscious movements, including blinking, smiling or swinging your arms when you walk.
# 
# 
# **Speech changes:** You may speak softly, quickly, slur or hesitate before talking. Your speech may be more of a monotone rather than have the usual inflections.
# 
# 
# **Writing changes:** It may become hard to write, and your writing may appear sm

# # Importing necessary library

# In[49]:


# Make necessary imports:
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, classification_report, roc_auc_score, plot_roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# # Loading the dataset using pandas

# In[2]:


data=pd.read_csv("parkinsons.data")


# # Exploring the dataset

# In[3]:


# using .columns we get the column names of the dataset

data.columns


# In[4]:


# using .shape we get to know about the number of rows and columns present in the dataset

data.shape


# In[5]:


# displaying the first rows from the dataset

data.head()


# In[6]:


# displaying the last rows from the dataset

data.tail()


# In[7]:


data.info


# In[8]:


# displaying the dataset of the columns in the dataset

data.dtypes


# # Checking for null values

# In[83]:


data.isnull().sum()


# Since we dont have any null values we dont need to preprocess it as of now, we can move to EDA.

# # EDA (Exploratory data analysis)

# In[9]:


# Plotting the histogram of dataset

data.hist(figsize=(25,16))
plt.show()


# In[10]:


# Plotting pairplot using Seaborn

sns.pairplot(data.iloc[: ,0:6])
plt.show()


# In[11]:


# Now we will find correlation and plot it using heatmap

data.corr()


# In[85]:


data.corr()['status'][:-1].sort_values().plot(kind='bar')
plt.show()


# In[81]:


f,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(data.corr(), annot = True, fmt= '.2f')
plt.show()


# # Get the features and labels from the DataFrame (dataset). The features are all the columns except ‘status’, and the labels are those in the ‘status’ column.

# In[13]:


features=data.loc[:,data.columns!='status'].values[:,1:]
labels=data.loc[:,'status'].values


# # The ‘status’ column has values 0 and 1 as labels; let’s get the counts of these labels for both- 0 and 1.

# In[14]:


print(labels[labels==1].shape[0], labels[labels==0].shape[0])


# We have 147 ones and 48 zeros in the status column in our dataset.

# Initialize a MinMaxScaler and scale the features to between -1 and 1 to normalize them. The MinMaxScaler transforms features by scaling them to a given range. The fit_transform() method fits to the data and then transforms it. We don’t need to scale the labels.

# In[15]:


scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels


# # Now, split the dataset into training and testing sets keeping 20% of the data for testing.

# In[16]:


x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)


# # Building the XGBoost Model

#  Initialize an XGBClassifier and train the model. This classifies using eXtreme Gradient Boosting- using gradient boosting algorithms for modern data science problems. It falls under the category of Ensemble Learning in ML, where we train and predict using many models to produce one superior output.

# In[73]:


model = XGBClassifier(learning_rate=0.1, max_depth=10,
                    scale_pos_weight=1.5, eval_metric='mlogloss')


# In[74]:


model.fit(x_train, y_train)


# In[75]:


# Finally, generate y_pred (predicted values for x_test)

y_pred=model.predict(x_test)


# In[76]:


# Calculate the accuracy for the model. Print it out.

print(accuracy_score(y_test, y_pred)*100)


# In[77]:


# Calculate the Confusion Matrix for the model. Print it out.

print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(model, x_test, y_test)
plt.show()


# In[78]:


# Displaying the classification report using sklearn

print(classification_report(y_test, y_pred))


# In[79]:


plot_roc_curve(model, x_test, y_test) 
plt.show()


# # Prediction using custom values

# In[86]:


# we took random values for this prediction, we can play with values to see the different output

newinput=[[274.688,240.005,174.287,0.01360,0.01008,0.00624,0.00564,0.01873,
           1.02308,0.256,0.51268,0.01365,0.81667,0.63804,0.10715,6.883,0.607567,0.158453,3.679772,0.131728]]


# In[87]:


# Here we will use our model to predict the output based on new values 

output=model.predict(newinput)
output


# In[88]:


# label 1 means True
# label 0 means False


if output == 1:
    print(True)
else:
    print(False)


# # Summary
# - In this Python machine learning project, we learned to detect the presence of Parkinson’s Disease in individuals using various factors. We used an XGBClassifier for this and made use of the sklearn library to prepare the dataset. 
# - This gives us an accuracy of 94.87%, which is great considering the number of lines of code in this project.
