#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd


# In[63]:


data=pd.read_csv(r'E:\Udemy\Machine Learning A-Z (Codes and Datasets)\Part 2 - Regression\Section 7 - Support Vector Regression (SVR)\Python\Position_Salaries.csv',encoding='latin1')


# In[64]:


data.head()


# In[65]:


data.describe()


# In[66]:


data


# In[67]:


X=data.iloc[:,1:-1].values


# In[68]:


X


# In[69]:


y=data.iloc[:,-1].values


# In[70]:


y


# # Feature Scaling

# In[71]:


from sklearn.preprocessing import StandardScaler


# In[72]:


scaler_x=StandardScaler()
scaler_y=StandardScaler()


# In[73]:


X=scaler_x.fit_transform(X)


# In[74]:


X


# In[75]:


y=y.reshape(len(y),1)


# In[76]:


y=scaler_y.fit_transform(y)


# In[77]:


y


# # Model Building

# In[78]:


from sklearn.svm import SVR


# In[79]:


reg=SVR(kernel='rbf')


# In[80]:


reg.fit(X,y)


# # Predicting New Result

# In[81]:


scaler_y.inverse_transform(reg.predict(scaler_x.transform([[6.5]])))


# In[82]:


import matplotlib.pyplot as plt


# In[83]:


plt.scatter(scaler_x.inverse_transform(X),scaler_y.inverse_transform(y),color='red')
plt.plot(scaler_x.inverse_transform(X),scaler_y.inverse_transform(reg.predict(X)))
plt.title("SVR")
plt.xlabel("Position")
plt.ylabel("Salary")


# In[ ]:




