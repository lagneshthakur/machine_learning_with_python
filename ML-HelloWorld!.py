
# coding: utf-8

# In[15]:


from sklearn import tree


# In[16]:


tree


# In[20]:


features = [[140,"smooth"],[130,"smooth"],[150,"bumpy"],[170,"bumpy"]]
labels = ["apple","apple","orange","orange"]

# For simplicity - change to 0 and 1
# 0 is bumpy, 1 is smooth
# 0 is apple, 1 is orange
features = [[140,1],[130,1],[150,0],[170,0]]
labels = [0,0,1,1]


clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)


# In[9]:


print(clf.predict([[130,1]]))


# In[13]:


print(clf.predict([[160,1]]))

