"""
author:silent_canary
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix,f1_score
import matplotlib.pyplot as plt
data=pd.read_csv("C:\\Users\\sairam\\Documents\\ML stuff\\spam_assassin.csv")
X,y=data["text"],data["target"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

vectorizer=TfidfVectorizer()
X_train_vec=vectorizer.fit_transform(X_train)
X_test_vec=vectorizer.transform(X_test)

model=LogisticRegression()
model.fit(X_train_vec,y_train)

prediction=model.predict(X_test_vec)
con_matrix=confusion_matrix(y_test,prediction)
plt.matshow(con_matrix,cmap=plt.cm.gray)
plt.show()
f1=f1_score(y_test, prediction)
print("f1 score : ",f1)