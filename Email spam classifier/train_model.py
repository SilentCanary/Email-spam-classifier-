import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

df = pd.read_csv("spam_assassin.csv") 
df_gmail = pd.read_csv("emails_training.csv")  

df_combined = pd.concat([df, df_gmail], ignore_index=True)
df_combined = df_combined.dropna(subset=['text'])

df_combined['text'] = df_combined['text'].astype(str)

X = df_combined['text']
y = df_combined['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


clf = LogisticRegression(max_iter=300)
clf.fit(X_train_vec, y_train)


pred = clf.predict(X_test_vec)
print("\n=== Classification Report ===")
print(classification_report(y_test, pred))


pickle.dump(clf, open("classifier.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n🎉 Model saved successfully as classifier.pkl & vectorizer.pkl")
