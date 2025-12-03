import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('COMBINED.csv', encoding='latin1')

data.dropna(inplace=True)
data.drop_duplicates(subset=['Comment'], inplace=True)

def clean_text(text):
    text = text.lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text
data['Comment'] = data['Comment'].apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Comment'])
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr=LogisticRegression()
lr.fit(X_train, y_train)
accuracy = lr.score(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

text=["Indians are so stinky and dirty."]
text_vectorized = vectorizer.transform(text)
prediction = lr.predict(text_vectorized)
print(f'Prediction for the sample text: {prediction[0]}')