from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import joblib

# Charger les données
with open('data.csv', encoding='utf-8') as f:
    X = [line.strip() for line in f if line.strip()]

with open('response.csv', encoding='utf-8') as f:
    y = [line.strip() for line in f if line.strip()]

# Encoder les phrases
model = SentenceTransformer('all-MiniLM-L6-v2')
X_encoded = model.encode(X)

# Entraîner le modèle
clf = LogisticRegression()
clf.fit(X_encoded, y)

# Sauvegarder
joblib.dump(clf, 'classifier_fr.pkl')
