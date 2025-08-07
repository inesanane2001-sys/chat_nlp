import fasttext
from sentence_transformers import SentenceTransformer
import joblib

# Charger le modèle fastText
lang_model = fasttext.load_model("lid.176.ftz")

# Entrée utilisateur
phrase = input("Votre phrase : ")
print(f"Phrase testée : {phrase}")

# Détecter la langue avec fastText
pred = lang_model.predict(phrase)
lang = pred[0][0].replace("__label__", "")  # ex: '__label__fr' → 'fr'
print(f"Langue détectée : {lang}")

# Charger le modèle SBERT
model = SentenceTransformer('all-MiniLM-L6-v2')

# Charger le bon classifieur
try:
    if lang == 'fr':
        clf = joblib.load('classifier_fr.pkl')
    elif lang == 'en':
        clf = joblib.load('classifier_en.pkl')
    else:
        print("Langue non supportée")
        exit()
except FileNotFoundError:
    print("Modèle non trouvé pour cette langue.")
    exit()

# Prédire
embedding = model.encode([phrase])
response = clf.predict(embedding)[0]
print("Réponse prédite :", response)
