from flask import Flask, request, jsonify
import fasttext
import pandas as pd

app = Flask(__name__)

# Charger le modèle FastText
model = fasttext.load_model("lid.176.ftz")

# CSV des réponses
responses_df = pd.read_csv("response.csv")

# Dictionnaire en mémoire pour stocker l'état par utilisateur (simplifié ici)
memory = {
    "last_unknown_phrase": None,
    "last_lang": "fr"
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    phrase = data.get("phrase", "").strip().lower()

    #  1. Gérer réponse spéciale O/N ou Y/N d'abord
    if phrase in {"o", "y", "oui", "yes"} and memory["last_unknown_phrase"]:
        return jsonify({
            "lang": memory["last_lang"],  # on utilise ce qu'on a détecté AVANT
            "response": f" Je vais chercher une réponse à : \"{memory['last_unknown_phrase']}\" (via GPT)"
        })

    if phrase in {"n", "non", "no"}:
        memory["last_unknown_phrase"] = None  # reset la mémoire
        return jsonify({
            "response": "Très bien, je n'apprendrai pas cette phrase."
        })

    #  2. Phrase normale → détecter la langue
    prediction, confidence = model.predict(phrase)
    lang = prediction[0].replace("__label__", "")
    confidence = confidence[0]
    memory["last_lang"] = lang

    #  3. Vérifier dans le CSV
    resp_row = responses_df[
        (responses_df['phrase'].str.lower() == phrase) &
        (responses_df['langue'] == lang)
    ]

    if not resp_row.empty:
        memory["last_unknown_phrase"] = None  # on reset car on a répondu
        return jsonify({
            "lang": lang,
            "confidence": round(confidence, 2),
            "response": resp_row['reponse'].values[0]
        })

    #  4. Phrase inconnue → proposer d'apprendre
    memory["last_unknown_phrase"] = phrase

    if lang == "fr":
        return jsonify({
            "lang": lang,
            "confidence": round(confidence, 2),
            "response": "Je ne comprends pas encore. Veux-tu que je l'apprenne ? (O/N)"
        })
    elif lang == "en":
        return jsonify({
            "lang": lang,
            "confidence": round(confidence, 2),
            "response": "I don't understand yet. Would you like me to learn it? (Y/N)"
        })
    else:
        return jsonify({
            "lang": lang,
            "confidence": round(confidence, 2),
            "response": "Langue non prise en charge."
        })
