from flask import Flask, request, jsonify
import fasttext
import pandas as pd
import os
from together import Together
import csv

# Initialisation de l'app Flask
app = Flask(__name__)

# Initialisation du client Together
client = Together(api_key="183031c2f56604dd565e8619bd06bb6d014648f6818120460649377d1934b336")

# Fonction GPT
def generate_response_with_together(prompt):
    try:
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",  # Modèle puissant et gratuit
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Erreur Together API : {e}")
        return "Erreur de communication avec le modèle Together."

# Chargement du modèle FastText
model_path = r"C:\Users\INES\Documents\chatbot_nlp\lid.176.ftz"
model = fasttext.load_model(model_path)

# Fichier CSV pour stocker les phrases apprises
csv_path = r"C:\Users\INES\Documents\chatbot_nlp\response.csv"
if os.path.exists(csv_path):
    responses_df = pd.read_csv(csv_path)
else:
    responses_df = pd.DataFrame(columns=["phrase", "langue", "reponse"])

# Mémoire temporaire
pending_learning = {
    "phrase": None,
    "lang": None,
    "response": None  # Ajout pour stocker temporairement la réponse générée
}

@app.route('/predict', methods=['POST'])
def predict():
    global responses_df, pending_learning

    data = request.get_json()
    phrase = data.get("phrase", "").strip().lower()

    # Si utilisateur répond "O" → Générer la réponse
    if phrase in ["y", "yes", "o", "oui", "نعم"]:
        last_phrase = pending_learning.get("phrase")
        last_lang = pending_learning.get("lang")

        if last_phrase and last_lang:
            # Si une réponse a déjà été générée mais pas enregistrée
            if pending_learning.get("response"):
                reponse = pending_learning["response"]
                responses_df = pd.concat([
                    responses_df,
                    pd.DataFrame([{
                        "phrase": last_phrase,
                        "langue": last_lang,
                        "reponse": reponse
                    }])
                ], ignore_index=True)
                responses_df.to_csv(csv_path, index=False)

                pending_learning["phrase"] = None
                pending_learning["lang"] = None
                pending_learning["response"] = None

                return jsonify({
                    "lang": last_lang,
                    "confidence": 1.0,
                    "response": "La réponse a été enregistrée."
                })

            # Sinon → générer une nouvelle réponse
            new_response = generate_response_with_together(last_phrase)
            pending_learning["response"] = new_response

            return jsonify({
                "lang": last_lang,
                "confidence": 1.0,
                "response": f"{new_response}\n\nSouhaites-tu enregistrer cette réponse ? (O/N)"
            })

        return jsonify({"response": "Aucune phrase à apprendre pour l’instant."})

    # Si utilisateur répond "N" ou "non"
    if phrase in ["n", "no", "non", "لا"]:
        pending_learning["phrase"] = None
        pending_learning["lang"] = None
        pending_learning["response"] = None
        return jsonify({"response": " Réponse ignorée."})

    # Détection de langue avec fastText
    prediction, confidence = model.predict(phrase)
    lang = prediction[0].replace("__label__", "")
    confidence = confidence[0]

    # Si phrase déjà connue
    match = responses_df[
        (responses_df["phrase"].str.lower() == phrase) & (responses_df["langue"] == lang)
    ]
    if not match.empty:
        return jsonify({
            "lang": lang,
            "confidence": round(confidence, 2),
            "response": match["reponse"].values[0]
        })

    # Sinon stocker pour apprentissage
    pending_learning["phrase"] = phrase
    pending_learning["lang"] = lang
    pending_learning["response"] = None  # Efface ancienne réponse

    # Message selon langue
    messages = {
        "fr": "Je ne comprends pas encore. Veux-tu que je l'apprenne ? (O/N)",
        "en": "I don't understand yet. Would you like me to learn it? (Y/N)",
        "ar": "لا أفهم بعد. هل ترغب أن أتعلم هذه الجملة؟ (نعم/لا)"
    }

    default = "Langue non prise en charge pour l’apprentissage."
    message = messages.get(lang, default)

    return jsonify({
        "lang": lang,
        "confidence": round(confidence, 2),
        "response": message
    })

# Lancer le serveur Flask
if __name__ == "__main__":
    app.run(debug=True)
