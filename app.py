from flask import Flask, request, jsonify
import fasttext
import pandas as pd

# Charger le modèle fasttext (assure-toi que le chemin est correct)
model = fasttext.load_model("lid.176.ftz")

# Charger le CSV des réponses
responses_df = pd.read_csv("response.csv")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    phrase = data.get("phrase", "").strip().lower()

    # Détecter la langue avec fasttext
    prediction = model.predict(phrase)[0][0]
    lang = prediction.replace("__label__", "")

    # Chercher les lignes dans responses_df où la phrase du CSV est contenue dans la phrase reçue
    matched_rows = responses_df[responses_df['phrase'].apply(lambda x: x in phrase)]

    if not matched_rows.empty:
        # Filtrer par langue détectée
        resp_row = matched_rows[matched_rows['langue'] == lang]
        if not resp_row.empty:
            response = resp_row['reponse'].values[0]
            return jsonify({"lang": lang, "response": response})

    # Réponse par défaut si aucune correspondance
    return jsonify({"lang": lang, "response": "Je ne comprends pas encore. Souhaitez-vous que j'apprenne ? (Y/N)"})

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, jsonify
import fasttext
import pandas as pd

# Charger le modèle fasttext (assure-toi que le chemin est correct)
model = fasttext.load_model("lid.176.ftz")

# Charger le CSV des réponses
responses_df = pd.read_csv("response.csv")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    phrase = data.get("phrase", "").strip().lower()

    # Détecter la langue avec fasttext
    prediction = model.predict(phrase)[0][0]
    lang = prediction.replace("__label__", "")

    # Chercher les lignes dans responses_df où la phrase du CSV est contenue dans la phrase reçue
    matched_rows = responses_df[responses_df['phrase'].apply(lambda x: x in phrase)]

    if not matched_rows.empty:
        # Filtrer par langue détectée
        resp_row = matched_rows[matched_rows['langue'] == lang]
        if not resp_row.empty:
            response = resp_row['reponse'].values[0]
            return jsonify({"lang": lang, "response": response})

    # Réponse par défaut si aucune correspondance
    return jsonify({"lang": lang, "response": "Je ne comprends pas encore. Souhaitez-vous que j'apprenne ? (Y/N)"})

if __name__ == '__main__':
    app.run(debug=True)
