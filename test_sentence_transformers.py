from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # modèle léger, rapide à télécharger
sentences = ["Bonjour, comment ça va ?", "Salut, quoi de neuf ?"]
embeddings = model.encode(sentences)

for sentence, embedding in zip(sentences, embeddings):
    print(f"Phrase: {sentence}")
    print(f"Embedding vector (extrait) : {embedding[:5]}...\n")
