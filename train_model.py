import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

print("Loading dataset...")
df = pd.read_csv("ViMedical_Disease.csv")

questions = df["Question"].astype(str).tolist()
diseases = df["Disease"].astype(str).tolist()

print("Loading embedding model...")
model = SentenceTransformer("keepitreal/vietnamese-sbert")

print("Embedding data...")
embeddings = model.encode(
    questions,
    show_progress_bar=True,
    batch_size=32
)

embeddings = np.array(embeddings).astype("float32")

print("Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("Saving index...")
faiss.write_index(index, "medical.index")

print("Saving metadata...")
with open("medical_metadata.pkl", "wb") as f:
    pickle.dump({
        "questions": questions,
        "diseases": diseases
    }, f)

print("DONE TRAINING")