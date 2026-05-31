import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading model...")
model = SentenceTransformer("keepitreal/vietnamese-sbert")

print("Loading index...")
index = faiss.read_index("medical.index")

print("Loading metadata...")
with open("medical_metadata.pkl", "rb") as f:
    data = pickle.load(f)

questions = data["questions"]
diseases = data["diseases"]

def search(query, k=5):

    query_vector = model.encode([query]).astype("float32")

    distances, indices = index.search(query_vector, k)

    results = []

    for i in indices[0]:
        results.append({
            "disease": diseases[i],
            "question": questions[i]
        })

    return results


# test
query = "tôi hay quên và mất trí nhớ"
results = search(query)

print("\nQuery:", query)
print("\nTop results:\n")

for r in results:
    print("Disease:", r["disease"])
    print("Match:", r["question"])
    print("--------------------")