import os, json, hashlib
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer


#--->
#Grundkonfigurieren
#--->
PERSIST_DIR     = "./chroma_db"
COLLECTION      = "schwerpunkte"
FAELLE_JSON     = "faelle.json"
MODEL_DIR       = "custom_model"

BATCH_SIZE      = 64

#--->
#stabile ID aufbauen pro schwerpunkt, stufe, text
#--->
def make_id(sp: str, stufe: int, text: str) -> str:
    raw = f"{sp.lower().strip()}\n{stufe}\n{text.strip}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def main():
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError("custom_model/ nicht gefunden. Erst tuning.py ausführen")
    
    if not os.path.isfile(FAELLE_JSON):
        raise FileNotFoundError("faelle.json nicht gefunden.")
    
    with open(FAELLE_JSON, "r", encoding= "utf-8") as f:
        data = json.load(f)
    
    model = SentenceTransformer(MODEL_DIR)
    #ChromaDB initialisieren
    client = PersistentClient(path=PERSIST_DIR)
    col = client.get_or_create_collection(COLLECTION)

    #  Daten vorbereiten
    docs, metas, ids = [], [], []
    for e in data:
        sp      = str(e["schwerpunkt"]).strip()
        stufe   = int(e["stufe"])
        text    = str(e["fallschilderung"]).strip()
        typ     = str(e.get("typ","")).strip()

        docs.append(text)
        metas.append({"schwerpunkt": sp, "stufe": stufe, "typ": typ})
        ids.append(make_id(sp, stufe, text))

    # Batchweise Reindexieren
    for i in range (0, len(docs), BATCH_SIZE):
        batch_docs  = docs[i:i+BATCH_SIZE]
        batch_ids   = ids[i:i+BATCH_SIZE]
        batch_metas = metas[i:i+BATCH_SIZE]
        # neue Embeddings berechnen
        vecs    = model.encode(batch_docs, show_progress_bar=False).tolist()
        # es soll in Chroma speichern 
        # Falls ID existiert ersetzen
        # Falls ID nicht existiert neu einfügen
        col.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=vecs)

    print("Fertig. Collection:", COLLECTION, "| count", col.count())

if __name__ == "__main__":
    main()           