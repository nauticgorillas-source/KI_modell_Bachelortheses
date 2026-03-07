import streamlit as st
import pandas as pd
import json
import os
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import hashlib

#--->
#Grundkonfiguration
#--->
EXCEL_FILE = "stichprobe.xlsx"
JSON_FILE = "faelle.json"
CHROMA_DIR = "./chroma_db"


# --->
# Hilfsfunktionen
# --->
def load_json():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_json(data):
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def make_id(sp, stufe, text):
    raw = f"{sp.lower().strip()}\n{stufe}\n{text.strip()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()



# --->
# Streamlit Setup aufbauuu
# --->
st.title(" Datenimport – Excel → JSON → ChromaDB")

if st.button(" Excel importieren & einpflegen"):
    st.write(" Lade Modell…")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    st.write(" Lade Excel…")
    excel_data = pd.read_excel(EXCEL_FILE, engine="openpyxl", sheet_name= None)
    df = pd.concat(excel_data.values(), ignore_index= True)

    st.write(" Lade faelle.json…")
    json_data = load_json()

    existing = {(e["schwerpunkt"].lower().strip(), int(e["stufe"]), e["fallschilderung"].strip()) for e in json_data}

    client = PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection("faelle")

    new_count = 0
    dup_count = 0

    for _, row in df.iterrows():
        sp = str(row["schwerpunkt"]).strip().lower()
        stufe = int(row["stufe"])
        text = str(row["fallschilderung"]).strip()

        key = (sp, stufe, text)

        if key in existing:
            dup_count += 1
            continue

        json_data.append({
            "schwerpunkt": sp,
            "stufe": stufe,
            "fallschilderung": text
        })
        
        # Embedding + Chroma-ID erzeugen
        vec = model.encode(text).tolist()
        cid = make_id(sp, stufe, text)

        collection.upsert(
            ids=[cid],
            documents=[text],
            embeddings=[vec],
            metadatas=[{"schwerpunkt": sp, "stufe": stufe}]
        )

        existing.add(key)
        new_count += 1

    save_json(json_data)

    st.success(f" Fertig! Neu eingefügt: {new_count}, Duplikate übersprungen: {dup_count}")