import chromadb
from collections import Counter

client = chromadb.PersistentClient(path="chroma_db")
# falls du den Namen noch nicht kennst, nimm den aus list_collections()
col = client.get_collection(client.list_collections()[0].name)

data = col.get(include=["metadatas"], limit=5000)  # bei 900 safe
metas = data["metadatas"]

key_counter = Counter()
for m in metas:
    key_counter.update(m.keys())

print("Metadaten-Keys:")
for k, v in key_counter.most_common():
    print(f"{k}: {v}")
