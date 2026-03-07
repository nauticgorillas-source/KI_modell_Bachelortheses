import chromadb

client = chromadb.PersistentClient(path="chroma_db")
col = client.get_collection("schwerpunkte")

print("count:", col.count())

sample = col.get(limit=3, include=["metadatas", "documents"])  # <- ids NICHT hier rein!
for i in range(len(sample["ids"])):  # ids sind trotzdem da
    print("\n--- item", i+1, "---")
    print("id:", sample["ids"][i])
    print("meta:", sample["metadatas"][i])
    print("doc snippet:", sample["documents"][i][:200])
