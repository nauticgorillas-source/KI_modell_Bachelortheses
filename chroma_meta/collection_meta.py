import chromadb

client = chromadb.PersistentClient(path="chroma_db")
col = client.get_collection("schwerpunkte")

print("collection metadata:", col.metadata)
