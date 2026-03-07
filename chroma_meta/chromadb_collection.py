import chromadb

client = chromadb.PersistentClient(path="chroma_db")

cols = client.list_collections()
print("Collections:")
for c in cols:
    print("-", c.name)

