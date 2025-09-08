import weaviate
from weaviate.classes.query import MetadataQuery
from sentence_transformers import SentenceTransformer

# Connect to your running container
print("🔌 Connecting to Weaviate ...")
client = weaviate.connect_to_custom(
    http_host="127.0.0.1", http_port=8080, http_secure=False,
    grpc_host="127.0.0.1", grpc_port=50051, grpc_secure=False,
)
print("✅ Connected to Weaviate client")
print("📂 Listing available collections ...")
collections = client.collections.list_all()
# print("   ->", collections)

if "Movies" not in collections:
    print("❌ Collection 'Movies' not found! Did you run ingest_movies.py?")
    client.close()
    raise SystemExit(1)

print("📂 Getting collection 'Movies' ...")
movies = client.collections.get("Movies")
print("✅ Got collection")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

query_text = "space adventure"
query_vec = model.encode(query_text, normalize_embeddings=True).tolist()


print("📊 Running hybrid query: 'space adventure' ...")
res = movies.query.hybrid(
    query=query_text,
    vector=query_vec,
    alpha=0.6,
    limit=5,
    return_metadata=MetadataQuery(score=True),
)

print("✅ Query executed, parsing results ...\n")
if not res.objects:
    print("⚠️ No results found")
else:
    for i, obj in enumerate(res.objects, start=1):
        p = obj.properties or {}
        print(f"{i}. {p.get('title')} ({p.get('release_year')}) "
              f"[score={obj.metadata.score:.4f}]")
        print("   ", p.get("description"))
        print()

client.close()
print("🔒 Connection closed")