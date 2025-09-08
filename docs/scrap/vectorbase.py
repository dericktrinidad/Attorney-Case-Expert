import weaviate

client = weaviate.connect_to_custom(
    http_host="127.0.0.1", http_port=8080,
    http_secure=False,
    grpc_host="127.0.0.1", grpc_port=50051,
    grpc_secure=False,
)

movies = client.collections.get("Movies")

count = movies.aggregate.over_all(total_count=True).total_count
print(f"Collection 'Movies' has {count} objects")

client.close()
