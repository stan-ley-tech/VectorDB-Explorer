from tests.conftest import requires_postgres


@requires_postgres
def test_create_collection_and_fetch_it_back(client):
    response = client.post("/collections", json={"name": "docs", "dim": 32, "metric": "cosine"})
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "docs"
    assert body["dim"] == 32

    fetched = client.get(f"/collections/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["name"] == "docs"


@requires_postgres
def test_creating_a_duplicate_collection_name_is_rejected(client):
    client.post("/collections", json={"name": "dupes", "dim": 8})
    response = client.post("/collections", json={"name": "dupes", "dim": 8})
    assert response.status_code == 409


@requires_postgres
def test_get_missing_collection_is_404(client):
    response = client.get("/collections/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


@requires_postgres
def test_upsert_vectors_generates_embeddings_when_not_supplied(client):
    collection = client.post("/collections", json={"name": "auto-embed", "dim": 64}).json()

    response = client.post(
        f"/collections/{collection['id']}/vectors",
        json={
            "vectors": [
                {"external_id": "a", "document": "authentication middleware for API requests"},
                {"external_id": "b", "document": "deploying containers to production"},
            ]
        },
    )
    assert response.status_code == 201
    vectors = response.json()
    assert len(vectors) == 2
    assert {v["external_id"] for v in vectors} == {"a", "b"}


@requires_postgres
def test_upsert_with_wrong_dimension_embedding_is_rejected(client):
    collection = client.post("/collections", json={"name": "dim-check", "dim": 16}).json()

    response = client.post(
        f"/collections/{collection['id']}/vectors",
        json={"vectors": [{"external_id": "a", "embedding": [0.1, 0.2, 0.3]}]},
    )
    assert response.status_code == 422


@requires_postgres
def test_collection_stats_reflects_document_count(client):
    collection = client.post("/collections", json={"name": "stats-check", "dim": 8}).json()
    client.post(
        f"/collections/{collection['id']}/vectors",
        json={"vectors": [{"external_id": str(i), "document": f"document number {i}"} for i in range(3)]},
    )

    stats = client.get(f"/collections/{collection['id']}/stats").json()
    assert stats["document_count"] == 3
    assert stats["dimensions"] == 8
    assert stats["index_type"] is None


@requires_postgres
def test_search_by_text_returns_best_match_first(client):
    collection = client.post("/collections", json={"name": "search-basic", "dim": 128}).json()
    client.post(
        f"/collections/{collection['id']}/vectors",
        json={
            "vectors": [
                {"external_id": "auth", "document": "authentication middleware validates API requests"},
                {"external_id": "weather", "document": "the weather today is sunny with a light breeze"},
            ]
        },
    )

    response = client.post(
        f"/collections/{collection['id']}/search", json={"query": "how do I authenticate an API request", "top_k": 2}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["external_id"] == "auth"
    assert body["used_index"] is False


@requires_postgres
def test_search_with_metadata_filter_excludes_non_matching_vectors(client):
    collection = client.post("/collections", json={"name": "filter-check", "dim": 32}).json()
    client.post(
        f"/collections/{collection['id']}/vectors",
        json={
            "vectors": [
                {"external_id": "a", "document": "api security guide", "metadata": {"category": "security"}},
                {"external_id": "b", "document": "api security overview", "metadata": {"category": "ops"}},
            ]
        },
    )

    response = client.post(
        f"/collections/{collection['id']}/search",
        json={"query": "api security", "top_k": 10, "filter": {"category": "security"}},
    )
    body = response.json()
    assert [r["external_id"] for r in body["results"]] == ["a"]


@requires_postgres
def test_update_vector_document_and_metadata(client):
    collection = client.post("/collections", json={"name": "update-check", "dim": 16}).json()
    created = client.post(
        f"/collections/{collection['id']}/vectors",
        json={"vectors": [{"external_id": "a", "document": "original text", "metadata": {"v": 1}}]},
    ).json()[0]

    response = client.patch(
        f"/collections/{collection['id']}/vectors/{created['id']}",
        json={"document": "updated text", "metadata": {"v": 2}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["document"] == "updated text"
    assert body["metadata"] == {"v": 2}


@requires_postgres
def test_delete_vector_removes_it(client):
    collection = client.post("/collections", json={"name": "delete-check", "dim": 16}).json()
    created = client.post(
        f"/collections/{collection['id']}/vectors",
        json={"vectors": [{"external_id": "a", "document": "to be deleted"}]},
    ).json()[0]

    delete_response = client.delete(f"/collections/{collection['id']}/vectors/{created['id']}")
    assert delete_response.status_code == 204

    get_response = client.get(f"/collections/{collection['id']}/vectors/{created['id']}")
    assert get_response.status_code == 404


@requires_postgres
def test_import_csv_dataset(client):
    collection = client.post("/collections", json={"name": "import-check", "dim": 64}).json()

    csv_content = "text,category\nfirst document,a\nsecond document,b\n"
    response = client.post(
        f"/collections/{collection['id']}/import",
        params={"format": "csv"},
        files={"file": ("dataset.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["imported"] == 2
    assert body["failed"] == 0

    stats = client.get(f"/collections/{collection['id']}/stats").json()
    assert stats["document_count"] == 2


@requires_postgres
def test_export_json_round_trips_imported_data(client):
    collection = client.post("/collections", json={"name": "export-check", "dim": 32}).json()
    client.post(
        f"/collections/{collection['id']}/vectors",
        json={"vectors": [{"external_id": "a", "document": "exportable document"}]},
    )

    response = client.get(f"/collections/{collection['id']}/export", params={"format": "json"})
    assert response.status_code == 200
    exported = response.json()
    assert len(exported) == 1
    assert exported[0]["external_id"] == "a"
    assert len(exported[0]["embedding"]) == 32


@requires_postgres
def test_benchmark_without_native_engine_still_reports_brute_force_results(client):
    collection = client.post("/collections", json={"name": "bench-check", "dim": 16}).json()

    response = client.post(
        f"/collections/{collection['id']}/benchmark",
        json={
            "name": "small smoke test",
            "config": {
                "dataset_size": 100,
                "dimensions": 16,
                "top_k": 5,
                "metric": "cosine",
                "nlist": 4,
                "nprobe": 2,
                "num_queries": 10,
                "concurrency": 2,
            },
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "completed"
    assert any(r["mode"] == "brute_force" for r in body["results"])

    fetched = client.get(f"/benchmarks/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == body["id"]
