# Vector Embedder

A lightweight FastAPI service for generating vector embeddings from text using [sentence-transformers](https://www.sbert.net/) models.
Includes Redis caching, API key authentication, batch support.

## Features

* Supports any HuggingFace model (e.g. `sentence-transformers`, `e5`, `bge`, `intfloat`, etc.)
* Single and batch embedding endpoints
* Select model via query or environment variable
* Redis caching (or fallback to in-memory LRU)
* API key protection for all endpoints (except `/healthz`)
* Docker-ready & CI/CD pipeline via GitHub Actions
* Optimized multi-architecture builds (`amd64`, `arm64`) with caching

## Getting Started

### ▶Run with Docker

```bash
docker run -p 8000:8000 \
  -e KEY=my-secret-key \
  -e EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
  sadovyiov/vector-embedder:latest
```

### ▶Run with docker-compose

```yaml
services:
  embedder:
    image: sadovyiov/vector-embedder:latest
    platform: linux/amd64
    ports:
      - "8000:8000"
    environment:
      - KEY=my-secret-key
      - REDIS_HOST=redis
      - EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

  redis:
    image: redis:7
```

## API Endpoints

> All endpoints require the `Key: your-secret-key` header except `/healthz`.

### `POST /embed`

Returns embedding for a single string.

```json
{
  "text": "Shimano Twin Power FD C3000",
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

Example response:

```json
{
  "embedding": [0.12, 0.23, ...],
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "cached": true
}
```

### `POST /embed-batch`

Returns list of embeddings for a list of strings.

```json
{
  "texts": ["Shimano", "Rod pod"],
  "model": "intfloat/e5-small-v2"
}
```

### `GET /healthz`

Simple health check. No API key required.

```json
{
  "status": "ok",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "cache": "redis"
}
```

## Authorization

All API endpoints except `/healthz` require this header:

```http
Key: your-secret-key
```

## Running Tests

```bash
export KEY=test-key
pytest
```

Tests are located in the `tests/` directory.

## License

MIT License. See `LICENSE` file for details.