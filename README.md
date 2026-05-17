# CV RAG API

A complete FastAPI application that turns a notebook-style CV RAG workflow into a production-friendly API.

The API retrieves relevant CV chunks from Pinecone, sends them to an LLM through OpenRouter, and returns an answer grounded only in the retrieved CV context.

## Project structure

```text
cv-rag-api/
├── app/
│   ├── __init__.py
│   ├── main.py      # FastAPI routes
│   └── rag.py       # Pinecone + embeddings + OpenRouter RAG logic
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 1. Create your `.env` file

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Then fill in:

```env
PINECONE_API_KEY=your_pinecone_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
PINECONE_INDEX=cv-rag
OPENROUTER_MODEL=mistralai/mistral-7b-instruct:free
RETRIEVER_TOP_K=5
```

Important: your Pinecone index must already contain CV chunks embedded with the same embedding model.

## 2. Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/docs
```

## 3. Run with Docker

```bash
docker compose up --build
```

Open:

```text
http://localhost:8000/docs
```

## Endpoints

### `GET /health`

Checks whether the RAG pipeline initialized correctly.

### `POST /ask`

Request:

```json
{
  "question": "Which candidate has Python and FastAPI experience?",
  "include_context": true
}
```

Response:

```json
{
  "question": "Which candidate has Python and FastAPI experience?",
  "answer": "...",
  "context": [
    {
      "content": "retrieved CV chunk text",
      "metadata": {}
    }
  ]
}
```

### `POST /ask-stream`

Streams the answer as plain text.

Example:

```bash
curl -N -X POST http://localhost:8000/ask-stream \
  -H "Content-Type: application/json" \
  -d '{"question":"Who has React experience?"}'
```

## Common Pinecone metadata names

The retriever looks for text in any of these metadata fields:

- `text`
- `chunk_text`
- `page_content`
- `content`

If your Pinecone records use a different metadata key, update `app/rag.py` inside `PineconeRetriever.invoke()`.

## Notes

- Do not commit `.env` to GitHub.
- Make sure the Pinecone index dimension matches the selected embedding model.
- `BAAI/bge-small-en-v1.5` produces 384-dimensional embeddings.
