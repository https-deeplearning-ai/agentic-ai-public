# Reflective Research Agent (FastAPI + Postgres, single container)

A FastAPI web app that plans a research workflow, runs tool-using agents (Tavily, arXiv, Wikipedia), and stores task state/results in Postgres.
This repo includes a Docker setup that runs **Postgres + the API in one container** (for local/dev).

## Features

* `/` serves a simple UI (Jinja2 template) to kick off a research task.
* `/generate_report` kicks off a threaded, multi-step agent workflow (planner → research/writer/editor).
* `/task_progress/{task_id}` live status for each step/substep.
* `/task_status/{task_id}` final status + report.

---

## Project layout (key paths)

```
.
├─ main.py                      # FastAPI app (your file shown above)
├─ src/
│  ├─ planning_agent.py         # planner_agent(), executor_agent_step()
│  ├─ agents.py                 # research_agent, writer_agent, editor_agent  (example)
│  └─ research_tools.py         # tavily_search_tool, arxiv_search_tool, wikipedia_search_tool
├─ templates/
│  └─ index.html                # UI page rendered by "/"
├─ static/                      # optional static assets (css/js)
├─ docker/
│  └─ entrypoint.sh             # starts Postgres, prepares DB, then launches Uvicorn
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

> Make sure `templates/index.html` and (optionally) `static/` exist and are copied into the image.

---

## Prerequisites

* **Docker** (Desktop on Windows/macOS, or engine on Linux).


* API keys stored in a `.env` file:

  ```
  OPENAI_API_KEY=your-open-api-key
  TAVILY_API_KEY=your-tavily-api-key
  ```

* Python deps are installed by Docker from `requirements.txt`:

  * `fastapi`, `uvicorn`, `sqlalchemy`, `python-dotenv`, `jinja2`, `requests`, `wikipedia`, etc.
  * Plus any libs used by your `aisuite` client.

---

## Environment variables

The app reads `DATABASE_URL` and supports generic LLM configuration for OpenAI-compatible servers.

* The container’s entrypoint sets a sane default for local dev:

  ```
  postgresql://app:local@127.0.0.1:5432/appdb
  ```
* To use Tavily:

  * Provide `TAVILY_API_KEY` (via `.env` or `-e`).

LLM configuration (generic, CPU-friendly options):

```
# Choose a provider (informational): ollama | lmstudio | openai
LLM_PROVIDER=ollama

# Base URL for an OpenAI-compatible server
# - Ollama:   http://localhost:11434/v1
# - LMStudio: http://localhost:1234/v1
LLM_BASE_URL=http://localhost:11434/v1

# API key (string; can be a dummy for local servers)
LLM_API_KEY=ollama

# Default model identifier
# - Examples (CPU quantized):
#   qwen2.5:7b-instruct-q4_K_M
#   llama3.1:8b-instruct-q4_K_M
# - For OpenAI (optional): openai:gpt-4.1-mini
LLM_MODEL=qwen2.5:7b-instruct-q4_K_M
```

Backward compatibility: if `LLM_BASE_URL`/`LLM_API_KEY` are not set, `OPENAI_BASE_URL`/`OPENAI_API_KEY` are used if present.

Optional (if you want to override defaults done by the entrypoint):

* `POSTGRES_USER` (default `app`)
* `POSTGRES_PASSWORD` (default `local`)
* `POSTGRES_DB` (default `appdb`)

---

## Build & Run (local/dev)

### 1) Build

```bash
docker build -t fastapi-postgres-service .
```

### 2) Run (foreground)

**IMPORTANT**: Add `--add-host=host.docker.internal:host-gateway` to allow the container to reach Ollama on your host machine.

```bash
docker run --rm -it \
  --add-host=host.docker.internal:host-gateway \
  -p 8000:8000 \
  -p 5432:5432 \
  --name fpsvc \
  --env-file .env \
  fastapi-postgres-service
```

You should see logs like:

```
🚀 Starting Postgres cluster 17/main...
✅ Postgres is ready
CREATE ROLE
CREATE DATABASE
🔗 DATABASE_URL=postgresql://app:local@127.0.0.1:5432/appdb
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3) Open the app

* UI: [http://localhost:8000/](http://localhost:8000/)
* Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API quickstart

### Kick off a run

```bash
# Using default model from env
curl -X POST http://localhost:8000/generate_report \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Large Language Models for scientific discovery"}'

# Overriding model per request (works with local OpenAI-compatible servers)
curl -X POST http://localhost:8000/generate_report \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Large Language Models for scientific discovery", "model":"qwen2.5:7b-instruct-q4_K_M"}'
# -> {"task_id": "UUID..."}
```

### Poll progress

```bash
curl http://localhost:8000/task_progress/<TASK_ID>
```

### Final status + report

```bash
curl http://localhost:8000/task_status/<TASK_ID>
```

---

## Troubleshooting

**I open [http://localhost:8000](http://localhost:8000) and see nothing / errors**

* Confirm `templates/index.html` exists inside the container:

  ```bash
  docker exec -it fpsvc bash -lc "ls -l /app/templates && ls -l /app/static || true"
  ```
* Watch logs while you load the page:

  ```bash
  docker logs -f fpsvc
  ```

**Container asks for a Postgres password on startup**

* The entrypoint uses **UNIX socket + peer auth** for admin tasks (no password).
  Ensure you’re not calling `psql -h 127.0.0.1 -U postgres` in the script—use:

  ```bash
  su -s /bin/bash postgres -c "psql -c '...'"
  ```

**`DATABASE_URL not set` error**

* The entrypoint exports a default DSN. If you overrode it, ensure it’s valid:

  ```
  postgresql://<user>:<password>@<host>:<port>/<database>
  ```

**Tables disappear on restart**

* In your `main.py` you call `Base.metadata.drop_all(...)` on startup.
  Comment it out or guard with an env flag:

  ```python
  if os.getenv("RESET_DB_ON_STARTUP") == "1":
      Base.metadata.drop_all(bind=engine)
  ```

**Tavily / arXiv / Wikipedia errors**

* Provide `TAVILY_API_KEY` and ensure network access, provide in the root dir and `.env` file as follows:
```
# OpenAI API Key
OPENAI_API_KEY=your-open-api-key
TAVILY_API_KEY=your-tavily-api-key
```

* Wikipedia rate limits sometimes; try later or handle exceptions gracefully.

---

## Development tips

* **Hot reload** (optional): For dev, you can run Uvicorn with `--reload` if you mount your code:

  ```bash
  docker run --rm -it -p 8000:8000 -p 5432:5432 \
    -v "$PWD":/app \
    --name fpsvc fastapi-postgres-service \
    bash -lc "pg_ctlcluster \$(psql -V | awk '{print \$3}' | cut -d. -f1) main start && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
  ```

* **Connect to DB from host:**

  ```bash
  psql "postgresql://app:local@localhost:5432/appdb"
  ```

---
