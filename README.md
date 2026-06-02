# The Agentic Pipeline

## What You'll Build

By the end of this guide, you'll have a working AI pipeline that fetches news articles, uses an LLM to filter them for business relevance, and stores everything in a local database. This is the foundation for a system that generates startup ideas from daily news.

The architecture diagram and tech stack are included in the course materials.

---

## Prerequisites

- **OS:** macOS, Linux, or WSL2 on Windows
- **Python:** 3.13+
- **uv:** Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management
- **Agentic Coding Tool:** Claude Code ([install instructions](https://code.claude.com/docs/en/getting-started)), Cursor, Codex, Windsurf, AntiGravity, or similar
- **Gemini API Key:** Get a free key at [Google AI Studio](https://aistudio.google.com/api-keys). Note: the free tier has rate limits; for sustained use, set up billing in Google Cloud Console.
- **NewsAPI.ai API Key:** Sign up at [newsapi.ai](https://newsapi.ai) and get your API key. The free tier is generous enough to build the pipeline; for daily production runs, you'll need a paid plan. Use [this link](https://gammavibe.com/go/newsapi) for a discounted $40/month plan (normally $90+). *Disclosure: This is a referral link — you get 10% off, I get a small commission.*

---

## Setup

### 1. Initialize Project

Create a new project directory with Git initialized (however you prefer — git init, clone from GitHub, etc.).

Copy the course starter files (CLAUDE.md, README.md, .gitignore, env.example) into the project root.

### 2. Configure Environment

Copy the example environment file and add your API keys:

```bash
cp env.example .env
```

Open `.env` and populate:
- `NEWSAPI_API_KEY` — Your NewsAPI.ai API key
- `GEMINI_API_KEY` — Your Gemini API key

---

## The Build Protocol

This pipeline is built using agentic engineering. The `CLAUDE.md` file contains system context that aligns your coding agent with the project architecture.

The examples below use Claude Code, but the prompts work with any agentic coding tool (Cursor, Codex, Windsurf, AntiGravity, etc.). If using a different tool, you may need to rename `CLAUDE.md` to match your tool's conventions (e.g., `AGENTS.md`).

Below are the first few prompts to get you started.

---

### Phase 1: Project Scaffolding

Initialize the project structure.

**Prompt:**

```
Set up the project structure as described in @CLAUDE.md. Create the folder structure, empty `__init__.py` files, and a working CLI. The run command should just print "Running idea-pipeline..." for now. Make sure `uv run idea-pipeline run` and `uv run idea-pipeline --help` both work.
```

---

### Phase 2: Database Foundation

Set up SQLite with Alembic (for migrations) and the Run model. The Run model represents a full pipeline execution and tracks status and progress. All other models will reference a Run via `run_id`.

**Prompt:**

```
Set up SQLite with Alembic and the Run model.

Database setup in `db/database.py`: synchronous SQLite engine and session factory, database file at `data/idea_pipeline.db`.

Run model in `core/models.py`:
* `RunStatus` enum: IN_PROGRESS, COMPLETED, FAILED  
* `Run` table: id (UUID), started_at, completed_at (nullable), status, last_completed_step_key (nullable string for resumability)

RunRepository in `db/repositories.py`:
* `create()`: new run with IN_PROGRESS status  
* `get_by_id()`  
* `update_status(run_id, status, last_completed_step_key)`: also sets completed_at if status is COMPLETED

Tests: pytest fixture for in-memory SQLite, test each repository method.

Create the initial Alembic migration.

Update @CLAUDE.md with Alembic commands in Development Commands.
```

---

### Phase 3: News Ingestion

Before fetching articles, we need the Article model. This follows the Base/Table pattern: `ArticleBase` defines fields used by AI agents, while `Article` adds database-specific fields and is used for persistence.

**Prompt:**

```
Add the Article model following the Base/Table pattern from @CLAUDE.md (since Article will later interact with AI agents).

`ArticleBase` fields: `news_service` (e.g. "newsapi.ai"), `news_service_article_key` (provider's unique ID), `url`, `source`, `title`, `body`, `published_at`.

`Article` adds: `id` (UUID), `created_at`, `run_id` (foreign key to runs). Unique constraint on (`news_service`, `news_service_article_key`) to prevent duplicates.

ArticleRepository in `db/repositories.py`:

* `upsert_many(articles, run_id)`: bulk insert, skip duplicates silently (ON CONFLICT DO NOTHING), return count of inserted rows

Tests for upsert_many, including verifying duplicate handling (inserting same article twice should only create one row).

Create an Alembic migration for the articles table.
```

Now implement the Fetch step. First, set up your NewsAPI.ai Topic Page:

1. Go to the [Media Monitoring page](https://eventregistry.org/monitoring) and create a new Topic Page
2. Add categories (Business, Technology, Science) and sources (VentureBeat, Business Insider, The Verge)
3. Copy the `uri` parameter from the URL
4. Add it to your `.env` as `NEWSAPI_TOPIC_URI`

You can refine your Topic Page later without code changes.

**Prompt:**

```
Implement the Fetch pipeline step.

Create a `PipelineStep` base class that defines the pattern for all steps: loading inputs, processing, persisting to database, and printing stats. Then implement `FetchStep` to retrieve articles from our newsapi.ai topic page and store them using the Article model.

We also need a Settings class (in `core/settings.py`; extends Pydantic BaseSettings) for configuration, with a module-level `settings` instance that other files can import directly. Add `newsapi_api_key` and `newsapi_topic_uri` settings.

The CLI's `run` command should create a new Run, execute FetchStep, and update the run status when done.

Reference @CLAUDE.md for conventions.
```

---

### Phase 4: AI-Powered Filtering

This is the first agentic step. The LLM reviews each article and decides whether it contains useful business signals (typically about half pass, depending on your Topic Page).

First, create the model:

**Prompt:**

```
Add the TriageDecision model (Base/Table pattern since it's an LLM output).

`TriageDecisionBase`: `article_id` (UUID), `keep` (boolean), `reason` (string).

`TriageDecision` adds: `id` (UUID), `run_id` (foreign key), `created_at`.

Add to ArticleRepository: `get_articles_pending_triage(run_id)` - returns articles from this run that don't have a triage decision yet.

Add `TriageDecisionRepository` with `create_many(decisions)`.

Tests for new repository methods. Create Alembic migration.
```

Now implement the step:

**Prompt:**

```
Implement TriageStep to filter articles for business relevance.

First, create `AIModelFactory` in `pipeline/ai_models.py` that returns the appropriate Gemini model for each step. Triage uses `gemini-2.5-flash-lite`.

TriageStep should:

* Load articles that are pending triage decisions  
* Process in batches of 40 articles  
* For each batch, call a Pydantic AI agent that returns structured `TriageDecisionBase` objects  
* Persist decisions to database

Important: When sending articles to the LLM, map article UUIDs to simple sequential indices (0, 1, 2...) and include that index in the prompt. The LLM returns decisions referencing these indices, which we map back to UUIDs before persisting. This avoids issues with LLMs truncating or mangling UUIDs.

The agent's system prompt should define a news editor persona filtering for startup/business relevance.

Update CLI to run TriageStep after FetchStep.
```

---

## About These Prompts

This README includes starter prompts to establish the project foundation: scaffolding, database setup, news ingestion, and the first AI-powered step (Triage).

The course walks through these same steps with full context — why we make certain decisions, how to evaluate what the AI generates, and when to push back — then continues through the remaining pipeline steps to completion. Your build will diverge from mine in subtle ways; that's expected and intentional. The prompts become more outcome-oriented as patterns establish themselves.

These files are your reference point. Keep them in your project root, and update CLAUDE.md as your codebase evolves.
