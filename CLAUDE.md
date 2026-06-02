# CLAUDE.md

This file provides guidance to Claude Code (or other AI coding assistants) when working on this project.

## Project Overview

This is an agentic AI pipeline that transforms news articles into startup idea newsletters. The pipeline:

1. Fetches news articles from an API
2. Uses AI agents to filter, extract, and synthesize business insights
3. Generates a detailed startup concept with business model
4. Outputs a newsletter post as markdown with a generated header image

The pipeline is designed to run daily, processing fresh news into actionable startup concepts.

## Tech Stack

- **Python 3.13+** with **uv** for dependency management
- **Pydantic AI** for AI agent orchestration
- **SQLModel** for data models (combines Pydantic + SQLAlchemy)
- **SQLite** for local database storage
- **ChromaDB** for vector embeddings and similarity search
- **Google Gemini** as the LLM provider (via Pydantic AI)

## Project Structure

```
src/idea_pipeline/
├── core/                   # Shared components
│   ├── models.py           # SQLModel data models
│   └── settings.py         # Configuration (API keys, paths)
├── db/                     # Database layer
│   ├── database.py         # SQLite engine and session management
│   └── repositories.py     # Data access classes
├── pipeline/               # Pipeline implementation
│   ├── ai_models.py        # AI model factory and configuration
│   ├── base.py             # PipelineStep base class
│   └── steps.py            # Step implementations
└── cli.py                  # Command-line interface
tests/
├── conftest.py             # Pytest fixtures (in-memory DB, etc.)
└── test_*.py               # Test files
```

## Pipeline Steps

Each step reads its own inputs from the database and persists its outputs. Steps are independent and resumable.

1. **Fetch** — Retrieve news articles from News API (newsapi.ai, SDK: `eventregistry`), store in database
2. **Triage** — AI filters articles for business relevance (keep/discard decisions)
3. **Extraction** — AI extracts business signals and market facts from kept articles
4. **Synthesis** — AI generates 3 startup candidates from insights (uses ~2-day rolling window capped at 1000 insights, one per archetype: META_TREND, FRICTION_POINT, RABBIT_HOLE)
5. **Deep Dive** — Selects best candidate via scoring, AI expands into full business model
6. **Writer** — AI generates the final newsletter post as markdown
7. **Visualizer** — AI generates an image prompt and caption for the header image
8. **ImageGenerator** — AI generates the actual header image from the prompt
9. **PodcastScript** — AI generates a ~4-minute podcast script as a conversation between two hosts: Ryan (enthusiastic founder/angel) and Priya (dry, analytical ex-consultant, British accent). Writes to `output/podcast_script.txt`
10. **PodcastAudio** — Calls the Gemini TTS API directly (not via Pydantic AI) to synthesise the script into audio. Uses multi-speaker voice config (Ryan: Sadachbia, Priya: Gacrux). Converts raw PCM output to MP3 via ffmpeg. Writes to `output/podcast.mp3`

## Development Commands

```bash
# Install dependencies
uv sync

# Run the full pipeline
uv run idea-pipeline run

# Show help and available commands
uv run idea-pipeline --help

# Run tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Apply database migrations
uv run alembic upgrade head

# Create a new migration after model changes
uv run alembic revision --autogenerate -m "description"

# Check current migration status
uv run alembic current
```

The CLI is defined in `pyproject.toml` under `[project.scripts]`:
```toml
[project.scripts]
idea-pipeline = "idea_pipeline.cli:main"
```

## AI Model Selection

The `AIModelFactory` in `pipeline/ai_models.py` selects appropriate models for each step to optimize cost and quality. Use `gemini-2.5-pro` as the default, and only add explicit cases when a step requires a different model.

| Step           | Model                        | Rationale                                 |
|----------------|------------------------------|-------------------------------------------|
| Triage         | `gemini-2.5-flash-lite`      | High volume, simple yes/no decisions      |
| Extraction     | `gemini-2.5-flash`           | Structured output, moderate complexity    |
| Synthesis      | `gemini-2.5-pro`             | Creative ideation, needs strong reasoning |
| Deep Dive      | `gemini-2.5-pro`             | Complex business analysis                 |
| Writer         | `gemini-2.5-pro`             | Long-form content generation              |
| Visualizer     | `gemini-2.5-pro`             | Creative prompt generation                |
| ImageGenerator | `gemini-3-pro-image-preview` | Image generation model                    |
| PodcastScript  | `gemini-2.5-pro`             | Creative dialogue generation              |
| PodcastAudio   | `gemini-2.5-flash-preview-tts` | TTS — called directly via google-genai SDK, not Pydantic AI |

This keeps development costs low while maintaining quality where it matters.

## Key Conventions

### Data Models

- All models inherit from `SQLModel`
- Use the Base/Table pattern for models that interact with AI agents: `ArticleBase` (fields only) → `Article(ArticleBase, table=True)` (adds id, timestamps, primary key)
- Internal models (like `Run`) that never interact with AI agents can be a single table class—no Base needed
- All fields should include descriptions via `Field(description="...")` — these provide context to AI agents about what each field represents
- Models live in `core/models.py`

### Pipeline Steps

- Each step is a class inheriting from `PipelineStep`
- Steps are stateless: fetch input from DB, process, persist output to DB
- Steps are responsible for querying their own inputs (e.g., rolling windows)

### AI Agents

Use `GoogleModel` with `GoogleProvider` to create Gemini-powered agents:

```python
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from idea_pipeline.core.settings import settings

provider = GoogleProvider(api_key=settings.gemini_api_key)
model = GoogleModel("gemini-2.5-flash", provider=provider)

agent: Agent[None, MyOutput] = Agent(
    model,
    system_prompt="You are a helpful assistant.",
    output_type=MyOutput,
)
```

Key points:
- Use `GoogleProvider(api_key=...)` to pass the API key — do not use the deprecated `GeminiModel` class
- Use `AIModelFactory.get_model(step_key)` to get the appropriate model for each pipeline step
- System prompts should define a clear persona and task
- Use Pydantic models for structured responses (not raw text parsing)

### Database

- Use plural table names (e.g., `runs`, `articles`, `candidates`)
- Use repository pattern: `RunRepository`, `ArticleRepository`, etc.
- Repositories handle all SQL queries
- Never write raw SQL in pipeline steps

## Testing Philosophy

We focus on testing **deterministic, high-value logic** rather than AI agent behavior:

**What we test:**
- Scoring and ranking functions
- Data transformations and filtering logic
- Repository queries and database operations
- Input validation and edge cases

**What we don't test:**
- AI agent outputs—the "magic" is in the model; mocking it just tests that mocks return what you told them to
- Integration tests for AI steps—these would either make real API calls (expensive, slow, non-deterministic) or mock the calls (testing nothing meaningful). We trust Pydantic AI to do what its docs say.

**Database testing:** SQLite makes testing simple. Use an **in-memory database** (`:memory:`) for fast, isolated tests with no cleanup required.

**TDD approach:** When implementing deterministic logic (scoring, filtering, transformations), prefer writing tests first. Write the test, verify it fails, then implement the code to make it pass.

## Environment Variables

Required in `.env`:
```
NEWSAPI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

## Reminders for Every Change

When making changes to the codebase:

1. **Tests**: Add or update tests for deterministic logic; run `uv run pytest` to verify all tests pass
2. **Models**: If you add/modify data models, follow the Base/Table pattern for AI-facing models; internal models can be a single table class
3. **Steps**: If you add/modify pipeline steps:
   - Update this file to reflect the change
   - Ensure the step executes when running the CLI
   - Add the step's model selection to `AIModelFactory` (only if it differs from the Gemini 2.5 Pro default)
4. **Dependencies**: Use `uv add <package>` to add new dependencies
5. **Pipeline**: Never run `uv run idea-pipeline run` yourself; suggest to the user to run it instead (incurs API costs and can be slow)

## TODO

- **Revisit Synthesis rolling window**: The Synthesis step currently uses a 2-day window (capped at 1000 insights) as a temporary measure to avoid overwhelming the model with overly verbose insights. Once the terser extraction prompts have been validated over a few runs, revert to a 4-day rolling window and re-evaluate whether the 1000-insight cap is still needed.
