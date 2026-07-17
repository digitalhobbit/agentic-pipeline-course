# The Agentic Pipeline - Reference Implementation

An autonomous AI pipeline that turns each day's news into a fully-formed startup idea — complete with a business model, a written newsletter post, a generated header image, and a two-host podcast.

This is the reference implementation from **The Agentic Pipeline** course by [GammaVibe](https://gammavibe.com). It shows **one possible end state**: a working build you can run, read, and learn from. Your own build will diverge based on the decisions you make, what the AI generates, and how you iterate — and that's the point.

- 🎓 **The course — free on YouTube:** [gammavibe.com/agentic-pipeline-course](https://gammavibe.com/agentic-pipeline-course/) — who it's for, what to expect, and the link to the full video series.
- 📘 **Free blueprint:** [gammavibe.com/blueprint](https://gammavibe.com/blueprint) — the architecture, tech stack, and starter prompts (the same files that bootstrapped this repo).
- 🚀 **See it in action:** [gammavibe.com](https://gammavibe.com) — daily posts, images, and podcasts from the production pipeline, plus the full archive.

> **This repo is the destination; the course is the journey.** The README you're reading describes how to run the finished pipeline. To build it yourself from scratch, [take the free course](https://gammavibe.com/agentic-pipeline-course/) — everything you need to follow along is in this repo (see [Build It Yourself](#build-it-yourself)).

---

## How It Works

The pipeline runs daily. It fetches news from curated sources, uses AI agents to filter articles and extract business signals, synthesizes startup candidates, expands the best one into a full business model, and produces a complete newsletter — written post, header image, and podcast. Every step persists its output to a local database, making the pipeline resumable and debuggable.

(See the [architecture diagram](#diagrams) below.)

---

## Tech Stack

The course teaches the simplified stack with SQLite and ChromaDB and markdown output. The production version uses PostgreSQL and pgvector and publishes to Ghost CMS.

(See the [tech stack diagram](#diagrams) below.)

---

## Quickstart

### Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** for dependency management
- **ffmpeg** — used by the PodcastAudio step to convert the generated audio to MP3 (`brew install ffmpeg`, `sudo apt install ffmpeg`, or `winget install ffmpeg`)
- **Gemini API key** — get a free key at [Google AI Studio](https://aistudio.google.com/api-keys); note: the free tier has rate limits; for sustained use, set up billing in Google Cloud Console
- **NewsAPI.ai API key + Topic Page URI** — sign up at [newsapi.ai](https://newsapi.ai), then create a Topic Page and copy its URI. See [Phase 3: News Ingestion](BLUEPRINT.md#phase-3-news-ingestion) in the blueprint for the Topic Page setup steps. *(For daily production runs you'll want a paid plan — [this referral link](https://gammavibe.com/go/newsapi) gets you a discounted $40 plan; I get a small commission.)*

### Setup

```bash
# 1. Install dependencies
uv sync

# 2. Configure your API keys
cp env.example .env
# then open .env and fill in GEMINI_API_KEY, NEWSAPI_API_KEY, and NEWSAPI_TOPIC_URI

# 3. Create the database
uv run alembic upgrade head

# 4. Run the full pipeline
uv run idea-pipeline run
```

Generated output lands in `output/` (newsletter markdown, header image, podcast script + MP3). The SQLite database and vector store live in `data/`.

> **Note:** AI models evolve quickly. [RECOMMENDATIONS.md](RECOMMENDATIONS.md) lists the currently recommended Gemini models — they supersede the choices baked into this repo and shown in the course videos.

### Useful commands

```bash
uv run idea-pipeline run --verbose   # detailed step-by-step logging
uv run idea-pipeline run --resume    # resume the most recent failed/incomplete run
uv run idea-pipeline --help          # all options
uv run pytest                        # run the test suite
```

---

## Build It Yourself

This repo is the finished reference. The **free video course** walks through building it from an empty directory using agentic engineering — why each decision is made, how to evaluate what the AI produces, and when to push back.

**Start at [gammavibe.com/agentic-pipeline-course](https://gammavibe.com/agentic-pipeline-course/)** — who the course is for, what to expect, and the link to the videos on YouTube.

### Starter files

The [starter_files/](starter_files/) folder contains everything you need to follow along. The **Project Setup** video (the second video in the course) has instructions for how to use these files.

| File | Description |
|------|-------------|
| [.gitignore](starter_files/.gitignore) | Ignore data and output folders, env file, and Python artifacts from your Git repository |
| [env.example](starter_files/env.example) | Placeholders for environment variables (API keys etc.); copy this to `.env` to add your own keys |
| [CLAUDE.md](starter_files/CLAUDE.md) | Guidance for Claude Code (you can also rename this to `AGENTS.md` to work with other coding agents) |
| [README.md](starter_files/README.md) | Project info and the first few Claude Code prompts that you can copy & paste |
| [Diagram.png](starter_files/Diagram.png) | The pipeline architecture and steps |
| [TechStack.png](starter_files/TechStack.png) | Overview of the tech stack we'll use for the course |

### Course companion materials

- **[RECOMMENDATIONS.md](RECOMMENDATIONS.md)** — up-to-date AI model recommendations and current guidance on Claude Code. Models have evolved since the videos were recorded; this file supersedes the model choices shown in the course and in `CLAUDE.md`.
- **[diagrams/](diagrams/)** — all the diagrams that appear in the course, usually prefixed by module name (e.g. `Extraction - ...`).
- **[BLUEPRINT.md](BLUEPRINT.md)** — the architecture, conventions, and the first few build prompts.
- **[CLAUDE.md](CLAUDE.md)** — the full project conventions that guide the coding agent.

---

## Diagrams

**Pipeline architecture**

![How It Works](Diagram.png)

*The diagram shows the core pipeline. The repo also includes bonus **PodcastScript** and **PodcastAudio** steps that turn the finished idea into a two-host audio episode.*

**Tech stack**

![Tech Stack](TechStack.png)

*Looking for more? All the diagrams that appear in the course are in the [diagrams/](diagrams/) folder.*

---

## License

Licensed under the MIT License — see [LICENSE](LICENSE) for details.
