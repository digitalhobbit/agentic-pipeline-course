# Recommendations

**Last Updated:**

 July 27, 2026

The AI space evolves quickly. While the course was recorded with specific model and tool versions, the principles are deliberately tool agnostic and still apply. This lesson holds the current model version recommendations. These supersede the guidance in the `CLAUDE.md` file.

## Gemini API

The guidance in the `CLAUDE.md` and course videos is still completely valid. This mainly uses Gemini 2.5 Pro, Flash, and Flash Lite depending on the step, as well as Gemini 3 Pro Image Preview for images.

That said, I suggest the following substitutions:

* **Extraction:**
  * Old: `gemini-2.5-flash`
  * New: `gemini-3.5-flash-lite`
* **Synthesis, Deep Dive, Writer, etc.:**
  * Old: `gemini-2.5-pro`
  * New: `gemini-3.6-flash`
* **Embedding:**
  * Old: `gemini-embedding-001`
  * New: `gemini-embedding-2`
  * Note: Don't change this after you already have embeddings created with the older model, as these aren't compatible with the new ones. You would need to run a backfill that replaces the old embeddings with new ones.  
* **Image Generation:**
  * Old: `gemini-3-pro-image-preview`
  * New: `gemini-3-pro-image`

As you see, lower-tier models in a newer generation often perform better than higher-tier models of the previous generation. So 3.1 Flash-Lite model is a good replacement for the 2.5 Flash model, and 3.5 Flash comfortably replaces 2.5 Pro. Overall cost should be roughly equivalent with these substitutions. You can also refer to the [Gemini Developer API pricing page](https://ai.google.dev/gemini-api/docs/pricing) for details.

As stated in the video, I generally avoid preview models, as these can have lower availability and rate limits.

## Local Models

The repo can also run the text steps against local models served by [LM Studio](https://lmstudio.ai/). This is experimental and the quality is well below the Gemini models, but it lets you run the pipeline without API costs. Use `uv run idea-pipeline run --provider lmstudio`, or set `LLM_PROVIDER=lmstudio` in your `.env`.

The models are configured in `src/idea_pipeline/pipeline/ai_models.py`:

* **Triage, Extraction:** `google/gemma-4-e4b`
* **Everything else:** `google/gemma-4-12b-qat`

Image generation, text to speech, and embeddings have no local equivalent yet, so those steps always use Gemini. You still need a `GEMINI_API_KEY`.

**Important:** in LM Studio, lower the sampling settings for `gemma-4-12b-qat` to **temperature 0.35** and **Top P Sampling 0.9**. The defaults (1.0 and 0.95) are too loose. Pydantic AI has LM Studio constrain the output to a JSON schema, and that schema lets the model end a text field at any point. At the default temperature it eventually picks a closing quote mid-sentence, which cuts the newsletter short after two or three sections. The result still parses as valid JSON, so nothing in the pipeline flags it as a failure.

Depending on your hardware, you may want to experiment with different local models, such as Gemma 4 31B or Qwen 3.6.

## Claude Code

The course videos mainly use Claude Code with Opus 4.6. The most recent Opus version is 5.0 (released on July 24, 2026). The key difference between Opus 4.6 and Opus 4.7/4.8/5.0 is that **Anthropic recommends using a higher effort level ("High" or "xHigh" instead of "Medium") with the newer models**. I use "xHigh" most of the time.

Also note that Claude Code itself (independent of the model) is evolving rapidly. Anthropic frequently tests new features and UI changes, so the UI may look slightly different than in the videos.
