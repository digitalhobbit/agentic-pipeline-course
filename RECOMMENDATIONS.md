# Recommendations

**Last Updated:**

 July 16, 2026

The AI space evolves quickly. While the course was recorded with specific model and tool versions, the principles are deliberately tool agnostic and still apply. This lesson holds the current model version recommendations. These supersede the guidance in the `CLAUDE.md` file.

## Gemini API

The guidance in the `CLAUDE.md` and course videos is still completely valid. This mainly uses Gemini 2.5 Pro, Flash, and Flash Lite depending on the step, as well as Gemini 3 Pro Image Preview for images.

That said, I suggest the following substitutions:

* **Extraction:**
  * Old: `gemini-2.5-flash`
  * New: `gemini-3.1-flash-lite`
* **Synthesis, Deep Dive, Writer, etc.:**
  * Old: `gemini-2.5-pro`
  * New: `gemini-3.5-flash`
* **Embedding:**
  * Old: `gemini-embedding-001`
  * New: `gemini-embedding-2`
  * Note: Don't change this after you already have embeddings created with the older model, as these aren't compatible with the new ones. You would need to run a backfill that replaces the old embeddings with new ones.  
* **Image Generation:**
  * Old: `gemini-3-pro-image-preview`
  * New: `gemini-3-pro-image`

As you see, lower-tier models in a newer generation often perform better than higher-tier models of the previous generation. So 3.1 Flash-Lite model is a good replacement for the 2.5 Flash model, and 3.5 Flash comfortably replaces 2.5 Pro. Overall cost should be roughly equivalent with these substitutions. You can also refer to the [Gemini Developer API pricing page](https://ai.google.dev/gemini-api/docs/pricing) for details.

As stated in the video, I generally avoid preview models, as these can have lower availability and rate limits.

## Claude Code

The course videos mainly use Claude Code with Opus 4.6. The most recent Opus version is 4.8 (released on May 28, 2026). The key difference between Opus 4.6 and Opus 4.7/4.8 is that **Anthropic recommends using a higher effort level ("High" or "xHigh" instead of "Medium") with the newer models**. I use "xHigh" most of the time.

Also note that Claude Code itself (independent of the model) is evolving rapidly. Anthropic frequently tests new features and UI changes, so the UI may look slightly different than in the videos.
