# multimodal-clickbait-detection

Multimodal machine learning system for detecting misleading YouTube content by analyzing titles, thumbnails, descriptions, tags, transcripts, and viewer comments to estimate clickbait risk and explain cross-modal inconsistencies.

## Dataset collection

The repository includes a small collector for a manually curated, labeled set. Team members find videos and enter their YouTube URLs, category, and human-assigned label in `data/submissions.csv`. The collector does not infer labels.

### Setup

1. Use Python 3.10 or newer and install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Create a YouTube Data API v3 key in Google Cloud and enable the YouTube Data API v3 for that project. Copy `.env.example` to `.env` and set `YOUTUBE_API_KEY`. Keep `.env` private; it is ignored by Git.

3. Review the ten canonical category names in `dataset_config.json`; submitted category names match case-insensitively and are saved using this canonical spelling.

4. Fill `data/submissions.csv` (`video_url,category,label,reviewer,notes`). Use one row per video and a category from the config (category matching ignores case and surrounding whitespace). Labels accept `0`/`normal` for normal and `1`/`clickbait` for clickbait; text labels ignore case and surrounding whitespace. Labels are stored as integers (`0` or `1`). The column remains named `label`.

5. Collect data:

   ```bash
   python scripts/collect_dataset.py
   ```

   To set the input path or maximum comments per collection:

   ```bash
   python scripts/collect_dataset.py --input data/submissions.csv --max-comments 20
   ```

   The script validates the entire CSV before contacting YouTube, reports malformed rows, and checks duplicates across URL formats and against `videos.json`. Every duplicate CSV occurrence is skipped. Existing video IDs are skipped unless `--refresh` is supplied; conflicting labels are never overwritten. It saves `videos.json` atomically after each video. Public API errors and unavailable modalities are recorded in the manifest; a failed video does not stop later rows.

6. Check the target counts:

   ```bash
   python scripts/collect_dataset.py --audit
   ```

   The audit expects 10 configured categories × 100 videos, with 50 normal (`0`) and 50 clickbait (`1`) per category. It reports gaps; it does not fabricate or relabel examples.

### Files written

- `videos.json`: one record per accepted video, including metadata, label provenance, and relative paths.
- `Thumbnail/<video_id>.jpg`: downloaded thumbnail (format follows the URL extension).
- `transcript/<video_id>.json`: transcript segments and transcript status.
- `comments/<video_id>_top.json` and `comments/<video_id>_newest.json`: up to 20 top-level comments each, ordered by relevance and publication time respectively. A comment may appear in both lists because they use different selection criteria. Author names and channel IDs are not collected. Disabled or unavailable comments are recorded without failing video collection.

Generated media and annotation data should only be committed after the team has reviewed labels and checked applicable YouTube API terms and dataset-sharing expectations. Never commit API keys. The collector writes files locally; review the changes before making a Git commit.

### Annotation guidance

Use consistent written criteria for what counts as clickbait before labeling. Judge whether the title and thumbnail materially mislead or exaggerate the video's content, rather than whether the video is popular or entertaining. Resolve disagreements with a second reviewer and document the adjudicated label and rationale in the CSV notes. Keep categories mutually understandable and apply them consistently.

