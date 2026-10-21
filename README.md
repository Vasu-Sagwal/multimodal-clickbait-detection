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

3. Replace the ten example category names in `dataset_config.json` with the project's actual categories.

4. Fill `data/submissions.csv`. Use one row per video, a category from the config, and exactly `normal` or `clickbait` for the label. Assign labels using the team's written annotation guidelines and record the annotator in `reviewer`.

5. Collect data:

   ```bash
   python scripts/collect_dataset.py
   ```

   To set the input path or limit comments per video:

   ```bash
   python scripts/collect_dataset.py --input data/submissions.csv --max-comments 100
   ```

   The script is resumable: existing video IDs are skipped unless `--refresh` is supplied. It saves `videos.json` after each row. Public API errors and unavailable modalities are recorded in the manifest; a failed video does not stop later rows.

6. Check the target counts:

   ```bash
   python scripts/collect_dataset.py --audit
   ```

   The audit expects 10 configured categories × 20 videos, with 10 normal and 10 clickbait per category. It reports gaps; it does not fabricate or relabel examples.

### Files written

- `videos.json`: one record per accepted video, including metadata, label provenance, and relative paths.
- `Thumbnail/<video_id>.jpg`: downloaded thumbnail (format follows the URL extension).
- `transcript/<video_id>.json`: transcript segments and transcript status.
- `comments/<video_id>.json`: up to the configured number of top-level comments. Author names and channel IDs are not collected.

Generated media and annotation data should only be committed after the team has reviewed labels and checked applicable YouTube API terms and dataset-sharing expectations. Never commit API keys. The collector only commits files locally; review the diff before making a Git commit.

### Annotation guidance

Use consistent written criteria for what counts as clickbait before labeling. Judge whether the title and thumbnail materially mislead or exaggerate the video's content, rather than whether the video is popular or entertaining. Resolve disagreements with a second reviewer and document the adjudicated label and rationale in the CSV notes. Keep categories mutually understandable and apply them consistently.
