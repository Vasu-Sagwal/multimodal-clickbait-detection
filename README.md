# multimodal-clickbait-detection

Multimodal machine learning system for detecting misleading YouTube content using video metadata, thumbnails, transcripts, and viewer comments.

## Dataset collection

`data/submissions.csv` is the source of truth for the submitted videos and human annotations. Each run reads the current CSV, so adding, removing, or editing its rows does not require editing `videos.json` by hand. Existing modality files are reused when the video ID is already complete. Labels and categories come from the CSV and are never inferred by the collector.

### Setup

1. Install Python 3.10+ and dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and set `YOUTUBE_API_KEYS` to one or more comma-separated YouTube Data API v3 keys. Keys should belong to one Google Cloud project: keys within a project share its quota. The collector rotates away from invalid credentials but never changes keys to bypass project quota. `.env` is ignored by Git.

3. Prepare `data/submissions.csv`. It accepts a `video_url`, `url`, or `video_id` column and requires `category` and `label`; `reviewer` and `notes` are optional. Other CSV columns are preserved only as input and do not affect collection. Labels accept `0`/`normal` and `1`/`clickbait`. Categories listed in `dataset_config.json` use their canonical spelling; other category values remain unchanged and appear as validation warnings.

4. Run the collector:

   ```bash
   python scripts/collect_dataset.py
   ```

   It validates all rows before making requests, reports malformed and duplicate IDs, retrieves metadata in batches of 50, collects up to 25 relevance-ranked and 25 newest top-level comments separately, downloads the best available thumbnail, and fetches English-first transcripts without using YouTube API quota. Set `--input`, `--max-comments`, `--daily-quota`, or `--quota-buffer` to override defaults.

5. Run it again after editing the CSV to synchronize the final manifest with the current submitted rows. Previously completed modalities are skipped. Interrupted runs can be restarted safely; incomplete work continues from the checkpoint. On daily quota exhaustion, the program saves its checkpoint, waits for YouTube's midnight Pacific Time quota reset, then resumes automatically. Use `--no-wait-for-quota` to save and exit instead. The quota ledger is stored by `YOUTUBE_QUOTA_PROJECT`.

### Outputs

- `videos.json`: final current submission records, preserving categories, labels, reviewer, notes, metadata, modality paths, and collection statuses.
- `collection_state.json`: local per-video modality checkpoint and quota ledger (ignored by Git).
- `collection_report.json`: latest validation, completion, error, quota, and remaining-work report (ignored by Git).
- `Thumbnail/`, `transcript/`, and `comments/`: modality files in the existing directory layout.

Only `videos.list` and `commentThreads.list` are charged to YouTube Data API quota. A metadata batch costs one unit; each comment page costs one unit. Extraction stops before the configured safety buffer and saves progress. YouTube's project quota error also pauses collection until quota resets. API keys are never written to output files or included in error text.

Review generated data and applicable YouTube API terms before sharing or committing it. Never commit `.env` or API credentials.
