# multimodal-clickbait-detection

Multimodal machine learning system for detecting misleading YouTube content using video metadata, thumbnails, transcripts, and viewer comments.

## Dataset collection

`data/submissions.csv` is the source of truth for the submitted videos and human annotations. Each run reads the current CSV, so adding, removing, or editing its rows does not require editing `videos.json` by hand. Existing modality files are reused when the video ID is already complete. Labels and categories come from the CSV and are never inferred by the collector.

### Setup

1. Install Python 3.10+ and dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env`. Set `YOUTUBE_API_KEYS` to comma-separated YouTube Data API v3 keys and set `YOUTUBE_API_KEY_PROJECTS` to the corresponding Google Cloud project IDs in the same order. Keys mapped to one project share its quota; different project IDs get separate daily ledgers. The collector rotates away from invalid credentials, stops using every key assigned to a project after that project's quota is exhausted, and continues with another configured project. `.env` is ignored by Git. For one project, one `YOUTUBE_QUOTA_PROJECT` value remains supported as a fallback.

3. The fixed dataset profile in `dataset_config.json` matches the current CSV: 11,194 videos, 11 categories, and 5,597 examples per label. The `video_url,category,label` columns are sufficient; `reviewer` and `notes` are optional. The collector checks the actual totals against the profile and reports mismatches without changing the CSV or fabricating or dropping records.

4. Run the collector:

   ```bash
   python scripts/collect_dataset.py
   ```

   It validates all rows before making requests, reports malformed and duplicate IDs, retrieves metadata in batches of 50, collects up to 25 relevance-ranked and 25 newest top-level comments separately, downloads the best available thumbnail, and fetches English-first transcripts without using YouTube API quota. Set `--input`, `--max-comments`, `--daily-quota`, or `--quota-buffer` to override defaults.

5. Run it again after editing the CSV to synchronize the final manifest with the current submitted rows. Previously completed modalities are skipped. Interrupted runs can be restarted safely; incomplete work continues from the checkpoint. When all configured projects hit their daily quota limits, the program saves its checkpoint, waits for YouTube's midnight Pacific Time quota reset, then resumes automatically. Use `--no-wait-for-quota` to save and exit instead. Quota use is tracked separately by project ID.

### Outputs

- `videos.json`: final current submission records, preserving categories, labels, reviewer, notes, metadata, modality paths, and collection statuses.
- `collection_state.json`: local per-video modality checkpoint and quota ledger (ignored by Git).
- `collection_report.json`: latest dataset-profile check, modality completion/errors, per-project quota use, and remaining work (ignored by Git).
- `Thumbnail/`, `transcript/`, and `comments/`: modality files in the existing directory layout.

Only `videos.list` and `commentThreads.list` are charged to YouTube Data API quota. A metadata batch costs one unit; each comment page costs one unit. Extraction stops before the configured safety buffer and saves progress. YouTube's project quota error also pauses collection until quota resets. API keys are never written to output files or included in error text.

Review generated data and applicable YouTube API terms before sharing or committing it. Never commit `.env` or API credentials.
