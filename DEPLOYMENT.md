# Website and local inference

The site is a static Vercel front end. FastAPI and all four model components run on the computer where `start_local_api.ps1` is running. Vercel does not receive the model weights or run inference.

## Deploy the website

1. In Vercel, import `Vasu-Sagwal/multimodal-clickbait-detection` from GitHub.
2. Set **Root Directory** to `web`, choose the **Other** framework preset, and deploy. The site is static and needs no build command or environment secrets.
3. Use the resulting `https://…vercel.app` URL to open the page.

## Run inference on this computer

1. Clone the repository on the computer that will run the models, then download its Git LFS weights:

   ```powershell
   git lfs install
   git lfs pull --include="models/text_classifier/best_model.pt"
   ```

   `models/text_classifier/best_model.pt` must be the real model file, not the small LFS pointer. The pipeline also needs `models/ann_model.joblib`, `models/resnet18_best.pt`, `models/fusion_weights.joblib`, and `models/text_classifier/tokenizer/`.

2. Install the local inference dependencies:

   ```powershell
   python -m pip install -r requirements-local-api.txt
   ```

3. Copy `.env.example` to `.env` and set `YOUTUBE_API_KEY`. Keep `.env` private and out of GitHub. If you use a custom website domain, add it to `WEB_ALLOWED_ORIGINS` in `.env`.
4. Run `start_local_api.ps1` and leave its window open. It loads the models once and prints a local API token.
5. On the Vercel page, open **Connect your local model**, enter `http://127.0.0.1:8000`, paste the token, and choose **Check connection**. If Chrome or Edge asks to let the site access the local network, allow it.
6. Paste a YouTube video URL and choose **Analyze video**.

The browser sends the URL to FastAPI on the same computer. FastAPI fetches the public video details and thumbnail, runs the saved text, image, metadata, and fusion models locally, then returns only the prediction and scores to the page. The API key and token stay on the local computer; the token is not stored by the site.

If you open the site on a different computer and want inference to run on this computer, `127.0.0.1` will point to the other computer. You can instead run a Cloudflare quick tunnel on the model computer (`cloudflared tunnel --url http://127.0.0.1:8000`) and enter its HTTPS URL on the site. Keep the API token private; anyone with that token can submit inference requests while the tunnel is active. Your computer and both local processes must stay online for the tunnel to work.

## Verify the pipeline

Run the existing pipeline directly from the repository root:

```powershell
python testing/test_pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

This command uses the local `.env` key and the saved model files. The FastAPI service calls the same pipeline functions and loads the models once at startup.
