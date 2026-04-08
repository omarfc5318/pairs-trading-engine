# Deploying to Streamlit Cloud

This guide covers deploying the pairs trading dashboard to [Streamlit Community Cloud](https://streamlit.io/cloud) (free tier). The dashboard loads all pre-computed data from `outputs/` and `data/clean/` — no live computation happens on the server.

---

## Prerequisites

- A **GitHub account** (free)
- A **Streamlit Community Cloud account** (free, sign in with GitHub)
- All output files committed to your repository (see Step 1)

---

## Step 1 — Commit output files to GitHub

Streamlit Cloud clones your repository and serves the app from it. The dashboard requires pre-computed data files that are **not regenerated on the server**, so they must be in the repo.

**Files that must be committed:**

```
outputs/
├── backtest/
│   ├── daily_pnl.csv
│   ├── kalman_daily_pnl.csv
│   ├── regime_daily_pnl.csv
│   ├── trade_log.csv
│   ├── kalman_trade_log.csv
│   ├── regime_trade_log.csv
│   └── factor_results.json
└── pairs/
    ├── coint_results.csv
    └── selected_pairs.csv

data/
└── clean/
    ├── log_prices.parquet
    └── prices.parquet
```

If you haven't run the pipeline yet, generate them locally first:

```bash
pip install -r requirements.txt
python main.py --no-fetch   # if raw/ parquets exist
# or
python main.py              # full run including data fetch
```

Then push everything to GitHub:

```bash
git add outputs/ data/clean/ .streamlit/ dashboard/ requirements.txt packages.txt
git commit -m "Add pre-computed outputs for Streamlit Cloud deployment"
git push origin main
```

> **Note on file size:** All output files total ~3 MB, well within GitHub's 100 MB per-file limit and Streamlit Cloud's repository size limits.

---

## Step 2 — Push the repository to GitHub

If you haven't already created a GitHub repository:

1. Go to [github.com/new](https://github.com/new).

   *Screenshot description: GitHub's "Create a new repository" page. Fill in the "Repository name" field (e.g. `pairs-trading`), choose Public or Private, leave "Add a README" unchecked (you already have one), and click the green "Create repository" button.*

2. Copy the remote URL shown on the next page (e.g. `https://github.com/your-username/pairs-trading.git`).

3. From your local project root:

   ```bash
   git init                                              # if not already a git repo
   git remote add origin https://github.com/your-username/pairs-trading.git
   git add .
   git commit -m "Initial commit"
   git push -u origin main
   ```

   *Screenshot description: Terminal showing the push succeeding — output like `Branch 'main' set up to track remote branch 'main' from 'origin'` and `main -> main`.*

---

## Step 3 — Sign in to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io).

   *Screenshot description: The Streamlit Cloud landing page showing a "Sign in with GitHub" button in the center. Click it.*

2. Authorize the Streamlit OAuth app when GitHub prompts you.

   *Screenshot description: GitHub's OAuth authorization dialog showing "Streamlit wants to access your account". Click the green "Authorize streamlit" button.*

3. You land on the **My apps** dashboard — a dark-themed page listing any existing deployed apps (empty on first use).

---

## Step 4 — Create a new app

1. Click the **"New app"** button (top-right of the My apps page).

   *Screenshot description: The My apps page with a blue "New app" button in the upper right corner. Click it.*

2. The "Deploy an app" dialog appears with three fields:

   | Field | Value to enter |
   |---|---|
   | **Repository** | `your-username/pairs-trading` |
   | **Branch** | `main` |
   | **Main file path** | `dashboard/app.py` |

   *Screenshot description: A modal dialog titled "Deploy an app". The Repository field has a dropdown/search box — type your repo name and select it from the autocomplete list. Branch defaults to `main`. The "Main file path" field should be changed from the default `streamlit_app.py` to `dashboard/app.py`.*

3. Leave **App URL** as the auto-generated slug, or customise it (e.g. `pairs-trading-demo`).

4. Click **"Deploy!"**

   *Screenshot description: The completed form with all three fields filled in and the blue "Deploy!" button at the bottom right. Click it.*

---

## Step 5 — Watch the build logs

After clicking Deploy, Streamlit Cloud:

1. Clones your repository.
2. Installs system packages from `packages.txt` (none for this project).
3. Installs Python packages from `requirements.txt` (~30–90 seconds on first deploy).
4. Launches the Streamlit app.

*Screenshot description: A build log console showing lines like `Collecting streamlit==1.56.0`, `Installing collected packages: ...`, and finally `You can now view your Streamlit app in your browser` in green. The left sidebar shows a spinning indicator that turns into a green checkmark when the build succeeds.*

> **If the build fails:** Click "Manage app" → "Logs" to see the full traceback. Common issues are listed in the [Troubleshooting](#troubleshooting) section below.

---

## Step 6 — Verify the deployed app

Once the build completes, your app opens automatically at:

```
https://<your-slug>.streamlit.app
```

*Screenshot description: The live dashboard in a browser at the streamlit.app URL. The dark-themed sidebar shows the four navigation items (Strategy Overview, Pair Analysis, Risk Analysis, Factor Decomposition). The Strategy Overview page displays the performance comparison cards and equity curve chart.*

Walk through each page to confirm data loads correctly:

- **Strategy Overview** — equity curves render, donut chart shows 88.3%/11.7%
- **Pair Analysis** — dropdown works for all three pairs, spread/z-score/hedge charts appear
- **Risk Analysis** — cost sensitivity line chart, rolling Sharpe, max drawdown bar chart
- **Factor Decomposition** — factor loadings bar chart and regression table load from `factor_results.json`

---

## Step 7 — Share the URL

Copy the URL from the browser address bar and share it. The app is publicly accessible (if your repo is public) or accessible to anyone you invite via Streamlit Cloud's sharing settings.

To manage access: click **"Share"** (top-right of the app) → **"Invite viewers"** → enter GitHub usernames or email addresses.

*Screenshot description: A "Share this app" popover with an "Invite viewers" text field and a list of current viewers. There is also a toggle to make the app public (visible to anyone with the link).*

---

## Keeping the app updated

Streamlit Cloud automatically redeploys on every push to the tracked branch:

```bash
# Make local changes, then:
git add .
git commit -m "Update dashboard or outputs"
git push origin main
```

*Streamlit Cloud detects the push and triggers a redeployment within ~30 seconds. The app shows a "Rerunning" spinner during the update.*

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'hmmlearn'`
Your `requirements.txt` was not picked up. Make sure it is in the **repository root** (not inside `dashboard/`), and that the filename is exactly `requirements.txt`.

### `FileNotFoundError: outputs/backtest/daily_pnl.csv`
The pre-computed output files were not committed to the repository. Re-run Step 1 and push the `outputs/` directory. The app displays a descriptive error banner listing every missing file if this occurs at runtime.

### `st.error: Missing output files`
Same as above — the sanity check in `dashboard/app.py` caught a missing file. Commit the files listed in the error message.

### App loads but charts are blank
Usually a `pyarrow` version mismatch causing silent failures reading `.parquet` files. Pin `pyarrow==23.0.1` in `requirements.txt` (already done) and redeploy.

### Build times out (>10 min)
This can happen if `scikit-learn` or `scipy` have no pre-built wheel for the Streamlit Cloud Python version. The pinned versions in `requirements.txt` all have wheels for Python 3.11 on Linux x86_64 (the Streamlit Cloud environment).

---

## File reference

| File | Purpose |
|---|---|
| `requirements.txt` | Python dependencies with pinned versions |
| `packages.txt` | System-level apt packages (empty for this project) |
| `.streamlit/config.toml` | Dark theme + server settings read by Streamlit Cloud |
| `dashboard/app.py` | App entry point — set as "Main file path" in the deploy dialog |
| `outputs/` | Pre-computed backtest data — must be committed |
| `data/clean/` | Pre-computed price data — must be committed |
