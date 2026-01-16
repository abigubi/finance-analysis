# Deployment Guide - Falling Knife Detector

## 🚀 Streamlit Cloud Deployment

### Main File Path
```
Quant/Falling_Knife_Web.py
```

### Settings:
- **Repository:** `your-username/finance-analysis` (or your repo name)
- **Branch:** `main` (or `master`)
- **Main file path:** `Quant/Falling_Knife_Web.py`
- **App URL (optional):** Leave empty or set custom name

### Advanced Settings (if needed):
- **Python version:** 3.9 or higher
- **Dependencies:** Automatically installs from `requirements.txt`

### Steps:
1. Push your code to GitHub
2. Go to https://share.streamlit.io
3. Click "New app"
4. Connect your GitHub account
5. Select repository: `finance-analysis`
6. Branch: `main`
7. Main file path: `Quant/Falling_Knife_Web.py`
8. Click "Deploy"

---

## 🟢 Replit Deployment

### Setup:
1. Import your GitHub repository to Replit
2. Replit will automatically detect `requirements.txt`
3. The `.replit` file will configure the run command
4. Click "Run" button

### Manual Setup (if needed):
1. In Replit Shell, run:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run Quant/Falling_Knife_Web.py --server.port=8501 --server.address=0.0.0.0
   ```

### Replit Webview:
- Replit will automatically show the app in the Webview panel
- The app will be accessible at the provided URL

---

## 📦 Required Files

Make sure these files are in your repository:

```
Finance Analysis/
├── requirements.txt          # Dependencies
├── .replit                   # Replit configuration
├── replit.nix                # Replit environment
├── Quant/
│   └── Falling_Knife_Web.py  # Main app file
└── README.md                 # Optional
```

---

## 🔧 Troubleshooting

### Streamlit Cloud:
- **Error: "This file does not exist"**
  - Check the main file path is correct: `Quant/Falling_Knife_Web.py`
  - Make sure the file is committed and pushed to GitHub
  - Verify the branch name matches

### Replit:
- **App won't start:**
  - Check that `requirements.txt` is in the root directory
  - Verify `.replit` file exists
  - Try running `pip install -r requirements.txt` manually

- **Port issues:**
  - Replit uses `$PORT` environment variable
  - The `.replit` file should handle this automatically

---

## 📝 Notes

- The app uses `yfinance` which requires internet connection
- No API keys needed
- Works on both Replit and Streamlit Cloud
- Free tier available on both platforms
