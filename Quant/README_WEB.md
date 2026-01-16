# Falling Knife Detector - Web Application

A beautiful, interactive web application for detecting long-term bottom opportunities in stocks using panic signal analysis.

## Features

- 🌐 **Web-based Interface**: Run in your browser, no command line needed
- 📊 **Interactive Charts**: Zoom, pan, and hover for detailed information
- 🔄 **Dynamic Ticker Switching**: Easily switch between different stocks
- 📈 **Real-time Analysis**: Instant panic cluster detection and severity scoring
- 🎯 **Strong Bottom Detection**: Identify the best swing entry points

## Installation

1. Install the required packages:
```bash
pip install -r requirements_web.txt
```

Or install individually:
```bash
pip install streamlit plotly pandas numpy yfinance
```

## Running the Application

1. Navigate to the Quant directory:
```bash
cd Quant
```

2. Run the Streamlit app:
```bash
streamlit run Falling_Knife_Web.py
```

3. The app will automatically open in your default browser at `http://localhost:8501`

## Usage

1. **Enter Ticker**: Type any stock ticker symbol in the sidebar (e.g., AAPL, TSLA, MSFT)
2. **Set Date Range**: Choose your start and end dates for historical analysis
3. **Click Analyze**: Click the "Analyze" button to process the ticker
4. **Explore Results**: 
   - View the interactive chart with panic signals
   - Check the metrics dashboard
   - Review cluster details in the table
   - Expand strong bottom candidates for detailed information

## Features Explained

- **Severity 2 (▲)**: Serious panic clusters - multiple panic types or clustering
- **Severity 3 (●)**: Full capitulation - all three panic conditions met
- **Strong Bottom (*)**: Best swing entry points - long-term bottom candidates

## Hosting Options

### Local Network Access
To access from other devices on your network:
```bash
streamlit run Falling_Knife_Web.py --server.address 0.0.0.0
```

### Cloud Hosting
You can deploy this app to:
- **Streamlit Cloud** (free): https://streamlit.io/cloud
- **Heroku**: Use a Procfile and requirements.txt
- **AWS/GCP/Azure**: Deploy as a containerized app
- **PythonAnywhere**: Upload and run as a web app

### Docker (Optional)
Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements_web.txt .
RUN pip install -r requirements_web.txt
COPY Quant/Falling_Knife_Web.py .
EXPOSE 8501
CMD ["streamlit", "run", "Falling_Knife_Web.py", "--server.address", "0.0.0.0"]
```

## Tips

- Use the chart's zoom and pan features to focus on specific time periods
- Hover over markers to see detailed information
- Strong bottom candidates are the best opportunities for swing trading
- Try different date ranges to see how signals change over time
