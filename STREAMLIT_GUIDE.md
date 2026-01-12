# Streamlit Deployment Guide

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r streamlit_requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
Final_Year_Project/
├── app.py                          # Main Streamlit application
├── streamlit_requirements.txt      # Dependencies
├── pages/                          # Multi-page components
│   ├── __init__.py
│   ├── prediction.py              # Prediction interface
│   ├── comparison.py              # Model comparison
│   └── explorer.py                # Data explorer
├── utils/                          # Utility functions
│   ├── __init__.py
│   └── model_loader.py            # Model loading utilities
├── .streamlit/                     # Streamlit configuration
│   └── config.toml                # Theme and server settings
├── models/                         # Trained models
│   ├── cnn_model.keras
│   ├── gru_model.keras
│   └── hybrid_model.keras
├── config/                         # Configuration files
│   ├── crop_zone_suitability_5crops.json
│   └── regions_and_state.json
└── project_data/                   # Data directory
    └── processed_data/
```

---

## 🌟 Features

### 1. Home Page
- Project overview and key metrics
- Model architecture explanations
- Quick navigation guide

### 2. Prediction Interface
- Interactive input forms for climate variables
- Crop and zone selection
- Model selection (CNN, GRU, Hybrid)
- Real-time predictions with confidence scores

### 3. Model Comparison
- Performance metrics comparison
- Side-by-side prediction analysis
- Detailed architecture documentation

### 4. Data Explorer
- Dataset overview and statistics
- Climate data visualizations
- Crop yield distributions
- Temporal trend analysis

---

## 🔧 Configuration

### Theme Customization

Edit `.streamlit/config.toml` to customize:
- Primary color (default: green for agriculture)
- Background colors
- Font styles
- Server settings

### Model Integration

To integrate actual predictions:

1. Update `utils/model_loader.py` with your preprocessing pipeline
2. Ensure preprocessing matches training methodology
3. Add scaling/normalization as needed
4. Update feature preparation in `prepare_input()`

---

## 📊 Deployment Options

### Option 1: Streamlit Cloud (Easiest)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy! (Free tier available)

**Pros:**
- Free hosting
- Automatic HTTPS
- Easy updates via git push
- No server management

### Option 2: Heroku

```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

### Option 3: AWS EC2 / Azure / GCP

```bash
# On server
pip install -r streamlit_requirements.txt
streamlit run app.py --server.port 80 --server.address 0.0.0.0
```

### Option 4: Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install -r streamlit_requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

---

## 🔐 Production Checklist

- [ ] Add authentication if needed (streamlit-authenticator)
- [ ] Set up HTTPS (via cloud provider or nginx reverse proxy)
- [ ] Configure proper error handling
- [ ] Add logging for predictions
- [ ] Optimize model loading (cache models)
- [ ] Set resource limits (memory, CPU)
- [ ] Add monitoring (e.g., Datadog, New Relic)
- [ ] Implement rate limiting if public
- [ ] Add user analytics (optional)
- [ ] Set up backup strategy

---

## 🎨 Customization Tips

### Add New Pages

Create new file in `pages/` directory:

```python
# pages/new_page.py
import streamlit as st

def show_new_page():
    st.title("New Feature")
    # Your code here
```

Update `app.py` navigation:

```python
elif page == "New Feature":
    from pages import new_page
    new_page.show_new_page()
```

### Add Visualizations

Use Plotly for interactive charts:

```python
import plotly.express as px

fig = px.line(data, x='year', y='yield')
st.plotly_chart(fig)
```

---

## 🐛 Troubleshooting

### Models Not Loading

- Check `models/` directory exists
- Verify `.keras` files are present
- Ensure TensorFlow version compatibility

### Import Errors

```bash
pip install --upgrade -r streamlit_requirements.txt
```

### Port Already in Use

```bash
streamlit run app.py --server.port 8502
```

### Slow Loading

- Add `@st.cache_data` to expensive operations
- Preload models at startup
- Optimize data loading

---

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/get-started)
- [Component Library](https://streamlit.io/components)

---

## 🆘 Support

For issues or questions:
1. Check [Streamlit Forum](https://discuss.streamlit.io)
2. Review [GitHub Issues](https://github.com/streamlit/streamlit/issues)
3. Read project documentation

---

**Ready to deploy!** 🎉

Run `streamlit run app.py` to start developing.
