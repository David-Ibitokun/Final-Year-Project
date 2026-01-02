# Django Deployment Checklist

## Pre-Deployment Checklist

### 1. Model Files Verification
- [ ] Copy `fnn_model.keras` to Django `models/` directory
- [ ] Copy `lstm_model.keras` to Django `models/` directory
- [ ] Copy `hybrid_model.keras` to Django `models/` directory
- [ ] Copy `fnn_scaler.pkl` to Django `models/` directory
- [ ] Copy `lstm_scaler.pkl` to Django `models/` directory
- [ ] Copy `hybrid_temp_scaler.pkl` to Django `models/` directory
- [ ] Copy `hybrid_stat_scaler.pkl` to Django `models/` directory
- [ ] Copy `le_crop.pkl` to Django `models/` directory
- [ ] Copy `le_zone.pkl` to Django `models/` directory

### 2. Configuration Files
- [ ] Copy `crop_zone_suitability_5crops.json` to Django `config/` directory
- [ ] Create `.env` file with all required variables
- [ ] Add `.env` to `.gitignore`
- [ ] Generate strong SECRET_KEY for production

### 3. Django Setup
- [ ] Create virtual environment
- [ ] Install all dependencies from requirements.txt
- [ ] Run `python manage.py makemigrations`
- [ ] Run `python manage.py migrate`
- [ ] Create superuser account
- [ ] Collect static files
- [ ] Create `logs/` directory

### 4. Local Testing
- [ ] Start development server
- [ ] Test admin panel (http://localhost:8000/admin)
- [ ] Test API endpoint (/api/predictions/)
- [ ] Test prediction with sample data
- [ ] Check model loading logs
- [ ] Verify predictions are reasonable

### 5. Production Configuration
- [ ] Set DEBUG=False in .env
- [ ] Configure ALLOWED_HOSTS
- [ ] Set up PostgreSQL database
- [ ] Configure CORS for your frontend
- [ ] Set up Gunicorn service
- [ ] Configure Nginx
- [ ] Obtain SSL certificate (Let's Encrypt)
- [ ] Set up firewall rules

### 6. Security Hardening
- [ ] Use strong SECRET_KEY (not default)
- [ ] Enable HTTPS only
- [ ] Configure secure cookies
- [ ] Set up CSRF protection
- [ ] Implement rate limiting
- [ ] Add authentication to sensitive endpoints
- [ ] Regular security updates

### 7. Monitoring & Maintenance
- [ ] Set up Sentry or error tracking
- [ ] Configure log rotation
- [ ] Set up database backups
- [ ] Monitor disk space
- [ ] Monitor memory usage
- [ ] Set up uptime monitoring

### 8. Documentation
- [ ] Document API endpoints
- [ ] Create API usage examples
- [ ] Document deployment process
- [ ] Create troubleshooting guide

## Quick Commands Reference

### Development
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Run migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Run dev server
python manage.py runserver
```

### Production
```bash
# Collect static files
python manage.py collectstatic --noinput

# Start Gunicorn
gunicorn crop_yield_project.wsgi:application --bind 0.0.0.0:8000

# Restart services
sudo systemctl restart crop_yield
sudo systemctl restart nginx

# Check logs
sudo journalctl -u crop_yield -f
sudo tail -f /var/log/nginx/error.log
```

### Docker
```bash
# Build and run
docker-compose up -d --build

# Run migrations
docker-compose exec web python manage.py migrate

# View logs
docker-compose logs -f web

# Stop containers
docker-compose down
```

## Testing Endpoints

### Test Prediction API
```bash
curl -X POST http://localhost:8000/api/predictions/ \
  -H "Content-Type: application/json" \
  -d '{
    "crop": "Maize",
    "geopolitical_zone": "NC",
    "state": "Kaduna",
    "avg_temp_c": 28.5,
    "rainfall_mm": 800,
    "avg_humidity": 65,
    "co2_ppm": 420,
    "soil_ph": 6.5,
    "nitrogen_ppm": 50,
    "phosphorus_ppm": 15,
    "potassium_ppm": 100
  }'
```

### Test Health Check
```bash
curl http://localhost:8000/api/predictions/
```

## Critical File Locations

```
crop_yield_project/
├── .env                           # Environment variables (DO NOT COMMIT)
├── manage.py
├── requirements.txt               # Python dependencies
├── models/                        # ML models (9 files)
│   ├── fnn_model.keras
│   ├── lstm_model.keras
│   ├── hybrid_model.keras
│   ├── fnn_scaler.pkl
│   ├── lstm_scaler.pkl
│   ├── hybrid_temp_scaler.pkl
│   ├── hybrid_stat_scaler.pkl
│   ├── le_crop.pkl
│   └── le_zone.pkl
├── config/
│   └── crop_zone_suitability_5crops.json
├── logs/                          # Application logs
│   └── django.log
└── crop_yield_project/
    └── settings.py                # Django configuration
```

## Common Issues & Solutions

### Issue: Model files not found
**Solution**: Verify all 9 model files are in `models/` directory with correct names

### Issue: Database connection error
**Solution**: Check .env DB settings and ensure PostgreSQL is running

### Issue: Static files not loading
**Solution**: Run `python manage.py collectstatic` and check Nginx config

### Issue: Import errors
**Solution**: Activate virtual environment and install requirements

### Issue: Permission denied on model files
**Solution**: `chmod 644 models/*.keras models/*.pkl`

### Issue: High memory usage
**Solution**: Reduce Gunicorn workers or implement model caching

---

## Support Contacts

- Django Issues: Check Django logs in `logs/django.log`
- Model Issues: Verify model files and scaler compatibility
- Server Issues: Check Nginx/Gunicorn logs
- Database Issues: Check PostgreSQL logs

---

Last Updated: December 29, 2025
