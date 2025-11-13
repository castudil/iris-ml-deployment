# 🐳 Docker Deployment Guide

## Quick Start

### Prerequisites
- Docker Desktop installed
- Docker Compose installed

### Deploy with Docker Compose

1. **Build and start all services:**
```bash
docker-compose up -d --build
```

2. **Access the application:**
- Frontend: http://localhost
- Backend API: http://localhost:8000
- API Health Check: http://localhost:8000/api/health/

3. **Stop services:**
```bash
docker-compose down
```

4. **View logs:**
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

## 🏗️ Architecture

```
┌─────────────────┐
│   Frontend      │
│  (React + Vite) │
│  Nginx:80       │
└────────┬────────┘
         │
         │ HTTP
         │
┌────────▼────────┐
│   Backend       │
│  (Django API)   │
│  Port:8000      │
└────────┬────────┘
         │
         │ Loads
         │
┌────────▼────────┐
│   ML Models     │
│  (PKL files)    │
└─────────────────┘
```

## 📦 Services

### Backend (Django)
- **Image**: Python 3.11-slim
- **Port**: 8000
- **Features**:
  - Loads ML models on startup
  - REST API endpoints
  - CORS enabled
  - Health check endpoint

### Frontend (React)
- **Image**: Node 20 (build) + Nginx Alpine (serve)
- **Port**: 80
- **Features**:
  - Multi-stage build (optimized)
  - Static file serving with Nginx
  - Gzip compression
  - Cache headers

## 🔧 Development

### Rebuild a specific service:
```bash
docker-compose up -d --build backend
docker-compose up -d --build frontend
```

### Execute commands in containers:
```bash
# Django shell
docker-compose exec backend python manage.py shell

# Django migrations
docker-compose exec backend python manage.py migrate

# Bash access
docker-compose exec backend bash
docker-compose exec frontend sh
```

## 🚀 Production Deployment

### 1. Update environment variables:
```bash
cp .env.example .env
# Edit .env with production values
```

### 2. Build for production:
```bash
docker-compose -f docker-compose.yml up -d --build
```

### 3. Security checklist:
- [ ] Change `DJANGO_SECRET_KEY` in `.env`
- [ ] Set `DEBUG=False`
- [ ] Update `ALLOWED_HOSTS`
- [ ] Update `CORS_ALLOWED_ORIGINS`
- [ ] Use HTTPS in production
- [ ] Set up proper database (PostgreSQL)
- [ ] Configure volume backups

## 📊 Health Checks

Backend health check runs every 30 seconds:
```bash
curl http://localhost:8000/api/health/
```

Expected response:
```json
{
  "status": "ok",
  "message": "ML API is running",
  "models_loaded": 2,
  "available_models": ["rf_model", "svc_model"],
  "metadata_loaded": true
}
```

## 🔍 Troubleshooting

### Backend not starting:
```bash
docker-compose logs backend
```

### Frontend not building:
```bash
docker-compose logs frontend
docker-compose exec frontend npm run build
```

### Models not loading:
Ensure `models/` directory contains:
- `rf_model.pkl`
- `svc_model.pkl`
- `results.json`

### CORS errors:
Check `CORS_ALLOWED_ORIGINS` in `backend/iris_api/iris_api/settings.py`

## 🧹 Cleanup

### Remove containers and images:
```bash
docker-compose down --rmi all

# Remove volumes too
docker-compose down -v --rmi all
```

### Remove unused Docker resources:
```bash
docker system prune -a
```

## 📝 Notes

- Models are mounted as volume for easy updates
- Frontend is built at image creation time
- Backend runs migrations on startup
- Both services restart automatically unless stopped
