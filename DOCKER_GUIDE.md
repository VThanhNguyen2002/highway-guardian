# 🐳 Docker Deployment Guide - Highway Guardian

## 🚀 Quick Start (1 Command)

```bash
# Windows
start-docker.bat

# Linux/Mac
docker-compose up --build -d
```

## 📋 Prerequisites

- Docker Desktop installed and running
- Models in `models/` folder:
  - `models/yolo/best.pt`
  - `models/cnn/bien_bao_mobilenetv2_MERGED_BALANCED_model (1).h5`

## 🏗️ Architecture

```
Docker Containers:
├── backend (Port 8000)
│   ├── FastAPI
│   ├── YOLO + CNN models
│   └── Python 3.10
│
└── frontend (Port 8080)
    ├── Nginx
    ├── Vue.js (built)
    └── Proxy to backend
```

## 📝 Commands

### Start Containers
```bash
# Build and start
docker-compose up --build -d

# Start existing containers
docker-compose up -d
```

### View Logs
```bash
# All containers
docker-compose logs -f

# Specific container
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Stop Containers
```bash
# Stop
docker-compose stop

# Stop and remove
docker-compose down

# Stop, remove, and clean volumes
docker-compose down -v
```

### Rebuild
```bash
# Rebuild specific service
docker-compose build backend
docker-compose build frontend

# Rebuild all
docker-compose build --no-cache
```

## 🔧 Configuration

### Backend Environment
Edit `Dockerfile.backend` or add `.env` file:
```env
PYTHONUNBUFFERED=1
API_HOST=0.0.0.0
API_PORT=8000
```

### Frontend Environment
Edit `frontend/.env`:
```env
VITE_API_URL=http://localhost:8000
VITE_FIREBASE_API_KEY=your-key
```

## 🌐 Access

- **Frontend**: http://localhost:8080
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Check what's using port 8000
netstat -ano | findstr :8000

# Kill process
taskkill /PID <PID> /F

# Or change port in docker-compose.yml
ports:
  - "8001:8000"  # Use 8001 instead
```

### Container Won't Start
```bash
# Check logs
docker-compose logs backend

# Restart container
docker-compose restart backend

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Models Not Found
```bash
# Check volume mount
docker-compose exec backend ls -la /app/models

# Ensure models exist locally
ls models/yolo/
ls models/cnn/
```

### Permission Issues (Linux/Mac)
```bash
# Fix permissions
chmod -R 755 models/
chmod -R 755 src/

# Run with sudo
sudo docker-compose up -d
```

## 📦 Production Deployment

### 1. Update docker-compose.prod.yml
```yaml
version: '3.8'

services:
  backend:
    image: your-registry/highway-guardian-backend:latest
    restart: always
    environment:
      - ENVIRONMENT=production
    
  frontend:
    image: your-registry/highway-guardian-frontend:latest
    restart: always
```

### 2. Build and Push
```bash
# Build
docker build -t your-registry/highway-guardian-backend:latest -f Dockerfile.backend .
docker build -t your-registry/highway-guardian-frontend:latest -f frontend/Dockerfile ./frontend

# Push
docker push your-registry/highway-guardian-backend:latest
docker push your-registry/highway-guardian-frontend:latest
```

### 3. Deploy
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 🔐 Security

### Production Checklist
- [ ] Change default ports
- [ ] Use environment variables for secrets
- [ ] Enable HTTPS
- [ ] Set up firewall rules
- [ ] Use Docker secrets
- [ ] Limit container resources

### Example with Secrets
```yaml
services:
  backend:
    secrets:
      - firebase_key
    environment:
      - FIREBASE_KEY_FILE=/run/secrets/firebase_key

secrets:
  firebase_key:
    file: ./secrets/firebase.json
```

## 📊 Monitoring

### Container Stats
```bash
# Real-time stats
docker stats

# Specific container
docker stats highway-guardian-backend
```

### Health Checks
```bash
# Backend health
curl http://localhost:8000/

# Frontend health
curl http://localhost:8080/
```

## 🔄 Updates

### Update Code
```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose up --build -d
```

### Update Models
```bash
# Copy new models
cp new_model.pt models/yolo/

# Restart backend
docker-compose restart backend
```

## 💾 Backup

### Backup Models
```bash
# Create backup
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/

# Restore
tar -xzf models-backup-20250121.tar.gz
```

### Backup Data
```bash
# Export volumes
docker run --rm -v highway-guardian_models:/data -v $(pwd):/backup alpine tar czf /backup/volumes-backup.tar.gz /data
```

## 🎯 Tips

1. **Development**: Use `docker-compose.yml` with volume mounts for hot reload
2. **Production**: Use `docker-compose.prod.yml` with built images
3. **Testing**: Use separate `docker-compose.test.yml`
4. **Logs**: Always check logs when debugging
5. **Resources**: Limit CPU/memory in production

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Docker](https://fastapi.tiangolo.com/deployment/docker/)
- [Vue.js Docker](https://vuejs.org/guide/best-practices/production-deployment.html)

---

*Last Updated: 2025-01-21*
