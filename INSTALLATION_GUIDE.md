# 🚀 Complete Installation & Startup Guide

## ✅ Prerequisites (Already Installed)

You have all required dependencies:
- ✓ Python 3.11.14
- ✓ pip 24.0
- ✓ Node.js 22.21.0
- ✓ npm 10.9.4

---

## 📦 Installation Complete!

All dependencies have been installed:
- ✓ Backend Python packages (FastAPI, SQLAlchemy, etc.)
- ✓ Frontend npm packages (React, Material-UI, Plotly, Three.js)

---

## 🎯 How to Run the Dashboard

You have **3 options** to run the dashboard:

### **Option 1: Quick Start Script (Recommended)**

The easiest way to get started:

```bash
cd /home/user/MC07_tuning
./start_dashboard.sh
```

This will show you options:
1. Docker Compose (not available without Docker)
2. Manual (Backend + Frontend) ← Choose this
3. Backend only
4. Frontend only

**Choose option 2** to start both services.

---

### **Option 2: Manual Startup (2 Terminals)**

**Terminal 1 - Backend:**
```bash
cd /home/user/MC07_tuning/backend
python3 api_server.py
```

**Terminal 2 - Frontend:**
```bash
cd /home/user/MC07_tuning/frontend
npm start
```

---

### **Option 3: Step by Step (Recommended for First Time)**

**Step 1: Start Backend**
```bash
cd /home/user/MC07_tuning/backend
python3 api_server.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
🚀 Drone Tuning API Server Started
📡 WebSocket endpoint: ws://localhost:8000/ws/{run_id}
📊 API docs: http://localhost:8000/docs
```

**Step 2: In a NEW terminal, Start Frontend**
```bash
cd /home/user/MC07_tuning/frontend
npm start
```

You should see:
```
Compiled successfully!

You can now view drone-tuning-dashboard in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.x.x:3000
```

---

## 🌐 Access the Dashboard

Once both services are running:

- **Frontend (Main Dashboard)**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (Interactive Swagger UI)
- **Health Check**: http://localhost:8000/health

---

## 🧪 Testing the Dashboard

### 1. Open the Dashboard
Navigate to http://localhost:3000 in your browser

### 2. Start a Test Optimization
- Click **"Start New Optimization"** button
- Configure settings:
  - **Algorithm**: Genetic Algorithm
  - **Phase**: Phase 1: Rate Controllers
  - **Generations**: 10 (for quick test)
  - **Population Size**: 5
  - **Parallel Instances**: 2
- Click **"Start Optimization"**

### 3. Watch Real-Time Updates
You should see:
- ✅ Fitness chart updating in real-time
- ✅ Best parameters changing
- ✅ Progress metrics updating
- ✅ SITL instance status

### 4. Explore Other Tabs
- **Telemetry**: View flight data and 3D trajectory
- **Analysis**: Parameter correlations
- **Control**: Bode plots and frequency response

---

## 🔧 Troubleshooting

### Backend Issues

**Problem: Port 8000 already in use**
```bash
# Find what's using port 8000
lsof -ti:8000

# Kill the process
kill -9 $(lsof -ti:8000)

# Or use a different port
# Edit backend/api_server.py, change port=8000 to port=8001
```

**Problem: Import errors (optimization_system not found)**
```bash
# Make sure you're in the backend directory
cd /home/user/MC07_tuning/backend

# The api_server.py automatically adds parent directory to path
# If issues persist, run from project root:
cd /home/user/MC07_tuning
python3 -m backend.api_server
```

**Problem: Database connection errors**
```bash
# The dashboard works without a database (in-memory mode)
# Database is optional for persistence
# If you see database errors, they can be ignored for basic testing
```

### Frontend Issues

**Problem: Port 3000 already in use**
```bash
# Kill the process
kill -9 $(lsof -ti:3000)

# Or React will ask if you want to use a different port
# Just press 'y' when prompted
```

**Problem: Cannot connect to backend**
- Make sure backend is running on port 8000
- Check browser console for errors (F12)
- Verify http://localhost:8000/health returns healthy status

**Problem: WebSocket connection errors**
- Ensure backend is running
- Check CORS settings (already configured in api_server.py)
- Try refreshing the page

**Problem: Blank page or errors**
```bash
# Clear npm cache and rebuild
cd frontend
rm -rf node_modules package-lock.json
npm install
npm start
```

---

## 📊 API Testing (Optional)

Test the backend API directly:

```bash
# Health check
curl http://localhost:8000/health

# Get parameter bounds
curl http://localhost:8000/api/parameters/bounds

# Get default parameters
curl http://localhost:8000/api/parameters/defaults
```

Or visit the interactive API docs:
http://localhost:8000/docs

---

## 🛑 Stopping the Services

**If using start_dashboard.sh:**
- Press `Ctrl+C` in the terminal
- Or use the PIDs shown when services started

**If running manually:**
- Press `Ctrl+C` in each terminal window

**To find and kill processes:**
```bash
# Find processes
ps aux | grep -E "(api_server|npm start)"

# Kill backend
pkill -f api_server.py

# Kill frontend
pkill -f "npm start"

# Or kill by port
kill -9 $(lsof -ti:8000)  # Backend
kill -9 $(lsof -ti:3000)  # Frontend
```

---

## 🔄 Restarting Services

```bash
# Just run the startup command again
cd /home/user/MC07_tuning/backend
python3 api_server.py

# In another terminal
cd /home/user/MC07_tuning/frontend
npm start
```

---

## 📝 Next Steps

### Integrate with Existing Optimization System

The dashboard currently runs with **mock data**. To integrate with your actual optimization system:

1. **Edit `backend/api_server.py`**
2. **Find the `run_optimization()` function** (around line 400)
3. **Replace mock code with actual optimizer calls**:

```python
# Current (mock):
import random
best_fitness = 0.5 + (generation / config.generations) * 0.4

# Replace with:
from optimization_system.optimizer import GeneticOptimizer
from optimization_system.sitl_manager import SITLManager
from optimization_system.performance_evaluator import PerformanceEvaluator

optimizer = GeneticOptimizer(...)
results = optimizer.run_generation(generation)
best_fitness = results['best_fitness']
```

### Enable Database Persistence (Optional)

To store optimization runs permanently:

1. **Install PostgreSQL** (if not already installed)
2. **Create database:**
```bash
sudo -u postgres psql
CREATE DATABASE drone_tuning;
CREATE USER drone_user WITH PASSWORD 'drone_password';
GRANT ALL PRIVILEGES ON DATABASE drone_tuning TO drone_user;
\q
```

3. **Initialize database schema:**
```bash
cd /home/user/MC07_tuning/backend
python3 database.py init
```

4. **Update environment variable** in docker-compose.yml or .env file

---

## 🎨 Dashboard Features

### Real-Time Monitoring
- Live fitness evolution charts
- WebSocket updates every second
- Progress tracking with ETA
- Parallel SITL instance monitoring

### Visualizations
- 📈 **Fitness Chart**: Best and average fitness over generations
- 🎯 **Parameter Panel**: Current best parameters organized by category
- 📊 **Metrics Summary**: Rise time, settling time, overshoot, etc.
- ⚙️ **SITL Status**: Real-time instance status
- 🛸 **3D Trajectory**: Interactive flight path visualization
- 📈 **Telemetry Charts**: Attitude, rates, position time-series
- 📊 **Bode Plots**: Frequency response with stability margins
- 🔥 **Correlation Heatmap**: Parameter relationships

### Controls
- ▶️ Start optimization with custom config
- ⏸️ Pause/Resume during runtime
- ⏹️ Stop and save results
- 💾 Export optimized parameters (.parm file)

---

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **Dashboard README**: `/home/user/MC07_tuning/DASHBOARD_README.md`
- **Project README**: `/home/user/MC07_tuning/README.md`

---

## ✅ Quick Verification Checklist

Before running optimization:
- [ ] Backend running on port 8000
- [ ] Frontend accessible at http://localhost:3000
- [ ] Health check passes: http://localhost:8000/health
- [ ] API docs load: http://localhost:8000/docs
- [ ] Dashboard shows main page with "Start New Optimization" button
- [ ] No errors in browser console (F12)
- [ ] No errors in backend terminal

---

## 🎯 Quick Start Command Summary

**One-line startup (2 terminals needed):**

Terminal 1:
```bash
cd /home/user/MC07_tuning/backend && python3 api_server.py
```

Terminal 2:
```bash
cd /home/user/MC07_tuning/frontend && npm start
```

Then open: http://localhost:3000

---

**Need help?** Check the troubleshooting section above or review logs in the terminal windows.

**Happy Tuning! 🚁**
