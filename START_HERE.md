# 🚀 Quick Start - Drone PID Tuning Dashboard

## ✅ Installation Complete!

All dependencies are installed and the dashboard is ready to run.

---

## 🎯 How to Run (Choose ONE method)

### **Method 1: Quick Start (2 Commands)**

**Terminal 1 - Start Backend:**
```bash
cd /home/user/MC07_tuning/backend
python3 api_server.py
```

**Terminal 2 - Start Frontend:**
```bash
cd /home/user/MC07_tuning/frontend
npm start
```

**Then open:** http://localhost:3000

---

### **Method 2: Using Startup Script**

```bash
cd /home/user/MC07_tuning
./start_dashboard.sh
# Choose option 2: Manual (Backend + Frontend)
```

---

## 📊 What You'll See

### Backend Output:
```
🚀 Drone Tuning API Server Started
📡 WebSocket endpoint: ws://localhost:8000/ws/{run_id}
📊 API docs: http://localhost:8000/docs
❤️  Health check: http://localhost:8000/health

⚠️  Dashboard will run in DEMO MODE with mock data
```

**Note:** The warning about DEMO MODE is expected. The dashboard works perfectly with simulated data for testing.

### Frontend Output:
```
Compiled successfully!

You can now view drone-tuning-dashboard in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.x.x:3000
```

---

## 🎨 Using the Dashboard

### 1. Start an Optimization

1. Click **"Start New Optimization"**
2. Configure:
   - **Algorithm**: Genetic Algorithm
   - **Phase**: Phase 1: Rate Controllers
   - **Generations**: 20 (for quick test)
   - **Population**: 10
   - **Parallel Instances**: 4
3. Click **"Start Optimization"**

### 2. Watch Real-Time Updates

- **Fitness Chart**: Updates every ~2 seconds
- **Best Parameters**: Changes as optimization progresses
- **Progress Bar**: Shows completion percentage
- **SITL Status**: Shows active simulation instances

### 3. Explore Features

**Dashboard Tab:**
- Real-time fitness evolution
- Current best parameters
- Progress metrics
- SITL instance status

**Telemetry Tab:**
- 3D trajectory viewer
- Flight data charts (attitude, rates, position)
- Performance metrics

**Analysis Tab:**
- Parameter correlation heatmap
- Statistical analysis

**Control Tab:**
- Bode plots (frequency response)
- Stability margins
- Phase/Gain analysis

---

## 🌐 Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **Dashboard** | http://localhost:3000 | Main web interface |
| **API** | http://localhost:8000 | Backend REST API |
| **API Docs** | http://localhost:8000/docs | Interactive Swagger UI |
| **Health** | http://localhost:8000/health | System health check |

---

## 🛑 Stopping Services

Press `Ctrl+C` in each terminal window

Or kill by port:
```bash
kill -9 $(lsof -ti:8000)  # Backend
kill -9 $(lsof -ti:3000)  # Frontend
```

---

## 🔧 Current Status

### ✅ Working Features:
- ✓ Backend API server (FastAPI)
- ✓ Frontend dashboard (React)
- ✓ Real-time WebSocket updates
- ✓ Mock optimization simulation
- ✓ All visualizations (charts, 3D, Bode plots)
- ✓ Parameter management
- ✓ Export functionality

### 🔄 Demo Mode:
The dashboard is currently running in **DEMO MODE** with simulated data. This means:
- ✓ You can test all features
- ✓ See how optimization looks
- ✓ Explore all visualizations
- ⚠️ No actual SITL simulation (mock data only)

### 🚀 To Use Real Optimization:
To connect to your actual optimization system:
1. Install DEAP and Optuna: `pip install deap==1.4.1 optuna==3.4.0`
2. Backend will automatically detect and use real optimization
3. See `INSTALLATION_GUIDE.md` for details

---

## 📚 Documentation

- **Quick Start**: This file (START_HERE.md)
- **Full Installation**: INSTALLATION_GUIDE.md
- **Dashboard Features**: DASHBOARD_README.md
- **Troubleshooting**: INSTALLATION_GUIDE.md (bottom section)

---

## 🐛 Common Issues

**Backend won't start:**
```bash
# Check if port is in use
lsof -ti:8000
# Kill if needed
kill -9 $(lsof -ti:8000)
```

**Frontend won't start:**
```bash
# Check if port is in use
lsof -ti:3000
# Or let React use different port (it will ask)
```

**Can't see updates:**
- Make sure both backend AND frontend are running
- Check browser console (F12) for errors
- Verify http://localhost:8000/health shows "healthy"

---

## ✨ Features Overview

### Real-Time Monitoring
- Live fitness charts
- WebSocket streaming
- Instant parameter updates
- Progress tracking with ETA

### Visualizations
- 📈 Plotly.js interactive charts
- 🛸 Three.js 3D trajectory viewer
- 📊 Bode plots with stability analysis
- 🔥 Correlation heatmaps

### Controls
- ▶️ Start/Pause/Resume optimization
- ⏹️ Stop and save results
- 💾 Export parameters (.parm file)
- ⚙️ Configure algorithm settings

---

## 🎯 Next Steps

1. **Run the dashboard** (use commands above)
2. **Test with mock data** (explore all features)
3. **Review visualizations** (all 4 tabs)
4. **Export test parameters** (try export feature)
5. **Read full guide** (INSTALLATION_GUIDE.md)

---

## 📞 Need Help?

1. Check **INSTALLATION_GUIDE.md** for detailed troubleshooting
2. Check **DASHBOARD_README.md** for feature documentation
3. View API docs at http://localhost:8000/docs
4. Check browser console (F12) for frontend errors
5. Check terminal for backend errors

---

## ✅ Verification Checklist

Before running optimization:
- [ ] Backend running (see startup messages)
- [ ] Frontend compiled successfully
- [ ] http://localhost:3000 loads
- [ ] http://localhost:8000/health shows "healthy"
- [ ] No red errors in browser console (F12)

---

**Ready to start? Run the two commands above and enjoy! 🚁**
