# 🚁 Drone PID Tuning Dashboard

Real-time web-based dashboard for automated drone PID tuning with ArduPilot SITL.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## ✨ Features

### Real-Time Monitoring
- **Live Fitness Evolution**: Watch optimization progress in real-time
- **WebSocket Updates**: Instant notifications of generation completions
- **Parallel SITL Status**: Monitor all simulation instances simultaneously
- **Progress Metrics**: ETA, trial counts, and performance indicators

### Visualization Components
- **📈 Fitness Chart**: Track best and average fitness across generations
- **🎯 Parameter Panel**: View current best parameters by category
- **📊 Metrics Summary**: Key performance indicators at a glance
- **⚙️ SITL Status**: Real-time instance monitoring

### Advanced Analysis
- **🛸 3D Trajectory Viewer**: Interactive flight path visualization with Three.js
- **📈 Telemetry Charts**: Time-series plots for attitude, rates, position
- **📊 Bode Plots**: Frequency response analysis with stability margins
- **🔥 Correlation Heatmap**: Parameter correlation analysis
- **🎯 Nyquist Plots**: Stability criterion visualization (ready to implement)

### Control Systems Engineering
- Frequency domain analysis (Bode, Nyquist)
- Time domain analysis (step response)
- Stability margin calculations
- Bandwidth verification

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Web Browser                           │
│  React Frontend (Port 3000)                             │
│  - Redux State Management                               │
│  - Material-UI Components                               │
│  - Plotly.js Charts                                     │
│  - Three.js 3D Graphics                                 │
└─────────────────────────────────────────────────────────┘
                        ↕ WebSocket / REST API
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Backend (Port 8000)           │
│  - RESTful API Endpoints                                │
│  - WebSocket Real-time Updates                          │
│  - Background Task Optimization                         │
└─────────────────────────────────────────────────────────┘
                        ↕
┌─────────────────────────────────────────────────────────┐
│              Existing Optimization System               │
│  - Genetic Algorithm / Bayesian Optimization            │
│  - Parallel SITL Manager                                │
│  - Performance Evaluator                                │
│  - Flight Data Logger                                   │
└─────────────────────────────────────────────────────────┘
                        ↕
┌─────────────────────────────────────────────────────────┐
│                   Data Layer                            │
│  PostgreSQL + TimescaleDB | Redis | File System         │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Access:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Option 2: Manual Setup

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python api_server.py
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

## 📦 Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL 14+ (optional, for persistence)
- Redis 7+ (optional, for pub/sub)
- ArduPilot SITL environment

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize database (optional)
python database.py init

# Run server
python api_server.py
```

Server will start on http://localhost:8000

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

Frontend will open at http://localhost:3000

## 📖 Usage

### Starting an Optimization

1. **Open Dashboard**: Navigate to http://localhost:3000
2. **Click "Start New Optimization"**
3. **Configure Settings**:
   - Algorithm: Genetic or Bayesian
   - Phase: Rate, Attitude, Position, or Advanced
   - Generations: Number of iterations
   - Population Size: Individuals per generation
   - Parallel Instances: SITL instances to run
4. **Click "Start Optimization"**

### Monitoring Progress

- **Dashboard Tab**: Real-time fitness evolution and metrics
- **Telemetry Tab**: Detailed flight data and 3D trajectory
- **Analysis Tab**: Parameter correlations and statistics
- **Control Tab**: Frequency response analysis

### Controlling Optimization

- **Pause**: Temporarily halt optimization
- **Resume**: Continue from paused state
- **Stop**: Terminate and save results
- **Export**: Download optimized parameters (.parm file)

## 📡 API Documentation

### REST Endpoints

#### Optimization Control
```
POST   /api/optimization/start        # Start new run
GET    /api/optimization/{run_id}/status  # Get status
POST   /api/optimization/{run_id}/pause   # Pause
POST   /api/optimization/{run_id}/resume  # Resume
POST   /api/optimization/{run_id}/stop    # Stop
GET    /api/optimization/list             # List all runs
```

#### Telemetry
```
GET    /api/telemetry/{run_id}/trials         # List trials
GET    /api/telemetry/{run_id}/trial/{id}     # Get trial data
```

#### Parameters
```
GET    /api/parameters/bounds                 # Get bounds
GET    /api/parameters/defaults               # Get defaults
POST   /api/parameters/export/{run_id}        # Export .parm file
```

#### Analysis
```
GET    /api/analysis/{run_id}/correlation          # Correlation matrix
GET    /api/analysis/{run_id}/frequency_response   # Bode data
```

### WebSocket Events

Connect to: `ws://localhost:8000/ws/{run_id}`

**Server → Client Events:**
- `initial_state`: Initial run state
- `generation_complete`: Generation finished
- `trial_complete`: Single trial finished
- `new_best`: New best fitness found
- `status_change`: Status update
- `optimization_complete`: Run finished
- `error`: Error occurred

**Client → Server Events:**
- `ping`: Keepalive
- `get_status`: Request current status

## 🛠️ Development

### Project Structure

```
MC07_tuning/
├── backend/
│   ├── api_server.py           # FastAPI application
│   ├── database.py             # SQLAlchemy models
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/         # React components
│   │   ├── pages/              # Page components
│   │   ├── store/              # Redux store
│   │   ├── utils/              # API & WebSocket utils
│   │   ├── App.js
│   │   └── index.js
│   ├── public/
│   ├── package.json
│   └── Dockerfile
├── optimization_system/        # Existing optimization code
├── docker-compose.yml
└── DASHBOARD_README.md
```

### Adding New Visualizations

1. Create component in `frontend/src/components/`
2. Add Redux state if needed in `store/store.js`
3. Create API endpoint in `backend/api_server.py`
4. Import and use in appropriate page

Example:
```javascript
// frontend/src/components/MyChart.js
import React from 'react';
import Plot from 'react-plotly.js';

function MyChart({ data }) {
  return <Plot data={data} layout={{...}} />;
}

export default MyChart;
```

### API Integration

Connect backend to existing optimization system:

```python
# In backend/api_server.py
from optimization_system.optimizer import GeneticOptimizer

async def run_optimization(run_id, config):
    optimizer = GeneticOptimizer(...)

    # Run with callbacks
    for generation in range(config.generations):
        results = optimizer.run_generation(generation)

        # Broadcast to WebSocket clients
        await manager.broadcast(run_id, {
            'type': 'generation_complete',
            'generation': generation,
            'best_fitness': results['best_fitness']
        })
```

## 🐛 Troubleshooting

### Backend Issues

**Port 8000 already in use:**
```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9

# Or change port in api_server.py
uvicorn.run(app, port=8001)
```

**Database connection failed:**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Or use in-memory mode (no database)
# Comment out database imports in api_server.py
```

**Import errors:**
```bash
# Ensure optimization_system is in Python path
export PYTHONPATH=$PYTHONPATH:/path/to/MC07_tuning
```

### Frontend Issues

**npm install fails:**
```bash
# Clear cache and retry
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

**WebSocket connection refused:**
- Check backend is running on port 8000
- Verify WebSocket URL in `frontend/src/utils/websocket.js`
- Check CORS settings in `backend/api_server.py`

**Plotly not rendering:**
```bash
# Reinstall plotly dependencies
npm install plotly.js react-plotly.js --force
```

### Docker Issues

**Containers won't start:**
```bash
# Check logs
docker-compose logs backend
docker-compose logs frontend

# Rebuild images
docker-compose build --no-cache

# Remove volumes and restart
docker-compose down -v
docker-compose up -d
```

## 🎯 Next Steps

### Immediate Enhancements
- [ ] Connect backend to actual optimization system
- [ ] Implement database persistence
- [ ] Add authentication
- [ ] Enable SSL/HTTPS

### Advanced Features
- [ ] Multi-user support
- [ ] Historical run comparison
- [ ] Advanced parameter sensitivity analysis
- [ ] Export optimization reports
- [ ] Real-time 3D drone visualization during flight

### Performance Optimizations
- [ ] Telemetry data downsampling
- [ ] Lazy loading for large datasets
- [ ] WebSocket message compression
- [ ] Database query optimization

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Plotly.js](https://plotly.com/javascript/)
- [Three.js](https://threejs.org/)
- [ArduPilot SITL](https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html)

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For issues or questions:
- Create an issue on GitHub
- Check troubleshooting section above
- Review API documentation at http://localhost:8000/docs

---

**Happy Tuning! 🚁**
