# Quick Start Guide

## Starting the System

### Option 1: Using the startup script (Recommended)
```bash
./start_backend.sh
```

This will:
- Clean up any stuck processes
- Start the backend API server
- Display status and URLs

### Option 2: Manual startup
```bash
# 1. Clean up first
./cleanup.sh

# 2. Start backend
cd backend
python3 api_server.py
```

## Frontend (Dashboard)

In a separate terminal:
```bash
cd frontend
npm start
```

The dashboard will open at http://localhost:3000

## Stopping the System

### Option 1: Clean shutdown
Press `Ctrl+C` in the backend terminal. The server will automatically clean up SITL processes.

### Option 2: Force cleanup (if processes are stuck)
```bash
./cleanup.sh
```

## Troubleshooting

### Backend won't start / Port already in use
```bash
./cleanup.sh
./start_backend.sh
```

### Terminal hangs on Ctrl+C
```bash
# In another terminal:
./cleanup.sh
```

### Check what's running
```bash
ps aux | grep -E "(api_server|arducopter|mavproxy)"
```

### Kill specific process
```bash
# Find process ID
ps aux | grep api_server

# Kill by PID
kill -9 <PID>
```

## URLs

- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Common Commands

```bash
# Clean up everything
./cleanup.sh

# Start backend
./start_backend.sh

# Start frontend (separate terminal)
cd frontend && npm start

# Check backend logs
cd backend && tail -f *.log

# Check SITL instances
ps aux | grep arducopter
```

## Workflow

1. Start backend: `./start_backend.sh`
2. Start frontend: `cd frontend && npm start`
3. Open dashboard: http://localhost:3000
4. Click "Start Optimization"
5. Monitor real-time updates
6. When done: Ctrl+C in both terminals
7. If stuck: Run `./cleanup.sh`
