"""
FastAPI Server for Drone Tuning System
Provides REST API and WebSocket endpoints for real-time monitoring
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
import json
import logging
from datetime import datetime
import uuid
import sys
import os

# Setup logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add optimization_system to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Try to import optimization system (optional - dashboard works standalone)
try:
    from optimization_system.optimizer import GeneticOptimizer, BayesianOptimizer
    from optimization_system.sitl_manager import SITLManager
    from optimization_system.performance_evaluator import PerformanceEvaluator
    from optimization_system.flight_logger import FlightDataLogger
    from optimization_system.config import PARAMETER_BOUNDS, DRONE_SPECS
    OPTIMIZATION_AVAILABLE = True
    logger.info("✓ Optimization system loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️  Optimization system not fully available: {e}")
    logger.warning("🔧 Dashboard will run in DEMO MODE with mock data")
    OPTIMIZATION_AVAILABLE = False
    # Mock parameter bounds for demo mode
    PARAMETER_BOUNDS = {
        "ATC_RAT_RLL_P": {"min": 0.08, "max": 0.25},
        "ATC_RAT_RLL_I": {"min": 0.05, "max": 0.15},
        "ATC_RAT_RLL_D": {"min": 0.003, "max": 0.012},
        "ATC_RAT_PIT_P": {"min": 0.08, "max": 0.25},
        "ATC_RAT_PIT_I": {"min": 0.05, "max": 0.15},
        "ATC_RAT_PIT_D": {"min": 0.003, "max": 0.012},
        "ATC_RAT_YAW_P": {"min": 0.3, "max": 0.6},
        "ATC_RAT_YAW_I": {"min": 0.03, "max": 0.06},
        "ATC_ANG_RLL_P": {"min": 3.5, "max": 6.0},
        "ATC_ANG_PIT_P": {"min": 3.5, "max": 6.0},
        "ATC_ANG_YAW_P": {"min": 3.5, "max": 6.0}
    }
    DRONE_SPECS = {
        "mass": 30.0,
        "frame_type": "X-frame quadcopter"
    }

# Initialize FastAPI app
app = FastAPI(
    title="Drone Tuning API",
    version="1.0.0",
    description="Real-time API for automated drone PID tuning"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class OptimizationConfig(BaseModel):
    algorithm: str = Field(..., description="Optimization algorithm: 'genetic' or 'bayesian'")
    phase: str = Field(..., description="Phase: 'phase1_rate', 'phase2_attitude', 'phase3_position', 'phase4_advanced'")
    generations: int = Field(default=100, ge=1, le=1000)
    population_size: int = Field(default=20, ge=4, le=100)
    parallel_instances: int = Field(default=4, ge=1, le=20)
    mutation_rate: float = Field(default=0.2, ge=0.0, le=1.0)
    crossover_rate: float = Field(default=0.7, ge=0.0, le=1.0)

class OptimizationStatus(BaseModel):
    run_id: str
    status: str  # 'running', 'paused', 'completed', 'failed', 'stopped'
    current_generation: int
    total_generations: int
    completed_trials: int
    best_fitness: float
    best_parameters: Dict[str, float]
    start_time: str
    elapsed_time: float
    estimated_time_remaining: Optional[float] = None

class TrialResult(BaseModel):
    trial_id: int
    generation: int
    parameters: Dict[str, float]
    fitness: float
    crashed: bool
    metrics: Dict[str, Any]
    timestamp: str

# ============================================================
# GLOBAL STATE
# ============================================================

# In-memory storage (in production, use Redis/database)
active_runs: Dict[str, Dict[str, Any]] = {}
websocket_connections: Dict[str, List[WebSocket]] = {}

# ============================================================
# WEBSOCKET CONNECTION MANAGER
# ============================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, run_id: str, websocket: WebSocket):
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = []
        self.active_connections[run_id].append(websocket)
        logger.info(f"Client connected to run {run_id}. Total connections: {len(self.active_connections[run_id])}")

    def disconnect(self, run_id: str, websocket: WebSocket):
        if run_id in self.active_connections:
            if websocket in self.active_connections[run_id]:
                self.active_connections[run_id].remove(websocket)
                logger.info(f"Client disconnected from run {run_id}")
            if not self.active_connections[run_id]:
                del self.active_connections[run_id]

    async def broadcast(self, run_id: str, message: dict):
        """Send message to all clients watching this run"""
        if run_id in self.active_connections:
            dead_connections = []
            for connection in self.active_connections[run_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send to client: {e}")
                    dead_connections.append(connection)

            # Remove dead connections
            for dead in dead_connections:
                self.disconnect(run_id, dead)

    async def send_to_client(self, websocket: WebSocket, message: dict):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message to client: {e}")

manager = ConnectionManager()

# ============================================================
# OPTIMIZATION CONTROL ENDPOINTS
# ============================================================

@app.post("/api/optimization/start", response_model=Dict[str, str])
async def start_optimization(
    config: OptimizationConfig,
    background_tasks: BackgroundTasks
):
    """Start a new optimization run"""
    run_id = str(uuid.uuid4())[:8]

    logger.info(f"Starting optimization run {run_id}")
    logger.info(f"Algorithm: {config.algorithm}, Phase: {config.phase}")
    logger.info(f"Generations: {config.generations}, Population: {config.population_size}")

    # Initialize run state
    active_runs[run_id] = {
        'status': 'starting',
        'config': config.dict(),
        'start_time': datetime.now().isoformat(),
        'current_generation': 0,
        'best_fitness': 0.0,
        'best_parameters': {},
        'fitness_history': [],
        'avg_fitness_history': [],
        'completed_trials': 0,
        'trial_results': [],
        'phase_history': []
    }

    # Start optimization in background
    background_tasks.add_task(run_optimization, run_id, config)

    return {
        "run_id": run_id,
        "status": "started",
        "message": f"Optimization {run_id} started successfully"
    }

@app.get("/api/optimization/{run_id}/status", response_model=OptimizationStatus)
async def get_optimization_status(run_id: str):
    """Get current status of an optimization run"""
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")

    run_data = active_runs[run_id]
    elapsed = (datetime.now() - datetime.fromisoformat(run_data['start_time'])).total_seconds()

    # Estimate time remaining
    eta = None
    if run_data['current_generation'] > 0:
        avg_time_per_gen = elapsed / run_data['current_generation']
        remaining_gens = run_data['config']['generations'] - run_data['current_generation']
        eta = avg_time_per_gen * remaining_gens

    return OptimizationStatus(
        run_id=run_id,
        status=run_data['status'],
        current_generation=run_data['current_generation'],
        total_generations=run_data['config']['generations'],
        completed_trials=run_data['completed_trials'],
        best_fitness=run_data['best_fitness'],
        best_parameters=run_data['best_parameters'],
        start_time=run_data['start_time'],
        elapsed_time=elapsed,
        estimated_time_remaining=eta
    )

@app.post("/api/optimization/{run_id}/pause")
async def pause_optimization(run_id: str):
    """Pause an ongoing optimization"""
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")

    active_runs[run_id]['status'] = 'paused'
    await manager.broadcast(run_id, {
        'type': 'status_change',
        'status': 'paused'
    })
    logger.info(f"Optimization {run_id} paused")
    return {"status": "paused"}

@app.post("/api/optimization/{run_id}/resume")
async def resume_optimization(run_id: str):
    """Resume a paused optimization"""
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")

    active_runs[run_id]['status'] = 'running'
    await manager.broadcast(run_id, {
        'type': 'status_change',
        'status': 'running'
    })
    logger.info(f"Optimization {run_id} resumed")
    return {"status": "resumed"}

@app.post("/api/optimization/{run_id}/stop")
async def stop_optimization(run_id: str):
    """Stop an optimization and save results"""
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")

    active_runs[run_id]['status'] = 'stopped'
    await manager.broadcast(run_id, {
        'type': 'status_change',
        'status': 'stopped'
    })

    logger.info(f"Optimization {run_id} stopped")

    # Return final results
    results = {
        'run_id': run_id,
        'status': 'stopped',
        'best_fitness': active_runs[run_id]['best_fitness'],
        'best_parameters': active_runs[run_id]['best_parameters'],
        'fitness_history': active_runs[run_id]['fitness_history']
    }
    return results

@app.get("/api/optimization/list")
async def list_optimizations():
    """List all optimization runs"""
    return {
        "runs": [
            {
                "run_id": run_id,
                "status": data['status'],
                "start_time": data['start_time'],
                "best_fitness": data['best_fitness'],
                "phase": data['config']['phase']
            }
            for run_id, data in active_runs.items()
        ]
    }

# ============================================================
# WEBSOCKET ENDPOINT FOR REAL-TIME UPDATES
# ============================================================

@app.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """
    WebSocket endpoint for real-time optimization updates

    Messages sent to client:
    - generation_complete: { generation, best_fitness, avg_fitness, best_parameters }
    - trial_complete: { trial_id, fitness, parameters, crashed }
    - new_best: { fitness, parameters }
    - status_change: { status }
    - error: { message }
    """
    await manager.connect(run_id, websocket)

    try:
        # Send current state immediately
        if run_id in active_runs:
            await manager.send_to_client(websocket, {
                'type': 'initial_state',
                'data': {
                    'status': active_runs[run_id]['status'],
                    'current_generation': active_runs[run_id]['current_generation'],
                    'best_fitness': active_runs[run_id]['best_fitness'],
                    'best_parameters': active_runs[run_id]['best_parameters'],
                    'fitness_history': active_runs[run_id]['fitness_history']
                }
            })

        # Keep connection alive and handle client messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)

                # Handle client commands
                if data == "ping":
                    await manager.send_to_client(websocket, {'type': 'pong'})
                elif data == "get_status":
                    if run_id in active_runs:
                        await manager.send_to_client(websocket, {
                            'type': 'status_update',
                            'data': active_runs[run_id]
                        })
            except asyncio.TimeoutError:
                # Send keepalive ping
                await manager.send_to_client(websocket, {'type': 'ping'})

    except WebSocketDisconnect:
        manager.disconnect(run_id, websocket)
        logger.info(f"Client disconnected from run {run_id}")
    except Exception as e:
        logger.error(f"WebSocket error for run {run_id}: {e}")
        manager.disconnect(run_id, websocket)

# ============================================================
# TELEMETRY ENDPOINTS
# ============================================================

@app.get("/api/telemetry/{run_id}/trials")
async def get_trials_list(run_id: str):
    """Get list of all trials for a run"""
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")

    return {
        'run_id': run_id,
        'trials': active_runs[run_id].get('trial_results', [])
    }

@app.get("/api/telemetry/{run_id}/trial/{trial_id}")
async def get_trial_telemetry(run_id: str, trial_id: int):
    """Get detailed telemetry for a specific trial"""
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")

    # In a real implementation, this would load from database or flight logs
    # For now, return mock data
    import numpy as np
    t = np.linspace(0, 30, 300)  # 30 seconds at 10 Hz

    return {
        'trial_id': trial_id,
        'run_id': run_id,
        'timestamps': t.tolist(),
        'attitude': {
            'roll': (10 * np.sin(t * 0.5) * np.exp(-t * 0.05)).tolist(),
            'pitch': (8 * np.sin(t * 0.3) * np.exp(-t * 0.05)).tolist(),
            'yaw': (t * 2).tolist()
        },
        'rates': {
            'roll': (5 * np.cos(t * 0.5) * np.exp(-t * 0.05)).tolist(),
            'pitch': (4 * np.cos(t * 0.3) * np.exp(-t * 0.05)).tolist(),
            'yaw': (2 * np.ones_like(t)).tolist()
        },
        'position': {
            'x': (t * 0.1 * np.sin(t * 0.2)).tolist(),
            'y': (t * 0.05 * np.cos(t * 0.2)).tolist(),
            'z': (10 * (1 - np.exp(-t * 0.3))).tolist()
        },
        'motors': {
            'motor1': (1500 + 100 * np.sin(t * 0.5)).tolist(),
            'motor2': (1500 + 100 * np.cos(t * 0.5)).tolist(),
            'motor3': (1500 + 100 * np.sin(t * 0.5 + np.pi)).tolist(),
            'motor4': (1500 + 100 * np.cos(t * 0.5 + np.pi)).tolist()
        },
        'metrics': {
            'rise_time': 1.2,
            'settling_time': 3.5,
            'overshoot': 8.3,
            'steady_state_error': 0.12,
            'rms_error': 0.45
        }
    }

# ============================================================
# PARAMETERS ENDPOINTS
# ============================================================

@app.get("/api/parameters/bounds")
async def get_parameter_bounds():
    """Get min/max bounds for all parameters"""
    return PARAMETER_BOUNDS

@app.get("/api/parameters/defaults")
async def get_default_parameters():
    """Get default parameter values"""
    # Return conservative defaults
    return {
        "ATC_RAT_RLL_P": 0.135,
        "ATC_RAT_RLL_I": 0.090,
        "ATC_RAT_RLL_D": 0.008,
        "ATC_RAT_PIT_P": 0.135,
        "ATC_RAT_PIT_I": 0.090,
        "ATC_RAT_PIT_D": 0.008,
        "ATC_RAT_YAW_P": 0.450,
        "ATC_RAT_YAW_I": 0.045,
        "ATC_ANG_RLL_P": 4.5,
        "ATC_ANG_PIT_P": 4.5,
        "ATC_ANG_YAW_P": 4.5
    }

@app.post("/api/parameters/export/{run_id}")
async def export_parameters(run_id: str, format: str = "parm"):
    """Export optimized parameters to file"""
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")

    params = active_runs[run_id]['best_parameters']

    if format == "parm":
        # Generate ArduPilot .parm file
        filename = f"/tmp/optimized_params_{run_id}.parm"
        with open(filename, 'w') as f:
            for param, value in params.items():
                f.write(f"{param},{value}\n")
        return FileResponse(filename, filename=f"optimized_params_{run_id}.parm")
    else:
        # Return JSON
        return params

# ============================================================
# ANALYSIS ENDPOINTS
# ============================================================

@app.get("/api/analysis/{run_id}/correlation")
async def get_correlation_matrix(run_id: str):
    """Get parameter correlation matrix"""
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")

    # Mock correlation data (in real implementation, calculate from trial history)
    import numpy as np
    params = list(PARAMETER_BOUNDS.keys())[:10]  # First 10 params
    n = len(params)

    # Generate random correlation matrix
    matrix = np.random.rand(n, n)
    matrix = (matrix + matrix.T) / 2  # Make symmetric
    np.fill_diagonal(matrix, 1.0)

    return {
        'parameters': params,
        'matrix': matrix.tolist()
    }

@app.get("/api/analysis/{run_id}/frequency_response")
async def get_frequency_response(run_id: str, axis: str = "roll"):
    """Calculate Bode plot data for specific axis"""
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")

    params = active_runs[run_id]['best_parameters']

    # Calculate Bode plot
    from scipy import signal
    import numpy as np

    P = params.get(f'ATC_RAT_{axis.upper()}_P', 0.15)
    I = params.get(f'ATC_RAT_{axis.upper()}_I', 0.10)
    D = params.get(f'ATC_RAT_{axis.upper()}_D', 0.008)

    # PID transfer function
    num = [D, P, I]
    den = [1, 0, 0]

    # Frequency sweep
    w = np.logspace(-1, 3, 500)  # 0.1 to 1000 rad/s
    w_hz, mag, phase = signal.bode((num, den), w)

    # Find crossover frequency
    crossover_idx = np.where(mag >= 0)[0]
    crossover_freq = w_hz[crossover_idx[0]] / (2 * np.pi) if len(crossover_idx) > 0 else 10.0

    # Calculate margins
    phase_at_crossover = phase[crossover_idx[0]] if len(crossover_idx) > 0 else -90
    phase_margin = 180 + phase_at_crossover

    return {
        'frequencies': (w_hz / (2 * np.pi)).tolist(),
        'magnitude_db': mag.tolist(),
        'phase_deg': phase.tolist(),
        'crossover_freq': float(crossover_freq),
        'phase_margin': float(phase_margin),
        'gain_margin': 10.0  # Simplified
    }

# ============================================================
# OPTIMIZATION EXECUTION (BACKGROUND TASK)
# ============================================================

async def run_optimization(run_id: str, config: OptimizationConfig):
    """
    Main optimization loop - runs in background
    Integrates with real optimization system or uses mock data
    """
    try:
        # Update status
        active_runs[run_id]['status'] = 'running'
        await manager.broadcast(run_id, {
            'type': 'status_change',
            'status': 'running'
        })

        logger.info(f"[{run_id}] Starting optimization with {config.algorithm}")

        if OPTIMIZATION_AVAILABLE:
            # ===== REAL OPTIMIZATION WITH SITL =====
            logger.info(f"[{run_id}] Using REAL optimization system with ArduPilot SITL")

            # Initialize SITL manager and evaluator
            # Try multiple possible ArduPilot locations
            possible_paths = [
                os.path.expanduser("~/Documents/MC07_tuning/ardupilot"),
                os.path.expanduser("~/MC07_tuning/ardupilot"),
                "/home/user/MC07_tuning/ardupilot"
            ]

            ardupilot_path = None
            for path in possible_paths:
                if os.path.exists(os.path.join(path, "build/sitl/bin/arducopter")):
                    ardupilot_path = path
                    logger.info(f"[{run_id}] Found ArduPilot at: {ardupilot_path}")
                    break

            if ardupilot_path is None:
                raise FileNotFoundError(
                    f"ArduPilot not found. Searched:\n" +
                    "\n".join(f"  - {p}" for p in possible_paths)
                )

            sitl_manager = SITLManager(
                num_instances=config.parallel_instances,
                ardupilot_path=ardupilot_path
            )
            evaluator = PerformanceEvaluator()

            # Get parameter bounds for the selected phase
            from optimization_system.config import OPTIMIZATION_PHASES
            phase_config = OPTIMIZATION_PHASES.get(config.phase, OPTIMIZATION_PHASES['phase1_rate'])
            param_bounds = phase_config['bounds']

            # Create optimizer with correct parameter names
            if config.algorithm == 'genetic':
                optimizer = GeneticOptimizer(
                    sitl_manager=sitl_manager,
                    evaluator=evaluator,
                    max_generations=config.generations,  # GeneticOptimizer uses max_generations
                    population_size=config.population_size
                )
            else:
                optimizer = BayesianOptimizer(
                    sitl_manager=sitl_manager,
                    evaluator=evaluator,
                    max_iterations=config.generations  # BayesianOptimizer uses max_iterations
                )

            # Extract parameter names and bounds
            parameters = list(param_bounds.keys())

            logger.info(f"[{run_id}] Starting optimization for phase: {config.phase}")
            logger.info(f"[{run_id}] Optimizing {len(parameters)} parameters")

            # Run optimization (this will block until complete)
            # Returns: (best_params, best_fitness, convergence_history)
            best_params, best_fitness, convergence_history = optimizer.optimize(
                phase_name=config.phase,
                parameters=parameters,
                bounds=param_bounds,
                resume_from=None
            )

            # Update final state after optimization completes
            active_runs[run_id]['best_parameters'] = best_params
            active_runs[run_id]['best_fitness'] = best_fitness
            active_runs[run_id]['fitness_history'] = convergence_history

            # Clean up SITL instances
            sitl_manager.cleanup()

            # Mark as completed
            active_runs[run_id]['status'] = 'completed'
            active_runs[run_id]['best_fitness'] = best_fitness
            active_runs[run_id]['best_parameters'] = best_params

            await manager.broadcast(run_id, {
                'type': 'optimization_complete',
                'final_fitness': best_fitness,
                'final_parameters': best_params
            })

            logger.info(f"[{run_id}] REAL optimization completed - Best fitness: {best_fitness:.4f}")

        else:
            # ===== MOCK OPTIMIZATION (DEMO MODE) =====
            logger.info(f"[{run_id}] Using MOCK optimization (DEMO MODE)")

            for generation in range(config.generations):
                # Check if paused or stopped
                while active_runs[run_id]['status'] == 'paused':
                    await asyncio.sleep(1)

                if active_runs[run_id]['status'] == 'stopped':
                    logger.info(f"[{run_id}] Optimization stopped by user")
                    break

                # Simulate generation
                await asyncio.sleep(2)

                # Generate mock results
                import random
                best_fitness = 0.5 + (generation / config.generations) * 0.4 + random.uniform(-0.05, 0.05)
                avg_fitness = best_fitness - random.uniform(0.1, 0.2)

                best_params = {
                    'ATC_RAT_RLL_P': 0.15 + random.uniform(-0.02, 0.02),
                    'ATC_RAT_RLL_I': 0.10 + random.uniform(-0.01, 0.01),
                    'ATC_RAT_RLL_D': 0.008 + random.uniform(-0.001, 0.001),
                    'ATC_RAT_PIT_P': 0.15 + random.uniform(-0.02, 0.02),
                    'ATC_RAT_PIT_I': 0.10 + random.uniform(-0.01, 0.01),
                    'ATC_RAT_PIT_D': 0.008 + random.uniform(-0.001, 0.001),
                }

                # Update state
                active_runs[run_id]['current_generation'] = generation + 1
                active_runs[run_id]['best_fitness'] = best_fitness
                active_runs[run_id]['best_parameters'] = best_params
                active_runs[run_id]['fitness_history'].append(best_fitness)
                active_runs[run_id]['avg_fitness_history'].append(avg_fitness)
                active_runs[run_id]['completed_trials'] += config.population_size

                # Broadcast update
                await manager.broadcast(run_id, {
                    'type': 'generation_complete',
                    'generation': generation + 1,
                    'best_fitness': best_fitness,
                    'avg_fitness': avg_fitness,
                    'best_parameters': best_params,
                    'timestamp': datetime.now().isoformat()
                })

                logger.info(f"[{run_id}] Generation {generation+1}/{config.generations} - Best: {best_fitness:.4f}")

            # Mark as completed
            active_runs[run_id]['status'] = 'completed'
            await manager.broadcast(run_id, {
                'type': 'optimization_complete',
                'final_fitness': active_runs[run_id]['best_fitness'],
                'final_parameters': active_runs[run_id]['best_parameters']
            })

            logger.info(f"[{run_id}] MOCK optimization completed")

    except Exception as e:
        logger.error(f"[{run_id}] Optimization failed: {e}", exc_info=True)
        active_runs[run_id]['status'] = 'failed'
        await manager.broadcast(run_id, {
            'type': 'error',
            'message': str(e)
        })

# ============================================================
# DRONE SPECS ENDPOINT
# ============================================================

@app.get("/api/drone/specs")
async def get_drone_specs():
    """Get drone specifications"""
    return DRONE_SPECS

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_runs": len(active_runs),
        "websocket_connections": sum(len(conns) for conns in manager.active_connections.values())
    }

# ============================================================
# STARTUP & SHUTDOWN
# ============================================================

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("🚀 Drone Tuning API Server Started")
    logger.info("=" * 60)
    logger.info(f"📡 WebSocket endpoint: ws://localhost:8000/ws/{{run_id}}")
    logger.info(f"📊 API docs: http://localhost:8000/docs")
    logger.info(f"❤️  Health check: http://localhost:8000/health")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Server shutting down...")
    # Cleanup active runs
    for run_id in list(active_runs.keys()):
        if active_runs[run_id]['status'] == 'running':
            active_runs[run_id]['status'] = 'stopped'

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
