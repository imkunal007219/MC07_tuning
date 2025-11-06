"""
Database models and schema for drone tuning system
Uses SQLAlchemy with PostgreSQL/TimescaleDB for time-series data
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, JSON, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://drone_user:drone_password@localhost:5432/drone_tuning"
)

# Create engine
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ============================================================
# DATABASE MODELS
# ============================================================

class OptimizationRun(Base):
    """Optimization run metadata"""
    __tablename__ = "optimization_runs"

    id = Column(String(8), primary_key=True)  # run_id
    algorithm = Column(String(20), nullable=False)  # 'genetic' or 'bayesian'
    phase = Column(String(50), nullable=False)  # 'phase1_rate', etc.
    status = Column(String(20), default='running')  # 'running', 'paused', 'completed', 'failed', 'stopped'

    # Timing
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)

    # Configuration
    config = Column(JSON)  # Full config dict
    generations = Column(Integer)
    population_size = Column(Integer)
    parallel_instances = Column(Integer)

    # Results
    best_fitness = Column(Float, default=0.0)
    best_parameters = Column(JSON)  # Dict of parameter: value

    # History
    fitness_history = Column(JSON)  # List of best fitness per generation
    avg_fitness_history = Column(JSON)  # List of avg fitness per generation

    # Relationships
    trials = relationship("Trial", back_populates="run", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            'id': self.id,
            'algorithm': self.algorithm,
            'phase': self.phase,
            'status': self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'config': self.config,
            'best_fitness': self.best_fitness,
            'best_parameters': self.best_parameters,
            'fitness_history': self.fitness_history,
            'avg_fitness_history': self.avg_fitness_history
        }


class Trial(Base):
    """Individual optimization trial"""
    __tablename__ = "trials"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(8), ForeignKey('optimization_runs.id'), nullable=False)

    # Trial info
    generation = Column(Integer, nullable=False)
    individual_id = Column(Integer, nullable=False)  # Index within generation
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Parameters tested
    parameters = Column(JSON, nullable=False)

    # Results
    fitness = Column(Float, nullable=False)
    crashed = Column(Boolean, default=False)

    # Detailed metrics
    metrics = Column(JSON)  # All performance metrics

    # Flight data reference
    log_file = Column(String(255), nullable=True)  # Path to flight log

    # Relationships
    run = relationship("OptimizationRun", back_populates="trials")
    telemetry = relationship("TelemetryPoint", back_populates="trial", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            'id': self.id,
            'run_id': self.run_id,
            'generation': self.generation,
            'individual_id': self.individual_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'parameters': self.parameters,
            'fitness': self.fitness,
            'crashed': self.crashed,
            'metrics': self.metrics
        }


class TelemetryPoint(Base):
    """
    Time-series telemetry data

    Note: In production with TimescaleDB, this should be a hypertable
    partitioned by timestamp for efficient time-series queries
    """
    __tablename__ = "telemetry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trial_id = Column(Integer, ForeignKey('trials.id'), nullable=False)

    # Timestamp (milliseconds since trial start)
    timestamp_ms = Column(Integer, nullable=False)

    # Attitude (degrees)
    roll = Column(Float)
    pitch = Column(Float)
    yaw = Column(Float)

    # Angular rates (deg/s)
    roll_rate = Column(Float)
    pitch_rate = Column(Float)
    yaw_rate = Column(Float)

    # Position (meters, NED frame)
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)

    # Velocity (m/s)
    vx = Column(Float)
    vy = Column(Float)
    vz = Column(Float)

    # Motors (PWM, 1000-2000)
    motor1 = Column(Integer)
    motor2 = Column(Integer)
    motor3 = Column(Integer)
    motor4 = Column(Integer)

    # Target values (for tracking error calculation)
    target_roll = Column(Float, nullable=True)
    target_pitch = Column(Float, nullable=True)
    target_yaw = Column(Float, nullable=True)
    target_altitude = Column(Float, nullable=True)

    # Relationships
    trial = relationship("Trial", back_populates="telemetry")


class ParameterBound(Base):
    """Parameter bounds and metadata"""
    __tablename__ = "parameter_bounds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    parameter_name = Column(String(50), unique=True, nullable=False)

    # Bounds
    min_value = Column(Float, nullable=False)
    max_value = Column(Float, nullable=False)
    default_value = Column(Float, nullable=True)

    # Metadata
    description = Column(Text)
    unit = Column(String(20))
    category = Column(String(50))  # 'rate', 'attitude', 'position', 'filter', etc.
    phase = Column(String(50))  # Which optimization phase this belongs to


class AnalysisResult(Base):
    """Stored analysis results (correlations, sensitivity, etc.)"""
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(8), ForeignKey('optimization_runs.id'))
    analysis_type = Column(String(50))  # 'correlation', 'sensitivity', 'pareto', etc.
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Analysis data
    data = Column(JSON)  # Flexible storage for different analysis types

    # Optional description
    description = Column(Text)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_database():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully")


def drop_all_tables():
    """Drop all tables (use with caution!)"""
    Base.metadata.drop_all(bind=engine)
    print("⚠️  All tables dropped")


def get_db():
    """Dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_optimization_run(db, run_id: str, config: dict) -> OptimizationRun:
    """Create new optimization run record"""
    run = OptimizationRun(
        id=run_id,
        algorithm=config['algorithm'],
        phase=config['phase'],
        config=config,
        generations=config['generations'],
        population_size=config['population_size'],
        parallel_instances=config['parallel_instances'],
        best_parameters={},
        fitness_history=[],
        avg_fitness_history=[]
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def update_run_status(db, run_id: str, status: str):
    """Update run status"""
    run = db.query(OptimizationRun).filter(OptimizationRun.id == run_id).first()
    if run:
        run.status = status
        if status in ['completed', 'stopped', 'failed']:
            run.end_time = datetime.utcnow()
        db.commit()


def add_trial_result(db, run_id: str, generation: int, individual_id: int,
                     parameters: dict, fitness: float, crashed: bool, metrics: dict,
                     log_file: str = None) -> Trial:
    """Add trial result to database"""
    trial = Trial(
        run_id=run_id,
        generation=generation,
        individual_id=individual_id,
        parameters=parameters,
        fitness=fitness,
        crashed=crashed,
        metrics=metrics,
        log_file=log_file
    )
    db.add(trial)
    db.commit()
    db.refresh(trial)
    return trial


def update_best_result(db, run_id: str, fitness: float, parameters: dict):
    """Update best fitness and parameters for a run"""
    run = db.query(OptimizationRun).filter(OptimizationRun.id == run_id).first()
    if run:
        run.best_fitness = fitness
        run.best_parameters = parameters
        db.commit()


def add_generation_history(db, run_id: str, best_fitness: float, avg_fitness: float):
    """Append generation results to history"""
    run = db.query(OptimizationRun).filter(OptimizationRun.id == run_id).first()
    if run:
        if run.fitness_history is None:
            run.fitness_history = []
        if run.avg_fitness_history is None:
            run.avg_fitness_history = []

        run.fitness_history.append(best_fitness)
        run.avg_fitness_history.append(avg_fitness)
        db.commit()


def add_telemetry_batch(db, trial_id: int, telemetry_data: list):
    """Add batch of telemetry points"""
    points = []
    for data in telemetry_data:
        point = TelemetryPoint(
            trial_id=trial_id,
            timestamp_ms=data['timestamp_ms'],
            roll=data.get('roll'),
            pitch=data.get('pitch'),
            yaw=data.get('yaw'),
            roll_rate=data.get('roll_rate'),
            pitch_rate=data.get('pitch_rate'),
            yaw_rate=data.get('yaw_rate'),
            x=data.get('x'),
            y=data.get('y'),
            z=data.get('z'),
            vx=data.get('vx'),
            vy=data.get('vy'),
            vz=data.get('vz'),
            motor1=data.get('motor1'),
            motor2=data.get('motor2'),
            motor3=data.get('motor3'),
            motor4=data.get('motor4'),
            target_roll=data.get('target_roll'),
            target_pitch=data.get('target_pitch'),
            target_yaw=data.get('target_yaw'),
            target_altitude=data.get('target_altitude')
        )
        points.append(point)

    db.bulk_save_objects(points)
    db.commit()


def get_trial_telemetry(db, trial_id: int) -> list:
    """Retrieve all telemetry for a trial"""
    points = db.query(TelemetryPoint).filter(
        TelemetryPoint.trial_id == trial_id
    ).order_by(TelemetryPoint.timestamp_ms).all()

    # Convert to dict format
    telemetry = {
        'timestamps': [],
        'attitude': {'roll': [], 'pitch': [], 'yaw': []},
        'rates': {'roll': [], 'pitch': [], 'yaw': []},
        'position': {'x': [], 'y': [], 'z': []},
        'velocity': {'vx': [], 'vy': [], 'vz': []},
        'motors': {'motor1': [], 'motor2': [], 'motor3': [], 'motor4': []}
    }

    for point in points:
        telemetry['timestamps'].append(point.timestamp_ms / 1000.0)  # Convert to seconds
        telemetry['attitude']['roll'].append(point.roll)
        telemetry['attitude']['pitch'].append(point.pitch)
        telemetry['attitude']['yaw'].append(point.yaw)
        telemetry['rates']['roll'].append(point.roll_rate)
        telemetry['rates']['pitch'].append(point.pitch_rate)
        telemetry['rates']['yaw'].append(point.yaw_rate)
        telemetry['position']['x'].append(point.x)
        telemetry['position']['y'].append(point.y)
        telemetry['position']['z'].append(point.z)
        telemetry['velocity']['vx'].append(point.vx)
        telemetry['velocity']['vy'].append(point.vy)
        telemetry['velocity']['vz'].append(point.vz)
        telemetry['motors']['motor1'].append(point.motor1)
        telemetry['motors']['motor2'].append(point.motor2)
        telemetry['motors']['motor3'].append(point.motor3)
        telemetry['motors']['motor4'].append(point.motor4)

    return telemetry


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "init":
        print("Initializing database...")
        init_database()
    elif len(sys.argv) > 1 and sys.argv[1] == "drop":
        confirm = input("Are you sure you want to drop all tables? (yes/no): ")
        if confirm.lower() == "yes":
            drop_all_tables()
        else:
            print("Cancelled")
    else:
        print("Usage:")
        print("  python database.py init   # Create tables")
        print("  python database.py drop   # Drop all tables")
