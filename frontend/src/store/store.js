/**
 * Redux Store for Drone Tuning Dashboard
 * Manages global state for optimization runs, telemetry, and analysis
 */

import { configureStore, createSlice } from '@reduxjs/toolkit';

// ============================================================
// OPTIMIZATION SLICE
// ============================================================

const optimizationSlice = createSlice({
  name: 'optimization',
  initialState: {
    currentRun: null,
    status: 'idle', // 'idle', 'running', 'paused', 'completed', 'failed', 'stopped', 'initializing'
    progress: {
      currentGeneration: 0,
      totalGenerations: 100,
      completedTrials: 0,
      estimatedTimeRemaining: 0
    },
    fitnessHistory: [],
    avgFitnessHistory: [],
    bestFitness: 0,
    bestParameters: {},
    currentParameters: {},
    allRuns: []
  },
  reducers: {
    setCurrentRun: (state, action) => {
      state.currentRun = action.payload;
    },
    setStatus: (state, action) => {
      state.status = action.payload;
    },
    updateProgress: (state, action) => {
      state.progress = { ...state.progress, ...action.payload };
    },
    updateFitness: (state, action) => {
      const { bestFitness, avgFitness } = action.payload;
      state.fitnessHistory.push(bestFitness);
      if (avgFitness !== undefined) {
        state.avgFitnessHistory.push(avgFitness);
      }
      state.bestFitness = bestFitness;
    },
    updateBestParameters: (state, action) => {
      state.bestParameters = action.payload;
      state.currentParameters = action.payload;
    },
    setInitialState: (state, action) => {
      const { status, current_generation, best_fitness, best_parameters, fitness_history } = action.payload;
      state.status = status;
      state.progress.currentGeneration = current_generation;
      state.bestFitness = best_fitness;
      state.bestParameters = best_parameters;
      if (fitness_history && fitness_history.length > 0) {
        state.fitnessHistory = fitness_history;
      }
    },
    addRun: (state, action) => {
      state.allRuns.push(action.payload);
    },
    resetOptimization: (state) => {
      state.currentRun = null;
      state.status = 'idle';
      state.fitnessHistory = [];
      state.avgFitnessHistory = [];
      state.bestFitness = 0;
      state.bestParameters = {};
      state.currentParameters = {};
      state.progress = {
        currentGeneration: 0,
        totalGenerations: 100,
        completedTrials: 0,
        estimatedTimeRemaining: 0
      };
    }
  }
});

export const {
  setCurrentRun,
  setStatus,
  updateProgress,
  updateFitness,
  updateBestParameters,
  setInitialState,
  addRun,
  resetOptimization
} = optimizationSlice.actions;

// ============================================================
// TELEMETRY SLICE
// ============================================================

const telemetrySlice = createSlice({
  name: 'telemetry',
  initialState: {
    currentTrial: null,
    selectedTrial: null,
    data: {
      timestamps: [],
      attitude: { roll: [], pitch: [], yaw: [] },
      rates: { roll: [], pitch: [], yaw: [] },
      position: { x: [], y: [], z: [] },
      velocity: { vx: [], vy: [], vz: [] },
      motors: { motor1: [], motor2: [], motor3: [], motor4: [] }
    },
    metrics: {},
    trials: []
  },
  reducers: {
    setCurrentTrial: (state, action) => {
      state.currentTrial = action.payload;
    },
    setSelectedTrial: (state, action) => {
      state.selectedTrial = action.payload;
    },
    setTelemetryData: (state, action) => {
      state.data = action.payload;
    },
    setMetrics: (state, action) => {
      state.metrics = action.payload;
    },
    setTrials: (state, action) => {
      state.trials = action.payload;
    },
    clearTelemetry: (state) => {
      state.data = {
        timestamps: [],
        attitude: { roll: [], pitch: [], yaw: [] },
        rates: { roll: [], pitch: [], yaw: [] },
        position: { x: [], y: [], z: [] },
        velocity: { vx: [], vy: [], vz: [] },
        motors: { motor1: [], motor2: [], motor3: [], motor4: [] }
      };
      state.metrics = {};
    }
  }
});

export const {
  setCurrentTrial,
  setSelectedTrial,
  setTelemetryData,
  setMetrics,
  setTrials,
  clearTelemetry
} = telemetrySlice.actions;

// ============================================================
// ANALYSIS SLICE
// ============================================================

const analysisSlice = createSlice({
  name: 'analysis',
  initialState: {
    correlationMatrix: null,
    parameterNames: [],
    bodeData: null,
    nyquistData: null,
    stepResponseData: null,
    sensitivityData: null
  },
  reducers: {
    setCorrelationMatrix: (state, action) => {
      state.correlationMatrix = action.payload.matrix;
      state.parameterNames = action.payload.parameters;
    },
    setBodeData: (state, action) => {
      state.bodeData = action.payload;
    },
    setNyquistData: (state, action) => {
      state.nyquistData = action.payload;
    },
    setStepResponseData: (state, action) => {
      state.stepResponseData = action.payload;
    },
    setSensitivityData: (state, action) => {
      state.sensitivityData = action.payload;
    }
  }
});

export const {
  setCorrelationMatrix,
  setBodeData,
  setNyquistData,
  setStepResponseData,
  setSensitivityData
} = analysisSlice.actions;

// ============================================================
// UI SLICE
// ============================================================

const uiSlice = createSlice({
  name: 'ui',
  initialState: {
    selectedTab: 'dashboard',
    selectedAxis: 'roll',
    theme: 'dark',
    notifications: [],
    loading: false,
    error: null
  },
  reducers: {
    setSelectedTab: (state, action) => {
      state.selectedTab = action.payload;
    },
    setSelectedAxis: (state, action) => {
      state.selectedAxis = action.payload;
    },
    setTheme: (state, action) => {
      state.theme = action.payload;
    },
    addNotification: (state, action) => {
      state.notifications.push({
        id: Date.now(),
        ...action.payload
      });
    },
    removeNotification: (state, action) => {
      state.notifications = state.notifications.filter(n => n.id !== action.payload);
    },
    setLoading: (state, action) => {
      state.loading = action.payload;
    },
    setError: (state, action) => {
      state.error = action.payload;
    }
  }
});

export const {
  setSelectedTab,
  setSelectedAxis,
  setTheme,
  addNotification,
  removeNotification,
  setLoading,
  setError
} = uiSlice.actions;

// ============================================================
// CONFIGURE STORE
// ============================================================

const store = configureStore({
  reducer: {
    optimization: optimizationSlice.reducer,
    telemetry: telemetrySlice.reducer,
    analysis: analysisSlice.reducer,
    ui: uiSlice.reducer
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // Ignore these paths in the state
        ignoredActions: [],
        ignoredPaths: []
      }
    })
});

export default store;
