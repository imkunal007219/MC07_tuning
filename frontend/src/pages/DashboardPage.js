/**
 * Dashboard Page
 * Main optimization monitoring interface
 */

import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  Box,
  Grid,
  Card,
  CardHeader,
  CardContent,
  Button,
  Alert,
  CircularProgress,
  Typography,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  Download,
  Settings
} from '@mui/icons-material';

import FitnessChart from '../components/FitnessChart';
import ParameterPanel from '../components/ParameterPanel';
import SITLStatus from '../components/SITLStatus';
import MetricsSummary from '../components/MetricsSummary';

import { optimizationAPI, parametersAPI } from '../utils/api';
import wsManager from '../utils/websocket';
import {
  setCurrentRun,
  setStatus,
  updateProgress,
  updateFitness,
  updateBestParameters,
  setInitialState,
  resetOptimization
} from '../store/store';

function DashboardPage() {
  const dispatch = useDispatch();

  // Redux state
  const currentRun = useSelector(state => state.optimization.currentRun);
  const status = useSelector(state => state.optimization.status);
  const fitnessHistory = useSelector(state => state.optimization.fitnessHistory);
  const avgFitnessHistory = useSelector(state => state.optimization.avgFitnessHistory);
  const bestParameters = useSelector(state => state.optimization.bestParameters);
  const progress = useSelector(state => state.optimization.progress);

  // Local state
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [initializationMessage, setInitializationMessage] = useState('');
  const [initializationProgress, setInitializationProgress] = useState({ current: 0, total: 0 });
  const [config, setConfig] = useState({
    algorithm: 'genetic',
    phase: 'phase1_rate',
    generations: 50,
    population_size: 20,
    parallel_instances: 4,
    mutation_rate: 0.2,
    crossover_rate: 0.7
  });

  // Validate configuration
  const validateConfig = () => {
    const errors = [];

    if (config.generations < 1 || config.generations > 1000) {
      errors.push('Generations must be between 1 and 1000');
    }
    if (config.population_size < 4 || config.population_size > 100) {
      errors.push('Population size must be between 4 and 100');
    }
    if (config.parallel_instances < 1 || config.parallel_instances > 20) {
      errors.push('Parallel instances must be between 1 and 20');
    }
    if (config.mutation_rate < 0 || config.mutation_rate > 1) {
      errors.push('Mutation rate must be between 0.0 and 1.0');
    }
    if (config.crossover_rate < 0 || config.crossover_rate > 1) {
      errors.push('Crossover rate must be between 0.0 and 1.0');
    }

    return errors;
  };

  // Start optimization
  const handleStart = async () => {
    // Validate configuration
    const errors = validateConfig();
    if (errors.length > 0) {
      alert('Configuration errors:\n\n' + errors.join('\n'));
      return;
    }

    try {
      const response = await optimizationAPI.start(config);
      const runId = response.data.run_id;

      dispatch(setCurrentRun(runId));
      dispatch(setStatus('initializing')); // Changed from 'running' to 'initializing'
      dispatch(resetOptimization());
      dispatch(updateProgress({ totalGenerations: config.generations }));

      // Reset initialization state
      setInitializationMessage('Connecting to backend...');
      setInitializationProgress({ current: 0, total: config.parallel_instances });

      // Connect to WebSocket
      wsManager.connect(runId, () => {
        console.log('WebSocket connected for run:', runId);
      });

      // Register event handlers
      wsManager.on('initial_state', (data) => {
        dispatch(setInitialState(data.data));
      });

      wsManager.on('generation_complete', (data) => {
        dispatch(updateProgress({
          currentGeneration: data.generation,
          completedTrials: progress.completedTrials + config.population_size
        }));
        dispatch(updateFitness({
          bestFitness: data.best_fitness,
          avgFitness: data.avg_fitness
        }));
        dispatch(updateBestParameters(data.best_parameters));
      });

      wsManager.on('status_change', (data) => {
        dispatch(setStatus(data.status));
      });

      wsManager.on('initialization_progress', (data) => {
        setInitializationMessage(data.message);
        setInitializationProgress({ current: data.current, total: data.total });
      });

      wsManager.on('optimization_complete', (data) => {
        dispatch(setStatus('completed'));
        dispatch(updateBestParameters(data.final_parameters));
      });

      wsManager.on('error', (data) => {
        dispatch(setStatus('failed'));
        console.error('Optimization error:', data.message);
      });

      setConfigDialogOpen(false);

    } catch (error) {
      console.error('Failed to start optimization:', error);
      const errorMsg = error.response?.data?.detail || error.message;
      alert('Failed to start optimization:\n\n' + JSON.stringify(errorMsg, null, 2));
    }
  };

  // Pause optimization
  const handlePause = async () => {
    if (!currentRun) return;
    try {
      await optimizationAPI.pause(currentRun);
      dispatch(setStatus('paused'));
    } catch (error) {
      console.error('Failed to pause:', error);
    }
  };

  // Resume optimization
  const handleResume = async () => {
    if (!currentRun) return;
    try {
      await optimizationAPI.resume(currentRun);
      dispatch(setStatus('running'));
    } catch (error) {
      console.error('Failed to resume:', error);
    }
  };

  // Stop optimization
  const handleStop = async () => {
    if (!currentRun) return;
    try {
      await optimizationAPI.stop(currentRun);
      dispatch(setStatus('stopped'));
      wsManager.disconnect();
    } catch (error) {
      console.error('Failed to stop:', error);
    }
  };

  // Export parameters
  const handleExport = async () => {
    if (!currentRun) return;
    try {
      const response = await parametersAPI.export(currentRun, 'parm');
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `optimized_params_${currentRun}.parm`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error('Failed to export:', error);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (currentRun) {
        wsManager.disconnect();
      }
    };
  }, [currentRun]);

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Real-Time Optimization Monitor
      </Typography>

      {/* Control Buttons */}
      <Box sx={{ mb: 3, display: 'flex', gap: 1 }}>
        <Button
          variant="contained"
          color="primary"
          startIcon={<PlayArrow />}
          onClick={() => setConfigDialogOpen(true)}
          disabled={status === 'initializing' || status === 'running' || status === 'paused'}
        >
          Start New Optimization
        </Button>

        {status === 'running' && (
          <Button
            variant="outlined"
            color="warning"
            startIcon={<Pause />}
            onClick={handlePause}
          >
            Pause
          </Button>
        )}

        {status === 'paused' && (
          <Button
            variant="outlined"
            color="success"
            startIcon={<PlayArrow />}
            onClick={handleResume}
          >
            Resume
          </Button>
        )}

        {(status === 'running' || status === 'paused') && (
          <Button
            variant="outlined"
            color="error"
            startIcon={<Stop />}
            onClick={handleStop}
          >
            Stop
          </Button>
        )}

        {(status === 'completed' || status === 'stopped') && currentRun && (
          <Button
            variant="outlined"
            color="info"
            startIcon={<Download />}
            onClick={handleExport}
          >
            Export Parameters
          </Button>
        )}
      </Box>

      {/* Status Alert */}
      {status !== 'idle' && (
        <Alert
          severity={
            status === 'initializing' ? 'info' :
            status === 'running' ? 'info' :
            status === 'paused' ? 'warning' :
            status === 'completed' ? 'success' : 'error'
          }
          sx={{ mb: 3 }}
        >
          <Typography variant="body1" fontWeight="bold">
            Status: {status.toUpperCase()}
          </Typography>

          {/* Show initialization progress */}
          {status === 'initializing' && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="body2">
                {initializationMessage}
              </Typography>
              {initializationProgress.total > 0 && (
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                  <Box sx={{ width: '100%', mr: 1 }}>
                    <Box
                      sx={{
                        width: `${(initializationProgress.current / initializationProgress.total) * 100}%`,
                        height: 8,
                        bgcolor: 'primary.main',
                        borderRadius: 1,
                        transition: 'width 0.3s ease'
                      }}
                    />
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {initializationProgress.current}/{initializationProgress.total}
                  </Typography>
                </Box>
              )}
            </Box>
          )}

          {/* Show optimization progress */}
          {currentRun && status !== 'initializing' && (
            <Typography variant="body2">
              Run ID: {currentRun} | Generation: {progress.currentGeneration}/{progress.totalGenerations} |
              Trials: {progress.completedTrials}
              {progress.estimatedTimeRemaining && ` | ETA: ${Math.round(progress.estimatedTimeRemaining / 60)}m`}
            </Typography>
          )}
        </Alert>
      )}

      {/* Dashboard Grid */}
      <Grid container spacing={3}>
        {/* Fitness Evolution Chart */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardHeader title="📈 Fitness Evolution" />
            <CardContent>
              <FitnessChart
                fitnessHistory={fitnessHistory}
                avgFitnessHistory={avgFitnessHistory}
                totalGenerations={progress.totalGenerations}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Best Parameters */}
        <Grid item xs={12} lg={4}>
          <ParameterPanel parameters={bestParameters} />
        </Grid>

        {/* Metrics Summary */}
        <Grid item xs={12} md={6}>
          <MetricsSummary
            bestFitness={fitnessHistory[fitnessHistory.length - 1] || 0}
            currentGeneration={progress.currentGeneration}
            totalGenerations={progress.totalGenerations}
            completedTrials={progress.completedTrials}
          />
        </Grid>

        {/* SITL Status */}
        <Grid item xs={12} md={6}>
          <SITLStatus instances={config.parallel_instances} />
        </Grid>
      </Grid>

      {/* Configuration Dialog */}
      <Dialog open={configDialogOpen} onClose={() => setConfigDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Optimization Configuration</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
            <FormControl fullWidth>
              <InputLabel>Algorithm</InputLabel>
              <Select
                value={config.algorithm}
                label="Algorithm"
                onChange={(e) => setConfig({ ...config, algorithm: e.target.value })}
              >
                <MenuItem value="genetic">Genetic Algorithm</MenuItem>
                <MenuItem value="bayesian">Bayesian Optimization</MenuItem>
              </Select>
            </FormControl>

            <FormControl fullWidth>
              <InputLabel>Phase</InputLabel>
              <Select
                value={config.phase}
                label="Phase"
                onChange={(e) => setConfig({ ...config, phase: e.target.value })}
              >
                <MenuItem value="phase1_rate">Phase 1: Rate Controllers</MenuItem>
                <MenuItem value="phase2_attitude">Phase 2: Attitude Controllers</MenuItem>
                <MenuItem value="phase3_position">Phase 3: Position Controllers</MenuItem>
                <MenuItem value="phase4_advanced">Phase 4: Advanced Parameters</MenuItem>
              </Select>
            </FormControl>

            <TextField
              label="Generations"
              type="number"
              value={config.generations}
              onChange={(e) => setConfig({ ...config, generations: parseInt(e.target.value) })}
              fullWidth
              inputProps={{ min: 1, max: 1000 }}
              helperText="Range: 1-1000 generations"
            />

            <TextField
              label="Population Size"
              type="number"
              value={config.population_size}
              onChange={(e) => setConfig({ ...config, population_size: parseInt(e.target.value) })}
              fullWidth
              inputProps={{ min: 4, max: 100 }}
              helperText="Range: 4-100 individuals (minimum 4 required)"
              error={config.population_size < 4 || config.population_size > 100}
            />

            <TextField
              label="Parallel Instances"
              type="number"
              value={config.parallel_instances}
              onChange={(e) => setConfig({ ...config, parallel_instances: parseInt(e.target.value) })}
              fullWidth
              inputProps={{ min: 1, max: 20 }}
              helperText="Range: 1-20 SITL instances"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfigDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleStart} variant="contained" color="primary">
            Start Optimization
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default DashboardPage;
