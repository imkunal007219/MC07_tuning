/**
 * Telemetry Page
 * View detailed flight telemetry and 3D trajectory
 */

import React, { useState, useEffect } from 'react';
import { useSelector } from 'react-redux';
import {
  Box,
  Grid,
  Card,
  CardHeader,
  CardContent,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel
} from '@mui/material';

import TelemetryChart from '../components/TelemetryChart';
import TrajectoryViewer3D from '../components/TrajectoryViewer3D';
import { telemetryAPI } from '../utils/api';

function TelemetryPage() {
  const currentRun = useSelector(state => state.optimization.currentRun);
  const [trials, setTrials] = useState([]);
  const [selectedTrial, setSelectedTrial] = useState(null);
  const [telemetryData, setTelemetryData] = useState(null);

  // Load trials list
  useEffect(() => {
    if (currentRun) {
      loadTrials();
    }
  }, [currentRun]);

  const loadTrials = async () => {
    try {
      const response = await telemetryAPI.getTrials(currentRun);
      setTrials(response.data.trials);
      if (response.data.trials.length > 0) {
        setSelectedTrial(response.data.trials[0].id);
      }
    } catch (error) {
      console.error('Failed to load trials:', error);
    }
  };

  // Load trial telemetry
  useEffect(() => {
    if (currentRun && selectedTrial) {
      loadTelemetry();
    }
  }, [currentRun, selectedTrial]);

  const loadTelemetry = async () => {
    try {
      const response = await telemetryAPI.getTrialData(currentRun, selectedTrial);
      setTelemetryData(response.data);
    } catch (error) {
      console.error('Failed to load telemetry:', error);
    }
  };

  if (!currentRun) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Telemetry Viewer
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Start an optimization run to view telemetry data.
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Flight Telemetry
      </Typography>

      {/* Trial Selector */}
      <FormControl sx={{ mb: 3, minWidth: 300 }}>
        <InputLabel>Select Trial</InputLabel>
        <Select
          value={selectedTrial || ''}
          label="Select Trial"
          onChange={(e) => setSelectedTrial(e.target.value)}
        >
          {trials.map((trial) => (
            <MenuItem key={trial.id} value={trial.id}>
              Trial #{trial.id} - Fitness: {trial.fitness?.toFixed(4) || 'N/A'}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      <Grid container spacing={3}>
        {/* 3D Trajectory */}
        <Grid item xs={12} lg={6}>
          <Card sx={{ height: 600 }}>
            <CardHeader title="🛸 3D Trajectory" />
            <CardContent>
              <TrajectoryViewer3D telemetryData={telemetryData} />
            </CardContent>
          </Card>
        </Grid>

        {/* Time-Series Charts */}
        <Grid item xs={12} lg={6}>
          <Card sx={{ height: 600 }}>
            <CardHeader title="📈 Attitude" />
            <CardContent>
              <TelemetryChart
                telemetryData={telemetryData}
                dataType="attitude"
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={6}>
          <Card>
            <CardHeader title="📈 Angular Rates" />
            <CardContent>
              <TelemetryChart
                telemetryData={telemetryData}
                dataType="rates"
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={6}>
          <Card>
            <CardHeader title="📈 Position" />
            <CardContent>
              <TelemetryChart
                telemetryData={telemetryData}
                dataType="position"
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Metrics */}
        {telemetryData?.metrics && (
          <Grid item xs={12}>
            <Card>
              <CardHeader title="📊 Performance Metrics" />
              <CardContent>
                <Grid container spacing={2}>
                  {Object.entries(telemetryData.metrics).map(([key, value]) => (
                    <Grid item xs={6} sm={3} key={key}>
                      <Box sx={{ p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
                        <Typography variant="caption" color="text.secondary">
                          {key.replace(/_/g, ' ').toUpperCase()}
                        </Typography>
                        <Typography variant="h6">
                          {typeof value === 'number' ? value.toFixed(3) : value}
                        </Typography>
                      </Box>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
}

export default TelemetryPage;
