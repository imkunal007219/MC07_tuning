/**
 * Control Systems Page
 * Frequency response analysis (Bode, Nyquist, etc.)
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
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';

import BodePlot from '../components/BodePlot';
import { analysisAPI } from '../utils/api';

function ControlPage() {
  const currentRun = useSelector(state => state.optimization.currentRun);
  const [axis, setAxis] = useState('roll');
  const [bodeData, setBodeData] = useState(null);

  useEffect(() => {
    if (currentRun) {
      loadBodeData();
    }
  }, [currentRun, axis]);

  const loadBodeData = async () => {
    try {
      const response = await analysisAPI.getFrequencyResponse(currentRun, axis);
      setBodeData(response.data);
    } catch (error) {
      console.error('Failed to load Bode data:', error);
    }
  };

  if (!currentRun) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Control Systems Analysis
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Start an optimization run to view control systems analysis.
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Control Systems Analysis
      </Typography>

      {/* Axis Selector */}
      <FormControl sx={{ mb: 3, minWidth: 200 }}>
        <InputLabel>Axis</InputLabel>
        <Select
          value={axis}
          label="Axis"
          onChange={(e) => setAxis(e.target.value)}
        >
          <MenuItem value="roll">Roll</MenuItem>
          <MenuItem value="pitch">Pitch</MenuItem>
          <MenuItem value="yaw">Yaw</MenuItem>
        </Select>
      </FormControl>

      <Grid container spacing={3}>
        {/* Bode Plot */}
        <Grid item xs={12}>
          <Card>
            <CardHeader title={`📊 Bode Plot - ${axis.toUpperCase()} Axis`} />
            <CardContent>
              <BodePlot data={bodeData} axis={axis} />
            </CardContent>
          </Card>
        </Grid>

        {/* Additional Analysis */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="📈 Step Response" />
            <CardContent>
              <Typography variant="body2" color="text.secondary">
                Step response analysis will be displayed here
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="🎯 Nyquist Plot" />
            <CardContent>
              <Typography variant="body2" color="text.secondary">
                Nyquist diagram will be displayed here
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default ControlPage;
