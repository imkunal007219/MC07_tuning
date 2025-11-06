/**
 * Telemetry Chart
 * Time-series plot for attitude, rates, position, etc.
 */

import React from 'react';
import Plot from 'react-plotly.js';
import { Box, Typography, CircularProgress } from '@mui/material';

function TelemetryChart({ telemetryData, dataType }) {
  if (!telemetryData || !telemetryData.timestamps) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <CircularProgress />
        <Typography variant="body2" sx={{ mt: 2 }} color="text.secondary">
          Loading telemetry data...
        </Typography>
      </Box>
    );
  }

  const { timestamps } = telemetryData;
  const data = telemetryData[dataType];

  if (!data) {
    return (
      <Typography variant="body2" color="text.secondary">
        No {dataType} data available
      </Typography>
    );
  }

  const traces = Object.entries(data).map(([key, values]) => ({
    x: timestamps,
    y: values,
    type: 'scatter',
    mode: 'lines',
    name: key.toUpperCase(),
    line: { width: 2 }
  }));

  const layout = {
    xaxis: {
      title: 'Time (s)',
      gridcolor: '#2c3e50'
    },
    yaxis: {
      title: getYAxisLabel(dataType),
      gridcolor: '#2c3e50'
    },
    plot_bgcolor: '#132f4c',
    paper_bgcolor: '#132f4c',
    font: { color: '#ffffff' },
    height: 400,
    margin: { t: 20, b: 50, l: 60, r: 40 },
    hovermode: 'x unified',
    legend: {
      x: 0.02,
      y: 0.98,
      bgcolor: 'rgba(0,0,0,0.5)'
    }
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false
  };

  return (
    <Plot
      data={traces}
      layout={layout}
      config={config}
      style={{ width: '100%' }}
      useResizeHandler={true}
    />
  );
}

function getYAxisLabel(dataType) {
  const labels = {
    attitude: 'Angle (degrees)',
    rates: 'Angular Rate (deg/s)',
    position: 'Position (m)',
    velocity: 'Velocity (m/s)',
    motors: 'PWM (us)'
  };
  return labels[dataType] || 'Value';
}

export default TelemetryChart;
