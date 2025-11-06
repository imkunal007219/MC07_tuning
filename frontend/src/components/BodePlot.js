/**
 * Bode Plot
 * Frequency response visualization with magnitude and phase
 */

import React from 'react';
import Plot from 'react-plotly.js';
import { Box, Typography, CircularProgress, Alert } from '@mui/material';

function BodePlot({ data, axis }) {
  if (!data) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <CircularProgress />
        <Typography variant="body2" sx={{ mt: 2 }} color="text.secondary">
          Loading frequency response data...
        </Typography>
      </Box>
    );
  }

  const { frequencies, magnitude_db, phase_deg, crossover_freq, phase_margin, gain_margin } = data;

  // Determine stability
  const isStable = phase_margin > 45 && gain_margin > 6;
  const stabilityLevel = phase_margin > 60 ? 'excellent' :
                         phase_margin > 45 ? 'good' :
                         phase_margin > 30 ? 'marginal' : 'unstable';

  // Magnitude plot
  const magTrace = {
    x: frequencies,
    y: magnitude_db,
    type: 'scatter',
    mode: 'lines',
    name: 'Magnitude',
    line: { color: '#2196f3', width: 2 }
  };

  const zeroDbLine = {
    x: frequencies,
    y: Array(frequencies.length).fill(0),
    type: 'scatter',
    mode: 'lines',
    name: '0 dB',
    line: { color: '#ff0000', dash: 'dash', width: 1 }
  };

  const magLayout = {
    xaxis: {
      type: 'log',
      title: 'Frequency (Hz)',
      gridcolor: '#2c3e50'
    },
    yaxis: {
      title: 'Magnitude (dB)',
      gridcolor: '#2c3e50'
    },
    plot_bgcolor: '#132f4c',
    paper_bgcolor: '#132f4c',
    font: { color: '#ffffff' },
    height: 300,
    margin: { t: 20, b: 50, l: 60, r: 40 },
    showlegend: true,
    legend: { x: 0.02, y: 0.98 },
    shapes: [
      {
        type: 'line',
        x0: crossover_freq,
        x1: crossover_freq,
        y0: -60,
        y1: 40,
        line: { color: '#4caf50', dash: 'dot', width: 2 }
      }
    ],
    annotations: [
      {
        x: Math.log10(crossover_freq),
        y: 0,
        text: `ωc = ${crossover_freq.toFixed(1)} Hz`,
        showarrow: true,
        arrowhead: 2,
        ax: 40,
        ay: -40,
        bgcolor: 'rgba(0,0,0,0.5)',
        font: { color: '#ffffff' }
      }
    ]
  };

  // Phase plot
  const phaseTrace = {
    x: frequencies,
    y: phase_deg,
    type: 'scatter',
    mode: 'lines',
    name: 'Phase',
    line: { color: '#ff9800', width: 2 }
  };

  const minus180Line = {
    x: frequencies,
    y: Array(frequencies.length).fill(-180),
    type: 'scatter',
    mode: 'lines',
    name: '-180°',
    line: { color: '#ff0000', dash: 'dash', width: 1 }
  };

  const phaseLayout = {
    xaxis: {
      type: 'log',
      title: 'Frequency (Hz)',
      gridcolor: '#2c3e50'
    },
    yaxis: {
      title: 'Phase (degrees)',
      gridcolor: '#2c3e50'
    },
    plot_bgcolor: '#132f4c',
    paper_bgcolor: '#132f4c',
    font: { color: '#ffffff' },
    height: 300,
    margin: { t: 20, b: 50, l: 60, r: 40 },
    showlegend: true,
    legend: { x: 0.02, y: 0.02 },
    shapes: [
      {
        type: 'line',
        x0: crossover_freq,
        x1: crossover_freq,
        y0: -270,
        y1: -90,
        line: { color: '#4caf50', dash: 'dot', width: 2 }
      }
    ],
    annotations: [
      {
        x: Math.log10(crossover_freq),
        y: -180 + phase_margin,
        text: `PM = ${phase_margin.toFixed(1)}°`,
        showarrow: true,
        arrowhead: 2,
        ax: 40,
        ay: -40,
        bgcolor: 'rgba(0,0,0,0.5)',
        font: { color: '#ffffff' }
      }
    ]
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false
  };

  return (
    <Box>
      {/* Magnitude Plot */}
      <Plot
        data={[magTrace, zeroDbLine]}
        layout={magLayout}
        config={config}
        style={{ width: '100%' }}
        useResizeHandler={true}
      />

      {/* Phase Plot */}
      <Plot
        data={[phaseTrace, minus180Line]}
        layout={phaseLayout}
        config={config}
        style={{ width: '100%' }}
        useResizeHandler={true}
      />

      {/* Stability Assessment */}
      <Alert severity={isStable ? 'success' : 'warning'} sx={{ mt: 2 }}>
        <Typography variant="subtitle1" fontWeight="bold">
          Stability Assessment: {stabilityLevel.toUpperCase()}
        </Typography>
        <Typography variant="body2">
          • Phase Margin: {phase_margin.toFixed(1)}° (Target: &gt; 45°, Excellent: &gt; 60°)
        </Typography>
        <Typography variant="body2">
          • Gain Margin: {gain_margin.toFixed(1)} dB (Target: &gt; 6 dB, Excellent: &gt; 12 dB)
        </Typography>
        <Typography variant="body2">
          • Crossover Frequency: {crossover_freq.toFixed(1)} Hz
        </Typography>

        {!isStable && (
          <Typography variant="body2" sx={{ mt: 1, fontWeight: 'bold' }}>
            ⚠️ Warning: System may exhibit oscillations. Consider reducing P gain or increasing D gain.
          </Typography>
        )}
      </Alert>
    </Box>
  );
}

export default BodePlot;
