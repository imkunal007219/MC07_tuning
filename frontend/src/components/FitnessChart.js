/**
 * Fitness Evolution Chart
 * Real-time line chart showing best and average fitness over generations
 */

import React from 'react';
import Plot from 'react-plotly.js';
import { Box, CircularProgress, Typography } from '@mui/material';

function FitnessChart({ fitnessHistory, avgFitnessHistory, totalGenerations }) {
  if (!fitnessHistory || fitnessHistory.length === 0) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <CircularProgress />
        <Typography variant="body2" sx={{ mt: 2 }} color="text.secondary">
          Waiting for optimization to start...
        </Typography>
      </Box>
    );
  }

  const generations = Array.from({ length: fitnessHistory.length }, (_, i) => i + 1);

  const traces = [
    {
      x: generations,
      y: fitnessHistory,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Best Fitness',
      line: {
        color: '#4caf50',
        width: 3
      },
      marker: {
        size: 6,
        color: '#4caf50'
      }
    }
  ];

  if (avgFitnessHistory && avgFitnessHistory.length > 0) {
    traces.push({
      x: generations,
      y: avgFitnessHistory,
      type: 'scatter',
      mode: 'lines',
      name: 'Avg Fitness',
      line: {
        color: '#2196f3',
        width: 2,
        dash: 'dot'
      }
    });
  }

  const layout = {
    xaxis: {
      title: 'Generation',
      gridcolor: '#2c3e50',
      range: [0, Math.max(totalGenerations, fitnessHistory.length + 5)]
    },
    yaxis: {
      title: 'Fitness',
      gridcolor: '#2c3e50',
      range: [0, 1.1]
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
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
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

export default FitnessChart;
