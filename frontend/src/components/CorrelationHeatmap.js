/**
 * Correlation Heatmap
 * Visualize parameter correlations
 */

import React from 'react';
import Plot from 'react-plotly.js';
import { Box, Typography, CircularProgress } from '@mui/material';

function CorrelationHeatmap({ data }) {
  if (!data || !data.matrix) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <CircularProgress />
        <Typography variant="body2" sx={{ mt: 2 }} color="text.secondary">
          Loading correlation data...
        </Typography>
      </Box>
    );
  }

  const { parameters, matrix } = data;

  const trace = {
    z: matrix,
    x: parameters,
    y: parameters,
    type: 'heatmap',
    colorscale: 'RdBu',
    zmid: 0,
    zmin: -1,
    zmax: 1,
    hoverongaps: false,
    hovertemplate:
      '<b>%{y}</b> vs <b>%{x}</b><br>' +
      'Correlation: %{z:.3f}<br>' +
      '<extra></extra>',
    colorbar: {
      title: 'Correlation',
      titleside: 'right'
    }
  };

  const layout = {
    title: 'Parameter Correlation Matrix',
    xaxis: {
      title: 'Parameter',
      tickangle: -45,
      gridcolor: '#2c3e50'
    },
    yaxis: {
      title: 'Parameter',
      gridcolor: '#2c3e50'
    },
    plot_bgcolor: '#132f4c',
    paper_bgcolor: '#132f4c',
    font: { color: '#ffffff' },
    height: 600,
    margin: { t: 50, b: 150, l: 150, r: 100 }
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false
  };

  return (
    <Plot
      data={[trace]}
      layout={layout}
      config={config}
      style={{ width: '100%' }}
      useResizeHandler={true}
    />
  );
}

export default CorrelationHeatmap;
