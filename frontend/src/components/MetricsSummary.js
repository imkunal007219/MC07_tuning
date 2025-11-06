/**
 * Metrics Summary
 * Shows key metrics in a card layout
 */

import React from 'react';
import { Card, CardHeader, CardContent, Grid, Box, Typography, LinearProgress } from '@mui/material';
import { TrendingUp, Speed, CheckCircle, Timer } from '@mui/icons-material';

function MetricsSummary({ bestFitness, currentGeneration, totalGenerations, completedTrials }) {
  const progress = (currentGeneration / totalGenerations) * 100;

  const metrics = [
    {
      label: 'Best Fitness',
      value: bestFitness ? bestFitness.toFixed(4) : '0.0000',
      icon: <TrendingUp sx={{ fontSize: 40 }} />,
      color: 'success.main'
    },
    {
      label: 'Generation',
      value: `${currentGeneration} / ${totalGenerations}`,
      icon: <Speed sx={{ fontSize: 40 }} />,
      color: 'primary.main'
    },
    {
      label: 'Completed Trials',
      value: completedTrials,
      icon: <CheckCircle sx={{ fontSize: 40 }} />,
      color: 'info.main'
    },
    {
      label: 'Progress',
      value: `${progress.toFixed(1)}%`,
      icon: <Timer sx={{ fontSize: 40 }} />,
      color: 'warning.main'
    }
  ];

  return (
    <Card>
      <CardHeader title="📊 Progress Metrics" />
      <CardContent>
        <Grid container spacing={2}>
          {metrics.map((metric, idx) => (
            <Grid item xs={6} key={idx}>
              <Box
                sx={{
                  p: 2,
                  border: 1,
                  borderColor: 'divider',
                  borderRadius: 1,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 2
                }}
              >
                <Box sx={{ color: metric.color }}>
                  {metric.icon}
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    {metric.label}
                  </Typography>
                  <Typography variant="h6" fontWeight="bold">
                    {metric.value}
                  </Typography>
                </Box>
              </Box>
            </Grid>
          ))}
        </Grid>

        {/* Progress Bar */}
        <Box sx={{ mt: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2" color="text.secondary">
              Overall Progress
            </Typography>
            <Typography variant="body2" fontWeight="bold">
              {progress.toFixed(1)}%
            </Typography>
          </Box>
          <LinearProgress variant="determinate" value={progress} sx={{ height: 8, borderRadius: 4 }} />
        </Box>
      </CardContent>
    </Card>
  );
}

export default MetricsSummary;
