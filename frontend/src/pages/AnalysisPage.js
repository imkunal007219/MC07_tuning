/**
 * Analysis Page
 * Parameter analysis and correlations
 */

import React, { useState, useEffect } from 'react';
import { useSelector } from 'react-redux';
import {
  Box,
  Grid,
  Card,
  CardHeader,
  CardContent,
  Typography
} from '@mui/material';

import CorrelationHeatmap from '../components/CorrelationHeatmap';
import { analysisAPI } from '../utils/api';

function AnalysisPage() {
  const currentRun = useSelector(state => state.optimization.currentRun);
  const [correlationData, setCorrelationData] = useState(null);

  useEffect(() => {
    if (currentRun) {
      loadAnalysis();
    }
  }, [currentRun]);

  const loadAnalysis = async () => {
    try {
      const response = await analysisAPI.getCorrelation(currentRun);
      setCorrelationData(response.data);
    } catch (error) {
      console.error('Failed to load analysis:', error);
    }
  };

  if (!currentRun) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Parameter Analysis
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Start an optimization run to view analysis.
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Parameter Analysis
      </Typography>

      <Grid container spacing={3}>
        {/* Correlation Heatmap */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardHeader title="📊 Parameter Correlation Matrix" />
            <CardContent>
              <CorrelationHeatmap data={correlationData} />
            </CardContent>
          </Card>
        </Grid>

        {/* Statistics */}
        <Grid item xs={12} lg={4}>
          <Card>
            <CardHeader title="📈 Statistics" />
            <CardContent>
              <Typography variant="body2" color="text.secondary">
                Statistical analysis will be displayed here
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default AnalysisPage;
