/**
 * SITL Status
 * Shows status of parallel SITL instances
 */

import React from 'react';
import { Card, CardHeader, CardContent, Box, Typography, Grid, Chip } from '@mui/material';
import { CheckCircle, HourglassEmpty, Error } from '@mui/icons-material';

function SITLStatus({ instances = 4 }) {
  // Mock instance status (in real implementation, get from backend)
  const instanceStatuses = Array.from({ length: instances }, (_, i) => ({
    id: i,
    status: i < 3 ? 'active' : 'idle',
    currentTrial: i < 3 ? Math.floor(Math.random() * 1000) : null
  }));

  const getStatusIcon = (status) => {
    switch (status) {
      case 'active':
        return <CheckCircle sx={{ color: 'success.main' }} />;
      case 'idle':
        return <HourglassEmpty sx={{ color: 'text.secondary' }} />;
      case 'error':
        return <Error sx={{ color: 'error.main' }} />;
      default:
        return null;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'active':
        return 'success';
      case 'idle':
        return 'default';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  const activeCount = instanceStatuses.filter(i => i.status === 'active').length;

  return (
    <Card>
      <CardHeader
        title="⚙️ SITL Instances"
        subheader={`${activeCount} / ${instances} Active`}
      />
      <CardContent>
        <Grid container spacing={1}>
          {instanceStatuses.map((instance) => (
            <Grid item xs={6} sm={4} md={3} key={instance.id}>
              <Box
                sx={{
                  p: 1.5,
                  border: 1,
                  borderColor: instance.status === 'active' ? 'success.main' : 'divider',
                  borderRadius: 1,
                  textAlign: 'center',
                  bgcolor: instance.status === 'active' ? 'success.dark' : 'background.paper',
                  opacity: instance.status === 'active' ? 1 : 0.6
                }}
              >
                <Box sx={{ display: 'flex', justifyContent: 'center', mb: 1 }}>
                  {getStatusIcon(instance.status)}
                </Box>
                <Typography variant="caption" display="block">
                  Instance {instance.id}
                </Typography>
                <Chip
                  label={instance.status}
                  size="small"
                  color={getStatusColor(instance.status)}
                  sx={{ mt: 0.5, fontSize: '0.7rem', height: 20 }}
                />
                {instance.currentTrial && (
                  <Typography variant="caption" display="block" color="text.secondary" sx={{ mt: 0.5 }}>
                    Trial #{instance.currentTrial}
                  </Typography>
                )}
              </Box>
            </Grid>
          ))}
        </Grid>

        {/* Summary */}
        <Box sx={{ mt: 2, p: 1.5, bgcolor: 'background.default', borderRadius: 1 }}>
          <Grid container spacing={2}>
            <Grid item xs={4}>
              <Typography variant="caption" color="text.secondary">
                Active
              </Typography>
              <Typography variant="h6" color="success.main">
                {activeCount}
              </Typography>
            </Grid>
            <Grid item xs={4}>
              <Typography variant="caption" color="text.secondary">
                Idle
              </Typography>
              <Typography variant="h6">
                {instanceStatuses.filter(i => i.status === 'idle').length}
              </Typography>
            </Grid>
            <Grid item xs={4}>
              <Typography variant="caption" color="text.secondary">
                Total
              </Typography>
              <Typography variant="h6">
                {instances}
              </Typography>
            </Grid>
          </Grid>
        </Box>
      </CardContent>
    </Card>
  );
}

export default SITLStatus;
