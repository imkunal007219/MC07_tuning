/**
 * Parameter Panel
 * Displays current best parameters
 */

import React, { useState } from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  Box,
  Typography,
  Chip,
  IconButton,
  Collapse,
  Table,
  TableBody,
  TableCell,
  TableRow
} from '@mui/material';
import { ExpandMore, ExpandLess } from '@mui/icons-material';

function ParameterPanel({ parameters }) {
  const [expanded, setExpanded] = useState(true);

  if (!parameters || Object.keys(parameters).length === 0) {
    return (
      <Card>
        <CardHeader title="🎯 Best Parameters" />
        <CardContent>
          <Typography variant="body2" color="text.secondary">
            No parameters yet...
          </Typography>
        </CardContent>
      </Card>
    );
  }

  // Group parameters by category
  const groupedParams = {};
  Object.keys(parameters).forEach(key => {
    const category = key.split('_')[0] + '_' + key.split('_')[1]; // e.g., "ATC_RAT", "ATC_ANG"
    if (!groupedParams[category]) {
      groupedParams[category] = [];
    }
    groupedParams[category].push({ key, value: parameters[key] });
  });

  return (
    <Card>
      <CardHeader
        title="🎯 Best Parameters"
        action={
          <IconButton onClick={() => setExpanded(!expanded)}>
            {expanded ? <ExpandLess /> : <ExpandMore />}
          </IconButton>
        }
      />
      <CardContent>
        <Collapse in={expanded}>
          {Object.entries(groupedParams).map(([category, params]) => (
            <Box key={category} sx={{ mb: 3 }}>
              <Typography variant="subtitle2" color="primary" gutterBottom>
                {category.replace('_', ' ')}
              </Typography>
              <Table size="small">
                <TableBody>
                  {params.map(({ key, value }) => (
                    <TableRow key={key}>
                      <TableCell sx={{ borderBottom: 'none', py: 0.5 }}>
                        <Typography variant="body2" color="text.secondary">
                          {key.split('_').pop()}
                        </Typography>
                      </TableCell>
                      <TableCell align="right" sx={{ borderBottom: 'none', py: 0.5 }}>
                        <Chip
                          label={typeof value === 'number' ? value.toFixed(4) : value}
                          size="small"
                          color="primary"
                          variant="outlined"
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Box>
          ))}
        </Collapse>

        {/* Summary */}
        <Box sx={{ mt: 2, p: 2, bgcolor: 'success.dark', borderRadius: 1, opacity: 0.8 }}>
          <Typography variant="body2" color="text.secondary">
            Total Parameters
          </Typography>
          <Typography variant="h5" color="success.light">
            {Object.keys(parameters).length}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
}

export default ParameterPanel;
