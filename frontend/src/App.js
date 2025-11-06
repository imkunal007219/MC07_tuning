/**
 * Main App Component
 * Drone PID Tuning Dashboard
 */

import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Provider } from 'react-redux';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Tabs,
  Tab,
  ThemeProvider,
  createTheme,
  CssBaseline
} from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';

import store from './store/store';
import { setSelectedTab } from './store/store';

// Pages
import DashboardPage from './pages/DashboardPage';
import TelemetryPage from './pages/TelemetryPage';
import AnalysisPage from './pages/AnalysisPage';
import ControlPage from './pages/ControlPage';

// Dark theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#2196f3',
    },
    secondary: {
      main: '#4caf50',
    },
    background: {
      default: '#0a1929',
      paper: '#132f4c',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 500,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
  },
});

function AppContent() {
  const dispatch = useDispatch();
  const selectedTab = useSelector(state => state.ui.selectedTab);

  const handleTabChange = (event, newValue) => {
    dispatch(setSelectedTab(newValue));
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      {/* Header */}
      <AppBar position="static" sx={{ backgroundImage: 'linear-gradient(to right, #0a1929, #1a2f4a)' }}>
        <Toolbar>
          <Typography variant="h5" component="div" sx={{ flexGrow: 1, fontWeight: 700 }}>
            🚁 Drone PID Tuning Dashboard
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Real-time Automated Optimization
          </Typography>
        </Toolbar>
        <Tabs
          value={selectedTab}
          onChange={handleTabChange}
          textColor="primary"
          indicatorColor="primary"
          sx={{ px: 2 }}
        >
          <Tab label="Dashboard" value="dashboard" />
          <Tab label="Telemetry" value="telemetry" />
          <Tab label="Analysis" value="analysis" />
          <Tab label="Control Systems" value="control" />
        </Tabs>
      </AppBar>

      {/* Main Content */}
      <Container maxWidth="xl" sx={{ mt: 3, mb: 3, flexGrow: 1 }}>
        {selectedTab === 'dashboard' && <DashboardPage />}
        {selectedTab === 'telemetry' && <TelemetryPage />}
        {selectedTab === 'analysis' && <AnalysisPage />}
        {selectedTab === 'control' && <ControlPage />}
      </Container>

      {/* Footer */}
      <Box
        component="footer"
        sx={{
          py: 2,
          px: 2,
          mt: 'auto',
          backgroundColor: (theme) => theme.palette.background.paper,
          borderTop: 1,
          borderColor: 'divider',
        }}
      >
        <Typography variant="body2" color="text.secondary" align="center">
          Drone PID Tuning System v1.0 | ArduPilot SITL Optimization
        </Typography>
      </Box>
    </Box>
  );
}

function App() {
  return (
    <Provider store={store}>
      <ThemeProvider theme={darkTheme}>
        <CssBaseline />
        <AppContent />
      </ThemeProvider>
    </Provider>
  );
}

export default App;
