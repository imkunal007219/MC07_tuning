/**
 * 3D Trajectory Viewer
 * Visualize drone flight path in 3D space using Three.js
 */

import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Line } from '@react-three/drei';
import { Box, Typography } from '@mui/material';

function Drone({ position, attitude }) {
  const meshRef = useRef();

  useFrame(() => {
    if (meshRef.current && attitude) {
      meshRef.current.rotation.x = (attitude.pitch || 0) * Math.PI / 180;
      meshRef.current.rotation.y = (attitude.yaw || 0) * Math.PI / 180;
      meshRef.current.rotation.z = (attitude.roll || 0) * Math.PI / 180;
    }
  });

  return (
    <mesh ref={meshRef} position={position}>
      {/* Drone body (X-frame quadcopter) */}
      <group>
        {/* Center body */}
        <mesh>
          <boxGeometry args={[0.3, 0.05, 0.3]} />
          <meshStandardMaterial color="#ff0000" />
        </mesh>

        {/* Arms */}
        <mesh position={[0.2, 0, 0.2]} rotation={[0, Math.PI / 4, 0]}>
          <boxGeometry args={[0.5, 0.02, 0.05]} />
          <meshStandardMaterial color="#333333" />
        </mesh>
        <mesh position={[-0.2, 0, -0.2]} rotation={[0, Math.PI / 4, 0]}>
          <boxGeometry args={[0.5, 0.02, 0.05]} />
          <meshStandardMaterial color="#333333" />
        </mesh>
        <mesh position={[0.2, 0, -0.2]} rotation={[0, -Math.PI / 4, 0]}>
          <boxGeometry args={[0.5, 0.02, 0.05]} />
          <meshStandardMaterial color="#333333" />
        </mesh>
        <mesh position={[-0.2, 0, 0.2]} rotation={[0, -Math.PI / 4, 0]}>
          <boxGeometry args={[0.5, 0.02, 0.05]} />
          <meshStandardMaterial color="#333333" />
        </mesh>

        {/* Propellers */}
        {[
          [0.35, 0.02, 0.35],
          [-0.35, 0.02, -0.35],
          [0.35, 0.02, -0.35],
          [-0.35, 0.02, 0.35]
        ].map((pos, idx) => (
          <mesh key={idx} position={pos}>
            <cylinderGeometry args={[0.15, 0.15, 0.01, 8]} />
            <meshStandardMaterial color="#00ff00" opacity={0.5} transparent />
          </mesh>
        ))}
      </group>
    </mesh>
  );
}

function TrajectoryPath({ points }) {
  if (!points || points.length < 2) return null;

  return (
    <Line
      points={points}
      color="lime"
      lineWidth={2}
    />
  );
}

function TrajectoryViewer3D({ telemetryData }) {
  // Extract position points
  const trajectoryPoints = useMemo(() => {
    if (!telemetryData || !telemetryData.position) return [];

    const { x, y, z } = telemetryData.position;
    const points = [];

    for (let i = 0; i < x.length; i++) {
      // Convert from NED to ENU (Three.js convention: Y is up)
      points.push([x[i], -z[i], -y[i]]);
    }

    return points;
  }, [telemetryData]);

  // Current drone position and attitude
  const currentPosition = useMemo(() => {
    if (trajectoryPoints.length === 0) return [0, 0, 0];
    return trajectoryPoints[trajectoryPoints.length - 1];
  }, [trajectoryPoints]);

  const currentAttitude = useMemo(() => {
    if (!telemetryData || !telemetryData.attitude) return { roll: 0, pitch: 0, yaw: 0 };

    const { roll, pitch, yaw } = telemetryData.attitude;
    const lastIdx = roll.length - 1;

    return {
      roll: roll[lastIdx] || 0,
      pitch: pitch[lastIdx] || 0,
      yaw: yaw[lastIdx] || 0
    };
  }, [telemetryData]);

  if (!telemetryData) {
    return (
      <Box sx={{ textAlign: 'center', py: 8 }}>
        <Typography variant="body2" color="text.secondary">
          No trajectory data available
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', height: 500 }}>
      <Canvas camera={{ position: [20, 20, 20], fov: 50 }}>
        {/* Lighting */}
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={0.8} />
        <pointLight position={[-10, -10, -10]} intensity={0.3} />

        {/* Grid floor */}
        <Grid args={[50, 50]} cellColor="#6f6f6f" sectionColor="#9d4b4b" />

        {/* Trajectory path */}
        <TrajectoryPath points={trajectoryPoints} />

        {/* Drone */}
        <Drone position={currentPosition} attitude={currentAttitude} />

        {/* Waypoint markers (start and end) */}
        {trajectoryPoints.length > 0 && (
          <>
            <mesh position={trajectoryPoints[0]}>
              <sphereGeometry args={[0.3, 16, 16]} />
              <meshStandardMaterial color="#00ff00" />
            </mesh>
            <mesh position={trajectoryPoints[trajectoryPoints.length - 1]}>
              <sphereGeometry args={[0.3, 16, 16]} />
              <meshStandardMaterial color="#ff0000" />
            </mesh>
          </>
        )}

        {/* Controls */}
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          maxPolarAngle={Math.PI / 2}
        />
      </Canvas>

      {/* Legend */}
      <Box sx={{ mt: 1, display: 'flex', gap: 2, justifyContent: 'center' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Box sx={{ width: 16, height: 16, bgcolor: '#00ff00', borderRadius: '50%' }} />
          <Typography variant="caption">Start</Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Box sx={{ width: 16, height: 16, bgcolor: '#ff0000', borderRadius: '50%' }} />
          <Typography variant="caption">End</Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Box sx={{ width: 16, height: 3, bgcolor: 'lime' }} />
          <Typography variant="caption">Path</Typography>
        </Box>
      </Box>
    </Box>
  );
}

export default TrajectoryViewer3D;
