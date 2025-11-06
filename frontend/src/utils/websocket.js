/**
 * WebSocket Manager for real-time updates
 */

import { io } from 'socket.io-client';

const WS_BASE_URL = process.env.REACT_APP_WS_URL || 'http://localhost:8000';

class WebSocketManager {
  constructor() {
    this.socket = null;
    this.runId = null;
    this.callbacks = {
      generation_complete: [],
      trial_complete: [],
      new_best: [],
      status_change: [],
      optimization_complete: [],
      error: [],
      initial_state: []
    };
  }

  connect(runId, onConnect) {
    if (this.socket && this.socket.connected) {
      console.warn('WebSocket already connected');
      return;
    }

    this.runId = runId;
    const wsUrl = `${WS_BASE_URL}/ws/${runId}`;

    console.log(`Connecting to WebSocket: ${wsUrl}`);

    this.socket = io(WS_BASE_URL, {
      path: `/ws/${runId}`,
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5
    });

    // Use raw WebSocket instead of Socket.IO for FastAPI compatibility
    this.socket = new WebSocket(wsUrl.replace('http', 'ws'));

    this.socket.onopen = () => {
      console.log('✅ WebSocket connected');
      if (onConnect) onConnect();
    };

    this.socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        this.handleMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.trigger('error', { message: 'WebSocket connection error' });
    };

    this.socket.onclose = () => {
      console.log('WebSocket disconnected');
    };
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
      this.runId = null;
      console.log('WebSocket disconnected');
    }
  }

  handleMessage(message) {
    const { type, ...data } = message;

    console.log('WebSocket message:', type, data);

    // Trigger registered callbacks for this message type
    if (this.callbacks[type]) {
      this.callbacks[type].forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in ${type} callback:`, error);
        }
      });
    }
  }

  on(eventType, callback) {
    if (!this.callbacks[eventType]) {
      this.callbacks[eventType] = [];
    }
    this.callbacks[eventType].push(callback);

    // Return unsubscribe function
    return () => {
      this.callbacks[eventType] = this.callbacks[eventType].filter(cb => cb !== callback);
    };
  }

  trigger(eventType, data) {
    if (this.callbacks[eventType]) {
      this.callbacks[eventType].forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in ${eventType} callback:`, error);
        }
      });
    }
  }

  send(message) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(typeof message === 'string' ? message : JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected');
    }
  }

  isConnected() {
    return this.socket && this.socket.readyState === WebSocket.OPEN;
  }
}

// Singleton instance
const wsManager = new WebSocketManager();

export default wsManager;
