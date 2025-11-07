/**
 * WebSocket Manager for real-time updates
 */

class WebSocketManager {
  constructor() {
    this.socket = null;
    this.runId = null;
    this.retryCount = 0;
    this.maxRetries = 5;
    this.retryTimeout = null;
    this.callbacks = {
      generation_complete: [],
      trial_complete: [],
      new_best: [],
      status_change: [],
      optimization_complete: [],
      error: [],
      initial_state: [],
      initialization_progress: []
    };
  }

  connect(runId, onConnect) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      console.warn('WebSocket already connected');
      return;
    }

    this.runId = runId;
    this.retryCount = 0;
    this._attemptConnect(onConnect);
  }

  _attemptConnect(onConnect) {
    const wsUrl = `ws://localhost:8000/ws/${this.runId}`;

    console.log(`Connecting to WebSocket: ${wsUrl} (Attempt ${this.retryCount + 1}/${this.maxRetries + 1})`);

    // Use raw WebSocket (not Socket.IO)
    this.socket = new WebSocket(wsUrl);

    this.socket.onopen = () => {
      console.log('✅ WebSocket connected successfully');
      this.retryCount = 0; // Reset retry count on success
      if (this.retryTimeout) {
        clearTimeout(this.retryTimeout);
        this.retryTimeout = null;
      }
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
    };

    this.socket.onclose = (event) => {
      console.log('WebSocket disconnected', event.code, event.reason);

      // Attempt retry if we haven't exceeded max retries
      if (this.retryCount < this.maxRetries) {
        // Exponential backoff: 2s, 4s, 8s, 16s, 32s
        const delay = Math.pow(2, this.retryCount + 1) * 1000;
        this.retryCount++;

        console.log(`⏳ Retrying WebSocket connection in ${delay/1000}s... (${this.retryCount}/${this.maxRetries})`);

        this.retryTimeout = setTimeout(() => {
          this._attemptConnect(onConnect);
        }, delay);
      } else {
        console.error('❌ WebSocket connection failed after maximum retries');
        this.trigger('error', {
          message: 'WebSocket connection failed after maximum retries. Backend may be initializing SITL instances.'
        });
      }
    };
  }

  disconnect() {
    // Clear any pending retry timeout
    if (this.retryTimeout) {
      clearTimeout(this.retryTimeout);
      this.retryTimeout = null;
    }

    if (this.socket) {
      this.socket.close();
      this.socket = null;
      this.runId = null;
      this.retryCount = 0;
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
