/**
 * Slate Viewer - Main JavaScript Module
 * Handles WebSocket communication and UI interactions for the Slate ML visualization tool
 */

class SlateViewer {
  constructor() {
    this.ws = null;
    this.reconnectDelay = 1000;
    this.maxReconnectDelay = 30000;
    this.wsPort = 8000;
    
    this.cumulativeScore = 0;
    this.checkpointRetryCount = 0;
    this.maxRetries = 5;
    this.checkpointsReceived = false;
    this.retryTimeout = null;
    
    // Playback functionality
    this.isPlaybackMode = false;
    this.currentPlaybackRun = null;
    this.currentFrameCursor = 0;
    this.isPlaybackPaused = true;
    this.isAwaitingFrame = false;
    this.shouldPauseAfterFrame = false;
    
    this.connect();
    this.scheduleRetry();
  }

  /**
   * Establish native WebSocket connection with auto-reconnect logic
   */
  connect() {
    const wsUrl = `ws://${window.location.hostname}:${this.wsPort}/ws/ui`;
    console.log(`Connecting to Slate Web Server at ${wsUrl}`);
    this.updateConnectionStatus('connecting', 'Connecting...');
    
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log("Socket connected, requesting checkpoints and run history");
      this.updateConnectionStatus('connected', 'Connected');
      this.reconnectDelay = 1000; // Reset backoff
      this.checkpointsReceived = false;
      this.checkpointRetryCount = 0;
      this.requestCheckpoints();
      this.requestRunHistory();
    };

    this.ws.onclose = () => {
      console.log("Socket disconnected");
      this.updateConnectionStatus('disconnected', 'Disconnected');
      this.checkpointsReceived = false;
      
      // Auto-reconnect with exponential backoff
      setTimeout(() => this.connect(), this.reconnectDelay);
      this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, this.maxReconnectDelay);
    };

    this.ws.onerror = (error) => {
      console.error("WebSocket Error:", error);
      // onclose will trigger immediately after this to handle reconnect
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const msgType = data.type;

      switch (msgType) {
        case "checkpoints_update":
          this.handleCheckpointsUpdate(data);
          break;
        case "frame_update":
          this.handleFrameUpdate(data);
          break;
        case "run_history_update":
          this.handleRunHistoryUpdate(data);
          break;
        case "playback:loaded":
          this.handlePlaybackLoaded(data);
          break;
        case "playback:frame":
          this.handlePlaybackFrame(data);
          break;
        case "playback:eos":
          this.handlePlaybackEOS(data);
          break;
        case "playback:error":
          this.handlePlaybackError(data);
          break;
        case "playback:seek:ok":
          this.handlePlaybackSeekOk(data);
          break;
        case "playback:save:ready":
          this.handlePlaybackSaveReady(data);
          break;
        default:
          console.warn("Unhandled message type:", msgType);
      }
    };
  }

  /**
   * Helper method to serialize and send payloads over the WebSocket
   */
  _send(msgType, payload = {}) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: msgType, ...payload }));
    } else {
      console.warn(`Cannot send message of type '${msgType}', socket is not open.`);
    }
  }

  /**
   * Request checkpoints from the ML client
   */
  requestCheckpoints() {
    console.log("Requesting checkpoints");
    this._send("send_checkpoints");
  }

  /**
   * Request run history from the server
   */
  requestRunHistory() {
    console.log("Requesting run history");
    this._send("get_run_history");
  }

  /**
   * Handle checkpoint list updates from the server
   */
  handleCheckpointsUpdate(msg) {
    console.log("Received checkpoints:", msg.payload.checkpoints);
    this.checkpointsReceived = true;
    
    if (this.retryTimeout) {
      clearTimeout(this.retryTimeout);
      this.retryTimeout = null;
    }
    
    const checkpoints = msg.payload.checkpoints;
    const selectElement = document.getElementById("ckpt_select");
    
    selectElement.classList.add('loading');
    
    setTimeout(() => {
      selectElement.innerHTML = "";
      
      if (checkpoints.length === 0) {
        this.addOption(selectElement, "No checkpoints available", true);
      } else {
        checkpoints.forEach((checkpoint) => {
          this.addOption(selectElement, checkpoint, false, checkpoint);
        });
      }
      
      selectElement.classList.remove('loading');
    }, 300);
    
    this.checkpointRetryCount = 0;
  }

  /**
   * Handle frame updates from the ML client (live mode only)
   */
  handleFrameUpdate(msg) {
    if (this.isPlaybackMode) return;

    const payload = msg.payload;
    const frameElement = document.getElementById("env_frame");
    
    frameElement.style.opacity = '0.7';
    frameElement.src = "data:image/jpeg;base64," + payload.frame;
    
    frameElement.onload = () => {
      frameElement.style.opacity = '1';
    };

    this.updateInfoDisplay(payload);
    this.syncCheckpointDropdown(payload.checkpoint);

    this.cumulativeScore += payload.reward;
    const scoreElement = document.getElementById("score");
    const newScore = Math.round(this.cumulativeScore);
    
    if (scoreElement.innerText !== newScore.toString()) {
      scoreElement.style.transform = 'scale(1.1)';
      scoreElement.style.color = '#28a745';
      scoreElement.innerText = newScore;
      
      setTimeout(() => {
        scoreElement.style.transform = 'scale(1)';
        scoreElement.style.color = '#28a745';
      }, 200);
    }
  }

  /**
   * Handle run history updates from the server
   */
  handleRunHistoryUpdate(msg) {
    console.log("Received run history:", msg.run_history);
    this.updateHistoryList(msg.run_history);
  }

  /**
   * Handle playback loaded event from the server
   */
  handlePlaybackLoaded(msg) {
    console.log("Playback loaded:", msg.payload);
    this.currentPlaybackRun = msg.payload;
    this.currentFrameCursor = 0;
    this.isPlaybackPaused = true;
    this.isAwaitingFrame = false;
    this.enterPlaybackMode();
  }

  /**
   * Handle playback frame event from the server
   */
  handlePlaybackFrame(msg) {
    if (!this.isPlaybackMode || !this.currentPlaybackRun) return;

    const frameData = msg.frame_data;
    const cursor = msg.cursor;

    const frameElement = document.getElementById("env_frame");
    frameElement.style.opacity = '0.7';
    
    let frameImage, reward, qValues, action, checkpoint;
    
    // 1. Robust Parsing: Handle potential stringified JSON from the database
    let parsedData = frameData;
    if (typeof frameData === 'string') {
      try {
        parsedData = JSON.parse(frameData);
      } catch (e) {
        // If it's a string but not JSON, fallback to treating it as raw base64
        parsedData = { frame: frameData };
      }
    }

    // 2. Extract data considering the nested "metadata" structure from Python
    if (parsedData && parsedData.frame) {
      frameImage = parsedData.frame;
      
      // Map to the nested metadata dict if it exists, otherwise fallback to flat
      const meta = parsedData.metadata || parsedData;

      reward = meta.reward;
      qValues = meta.q_values;
      action = meta.action;
      checkpoint = meta.checkpoint || this.currentPlaybackRun.checkpoint;
    } else {
      console.warn("Unexpected frame_data structure. Received:", frameData);
      
      // CRITICAL: We must still ACK the frame even if it's broken, 
      // otherwise the server's asyncio loop will wait forever for this frame.
      this.isAwaitingFrame = false;
      this._send("playback:ack");
      return;
    }

    // Update Image
    frameElement.src = "data:image/jpeg;base64," + frameImage;
    frameElement.onload = () => {
      frameElement.style.opacity = '1';
    };

    // Update UI Panels
    if (qValues !== undefined && action !== undefined) {
      this.updateInfoDisplay({
        q_values: qValues,
        action: action,
        reward: reward || 0,
        checkpoint: checkpoint
      });
    }

    if (reward !== undefined) {
      const scoreElement = document.getElementById("score");
      scoreElement.innerText = Math.round(reward);
    }

    // Update cursor and acknowledge receipt to server
    this.currentFrameCursor = cursor;
    this.isAwaitingFrame = false;
    this._send("playback:ack");

    if (this.shouldPauseAfterFrame) {
      this.shouldPauseAfterFrame = false;
      setTimeout(() => {
        this.pausePlayback();
      }, 50);
    }
  }

  /**
   * Handle playback end of stream event
   */
  handlePlaybackEOS(msg) {
    console.log("Playback end of stream at cursor:", msg.cursor);
    this.isPlaybackPaused = true;
    this.isAwaitingFrame = false;
    
    document.getElementById('playback_play').classList.remove('active');
    document.getElementById('playback_pause').classList.add('active');
    document.getElementById('playback_stop').classList.remove('active');
  }

  /**
   * Handle playback error event
   */
  handlePlaybackError(msg) {
    console.error("Playback error:", msg.message);
    alert(`Playback Error: ${msg.message}`);
  }

  /**
   * Handle playback seek OK event
   */
  handlePlaybackSeekOk(msg) {
    console.log("Seek successful, cursor:", msg.cursor);
    this.currentFrameCursor = msg.cursor;
  }

  /**
   * Handle playback save ready event
   */
  handlePlaybackSaveReady(msg) {
    console.log("Playback save ready:", msg);
    const downloadUrl = msg.download_url;
    if (downloadUrl) {
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = `${msg.run_id}.s4`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  }

  /**
   * Save the current playback run
   */
  savePlaybackRun() {
    if (!this.isPlaybackMode || !this.currentPlaybackRun) {
      console.warn("Cannot save: not in playback mode or no run selected");
      return;
    }
    console.log("Saving playback run:", this.currentPlaybackRun.id);
    this._send("playback:save");
  }

  /**
   * Update the history list UI
   */
  updateHistoryList(runHistory) {
    const historyList = document.getElementById("run_history_list");
    
    if (runHistory.length === 0) {
      historyList.innerHTML = '<div class="no-history">No recorded runs yet</div>';
      return;
    }

    historyList.innerHTML = runHistory.map((run, index) => {
      const timestamp = new Date(run.timestamp).toLocaleTimeString();
      return `
        <div class="history-item" onclick="slateViewer.startPlayback(${run.id})">
          <div class="history-item-header">
            <span class="history-item-id">Run ${run.id}</span>
            <span class="history-item-time">${timestamp}</span>
          </div>
          <div class="history-item-stats">
            <span>${run.total_steps} steps</span>
            <span class="history-item-reward">+${Math.round(run.total_reward)}</span>
          </div>
        </div>
      `;
    }).join('');
  }

  /**
   * Start playback of a specific run
   */
  startPlayback(runId) {
    console.log("Loading playback for run:", runId);
    this._send("playback:load", { run_id: runId });
  }

  /**
   * Enter playback mode
   */
  enterPlaybackMode() {
    this.isPlaybackMode = true;
    
    document.getElementById("playback_controls").style.display = "block";
    document.getElementById("playback_status").textContent = `Playing back Run ${this.currentPlaybackRun.id}`;
    
    const modeStatus = document.getElementById("mode_status");
    modeStatus.className = "mode-status playback-mode";
    modeStatus.querySelector(".mode-text").textContent = "Playback Mode";
    
    this.isPlaybackPaused = true;
    document.getElementById('playback_play').classList.add('active');
    document.getElementById('playback_pause').classList.remove('active');
    document.getElementById('playback_stop').classList.remove('active');
  }

  /**
   * Exit playback mode and return to live view
   */
  exitPlaybackMode() {
    this.isPlaybackMode = false;
    this.currentPlaybackRun = null;
    this.currentFrameCursor = 0;
    this.isPlaybackPaused = true;
    this.isAwaitingFrame = false;
    
    document.getElementById("playback_controls").style.display = "none";
    document.getElementById("playback_play").classList.remove("active");
    document.getElementById("playback_pause").classList.remove("active");
    document.getElementById("playback_stop").classList.remove("active");
    
    const modeStatus = document.getElementById("mode_status");
    modeStatus.className = "mode-status live-mode";
    modeStatus.querySelector(".mode-text").textContent = "Live Mode";
    
    this.cumulativeScore = 0;
    document.getElementById("score").innerText = "–";
  }

  /**
   * Resume playback (start streaming)
   */
  resumePlayback() {
    if (!this.isPlaybackMode || !this.currentPlaybackRun) return;

    this.isPlaybackPaused = false;
    this._send("playback:resume");
    
    document.getElementById('playback_play').classList.remove('active');
    document.getElementById('playback_pause').classList.add('active');
    document.getElementById('playback_stop').classList.remove('active');
  }

  /**
   * Pause playback
   */
  pausePlayback() {
    if (!this.isPlaybackMode) return;

    this.isPlaybackPaused = true;
    this._send("playback:pause");
    
    document.getElementById('playback_play').classList.add('active');
    document.getElementById('playback_pause').classList.remove('active');
    document.getElementById('playback_stop').classList.remove('active');
  }

  /**
   * Stop playback and seek to beginning
   */
  stopPlayback() {
    if (!this.isPlaybackMode || !this.currentPlaybackRun) return;

    this.isPlaybackPaused = true;
    this._send("playback:seek", { frame: 0 });
    
    document.getElementById('playback_play').classList.add('active');
    document.getElementById('playback_pause').classList.remove('active');
    document.getElementById('playback_stop').classList.add('active');
  }

  /**
   * Seek to a specific frame
   */
  seekToFrame(frameIndex) {
    if (!this.isPlaybackMode || !this.currentPlaybackRun) return;

    if (frameIndex < 0 || frameIndex >= this.currentPlaybackRun.total_steps) {
      console.warn("Seek frame out of range:", frameIndex);
      return;
    }
    this._send("playback:seek", { frame: frameIndex });
  }

  /**
   * Step forward one frame
   */
  stepForward() {
    if (!this.isPlaybackMode || !this.currentPlaybackRun) return;

    if (!this.isPlaybackPaused) this.pausePlayback();
    const nextFrame = Math.min(this.currentFrameCursor + 1, this.currentPlaybackRun.total_steps - 1);
    this.seekToFrameAndFetch(nextFrame);
  }

  /**
   * Step backward one frame
   */
  stepBackward() {
    if (!this.isPlaybackMode || !this.currentPlaybackRun) return;

    if (!this.isPlaybackPaused) this.pausePlayback();
    const prevFrame = Math.max(this.currentFrameCursor - 1, 0);
    this.seekToFrameAndFetch(prevFrame);
  }

  /**
   * Seek to a frame and fetch it (for stepping while paused)
   */
  seekToFrameAndFetch(frameIndex) {
    if (!this.isPlaybackMode || !this.currentPlaybackRun) return;

    if (frameIndex < 0 || frameIndex >= this.currentPlaybackRun.total_steps) {
      console.warn("Seek frame out of range:", frameIndex);
      return;
    }

    if (this.isPlaybackPaused) {
      this.shouldPauseAfterFrame = true;
      this._send("playback:seek", { frame: frameIndex });
      this._send("playback:resume");
    } else {
      this._send("playback:seek", { frame: frameIndex });
    }
  }

  /**
   * Update the info display with current environment data
   */
  updateInfoDisplay(payload) {
    const qvals = payload.q_values.map((v) => Number(v).toFixed(2));
    document.getElementById("info").innerHTML = `
      <div>Q-values: [${qvals.join(", ")}]</div>
      <div>Action: ${payload.action}</div>
      <div>Reward: +${payload.reward}</div>
      <div>Checkpoint: ${payload.checkpoint}</div>
    `;
  }

  /**
   * Sync the checkpoint dropdown with the currently running checkpoint
   */
  syncCheckpointDropdown(checkpoint) {
    const selectElement = document.getElementById("ckpt_select");
    if (checkpoint && selectElement.value !== checkpoint) {
      selectElement.value = checkpoint;
    }
  }

  /**
   * Add an option to a select element
   */
  addOption(selectElement, text, disabled = false, value = null) {
    const option = document.createElement("option");
    option.textContent = text;
    option.disabled = disabled;
    if (value) option.value = value;
    selectElement.appendChild(option);
  }

  /**
   * Schedule retry mechanism for checkpoint requests
   */
  scheduleRetry() {
    if (this.retryTimeout) {
      clearTimeout(this.retryTimeout);
    }
    
    if (this.checkpointRetryCount < this.maxRetries && !this.checkpointsReceived) {
      this.checkpointRetryCount++;
      console.log(`Scheduling retry ${this.checkpointRetryCount}/${this.maxRetries} in 3 seconds`);
      
      this.retryTimeout = setTimeout(() => {
        if (!this.checkpointsReceived) {
          console.log(`Retrying checkpoint request (${this.checkpointRetryCount}/${this.maxRetries})`);
          this.requestCheckpoints();
          this.scheduleRetry();
        }
      }, 3000);
    } else if (!this.checkpointsReceived) {
      console.log("Max retries reached for checkpoint request");
      const selectElement = document.getElementById("ckpt_select");
      selectElement.innerHTML = '<option disabled selected>No ML client connected</option>';
    }
  }

  /**
   * Send a command to the ML client
   */
  sendCommand(command) {
    if (this.isPlaybackMode) {
      console.log("Ignoring command in playback mode:", command);
      return;
    }
    
    if (command === "reset") {
      this.cumulativeScore = 0;
    }
    this._send(command);
  }

  /**
   * Handle checkpoint selection
   */
  onSelectCheckpoint(selectElement) {
    this._send("select_checkpoint", { checkpoint: selectElement.value });
  }

  /**
   * Update connection status indicator
   */
  updateConnectionStatus(status, text) {
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');
    
    if (statusDot && statusText) {
      statusDot.className = `status-dot ${status}`;
      statusText.textContent = text;
    }
  }
}

// Global functions for HTML onclick handlers
let slateViewer;

function sendCommand(command) {
  slateViewer.sendCommand(command);
}

function onSelectCheckpoint(element) {
  slateViewer.onSelectCheckpoint(element);
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
  slateViewer = new SlateViewer();
  
  // Add playback control event listeners
  document.getElementById('back_to_live').addEventListener('click', () => {
    slateViewer.exitPlaybackMode();
  });
  
  document.getElementById('playback_play').addEventListener('click', () => {
    slateViewer.resumePlayback();
  });
  
  document.getElementById('playback_pause').addEventListener('click', () => {
    slateViewer.pausePlayback();
  });
  
  document.getElementById('playback_stop').addEventListener('click', () => {
    slateViewer.stopPlayback();
  });

  document.getElementById('playback_step_forward').addEventListener('click', () => {
    slateViewer.stepForward();
  });

  document.getElementById('playback_step_back').addEventListener('click', () => {
    slateViewer.stepBackward();
  });

  document.getElementById('save_run').addEventListener('click', () => {
    slateViewer.savePlaybackRun();
  });
});