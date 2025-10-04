/**
 * Slate Viewer - Main JavaScript Module
 * Handles WebSocket communication and UI interactions for the Slate ML visualization tool
 */

class SlateViewer {
  constructor() {
    this.socket = io();
    this.cumulativeScore = 0;
    this.checkpointRetryCount = 0;
    this.maxRetries = 5;
    this.checkpointsReceived = false;
    this.retryTimeout = null;
    
    // Playback functionality
    this.isPlaybackMode = false;
    this.currentPlaybackData = null;
    this.playbackIndex = 0;
    this.playbackInterval = null;
    this.playbackSpeed = 100; // ms between frames
    
    this.initializeEventListeners();
    this.requestCheckpoints();
    this.requestRunHistory();
    this.scheduleRetry();
  }

  /**
   * Initialize all Socket.IO event listeners
   */
  initializeEventListeners() {
    this.socket.on("connect", () => {
      console.log("Socket connected, requesting checkpoints and run history");
      this.updateConnectionStatus('connected', 'Connected');
      this.checkpointsReceived = false;
      this.checkpointRetryCount = 0;
      this.requestCheckpoints();
      this.requestRunHistory();
    });

    this.socket.on("disconnect", () => {
      console.log("Socket disconnected");
      this.updateConnectionStatus('disconnected', 'Disconnected');
      this.checkpointsReceived = false;
    });

    this.socket.on("checkpoints_update", (msg) => {
      this.handleCheckpointsUpdate(msg);
    });

    this.socket.on("frame_update", (msg) => {
      this.handleFrameUpdate(msg);
    });

    this.socket.on("run_history_update", (msg) => {
      this.handleRunHistoryUpdate(msg);
    });

    this.socket.on("playback_data", (msg) => {
      this.handlePlaybackData(msg);
    });
  }

  /**
   * Request checkpoints from the ML client
   */
  requestCheckpoints() {
    console.log("Requesting checkpoints");
    this.socket.emit("send_checkpoints");
  }

  /**
   * Request run history from the server
   */
  requestRunHistory() {
    console.log("Requesting run history");
    this.socket.emit("get_run_history");
  }

  /**
   * Handle checkpoint list updates from the server
   * @param {Object} msg - Message containing checkpoint data
   */
  handleCheckpointsUpdate(msg) {
    console.log("Received checkpoints:", msg.payload.checkpoints);
    this.checkpointsReceived = true;
    
    // Clear any pending retry
    if (this.retryTimeout) {
      clearTimeout(this.retryTimeout);
      this.retryTimeout = null;
    }
    
    const checkpoints = msg.payload.checkpoints;
    const selectElement = document.getElementById("ckpt_select");
    
    // Add loading animation
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
    
    // Reset retry count on successful response
    this.checkpointRetryCount = 0;
  }

  /**
   * Handle frame updates from the ML client
   * @param {Object} msg - Message containing frame data
   */
  handleFrameUpdate(msg) {
    const payload = msg.payload;
    const frameElement = document.getElementById("env_frame");
    
    // Add subtle animation to frame updates
    frameElement.style.opacity = '0.7';
    
    // Update frame image
    frameElement.src = "data:image/jpeg;base64," + payload.frame;
    
    // Restore opacity after image loads
    frameElement.onload = () => {
      frameElement.style.opacity = '1';
    };

    // Update info display
    this.updateInfoDisplay(payload);

    // Keep dropdown in sync with currently running checkpoint
    this.syncCheckpointDropdown(payload.checkpoint);

    // Update cumulative score with animation
    this.cumulativeScore += payload.reward;
    const scoreElement = document.getElementById("score");
    const newScore = Math.round(this.cumulativeScore);
    
    // Animate score change
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
   * @param {Object} msg - Message containing run history data
   */
  handleRunHistoryUpdate(msg) {
    console.log("Received run history:", msg.run_history);
    this.updateHistoryList(msg.run_history);
  }

  /**
   * Handle playback data from the server
   * @param {Object} msg - Message containing playback data
   */
  handlePlaybackData(msg) {
    console.log("Received playback data:", msg.payload);
    this.currentPlaybackData = msg.payload;
    this.playbackIndex = 0;
    this.enterPlaybackMode();
  }

  /**
   * Update the history list UI
   * @param {Array} runHistory - Array of run data
   */
  updateHistoryList(runHistory) {
    const historyList = document.getElementById("run_history_list");
    
    if (runHistory.length === 0) {
      historyList.innerHTML = '<div class="no-history">No recorded runs yet</div>';
      return;
    }

    historyList.innerHTML = runHistory.map((run, index) => {
      const timestamp = new Date(run.timestamp).toLocaleTimeString();
      const duration = Math.round(run.duration);
      return `
        <div class="history-item" onclick="slateViewer.startPlayback(${run.id})">
          <div class="history-item-header">
            <span class="history-item-id">Run ${run.id}</span>
            <span class="history-item-time">${timestamp}</span>
          </div>
          <div class="history-item-stats">
            <span>${run.total_steps} steps</span>
            <span>${duration}s</span>
            <span class="history-item-reward">+${Math.round(run.total_reward)}</span>
          </div>
        </div>
      `;
    }).join('');
  }

  /**
   * Start playback of a specific run
   * @param {number} runId - ID of the run to playback
   */
  startPlayback(runId) {
    console.log("Starting playback of run:", runId);
    this.socket.emit("playback_run", { run_id: runId });
  }

  /**
   * Enter playback mode
   */
  enterPlaybackMode() {
    this.isPlaybackMode = true;
    this.playbackIndex = 0;
    
    // Show playback controls
    document.getElementById("playback_controls").style.display = "block";
    
    // Update playback status
    document.getElementById("playback_status").textContent = 
      `Playing back Run ${this.currentPlaybackData.id}`;
    
    // Start playback
    this.playPlayback();
  }

  /**
   * Exit playback mode and return to live view
   */
  exitPlaybackMode() {
    this.isPlaybackMode = false;
    this.currentPlaybackData = null;
    this.playbackIndex = 0;
    
    // Stop any ongoing playback
    if (this.playbackInterval) {
      clearInterval(this.playbackInterval);
      this.playbackInterval = null;
    }
    
    // Hide playback controls
    document.getElementById("playback_controls").style.display = "none";
    
    // Reset playback buttons
    document.getElementById("playback_play").classList.remove("active");
    document.getElementById("playback_pause").classList.remove("active");
    document.getElementById("playback_stop").classList.remove("active");
  }

  /**
   * Play the current playback data
   */
  playPlayback() {
    if (!this.currentPlaybackData || this.playbackIndex >= this.currentPlaybackData.frames.length) {
      this.stopPlayback();
      return;
    }

    // Update frame
    const frameElement = document.getElementById("env_frame");
    frameElement.src = "data:image/jpeg;base64," + this.currentPlaybackData.frames[this.playbackIndex];
    
    // Update info display
    const metadata = this.currentPlaybackData.metadata[this.playbackIndex];
    this.updateInfoDisplay({
      q_values: metadata.q_values,
      reward: metadata.reward,
      checkpoint: this.currentPlaybackData.checkpoint
    });

    // Update score
    const scoreElement = document.getElementById("score");
    scoreElement.innerText = Math.round(metadata.reward);

    this.playbackIndex++;
    
    // Continue playback
    this.playbackInterval = setTimeout(() => {
      this.playPlayback();
    }, this.playbackSpeed);
  }

  /**
   * Pause playback
   */
  pausePlayback() {
    if (this.playbackInterval) {
      clearInterval(this.playbackInterval);
      this.playbackInterval = null;
    }
  }

  /**
   * Stop playback
   */
  stopPlayback() {
    this.pausePlayback();
    this.playbackIndex = 0;
    document.getElementById("playback_play").classList.remove("active");
    document.getElementById("playback_pause").classList.remove("active");
    document.getElementById("playback_stop").classList.add("active");
  }

  /**
   * Update the info display with current environment data
   * @param {Object} payload - Frame update payload
   */
  updateInfoDisplay(payload) {
    const qvals = payload.q_values.map((v) => Number(v).toFixed(2));
    document.getElementById("info").innerHTML = `
      <div>Q-values: [${qvals.join(", ")}]</div>
      <div>Action: [auto]</div>
      <div>Reward: +${payload.reward}</div>
      <div>Checkpoint: ${payload.checkpoint}</div>
    `;
  }

  /**
   * Sync the checkpoint dropdown with the currently running checkpoint
   * @param {string} checkpoint - Current checkpoint name
   */
  syncCheckpointDropdown(checkpoint) {
    const selectElement = document.getElementById("ckpt_select");
    if (checkpoint && selectElement.value !== checkpoint) {
      selectElement.value = checkpoint;
    }
  }

  /**
   * Add an option to a select element
   * @param {HTMLElement} selectElement - The select element
   * @param {string} text - Option text
   * @param {boolean} disabled - Whether the option is disabled
   * @param {string} value - Option value (optional)
   */
  addOption(selectElement, text, disabled = false, value = null) {
    const option = document.createElement("option");
    option.textContent = text;
    option.disabled = disabled;
    if (value) {
      option.value = value;
    }
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
          this.scheduleRetry(); // Schedule next retry
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
   * @param {string} command - Command to send
   */
  sendCommand(command) {
    // Don't send commands to ML client if in playback mode
    if (this.isPlaybackMode) {
      console.log("Ignoring command in playback mode:", command);
      return;
    }
    
    if (command === "reset") {
      this.cumulativeScore = 0;
    }
    this.socket.emit(command);
  }

  /**
   * Handle checkpoint selection
   * @param {HTMLElement} selectElement - The select element that was changed
   */
  onSelectCheckpoint(selectElement) {
    this.socket.emit("select_checkpoint", { checkpoint: selectElement.value });
  }

  /**
   * Update connection status indicator
   * @param {string} status - Connection status ('connected', 'disconnected', 'connecting')
   * @param {string} text - Status text to display
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
    slateViewer.playPlayback();
    document.getElementById('playback_play').classList.add('active');
    document.getElementById('playback_pause').classList.remove('active');
    document.getElementById('playback_stop').classList.remove('active');
  });
  
  document.getElementById('playback_pause').addEventListener('click', () => {
    slateViewer.pausePlayback();
    document.getElementById('playback_play').classList.remove('active');
    document.getElementById('playback_pause').classList.add('active');
    document.getElementById('playback_stop').classList.remove('active');
  });
  
  document.getElementById('playback_stop').addEventListener('click', () => {
    slateViewer.stopPlayback();
  });
});
