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
    this.currentPlaybackRun = null;
    this.currentFrameCursor = 0;
    this.isPlaybackPaused = true;
    this.isAwaitingFrame = false;
    this.shouldPauseAfterFrame = false;
    
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

    this.socket.on("playback:loaded", (msg) => {
      this.handlePlaybackLoaded(msg);
    });

    this.socket.on("playback:frame", (msg) => {
      this.handlePlaybackFrame(msg);
    });

    this.socket.on("playback:eos", (msg) => {
      this.handlePlaybackEOS(msg);
    });

    this.socket.on("playback:error", (msg) => {
      this.handlePlaybackError(msg);
    });

    this.socket.on("playback:seek:ok", (msg) => {
      this.handlePlaybackSeekOk(msg);
    });

    this.socket.on("playback:save:ready", (msg) => {
      this.handlePlaybackSaveReady(msg);
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
   * Handle frame updates from the ML client (live mode only)
   * @param {Object} msg - Message containing frame data
   */
  handleFrameUpdate(msg) {
    if (this.isPlaybackMode) {
      return;
    }

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
   * Handle playback loaded event from the server
   * @param {Object} msg - Message containing run metadata
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
   * @param {Object} msg - Message containing frame data
   */
  handlePlaybackFrame(msg) {
    if (!this.isPlaybackMode || !this.currentPlaybackRun) {
      return;
    }

    const frameData = msg.frame_data;
    const cursor = msg.cursor;

    // Update frame display
    const frameElement = document.getElementById("env_frame");
    frameElement.style.opacity = '0.7';
    
    // Handle different frame_data structures
    let frameImage, reward, qValues, action, checkpoint;
    
    if (typeof frameData === 'string') {
      // If frame_data is just the base64 string
      frameImage = frameData;
    } else if (frameData && frameData.frame) {
      // If frame_data is an object with frame property
      frameImage = frameData.frame;
      reward = frameData.reward;
      qValues = frameData.q_values;
      action = frameData.action;
      checkpoint = frameData.checkpoint || this.currentPlaybackRun.checkpoint;
    } else {
      // Fallback: try to extract from metadata if available
      console.warn("Unexpected frame_data structure:", frameData);
      return;
    }

    frameElement.src = "data:image/jpeg;base64," + frameImage;
    frameElement.onload = () => {
      frameElement.style.opacity = '1';
    };

    // Update info display if we have the data
    if (qValues !== undefined && action !== undefined) {
      this.updateInfoDisplay({
        q_values: qValues,
        action: action,
        reward: reward || 0,
        checkpoint: checkpoint
      });
    }

    // Update score (for playback, show the frame's reward)
    if (reward !== undefined) {
      const scoreElement = document.getElementById("score");
      scoreElement.innerText = Math.round(reward);
    }

    // Update current cursor
    this.currentFrameCursor = cursor;

    // Send acknowledgment
    this.isAwaitingFrame = false;
    this.socket.emit("playback:ack", {});

    if (this.shouldPauseAfterFrame) {
      this.shouldPauseAfterFrame = false;
      setTimeout(() => {
        this.pausePlayback();
      }, 50);
    }
  }

  /**
   * Handle playback end of stream event
   * @param {Object} msg - Message containing cursor info
   */
  handlePlaybackEOS(msg) {
    console.log("Playback end of stream at cursor:", msg.cursor);
    this.isPlaybackPaused = true;
    this.isAwaitingFrame = false;
    
    // Update UI to show paused state
    document.getElementById('playback_play').classList.remove('active');
    document.getElementById('playback_pause').classList.add('active');
    document.getElementById('playback_stop').classList.remove('active');
  }

  /**
   * Handle playback error event
   * @param {Object} msg - Message containing error info
   */
  handlePlaybackError(msg) {
    console.error("Playback error:", msg.message);
    alert(`Playback Error: ${msg.message}`);
  }

  /**
   * Handle playback seek OK event
   * @param {Object} msg - Message containing cursor info
   */
  handlePlaybackSeekOk(msg) {
    console.log("Seek successful, cursor:", msg.cursor);
    this.currentFrameCursor = msg.cursor;
  }

  /**
   * Handle playback save ready event
   * @param {Object} msg - Message containing download URL
   */
  handlePlaybackSaveReady(msg) {
    console.log("Playback save ready:", msg);
    const downloadUrl = msg.download_url;
    if (downloadUrl) {
      // Trigger download
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
    this.socket.emit("playback:save", {});
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
   * @param {number} runId - ID of the run to playback
   */
  startPlayback(runId) {
    console.log("Loading playback for run:", runId);
    this.socket.emit("playback:load", { run_id: runId });
  }

  /**
   * Enter playback mode
   */
  enterPlaybackMode() {
    this.isPlaybackMode = true;
    
    // Show playback controls
    document.getElementById("playback_controls").style.display = "block";
    
    // Update playback status
    document.getElementById("playback_status").textContent = 
      `Playing back Run ${this.currentPlaybackRun.id}`;
    
    // Update mode indicator
    const modeStatus = document.getElementById("mode_status");
    modeStatus.className = "mode-status playback-mode";
    modeStatus.querySelector(".mode-text").textContent = "Playback Mode";
    
    // Start paused by default
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
    
    // Hide playback controls
    document.getElementById("playback_controls").style.display = "none";
    
    // Reset playback buttons
    document.getElementById("playback_play").classList.remove("active");
    document.getElementById("playback_pause").classList.remove("active");
    document.getElementById("playback_stop").classList.remove("active");
    
    // Reset mode indicator
    const modeStatus = document.getElementById("mode_status");
    modeStatus.className = "mode-status live-mode";
    modeStatus.querySelector(".mode-text").textContent = "Live Mode";
    
    // Reset score
    this.cumulativeScore = 0;
    document.getElementById("score").innerText = "–";
  }

  /**
   * Resume playback (start streaming)
   */
  resumePlayback() {
    if (!this.isPlaybackMode || !this.currentPlaybackRun) {
      return;
    }

    this.isPlaybackPaused = false;
    this.socket.emit("playback:resume", {});
    
    // Update UI
    document.getElementById('playback_play').classList.remove('active');
    document.getElementById('playback_pause').classList.add('active');
    document.getElementById('playback_stop').classList.remove('active');
  }

  /**
   * Pause playback
   */
  pausePlayback() {
    if (!this.isPlaybackMode) {
      return;
    }

    this.isPlaybackPaused = true;
    this.socket.emit("playback:pause", {});
    
    // Update UI
    document.getElementById('playback_play').classList.add('active');
    document.getElementById('playback_pause').classList.remove('active');
    document.getElementById('playback_stop').classList.remove('active');
  }

  /**
   * Stop playback and seek to beginning
   */
  stopPlayback() {
    if (!this.isPlaybackMode || !this.currentPlaybackRun) {
      return;
    }

    this.isPlaybackPaused = true;
    this.socket.emit("playback:seek", { frame: 0 });
    
    // Update UI
    document.getElementById('playback_play').classList.add('active');
    document.getElementById('playback_pause').classList.remove('active');
    document.getElementById('playback_stop').classList.add('active');
  }

  /**
   * Seek to a specific frame
   * @param {number} frameIndex - Frame index to seek to
   */
  seekToFrame(frameIndex) {
    if (!this.isPlaybackMode || !this.currentPlaybackRun) {
      return;
    }

    if (frameIndex < 0 || frameIndex >= this.currentPlaybackRun.total_steps) {
      console.warn("Seek frame out of range:", frameIndex);
      return;
    }

    this.socket.emit("playback:seek", { frame: frameIndex });
  }

  /**
   * Step forward one frame
   */
  stepForward() {
    if (!this.isPlaybackMode || !this.currentPlaybackRun) {
      return;
    }

    // Pause playback if playing
    if (!this.isPlaybackPaused) {
      this.pausePlayback();
    }

    // Step to next frame
    const nextFrame = Math.min(this.currentFrameCursor + 1, this.currentPlaybackRun.total_steps - 1);
    this.seekToFrameAndFetch(nextFrame);
  }

  /**
   * Step backward one frame
   */
  stepBackward() {
    if (!this.isPlaybackMode || !this.currentPlaybackRun) {
      return;
    }

    // Pause playback if playing
    if (!this.isPlaybackPaused) {
      this.pausePlayback();
    }

    // Step to previous frame
    const prevFrame = Math.max(this.currentFrameCursor - 1, 0);
    this.seekToFrameAndFetch(prevFrame);
  }

  /**
   * Seek to a frame and fetch it (for stepping while paused)
   * @param {number} frameIndex - Frame index to seek to
   */
  seekToFrameAndFetch(frameIndex) {
    if (!this.isPlaybackMode || !this.currentPlaybackRun) {
      return;
    }

    if (frameIndex < 0 || frameIndex >= this.currentPlaybackRun.total_steps) {
      console.warn("Seek frame out of range:", frameIndex);
      return;
    }

    // If paused, we need to briefly resume to get the frame, then pause again
    if (this.isPlaybackPaused) {
      // Set a flag to pause after receiving the next frame
      this.shouldPauseAfterFrame = true;
      
      // Seek to the frame
      this.socket.emit("playback:seek", { frame: frameIndex });
      
      // Resume briefly to trigger frame send
      this.socket.emit("playback:resume", {});
    } else {
      // If already playing, just seek
      this.socket.emit("playback:seek", { frame: frameIndex });
    }
  }

  /**
   * Update the info display with current environment data
   * @param {Object} payload - Frame update payload
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
