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
    
    this.initializeEventListeners();
    this.requestCheckpoints();
    this.scheduleRetry();
  }

  /**
   * Initialize all Socket.IO event listeners
   */
  initializeEventListeners() {
    this.socket.on("connect", () => {
      console.log("Socket connected, requesting checkpoints");
      this.checkpointsReceived = false;
      this.checkpointRetryCount = 0;
      this.requestCheckpoints();
    });

    this.socket.on("disconnect", () => {
      console.log("Socket disconnected");
      this.checkpointsReceived = false;
    });

    this.socket.on("checkpoints_update", (msg) => {
      this.handleCheckpointsUpdate(msg);
    });

    this.socket.on("frame_update", (msg) => {
      this.handleFrameUpdate(msg);
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
    selectElement.innerHTML = "";
    
    if (checkpoints.length === 0) {
      this.addOption(selectElement, "No checkpoints available", true);
    } else {
      checkpoints.forEach((checkpoint) => {
        this.addOption(selectElement, checkpoint, false, checkpoint);
      });
    }
    
    // Reset retry count on successful response
    this.checkpointRetryCount = 0;
  }

  /**
   * Handle frame updates from the ML client
   * @param {Object} msg - Message containing frame data
   */
  handleFrameUpdate(msg) {
    const payload = msg.payload;
    
    // Update frame image
    document.getElementById("env_frame").src = 
      "data:image/jpeg;base64," + payload.frame;

    // Update info display
    this.updateInfoDisplay(payload);

    // Keep dropdown in sync with currently running checkpoint
    this.syncCheckpointDropdown(payload.checkpoint);

    // Update cumulative score
    this.cumulativeScore += payload.reward;
    document.getElementById("score").innerText = Math.round(this.cumulativeScore);
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
});
