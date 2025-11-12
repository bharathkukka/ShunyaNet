// Application state and Data
let selectedModel = null;
let currentStream = null;
let isWebcamActive = false;
let emotionLines = null; // Will store the emotion lines from JSON

const projects = ["Emotion Recognition System", "Weapon Detection System", "Plant Disease Prediction", "Fall Detection System", "Plant Pesticide Recommendation", "Movie Recommendation System"];

// Load emotion lines from JSON file
async function loadEmotionLines() {
  try {
    const response = await fetch('./Lines.json');
    if (response.ok) {
      emotionLines = await response.json();
      console.log('Emotion lines loaded successfully');
    } else {
      console.error('Failed to load emotion lines');
    }
  } catch (error) {
    console.error('Error loading emotion lines:', error);
  }
}

// Project-specific predictions and contextual messages
const projectData = {
  "Emotion Recognition System": {
    predictions: [
      {"label": "happiness", "confidence": 0.94, "displayName": "Happy"},
      {"label": "sadness", "confidence": 0.89, "displayName": "Sad"},
      {"label": "anger", "confidence": 0.82, "displayName": "Angry"},
      {"label": " surprise", "confidence": 0.91, "displayName": "Surprised"},
      {"label": "neutral", "confidence": 0.76, "displayName": "Neutral"},
      {"label": "fear", "confidence": 0.88, "displayName": "Fearful"},
      {"label": "disgust", "confidence": 0.83, "displayName": "Disgusted"},
      {"label": "contempt", "confidence": 0.87, "displayName": "Contemptuous"}
    ],
    useEmotionLines: true // Flag to indicate this project uses the JSON file
  },
  "Weapon Detection System": {
    predictions: [
      {"label": "Weapon Detected", "confidence": 0.92},
      {"label": "No Weapon", "confidence": 0.96},
      {"label": "Suspicious Object", "confidence": 0.78}
    ],
    messages: {
      "Weapon Detected": "⚠️ ALERT: Potential weapon detected. Please contact security immediately and maintain safe distance.",
      "No Weapon": "✅ All clear! No weapons detected. Environment appears safe.",
      "Suspicious Object": "⚠️ Suspicious object detected. Please verify and proceed with caution."
    }
  },
  "Plant Disease Prediction": {
    predictions: [
      {"label": "Healthy", "confidence": 0.94},
      {"label": "Bacterial Blight", "confidence": 0.87},
      {"label": "Leaf Spot", "confidence": 0.82},
      {"label": "Powdery Mildew", "confidence": 0.89},
      {"label": "Root Rot", "confidence": 0.85}
    ],
    messages: {
      "Healthy": "🌿 Great news! Your plant looks healthy and vibrant. Keep up the excellent care!",
      "Bacterial Blight": "🦠 Bacterial infection detected. Remove affected leaves and apply copper-based fungicide. Improve air circulation.",
      "Leaf Spot": "🍃 Leaf spot disease identified. Remove infected leaves, avoid overhead watering, and apply appropriate fungicide.",
      "Powdery Mildew": "⚪ Powdery mildew detected. Increase air circulation, reduce humidity, and treat with neem oil or fungicide.",
      "Root Rot": "🌱 Root rot suspected. Check drainage, reduce watering frequency, and consider repotting with fresh soil."
    }
  },
  "Fall Detection System": {
    predictions: [
      {"label": "Normal Activity", "confidence": 0.95},
      {"label": "Fall Detected", "confidence": 0.88},
      {"label": "Sitting", "confidence": 0.92},
      {"label": "Lying Down", "confidence": 0.87}
    ],
    messages: {
      "Normal Activity": "✅ Normal movement detected. Person appears to be moving safely.",
      "Fall Detected": "🚨 EMERGENCY: Fall detected! Immediate assistance may be required. Check on the person immediately.",
      "Sitting": "💺 Person is sitting comfortably. Normal resting position detected.",
      "Lying Down": "🛏️ Person is lying down. This could be normal rest or require attention - please verify."
    }
  },
  "Plant Pesticide Recommendation": {
    predictions: [
      {"label": "Neem Oil", "confidence": 0.91},
      {"label": "Insecticidal Soap", "confidence": 0.87},
      {"label": "Copper Fungicide", "confidence": 0.89},
      {"label": "Bacillus Thuringiensis", "confidence": 0.84},
      {"label": "No Treatment Needed", "confidence": 0.95}
    ],
    messages: {
      "Neem Oil": "🌿 Recommendation: Apply neem oil spray. It's organic and effective against aphids, mites, and fungal issues.",
      "Insecticidal Soap": "🧼 Recommendation: Use insecticidal soap for soft-bodied insects. Safe for plants and environment.",
      "Copper Fungicide": "🔷 Recommendation: Apply copper-based fungicide for bacterial and fungal diseases. Follow label instructions.",
      "Bacillus Thuringiensis": "🦋 Recommendation: Use BT spray for caterpillars and larvae. Safe biological control method.",
      "No Treatment Needed": "✅ Good news! Your plant doesn't require any pesticide treatment. Maintain current care routine."
    }
  },
  "Movie Recommendation System": {
    predictions: [
      {"label": "Action", "confidence": 0.88},
      {"label": "Comedy", "confidence": 0.92},
      {"label": "Drama", "confidence": 0.85},
      {"label": "Horror", "confidence": 0.79},
      {"label": "Romance", "confidence": 0.91},
      {"label": "Sci-Fi", "confidence": 0.86}
    ],
    messages: {
      "Action": "🎬 Perfect for action lovers! Get ready for thrilling adventures, epic battles, and heart-pounding excitement!",
      "Comedy": "😂 Time to laugh out loud! Comedy is the best medicine. Enjoy some light-hearted fun and giggles!",
      "Drama": "🎭 Deep storytelling awaits! Prepare for emotional journeys and compelling character development.",
      "Horror": "👻 Spooky recommendation! Perfect for thrill-seekers who love spine-chilling experiences. Watch with friends!",
      "Romance": "💕 Love is in the air! Perfect for a cozy movie night with heartwarming stories and beautiful relationships.",
      "Sci-Fi": "🚀 Blast off to the future! Explore new worlds, advanced technology, and mind-bending concepts!"
    }
  }
};

function getCurrentTime() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function addMessage(text, type = 'info') {
  const messageContent = document.getElementById('messageContent');
  if (!messageContent) return;
  
  const messageDiv = document.createElement('div');
  messageDiv.className = `message-item ${type}`;
  messageDiv.innerHTML = `
    <span class="timestamp">${getCurrentTime()}</span>
    <span class="message">${text}</span>
  `;
  
  messageContent.appendChild(messageDiv);
  messageContent.scrollTop = messageContent.scrollHeight;
}

function hideAllMediaElements() {
  document.getElementById('placeholderContent').classList.add('hidden');
  document.getElementById('videoElement').classList.add('hidden');
  document.getElementById('imageElement').classList.add('hidden');
  document.getElementById('captureBtn').classList.add('hidden');
  document.getElementById('stopBtn').classList.add('hidden');
  document.getElementById('clearBtn').classList.add('hidden');
}

function showPlaceholderContent() {
  hideAllMediaElements();
  document.getElementById('placeholderContent').classList.remove('hidden');
}

function simulateAIPrediction() {
  if (!selectedModel) {
    addMessage('No project selected for prediction', 'error');
    return;
  }
  
  addMessage('Processing with AI model...', 'info');
  
  setTimeout(() => {
    const projectInfo = projectData[selectedModel];
    if (!projectInfo) {
      addMessage('Project Data not found', 'error');
      return;
    }
    
    const prediction = projectInfo.predictions[Math.floor(Math.random() * projectInfo.predictions.length)];
    const confidence = (prediction.confidence * 100).toFixed(1);
    
    // Update results display - use displayName if available, otherwise use label
    const displayLabel = prediction.displayName || prediction.label;
    document.getElementById('resultLabel').textContent = displayLabel;
    document.getElementById('accuracyValue').textContent = `${confidence}%`;
    
    // Add prediction message
    addMessage(`Prediction: ${displayLabel} (${confidence}% confidence) using ${selectedModel}`, 'success');
    
    // Handle contextual message based on project type
    if (selectedModel === "Emotion Recognition System" && projectInfo.useEmotionLines && emotionLines) {
      // Use emotion lines from JSON file
      const emotionKey = prediction.label;
      const emotionMessages = emotionLines[emotionKey];
      
      if (emotionMessages && emotionMessages.length > 0) {
        const randomMessage = emotionMessages[Math.floor(Math.random() * emotionMessages.length)];
        setTimeout(() => {
          addMessage(`💭 ${randomMessage}`, 'emotion');
        }, 800);
      } else {
        setTimeout(() => {
          addMessage(`Emotion detected: ${displayLabel}. Stay positive! 😊`, 'emotion');
        }, 800);
      }
    } else {
      // Use default messages for other projects
      const contextualMessage = projectInfo.messages && projectInfo.messages[prediction.label];
      if (contextualMessage) {
        setTimeout(() => {
          addMessage(contextualMessage, 'motivation');
        }, 800);
      }
    }
  }, 1500);
}

// Wait for DOM to be fully loaded
window.addEventListener('load', async function() {
  console.log('Application loading...');
  
  // Load emotion lines for Emotion Recognition System
  await loadEmotionLines();
  
  // Model Selection Dropdown
  const modelButton = document.getElementById('modelSelectBtn');
  const dropdown = document.getElementById('modelDropdown');
  
  if (modelButton && dropdown) {
    modelButton.onclick = function(e) {
      e.preventDefault();
      e.stopPropagation();
      dropdown.classList.toggle('hidden');
      addMessage(dropdown.classList.contains('hidden') ? 'Model dropdown closed' : 'Model dropdown opened', 'info');
      return false;
    };
    
    // Add click handlers to dropdown items
    dropdown.onclick = function(e) {
      e.preventDefault();
      e.stopPropagation();
      
      if (e.target.classList.contains('dropdown-item')) {
        const model = e.target.getAttribute('Data-model');
        selectedModel = model;
        modelButton.textContent = `${model} 👉`;
        dropdown.classList.add('hidden');
        addMessage(`Selected project: ${model}`, 'success');
        addMessage(`Ready to analyze with ${model}. Upload an image or start webcam.`, 'info');
      }
      return false;
    };
  }
  
  // Close dropdown when clicking elsewhere
  document.onclick = function(e) {
    if (dropdown && !modelButton.contains(e.target) && !dropdown.contains(e.target)) {
      dropdown.classList.add('hidden');
    }
  };
  
  // File Upload
  const deviceButton = document.getElementById('deviceBtn');
  const fileInput = document.getElementById('fileInput');
  
  if (deviceButton && fileInput) {
    deviceButton.onclick = function(e) {
      e.preventDefault();
      if (!selectedModel) {
        addMessage('Please select a project first!', 'error');
        return false;
      }
      addMessage('Opening file picker...', 'info');
      fileInput.click();
      return false;
    };
    
    fileInput.onchange = function(e) {
      const file = e.target.files[0];
      if (!file) return;
      
      if (!file.type.startsWith('image/')) {
        addMessage('Please select a valid image file!', 'error');
        return;
      }
      
      if (file.size > 10 * 1024 * 1024) {
        addMessage('File too large! Max 10MB allowed.', 'error');
        return;
      }
      
      addMessage(`Loading image: ${file.name}`, 'info');
      
      const reader = new FileReader();
      reader.onload = function(event) {
        if (currentStream) {
          currentStream.getTracks().forEach(track => track.stop());
          currentStream = null;
          isWebcamActive = false;
        }
        
        hideAllMediaElements();
        const img = document.getElementById('imageElement');
        img.src = event.target.result;
        img.classList.remove('hidden');
        document.getElementById('clearBtn').classList.remove('hidden');
        
        addMessage(`Image loaded successfully: ${file.name}`, 'success');
        simulateAIPrediction();
      };
      reader.readAsDataURL(file);
    };
  }
  
  // Webcam
  const webcamButton = document.getElementById('webcamBtn');
  const videoElement = document.getElementById('videoElement');
  
  if (webcamButton) {
    webcamButton.onclick = async function(e) {
      e.preventDefault();
      
      if (!selectedModel) {
        addMessage('Please select a project first!', 'error');
        return false;
      }
      
      if (isWebcamActive) {
        addMessage('Webcam is already active', 'info');
        return false;
      }
      
      if (!navigator.mediaDevices?.getUserMedia) {
        addMessage('Webcam not supported in this browser', 'error');
        return false;
      }
      
      try {
        addMessage('Requesting webcam access...', 'info');
        
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 }
        });
        
        currentStream = stream;
        isWebcamActive = true;
        
        hideAllMediaElements();
        videoElement.srcObject = stream;
        videoElement.classList.remove('hidden');
        document.getElementById('captureBtn').classList.remove('hidden');
        document.getElementById('stopBtn').classList.remove('hidden');
        
        addMessage('Webcam activated successfully!', 'success');
        
      } catch (error) {
        let msg = 'Webcam access failed';
        if (error.name === 'NotAllowedError') msg = 'Webcam access denied';
        else if (error.name === 'NotFoundError') msg = 'No webcam found';
        addMessage(msg, 'error');
      }
      
      return false;
    };
  }
  
  // Capture Photo
  const captureButton = document.getElementById('captureBtn');
  if (captureButton) {
    captureButton.onclick = function(e) {
      e.preventDefault();
      
      const canvas = document.getElementById('captureCanvas');
      const ctx = canvas.getContext('2d');
      
      canvas.width = videoElement.videoWidth;
      canvas.height = videoElement.videoHeight;
      ctx.drawImage(videoElement, 0, 0);
      
      canvas.toBlob(function(blob) {
        const url = URL.createObjectURL(blob);
        hideAllMediaElements();
        
        const img = document.getElementById('imageElement');
        img.src = url;
        img.classList.remove('hidden');
        document.getElementById('clearBtn').classList.remove('hidden');
        
        addMessage('Photo captured from webcam!', 'success');
        simulateAIPrediction();
      });
      
      return false;
    };
  }
  
  // Stop Webcam
  const stopButton = document.getElementById('stopBtn');
  if (stopButton) {
    stopButton.onclick = function(e) {
      e.preventDefault();
      
      if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
        isWebcamActive = false;
      }
      
      showPlaceholderContent();
      addMessage('Webcam stopped', 'system');
      return false;
    };
  }
  
  // Clear Display
  const clearButton = document.getElementById('clearBtn');
  if (clearButton) {
    clearButton.onclick = function(e) {
      e.preventDefault();
      
      if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
        isWebcamActive = false;
      }
      
      if (fileInput) fileInput.value = '';
      
      document.getElementById('resultLabel').textContent = 'No prediction yet';
      document.getElementById('accuracyValue').textContent = '--';
      
      showPlaceholderContent();
      addMessage('Display cleared', 'system');
      return false;
    };
  }
  
  const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');

if (startBtn && stopBtn) {
  startBtn.addEventListener('click', function() {
    startBtn.classList.add('hidden');
    stopBtn.classList.remove('hidden');
    // Placeholder: Add start functionality here
    addMessage('Start button clicked. System started.', 'info');
  });

  stopBtn.addEventListener('click', function() {
    stopBtn.classList.add('hidden');
    startBtn.classList.remove('hidden');
    // Placeholder: Add stop functionality here
    addMessage('Stop button clicked. System stopped.', 'info');
  });
}

  // Clear Log
  const clearLogButton = document.getElementById('clearLogBtn');
  if (clearLogButton) {
    clearLogButton.onclick = function(e) {
      e.preventDefault();
      
      const messageContent = document.getElementById('messageContent');
      messageContent.innerHTML = '';
      addMessage('Message log cleared', 'system');
      return false;
    };
  }
  
  // Initialize application
  addMessage('ML Portfolio Interface loaded successfully!', 'system');
  addMessage('Select a project and choose an input source to begin analysis.', 'info');
  
  console.log('Application initialized');
});

// Cleanup
window.addEventListener('beforeunload', function() {
  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
  }
});