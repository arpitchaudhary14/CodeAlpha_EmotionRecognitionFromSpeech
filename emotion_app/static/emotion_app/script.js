document.addEventListener('DOMContentLoaded', () => {
    const recordBtn = document.getElementById('recordBtn');
    const audioUpload = document.getElementById('audioUpload');
    const visualizer = document.getElementById('visualizer');
    const statusMessage = document.getElementById('statusMessage');
    const resultContainer = document.getElementById('resultContainer');
    const emotionDisplay = document.getElementById('emotionDisplay');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');

    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;

    // Handle File Upload
    audioUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            processAudio(file);
        }
    });

    // Handle Recording
    recordBtn.addEventListener('click', async () => {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                const rawBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
                
                try {
                    // Convert raw browser audio to proper WAV format compatible with Librosa
                    const wavBlob = await convertToWav(rawBlob);
                    const audioFile = new File([wavBlob], "recording.wav", { type: 'audio/wav' });
                    processAudio(audioFile);
                } catch (err) {
                    console.error("WAV conversion error:", err);
                    statusMessage.textContent = "Error: Failed to process microphone audio.";
                }
                
                // Stop all tracks to release the microphone
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            isRecording = true;
            
            // UI Updates
            recordBtn.classList.add('recording');
            recordBtn.innerHTML = '<span class="icon">⏹️</span> Stop Recording';
            visualizer.classList.add('active');
            statusMessage.textContent = 'Listening...';
            resultContainer.classList.remove('show');

        } catch (err) {
            alert('Microphone access denied or not available. Please allow permissions or use file upload.');
            console.error(err);
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        isRecording = false;
        
        // UI Updates
        recordBtn.classList.remove('recording');
        recordBtn.innerHTML = '<span class="icon">🎤</span> Start Recording';
        visualizer.classList.remove('active');
        statusMessage.textContent = 'Processing audio...';
    }

    async function processAudio(file) {
        statusMessage.textContent = 'ResoNate is Thinking...';
        resultContainer.classList.remove('show');
        
        const formData = new FormData();
        formData.append('audio', file);

        try {
            const response = await fetch('/api/predict/', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                displayResult(data.emotion, data.confidence, data.probabilities);
            } else {
                throw new Error(data.message || 'Server error during prediction');
            }
        } catch (error) {
            statusMessage.textContent = 'Error: ' + error.message;
            console.error(error);
        }
    }

    function displayResult(emotion, confidence, probabilities = {}) {
        statusMessage.textContent = 'Analysis complete.';
        
        // Setup Emotion Text and Color
        emotionDisplay.textContent = emotion.toUpperCase();
        emotionDisplay.className = 'emotion-display animated-title'; // Reset classes
        emotionDisplay.classList.add(`emotion-${emotion.toLowerCase()}`);
        
        // Define dynamic reactions, colors, and icons
        const emotionReaction = document.getElementById('emotionReaction');
        const emotionIcon = document.getElementById('emotionIcon');
        const confidenceWarning = document.getElementById('confidenceWarning');
        const probabilityBreakdown = document.getElementById('probabilityBreakdown');
        
        let reactionText = "";
        let svgIcon = "";
        let strokeColor = 'var(--primary-color)';

        const emotionLower = emotion.toLowerCase();
        
        if (emotionLower === 'happy') {
            strokeColor = 'var(--success-color)';
            reactionText = "High energy and elevated pitch detected! Keep the good vibes rolling. 🎉";
            svgIcon = `<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M8 14s1.5 2 4 2 4-2 4-2"></path><line x1="9" y1="9" x2="9.01" y2="9"></line><line x1="15" y1="9" x2="15.01" y2="9"></line></svg>`;
        } else if (emotionLower === 'sad') {
            strokeColor = 'var(--secondary-color)';
            reactionText = "Lower energy and drawn-out acoustic patterns detected. It sounds like a somber moment. 🌧️";
            svgIcon = `<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M16 16s-1.5-2-4-2-4 2-4 2"></path><line x1="9" y1="9" x2="9.01" y2="9"></line><line x1="15" y1="9" x2="15.01" y2="9"></line></svg>`;
        } else if (emotionLower === 'angry') {
            strokeColor = 'var(--warning-color)';
            reactionText = "High intensity and sharp pitch shifts detected! A strong emotional response. ⚡";
            svgIcon = `<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M16 16s-1.5-2-4-2-4 2-4 2"></path><path d="M7 8l2 2"></path><path d="M17 8l-2 2"></path></svg>`;
        } else if (emotionLower === 'neutral') {
            strokeColor = '#94a3b8';
            reactionText = "Calm, baseline speech patterns detected. Cool and collected. 🧊";
            svgIcon = `<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="8" y1="15" x2="16" y2="15"></line><line x1="9" y1="9" x2="9.01" y2="9"></line><line x1="15" y1="9" x2="15.01" y2="9"></line></svg>`;
        }

        if (emotionReaction) emotionReaction.textContent = reactionText;
        if (emotionIcon) {
            emotionIcon.innerHTML = svgIcon;
            emotionIcon.style.color = strokeColor;
        }
        
        // Setup Confidence Bar
        const percentage = Math.round(confidence * 100);
        confidenceText.textContent = `Confidence: ${percentage}%`;
        
        const confidenceTextVal = document.getElementById('confidenceTextVal');
        if (confidenceTextVal) {
            confidenceTextVal.textContent = percentage;
        }

        // Setup Confidence Warning System
        if (confidenceWarning) {
            confidenceWarning.style.display = 'block';
            if (confidence >= 0.80) {
                confidenceWarning.textContent = "High confidence prediction.";
                confidenceWarning.style.backgroundColor = "rgba(16, 185, 129, 0.1)";
                confidenceWarning.style.color = "var(--success-color)";
                confidenceWarning.style.border = "1px solid rgba(16, 185, 129, 0.3)";
            } else if (confidence >= 0.60) {
                confidenceWarning.textContent = "Moderate confidence prediction.";
                confidenceWarning.style.backgroundColor = "rgba(245, 158, 11, 0.1)";
                confidenceWarning.style.color = "var(--warning-color)";
                confidenceWarning.style.border = "1px solid rgba(245, 158, 11, 0.3)";
            } else {
                confidenceWarning.textContent = "Low confidence prediction. Please provide a clearer audio sample.";
                confidenceWarning.style.backgroundColor = "rgba(239, 68, 68, 0.1)";
                confidenceWarning.style.color = "#ef4444";
                confidenceWarning.style.border = "1px solid rgba(239, 68, 68, 0.3)";
            }
        }
        
        // Setup Probability Breakdown
        if (probabilityBreakdown && Object.keys(probabilities).length > 0) {
            probabilityBreakdown.style.display = 'block';
            for (const [emo, prob] of Object.entries(probabilities)) {
                const emoKey = emo.toLowerCase();
                const probPercent = Math.round(prob * 100);
                
                const valEl = document.getElementById(`prob-val-${emoKey}`);
                const barEl = document.getElementById(`prob-bar-${emoKey}`);
                
                if (valEl) valEl.textContent = `${probPercent}%`;
                
                // Small delay for CSS transition
                setTimeout(() => {
                    if (barEl) barEl.style.width = `${probPercent}%`;
                }, 150);
            }
        }

        // Trigger animations
        resultContainer.classList.add('show');
        
        // Small delay to allow transition to register
        setTimeout(() => {
            confidenceFill.style.width = `${percentage}%`;
            
            const circularProgress = document.getElementById('circularProgress');
            if (circularProgress) {
                const maxDashoffset = 251.2;
                const targetOffset = maxDashoffset - (maxDashoffset * percentage) / 100;
                circularProgress.style.strokeDashoffset = targetOffset;
                
                // Color the circle based on emotion
                circularProgress.style.stroke = strokeColor;
            }
        }, 100);
    }

    // --- Audio Conversion Utilities ---
    async function convertToWav(blob) {
        const arrayBuffer = await blob.arrayBuffer();
        // Use browser's native AudioContext to decode the raw webm/opus data
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        
        return audioBufferToWav(audioBuffer);
    }

    function audioBufferToWav(buffer) {
        const numOfChan = buffer.numberOfChannels;
        const length = buffer.length * numOfChan * 2 + 44;
        const out = new ArrayBuffer(length);
        const view = new DataView(out);
        const channels = [];
        let pos = 0;
        let offset = 0;
        let sample = 0;

        // write WAVE header
        setUint32(0x46464952); // "RIFF"
        setUint32(length - 8); // file length - 8
        setUint32(0x45564157); // "WAVE"

        setUint32(0x20746d66); // "fmt " chunk
        setUint32(16); // length = 16
        setUint16(1); // PCM (uncompressed)
        setUint16(numOfChan);
        setUint32(buffer.sampleRate);
        setUint32(buffer.sampleRate * 2 * numOfChan); // avg. bytes/sec
        setUint16(numOfChan * 2); // block-align
        setUint16(16); // 16-bit

        setUint32(0x61746164); // "data" - chunk
        setUint32(length - pos - 4); // chunk length

        // write interleaved data
        for (let i = 0; i < buffer.numberOfChannels; i++) {
            channels.push(buffer.getChannelData(i));
        }

        while (pos < length) {
            for (let i = 0; i < numOfChan; i++) {
                // interleave channels
                sample = Math.max(-1, Math.min(1, channels[i][offset])); // clamp
                sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0; // scale to 16-bit signed int
                view.setInt16(pos, sample, true); // write 16-bit sample
                pos += 2;
            }
            offset++; // next source sample
        }

        function setUint16(data) {
            view.setUint16(pos, data, true);
            pos += 2;
        }

        function setUint32(data) {
            view.setUint32(pos, data, true);
            pos += 4;
        }

        return new Blob([out], { type: "audio/wav" });
    }
});
