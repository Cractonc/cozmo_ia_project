const WS_URL = "ws://localhost:8000/ws";
let ws;
let isReconnecting = false;

// DOM Elements
const connectionLed = document.getElementById('connection-led');
const connectionText = document.getElementById('connection-text');
const videoFeed = document.getElementById('video-feed');
const stateBadge = document.getElementById('state-badge');
const currentActionBadge = document.getElementById('current-action-badge');
const cliffWarning = document.getElementById('cliff-warning');

// Photo Modal
const photoModal = document.getElementById('photo-modal');
const btnCloseModal = document.getElementById('btn-close-modal');
const modalImage = document.getElementById('modal-image');

window.openPhotoModal = function(src) {
    if (photoModal && modalImage) {
        modalImage.src = src;
        photoModal.classList.remove('hidden');
    }
};

if (btnCloseModal) {
    btnCloseModal.addEventListener('click', () => {
        photoModal.classList.add('hidden');
    });
}

// Odometry
const odomX = document.getElementById('odom-x');
const odomY = document.getElementById('odom-y');
const odomCap = document.getElementById('odom-cap');

// AI Logs
const aiLogsContainer = document.getElementById('ai-logs');
const terminalOutput = document.getElementById('terminal-output');

// Forms & Buttons
const commandForm = document.getElementById('command-form');
const commandInput = document.getElementById('command-input');
const btnStop = document.getElementById('btn-stop');
const btnForward = document.getElementById('btn-forward');
const btnLeft = document.getElementById('btn-left');
const btnRight = document.getElementById('btn-right');
const btnTraining = document.getElementById('btn-training');
const btnNNMode = document.getElementById('btn-nn-mode');

// Training Panel DOM Elements
const btnStartRecord = document.getElementById('btn-start-record');
const btnStopRecord = document.getElementById('btn-stop-record');
const btnExitTraining = document.getElementById('btn-exit-training');

// NN Panel DOM Elements
const btnNNStart = document.getElementById('btn-nn-start');
const btnNNStop = document.getElementById('btn-nn-stop');
const btnExitNN = document.getElementById('btn-exit-nn');
const btnNNKill = document.getElementById('btn-nn-kill');
const nnModelSelect = document.getElementById('nn-model-select');
const nnInferenceMs = document.getElementById('nn-inference-ms');
const nnFps = document.getElementById('nn-fps');
const nnCliffEvents = document.getElementById('nn-cliff-events');
const nnLeftBar = document.getElementById('nn-left-bar');
const nnLeftValue = document.getElementById('nn-left-value');
const nnRightBar = document.getElementById('nn-right-bar');
const nnRightValue = document.getElementById('nn-right-value');
const nnHeadFill = document.getElementById('nn-head-fill');
const nnHeadThumb = document.getElementById('nn-head-thumb');
const nnHeadValue = document.getElementById('nn-head-value');
const nnActiveBadge = document.getElementById('nn-active-badge');
const nnStatusBadge = document.getElementById('nn-status-badge');

// Hybrid Panel DOM Elements
const btnHybrid = document.getElementById('btn-hybrid');
const btnHybridStart = document.getElementById('btn-hybrid-start');
const btnHybridStop = document.getElementById('btn-hybrid-stop');
const btnExitHybrid = document.getElementById('btn-exit-hybrid');
const btnHybridKill = document.getElementById('btn-hybrid-kill');
const hybridModelSelect = document.getElementById('hybrid-model-select');
const hybridObjectiveInput = document.getElementById('hybrid-objective');
const hybridSceneDesc = document.getElementById('hybrid-scene-desc');
const hybridStrategy = document.getElementById('hybrid-strategy');
const hybridConfiance = document.getElementById('hybrid-confiance');
const hybridFps = document.getElementById('hybrid-fps');
const hybridInferenceMs = document.getElementById('hybrid-inference-ms');
const hybridCliffEvents = document.getElementById('hybrid-cliff-events');

function initWebSocket() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        isReconnecting = false;
        connectionLed.classList.add('connected');
        connectionText.textContent = 'Connecté';
        appendTerminalLog('Système connecté au backend Cozmo.', 'system');
    };

    ws.onclose = () => {
        connectionLed.classList.remove('connected');
        connectionText.textContent = 'Déconnecté';
        if (!isReconnecting) {
            appendTerminalLog('Connexion perdue. Tentative de reconnexion...', 'system');
            isReconnecting = true;
        }
        setTimeout(initWebSocket, 2000);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleIncomingData(data);
    };
}

function handleIncomingData(data) {
    switch(data.type) {
        case 'video_frame':
            // data.frame is base64 jpeg
            videoFeed.src = "data:image/jpeg;base64," + data.frame;
            break;
        
        case 'state_update':
            // Update Odometry
            if (data.odometry) {
                odomX.textContent = (data.odometry.x / 10).toFixed(2);
                odomY.textContent = (data.odometry.y / 10).toFixed(2);
                odomCap.textContent = data.odometry.cap.toFixed(1);
                
                if (typeof Map3D !== 'undefined') {
                    Map3D.updateCozmo(data.odometry.x, data.odometry.y, data.odometry.cap, data.session_id);
                }

                if (trainingModeActive) {
                    document.getElementById('train-pos-xy').textContent = 
                        `${(data.odometry.x / 10).toFixed(1)} / ${(data.odometry.y / 10).toFixed(1)}`;
                    document.getElementById('train-pos-cap').textContent = data.odometry.cap.toFixed(1);
                }
            }
            
            // Update State Badge
            if (data.state) {
                stateBadge.textContent = data.state;
                if (data.state === 'ACT') {
                    stateBadge.className = 'badge active';
                } else if (data.state === 'THINK') {
                    stateBadge.className = 'badge warning';
                    currentActionBadge.textContent = '💤 En attente';
                } else {
                    stateBadge.className = 'badge';
                }
            }

            // Update Cliff Warning
            if (data.hasOwnProperty('cliff_detected')) {
                if (data.cliff_detected) {
                    cliffWarning.classList.remove('hidden');
                    if (typeof Map3D !== 'undefined') {
                        Map3D.addHazardMarker(data.session_id);
                    }
                } else {
                    cliffWarning.classList.add('hidden');
                }
            }

            if (trainingModeActive && data.wheels) {
                document.getElementById('train-wheels-speed').textContent = 
                    `${data.wheels.left.toFixed(0)} / ${data.wheels.right.toFixed(0)}`;
            }
            break;

        case 'training_status':
            document.getElementById('train-frames').textContent = data.frame_count;
            document.getElementById('train-duration').textContent = data.duration_s.toFixed(2);
            document.getElementById('train-size').textContent = data.file_size_mb.toFixed(2);
            
            const recBadge = document.getElementById('training-rec-badge');
            const statusBadge = document.getElementById('training-status-badge');
            
            if (data.recording) {
                recBadge.classList.remove('hidden');
                statusBadge.textContent = 'Enregistrement...';
                statusBadge.className = 'badge danger';
                btnStartRecord.disabled = true;
                btnStopRecord.disabled = false;
                btnExitTraining.disabled = true;
            } else {
                recBadge.classList.add('hidden');
                statusBadge.textContent = 'Prêt';
                statusBadge.className = 'badge';
                btnStartRecord.disabled = false;
                btnStopRecord.disabled = true;
                btnExitTraining.disabled = false;
            }
            break;

        case 'nn_status':
            if (nnInferenceMs) nnInferenceMs.textContent = data.inference_ms.toFixed(1);
            if (nnFps) nnFps.textContent = data.fps.toFixed(1);
            if (nnCliffEvents) nnCliffEvents.textContent = data.cliff_events;
            
            // Update wheel speeds visualizer
            updateWheelBar('nn-left-bar', 'nn-left-value', data.left_speed);
            updateWheelBar('nn-right-bar', 'nn-right-value', data.right_speed);
            
            // Update head angle visualizer
            updateHeadSlider(data.head_angle);
            
            // Badges & Buttons state
            if (data.active) {
                if (nnActiveBadge) nnActiveBadge.classList.remove('hidden');
                if (nnStatusBadge) {
                    nnStatusBadge.textContent = 'En cours';
                    nnStatusBadge.className = 'badge danger';
                }
                if (btnNNStart) btnNNStart.disabled = true;
                if (btnNNStop) btnNNStop.disabled = false;
                if (nnModelSelect) nnModelSelect.disabled = true;
                if (btnExitNN) btnExitNN.disabled = true;
            } else {
                if (nnActiveBadge) nnActiveBadge.classList.add('hidden');
                if (nnStatusBadge) {
                    nnStatusBadge.textContent = 'Arrêté';
                    nnStatusBadge.className = 'badge';
                }
                if (btnNNStart) btnNNStart.disabled = false;
                if (btnNNStop) btnNNStop.disabled = true;
                if (nnModelSelect) nnModelSelect.disabled = false;
                if (btnExitNN) btnExitNN.disabled = false;
            }
            break;

        case 'hybrid_status':
            // Update wheel bars
            updateWheelBar('hybrid-left-bar', 'hybrid-left-value', data.left_speed);
            updateWheelBar('hybrid-right-bar', 'hybrid-right-value', data.right_speed);
            updateHeadSliderById('hybrid-head-fill', 'hybrid-head-thumb', 'hybrid-head-value', data.head_angle);
            
            // Update metrics
            if (hybridFps) hybridFps.textContent = data.fps.toFixed(1);
            if (hybridInferenceMs) hybridInferenceMs.textContent = data.inference_ms.toFixed(1);
            if (hybridCliffEvents) hybridCliffEvents.textContent = data.cliff_events;
            
            // Update compass
            updateCompass(data.cap_actuel, data.cap_cible);
            
            // Update scene & strategy text
            if (hybridSceneDesc && data.description_scene) hybridSceneDesc.textContent = data.description_scene;
            if (hybridStrategy && data.strategie) hybridStrategy.textContent = data.strategie;
            if (hybridConfiance && data.confiance !== undefined) {
                const pct = Math.round(data.confiance * 100);
                hybridConfiance.textContent = pct + '%';
                hybridConfiance.style.color = pct >= 70 ? 'var(--success)' : pct >= 40 ? 'var(--warning)' : 'var(--danger)';
            }
            
            // Update badges
            const nnBadgeH = document.getElementById('hybrid-nn-badge');
            const geminiBadgeH = document.getElementById('hybrid-gemini-badge');
            if (nnBadgeH) nnBadgeH.classList.toggle('hidden', !data.nn_active);
            if (geminiBadgeH) geminiBadgeH.classList.toggle('hidden', !data.gemini_active);
            
            // Button states
            if (data.nn_active || data.gemini_active) {
                if (btnHybridStart) btnHybridStart.disabled = true;
                if (btnHybridStop) btnHybridStop.disabled = false;
                if (hybridModelSelect) hybridModelSelect.disabled = true;
                if (hybridObjectiveInput) hybridObjectiveInput.disabled = true;
                if (btnExitHybrid) btnExitHybrid.disabled = true;
            }
            
            if (data.objectif_atteint) {
                appendTerminalLog('🎯 Objectif atteint !', 'system');
            }
            break;
            
        case 'hybrid_gemini':
            // Log the Gemini strategic decision
            const timeG = new Date().toLocaleTimeString();
            const cardG = document.createElement('div');
            cardG.className = 'log-card cot';
            cardG.innerHTML = `
                <div class="log-title">
                    <span>🧭 Gemini Stratégique</span>
                    <span class="log-time">${timeG}</span>
                </div>
                <div class="log-body">
                    <strong>Cap cible :</strong> ${data.cap_cible.toFixed(1)}°<br>
                    <strong>Scène :</strong> ${data.description_scene || 'N/A'}<br>
                    <strong>Stratégie :</strong> ${data.strategie || 'N/A'}<br>
                    <strong>Confiance :</strong> ${(data.confiance * 100).toFixed(0)}%
                </div>
            `;
            if (aiLogsContainer) {
                aiLogsContainer.appendChild(cardG);
                aiLogsContainer.scrollTop = aiLogsContainer.scrollHeight;
            }
            break;
            
        case 'hybrid_stopped':
            // Reset UI
            if (btnHybridStart) btnHybridStart.disabled = false;
            if (btnHybridStop) btnHybridStop.disabled = true;
            if (hybridModelSelect) hybridModelSelect.disabled = false;
            if (hybridObjectiveInput) hybridObjectiveInput.disabled = false;
            if (btnExitHybrid) btnExitHybrid.disabled = false;
            const nnB = document.getElementById('hybrid-nn-badge');
            const gemB = document.getElementById('hybrid-gemini-badge');
            if (nnB) nnB.classList.add('hidden');
            if (gemB) gemB.classList.add('hidden');
            updateCompass(0, 0);
            break;

        case 'ai_cot':
            // Log Chain of Thought
            const actionStr = formatActionText(data.action, data.valeur);
            currentActionBadge.textContent = actionStr;
            appendAILog(data, actionStr);

            if (typeof Map3D !== 'undefined' && data.odometry) {
                if (data.frame) {
                    Map3D.addPhotoMemory(data.frame, data.odometry.x, data.odometry.y, data.odometry.cap, data.session_id);
                }
                Map3D.setDestination(data.odometry.x, data.odometry.y, data.odometry.cap, data.action, data.valeur, data.session_id);
            }
            break;

        case 'terminal_log':
            appendTerminalLog(data.message, data.source || 'robot');
            break;
    }
}

function formatActionText(action, valeur) {
    if (action === 'avancer') return `⬆️ Avance de ${valeur || 0} cm`;
    if (action === 'reculer') return `⬇️ Recule de ${valeur || 0} cm`;
    if (action === 'pivoter_gauche' || action === 'tourner_gauche') return `⬅️ Gauche ${valeur || 0}°`;
    if (action === 'pivoter_droite' || action === 'tourner_droite') return `➡️ Droite ${valeur || 0}°`;
    if (action === 'stop') return `🛑 Stop`;
    if (action === 'none') return `💤 En attente`;
    return `⚙️ ${action}`;
}

function appendAILog(data, actionStr) {
    const time = new Date().toLocaleTimeString();
    const card = document.createElement('div');
    card.className = `log-card ${data.level || 'cot'}`;
    
    card.innerHTML = `
        <div class="log-title">
            <span>🧠 Décision IA</span>
            <span class="log-time">${time}</span>
        </div>
        <div class="log-body">
            <strong>Étape :</strong> ${data.etape || 'N/A'}<br>
            <strong>Diagnostic :</strong> ${data.diagnostic || 'N/A'}<br>
            <strong>Action :</strong> ${actionStr}
        </div>
        <div class="log-json">${JSON.stringify(data.raw || data, null, 2)}</div>
    `;
    
    aiLogsContainer.appendChild(card);
    aiLogsContainer.scrollTop = aiLogsContainer.scrollHeight;
}

function appendTerminalLog(message, source) {
    const entry = document.createElement('div');
    entry.className = `log-entry ${source}`;
    
    let prefix = '🤖 ';
    if (source === 'system') prefix = '⚙️ ';
    if (source === 'user') prefix = '❯ ';

    entry.textContent = `${prefix}${message}`;
    terminalOutput.appendChild(entry);
    terminalOutput.scrollTop = terminalOutput.scrollHeight;
}

function sendCommand(cmd) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'command', command: cmd }));
        // Le log local est retiré car le backend broadcast déjà la commande
    }
}

// Event Listeners
commandForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const cmd = commandInput.value.trim();
    if (cmd) {
        sendCommand(cmd);
        commandInput.value = '';
    }
});

btnStop.addEventListener('click', () => sendCommand('stop'));
btnForward.addEventListener('click', () => sendCommand('avancer 10'));
btnLeft.addEventListener('click', () => sendCommand('tourner_gauche 90'));
btnRight.addEventListener('click', () => sendCommand('tourner_droite 90'));

// Initialize
initWebSocket();

// =====================================================================
// --- ENTRAÎNEMENT MODE (Clavier + Enregistrement) ---
// =====================================================================
let trainingModeActive = false;
let trainingDriveInterval = null;
let pressedKeys = {};
let currentHeadTargetAngle = 30.0;

const SPEED_FORWARD = 80;
const SPEED_TURN = 50;
const SPEED_CURVE = 30;

function calculateWheelSpeeds() {
    let left = 0;
    let right = 0;
    
    let fwd = pressedKeys["KeyW"] || pressedKeys["KeyZ"];
    let bwd = pressedKeys["KeyS"];
    let leftTurn = pressedKeys["KeyA"] || pressedKeys["KeyQ"];
    let rightTurn = pressedKeys["KeyD"];
    
    if (fwd && leftTurn) {
        left = SPEED_CURVE;
        right = SPEED_FORWARD;
    } else if (fwd && rightTurn) {
        left = SPEED_FORWARD;
        right = SPEED_CURVE;
    } else if (bwd && leftTurn) {
        left = -SPEED_CURVE;
        right = -SPEED_FORWARD * 0.75;
    } else if (bwd && rightTurn) {
        left = -SPEED_FORWARD * 0.75;
        right = -SPEED_CURVE;
    } else if (fwd) {
        left = SPEED_FORWARD;
        right = SPEED_FORWARD;
    } else if (bwd) {
        left = -SPEED_FORWARD * 0.75;
        right = -SPEED_FORWARD * 0.75;
    } else if (leftTurn) {
        left = -SPEED_TURN;
        right = SPEED_TURN;
    } else if (rightTurn) {
        left = SPEED_TURN;
        right = -SPEED_TURN;
    }
    
    return { left, right };
}

function updateKeycapUI(code, isPressed) {
    let elementId = `key-${code}`;
    if (code === "KeyW" || code === "KeyZ") elementId = "key-KeyW";
    if (code === "KeyA" || code === "KeyQ") elementId = "key-KeyA";
    
    const keycap = document.getElementById(elementId);
    if (keycap) {
        if (isPressed) {
            keycap.classList.add('active');
        } else {
            keycap.classList.remove('active');
        }
    }
}

function enterTrainingMode() {
    trainingModeActive = true;
    pressedKeys = {};
    
    // Switch panels
    document.getElementById('main-dashboard').classList.add('hidden');
    document.getElementById('training-panel').classList.remove('hidden');
    
    // Move video element
    const videoFeed = document.getElementById('video-feed');
    document.getElementById('training-video-wrapper').appendChild(videoFeed);
    
    // Notify backend
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'training_mode', active: true }));
    }
    
    // Reset keycaps
    document.querySelectorAll('.keycap').forEach(el => el.classList.remove('active'));
    
    // Start interval
    if (trainingDriveInterval) clearInterval(trainingDriveInterval);
    trainingDriveInterval = setInterval(() => {
        if (!trainingModeActive) return;
        const speeds = calculateWheelSpeeds();
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'training_drive',
                left: speeds.left,
                right: speeds.right
            }));
        }
    }, 50);
}

function exitTrainingMode() {
    trainingModeActive = false;
    if (trainingDriveInterval) {
        clearInterval(trainingDriveInterval);
        trainingDriveInterval = null;
    }
    
    // Switch panels back
    document.getElementById('training-panel').classList.add('hidden');
    document.getElementById('main-dashboard').classList.remove('hidden');
    
    // Move video element back
    const videoFeed = document.getElementById('video-feed');
    document.querySelector('.video-container .video-wrapper').appendChild(videoFeed);
    
    // Notify backend
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'training_mode', active: false }));
    }
}

function sendHeadAngle(angle) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'training_head', angle: angle }));
    }
}

// Event Listeners for Keyboard
window.addEventListener('keydown', (e) => {
    if (!trainingModeActive) return;
    if (["KeyW", "KeyS", "KeyA", "KeyD", "KeyZ", "KeyQ", "ArrowUp", "ArrowDown"].includes(e.code)) {
        e.preventDefault();
    }
    pressedKeys[e.code] = true;
    updateKeycapUI(e.code, true);
    
    if (e.code === "ArrowUp") {
        currentHeadTargetAngle = Math.min(44.5, currentHeadTargetAngle + 2.0);
        sendHeadAngle(currentHeadTargetAngle);
    } else if (e.code === "ArrowDown") {
        currentHeadTargetAngle = Math.max(-25.0, currentHeadTargetAngle - 2.0);
        sendHeadAngle(currentHeadTargetAngle);
    }
});

window.addEventListener('keyup', (e) => {
    if (!trainingModeActive) return;
    pressedKeys[e.code] = false;
    updateKeycapUI(e.code, false);
});

// Button event listeners
if (btnTraining) btnTraining.addEventListener('click', enterTrainingMode);
if (btnExitTraining) btnExitTraining.addEventListener('click', exitTrainingMode);

if (btnStartRecord) {
    btnStartRecord.addEventListener('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'training_start' }));
        }
    });
}

if (btnStopRecord) {
    btnStopRecord.addEventListener('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'training_stop' }));
        }
    });
}

// =====================================================================
// --- NN PILOT MODE (Inférence & Visualisation) ---
// =====================================================================
let nnModeActive = false;

function updateWheelBar(elementId, valueId, speed) {
    const bar = document.getElementById(elementId);
    const valueText = document.getElementById(valueId);
    if (!bar || !valueText) return;
    
    valueText.textContent = `${speed.toFixed(1)} mm/s`;
    
    // Clip speed to [-150, 150]
    const clipped = Math.max(-150, Math.min(150, speed));
    const percentage = (clipped / 150) * 50; // Range: -50 to 50
    
    if (clipped >= 0) {
        bar.style.bottom = "50%";
        bar.style.height = `${percentage}%`;
        bar.style.backgroundColor = "var(--success)";
    } else {
        bar.style.bottom = `${50 + percentage}%`; // percentage is negative
        bar.style.height = `${-percentage}%`;
        bar.style.backgroundColor = "var(--danger)";
    }
}

function updateHeadSlider(angle) {
    if (!nnHeadFill || !nnHeadThumb || !nnHeadValue) return;
    
    nnHeadValue.textContent = `${angle.toFixed(1)}°`;
    
    const clipped = Math.max(-25, Math.min(45, angle));
    const pct = ((clipped + 25) / 70) * 100;
    
    nnHeadFill.style.width = `${pct}%`;
    nnHeadThumb.style.left = `${pct}%`;
}

function updateHeadSliderById(fillId, thumbId, valueId, angle) {
    const fill = document.getElementById(fillId);
    const thumb = document.getElementById(thumbId);
    const value = document.getElementById(valueId);
    if (!fill || !thumb || !value) return;
    value.textContent = angle.toFixed(1) + '°';
    const clipped = Math.max(-25, Math.min(45, angle));
    const pct = ((clipped + 25) / 70) * 100;
    fill.style.width = pct + '%';
    thumb.style.left = pct + '%';
}

function updateCompass(capActuel, capCible) {
    const needleCurrent = document.getElementById('compass-needle-current');
    const needleTarget = document.getElementById('compass-needle-target');
    const capActuelText = document.getElementById('compass-cap-actuel');
    const capCibleText = document.getElementById('compass-cap-cible');
    
    if (needleCurrent) needleCurrent.style.transform = 'rotate(' + capActuel + 'deg)';
    if (needleTarget) needleTarget.style.transform = 'rotate(' + capCible + 'deg)';
    if (capActuelText) capActuelText.textContent = capActuel.toFixed(1) + '°';
    if (capCibleText) capCibleText.textContent = capCible.toFixed(1) + '°';
}

async function loadModels() {
    if (!nnModelSelect) return;
    nnModelSelect.innerHTML = '<option value="">Chargement...</option>';
    try {
        const response = await fetch('/api/models');
        const models = await response.json();
        if (models.length === 0) {
            nnModelSelect.innerHTML = '<option value="">Aucun modèle (.pt) trouvé</option>';
            return;
        }
        nnModelSelect.innerHTML = '';
        models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.name;
            opt.textContent = m.displayName || m.name;
            nnModelSelect.appendChild(opt);
        });
    } catch (err) {
        console.error("Error loading models:", err);
        nnModelSelect.innerHTML = '<option value="">Erreur de chargement</option>';
    }
}

function enterNNMode() {
    nnModeActive = true;
    
    // Switch panels
    document.getElementById('main-dashboard').classList.add('hidden');
    document.getElementById('nn-panel').classList.remove('hidden');
    
    // Move video element
    const feed = document.getElementById('video-feed');
    const wrapper = document.getElementById('nn-video-wrapper');
    if (feed && wrapper) {
        wrapper.appendChild(feed);
    }
    
    // Reset visualizer values
    updateWheelBar('nn-left-bar', 'nn-left-value', 0.0);
    updateWheelBar('nn-right-bar', 'nn-right-value', 0.0);
    updateHeadSlider(0.0);
    if (nnInferenceMs) nnInferenceMs.textContent = "0.0";
    if (nnFps) nnFps.textContent = "0.0";
    if (nnCliffEvents) nnCliffEvents.textContent = "0";
    
    // Load models list
    loadModels();
}

function exitNNMode() {
    // Notify backend to stop
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'nn_stop' }));
    }
    
    nnModeActive = false;
    
    // Switch panels back
    document.getElementById('nn-panel').classList.add('hidden');
    document.getElementById('main-dashboard').classList.remove('hidden');
    
    // Move video element back
    const feed = document.getElementById('video-feed');
    const normalWrapper = document.querySelector('.video-container .video-wrapper');
    if (feed && normalWrapper) {
        normalWrapper.appendChild(feed);
    }
}

// Event listeners for NN control
if (btnNNMode) btnNNMode.addEventListener('click', enterNNMode);
if (btnExitNN) btnExitNN.addEventListener('click', exitNNMode);

if (btnNNStart) {
    btnNNStart.addEventListener('click', () => {
        if (!nnModelSelect || !nnModelSelect.value) {
            alert("Veuillez sélectionner un modèle.");
            return;
        }
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'nn_start',
                model: nnModelSelect.value
            }));
        }
    });
}

if (btnNNStop) {
    btnNNStop.addEventListener('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'nn_stop' }));
        }
    });
}

if (btnNNKill) {
    btnNNKill.addEventListener('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'nn_stop' }));
        }
    });
}

// =====================================================================
// --- HYBRID EXPLORATION MODE ---
// =====================================================================
let hybridModeActive = false;

async function loadHybridModels() {
    if (!hybridModelSelect) return;
    hybridModelSelect.innerHTML = '<option value="">Chargement...</option>';
    try {
        const response = await fetch('/api/models');
        const models = await response.json();
        if (models.length === 0) {
            hybridModelSelect.innerHTML = '<option value="">Aucun modèle trouvé</option>';
            return;
        }
        hybridModelSelect.innerHTML = '<option value="">Auto (dernier modèle)</option>';
        models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.name;
            opt.textContent = m.displayName || m.name;
            hybridModelSelect.appendChild(opt);
        });
    } catch (err) {
        console.error('Error loading models:', err);
        hybridModelSelect.innerHTML = '<option value="">Erreur de chargement</option>';
    }
}

function enterHybridMode() {
    hybridModeActive = true;
    document.getElementById('main-dashboard').classList.add('hidden');
    document.getElementById('hybrid-panel').classList.remove('hidden');
    
    const feed = document.getElementById('video-feed');
    const wrapper = document.getElementById('hybrid-video-wrapper');
    if (feed && wrapper) wrapper.appendChild(feed);
    
    // Reset UI
    updateWheelBar('hybrid-left-bar', 'hybrid-left-value', 0.0);
    updateWheelBar('hybrid-right-bar', 'hybrid-right-value', 0.0);
    updateHeadSliderById('hybrid-head-fill', 'hybrid-head-thumb', 'hybrid-head-value', 0.0);
    updateCompass(0, 0);
    if (hybridSceneDesc) hybridSceneDesc.textContent = 'En attente...';
    if (hybridStrategy) hybridStrategy.textContent = 'En attente...';
    if (hybridConfiance) hybridConfiance.textContent = '-';
    if (hybridFps) hybridFps.textContent = '0.0';
    if (hybridInferenceMs) hybridInferenceMs.textContent = '0.0';
    if (hybridCliffEvents) hybridCliffEvents.textContent = '0';
    
    loadHybridModels();
}

function exitHybridMode() {
    // Send stop first
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'hybrid_stop' }));
    }
    hybridModeActive = false;
    document.getElementById('hybrid-panel').classList.add('hidden');
    document.getElementById('main-dashboard').classList.remove('hidden');
    
    const feed = document.getElementById('video-feed');
    const normalWrapper = document.querySelector('.video-container .video-wrapper');
    if (feed && normalWrapper) normalWrapper.appendChild(feed);
}

// Event Listeners
if (btnHybrid) btnHybrid.addEventListener('click', enterHybridMode);
if (btnExitHybrid) btnExitHybrid.addEventListener('click', exitHybridMode);

if (btnHybridStart) {
    btnHybridStart.addEventListener('click', function() {
        const objective = hybridObjectiveInput ? hybridObjectiveInput.value.trim() : '';
        if (!objective) {
            alert('Veuillez entrer un objectif (ex: explore la pièce)');
            return;
        }
        const model = hybridModelSelect ? hybridModelSelect.value : '';
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'hybrid_start',
                model: model,
                objective: objective
            }));
        }
    });
}

if (btnHybridStop) {
    btnHybridStop.addEventListener('click', function() {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'hybrid_stop' }));
        }
    });
}

if (btnHybridKill) {
    btnHybridKill.addEventListener('click', function() {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'hybrid_stop' }));
        }
    });
}
