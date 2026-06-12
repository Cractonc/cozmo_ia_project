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
