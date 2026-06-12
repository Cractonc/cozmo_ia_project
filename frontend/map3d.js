const Map3D = (function() {
    let scene, camera, renderer, controls;
    let cozmoGroup;
    let targetMesh, lineMesh;
    let isInitialized = false;

    // Session management
    const sessions = {};
    const sessionColors = [0x3b82f6, 0x8b5cf6, 0x10b981, 0xf59e0b, 0xec4899];
    let colorIndex = 0;

    // Interaction
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const clickablePhotos = [];

    // Follow camera
    let followCam = false;

    // DOM
    const overlay = document.getElementById('map3d-overlay');
    const container = document.getElementById('map3d-canvas');
    const btnOpen = document.getElementById('btn-map3d');
    const btnClose = document.getElementById('btn-close-map');
    const sessionsList = document.getElementById('sessions-list');
    const followCamToggle = document.getElementById('follow-cam-toggle');

    function init() {
        if (isInitialized) return;
        
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a14);

        camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 1, 20000);
        camera.position.set(0, 1500, 2000);

        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(renderer.domElement);

        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        const gridHelper = new THREE.GridHelper(5000, 50, 0x444444, 0x222222);
        scene.add(gridHelper);

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
        dirLight.position.set(1000, 2000, 1000);
        scene.add(dirLight);

        buildCozmoModel();

        const homeGeo = new THREE.CylinderGeometry(20, 20, 5, 16);
        const homeMat = new THREE.MeshPhongMaterial({ color: 0x10b981 });
        const home = new THREE.Mesh(homeGeo, homeMat);
        home.position.set(0, 2.5, 0);
        scene.add(home);

        window.addEventListener('resize', onWindowResize, false);
        
        // Raycaster event
        container.addEventListener('click', onMouseClick, false);
        container.addEventListener('touchstart', onMouseClick, {passive: false});
        
        // Follow cam toggle
        if (followCamToggle) {
            followCamToggle.addEventListener('change', (e) => {
                followCam = e.target.checked;
            });
        }

        animate();
        isInitialized = true;
    }

    function buildCozmoModel() {
        cozmoGroup = new THREE.Group();
        
        // Tracks (Black Cylinders)
        const trackGeo = new THREE.CylinderGeometry(30, 30, 110, 16);
        const trackMat = new THREE.MeshPhongMaterial({ color: 0x111111 });
        const trackL = new THREE.Mesh(trackGeo, trackMat);
        trackL.rotation.x = Math.PI / 2;
        trackL.position.set(0, 30, -45);
        const trackR = trackL.clone();
        trackR.position.set(0, 30, 45);
        
        // Body (Red and White)
        const bodyGeo = new THREE.BoxGeometry(100, 50, 70);
        const bodyMat = new THREE.MeshPhongMaterial({ color: 0xffffff });
        const body = new THREE.Mesh(bodyGeo, bodyMat);
        body.position.set(-10, 45, 0);

        const accentGeo = new THREE.BoxGeometry(90, 10, 72);
        const accentMat = new THREE.MeshPhongMaterial({ color: 0xef4444 }); // Red
        const accent = new THREE.Mesh(accentGeo, accentMat);
        accent.position.set(-10, 25, 0);

        // Head (White box, black screen)
        const headGroup = new THREE.Group();
        const headGeo = new THREE.BoxGeometry(50, 45, 60);
        const headMat = new THREE.MeshPhongMaterial({ color: 0xffffff });
        const head = new THREE.Mesh(headGeo, headMat);
        
        const screenGeo = new THREE.PlaneGeometry(45, 30);
        const screenMat = new THREE.MeshBasicMaterial({ color: 0x000000 });
        const screen = new THREE.Mesh(screenGeo, screenMat);
        screen.position.set(26, 0, 0);
        screen.rotation.y = Math.PI / 2;
        
        headGroup.add(head);
        headGroup.add(screen);
        headGroup.position.set(45, 60, 0);
        headGroup.rotation.z = -Math.PI / 8; // Tilt slightly up

        // Lift arm (Red)
        const armGeo = new THREE.BoxGeometry(60, 10, 10);
        const armMat = new THREE.MeshPhongMaterial({ color: 0xef4444 });
        const armL = new THREE.Mesh(armGeo, armMat);
        armL.position.set(40, 30, -40);
        const armR = armL.clone();
        armR.position.set(40, 30, 40);

        cozmoGroup.add(trackL);
        cozmoGroup.add(trackR);
        cozmoGroup.add(body);
        cozmoGroup.add(accent);
        cozmoGroup.add(headGroup);
        cozmoGroup.add(armL);
        cozmoGroup.add(armR);
        
        scene.add(cozmoGroup);
    }

    function getSession(sessionId) {
        if (!sessionId) sessionId = "Mouvements Libres";
        
        if (!sessions[sessionId]) {
            const group = new THREE.Group();
            scene.add(group);
            
            const color = sessionColors[colorIndex % sessionColors.length];
            colorIndex++;

            const pathLineMat = new THREE.LineBasicMaterial({ color: color, linewidth: 2 });
            const pathLineGeo = new THREE.BufferGeometry();
            const pathLine = new THREE.Line(pathLineGeo, pathLineMat);
            pathLine.frustumCulled = false; // Fix: empêche le trait de disparaître si l'origine n'est pas visible
            group.add(pathLine);

            sessions[sessionId] = {
                id: sessionId,
                group: group,
                color: color,
                pathPoints: [],
                pathLine: pathLine
            };

            updateSessionsUI();
        }
        return sessions[sessionId];
    }

    function updateSessionsUI() {
        if (!sessionsList) return;
        sessionsList.innerHTML = '';
        
        Object.values(sessions).forEach(session => {
            const label = document.createElement('label');
            label.className = 'session-item';
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = session.group.visible;
            checkbox.addEventListener('change', (e) => {
                session.group.visible = e.target.checked;
            });
            
            const badge = document.createElement('span');
            badge.className = 'session-color-badge';
            badge.style.backgroundColor = '#' + session.color.toString(16).padStart(6, '0');
            
            const text = document.createTextNode(` ${session.id}`);
            
            label.appendChild(checkbox);
            label.appendChild(badge);
            label.appendChild(text);
            sessionsList.appendChild(label);
        });
    }

    function onMouseClick(event) {
        if (!isInitialized || overlay.classList.contains('hidden')) return;
        
        let clientX = event.clientX;
        let clientY = event.clientY;
        
        if (event.touches && event.touches.length > 0) {
            clientX = event.touches[0].clientX;
            clientY = event.touches[0].clientY;
        }
        
        const rect = container.getBoundingClientRect();
        mouse.x = ((clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        
        // Only check photos that are visible
        const visiblePhotos = clickablePhotos.filter(p => {
            let obj = p;
            while (obj) {
                if (!obj.visible) return false;
                obj = obj.parent;
            }
            return true;
        });
        
        const intersects = raycaster.intersectObjects(visiblePhotos, false);
        
        if (intersects.length > 0) {
            const photoData = intersects[0].object.userData;
            if (photoData && photoData.src && window.openPhotoModal) {
                window.openPhotoModal(photoData.src);
            }
        }
    }

    function onWindowResize() {
        if (!isInitialized || overlay.classList.contains('hidden')) return;
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    }

    function animate() {
        requestAnimationFrame(animate);
        
        // Photos always face camera
        clickablePhotos.forEach(photo => {
            photo.lookAt(camera.position);
        });

        // Follow cam
        if (followCam && cozmoGroup) {
            controls.target.copy(cozmoGroup.position);
        }

        controls.update();
        renderer.render(scene, camera);
    }

    function toggleMap() {
        overlay.classList.toggle('hidden');
        if (!overlay.classList.contains('hidden')) {
            if (!isInitialized) init();
            onWindowResize();
        }
    }

    if(btnOpen) btnOpen.addEventListener('click', toggleMap);
    if(btnClose) btnClose.addEventListener('click', toggleMap);

    return {
        updateCozmo: function(x, y, cap, sessionId) {
            if (!isInitialized) return;
            cozmoGroup.position.set(x, 0, -y);
            const rad = cap * Math.PI / 180;
            cozmoGroup.rotation.y = rad;

            // Path Tracing
            const session = getSession(sessionId);
            // Append point only if it moved significantly
            const lastPoint = session.pathPoints[session.pathPoints.length - 1];
            if (!lastPoint || lastPoint.distanceTo(new THREE.Vector3(x, 5, -y)) > 10) {
                session.pathPoints.push(new THREE.Vector3(x, 5, -y));
                session.pathLine.geometry.setFromPoints(session.pathPoints);
            }
        },
        
        setDestination: function(startX, startY, startCap, action, valeur, sessionId) {
            // Fonctionnalité désactivée : on ne dessine plus le pointillé rouge 
            // pour ne pas surcharger la carte vu qu'on a déjà le tracé au sol.
            return;
        },

        addPhotoMemory: function(img_b64, x, y, cap, sessionId) {
            if (!isInitialized || !img_b64) return;
            const session = getSession(sessionId);

            const src = "data:image/jpeg;base64," + img_b64;
            const image = new Image();
            image.src = src;
            image.onload = () => {
                const texture = new THREE.Texture(image);
                texture.needsUpdate = true;
                
                // Polaroid style frame
                const frameGeo = new THREE.PlaneGeometry(130, 110);
                const frameMat = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });
                const frame = new THREE.Mesh(frameGeo, frameMat);
                
                const picGeo = new THREE.PlaneGeometry(120, 90);
                const picMat = new THREE.MeshBasicMaterial({ map: texture });
                const pic = new THREE.Mesh(picGeo, picMat);
                pic.position.set(0, 5, 1); // slightly forward to avoid z-fighting
                
                frame.add(pic);
                
                // Position at a higher height so it doesn't intersect with the ground or other elements
                const photoHeight = 85;
                frame.position.set(x, photoHeight, -y);
                frame.userData = { src: src }; // For raycaster click
                
                // Thin line for the stem instead of a thick cylinder, attaching to the bottom of the photo
                const stemPoints = [
                    new THREE.Vector3(x, 0, -y),
                    new THREE.Vector3(x, photoHeight - 55, -y) // Bottom edge of the frame (110 height / 2 = 55)
                ];
                const stemGeo = new THREE.BufferGeometry().setFromPoints(stemPoints);
                const stemMat = new THREE.LineBasicMaterial({ color: 0xcccccc, linewidth: 2 });
                const stem = new THREE.Line(stemGeo, stemMat);
                
                session.group.add(frame);
                session.group.add(stem);
                
                clickablePhotos.push(frame);
            };
        },

        addHazardMarker: function(sessionId) {
            if (!isInitialized || !cozmoGroup) return;
            const session = getSession(sessionId);

            // Create a warning cone or cross at Cozmo's current position + slightly ahead
            const dist = 40; // mm ahead
            const rad = cozmoGroup.rotation.y;
            const hx = cozmoGroup.position.x + dist * Math.cos(rad);
            const hz = cozmoGroup.position.z - dist * Math.sin(rad); // Z is negative Y

            const geo = new THREE.ConeGeometry(20, 40, 16);
            const mat = new THREE.MeshPhongMaterial({ color: 0xf59e0b }); // Warning amber
            const cone = new THREE.Mesh(geo, mat);
            
            cone.position.set(hx, 20, hz);
            cone.rotation.x = Math.PI; // point down
            
            session.group.add(cone);
        }
    };
})();
