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
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.0;
        container.appendChild(renderer.domElement);

        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        // Empêche de passer sous le sol, tout en gardant un micro-angle pour toujours voir le parquet
        controls.maxPolarAngle = (Math.PI / 2) - 0.03; 

        // --- Sol Parquet Réaliste ---
        const floorGeo = new THREE.PlaneGeometry(10000, 10000);
        const floorMat = new THREE.MeshStandardMaterial({
            map: createParquetTexture(),
            roughness: 0.6,
            metalness: 0.1
        });
        const floor = new THREE.Mesh(floorGeo, floorMat);
        floor.rotation.x = -Math.PI / 2;
        floor.receiveShadow = true;
        scene.add(floor);

        // --- Eclairage ---
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        scene.add(ambientLight);
        
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(1000, 2000, 1000);
        dirLight.castShadow = true;
        dirLight.shadow.mapSize.width = 2048;
        dirLight.shadow.mapSize.height = 2048;
        dirLight.shadow.camera.near = 0.5;
        dirLight.shadow.camera.far = 5000;
        dirLight.shadow.camera.left = -2000;
        dirLight.shadow.camera.right = 2000;
        dirLight.shadow.camera.top = 2000;
        dirLight.shadow.camera.bottom = -2000;
        scene.add(dirLight);

        // --- HDRI ---
        const pmremGenerator = new THREE.PMREMGenerator(renderer);
        pmremGenerator.compileEquirectangularShader();
        new THREE.RGBELoader()
            .load('https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/equirectangular/royal_esplanade_1k.hdr', function (texture) {
                const envMap = pmremGenerator.fromEquirectangular(texture).texture;
                scene.environment = envMap;
                texture.dispose();
                pmremGenerator.dispose();
            });

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
        scene.add(cozmoGroup);

        const loader = new THREE.GLTFLoader();
        // Essai de charger un vrai modèle 3D s'il existe
        loader.load(
            'assets/cozmo.glb', // URL du modèle
            function (gltf) {
                const model = gltf.scene;
                
                // On applique les matériaux PBR réalistes sur le modèle chargé
                model.traverse((child) => {
                    if (child.isMesh) {
                        child.castShadow = true;
                        child.receiveShadow = true;
                        
                        // Logique simplifiée pour assigner des matériaux selon le nom des mesh
                        if (child.name.toLowerCase().includes('screen') || child.name.toLowerCase().includes('face')) {
                            child.material = new THREE.MeshStandardMaterial({
                                color: 0x000000,
                                roughness: 0.1, // Très lisse (verre)
                                metalness: 0.8
                            });
                        } else if (child.name.toLowerCase().includes('track') || child.name.toLowerCase().includes('wheel')) {
                            child.material = new THREE.MeshStandardMaterial({
                                color: 0x111111,
                                roughness: 0.9, // Caoutchouc mat
                                metalness: 0.0
                            });
                        } else {
                            // Plastique blanc/rouge par défaut
                            child.material = new THREE.MeshStandardMaterial({
                                color: child.material ? child.material.color : 0xffffff,
                                roughness: 0.5, // Plastique rugueux
                                metalness: 0.1
                            });
                        }
                    }
                });
                
                // Ajustement de l'échelle si nécessaire (à adapter selon le GLTF)
                model.scale.set(10, 10, 10);
                model.position.set(0, 10, 0); // Ajuster la hauteur
                
                cozmoGroup.add(model);
            },
            undefined, // onProgress
            function (error) {
                console.warn("Modèle cozmo.glb introuvable. Chargement du modèle de secours procédural PBR.");
                buildProceduralPBRCozmo();
            }
        );
    }

    function buildProceduralPBRCozmo() {
        // Matériaux PBR
        const plasticWhite = new THREE.MeshStandardMaterial({ color: 0xeeeeee, roughness: 0.4, metalness: 0.1 });
        const plasticRed = new THREE.MeshStandardMaterial({ color: 0xcc1111, roughness: 0.5, metalness: 0.2 });
        const plasticBlack = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.6, metalness: 0.1 });
        const screenMat = new THREE.MeshStandardMaterial({ color: 0x020202, roughness: 0.1, metalness: 0.8 });
        const eyeMat = new THREE.MeshBasicMaterial({ color: 0x00aaff });
        const chassisMat = new THREE.MeshStandardMaterial({ color: 0x555555, roughness: 0.5, metalness: 0.6 });

        // Texture procédurale des chenilles
        function createTreadTexture() {
            const canvas = document.createElement('canvas');
            canvas.width = 128;
            canvas.height = 512;
            const ctx = canvas.getContext('2d');
            
            ctx.fillStyle = '#111111';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            ctx.fillStyle = '#2a2a2a';
            for (let y = 0; y < canvas.height; y += 32) {
                ctx.fillRect(0, y + 8, canvas.width, 16);
            }
            
            const texture = new THREE.CanvasTexture(canvas);
            texture.wrapS = THREE.RepeatWrapping;
            texture.wrapT = THREE.RepeatWrapping;
            texture.repeat.set(1, 2);
            return texture;
        }
        const texturedRubberMat = new THREE.MeshStandardMaterial({ 
            color: 0x1a1a1a, 
            roughness: 0.9, 
            metalness: 0.0,
            map: createTreadTexture()
        });

        // Forme de la grande roue avant avec les 3 trous
        function createFrontWheelGeo() {
            const shape = new THREE.Shape();
            const radius = 16;
            shape.absarc(0, 0, radius, 0, Math.PI * 2, false);
            
            for(let i=0; i<3; i++) {
                const angle = (i * Math.PI * 2 / 3);
                const hole = new THREE.Path();
                const innerR = 5;
                const outerR = 12;
                const spread = 0.6;
                
                hole.moveTo(Math.cos(angle) * innerR, Math.sin(angle) * innerR);
                hole.lineTo(Math.cos(angle + spread) * outerR, Math.sin(angle + spread) * outerR);
                hole.lineTo(Math.cos(angle - spread) * outerR, Math.sin(angle - spread) * outerR);
                hole.lineTo(Math.cos(angle) * innerR, Math.sin(angle) * innerR);
                
                shape.holes.push(hole);
            }
            
            const extrudeSettings = { depth: 6, bevelEnabled: true, bevelSegments: 2, steps: 1, bevelSize: 0.5, bevelThickness: 0.5 };
            const geo = new THREE.ExtrudeGeometry(shape, extrudeSettings);
            geo.center();
            return geo;
        }

        const frontWheelGeo = createFrontWheelGeo();

        // Fonction utilitaire pour créer une chenille complète avec le châssis
        function createTrack(zOffset) {
            const trackGroup = new THREE.Group();
            
            // Chenille (forme trapézoïdale)
            const trackShape = new THREE.Shape();
            trackShape.moveTo(-30, -10); // Arrière bas
            trackShape.lineTo(35, -10);  // Avant bas
            trackShape.lineTo(40, 12);   // Avant haut
            trackShape.lineTo(-25, 6);   // Arrière haut
            trackShape.lineTo(-30, -10);

            const trackExtrude = { depth: 16, bevelEnabled: true, bevelThickness: 1, bevelSize: 1, bevelSegments: 2 };
            const trackGeo = new THREE.ExtrudeGeometry(trackShape, trackExtrude);
            trackGeo.center();
            const tread = new THREE.Mesh(trackGeo, texturedRubberMat);
            tread.position.set(5, 15, zOffset);
            tread.castShadow = true; tread.receiveShadow = true;
            trackGroup.add(tread);

            // Roue avant (grande avec trous)
            const frontWheelZ = zOffset > 0 ? zOffset + 9 : zOffset - 9;
            const wheelFront = new THREE.Mesh(frontWheelGeo, plasticWhite);
            wheelFront.position.set(38, 16, frontWheelZ);
            wheelFront.castShadow = true; wheelFront.receiveShadow = true;
            
            // Roue arrière (plus petite)
            const backWheelGeo = new THREE.CylinderGeometry(10, 10, 6, 32);
            backWheelGeo.rotateX(Math.PI / 2);
            const wheelBack = new THREE.Mesh(backWheelGeo, plasticWhite);
            wheelBack.position.set(-20, 11, frontWheelZ);
            wheelBack.castShadow = true; wheelBack.receiveShadow = true;
            
            // Châssis métallique intérieur (remplit le vide de la chenille)
            const chassisGeo = new THREE.BoxGeometry(50, 14, 14);
            const chassisSide = new THREE.Mesh(chassisGeo, chassisMat);
            chassisSide.position.set(5, 15, zOffset);
            chassisSide.castShadow = true; chassisSide.receiveShadow = true;
            trackGroup.add(chassisSide);

            trackGroup.add(wheelFront, wheelBack);
            return trackGroup;
        }

        // Ajout des deux chenilles
        cozmoGroup.add(createTrack(35));
        cozmoGroup.add(createTrack(-35));

        // Corps central (Body) et Backpack
        const bodyGroup = new THREE.Group();
        
        // 1. Carénage principal blanc (profil adouci)
        const bodyShape = new THREE.Shape();
        bodyShape.moveTo(-25, 0);   // Arrière bas
        bodyShape.lineTo(10, 0);    // Avant bas
        bodyShape.bezierCurveTo(20, 0, 25, 10, 25, 20); // Courbe avant
        bodyShape.lineTo(15, 35);   // Haut avant
        bodyShape.bezierCurveTo(10, 40, -10, 42, -20, 35); // Courbe supérieure
        bodyShape.lineTo(-30, 20);  // Arrière haut
        bodyShape.bezierCurveTo(-35, 10, -30, 0, -25, 0); // Courbe arrière
        
        const bodyExtrude = { depth: 36, bevelEnabled: true, bevelSegments: 4, steps: 1, bevelSize: 2, bevelThickness: 2 };
        const mainBodyGeo = new THREE.ExtrudeGeometry(bodyShape, bodyExtrude);
        const mainBody = new THREE.Mesh(mainBodyGeo, plasticWhite);
        mainBody.position.set(-10, 15, -18); // Centrage sur Z (-36/2 = -18)
        mainBody.castShadow = true; mainBody.receiveShadow = true;
        bodyGroup.add(mainBody);

        // 2. Lignes rouges latérales
        const redLineGeo = new THREE.BoxGeometry(24, 3, 3);
        const redLineL = new THREE.Mesh(redLineGeo, plasticRed);
        redLineL.position.set(-15, 42, 18);
        redLineL.rotation.z = Math.PI / 12;
        redLineL.castShadow = true; redLineL.receiveShadow = true;
        
        const redLineR = new THREE.Mesh(redLineGeo, plasticRed);
        redLineR.position.set(-15, 42, -18);
        redLineR.rotation.z = Math.PI / 12;
        redLineR.castShadow = true; redLineR.receiveShadow = true;
        bodyGroup.add(redLineL, redLineR);

        // 3. Backpack (Sac à dos) supérieur avec LEDs
        const backpackGroup = new THREE.Group();
        
        // Base rouge du backpack
        const backpackGeo = new THREE.BoxGeometry(24, 6, 28);
        const backpackBase = new THREE.Mesh(backpackGeo, plasticRed);
        backpackBase.castShadow = true; backpackBase.receiveShadow = true;
        backpackGroup.add(backpackBase);
        
        // Partie supérieure noire du backpack
        const backpackTopGeo = new THREE.BoxGeometry(20, 3, 24);
        const backpackTop = new THREE.Mesh(backpackTopGeo, plasticBlack);
        backpackTop.position.set(0, 4, 0);
        backpackGroup.add(backpackTop);
        
        // LEDs de statut
        const ledGeo = new THREE.BoxGeometry(3, 1, 10); 
        const ledMat1 = new THREE.MeshStandardMaterial({ color: 0x0044ff, emissive: 0x0044ff, emissiveIntensity: 0.8 });
        const ledMat2 = new THREE.MeshStandardMaterial({ color: 0x00ff44, emissive: 0x00ff44, emissiveIntensity: 0.8 });
        const ledMat3 = new THREE.MeshStandardMaterial({ color: 0xff0044, emissive: 0xff0044, emissiveIntensity: 0.8 });
        
        const led1 = new THREE.Mesh(ledGeo, ledMat1);
        led1.position.set(-5, 6, 0);
        const led2 = new THREE.Mesh(ledGeo, ledMat2);
        led2.position.set(0, 6, 0);
        const led3 = new THREE.Mesh(ledGeo, ledMat3);
        led3.position.set(5, 6, 0);
        
        backpackGroup.add(led1, led2, led3);
        
        backpackGroup.position.set(-20, 53, 0);
        backpackGroup.rotation.z = -Math.PI / 16; // Inclinaison
        bodyGroup.add(backpackGroup);
        
        cozmoGroup.add(bodyGroup);

        // 4. Tête (Head) et Écran
        const headGroup = new THREE.Group();
        
        // Carénage blanc (Forme complexe profilée)
        const headShape = new THREE.Shape();
        headShape.moveTo(-15, -15);
        headShape.lineTo(10, -15);
        headShape.bezierCurveTo(18, -15, 20, -5, 18, 5); // Menton
        headShape.lineTo(10, 18); // Front fuyant
        headShape.bezierCurveTo(5, 22, -5, 22, -10, 20); // Sommet
        headShape.lineTo(-18, 5); // Arrière
        headShape.bezierCurveTo(-20, -5, -20, -15, -15, -15);
        
        const headExtrude = { depth: 40, bevelEnabled: true, bevelThickness: 3, bevelSize: 3, bevelSegments: 3 };
        const headGeo = new THREE.ExtrudeGeometry(headShape, headExtrude);
        headGeo.center();
        const headCasing = new THREE.Mesh(headGeo, plasticWhite);
        headCasing.castShadow = true; headCasing.receiveShadow = true;
        headGroup.add(headCasing);

        // Texture animable de l'écran (Faceplate)
        function createEyesTexture() {
            const canvas = document.createElement('canvas');
            canvas.width = 512;
            canvas.height = 256;
            const ctx = canvas.getContext('2d');
            
            // Fond noir de l'écran
            ctx.fillStyle = '#020202';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Style des yeux (Bleu lumineux)
            const eyeColor = '#00aaff';
            ctx.fillStyle = eyeColor;
            ctx.shadowColor = eyeColor;
            ctx.shadowBlur = 15; // Halo lumineux
            
            // Fonction utilitaire pour dessiner un rectangle aux coins arrondis
            function roundRect(ctx, x, y, width, height, radius) {
                ctx.beginPath();
                ctx.moveTo(x + radius, y);
                ctx.lineTo(x + width - radius, y);
                ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
                ctx.lineTo(x + width, y + height - radius);
                ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
                ctx.lineTo(x + radius, y + height);
                ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
                ctx.lineTo(x, y + radius);
                ctx.quadraticCurveTo(x, y, x + radius, y);
                ctx.closePath();
                ctx.fill();
            }
            
            // Dessin des deux yeux
            roundRect(ctx, 130, 60, 100, 130, 25); // Oeil Gauche
            roundRect(ctx, 282, 60, 100, 130, 25); // Oeil Droit
            
            // Effet Scanline (lignes horizontales semi-transparentes pour faire "écran")
            ctx.shadowBlur = 0; // Désactiver l'ombre pour les lignes
            ctx.fillStyle = 'rgba(0, 0, 0, 0.25)';
            for (let i = 0; i < canvas.height; i += 4) {
                ctx.fillRect(0, i, canvas.width, 2);
            }
            
            return new THREE.CanvasTexture(canvas);
        }
        
        const eyesTexture = createEyesTexture();
        const faceplateMat = new THREE.MeshStandardMaterial({ 
            color: 0x050505, // Presque noir
            emissiveMap: eyesTexture, 
            emissive: 0xffffff, // Autorise la couleur de la map
            emissiveIntensity: 0.9,
            roughness: 0.05, // Très lisse (vitre)
            metalness: 0.9
        });

        // Bloc noir de la tête (remplace l'écran incurvé)
        const screenBackGeo = new THREE.BoxGeometry(8, 28, 38);
        const screenBack = new THREE.Mesh(screenBackGeo, plasticBlack);
        screenBack.position.set(13, 11, 0); // Centré dans le trou du casque blanc
        screenBack.rotation.z = Math.PI / 6; // Inclinaison de 30° pour épouser le front
        
        // Le plan qui affiche les yeux, collé sur la face avant (+X) du bloc noir
        const screenPlaneGeo = new THREE.PlaneGeometry(38, 28);
        const screenPlane = new THREE.Mesh(screenPlaneGeo, faceplateMat);
        screenPlane.position.set(4.1, 0, 0); // Juste devant la face +X (largeur/2 = 4)
        screenPlane.rotation.y = Math.PI / 2; // Tourné vers l'avant (+X)
        
        // On attache l'écran au bloc noir
        screenBack.add(screenPlane);
        
        headGroup.add(screenBack);
        
        // Caméra Infrarouge (renfoncement et lentille)
        const cameraGroup = new THREE.Group();
        const camHoleGeo = new THREE.BoxGeometry(4, 3, 10);
        const camHole = new THREE.Mesh(camHoleGeo, plasticBlack);
        camHole.position.set(18, -12, 0);
        
        const lensGeo = new THREE.CylinderGeometry(1.5, 1.5, 1, 16);
        lensGeo.rotateZ(Math.PI / 2);
        const lensMat = new THREE.MeshStandardMaterial({ color: 0x111122, roughness: 0.0, metalness: 0.9 });
        const lens = new THREE.Mesh(lensGeo, lensMat);
        lens.position.set(19.5, -12, 0);
        
        cameraGroup.add(camHole, lens);
        headGroup.add(cameraGroup);
        
        // Position et inclinaison globale de la tête
        headGroup.position.set(16, 40, 0); 
        headGroup.rotation.z = Math.PI / 16; // Regarde légèrement vers le haut
        cozmoGroup.add(headGroup);

        // Bras de levage (Lift Arms)
        function createArm(zOffset) {
            const armRoot = new THREE.Group();
            // Point de pivot de l'épaule
            armRoot.position.set(-10, 36, 0);
            
            const armGroup = new THREE.Group();
            armGroup.rotation.z = Math.PI / 10; // Inclinaison globale du bras vers le bas

            const isLeft = zOffset > 0;
            const armZ = isLeft ? zOffset - 4 : zOffset + 4; 

            // 1. Structure principale du bras (squelette ajouré)
            const armShape = new THREE.Shape();
            
            // Dessin du contour autour du pivot (0,0)
            armShape.moveTo(-8, 8);    
            armShape.lineTo(45, 4);     // Vers le poignet haut
            armShape.lineTo(48, -4);    // Poignet bas
            armShape.lineTo(25, -4);    
            armShape.lineTo(15, -12);   // Décrochement sous le bras
            armShape.lineTo(-5, -12);  
            armShape.lineTo(-12, -5);   
            armShape.lineTo(-8, 8);    
            
            // Grand trou central (le "squelette")
            const hole1 = new THREE.Path();
            hole1.moveTo(0, 3);
            hole1.lineTo(38, 0);
            hole1.lineTo(38, -1);
            hole1.lineTo(22, -1);
            hole1.lineTo(12, -8);
            hole1.lineTo(0, -8);
            hole1.lineTo(0, 3);
            armShape.holes.push(hole1);
            
            const armExtrude = { depth: 4, bevelEnabled: true, bevelThickness: 0.5, bevelSize: 0.5, bevelSegments: 2 };
            const armGeo = new THREE.ExtrudeGeometry(armShape, armExtrude);
            // Centrage en Z uniquement pour l'épaisseur
            armGeo.translate(0, 0, -2); 
            
            const mainArm = new THREE.Mesh(armGeo, plasticWhite);
            mainArm.position.set(0, 0, armZ); 
            mainArm.castShadow = true; mainArm.receiveShadow = true;
            armGroup.add(mainArm);

            // 2. Le grand disque rouge (pivot de l'épaule)
            const jointZ = isLeft ? armZ + 2 : armZ - 2;
            const jointGeo = new THREE.CylinderGeometry(14, 14, 6, 32);
            jointGeo.rotateX(Math.PI / 2);
            const joint = new THREE.Mesh(jointGeo, plasticRed);
            joint.position.set(0, 0, jointZ);
            joint.castShadow = true; joint.receiveShadow = true;
            
            // Cercle blanc intérieur du pivot
            const jointInnerGeo = new THREE.CylinderGeometry(8, 8, 6.5, 32);
            jointInnerGeo.rotateX(Math.PI / 2);
            const jointInner = new THREE.Mesh(jointInnerGeo, plasticWhite);
            jointInner.position.set(0, 0, jointZ);
            armGroup.add(joint, jointInner);

            // 3. La fourche avant et son patin noir
            const forkGroup = new THREE.Group();
            
            // Bras vertical de la fourche
            const forkVertGeo = new THREE.BoxGeometry(8, 28, 8);
            const forkVert = new THREE.Mesh(forkVertGeo, plasticWhite);
            forkVert.position.set(0, -10, 0);
            forkVert.castShadow = true; forkVert.receiveShadow = true;
            
            // Dent horizontale de la fourche
            const forkHorizGeo = new THREE.BoxGeometry(18, 4, 8);
            const forkHoriz = new THREE.Mesh(forkHorizGeo, plasticWhite);
            forkHoriz.position.set(9, -22, 0);
            forkHoriz.castShadow = true; forkHoriz.receiveShadow = true;
            
            // Patin de grip en caoutchouc noir (dépasse un peu vers l'intérieur et le haut)
            const gripGeo = new THREE.BoxGeometry(16, 6, 9);
            const grip = new THREE.Mesh(gripGeo, plasticBlack);
            grip.position.set(9, -21, 0); 
            
            forkGroup.add(forkVert, forkHoriz, grip);
            
            // Positionnement de la fourche au bout du bras
            forkGroup.position.set(45, 0, armZ);
            // Compensation de l'inclinaison pour rester droit (horizontal par rapport au sol)
            forkGroup.rotation.z = -Math.PI / 10; 
            
            armGroup.add(forkGroup);
            armRoot.add(armGroup);

            return armRoot;
        }

        cozmoGroup.add(createArm(42));
        cozmoGroup.add(createArm(-42));
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

        // Follow cam (Smooth Gimbal Effect)
        if (followCam && cozmoGroup) {
            // Lerp pour un suivi fluide style drone au lieu d'une copie instantanée
            controls.target.lerp(cozmoGroup.position, 0.05);
        }

        controls.update();
        renderer.render(scene, camera);
    }

    function toggleMap() {
        overlay.classList.toggle('hidden');
        if (!overlay.classList.contains('hidden')) {
            if (!isInitialized) init();
            onWindowResize();
            
            // GSAP Animation: Effet d'entrée cinématique (Drone swooping in)
            if (window.gsap && camera) {
                const startX = cozmoGroup ? cozmoGroup.position.x : 0;
                const startZ = cozmoGroup ? cozmoGroup.position.z : 0;
                
                // On place la caméra très haut
                camera.position.set(startX, 2500, startZ + 2000);
                
                // On la fait descendre de façon fluide et dramatique
                gsap.to(camera.position, {
                    duration: 2.5,
                    y: 800,
                    z: startZ + 1000,
                    ease: "power3.out"
                });
            }
        }
    }

    if(btnOpen) btnOpen.addEventListener('click', toggleMap);
    if(btnClose) btnClose.addEventListener('click', toggleMap);

    function createParquetTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 2048; // Plus large pour de longues planches
        canvas.height = 1024;
        const ctx = canvas.getContext('2d');
        
        // Couleur de base bois classique
        ctx.fillStyle = '#8c603b';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        ctx.lineWidth = 4;
        ctx.strokeStyle = '#4a2b12'; // Interstices plus sombres
        
        // 1024 / 128 = exactement 8 planches sur la hauteur. Plus de planches coupées très fines à la jointure.
        const plankWidth = 128; 
        // 2048 / 1024 = exactement 2 planches sur la largeur. Fini les "mini lattes" ou carrés.
        const plankLength = 1024; 
        
        let lastOffset = 0;
        
        for (let y = 0; y <= canvas.height; y += plankWidth) {
            // Ligne horizontale
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(canvas.width, y);
            ctx.stroke();
            
            // On veut un décalage d'au moins 30% de la longueur de la planche par rapport à la ligne précédente
            // Cela garantit une esthétique parfaite "en coupe perdue" sans que deux joints ne s'alignent jamais
            let minShift = plankLength * 0.3;
            let maxShift = plankLength * 0.7;
            let shift = minShift + Math.random() * (maxShift - minShift);
            
            let xOffset = (lastOffset + shift) % plankLength;
            lastOffset = xOffset;
            
            // On dessine légèrement hors du canvas (-plankLength) pour garantir un raccordement sans couture aux bords
            for (let x = xOffset - plankLength; x <= canvas.width + plankLength; x += plankLength) {
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(x, y + plankWidth);
                ctx.stroke();
            }
            
            // Grain de bois subtil
            ctx.strokeStyle = 'rgba(0,0,0,0.06)';
            for(let i = 0; i < 15; i++) {
                ctx.beginPath();
                ctx.moveTo(0, y + Math.random() * plankWidth);
                ctx.lineTo(canvas.width, y + Math.random() * plankWidth);
                ctx.stroke();
            }
            ctx.strokeStyle = '#4a2b12';
        }
        
        const texture = new THREE.CanvasTexture(canvas);
        texture.wrapS = THREE.RepeatWrapping;
        texture.wrapT = THREE.RepeatWrapping;
        // Répétition calculée pour obtenir des lattes de ~12cm de large et ~120cm de long (ratio 1:10)
        texture.repeat.set(2.8, 7); 

        
        // Optionnel : Générer une texture de Normal Map basique à partir de ça pourrait être complexe en JS pur, 
        // on se contente du Diffuse et du shader Standard.
        return texture;
    }

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
                
                // Polaroid style frame (maintenant avec une très fine épaisseur pour plus de réalisme)
                const frameGeo = new THREE.BoxGeometry(110, 2, 130);
                const frameMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.8 });
                const frame = new THREE.Mesh(frameGeo, frameMat);
                frame.castShadow = true; frame.receiveShadow = true;
                
                const picGeo = new THREE.PlaneGeometry(90, 100);
                // Le côté brillant de la photo Polaroid
                const picMat = new THREE.MeshStandardMaterial({ 
                    map: texture, 
                    roughness: 0.1, 
                    metalness: 0.1 
                });
                const pic = new THREE.Mesh(picGeo, picMat);
                pic.position.set(0, 1.1, -5); // Décalé vers le haut (Y local de la box) et un peu vers le bord (pour le blanc du Polaroid)
                pic.rotation.x = -Math.PI / 2; // Couché à plat
                
                frame.add(pic);
                frame.userData = { src: src }; // For raycaster click
                
                // Position initiale : "éjectée" depuis Cozmo
                const rad = cap * Math.PI / 180;
                const startX = x;
                const startZ = -y;
                const startY = 80; // Hauteur d'éjection
                
                // Position d'arrivée : derrière Cozmo
                const dropDist = 100;
                let targetX = startX - dropDist * Math.cos(rad);
                let targetZ = startZ + dropDist * Math.sin(rad);
                
                // Dispersion aléatoire pour éviter le Z-fighting exact (et superposition)
                const randomOffset = 50;
                targetX += (Math.random() - 0.5) * randomOffset;
                targetZ += (Math.random() - 0.5) * randomOffset;
                
                // Hauteur d'arrivée avec un micro-décalage Y aléatoire (anti-clipping des photos entre elles)
                const targetY = 1 + Math.random() * 5; 
                
                // Rotation d'arrivée aléatoire pour un effet "pile de photos jetées"
                const targetRotY = rad + (Math.random() - 0.5) * Math.PI;

                frame.position.set(startX, startY, startZ);
                frame.rotation.set(-Math.PI / 4, rad, 0); // Elle sort en biais

                session.group.add(frame);
                clickablePhotos.push(frame);

                // Animation GSAP de la chute
                if (window.gsap) {
                    gsap.to(frame.position, {
                        duration: 1.2,
                        x: targetX,
                        z: targetZ,
                        ease: "power1.out"
                    });
                    
                    gsap.to(frame.position, {
                        duration: 1.2,
                        y: targetY,
                        ease: "bounce.out" // Effet de rebond sur le sol
                    });

                    gsap.to(frame.rotation, {
                        duration: 1.2,
                        x: 0, // A plat sur le sol
                        y: targetRotY,
                        z: 0,
                        ease: "power2.out"
                    });
                } else {
                    frame.position.set(targetX, targetY, targetZ);
                    frame.rotation.set(0, targetRotY, 0);
                }
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
