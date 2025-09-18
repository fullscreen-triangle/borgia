// 3D Molecular Property Space using Three.js
function create3DPropertySpace(data) {
    const container = document.getElementById('property-3d');
    const width = container.clientWidth;
    const height = 500;

    // Scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setSize(width, height);
    renderer.setClearColor(0xf8f9fa);
    container.appendChild(renderer.domElement);

    // Extract molecular data
    const molecules = data.molecules.slice(0, 50); // Limit for performance
    
    // Normalize properties for 3D coordinates
    const logpExtent = d3.extent(molecules, d => d.logp);
    const tpsaExtent = d3.extent(molecules, d => d.tpsa);
    const mwExtent = d3.extent(molecules, d => d.molecular_weight);
    const freqExtent = d3.extent(molecules, d => d.clock_properties.base_frequency_hz);

    // Create points
    const geometry = new THREE.BufferGeometry();
    const positions = [];
    const colors = [];
    const sizes = [];

    molecules.forEach(mol => {
        // Position based on LogP, TPSA, MW
        const x = ((mol.logp - logpExtent[0]) / (logpExtent[1] - logpExtent[0]) - 0.5) * 10;
        const y = ((mol.tpsa - tpsaExtent[0]) / (tpsaExtent[1] - tpsaExtent[0]) - 0.5) * 10;
        const z = ((mol.molecular_weight - mwExtent[0]) / (mwExtent[1] - mwExtent[0]) - 0.5) * 10;
        
        positions.push(x, y, z);
        
        // Color based on frequency
        const freqNorm = (mol.clock_properties.base_frequency_hz - freqExtent[0]) / (freqExtent[1] - freqExtent[0]);
        const color = new THREE.Color();
        color.setHSL(0.7 * (1 - freqNorm), 0.8, 0.5);
        colors.push(color.r, color.g, color.b);
        
        // Size based on processing rate
        sizes.push(Math.log(mol.processor_properties.processing_rate_ops_per_sec) / 10);
    });

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));

    // Shader material for variable point sizes
    const material = new THREE.ShaderMaterial({
        uniforms: {
            pointTexture: { value: new THREE.TextureLoader().load('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==') }
        },
        vertexShader: `
            attribute float size;
            varying vec3 vColor;
            void main() {
                vColor = color;
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                gl_PointSize = size * (300.0 / -mvPosition.z);
                gl_Position = projectionMatrix * mvPosition;
            }
        `,
        fragmentShader: `
            varying vec3 vColor;
            void main() {
                float r = distance(gl_PointCoord, vec2(0.5, 0.5));
                if (r > 0.5) discard;
                gl_FragColor = vec4(vColor, 1.0 - r * 2.0);
            }
        `,
        vertexColors: true,
        transparent: true
    });

    const points = new THREE.Points(geometry, material);
    scene.add(points);

    // Add axes
    const axesHelper = new THREE.AxesHelper(6);
    scene.add(axesHelper);

    // Add axis labels (simplified)
    const loader = new THREE.FontLoader();
    // Note: In real implementation, you'd load a font and add text geometry

    // Camera position
    camera.position.set(8, 8, 8);
    camera.lookAt(0, 0, 0);

    // Controls (simplified mouse interaction)
    let mouseX = 0, mouseY = 0;
    let isMouseDown = false;

    container.addEventListener('mousedown', (event) => {
        isMouseDown = true;
        mouseX = event.clientX;
        mouseY = event.clientY;
    });

    container.addEventListener('mouseup', () => {
        isMouseDown = false;
    });

    container.addEventListener('mousemove', (event) => {
        if (!isMouseDown) return;
        
        const deltaX = event.clientX - mouseX;
        const deltaY = event.clientY - mouseY;
        
        const spherical = new THREE.Spherical();
        spherical.setFromVector3(camera.position);
        spherical.theta -= deltaX * 0.01;
        spherical.phi += deltaY * 0.01;
        spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
        
        camera.position.setFromSpherical(spherical);
        camera.lookAt(0, 0, 0);
        
        mouseX = event.clientX;
        mouseY = event.clientY;
    });

    // Zoom with mouse wheel
    container.addEventListener('wheel', (event) => {
        const scale = event.deltaY > 0 ? 1.1 : 0.9;
        camera.position.multiplyScalar(scale);
    });

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
    }
    animate();

    // Store renderer for export
    container._threeRenderer = renderer;
    container._threeScene = scene;
    container._threeCamera = camera;
}

// 3D Quantum Surface Plot using Plotly
function create3DQuantumSurface(data) {
    const container = document.getElementById('quantum-surface');
    
    // Extract quantum state data
    const measurement = data.quantum_scale.measurements[0];
    const timePoints = measurement.time_femtoseconds.slice(0, 100); // Limit for performance
    const quantumStates = measurement.quantum_states.slice(0, 100);
    
    // Create 2D grid for surface
    const gridSize = 10;
    const x = [];
    const y = [];
    const z = [];
    
    for (let i = 0; i < gridSize; i++) {
        const row = [];
        for (let j = 0; j < gridSize; j++) {
            const timeIndex = Math.floor((i * gridSize + j) * timePoints.length / (gridSize * gridSize));
            if (timeIndex < quantumStates.length) {
                row.push(quantumStates[timeIndex] + Math.sin(i * 0.5) * Math.cos(j * 0.5) * 0.1);
            } else {
                row.push(0);
            }
        }
        z.push(row);
    }
    
    // Generate x and y coordinates
    for (let i = 0; i < gridSize; i++) {
        x.push(i);
        y.push(i);
    }
    
    const trace = {
        x: x,
        y: y,
        z: z,
        type: 'surface',
        colorscale: 'Viridis',
        showscale: true,
        colorbar: {
            title: 'Quantum State Value',
            titleside: 'right'
        }
    };
    
    const layout = {
        title: {
            text: '3D Quantum State Surface',
            font: { size: 16 }
        },
        scene: {
            xaxis: { title: 'Spatial Dimension X' },
            yaxis: { title: 'Spatial Dimension Y' },
            zaxis: { title: 'Quantum State Value' },
            camera: {
                eye: { x: 1.5, y: 1.5, z: 1.5 }
            }
        },
        margin: { l: 0, r: 0, b: 0, t: 50 }
    };
    
    const config = {
        displayModeBar: true,
        modeBarButtonsToAdd: [{
            name: 'Export SVG',
            icon: Plotly.Icons.camera,
            click: function(gd) {
                Plotly.downloadImage(gd, {
                    format: 'svg',
                    width: 800,
                    height: 600,
                    filename: 'quantum_surface_3d'
                });
            }
        }]
    };
    
    Plotly.newPlot(container, [trace], layout, config);
}
