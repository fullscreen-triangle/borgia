/**
 * MoleculeViewer3D — GLB molecular model with harmonic loop overlay
 * =================================================================
 * Loads a GLB molecule, extracts mesh normals for shape entropy,
 * and overlays loop membership as vertex colours.
 *
 * When no GLB is available, shows a procedural sphere placeholder.
 */
import { Suspense, useRef, useMemo, useState, useEffect, useCallback } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useGLTF, OrbitControls, Environment } from "@react-three/drei";
import * as THREE from "three";
import { computeShapeEntropy } from "@/lib/sentropy";

function GLBMolecule({ url, loopNodes = new Set(), onNormalsExtracted }) {
  const ref = useRef();
  const { scene } = useGLTF(url);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (!scene || loaded) return;
    // Extract normals from all meshes for shape entropy
    const normals = [];
    scene.traverse((child) => {
      if (child.isMesh && child.geometry) {
        const normalAttr = child.geometry.getAttribute("normal");
        if (normalAttr) {
          for (let i = 0; i < normalAttr.count; i++) {
            normals.push({
              x: normalAttr.getX(i),
              y: normalAttr.getY(i),
              z: normalAttr.getZ(i),
            });
          }
        }
        // Add vertex colours based on loop membership (if applicable)
        // For now, tint the material
        if (child.material) {
          child.material = child.material.clone();
          child.material.transparent = true;
          child.material.opacity = 0.9;
        }
      }
    });

    if (onNormalsExtracted && normals.length > 0) {
      onNormalsExtracted(normals);
    }
    setLoaded(true);
  }, [scene, loaded, loopNodes, onNormalsExtracted]);

  useFrame((state) => {
    if (ref.current) {
      ref.current.rotation.y = state.clock.elapsedTime * 0.1;
    }
  });

  return (
    <group ref={ref} scale={1.5}>
      <primitive object={scene} />
    </group>
  );
}

function PlaceholderMolecule({ modes = [], loopNodes = new Set() }) {
  const ref = useRef();

  // Create procedural representation: spheres for atoms arranged by mode count
  const atoms = useMemo(() => {
    const N = Math.max(modes.length, 3);
    const positions = [];
    for (let i = 0; i < N; i++) {
      const angle = (i / N) * Math.PI * 2;
      const r = 0.8 + (modes[i] || 1000) / 8000;
      positions.push({
        pos: [Math.cos(angle) * r, Math.sin(angle) * r, (Math.random() - 0.5) * 0.4],
        size: 0.15 + (modes[i] || 1000) / 20000,
        isLoop: loopNodes.has(i),
      });
    }
    return positions;
  }, [modes, loopNodes]);

  useFrame((state) => {
    if (ref.current) {
      ref.current.rotation.y = state.clock.elapsedTime * 0.15;
      ref.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.3) * 0.1;
    }
  });

  return (
    <group ref={ref}>
      {atoms.map((atom, i) => (
        <mesh key={i} position={atom.pos}>
          <sphereGeometry args={[atom.size, 16, 16]} />
          <meshStandardMaterial
            color={atom.isLoop ? "#58E6D9" : "#3b82f6"}
            emissive={atom.isLoop ? "#58E6D9" : "#000000"}
            emissiveIntensity={atom.isLoop ? 0.3 : 0}
            metalness={0.3}
            roughness={0.6}
          />
        </mesh>
      ))}
      {/* Bonds between adjacent atoms */}
      {atoms.slice(0, -1).map((atom, i) => {
        const next = atoms[i + 1];
        const start = new THREE.Vector3(...atom.pos);
        const end = new THREE.Vector3(...next.pos);
        const mid = start.clone().add(end).multiplyScalar(0.5);
        const dir = end.clone().sub(start);
        const len = dir.length();
        return (
          <mesh key={`bond-${i}`} position={[mid.x, mid.y, mid.z]}>
            <cylinderGeometry args={[0.03, 0.03, len, 8]} />
            <meshStandardMaterial color="#666" />
          </mesh>
        );
      })}
    </group>
  );
}

export default function MoleculeViewer3D({
  mol,
  width = 400,
  height = 350,
  onShapeEntropy,
}) {
  const [glbAvailable, setGlbAvailable] = useState(false);
  const [checking, setChecking] = useState(true);

  // Check if GLB exists
  useEffect(() => {
    if (!mol?.glbPath) {
      setGlbAvailable(false);
      setChecking(false);
      return;
    }
    fetch(mol.glbPath, { method: "HEAD" })
      .then((res) => {
        setGlbAvailable(res.ok);
        setChecking(false);
      })
      .catch(() => {
        setGlbAvailable(false);
        setChecking(false);
      });
  }, [mol?.glbPath]);

  // Loop node set for highlighting
  const loopNodes = useMemo(() => {
    const s = new Set();
    if (mol?.loopFP?.loops) {
      mol.loopFP.loops.forEach((l) => l.nodes.forEach((n) => s.add(n)));
    }
    return s;
  }, [mol?.loopFP]);

  const handleNormals = useCallback(
    (normals) => {
      if (onShapeEntropy) {
        const Sg = computeShapeEntropy(normals);
        onShapeEntropy(Sg);
      }
    },
    [onShapeEntropy]
  );

  if (!mol) return null;

  return (
    <div
      className="rounded-lg border border-neutral-800 overflow-hidden bg-[#0d0d0d]"
      style={{ width, height }}
    >
      <Canvas camera={{ position: [0, 0, 4], fov: 40 }}>
        <ambientLight intensity={0.4} />
        <directionalLight position={[5, 5, 5]} intensity={0.8} />
        <pointLight position={[-3, -3, 2]} intensity={0.3} color="#58E6D9" />
        <Suspense fallback={null}>
          {glbAvailable && !checking ? (
            <GLBMolecule
              url={mol.glbPath}
              loopNodes={loopNodes}
              onNormalsExtracted={handleNormals}
            />
          ) : (
            <PlaceholderMolecule modes={mol.modes} loopNodes={loopNodes} />
          )}
        </Suspense>
        <OrbitControls enableZoom enablePan={false} autoRotate={false} />
      </Canvas>
      {/* Overlay label */}
      <div className="relative -mt-8 ml-2 mb-2">
        <span className="text-xs text-neutral-500 bg-black/60 px-2 py-0.5 rounded">
          {glbAvailable ? "GLB" : "Procedural"} · {mol.name}
        </span>
      </div>
    </div>
  );
}
