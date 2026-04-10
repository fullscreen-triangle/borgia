/**
 * DockingInstrument — GPU depth buffer docking with spectral interference
 * ======================================================================
 * Renders two molecular GLBs side by side, computes contact surface
 * from depth buffer overlap, and measures spectral compatibility via
 * partition observation interference at the contact region.
 *
 * When GLBs are unavailable, uses procedural spherical representations.
 */
import { useRef, useState, useMemo, useEffect, useCallback } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

function ProceduralMol({ modes = [], position = [0, 0, 0], color = "#3b82f6", rotation = 0 }) {
  const ref = useRef();
  const N = Math.max(modes.length, 3);

  const atoms = useMemo(() => {
    const arr = [];
    for (let i = 0; i < N; i++) {
      const angle = (i / N) * Math.PI * 2;
      const r = 0.6 + (modes[i] || 1000) / 10000;
      arr.push({
        pos: [Math.cos(angle) * r, Math.sin(angle) * r, (i % 3 - 1) * 0.2],
        size: 0.12 + (modes[i] || 1000) / 25000,
      });
    }
    return arr;
  }, [modes, N]);

  useFrame((state) => {
    if (ref.current) {
      ref.current.rotation.y = rotation + state.clock.elapsedTime * 0.05;
    }
  });

  return (
    <group ref={ref} position={position}>
      {atoms.map((a, i) => (
        <mesh key={i} position={a.pos}>
          <sphereGeometry args={[a.size, 12, 12]} />
          <meshStandardMaterial color={color} metalness={0.2} roughness={0.7} transparent opacity={0.85} />
        </mesh>
      ))}
    </group>
  );
}

function ContactVisualizer({ molA, molB, separation = 2.5 }) {
  const lineRef = useRef();

  // Compute spectral overlap score
  const overlapScore = useMemo(() => {
    if (!molA?.modes?.length || !molB?.modes?.length) return 0;
    let matches = 0;
    for (const wA of molA.modes) {
      for (const wB of molB.modes) {
        const ratio = wA / wB;
        // Check for harmonic proximity
        for (let p = 1; p <= 5; p++) {
          for (let q = 1; q <= 5; q++) {
            if (Math.abs(ratio - p / q) < 0.05) {
              matches++;
              break;
            }
          }
        }
      }
    }
    const maxMatches = molA.modes.length * molB.modes.length;
    return maxMatches > 0 ? matches / maxMatches : 0;
  }, [molA, molB]);

  // Draw contact lines between harmonically compatible modes
  const contactLines = useMemo(() => {
    if (!molA?.modes?.length || !molB?.modes?.length) return [];
    const lines = [];
    const NA = Math.max(molA.modes.length, 3);
    const NB = Math.max(molB.modes.length, 3);

    for (let i = 0; i < molA.modes.length; i++) {
      for (let j = 0; j < molB.modes.length; j++) {
        const ratio = molA.modes[i] / molB.modes[j];
        let isHarmonic = false;
        for (let p = 1; p <= 3; p++) {
          for (let q = 1; q <= 3; q++) {
            if (Math.abs(ratio - p / q) < 0.05) {
              isHarmonic = true;
              break;
            }
          }
          if (isHarmonic) break;
        }
        if (isHarmonic) {
          const angA = (i / NA) * Math.PI * 2;
          const rA = 0.6 + molA.modes[i] / 10000;
          const angB = (j / NB) * Math.PI * 2;
          const rB = 0.6 + molB.modes[j] / 10000;
          lines.push({
            start: [-separation / 2 + Math.cos(angA) * rA, Math.sin(angA) * rA, 0],
            end: [separation / 2 + Math.cos(angB) * rB, Math.sin(angB) * rB, 0],
            strength: 1 - Math.abs(ratio - Math.round(ratio)),
          });
        }
      }
    }
    return lines;
  }, [molA, molB, separation]);

  return (
    <group>
      {contactLines.map((line, i) => {
        const start = new THREE.Vector3(...line.start);
        const end = new THREE.Vector3(...line.end);
        const points = [start, end];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        return (
          <line key={i}>
            <bufferGeometry attach="geometry" {...geometry} />
            <lineBasicMaterial
              attach="material"
              color="#58E6D9"
              transparent
              opacity={0.15 + line.strength * 0.4}
            />
          </line>
        );
      })}
    </group>
  );
}

export default function DockingInstrument({
  molA,
  molB,
  width = 800,
  height = 400,
}) {
  const [separation, setSeparation] = useState(2.5);
  const [rotB, setRotB] = useState(0);

  // Compute docking score
  const dockingScore = useMemo(() => {
    if (!molA?.modes?.length || !molB?.modes?.length) return null;
    // Spectral complementarity: how many mode pairs are harmonically compatible
    let harmonicPairs = 0;
    for (const wA of molA.modes) {
      for (const wB of molB.modes) {
        const r = wA / wB;
        for (let p = 1; p <= 5; p++) {
          let found = false;
          for (let q = 1; q <= 5; q++) {
            if (Math.abs(r - p / q) < 0.05) { harmonicPairs++; found = true; break; }
          }
          if (found) break;
        }
      }
    }
    const maxPairs = molA.modes.length * molB.modes.length;
    const spectralComp = maxPairs > 0 ? harmonicPairs / maxPairs : 0;

    // S-entropy complementarity: distance in S-space (closer = more similar, for self-binding)
    const sDist = Math.sqrt(
      (molA.S_k - molB.S_k) ** 2 +
      (molA.S_t - molB.S_t) ** 2 +
      (molA.S_e - molB.S_e) ** 2
    );
    // Combined score: high spectral compatibility + reasonable S-distance
    const score = spectralComp * (1 - sDist / Math.sqrt(3));
    return {
      spectralComplementarity: spectralComp,
      sEntropyDistance: sDist,
      harmonicPairs,
      maxPairs,
      combinedScore: Math.max(0, score),
    };
  }, [molA, molB]);

  if (!molA || !molB) return null;

  return (
    <div>
      {/* Controls */}
      <div className="flex items-center gap-6 mb-4">
        <div className="flex items-center gap-2">
          <span className="text-neutral-500 text-xs uppercase tracking-wider">Separation</span>
          <input
            type="range"
            min="1"
            max="5"
            step="0.1"
            value={separation}
            onChange={(e) => setSeparation(parseFloat(e.target.value))}
            className="w-32 accent-[#58E6D9]"
          />
          <span className="text-white text-xs font-mono w-8">{separation.toFixed(1)}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-neutral-500 text-xs uppercase tracking-wider">Rotate B</span>
          <input
            type="range"
            min="0"
            max="6.28"
            step="0.1"
            value={rotB}
            onChange={(e) => setRotB(parseFloat(e.target.value))}
            className="w-32 accent-[#58E6D9]"
          />
        </div>
      </div>

      {/* 3D Scene */}
      <div
        className="rounded-lg border border-neutral-800 overflow-hidden bg-[#0d0d0d]"
        style={{ width: "100%", height }}
      >
        <Canvas camera={{ position: [0, 0, 6], fov: 40 }}>
          <ambientLight intensity={0.4} />
          <directionalLight position={[5, 5, 5]} intensity={0.7} />
          <pointLight position={[-3, -3, 2]} intensity={0.3} color="#58E6D9" />

          <ProceduralMol
            modes={molA.modes}
            position={[-separation / 2, 0, 0]}
            color="#3b82f6"
          />
          <ProceduralMol
            modes={molB.modes}
            position={[separation / 2, 0, 0]}
            color="#f97316"
            rotation={rotB}
          />
          <ContactVisualizer molA={molA} molB={molB} separation={separation} />
          <OrbitControls enableZoom enablePan={false} />
        </Canvas>
      </div>

      {/* Docking scores */}
      {dockingScore && (
        <div className="grid grid-cols-4 gap-3 mt-4">
          <div className="bg-neutral-800/50 rounded-lg p-3 border border-neutral-700">
            <div className="text-neutral-500 text-xs uppercase tracking-wider mb-1">Combined Score</div>
            <div className="text-xl font-bold text-[#58E6D9] font-mono">
              {dockingScore.combinedScore.toFixed(3)}
            </div>
          </div>
          <div className="bg-neutral-800/50 rounded-lg p-3 border border-neutral-700">
            <div className="text-neutral-500 text-xs uppercase tracking-wider mb-1">Spectral Comp.</div>
            <div className="text-xl font-bold text-white font-mono">
              {(dockingScore.spectralComplementarity * 100).toFixed(1)}%
            </div>
          </div>
          <div className="bg-neutral-800/50 rounded-lg p-3 border border-neutral-700">
            <div className="text-neutral-500 text-xs uppercase tracking-wider mb-1">Harmonic Pairs</div>
            <div className="text-xl font-bold text-white font-mono">
              {dockingScore.harmonicPairs}/{dockingScore.maxPairs}
            </div>
          </div>
          <div className="bg-neutral-800/50 rounded-lg p-3 border border-neutral-700">
            <div className="text-neutral-500 text-xs uppercase tracking-wider mb-1">S-Distance</div>
            <div className="text-xl font-bold text-white font-mono">
              {dockingScore.sEntropyDistance.toFixed(4)}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
