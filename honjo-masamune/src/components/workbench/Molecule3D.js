/**
 * Molecule3D — the compound a program built, in three dimensions.
 *
 * MoleculeViewer3D loads a GLB. That is the wrong input here: nothing in this
 * workbench has a mesh file, because nothing looked the molecule up. `close
 * O(H,H)` derives geometry="bent" and angleDeg=104.5 from vacancy counting and
 * domain repulsion, and those two numbers plus a ligand count are a complete
 * specification of where the atoms go. So the geometry is *constructed* from
 * the run's own bindings rather than fetched.
 *
 * The consequence worth seeing: change the program and the shape changes,
 * because the shape was computed. A tabulated viewer cannot show that.
 *
 * Placement, per derived geometry:
 *   terminal      one ligand on +x
 *   linear        two ligands at ±x
 *   bent          two ligands in the yz-plane, separated by angleDeg
 *   trigonal      three ligands at 120 deg, tilted off the spin axis
 *   pyramidal     three ligands on a cone whose apex angle gives angleDeg
 *   tetrahedral   four ligands on alternate cube vertices
 */

import { Suspense, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

const T = {
  bg: "#1a1b26", panel: "#24253a", border: "#2f3146",
  text: "#c0caf5", dim: "#565f89", accent: "#7dcfff", ok: "#9ece6a",
};
const MONO = "'JetBrains Mono','Fira Code','SF Mono',Consolas,monospace";

/* Element colours: CPK where the element is common, a neutral otherwise. */
const CPK = {
  H: "#ffffff", C: "#3b3d57", N: "#3050f8", O: "#ff0d0d", F: "#90e050",
  Cl: "#1ff01f", Br: "#a62929", I: "#940094", S: "#ffff30", P: "#ff8000",
  B: "#ffb5b5", Si: "#f0c8a0", Na: "#ab5cf2", K: "#8f40d4", Mg: "#8aff00",
  Ca: "#3dff00", Li: "#cc80ff", Be: "#c2ff00", He: "#d9ffff", Ne: "#b3e3f5",
  Ar: "#80d1e3", Se: "#ffa100", Al: "#bfa6a6",
};
const RADIUS = {
  H: 0.30, He: 0.30, Li: 0.62, Be: 0.54, B: 0.50, C: 0.48, N: 0.46,
  O: 0.44, F: 0.42, Ne: 0.40, Na: 0.70, Mg: 0.64, Al: 0.60, Si: 0.58,
  P: 0.56, S: 0.54, Cl: 0.52, Ar: 0.50,
};
const colourOf = (s) => CPK[s] || "#9aa5ce";
const radiusOf = (s) => RADIUS[s] || 0.52;

const DEG = Math.PI / 180;

/**
 * Ligand positions in the central atom's frame, from the derived geometry.
 *
 * The angle is used, not assumed: a bent molecule with angleDeg=104.5 opens by
 * exactly 104.5 degrees, and a pyramidal one places its cone so the
 * ligand-central-ligand angle comes out at the derived value. That is the
 * difference between drawing the geometry and drawing its name.
 */
export function ligandPositions(geometry, angleDeg, n, bondLength = 1.6) {
  const P = [];
  const L = bondLength;
  const g = String(geometry || "").toLowerCase();

  if (g === "terminal" || n === 1) {
    P.push([L, 0, 0]);
  } else if (g === "linear") {
    P.push([L, 0, 0], [-L, 0, 0]);
  } else if (g === "bent") {
    // Opened about +y in the yz-plane, not the xy-plane.  The assembly spins
    // about y and the camera looks down -z, so a molecule lying in xy is
    // swept edge-on and its angle collapses to a straight line -- the one
    // reading this view exists to make visible.  In yz the opening is
    // subtended at every phase of the spin.
    const a = (angleDeg ?? 104.5) * DEG;
    P.push(
      [0, L * Math.cos(a / 2), L * Math.sin(a / 2)],
      [0, L * Math.cos(a / 2), -L * Math.sin(a / 2)],
    );
  } else if (g === "trigonal") {
    // Planar by derivation, so it must stay planar; tilted off the spin axis
    // for the same reason as bent, rather than laid flat where the spin
    // would edge it on.
    const TILT = 24 * DEG;
    for (let i = 0; i < 3; i++) {
      const t = i * 120 * DEG;
      const y0 = L * Math.sin(t);
      P.push([
        L * Math.cos(t),
        y0 * Math.cos(TILT),
        y0 * Math.sin(TILT),
      ]);
    }
  } else if (g === "pyramidal") {
    // Half-angle theta from the apex such that the ligand-central-ligand
    // angle equals angleDeg:  cos(a) = cos^2(t) + sin^2(t)*cos(120deg).
    const a = (angleDeg ?? 107) * DEG;
    const c = Math.max(-1, Math.min(1, (Math.cos(a) + 0.5) / 1.5));
    const t = Math.acos(Math.sqrt(Math.max(0, c)));
    for (let i = 0; i < 3; i++) {
      const p = i * 120 * DEG;
      P.push([
        L * Math.sin(t) * Math.cos(p),
        L * Math.cos(t),
        L * Math.sin(t) * Math.sin(p),
      ]);
    }
  } else if (g === "tetrahedral") {
    const k = L / Math.sqrt(3);
    P.push([k, k, k], [k, -k, -k], [-k, k, -k], [-k, -k, k]);
  } else {
    // Unknown geometry: spread evenly on a sphere rather than invent a shape.
    for (let i = 0; i < n; i++) {
      const y = 1 - (2 * i) / Math.max(1, n - 1);
      const r = Math.sqrt(Math.max(0, 1 - y * y));
      const t = i * Math.PI * (3 - Math.sqrt(5));
      P.push([L * r * Math.cos(t), L * y, L * r * Math.sin(t)]);
    }
  }
  return P.slice(0, Math.max(1, n));
}

function Bond({ from, to, colour }) {
  const { pos, quat, len } = useMemo(() => {
    const a = new THREE.Vector3(...from), b = new THREE.Vector3(...to);
    const d = new THREE.Vector3().subVectors(b, a);
    const q = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 1, 0), d.clone().normalize());
    return {
      pos: new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5),
      quat: q, len: d.length(),
    };
  }, [from, to]);
  return (
    <mesh position={pos} quaternion={quat}>
      <cylinderGeometry args={[0.085, 0.085, len, 16]} />
      <meshStandardMaterial color={colour} roughness={0.45} metalness={0.15} />
    </mesh>
  );
}

function Atom({ position, symbol, radius, onHover }) {
  const [over, setOver] = useState(false);
  return (
    <mesh
      position={position}
      onPointerOver={(e) => { e.stopPropagation(); setOver(true); onHover?.(symbol); }}
      onPointerOut={() => { setOver(false); onHover?.(null); }}
    >
      <sphereGeometry args={[radius * (over ? 1.12 : 1), 32, 32]} />
      <meshStandardMaterial
        color={colourOf(symbol)}
        roughness={0.28} metalness={0.2}
        emissive={colourOf(symbol)} emissiveIntensity={over ? 0.35 : 0.06}
      />
    </mesh>
  );
}

/** The angle between two bonds, drawn as an arc — the derived number, shown. */
function AngleArc({ a, b, angleDeg }) {
  const pts = useMemo(() => {
    if (angleDeg == null) return null;
    const va = new THREE.Vector3(...a).normalize();
    const vb = new THREE.Vector3(...b).normalize();
    const r = 0.62, out = [];
    for (let i = 0; i <= 40; i++) {
      const t = i / 40;
      const v = new THREE.Vector3().copy(va).lerp(vb, t).normalize()
        .multiplyScalar(r);
      out.push(v);
    }
    return out;
  }, [a, b, angleDeg]);
  const geo = useMemo(
    () => (pts ? new THREE.BufferGeometry().setFromPoints(pts) : null), [pts]);
  if (!geo) return null;
  return (
    <line geometry={geo}>
      <lineBasicMaterial color={T.accent} transparent opacity={0.85} />
    </line>
  );
}

function Assembly({ central, ligands, geometry, angleDeg, spin, onHover }) {
  const g = useRef();
  useFrame((_, dt) => { if (spin && g.current) g.current.rotation.y += dt * 0.28; });

  const pos = useMemo(
    () => ligandPositions(geometry, angleDeg, ligands.length),
    [geometry, angleDeg, ligands.length]);

  return (
    <group ref={g}>
      <Atom position={[0, 0, 0]} symbol={central} radius={radiusOf(central)}
            onHover={onHover} />
      {pos.map((p, i) => (
        <group key={i}>
          <Bond from={[0, 0, 0]} to={p} colour="#6b7194" />
          <Atom position={p} symbol={ligands[i]} radius={radiusOf(ligands[i])}
                onHover={onHover} />
        </group>
      ))}
      {pos.length >= 2 && (
        <AngleArc a={pos[0]} b={pos[1]} angleDeg={angleDeg} />
      )}
    </group>
  );
}

/**
 * One compound. `compound` is a Compound value straight out of `res.named`.
 */
export default function Molecule3D({ compound, height = 260, spin = true }) {
  const [hover, setHover] = useState(null);
  const { central, ligand, ligands: nLig, geometry, angleDeg, formula } =
    compound || {};

  const ligandList = useMemo(() => {
    const n = Number(nLig) || 0;
    // The homonuclear case binds formula=[2,0] with one "ligand": the second
    // atom is another copy of the central, not a different element.
    if (Array.isArray(formula) && formula[1] === 0 && formula[0] === 2)
      return [central];
    return Array(Math.max(0, n)).fill(ligand);
  }, [central, ligand, nLig, formula]);

  if (!central || !ligandList.length) return null;

  const label = Array.isArray(formula)
    ? (formula[1] === 0
      ? `${central}${formula[0] > 1 ? formula[0] : ""}`
      : `${central}${formula[0] > 1 ? formula[0] : ""}${ligand}${formula[1] > 1 ? formula[1] : ""}`)
    : `${central}${ligand}${nLig}`;

  return (
    <div style={{
      border: `1px solid ${T.border}`, borderRadius: 4,
      background: T.bg, overflow: "hidden",
    }}>
      <div style={{ height, position: "relative" }}>
        <Canvas camera={{ position: [0, 1.4, 4.6], fov: 42 }} dpr={[1, 2]}>
          <color attach="background" args={[T.bg]} />
          <ambientLight intensity={0.55} />
          <directionalLight position={[4, 6, 5]} intensity={1.15} />
          <directionalLight position={[-5, -2, -4]} intensity={0.35}
                            color="#7dcfff" />
          <Suspense fallback={null}>
            <Assembly central={central} ligands={ligandList}
                      geometry={geometry} angleDeg={angleDeg}
                      spin={spin} onHover={setHover} />
          </Suspense>
          <OrbitControls enablePan={false} minDistance={2.6} maxDistance={11}
                         enableDamping dampingFactor={0.08} />
        </Canvas>
        <div style={{
          position: "absolute", left: 9, top: 8, pointerEvents: "none",
          fontFamily: MONO, fontSize: 11.5, color: T.text,
        }}>
          {label}
          <span style={{ color: T.dim }}>
            {"  "}{geometry}
            {angleDeg != null ? ` · ${angleDeg}°` : ""}
          </span>
        </div>
        {hover && (
          <div style={{
            position: "absolute", right: 9, top: 8, pointerEvents: "none",
            fontFamily: MONO, fontSize: 11, color: T.accent,
          }}>{hover}</div>
        )}
        <div style={{
          position: "absolute", left: 9, bottom: 7, pointerEvents: "none",
          fontFamily: MONO, fontSize: 9.5, color: T.dim,
        }}>
          geometry derived by the run · drag to orbit
        </div>
      </div>
    </div>
  );
}
