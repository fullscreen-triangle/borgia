import { Suspense, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { useGLTF, ContactShadows } from "@react-three/drei";

function BohrModel(props) {
  const ref = useRef();
  const { scene } = useGLTF("/models/nitrogen_bohr_model_7p_7n_7e.glb");

  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    ref.current.rotation.y = t * 0.15;
    ref.current.rotation.x = Math.sin(t * 0.3) * 0.05;
    ref.current.position.y = Math.sin(t * 0.5) * 0.03;
  });

  return (
    <group ref={ref} {...props}>
      <primitive object={scene} />
    </group>
  );
}

useGLTF.preload("/models/nitrogen_bohr_model_7p_7n_7e.glb");

export default function LandingScene() {
  return (
    <Canvas
      shadows
      dpr={[1, 2]}
      camera={{ position: [0, 0, 4], fov: 45 }}
      style={{ background: "transparent" }}
    >
      <ambientLight intensity={0.4} />
      <spotLight
        position={[5, 10, 5]}
        angle={0.3}
        penumbra={1}
        intensity={1.5}
        castShadow
      />
      <spotLight
        position={[-5, 5, -5]}
        angle={0.3}
        penumbra={1}
        intensity={0.8}
        color="#58E6D9"
      />
      <pointLight position={[0, -3, 0]} intensity={0.3} color="#58E6D9" />
      <Suspense fallback={null}>
        <BohrModel scale={0.5} position={[0, 0, 0]} />
        <ContactShadows
          frames={1}
          rotation-x={Math.PI / 2}
          position={[0, -1.5, 0]}
          far={3}
          width={5}
          height={5}
          blur={4}
          opacity={0.3}
        />
      </Suspense>
    </Canvas>
  );
}
