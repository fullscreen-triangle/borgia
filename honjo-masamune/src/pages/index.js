import { Suspense, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { useGLTF, ContactShadows } from "@react-three/drei";
import { useRouter } from "next/router";
import { motion } from "framer-motion";
import Head from "next/head";

function BohrModel(props) {
  const ref = useRef();
  const { scene } = useGLTF("/models/nitrogen_bohr_model_7p_7n_7e.glb");

  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    ref.current.rotation.y = t * 0.15;  // slow continuous rotation
    ref.current.rotation.x = Math.sin(t * 0.3) * 0.05;  // gentle tilt
    ref.current.position.y = Math.sin(t * 0.5) * 0.03;  // subtle float
  });

  return (
    <group ref={ref} {...props}>
      <primitive object={scene} />
    </group>
  );
}

useGLTF.preload("/models/nitrogen_bohr_model_7p_7n_7e.glb");

export default function Landing() {
  const router = useRouter();

  return (
    <>
      <Head>
        <title>Honjo Masamune</title>
        <meta name="description" content="Categorical spectroscopic instruments from bounded phase space geometry." />
      </Head>

      {/* Underlay: Title text behind the 3D model */}
      <div className="fixed inset-0 flex items-center justify-center z-0 select-none pointer-events-none">
        <motion.h1
          className="text-[12vw] md:text-[10vw] lg:text-[8vw] font-bold tracking-[0.3em] text-neutral-800/40 uppercase whitespace-nowrap"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 2, delay: 0.5 }}
          style={{ fontFamily: "var(--font-mont)" }}
        >
          Honjo Masamune
        </motion.h1>
      </div>

      {/* 3D Canvas on top */}
      <div
        className="fixed inset-0 z-10 cursor-pointer"
        onClick={() => router.push("/framework")}
      >
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
            <BohrModel scale={1.2} position={[0, 0, 0]} />
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
      </div>

      {/* Bottom indicator */}
      <motion.div
        className="fixed bottom-8 left-1/2 -translate-x-1/2 z-20 flex flex-col items-center gap-2 cursor-pointer"
        onClick={() => router.push("/framework")}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2, duration: 1 }}
      >
        <span className="text-neutral-500 text-xs tracking-[0.3em] uppercase">Enter</span>
        <motion.div
          className="w-[1px] h-8 bg-neutral-600"
          animate={{ scaleY: [1, 0.5, 1] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        />
      </motion.div>
    </>
  );
}
