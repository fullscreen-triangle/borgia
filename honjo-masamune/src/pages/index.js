import { useRef } from "react";
import { useRouter } from "next/router";
import { motion } from "framer-motion";
import Head from "next/head";
import dynamic from "next/dynamic";

// Dynamically import the 3D scene with SSR disabled —
// Three.js requires browser APIs (WebGL, window) that don't exist during static generation
const Scene = dynamic(() => import("../components/LandingScene"), { ssr: false });

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
        <Scene />
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
