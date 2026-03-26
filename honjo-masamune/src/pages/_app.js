import "@/styles/globals.css";
import { Montserrat } from "next/font/google";
import Head from "next/head";
import { useRouter } from "next/router";
import { useEffect } from "react";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { AnimatePresence } from "framer-motion";

const montserrat = Montserrat({ subsets: ["latin"], variable: "--font-mont" });

export default function App({ Component, pageProps }) {
  const router = useRouter();
  const isLanding = router.pathname === "/";

  useEffect(() => {
    let lenis;
    const initLenis = async () => {
      const Lenis = (await import("@studio-freight/lenis")).default;
      lenis = new Lenis({
        duration: 1.2,
        easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
        orientation: "vertical",
        smoothWheel: true,
      });

      function raf(time) {
        lenis.raf(time);
        requestAnimationFrame(raf);
      }
      requestAnimationFrame(raf);
    };

    // Only init smooth scroll on non-landing pages
    if (!isLanding) {
      initLenis();
    }

    return () => {
      if (lenis) lenis.destroy();
    };
  }, [isLanding]);

  return (
    <>
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
        <title>Honjo Masamune</title>
      </Head>
      <main className={`${montserrat.variable} font-mont bg-[#0a0a0a] w-full min-h-screen`}>
        {!isLanding && <Navbar />}
        <AnimatePresence initial={false} mode="wait">
          <Component key={router.asPath} {...pageProps} />
        </AnimatePresence>
        {!isLanding && <Footer />}
      </main>
    </>
  );
}
