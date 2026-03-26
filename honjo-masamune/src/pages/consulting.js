import { useEffect } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/dist/ScrollTrigger";
import Head from "next/head";
import { motion } from "framer-motion";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

export default function Consulting() {
  useEffect(() => {
    const sections = document.querySelectorAll(".section-reveal");
    sections.forEach((section) => {
      gsap.fromTo(
        section,
        { opacity: 0, y: 60 },
        {
          opacity: 1,
          y: 0,
          duration: 1,
          scrollTrigger: {
            trigger: section,
            start: "top 85%",
            toggleActions: "play none none none",
          },
        }
      );
    });
    return () => ScrollTrigger.getAll().forEach((t) => t.kill());
  }, []);

  return (
    <>
      <Head>
        <title>Scientific Consulting | Honjo Masamune</title>
        <meta
          name="description"
          content="Bounded phase space expertise applied to your spectroscopic research challenges."
        />
      </Head>

      <main className="min-h-screen bg-[#0a0a0a]">
        {/* Hero */}
        <section className="pt-32 pb-24 px-6">
          <div className="max-w-5xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              <p className="text-[#58E6D9] text-sm tracking-[0.3em] uppercase mb-4">
                Consulting
              </p>
              <h1 className="text-5xl md:text-6xl font-bold text-white tracking-tight mb-6">
                Scientific Consulting
              </h1>
              <p className="text-neutral-400 text-lg leading-relaxed max-w-2xl">
                We bring bounded phase space expertise to your research
                challenges. From spectral identification to compound screening,
                our framework delivers categorical precision where traditional
                methods rely on heuristic matching.
              </p>
            </motion.div>
          </div>
        </section>

        {/* Divider */}
        <div className="max-w-5xl mx-auto px-6">
          <div className="h-px bg-neutral-800" />
        </div>

        {/* Section: Spectral Identification */}
        <section className="section-reveal py-24 px-6">
          <div className="max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-5 gap-12 items-start">
            <div className="lg:col-span-3">
              <h2 className="text-3xl md:text-4xl font-bold text-white tracking-tight mb-4">
                Spectral Identification
              </h2>
              <div className="mb-6">
                <h3 className="text-sm tracking-[0.2em] uppercase text-[#58E6D9] mb-2">
                  The Problem
                </h3>
                <p className="text-neutral-400 text-lg leading-relaxed">
                  You have an unidentified spectrum from an IR or Raman
                  measurement. Traditional spectral library matching is
                  sensitive to noise, resolution, and baseline correction. It
                  scales poorly with database size.
                </p>
              </div>
              <div className="mb-6">
                <h3 className="text-sm tracking-[0.2em] uppercase text-[#58E6D9] mb-2">
                  What We Do
                </h3>
                <p className="text-neutral-400 text-lg leading-relaxed">
                  Encode the vibrational frequencies into{" "}
                  <span className="text-neutral-200">S-entropy coordinates</span>,
                  query the categorical database, and return identification with
                  confidence metrics and nearest neighbors in entropy space.
                </p>
              </div>
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
                <p className="text-sm tracking-[0.2em] uppercase text-neutral-500 mb-3">
                  Example
                </p>
                <p className="text-neutral-300 text-base leading-relaxed">
                  A pharmaceutical lab measured 5 unknown peaks at{" "}
                  <span className="text-[#58E6D9] font-mono text-sm">
                    1033, 1060, 1165, 1345, 2844 cm
                    <sup>&minus;1</sup>
                  </span>
                  . Our encoding returned{" "}
                  <span className="text-white font-semibold">
                    CH&#8323;OH (methanol)
                  </span>{" "}
                  at depth 9 with 98.7% confidence.
                </p>
              </div>
            </div>
            <div className="lg:col-span-2 flex items-center justify-center">
              <div className="relative w-48 h-48">
                <div className="absolute inset-0 border border-[#58E6D9]/20 rounded-full" />
                <div className="absolute inset-4 border border-[#58E6D9]/15 rounded-full rotate-45" />
                <div className="absolute inset-8 border border-[#58E6D9]/10 rounded-full -rotate-12" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-3 h-3 bg-[#58E6D9]/40 rounded-full" />
                </div>
                <div className="absolute top-6 right-8 w-2 h-2 bg-[#58E6D9]/30 rounded-full" />
                <div className="absolute bottom-10 left-6 w-1.5 h-1.5 bg-[#58E6D9]/25 rounded-full" />
              </div>
            </div>
          </div>
        </section>

        {/* Divider */}
        <div className="max-w-5xl mx-auto px-6">
          <div className="h-px bg-neutral-800" />
        </div>

        {/* Section: Compound Screening */}
        <section className="section-reveal py-24 px-6">
          <div className="max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-5 gap-12 items-start">
            <div className="lg:col-span-3">
              <h2 className="text-3xl md:text-4xl font-bold text-white tracking-tight mb-4">
                Compound Screening
              </h2>
              <div className="mb-6">
                <h3 className="text-sm tracking-[0.2em] uppercase text-[#58E6D9] mb-2">
                  The Problem
                </h3>
                <p className="text-neutral-400 text-lg leading-relaxed">
                  You need to screen a large compound library for spectral
                  similarity to a target. Pairwise comparison against{" "}
                  <span className="text-neutral-200">N</span> compounds requires{" "}
                  <span className="text-neutral-200">O(N)</span> operations per
                  query, and real libraries contain hundreds of thousands of
                  entries.
                </p>
              </div>
              <div className="mb-6">
                <h3 className="text-sm tracking-[0.2em] uppercase text-[#58E6D9] mb-2">
                  What We Do
                </h3>
                <p className="text-neutral-400 text-lg leading-relaxed">
                  Encode your target, insert your library into the ternary trie,
                  and retrieve all compounds sharing ternary prefixes at your
                  desired resolution.{" "}
                  <span className="text-neutral-200">O(k)</span> per query
                  instead of{" "}
                  <span className="text-neutral-200">O(N)</span>, where{" "}
                  <span className="text-neutral-200">k</span> is the encoding
                  depth.
                </p>
              </div>
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
                <p className="text-sm tracking-[0.2em] uppercase text-neutral-500 mb-3">
                  Example
                </p>
                <p className="text-neutral-300 text-base leading-relaxed">
                  Screening{" "}
                  <span className="text-[#58E6D9] font-mono text-sm">
                    100,000
                  </span>{" "}
                  compounds against a target traditionally takes ~10&#8312;
                  operations. With trie search at depth 12:{" "}
                  <span className="text-white font-semibold">
                    12 operations per query
                  </span>
                  .
                </p>
              </div>
            </div>
            <div className="lg:col-span-2 flex items-center justify-center">
              <div className="relative w-48 h-48">
                {/* Tree/trie visual */}
                <svg
                  viewBox="0 0 200 200"
                  className="w-full h-full"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <line x1="100" y1="30" x2="50" y2="80" stroke="#58E6D9" strokeOpacity="0.2" strokeWidth="1" />
                  <line x1="100" y1="30" x2="100" y2="80" stroke="#58E6D9" strokeOpacity="0.2" strokeWidth="1" />
                  <line x1="100" y1="30" x2="150" y2="80" stroke="#58E6D9" strokeOpacity="0.2" strokeWidth="1" />
                  <line x1="50" y1="80" x2="30" y2="130" stroke="#58E6D9" strokeOpacity="0.12" strokeWidth="1" />
                  <line x1="50" y1="80" x2="50" y2="130" stroke="#58E6D9" strokeOpacity="0.12" strokeWidth="1" />
                  <line x1="50" y1="80" x2="70" y2="130" stroke="#58E6D9" strokeOpacity="0.12" strokeWidth="1" />
                  <line x1="100" y1="80" x2="85" y2="130" stroke="#58E6D9" strokeOpacity="0.12" strokeWidth="1" />
                  <line x1="100" y1="80" x2="100" y2="130" stroke="#58E6D9" strokeOpacity="0.12" strokeWidth="1" />
                  <line x1="100" y1="80" x2="115" y2="130" stroke="#58E6D9" strokeOpacity="0.12" strokeWidth="1" />
                  <line x1="150" y1="80" x2="130" y2="130" stroke="#58E6D9" strokeOpacity="0.12" strokeWidth="1" />
                  <line x1="150" y1="80" x2="150" y2="130" stroke="#58E6D9" strokeOpacity="0.12" strokeWidth="1" />
                  <line x1="150" y1="80" x2="170" y2="130" stroke="#58E6D9" strokeOpacity="0.12" strokeWidth="1" />
                  <circle cx="100" cy="30" r="4" fill="#58E6D9" fillOpacity="0.4" />
                  <circle cx="50" cy="80" r="3" fill="#58E6D9" fillOpacity="0.25" />
                  <circle cx="100" cy="80" r="3" fill="#58E6D9" fillOpacity="0.25" />
                  <circle cx="150" cy="80" r="3" fill="#58E6D9" fillOpacity="0.25" />
                  <circle cx="30" cy="130" r="2" fill="#58E6D9" fillOpacity="0.15" />
                  <circle cx="50" cy="130" r="2" fill="#58E6D9" fillOpacity="0.15" />
                  <circle cx="70" cy="130" r="2" fill="#58E6D9" fillOpacity="0.15" />
                  <circle cx="85" cy="130" r="2" fill="#58E6D9" fillOpacity="0.15" />
                  <circle cx="100" cy="130" r="2" fill="#58E6D9" fillOpacity="0.15" />
                  <circle cx="115" cy="130" r="2" fill="#58E6D9" fillOpacity="0.15" />
                  <circle cx="130" cy="130" r="2" fill="#58E6D9" fillOpacity="0.15" />
                  <circle cx="150" cy="130" r="2" fill="#58E6D9" fillOpacity="0.15" />
                  <circle cx="170" cy="130" r="2" fill="#58E6D9" fillOpacity="0.15" />
                </svg>
              </div>
            </div>
          </div>
        </section>

        {/* Divider */}
        <div className="max-w-5xl mx-auto px-6">
          <div className="h-px bg-neutral-800" />
        </div>

        {/* Section: Fermi Resonance Analysis */}
        <section className="section-reveal py-24 px-6">
          <div className="max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-5 gap-12 items-start">
            <div className="lg:col-span-3">
              <h2 className="text-3xl md:text-4xl font-bold text-white tracking-tight mb-4">
                Fermi Resonance Analysis
              </h2>
              <div className="mb-6">
                <h3 className="text-sm tracking-[0.2em] uppercase text-[#58E6D9] mb-2">
                  The Problem
                </h3>
                <p className="text-neutral-400 text-lg leading-relaxed">
                  You observe anomalous intensity sharing or unexpected spectral
                  features that do not correspond to fundamental modes.
                  Identifying and classifying these resonances requires detailed
                  anharmonic calculations.
                </p>
              </div>
              <div className="mb-6">
                <h3 className="text-sm tracking-[0.2em] uppercase text-[#58E6D9] mb-2">
                  What We Do
                </h3>
                <p className="text-neutral-400 text-lg leading-relaxed">
                  Compute the harmonic network topology, identify all rational
                  frequency relationships, and classify resonances by order{" "}
                  <span className="text-neutral-200 italic">&eta;</span> and
                  coupling strength. The framework detects Fermi resonances
                  directly from the spectral data without anharmonic force
                  field calculations.
                </p>
              </div>
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
                <p className="text-sm tracking-[0.2em] uppercase text-neutral-500 mb-3">
                  Example
                </p>
                <p className="text-neutral-300 text-base leading-relaxed">
                  CO&#8322;&apos;s Raman spectrum shows the famous{" "}
                  <span className="text-[#58E6D9] font-mono text-sm">
                    1285 / 1388 cm<sup>&minus;1</sup>
                  </span>{" "}
                  doublet. Our framework identifies this as an{" "}
                  <span className="text-white font-semibold">
                    &eta; = 2 harmonic edge
                  </span>
                  :{" "}
                  <span className="text-neutral-200 font-mono text-sm">
                    &nu;&#8321;/&nu;&#8322; = 2.081 &asymp; 2/1
                  </span>
                  , the Fermi resonance first described in 1931.
                </p>
              </div>
            </div>
            <div className="lg:col-span-2 flex items-center justify-center">
              <div className="relative w-48 h-48">
                {/* Harmonic network visual */}
                <svg
                  viewBox="0 0 200 200"
                  className="w-full h-full"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  {/* Edges */}
                  <line x1="100" y1="40" x2="45" y2="110" stroke="#58E6D9" strokeOpacity="0.25" strokeWidth="1" />
                  <line x1="100" y1="40" x2="155" y2="110" stroke="#58E6D9" strokeOpacity="0.25" strokeWidth="1" />
                  <line x1="45" y1="110" x2="155" y2="110" stroke="#58E6D9" strokeOpacity="0.15" strokeWidth="1" strokeDasharray="4 4" />
                  <line x1="100" y1="40" x2="100" y2="170" stroke="#58E6D9" strokeOpacity="0.12" strokeWidth="1" />
                  <line x1="45" y1="110" x2="100" y2="170" stroke="#58E6D9" strokeOpacity="0.15" strokeWidth="1" />
                  <line x1="155" y1="110" x2="100" y2="170" stroke="#58E6D9" strokeOpacity="0.15" strokeWidth="1" />
                  {/* Nodes */}
                  <circle cx="100" cy="40" r="5" fill="#58E6D9" fillOpacity="0.35" />
                  <circle cx="45" cy="110" r="4" fill="#58E6D9" fillOpacity="0.25" />
                  <circle cx="155" cy="110" r="4" fill="#58E6D9" fillOpacity="0.25" />
                  <circle cx="100" cy="170" r="3.5" fill="#58E6D9" fillOpacity="0.2" />
                  {/* Resonance highlight */}
                  <circle cx="100" cy="40" r="12" stroke="#58E6D9" strokeOpacity="0.1" strokeWidth="1" fill="none" />
                </svg>
              </div>
            </div>
          </div>
        </section>

        {/* Divider */}
        <div className="max-w-5xl mx-auto px-6">
          <div className="h-px bg-neutral-800" />
        </div>

        {/* Section: Custom Framework Applications */}
        <section className="section-reveal py-24 px-6">
          <div className="max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-5 gap-12 items-start">
            <div className="lg:col-span-3">
              <h2 className="text-3xl md:text-4xl font-bold text-white tracking-tight mb-4">
                Custom Framework Applications
              </h2>
              <div className="mb-6">
                <h3 className="text-sm tracking-[0.2em] uppercase text-[#58E6D9] mb-2">
                  The Problem
                </h3>
                <p className="text-neutral-400 text-lg leading-relaxed">
                  You have a novel bounded oscillatory system &mdash; protein
                  normal modes, phonon spectra, stellar absorption lines &mdash;
                  and want to apply S-entropy encoding to your domain.
                </p>
              </div>
              <div className="mb-6">
                <h3 className="text-sm tracking-[0.2em] uppercase text-[#58E6D9] mb-2">
                  What We Do
                </h3>
                <p className="text-neutral-400 text-lg leading-relaxed">
                  Adapt the encoding protocol to your domain, validate against
                  known data, and build a domain-specific categorical database.
                  Any system with bounded oscillatory behavior possesses the
                  mathematical structure required for{" "}
                  <span className="text-neutral-200">S-entropy encoding</span>.
                  We work with your team to define the appropriate phase space
                  bounds, construct the encoding, and verify it against your
                  existing measurements.
                </p>
              </div>
            </div>
            <div className="lg:col-span-2 flex items-center justify-center">
              <div className="relative w-48 h-48">
                {/* Abstract bounded region */}
                <div className="absolute inset-0 border border-[#58E6D9]/10 rounded-2xl rotate-6" />
                <div className="absolute inset-3 border border-[#58E6D9]/15 rounded-xl -rotate-3" />
                <div className="absolute inset-6 border border-dashed border-[#58E6D9]/12 rounded-lg rotate-2" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-16 h-16 border border-[#58E6D9]/20 rounded-full flex items-center justify-center">
                    <div className="w-6 h-6 border border-[#58E6D9]/30 rounded-full" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Divider */}
        <div className="max-w-5xl mx-auto px-6">
          <div className="h-px bg-neutral-800" />
        </div>

        {/* Section: Contact */}
        <section className="section-reveal py-32 px-6">
          <div className="max-w-5xl mx-auto text-center">
            <h2 className="text-3xl md:text-4xl font-bold text-white tracking-tight mb-6">
              Contact
            </h2>
            <p className="text-neutral-400 text-lg leading-relaxed mb-8 max-w-xl mx-auto">
              Reach out to discuss how the framework can address your research
              needs.
            </p>
            <a
              href="mailto:kundai.sachikonye@tum.de"
              className="text-[#58E6D9] text-lg hover:text-[#58E6D9]/80 transition-colors duration-300 font-mono"
            >
              kundai.sachikonye@tum.de
            </a>
            <p className="text-neutral-500 text-sm mt-3 tracking-wide">
              Technical University of Munich
            </p>
          </div>
        </section>
      </main>
    </>
  );
}
