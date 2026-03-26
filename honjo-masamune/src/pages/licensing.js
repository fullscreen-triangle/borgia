import { useEffect } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/dist/ScrollTrigger";
import Head from "next/head";
import { motion } from "framer-motion";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

export default function Licensing() {
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
        <title>Licensing | Honjo Masamune</title>
        <meta
          name="description"
          content="License categorical spectroscopic technology for your instruments and workflows."
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
                Licensing
              </p>
              <h1 className="text-5xl md:text-6xl font-bold text-white tracking-tight mb-6">
                Licensing
              </h1>
              <p className="text-neutral-400 text-lg leading-relaxed max-w-2xl">
                Integrate categorical spectroscopic technology into your
                instruments and workflows. We offer tiered licensing for
                individual components or the complete framework.
              </p>
            </motion.div>
          </div>
        </section>

        {/* Divider */}
        <div className="max-w-5xl mx-auto px-6">
          <div className="h-px bg-neutral-800" />
        </div>

        {/* Section: Available Technologies */}
        <section className="section-reveal py-24 px-6">
          <div className="max-w-5xl mx-auto">
            <h2 className="text-3xl md:text-4xl font-bold text-white tracking-tight mb-4">
              Available Technologies
            </h2>
            <p className="text-neutral-400 text-lg leading-relaxed mb-12 max-w-2xl">
              Three levels of integration, from encoding protocol to complete
              framework.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Tier 1 */}
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-8 flex flex-col">
                <div className="w-10 h-10 border border-[#58E6D9]/30 rounded-lg flex items-center justify-center mb-6">
                  <div className="w-4 h-4 border border-[#58E6D9]/50 rounded-sm" />
                </div>
                <h3 className="text-xl font-bold text-white mb-3">
                  S-Entropy Encoding Method
                </h3>
                <p className="text-neutral-400 text-base leading-relaxed mb-6 flex-grow">
                  License the encoding protocol: vibrational frequencies mapped
                  to{" "}
                  <span className="text-neutral-200 font-mono text-sm">
                    (S<sub>k</sub>, S<sub>t</sub>, S<sub>e</sub>)
                  </span>{" "}
                  coordinates. Integrate directly into existing spectral
                  analysis software.
                </p>
                <div className="border-t border-neutral-800 pt-4">
                  <p className="text-sm text-neutral-500 tracking-wide uppercase mb-2">
                    Suitable For
                  </p>
                  <p className="text-neutral-300 text-sm">
                    Instrument software vendors, analytical chemistry platforms
                  </p>
                </div>
              </div>

              {/* Tier 2 */}
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-8 flex flex-col">
                <div className="w-10 h-10 border border-[#58E6D9]/30 rounded-lg flex items-center justify-center mb-6">
                  <div className="w-4 h-4 bg-[#58E6D9]/20 rounded-sm" />
                </div>
                <h3 className="text-xl font-bold text-white mb-3">
                  Categorical Database Architecture
                </h3>
                <p className="text-neutral-400 text-base leading-relaxed mb-6 flex-grow">
                  License the ternary trie data structure and search algorithm.{" "}
                  <span className="text-neutral-200">O(k)</span> search
                  independent of database size. Includes encoding, trie
                  construction, fuzzy search, and property-based retrieval.
                </p>
                <div className="border-t border-neutral-800 pt-4">
                  <p className="text-sm text-neutral-500 tracking-wide uppercase mb-2">
                    Suitable For
                  </p>
                  <p className="text-neutral-300 text-sm">
                    Compound library providers, drug discovery platforms
                  </p>
                </div>
              </div>

              {/* Tier 3 */}
              <div className="bg-neutral-900/50 border border-[#58E6D9]/20 rounded-xl p-8 flex flex-col">
                <div className="w-10 h-10 border border-[#58E6D9]/40 rounded-lg flex items-center justify-center mb-6">
                  <div className="w-4 h-4 bg-[#58E6D9]/35 rounded-sm" />
                </div>
                <h3 className="text-xl font-bold text-white mb-3">
                  Full Framework License
                </h3>
                <p className="text-neutral-400 text-base leading-relaxed mb-4 flex-grow">
                  Complete bounded phase space framework including:
                </p>
                <ul className="text-neutral-300 text-sm space-y-2 mb-6 flex-grow">
                  <li className="flex items-start gap-2">
                    <span className="text-[#58E6D9] mt-1.5 block w-1 h-1 rounded-full bg-[#58E6D9]/60 flex-shrink-0" />
                    <span>
                      Atomic derivation (periodic table from first principles)
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-[#58E6D9] mt-1.5 block w-1 h-1 rounded-full bg-[#58E6D9]/60 flex-shrink-0" />
                    <span>
                      Harmonic molecular networks (self-clocking resonators)
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-[#58E6D9] mt-1.5 block w-1 h-1 rounded-full bg-[#58E6D9]/60 flex-shrink-0" />
                    <span>Categorical compound database</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-[#58E6D9] mt-1.5 block w-1 h-1 rounded-full bg-[#58E6D9]/60 flex-shrink-0" />
                    <span>All validation code and datasets</span>
                  </li>
                </ul>
                <div className="border-t border-neutral-800 pt-4">
                  <p className="text-sm text-neutral-500 tracking-wide uppercase mb-2">
                    Suitable For
                  </p>
                  <p className="text-neutral-300 text-sm">
                    Research institutions, national laboratories, instrument
                    manufacturers
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Divider */}
        <div className="max-w-5xl mx-auto px-6">
          <div className="h-px bg-neutral-800" />
        </div>

        {/* Section: Integration Support */}
        <section className="section-reveal py-24 px-6">
          <div className="max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-5 gap-12 items-start">
            <div className="lg:col-span-3">
              <h2 className="text-3xl md:text-4xl font-bold text-white tracking-tight mb-4">
                Integration Support
              </h2>
              <p className="text-neutral-400 text-lg leading-relaxed mb-8">
                We provide technical support for seamless integration into your
                existing systems.
              </p>
              <div className="space-y-6">
                <div className="flex items-start gap-4">
                  <div className="w-px h-12 bg-[#58E6D9]/30 mt-1 flex-shrink-0" />
                  <div>
                    <h4 className="text-white font-semibold mb-1">
                      API Documentation &amp; Reference Implementations
                    </h4>
                    <p className="text-neutral-400 text-sm leading-relaxed">
                      Complete API documentation with reference implementations
                      in Python and Rust.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <div className="w-px h-12 bg-[#58E6D9]/25 mt-1 flex-shrink-0" />
                  <div>
                    <h4 className="text-white font-semibold mb-1">
                      Custom Encoding
                    </h4>
                    <p className="text-neutral-400 text-sm leading-relaxed">
                      Custom encoding protocols for domain-specific frequency
                      sets and measurement configurations.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <div className="w-px h-12 bg-[#58E6D9]/20 mt-1 flex-shrink-0" />
                  <div>
                    <h4 className="text-white font-semibold mb-1">
                      Validation
                    </h4>
                    <p className="text-neutral-400 text-sm leading-relaxed">
                      Validation against your internal spectral databases to
                      ensure consistency with your existing measurements.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <div className="w-px h-12 bg-[#58E6D9]/15 mt-1 flex-shrink-0" />
                  <div>
                    <h4 className="text-white font-semibold mb-1">
                      Team Training
                    </h4>
                    <p className="text-neutral-400 text-sm leading-relaxed">
                      Training sessions for your technical team on the
                      mathematical foundations and practical usage.
                    </p>
                  </div>
                </div>
              </div>
            </div>
            <div className="lg:col-span-2 flex items-center justify-center">
              <div className="relative w-48 h-48">
                {/* Integration visual: layered rectangles */}
                <div className="absolute top-4 left-4 right-8 bottom-16 border border-[#58E6D9]/10 rounded-lg" />
                <div className="absolute top-8 left-8 right-4 bottom-12 border border-[#58E6D9]/15 rounded-lg" />
                <div className="absolute top-12 left-6 right-6 bottom-4 border border-[#58E6D9]/20 rounded-lg bg-[#58E6D9]/[0.02]" />
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
                  <div className="w-8 h-8 border border-[#58E6D9]/30 rounded rotate-45" />
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Divider */}
        <div className="max-w-5xl mx-auto px-6">
          <div className="h-px bg-neutral-800" />
        </div>

        {/* Section: Intellectual Property */}
        <section className="section-reveal py-24 px-6">
          <div className="max-w-5xl mx-auto">
            <h2 className="text-3xl md:text-4xl font-bold text-white tracking-tight mb-4">
              Intellectual Property
            </h2>
            <p className="text-neutral-400 text-lg leading-relaxed mb-10 max-w-2xl">
              All technologies are derived from original mathematical research,
              validated against reference data with zero adjustable parameters.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
                <p className="text-[#58E6D9] text-sm tracking-[0.2em] uppercase mb-2">
                  Publications
                </p>
                <p className="text-neutral-300 text-base leading-relaxed">
                  Published in peer-reviewed framework: 3 papers covering atomic
                  structure, molecular networks, and compound database
                  construction, supported by 14 foundational derivations.
                </p>
              </div>
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
                <p className="text-[#58E6D9] text-sm tracking-[0.2em] uppercase mb-2">
                  Validation
                </p>
                <p className="text-neutral-300 text-base leading-relaxed">
                  Validated against NIST reference data with zero adjustable
                  parameters. 39-compound database with exact spectral
                  reproduction.
                </p>
              </div>
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6 md:col-span-2">
                <p className="text-[#58E6D9] text-sm tracking-[0.2em] uppercase mb-2">
                  Methodology
                </p>
                <p className="text-neutral-300 text-base leading-relaxed">
                  Proprietary methodology derived from bounded phase space
                  geometry. All encoding protocols, data structures, and search
                  algorithms are original work with clearly established
                  provenance.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Divider */}
        <div className="max-w-5xl mx-auto px-6">
          <div className="h-px bg-neutral-800" />
        </div>

        {/* Section: Inquiries */}
        <section className="section-reveal py-32 px-6">
          <div className="max-w-5xl mx-auto text-center">
            <h2 className="text-3xl md:text-4xl font-bold text-white tracking-tight mb-6">
              Inquiries
            </h2>
            <p className="text-neutral-400 text-lg leading-relaxed mb-8 max-w-xl mx-auto">
              For licensing inquiries, contact:
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
