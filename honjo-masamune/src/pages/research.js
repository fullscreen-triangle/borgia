import { useEffect } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/dist/ScrollTrigger";
import Head from "next/head";
import { motion } from "framer-motion";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

export default function Research() {
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
        <title>Research Directions | Honjo Masamune</title>
        <meta
          name="description"
          content="Research directions in bounded phase space spectroscopy: compound coverage, protein analysis, materials science, astrophysics."
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
                Research
              </p>
              <h1 className="text-5xl md:text-6xl font-bold text-white tracking-tight mb-6">
                Research Directions
              </h1>
              <p className="text-neutral-400 text-lg leading-relaxed max-w-2xl">
                The bounded phase space framework opens multiple avenues for
                high-impact investigation. Each direction produces concrete
                intellectual property &mdash; publications, patents, validated
                software &mdash; with clearly defined scope.
              </p>
            </motion.div>
          </div>
        </section>

        {/* Divider */}
        <div className="max-w-5xl mx-auto px-6">
          <div className="h-px bg-neutral-800" />
        </div>

        {/* Section: Current Achievements */}
        <section className="section-reveal py-24 px-6">
          <div className="max-w-5xl mx-auto">
            <h2 className="text-3xl md:text-4xl font-bold text-white tracking-tight mb-4">
              Current Achievements
            </h2>
            <p className="text-neutral-400 text-lg leading-relaxed mb-10 max-w-2xl">
              The foundation on which all future directions build.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6 text-center">
                <p className="text-3xl font-bold text-white mb-1">3</p>
                <p className="text-neutral-500 text-sm tracking-wide">
                  Publications
                </p>
              </div>
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6 text-center">
                <p className="text-3xl font-bold text-white mb-1">14</p>
                <p className="text-neutral-500 text-sm tracking-wide">
                  Foundational Derivations
                </p>
              </div>
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6 text-center">
                <p className="text-3xl font-bold text-white mb-1">39</p>
                <p className="text-neutral-500 text-sm tracking-wide">
                  Validated Compounds
                </p>
              </div>
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6 text-center">
                <p className="text-3xl font-bold text-[#58E6D9] mb-1">O(k)</p>
                <p className="text-neutral-500 text-sm tracking-wide">
                  Search Complexity
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Divider */}
        <div className="max-w-5xl mx-auto px-6">
          <div className="h-px bg-neutral-800" />
        </div>

        {/* Section: Near-Term Directions */}
        <section className="section-reveal py-24 px-6">
          <div className="max-w-5xl mx-auto">
            <div className="flex items-center gap-4 mb-4">
              <h2 className="text-3xl md:text-4xl font-bold text-white tracking-tight">
                Near-Term Directions
              </h2>
              <span className="text-neutral-600 text-sm tracking-wide border border-neutral-800 rounded-full px-3 py-1">
                1&ndash;2 years
              </span>
            </div>
            <p className="text-neutral-400 text-lg leading-relaxed mb-12 max-w-2xl">
              Scaling the existing framework and establishing experimental
              validation.
            </p>

            <div className="space-y-8">
              {/* Direction 1 */}
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-8">
                <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 items-start">
                  <div className="lg:col-span-3">
                    <div className="flex items-center gap-3 mb-3">
                      <span className="text-[#58E6D9] font-mono text-sm">
                        01
                      </span>
                      <h3 className="text-xl font-bold text-white">
                        Extended Compound Coverage
                      </h3>
                    </div>
                    <p className="text-neutral-400 text-base leading-relaxed mb-4">
                      Expand from 39 to ~2,000 compounds (full NIST CCCBDB),
                      then to ~10&#8310; entries from commercial spectral
                      libraries. This establishes the categorical database as a
                      comprehensive replacement for traditional fingerprint-based
                      spectral identification systems.
                    </p>
                  </div>
                  <div className="lg:col-span-2">
                    <div className="border border-neutral-800 rounded-lg p-4">
                      <p className="text-sm text-neutral-500 tracking-wide uppercase mb-2">
                        Resulting IP
                      </p>
                      <p className="text-neutral-300 text-sm leading-relaxed">
                        A comprehensive spectral database with O(k) search,
                        replacing traditional fingerprint-based systems.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Direction 2 */}
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-8">
                <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 items-start">
                  <div className="lg:col-span-3">
                    <div className="flex items-center gap-3 mb-3">
                      <span className="text-[#58E6D9] font-mono text-sm">
                        02
                      </span>
                      <h3 className="text-xl font-bold text-white">
                        Experimental Validation
                      </h3>
                    </div>
                    <p className="text-neutral-400 text-base leading-relaxed mb-4">
                      Direct validation against measured IR and Raman spectra,
                      not just NIST tabulated values. Demonstrate real-time
                      spectral identification using the encoding protocol on live
                      instrument output.
                    </p>
                  </div>
                  <div className="lg:col-span-2">
                    <div className="border border-neutral-800 rounded-lg p-4">
                      <p className="text-sm text-neutral-500 tracking-wide uppercase mb-2">
                        Resulting IP
                      </p>
                      <p className="text-neutral-300 text-sm leading-relaxed">
                        Validated spectral identification methodology with
                        demonstrated real-time performance.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Direction 3 */}
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-8">
                <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 items-start">
                  <div className="lg:col-span-3">
                    <div className="flex items-center gap-3 mb-3">
                      <span className="text-[#58E6D9] font-mono text-sm">
                        03
                      </span>
                      <h3 className="text-xl font-bold text-white">
                        Protein Normal Mode Analysis
                      </h3>
                    </div>
                    <p className="text-neutral-400 text-base leading-relaxed mb-4">
                      Proteins are bounded oscillatory systems. Apply S-entropy
                      encoding to protein normal modes calculated from elastic
                      network models. This creates a protein fingerprinting
                      system based on vibrational topology rather than sequence
                      or structure.
                    </p>
                  </div>
                  <div className="lg:col-span-2">
                    <div className="border border-neutral-800 rounded-lg p-4">
                      <p className="text-sm text-neutral-500 tracking-wide uppercase mb-2">
                        Resulting IP
                      </p>
                      <p className="text-neutral-300 text-sm leading-relaxed">
                        A protein fingerprinting system based on vibrational
                        topology rather than sequence or structure.
                      </p>
                    </div>
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

        {/* Section: Medium-Term Directions */}
        <section className="section-reveal py-24 px-6">
          <div className="max-w-5xl mx-auto">
            <div className="flex items-center gap-4 mb-4">
              <h2 className="text-3xl md:text-4xl font-bold text-white tracking-tight">
                Medium-Term Directions
              </h2>
              <span className="text-neutral-600 text-sm tracking-wide border border-neutral-800 rounded-full px-3 py-1">
                2&ndash;5 years
              </span>
            </div>
            <p className="text-neutral-400 text-lg leading-relaxed mb-12 max-w-2xl">
              Extending categorical spectroscopy into new domains where bounded
              oscillatory structure exists.
            </p>

            <div className="space-y-8">
              {/* Direction 4 */}
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-8">
                <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 items-start">
                  <div className="lg:col-span-3">
                    <div className="flex items-center gap-3 mb-3">
                      <span className="text-[#58E6D9] font-mono text-sm">
                        04
                      </span>
                      <h3 className="text-xl font-bold text-white">
                        Materials Science &mdash; Phonon Spectra
                      </h3>
                    </div>
                    <p className="text-neutral-400 text-base leading-relaxed mb-4">
                      Crystalline materials have phonon dispersion curves:
                      bounded oscillatory spectra with well-defined Brillouin
                      zone boundaries. S-entropy encoding of phonon spectra
                      would enable categorical materials search and property
                      prediction from vibrational structure alone.
                    </p>
                  </div>
                  <div className="lg:col-span-2">
                    <div className="border border-neutral-800 rounded-lg p-4">
                      <p className="text-sm text-neutral-500 tracking-wide uppercase mb-2">
                        Resulting IP
                      </p>
                      <p className="text-neutral-300 text-sm leading-relaxed">
                        Materials database with categorical addressing and
                        property prediction from phonon topology.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Direction 5 */}
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-8">
                <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 items-start">
                  <div className="lg:col-span-3">
                    <div className="flex items-center gap-3 mb-3">
                      <span className="text-[#58E6D9] font-mono text-sm">
                        05
                      </span>
                      <h3 className="text-xl font-bold text-white">
                        Astrophysical Spectroscopy
                      </h3>
                    </div>
                    <p className="text-neutral-400 text-base leading-relaxed mb-4">
                      Stellar atmospheres produce absorption line spectra.
                      These are bounded oscillatory systems whose categorical
                      encoding could enable rapid spectral classification of
                      astronomical objects at survey scale, complementing
                      existing photometric methods with categorical precision.
                    </p>
                  </div>
                  <div className="lg:col-span-2">
                    <div className="border border-neutral-800 rounded-lg p-4">
                      <p className="text-sm text-neutral-500 tracking-wide uppercase mb-2">
                        Resulting IP
                      </p>
                      <p className="text-neutral-300 text-sm leading-relaxed">
                        Astronomical spectral classification at survey scale
                        using categorical encoding.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Direction 6 */}
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-8">
                <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 items-start">
                  <div className="lg:col-span-3">
                    <div className="flex items-center gap-3 mb-3">
                      <span className="text-[#58E6D9] font-mono text-sm">
                        06
                      </span>
                      <h3 className="text-xl font-bold text-white">
                        Categorical Retrosynthesis
                      </h3>
                    </div>
                    <p className="text-neutral-400 text-base leading-relaxed mb-4">
                      Chemical reactions as displacements in S-entropy space.
                      Reaction types that conserve certain S-entropy
                      relationships become constrained paths through the
                      categorical database. This enables synthesis planning
                      guided by phase space topology rather than pattern-matching
                      on molecular graphs.
                    </p>
                  </div>
                  <div className="lg:col-span-2">
                    <div className="border border-neutral-800 rounded-lg p-4">
                      <p className="text-sm text-neutral-500 tracking-wide uppercase mb-2">
                        Resulting IP
                      </p>
                      <p className="text-neutral-300 text-sm leading-relaxed">
                        Synthesis planning guided by phase space topology with
                        categorical reaction path enumeration.
                      </p>
                    </div>
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

        {/* Section: Research Partnerships */}
        <section className="section-reveal py-24 px-6">
          <div className="max-w-5xl mx-auto">
            <h2 className="text-3xl md:text-4xl font-bold text-white tracking-tight mb-4">
              Research Partnerships
            </h2>
            <p className="text-neutral-400 text-lg leading-relaxed mb-12 max-w-2xl">
              We seek research partners who share our vision of categorical
              spectroscopic technology. Each partnership produces concrete
              intellectual property &mdash; publications, patents, validated
              software &mdash; with clearly defined ownership.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-8">
                <div className="w-8 h-px bg-[#58E6D9]/40 mb-6" />
                <h3 className="text-xl font-bold text-white mb-3">
                  Research Collaboration
                </h3>
                <p className="text-neutral-400 text-base leading-relaxed">
                  Joint research programs with shared intellectual property.
                  Co-authored publications and co-developed software with
                  mutually agreed IP allocation.
                </p>
              </div>

              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-8">
                <div className="w-12 h-px bg-[#58E6D9]/40 mb-6" />
                <h3 className="text-xl font-bold text-white mb-3">
                  Sponsored Research
                </h3>
                <p className="text-neutral-400 text-base leading-relaxed">
                  Directed investigation of specific applications, with
                  first-right-of-access to resulting intellectual property.
                  Defined deliverables and timelines aligned to your objectives.
                </p>
              </div>

              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-8">
                <div className="w-16 h-px bg-[#58E6D9]/40 mb-6" />
                <h3 className="text-xl font-bold text-white mb-3">
                  Fellowship Support
                </h3>
                <p className="text-neutral-400 text-base leading-relaxed">
                  Funding for graduate students and postdoctoral researchers
                  extending the framework into new domains. Named fellowships
                  with publication and patent commitments.
                </p>
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
              To discuss research partnerships:
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
