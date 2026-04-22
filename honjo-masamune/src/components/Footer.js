import Link from "next/link";

const Footer = () => {
  return (
    <footer className="border-t border-neutral-800/50 bg-[#0a0a0a]">
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid grid-cols-3 md:grid-cols-1 gap-8">
          <div>
            <h3 className="text-white font-bold text-lg mb-4 tracking-wider">Honjo Masamune</h3>
            <p className="text-neutral-500 text-sm leading-relaxed">
              Categorical spectroscopic instruments derived from bounded phase space geometry.
            </p>
          </div>
          <div>
            <h4 className="text-neutral-300 font-semibold text-sm mb-4 uppercase tracking-wider">Navigate</h4>
            <div className="flex flex-col gap-2">
              <Link href="/framework" className="text-neutral-500 hover:text-[#58E6D9] text-sm transition-colors">Framework</Link>
              <Link href="/validation" className="text-neutral-500 hover:text-[#58E6D9] text-sm transition-colors">Validation</Link>
              <Link href="/database" className="text-neutral-500 hover:text-[#58E6D9] text-sm transition-colors">Database</Link>
              <Link href="/tools" className="text-neutral-500 hover:text-[#58E6D9] text-sm transition-colors">Tools</Link>
              <Link href="/consulting" className="text-neutral-500 hover:text-[#58E6D9] text-sm transition-colors">Consulting</Link>
              <Link href="/licensing" className="text-neutral-500 hover:text-[#58E6D9] text-sm transition-colors">Licensing</Link>
              <Link href="/research" className="text-neutral-500 hover:text-[#58E6D9] text-sm transition-colors">Research</Link>
            </div>
          </div>
          <div>
            <h4 className="text-neutral-300 font-semibold text-sm mb-4 uppercase tracking-wider">Contact</h4>
            <p className="text-neutral-500 text-sm">kundai.sachikonye@tum.de</p>
            <p className="text-neutral-500 text-sm mt-1">Technical University of Munich</p>
          </div>
        </div>
        <div className="border-t border-neutral-800/50 mt-8 pt-6">
          <p className="text-neutral-600 text-xs text-center">
            &copy; {new Date().getFullYear()} Honjo Masamune. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
