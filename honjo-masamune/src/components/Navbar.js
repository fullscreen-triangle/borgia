import Link from "next/link";
import React, { useState } from "react";
import { useRouter } from "next/router";
import { motion } from "framer-motion";

const NavLink = ({ href, title, className = "" }) => {
  const router = useRouter();
  const isActive = router.asPath === href;

  return (
    <Link
      href={href}
      className={`${className} relative group text-sm tracking-wider uppercase ${
        isActive ? "text-[#58E6D9]" : "text-neutral-400 hover:text-white"
      } transition-colors duration-300`}
    >
      {title}
      <span
        className={`inline-block h-[1px] bg-[#58E6D9] absolute left-0 -bottom-1 transition-[width] ease duration-300 ${
          isActive ? "w-full" : "w-0 group-hover:w-full"
        }`}
      />
    </Link>
  );
};

const MobileNavLink = ({ href, title, toggle }) => {
  const router = useRouter();
  const isActive = router.asPath === href;

  return (
    <button
      className={`text-lg ${isActive ? "text-[#58E6D9]" : "text-neutral-300"} py-2`}
      onClick={() => { toggle(); router.push(href); }}
    >
      {title}
    </button>
  );
};

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);

  const toggle = () => setIsOpen(!isOpen);

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-[#0a0a0a]/80 backdrop-blur-md border-b border-neutral-800/50">
      <nav className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center">
          <motion.div
            className="w-10 h-10 rounded-full border border-neutral-600 flex items-center justify-center text-sm font-bold text-white tracking-widest"
            whileHover={{ borderColor: "#58E6D9", color: "#58E6D9" }}
            transition={{ duration: 0.3 }}
          >
            HM
          </motion.div>
        </Link>

        {/* Desktop Nav - visible by default, hidden at lg (1023px) and below */}
        <div className="flex items-center gap-8 lg:hidden">
          <NavLink href="/framework" title="Framework" />
          <NavLink href="/database" title="Database" />
          <NavLink href="/consulting" title="Consulting" />
          <NavLink href="/licensing" title="Licensing" />
          <NavLink href="/research" title="Research" />
        </div>

        {/* Mobile Hamburger - hidden by default, shown at lg (1023px) and below */}
        <button
          className="hidden lg:flex flex-col gap-1.5 z-50"
          onClick={toggle}
          aria-label="Toggle menu"
        >
          <motion.span
            className="block w-6 h-[2px] bg-white"
            animate={isOpen ? { rotate: 45, y: 5 } : { rotate: 0, y: 0 }}
          />
          <motion.span
            className="block w-6 h-[2px] bg-white"
            animate={isOpen ? { opacity: 0 } : { opacity: 1 }}
          />
          <motion.span
            className="block w-6 h-[2px] bg-white"
            animate={isOpen ? { rotate: -45, y: -5 } : { rotate: 0, y: 0 }}
          />
        </button>

        {/* Mobile Menu */}
        {isOpen && (
          <motion.div
            className="fixed inset-0 bg-[#0a0a0a]/95 backdrop-blur-lg flex flex-col items-center justify-center gap-4 z-40"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <MobileNavLink href="/framework" title="Framework" toggle={toggle} />
            <MobileNavLink href="/database" title="Database" toggle={toggle} />
            <MobileNavLink href="/consulting" title="Consulting" toggle={toggle} />
            <MobileNavLink href="/licensing" title="Licensing" toggle={toggle} />
            <MobileNavLink href="/research" title="Research" toggle={toggle} />
          </motion.div>
        )}
      </nav>
    </header>
  );
};

export default Navbar;
