import Head from "next/head";
import Link from "next/link";
import Layout from "@/components/Layout";

const NotFound = () => {
  return (
    <>
      <Head>
        <title>404 | Honjo Masamune</title>
        <meta name="description" content="Page not found." />
      </Head>
      <main className="pt-24 min-h-screen flex items-center justify-center">
        <Layout className="flex flex-col items-center justify-center text-center">
          <h1 className="text-[8rem] font-bold text-neutral-800">404</h1>
          <p className="text-2xl text-neutral-400 mb-8">Page Not Found</p>
          <Link
            href="/"
            className="inline-block rounded-lg border border-neutral-600 px-6 py-3 text-sm font-semibold text-white uppercase tracking-wider hover:border-[#58E6D9] hover:text-[#58E6D9] transition-colors duration-300"
          >
            Go Home
          </Link>
        </Layout>
      </main>
    </>
  );
};

export default NotFound;
