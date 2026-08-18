import type { NextConfig } from "next";

const staticExport = process.env.STATIC_EXPORT === "1";
const repository = process.env.GITHUB_REPOSITORY?.split("/")[1];
const basePath = staticExport && repository ? `/${repository}` : "";

const nextConfig: NextConfig = {
  output: staticExport ? "export" : undefined,
  basePath,
  assetPrefix: basePath || undefined,
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
