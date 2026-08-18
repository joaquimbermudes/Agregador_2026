import type { Metadata } from "next";
import "./globals.css";

const siteUrl =
  process.env.NEXT_PUBLIC_SITE_URL ??
  "https://agregador-presidencial-2026.joaquimacbermudes.chatgpt.site";
const repository = process.env.GITHUB_REPOSITORY?.split("/")[1];
const basePath = process.env.STATIC_EXPORT === "1" && repository ? `/${repository}` : "";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: "Agregador presidencial 2026",
  description:
    "Pesquisas para presidente filtradas por um modelo acoplado de curto e longo prazo com Kalman, EM e suavização RTS.",
  icons: {
    icon: `${basePath}/favicon.svg`,
    shortcut: `${basePath}/favicon.svg`,
  },
  openGraph: {
    title: "Agregador presidencial 2026",
    description: "O sinal de curto prazo e a tendência estrutural das pesquisas para presidente.",
    type: "website",
    locale: "pt_BR",
    images: [{ url: `${basePath}/og.png`, width: 1738, height: 909, alt: "Agregador presidencial 2026" }],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="pt-BR">
      <body>{children}</body>
    </html>
  );
}
