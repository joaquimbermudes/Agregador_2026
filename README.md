# Agregador presidencial 2026

Site estático das pesquisas presidenciais de primeiro e segundo turno. A base é atualizada na Wikipédia e processada diariamente por um modelo de estado acoplado de curto e longo prazo, sem efeitos fixos por instituto. Os parâmetros são estimados por EM com suavização RTS e alimentam uma nova execução causal do filtro de Kalman.

## Execução local

Requer Python 3.12+, Node.js 22.13+ e pnpm 11.

```bash
python -m pip install -r scripts/requirements.txt
python scripts/notebook_pesquisas.py
python scripts/build_presidential_data.py
pnpm install --frozen-lockfile
pnpm dev
```

Para testar exatamente a versão estática publicada:

```bash
STATIC_EXPORT=1 pnpm run build:static
```

O resultado é salvo em `out/`.

## GitHub Pages e atualização diária

O workflow `.github/workflows/pages.yml` atualiza a base todos os dias às 06h17 no horário de Brasília, regenera os parâmetros, o JSON e os dois gráficos, compila a página e publica o diretório `out/` no GitHub Pages.

Esta pasta foi preparada como um repositório autônomo. No GitHub, escolha **Settings → Pages → Source → GitHub Actions**. Execuções manuais também podem ser iniciadas na aba **Actions**.

## Arquivos gerados

- `snapshot_pesquisas.json`: pesquisas presidenciais limpas.
- `app/data/presidente.json`: séries, estimativas atuais e parâmetros do modelo.
- `public/plots/`: gráficos filtrados de primeiro e segundo turno.
