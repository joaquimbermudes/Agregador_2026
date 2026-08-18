import dataJson from "./data/presidente.json";

type Estimate = {
  estimativa_pct: number;
  ic95_lo_pct: number;
  ic95_hi_pct: number;
};

type Turn = {
  titulo: string;
  quantidade_pesquisas: number;
  data_ultima_pesquisa: string;
  estimativa_atual: {
    curto: { filtrado: Record<string, Estimate> };
    longo: { filtrado: Record<string, Estimate> };
  };
  categorias: Array<{ id: string; nome: string; cor: string }>;
  grafico: string;
};

type PollData = {
  gerado_em: string;
  modelo: string;
  fonte: string;
  turnos: Record<"primeiro_turno" | "segundo_turno", Turn>;
};

const data = dataJson as PollData;

const repoName = process.env.GITHUB_REPOSITORY?.split("/")[1];
const basePath =
  process.env.STATIC_EXPORT === "1" && repoName ? `/${repoName}` : "";

const formatDate = (date: string) =>
  new Intl.DateTimeFormat("pt-BR", {
    day: "2-digit",
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(`${date}T12:00:00Z`));

const pct = (value: number) =>
  new Intl.NumberFormat("pt-BR", {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }).format(value);

function EstimateCard({ turn, compact = false }: { turn: Turn; compact?: boolean }) {
  return (
    <div className={`estimate-card ${compact ? "estimate-card--compact" : ""}`}>
      <div className="estimate-card__head">
        <div>
          <span className="eyebrow">Estimativa atual</span>
          <h3>{turn.titulo}</h3>
        </div>
        <span className="sample-count">{turn.quantidade_pesquisas} pesquisas</span>
      </div>

      <div className="candidate-grid">
        {turn.categorias.map((candidate) => {
          const short = turn.estimativa_atual.curto.filtrado[candidate.id];
          const long = turn.estimativa_atual.longo.filtrado[candidate.id];

          return (
            <div className="candidate" key={candidate.id}>
              <div className="candidate__identity">
                <span
                  className="candidate__dot"
                  style={{ backgroundColor: candidate.cor }}
                  aria-hidden="true"
                />
                <strong>{candidate.nome}</strong>
              </div>
              <div className="candidate__numbers">
                <div>
                  <span>Curto prazo</span>
                  <b>{pct(short.estimativa_pct)}%</b>
                  <small>
                    IC 95%: {pct(short.ic95_lo_pct)}–{pct(short.ic95_hi_pct)}
                  </small>
                </div>
                <div>
                  <span>Longo prazo</span>
                  <b>{pct(long.estimativa_pct)}%</b>
                  <small>
                    IC 95%: {pct(long.ic95_lo_pct)}–{pct(long.ic95_hi_pct)}
                  </small>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TurnSection({ id, turn }: { id: string; turn: Turn }) {
  return (
    <section className="turn-section" id={id}>
      <div className="section-heading">
        <div>
          <span className="section-index">{id === "primeiro-turno" ? "01" : "02"}</span>
          <h2>{turn.titulo}</h2>
        </div>
        <p>
          Última pesquisa incluída em {formatDate(turn.data_ultima_pesquisa)}.
          Resultados expressos como proporção dos votos válidos.
        </p>
      </div>

      <EstimateCard turn={turn} compact />

      <figure className="chart-frame">
        <img
          src={`${basePath}${turn.grafico}`}
          alt={`Evolução das estimativas de curto e longo prazo para ${turn.titulo.toLowerCase()}`}
          loading={id === "primeiro-turno" ? "eager" : "lazy"}
        />
        <figcaption>
          Linhas contínuas mostram a estimativa disponível em tempo real pelo filtro de
          Kalman; linhas pontilhadas mostram a leitura retrospectiva do suavizador RTS.
        </figcaption>
      </figure>
    </section>
  );
}

export default function Home() {
  const first = data.turnos.primeiro_turno;
  const second = data.turnos.segundo_turno;

  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Agregador 2026 — início">
          <span className="brand__mark">A26</span>
          <span>Agregador presidencial</span>
        </a>
        <nav aria-label="Navegação principal">
          <a href="#primeiro-turno">1º turno</a>
          <a href="#segundo-turno">2º turno</a>
          <a href="#metodologia">Método</a>
        </nav>
      </header>

      <section className="hero" id="top">
        <div className="hero__copy">
          <span className="status-pill">
            <span aria-hidden="true" /> Atualização diária
          </span>
          <p className="kicker">Eleições presidenciais · Brasil · 2026</p>
          <h1>O sinal por trás das pesquisas.</h1>
          <p className="hero__lead">
            Todas as pesquisas nacionais com Lula e Flávio Bolsonaro, reunidas em
            séries comparáveis para distinguir o movimento imediato da tendência
            estrutural.
          </p>
          <div className="hero__meta">
            <div>
              <span>Base atual</span>
              <strong>{first.quantidade_pesquisas + second.quantidade_pesquisas} pesquisas</strong>
            </div>
            <div>
              <span>Último dado</span>
              <strong>{formatDate(first.data_ultima_pesquisa)}</strong>
            </div>
            <div>
              <span>Modelo</span>
              <strong>Kalman + EM/RTS</strong>
            </div>
          </div>
        </div>

        <div className="hero__panel">
          <p className="panel-note">Leitura rápida</p>
          <EstimateCard turn={first} />
          <a className="text-link" href="#primeiro-turno">
            Ver série completa <span aria-hidden="true">↓</span>
          </a>
        </div>
      </section>

      <aside className="reading-guide" aria-label="Como ler o agregador">
        <div>
          <span className="guide-number">Curto</span>
          <p>Reage com mais velocidade às pesquisas recentes e capta mudanças de momento.</p>
        </div>
        <div>
          <span className="guide-number">Longo</span>
          <p>Move-se lentamente e representa a tendência estrutural do eleitorado.</p>
        </div>
        <div>
          <span className="guide-number">RTS</span>
          <p>Usa toda a série para revisar o passado; não representa informação disponível à época.</p>
        </div>
      </aside>

      <TurnSection id="primeiro-turno" turn={first} />
      <TurnSection id="segundo-turno" turn={second} />

      <section className="method" id="metodologia">
        <div className="method__title">
          <span className="eyebrow">Metodologia</span>
          <h2>Duas velocidades, um mesmo eleitorado.</h2>
        </div>
        <div className="method__copy">
          <p>
            O agregador usa um modelo de estado acoplado: uma intenção de voto de
            curto prazo com reversão à média e uma tendência de longo prazo. Os
            parâmetros desconhecidos são estimados por máxima verossimilhança com
            algoritmo EM e suavização RTS; em seguida, o filtro de Kalman é executado
            novamente com esses parâmetros.
          </p>
          <p>
            Não são aplicados vieses fixos por instituto. Pesquisas do primeiro turno
            incluem Lula, Flávio Bolsonaro e a soma de todos os demais candidatos em
            “Outros”. O segundo turno exibe apenas cenários Lula × Flávio Bolsonaro.
            Indecisos e abstenções são retirados para a conversão em votos válidos.
          </p>
          <div className="method__note">
            <strong>Importante</strong>
            <span>
              Estimativas do modelo não são previsão do resultado eleitoral. Intervalos
              refletem a incerteza estatística do modelo, não todos os erros possíveis
              de pesquisas e cobertura.
            </span>
          </div>
        </div>
      </section>

      <footer>
        <div>
          <strong>Agregador presidencial 2026</strong>
          <span>Dados públicos, método reproduzível.</span>
        </div>
        <p>
          Fonte: pesquisas compiladas na Wikipédia. Gerado automaticamente em
          {" "}{new Intl.DateTimeFormat("pt-BR", {
            day: "2-digit",
            month: "2-digit",
            year: "numeric",
            timeZone: "UTC",
          }).format(new Date(data.gerado_em))}.
        </p>
      </footer>
    </main>
  );
}
