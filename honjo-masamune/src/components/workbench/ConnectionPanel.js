/**
 * Connection panel: pair the workbench with a local engine.
 *
 * The flow the user follows:
 *   1. install and start `honjo serve` on their own machine
 *   2. copy the token it prints
 *   3. paste it here
 *
 * Until that happens the workbench runs the browser build, and says so.
 * The panel never claims a connection it has not verified, and it
 * distinguishes the three failure modes — engine absent, origin
 * refused, token wrong — because they need different fixes.
 */

import { useState, useCallback, useEffect } from "react";
import {
  STATUS,
  DEFAULT_ENDPOINT,
  probe,
  verifyToken,
  loadConnection,
  saveConnection,
  clearConnection,
} from "@/lib/engine";

const T = {
  panel: "#24253a",
  border: "#2f3146",
  bg: "#1a1b26",
  text: "#c0caf5",
  dim: "#565f89",
  accent: "#7dcfff",
  ok: "#9ece6a",
  warn: "#e0af68",
  err: "#f7768e",
  code: "#1e1f2e",
};

const DOT = {
  [STATUS.CONNECTED]: T.ok,
  [STATUS.CHECKING]: T.warn,
  [STATUS.UNAUTHORISED]: T.err,
  [STATUS.UNREACHABLE]: T.err,
  [STATUS.DISCONNECTED]: T.dim,
};

const LABEL = {
  [STATUS.CONNECTED]: "local engine",
  [STATUS.CHECKING]: "checking…",
  [STATUS.UNAUTHORISED]: "token rejected",
  [STATUS.UNREACHABLE]: "engine not found",
  [STATUS.DISCONNECTED]: "browser engine",
};

export function ConnectionBadge({ status, onClick }) {
  return (
    <button
      onClick={onClick}
      title="Engine connection"
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        padding: "3px 10px",
        background: "transparent",
        border: `1px solid ${T.border}`,
        borderRadius: 4,
        color: T.dim,
        fontSize: 11,
        cursor: "pointer",
        fontFamily: "inherit",
      }}
    >
      <span
        style={{
          width: 7,
          height: 7,
          borderRadius: 4,
          background: DOT[status] || T.dim,
          flexShrink: 0,
        }}
      />
      {LABEL[status] || "unknown"}
    </button>
  );
}

export default function ConnectionPanel({ open, onClose, connection, onChange }) {
  const [endpoint, setEndpoint] = useState(connection.endpoint || DEFAULT_ENDPOINT);
  const [token, setToken] = useState(connection.token || "");
  const [status, setStatus] = useState(connection.status || STATUS.DISCONNECTED);
  const [detail, setDetail] = useState("");
  const [info, setInfo] = useState(null);

  useEffect(() => {
    setEndpoint(connection.endpoint || DEFAULT_ENDPOINT);
    setToken(connection.token || "");
  }, [connection.endpoint, connection.token]);

  const connect = useCallback(async () => {
    setStatus(STATUS.CHECKING);
    setDetail("");
    setInfo(null);

    // Step 1: is anything there at all? This separates "engine not
    // running" from "token wrong", which are different user problems.
    const health = await probe(endpoint);
    if (health.status !== STATUS.CONNECTED) {
      setStatus(STATUS.UNREACHABLE);
      setDetail(health.detail || "no engine at this address");
      onChange({ endpoint, token: "", status: STATUS.UNREACHABLE });
      return;
    }
    setInfo(health);

    // Step 2: does the token work? /health does not check it, so a
    // separate authenticated call is required.
    const auth = await verifyToken(endpoint, token);
    setStatus(auth.status);
    setDetail(auth.detail || "");
    if (auth.status === STATUS.CONNECTED) {
      saveConnection({ endpoint, token });
      onChange({ endpoint, token, status: STATUS.CONNECTED });
    } else {
      onChange({ endpoint, token: "", status: auth.status });
    }
  }, [endpoint, token, onChange]);

  const disconnect = useCallback(() => {
    clearConnection();
    setToken("");
    setStatus(STATUS.DISCONNECTED);
    setDetail("");
    setInfo(null);
    onChange({ endpoint, token: "", status: STATUS.DISCONNECTED });
  }, [endpoint, onChange]);

  if (!open) return null;

  const connected = status === STATUS.CONNECTED;

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.65)",
        zIndex: 200,
        display: "flex",
        justifyContent: "center",
        alignItems: "flex-start",
        paddingTop: 70,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: T.panel,
          border: `1px solid ${T.border}`,
          borderRadius: 8,
          width: 560,
          maxHeight: "78vh",
          overflowY: "auto",
          padding: 24,
          fontFamily:
            "'JetBrains Mono','Fira Code','SF Mono',Consolas,monospace",
          color: T.text,
        }}
      >
        <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 6 }}>
          Run on your own machine
        </div>
        <div
          style={{
            fontSize: 12,
            color: T.dim,
            lineHeight: 1.6,
            marginBottom: 20,
          }}
        >
          The workbench can execute programs in the browser, but the
          reference compiler is the Rust build. Start it locally and paste
          its token here: source and results stay on your machine, and
          nothing is uploaded.
        </div>

        {/* --- step 1 ------------------------------------------------ */}
        <Step n={1} title="Start the engine">
          <Code>{`cargo install --path honjo-rs
honjo serve`}</Code>
          <div style={{ fontSize: 11, color: T.dim, marginTop: 6 }}>
            It prints a token and listens on{" "}
            <span style={{ color: T.accent }}>127.0.0.1:8731</span>. The token
            is new each time and dies when you stop the process.
          </div>
        </Step>

        {/* --- step 2 ------------------------------------------------ */}
        <Step n={2} title="Paste the token">
          <label style={{ fontSize: 11, color: T.dim }}>Endpoint</label>
          <input
            value={endpoint}
            onChange={(e) => setEndpoint(e.target.value)}
            spellCheck={false}
            style={inputStyle}
          />
          <label
            style={{ fontSize: 11, color: T.dim, marginTop: 10, display: "block" }}
          >
            Token
          </label>
          <input
            value={token}
            onChange={(e) => setToken(e.target.value.trim())}
            placeholder="32 hex characters"
            spellCheck={false}
            autoComplete="off"
            style={{
              ...inputStyle,
              letterSpacing: 1,
              color: token.length === 32 ? T.text : T.warn,
            }}
          />
          {token && token.length !== 32 && (
            <div style={{ fontSize: 11, color: T.warn, marginTop: 4 }}>
              expected 32 characters, got {token.length}
            </div>
          )}
        </Step>

        {/* --- status ------------------------------------------------ */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "10px 12px",
            background: T.bg,
            border: `1px solid ${T.border}`,
            borderRadius: 5,
            marginBottom: 14,
          }}
        >
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: 4,
              background: DOT[status],
              flexShrink: 0,
            }}
          />
          <div style={{ flex: 1, fontSize: 12 }}>
            <div>{LABEL[status]}</div>
            {detail && (
              <div style={{ fontSize: 11, color: T.dim, marginTop: 2 }}>
                {detail}
              </div>
            )}
            {connected && info && (
              <div style={{ fontSize: 11, color: T.dim, marginTop: 2 }}>
                {info.engine} v{info.version}
                {info.tokenStrength === "fallback" && (
                  <span style={{ color: T.warn }}>
                    {" "}
                    · token from fallback entropy
                  </span>
                )}
              </div>
            )}
          </div>
        </div>

        {status === STATUS.UNREACHABLE && (
          <Hint tone={T.err}>
            Nothing answered at that address. Check that{" "}
            <code style={codeInline}>honjo serve</code> is still running in a
            terminal, and that the port matches.
          </Hint>
        )}
        {status === STATUS.UNAUTHORISED && detail.includes("origin") && (
          <Hint tone={T.warn}>
            The engine is running but refused this page&apos;s origin. Restart
            it with{" "}
            <code style={codeInline}>
              honjo serve --origin{" "}
              {typeof window !== "undefined" ? window.location.origin : ""}
            </code>
            .
          </Hint>
        )}
        {status === STATUS.UNAUTHORISED && !detail.includes("origin") && (
          <Hint tone={T.err}>
            The engine is running but rejected the token. Copy it again from
            the terminal — a new one is generated every restart.
          </Hint>
        )}

        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={connect}
            disabled={status === STATUS.CHECKING}
            style={{
              ...btnStyle,
              background: T.accent,
              color: T.bg,
              opacity: status === STATUS.CHECKING ? 0.6 : 1,
            }}
          >
            {status === STATUS.CHECKING ? "checking…" : "Connect"}
          </button>
          {connected && (
            <button onClick={disconnect} style={{ ...btnStyle, background: T.border, color: T.text }}>
              Disconnect
            </button>
          )}
          <div style={{ flex: 1 }} />
          <button onClick={onClose} style={{ ...btnStyle, background: "transparent", color: T.dim, border: `1px solid ${T.border}` }}>
            Close
          </button>
        </div>

        <div
          style={{
            marginTop: 18,
            paddingTop: 14,
            borderTop: `1px solid ${T.border}`,
            fontSize: 11,
            color: T.dim,
            lineHeight: 1.6,
          }}
        >
          Without a connection the workbench uses its in-browser build of the
          same language. That build is a second implementation: it agrees with
          the reference on the test suite, but where the two disagree the Rust
          engine is authoritative. Every result is labelled with the engine
          that produced it.
        </div>
      </div>
    </div>
  );
}

/* ---------- small presentational helpers ---------- */

function Step({ n, title, children }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          marginBottom: 8,
        }}
      >
        <span
          style={{
            width: 18,
            height: 18,
            borderRadius: 9,
            background: T.border,
            color: T.text,
            fontSize: 10,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
          }}
        >
          {n}
        </span>
        <span style={{ fontSize: 12, fontWeight: 600 }}>{title}</span>
      </div>
      <div style={{ paddingLeft: 26 }}>{children}</div>
    </div>
  );
}

function Code({ children }) {
  return (
    <pre
      style={{
        background: T.code,
        border: `1px solid ${T.border}`,
        borderRadius: 4,
        padding: "8px 10px",
        margin: 0,
        fontSize: 11.5,
        color: T.ok,
        overflowX: "auto",
        fontFamily: "inherit",
      }}
    >
      {children}
    </pre>
  );
}

function Hint({ tone, children }) {
  return (
    <div
      style={{
        fontSize: 11,
        color: tone,
        lineHeight: 1.6,
        marginBottom: 14,
        paddingLeft: 10,
        borderLeft: `2px solid ${tone}`,
      }}
    >
      {children}
    </div>
  );
}

const inputStyle = {
  width: "100%",
  padding: "7px 10px",
  fontSize: 12,
  background: T.bg,
  border: `1px solid ${T.border}`,
  borderRadius: 4,
  color: T.text,
  outline: "none",
  boxSizing: "border-box",
  fontFamily: "inherit",
  marginTop: 4,
};

const btnStyle = {
  padding: "7px 16px",
  fontSize: 12,
  fontWeight: 600,
  border: "none",
  borderRadius: 4,
  cursor: "pointer",
  fontFamily: "inherit",
};

const codeInline = {
  background: T.code,
  padding: "1px 5px",
  borderRadius: 3,
  color: T.accent,
};
