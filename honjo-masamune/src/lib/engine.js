/**
 * Engine transport.
 *
 * Two ways to run a program, and the workbench must be honest about
 * which one produced a given result:
 *
 *   LOCAL   the Rust reference compiler, running on the user's machine,
 *           reached over loopback HTTP with a session token. This is
 *           the authoritative back end.
 *
 *   BROWSER the JavaScript build of the same front end, running in the
 *           page. No install required, but it is a second
 *           implementation and may diverge from the reference.
 *
 * The distinction is surfaced in every result rather than hidden, so a
 * number on screen can always be traced to the thing that computed it.
 */

import { evaluate as jsEvaluate, compile as jsCompile } from "@/lib/honjo";

export const ENGINE = {
  LOCAL: "rust-reference",
  BROWSER: "js-browser",
};

export const STATUS = {
  DISCONNECTED: "disconnected",
  CHECKING: "checking",
  CONNECTED: "connected",
  UNREACHABLE: "unreachable",
  UNAUTHORISED: "unauthorised",
};

const DEFAULT_ENDPOINT = "http://127.0.0.1:8731";

/** Milliseconds before a local-engine request is abandoned. */
const TIMEOUT_MS = 15000;

/** Health checks should fail fast: the engine is either there or not. */
const HEALTH_TIMEOUT_MS = 2500;

function withTimeout(ms) {
  const c = new AbortController();
  const t = setTimeout(() => c.abort(), ms);
  return { signal: c.signal, done: () => clearTimeout(t) };
}

/**
 * Ask a local engine whether it is alive.
 *
 * /health is deliberately unauthenticated: a user who has not yet
 * pasted a token still needs to learn whether the engine is running,
 * and the endpoint reveals only a version string.
 */
export async function probe(endpoint = DEFAULT_ENDPOINT) {
  const { signal, done } = withTimeout(HEALTH_TIMEOUT_MS);
  try {
    const res = await fetch(`${endpoint.replace(/\/$/, "")}/health`, {
      method: "GET",
      signal,
    });
    done();
    if (!res.ok) return { status: STATUS.UNREACHABLE, detail: `HTTP ${res.status}` };
    const body = await res.json();
    return {
      status: STATUS.CONNECTED,
      version: body.version,
      engine: body.engine,
      tokenStrength: body.token_strength,
    };
  } catch (err) {
    done();
    return {
      status: STATUS.UNREACHABLE,
      detail:
        err.name === "AbortError"
          ? "no response — is `honjo serve` running?"
          : err.message,
    };
  }
}

/**
 * Verify a token by making an authenticated request that does no work.
 *
 * We compile an empty program rather than calling /health, because
 * /health does not check the token and would report success for a
 * wrong one.
 */
export async function verifyToken(endpoint = DEFAULT_ENDPOINT, token = "") {
  if (!token) return { status: STATUS.UNAUTHORISED, detail: "no token supplied" };
  const { signal, done } = withTimeout(HEALTH_TIMEOUT_MS);
  try {
    const res = await fetch(`${endpoint.replace(/\/$/, "")}/compile`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ source: "floor 1.0" }),
      signal,
    });
    done();
    if (res.status === 401) {
      return { status: STATUS.UNAUTHORISED, detail: "token rejected" };
    }
    if (res.status === 403) {
      return {
        status: STATUS.UNAUTHORISED,
        detail:
          "origin refused — start the engine with --origin " +
          (typeof window !== "undefined" ? window.location.origin : "<your origin>"),
      };
    }
    if (!res.ok) {
      return { status: STATUS.UNREACHABLE, detail: `HTTP ${res.status}` };
    }
    return { status: STATUS.CONNECTED };
  } catch (err) {
    done();
    return {
      status: STATUS.UNREACHABLE,
      detail: err.name === "AbortError" ? "timed out" : err.message,
    };
  }
}

/**
 * Normalise a local-engine response into the shape the workbench uses.
 *
 * The Rust `RunResult` and the JS `evaluate` return different objects;
 * both are mapped here so that no consumer has to know which produced
 * the result it is holding.
 */
function normaliseLocal(body) {
  if (!body.ok && body.error) {
    return {
      ok: false,
      engine: ENGINE.LOCAL,
      error: body.error,
      stage: body.stage || "run",
      log: [],
      cutCount: 0,
      floor: null,
      named: {},
    };
  }
  return {
    ok: Boolean(body.ok),
    engine: ENGINE.LOCAL,
    error: null,
    stage: null,
    log: Array.isArray(body.log) ? body.log : [],
    cutCount: typeof body.cut_count === "number" ? body.cut_count : 0,
    floor: typeof body.floor === "number" ? body.floor : null,
    named: body.named || {},
  };
}

function normaliseBrowser(result) {
  // Verified against the JS build: it returns
  //   { cutCount, floor, registers, named, log, ok }
  // which is the Rust RunResult with camelCase keys plus `registers`,
  // an ordered list the Rust side does not expose. Both engines report
  // the same cutCount for the same program.
  return {
    ok: Boolean(result && result.ok),
    engine: ENGINE.BROWSER,
    error: null,
    stage: null,
    log: (result && result.log) || [],
    cutCount: (result && result.cutCount) || 0,
    floor: (result && result.floor) ?? null,
    named: (result && result.named) || {},
    registers: (result && result.registers) || [],
  };
}

/**
 * Run a program.
 *
 * When `token` is present and an endpoint is reachable the reference
 * compiler runs it. Otherwise the browser build runs it, and the
 * returned `engine` field says so. There is no silent substitution:
 * a caller that requires the reference engine can check the field and
 * refuse the result.
 */
export async function run({ source, endpoint = DEFAULT_ENDPOINT, token = "" }) {
  if (token) {
    const { signal, done } = withTimeout(TIMEOUT_MS);
    try {
      const res = await fetch(`${endpoint.replace(/\/$/, "")}/run`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ source }),
        signal,
      });
      done();
      if (res.status === 401) {
        return {
          ok: false,
          engine: ENGINE.LOCAL,
          error: "token rejected by the local engine",
          stage: "auth",
          log: [],
          cutCount: 0,
          floor: null,
          named: {},
        };
      }
      const body = await res.json();
      return normaliseLocal(body);
    } catch (err) {
      done();
      // Falling back silently would be the wrong behaviour: the user
      // asked for the reference engine. Report the failure and let the
      // caller decide.
      return {
        ok: false,
        engine: ENGINE.LOCAL,
        error:
          err.name === "AbortError"
            ? "local engine did not respond within 15s"
            : `local engine unreachable: ${err.message}`,
        stage: "transport",
        log: [],
        cutCount: 0,
        floor: null,
        named: {},
      };
    }
  }

  // No token: run in the browser.
  try {
    return normaliseBrowser(jsEvaluate(source));
  } catch (err) {
    return {
      ok: false,
      engine: ENGINE.BROWSER,
      error: String(err.message || err),
      stage: "run",
      log: [],
      cutCount: 0,
      floor: null,
      named: {},
    };
  }
}

/** Compile only, for the AST view and for syntax checking as you type. */
export async function compileOnly({
  source,
  endpoint = DEFAULT_ENDPOINT,
  token = "",
}) {
  if (token) {
    const { signal, done } = withTimeout(TIMEOUT_MS);
    try {
      const res = await fetch(`${endpoint.replace(/\/$/, "")}/compile`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ source }),
        signal,
      });
      done();
      const body = await res.json();
      return { ...body, engine: ENGINE.LOCAL };
    } catch (err) {
      done();
      return { ok: false, error: String(err.message || err), engine: ENGINE.LOCAL };
    }
  }
  try {
    jsCompile(source);
    return { ok: true, engine: ENGINE.BROWSER };
  } catch (err) {
    return { ok: false, error: String(err.message || err), engine: ENGINE.BROWSER };
  }
}

/**
 * Persist connection settings.
 *
 * The token is kept in sessionStorage rather than localStorage: it dies
 * with the engine process anyway, so persisting it across browser
 * sessions would only leave a dead secret on disk.
 */
const STORE_KEY = "honjo.connection";

export function loadConnection() {
  if (typeof window === "undefined") {
    return { endpoint: DEFAULT_ENDPOINT, token: "" };
  }
  try {
    const raw = window.sessionStorage.getItem(STORE_KEY);
    if (!raw) return { endpoint: DEFAULT_ENDPOINT, token: "" };
    const parsed = JSON.parse(raw);
    return {
      endpoint: parsed.endpoint || DEFAULT_ENDPOINT,
      token: parsed.token || "",
    };
  } catch {
    return { endpoint: DEFAULT_ENDPOINT, token: "" };
  }
}

export function saveConnection({ endpoint, token }) {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.setItem(
      STORE_KEY,
      JSON.stringify({ endpoint, token })
    );
  } catch {
    /* storage unavailable (private mode); connection is per-page only */
  }
}

export function clearConnection() {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.removeItem(STORE_KEY);
  } catch {
    /* nothing to clear */
  }
}

export { DEFAULT_ENDPOINT };
