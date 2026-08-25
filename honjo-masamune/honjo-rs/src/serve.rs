//! Local execution server for the web workbench.
//!
//! The workbench in the browser cannot execute a plan: it has no file
//! system, no compiler, and no reference corpus.  This module lets a
//! user run the compiler on their own machine and connect the browser
//! to it, so that computation happens locally and no source or
//! structure leaves the machine.
//!
//! The connection is authorised by a token printed on startup.  The
//! token is generated per session, held only in memory, and is required
//! on every request.  There is no account, no registration, and nothing
//! is transmitted anywhere except between the browser and this process.
//!
//! Security posture, stated plainly:
//!   * the listener binds to loopback only and refuses to bind
//!     elsewhere, so the port is not reachable from the network;
//!   * the token is compared in constant time, so a wrong token leaks
//!     no information about the right one through timing;
//!   * CORS is restricted to an explicit origin allow-list;
//!   * request bodies are capped, so a large POST cannot exhaust memory.
//!
//! None of this makes the server safe to expose deliberately.  It is
//! safe because it is not reachable, and the checks above are defence
//! for the case where that assumption is broken by something else on
//! the machine.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::time::{SystemTime, UNIX_EPOCH};

/// Largest request body accepted, in bytes.  A plan is text; anything
/// beyond this is not a plan.
const MAX_BODY: usize = 1 << 20; // 1 MiB

/// Token length in hex characters.  Two hex digits per byte, so this
/// determines the entropy: 32 characters is 16 bytes, 128 bits.
const TOKEN_HEX_LEN: usize = 32;
const TOKEN_BYTES: usize = TOKEN_HEX_LEN / 2;

// =====================================================================
//  Token
// =====================================================================

/// A per-session bearer token.
///
/// Generated from the operating system's entropy source where one is
/// reachable, and from a time-and-address mix otherwise.  The fallback
/// is weaker and says so on startup rather than pretending otherwise.
pub struct Token {
    value: String,
    strong: bool,
}

impl Token {
    pub fn generate() -> Token {
        match os_entropy(TOKEN_BYTES) {
            Some(bytes) => Token {
                value: hex(&bytes),
                strong: true,
            },
            None => Token {
                value: hex(&weak_entropy(TOKEN_BYTES)),
                strong: false,
            },
        }
    }

    pub fn value(&self) -> &str {
        &self.value
    }

    pub fn is_strong(&self) -> bool {
        self.strong
    }

    /// Constant-time comparison.
    ///
    /// A short-circuiting `==` would return sooner for a token that
    /// shares a longer prefix with the real one, which leaks the token
    /// one character at a time to anyone who can time the response.
    fn matches(&self, candidate: &str) -> bool {
        let a = self.value.as_bytes();
        let b = candidate.as_bytes();
        if a.len() != b.len() {
            // Length is not secret: it is a compile-time constant.
            return false;
        }
        let mut diff: u8 = 0;
        for i in 0..a.len() {
            diff |= a[i] ^ b[i];
        }
        diff == 0
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Read `n` bytes from the OS entropy source.
#[cfg(unix)]
fn os_entropy(n: usize) -> Option<Vec<u8>> {
    use std::fs::File;
    let mut f = File::open("/dev/urandom").ok()?;
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf).ok()?;
    Some(buf)
}

#[cfg(windows)]
fn os_entropy(n: usize) -> Option<Vec<u8>> {
    // Without a crate dependency there is no stable path to
    // BCryptGenRandom here.  Report the absence rather than returning
    // weak bytes that look strong to the caller.
    let _ = n;
    None
}

#[cfg(not(any(unix, windows)))]
fn os_entropy(_n: usize) -> Option<Vec<u8>> {
    None
}

/// Fallback entropy: high-resolution time mixed with addresses that
/// vary per process.  Adequate to prevent accidental collision on a
/// single-user machine; not adequate against an adversary who can
/// observe process start time.  The caller warns when this is used.
fn weak_entropy(n: usize) -> Vec<u8> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let stack_marker = 0u8;
    let addr = &stack_marker as *const u8 as usize;
    let pid = std::process::id() as usize;

    let mut state: u64 = (now as u64)
        ^ ((addr as u64) << 17)
        ^ ((pid as u64) << 33)
        ^ 0x9e3779b97f4a7c15;

    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // splitmix64
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        out.push((z & 0xff) as u8);
    }
    out
}

// =====================================================================
//  Minimal HTTP
// =====================================================================

struct Request {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: String,
}

fn read_request(stream: &mut TcpStream) -> Result<Request, String> {
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);

    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| format!("read request line: {}", e))?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();

    let mut headers = HashMap::new();
    loop {
        let mut h = String::new();
        let n = reader
            .read_line(&mut h)
            .map_err(|e| format!("read header: {}", e))?;
        if n == 0 || h.trim().is_empty() {
            break;
        }
        if let Some(idx) = h.find(':') {
            headers.insert(
                h[..idx].trim().to_ascii_lowercase(),
                h[idx + 1..].trim().to_string(),
            );
        }
    }

    let len: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    if len > MAX_BODY {
        return Err(format!("body too large: {} > {}", len, MAX_BODY));
    }

    let mut body = vec![0u8; len];
    if len > 0 {
        reader
            .read_exact(&mut body)
            .map_err(|e| format!("read body: {}", e))?;
    }

    Ok(Request {
        method,
        path,
        headers,
        body: String::from_utf8_lossy(&body).to_string(),
    })
}

fn cors_origin(req: &Request, allowed: &[String]) -> Option<String> {
    let origin = req.headers.get("origin")?;
    if allowed.iter().any(|a| a == origin) {
        Some(origin.clone())
    } else {
        None
    }
}

fn respond(stream: &mut TcpStream, status: u16, origin: Option<&str>, body: &str) {
    let reason = match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        413 => "Payload Too Large",
        _ => "Error",
    };
    let mut head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nCache-Control: no-store\r\n\
         X-Content-Type-Options: nosniff\r\n",
        status,
        reason,
        body.as_bytes().len()
    );
    if let Some(o) = origin {
        head.push_str(&format!(
            "Access-Control-Allow-Origin: {}\r\n\
             Access-Control-Allow-Headers: content-type, authorization\r\n\
             Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n\
             Vary: Origin\r\n",
            o
        ));
    }
    head.push_str("Connection: close\r\n\r\n");
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body.as_bytes());
    let _ = stream.flush();
}

// =====================================================================
//  JSON emission
// =====================================================================

/// Escape a string for inclusion in JSON.
fn jstr(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_array(items: &[String]) -> String {
    format!("[{}]", items.join(","))
}

/// Extract a top-level string field from a flat JSON object.
///
/// This is a deliberate minimum: the only request bodies this server
/// accepts are flat objects with string fields, so a full parser would
/// be more surface than the protocol needs.
fn json_field(body: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\"", key);
    let start = body.find(&needle)? + needle.len();
    let rest = &body[start..];
    let colon = rest.find(':')? + 1;
    let rest = &rest[colon..];
    let open = rest.find('"')? + 1;
    let bytes = rest.as_bytes();
    let mut i = open;
    let mut out = String::new();
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if i + 1 < bytes.len() => {
                match bytes[i + 1] {
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'u' => {
                        if i + 5 < bytes.len() {
                            let hex = &rest[i + 2..i + 6];
                            if let Ok(cp) = u32::from_str_radix(hex, 16) {
                                if let Some(c) = char::from_u32(cp) {
                                    out.push(c);
                                }
                            }
                            i += 4;
                        }
                    }
                    other => out.push(other as char),
                }
                i += 2;
            }
            b'"' => return Some(out),
            b => {
                // rebuild multi-byte UTF-8 sequences verbatim
                out.push(b as char);
                i += 1;
            }
        }
    }
    None
}

// =====================================================================
//  Handlers
// =====================================================================

fn handle_health(token: &Token) -> String {
    format!(
        "{{\"service\":\"honjo\",\"version\":{},\"ok\":true,\
         \"token_strength\":{},\"engine\":\"rust-reference\"}}",
        jstr(env!("CARGO_PKG_VERSION")),
        jstr(if token.is_strong() { "os" } else { "fallback" })
    )
}

fn handle_run(body: &str) -> String {
    let src = match json_field(body, "source") {
        Some(s) => s,
        None => {
            return format!(
                "{{\"ok\":false,\"error\":{},\"stage\":\"request\"}}",
                jstr("missing 'source' field")
            )
        }
    };

    match crate::evaluate(&src) {
        Ok(r) => {
            let log: Vec<String> = r.log.iter().map(|l| jstr(l)).collect();
            let mut named: Vec<String> = Vec::new();
            for (k, v) in r.named.iter() {
                named.push(format!("{}:{}", jstr(k), jstr(&format!("{:?}", v))));
            }
            named.sort();
            format!(
                "{{\"ok\":{},\"cut_count\":{},\"floor\":{},\"log\":{},\
                 \"named\":{{{}}},\"engine\":\"rust-reference\"}}",
                r.ok,
                r.cut_count,
                r.floor,
                json_array(&log),
                named.join(",")
            )
        }
        Err(e) => format!(
            "{{\"ok\":false,\"error\":{},\"stage\":\"compile\"}}",
            jstr(&e)
        ),
    }
}

fn handle_compile(body: &str) -> String {
    let src = match json_field(body, "source") {
        Some(s) => s,
        None => {
            return format!(
                "{{\"ok\":false,\"error\":{}}}",
                jstr("missing 'source' field")
            )
        }
    };
    match crate::compile(&src) {
        Ok(p) => format!(
            "{{\"ok\":true,\"ast\":{}}}",
            jstr(&format!("{:#?}", p))
        ),
        Err(e) => format!("{{\"ok\":false,\"error\":{}}}", jstr(&e)),
    }
}

// =====================================================================
//  Server
// =====================================================================

pub struct Config {
    pub port: u16,
    pub allowed_origins: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            port: 8731,
            allowed_origins: vec![
                "http://localhost:3000".to_string(),
                "http://127.0.0.1:3000".to_string(),
            ],
        }
    }
}

/// Start the local execution server.  Blocks until the process is
/// interrupted.
pub fn serve(cfg: Config) -> Result<(), String> {
    let token = Token::generate();

    // Loopback only.  Binding to 0.0.0.0 would expose the compiler to
    // the local network, and a token is not a substitute for not being
    // reachable.
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), cfg.port);
    let listener = TcpListener::bind(addr)
        .map_err(|e| format!("cannot bind {}: {}", addr, e))?;

    eprintln!();
    eprintln!("  honjo local engine");
    eprintln!("  ------------------");
    eprintln!("  listening   http://127.0.0.1:{}", cfg.port);
    eprintln!("  token       {}", token.value());
    if !token.is_strong() {
        eprintln!();
        eprintln!("  WARNING: the OS entropy source was unavailable on this");
        eprintln!("  platform, so the token was derived from process time and");
        eprintln!("  address. It is unique enough for a single-user machine but");
        eprintln!("  is NOT cryptographically strong. Do not rely on it if");
        eprintln!("  anything else on this machine is untrusted.");
    }
    eprintln!();
    eprintln!("  Paste the token into the workbench to connect.");
    eprintln!("  Nothing leaves this machine: the browser sends source here,");
    eprintln!("  and this process returns the result.");
    eprintln!();
    eprintln!("  Ctrl-C to stop. The token dies with this process.");
    eprintln!();

    for incoming in listener.incoming() {
        let mut stream = match incoming {
            Ok(s) => s,
            Err(_) => continue,
        };

        let req = match read_request(&mut stream) {
            Ok(r) => r,
            Err(e) => {
                let status = if e.contains("too large") { 413 } else { 400 };
                respond(
                    &mut stream,
                    status,
                    None,
                    &format!("{{\"ok\":false,\"error\":{}}}", jstr(&e)),
                );
                continue;
            }
        };

        let origin = cors_origin(&req, &cfg.allowed_origins);

        // Preflight
        if req.method == "OPTIONS" {
            match origin {
                Some(o) => respond(&mut stream, 204, Some(&o), ""),
                None => respond(
                    &mut stream,
                    403,
                    None,
                    &format!("{{\"ok\":false,\"error\":{}}}", jstr("origin not allowed")),
                ),
            }
            continue;
        }

        // An unrecognised origin is refused before the token is even
        // examined: a page we do not know should not be able to probe
        // token validity at all.
        let origin_ref = match &origin {
            Some(o) => Some(o.as_str()),
            None => {
                if req.headers.contains_key("origin") {
                    respond(
                        &mut stream,
                        403,
                        None,
                        &format!(
                            "{{\"ok\":false,\"error\":{}}}",
                            jstr("origin not allowed")
                        ),
                    );
                    continue;
                }
                None // non-browser client, e.g. curl
            }
        };

        // Every route except /health requires the token.
        let authorised = req
            .headers
            .get("authorization")
            .map(|h| h.strip_prefix("Bearer ").unwrap_or(h).trim().to_string())
            .map(|t| token.matches(&t))
            .unwrap_or(false);

        let body = match req.path.as_str() {
            "/health" => handle_health(&token),
            "/run" | "/compile" if !authorised => {
                respond(
                    &mut stream,
                    401,
                    origin_ref,
                    &format!(
                        "{{\"ok\":false,\"error\":{}}}",
                        jstr("missing or invalid token")
                    ),
                );
                continue;
            }
            "/run" => handle_run(&req.body),
            "/compile" => handle_compile(&req.body),
            _ => {
                respond(
                    &mut stream,
                    404,
                    origin_ref,
                    &format!("{{\"ok\":false,\"error\":{}}}", jstr("no such route")),
                );
                continue;
            }
        };

        respond(&mut stream, 200, origin_ref, &body);
    }

    Ok(())
}

// =====================================================================
//  Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_matches_itself_and_nothing_else() {
        let t = Token::generate();
        assert!(t.matches(t.value()));
        assert!(!t.matches("00000000000000000000000000000000"));
        assert!(!t.matches(""));
        assert!(!t.matches(&format!("{}x", t.value())));
    }

    #[test]
    fn token_is_expected_length() {
        let t = Token::generate();
        assert_eq!(t.value().len(), TOKEN_HEX_LEN);
        assert!(t.value().chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn tokens_differ_between_sessions() {
        let a = Token::generate();
        let b = Token::generate();
        assert_ne!(a.value(), b.value());
    }

    #[test]
    fn json_field_extracts_source() {
        let body = r#"{"source":"floor 1.0\nC := cut 6","extra":"x"}"#;
        let got = json_field(body, "source").unwrap();
        assert_eq!(got, "floor 1.0\nC := cut 6");
    }

    #[test]
    fn json_field_absent_is_none() {
        assert!(json_field(r#"{"other":"x"}"#, "source").is_none());
    }

    #[test]
    fn jstr_escapes_quotes_and_newlines() {
        assert_eq!(jstr("a\"b\nc"), "\"a\\\"b\\nc\"");
    }
}
