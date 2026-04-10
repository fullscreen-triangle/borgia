/**
 * PartitionObserver — GPU Partition Observation Apparatus
 * ======================================================
 * WebGL2 fragment shader that evaluates the partition observation function
 * at every pixel simultaneously. The rendering IS the observation.
 *
 * Implements Instruments I (observation) and II (interference).
 */
import { useEffect, useRef, useCallback, useState } from "react";

// ─── GLSL Shaders ───────────────────────────────────────────────────

const VERTEX_SHADER = `#version 300 es
in vec2 a_position;
out vec2 v_uv;
void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

const OBSERVATION_SHADER = `#version 300 es
precision highp float;

uniform vec3 u_sentropy;       // (S_k, S_t, S_e)
uniform int u_numModes;
uniform float u_modes[32];     // normalised frequencies
uniform float u_time;

in vec2 v_uv;
out vec4 fragColor;

float gaussian(float x, float center, float width) {
  float dx = (x - center) / width;
  return exp(-dx * dx);
}

void main() {
  float S_k = u_sentropy.x;
  float S_t = u_sentropy.y;
  float S_e = u_sentropy.z;

  // Cell width from configuration entropy
  float sigma = 0.015 * (1.0 - 0.5 * S_k);

  // Sum Gaussians at each vibrational mode
  float intensity = 0.0;
  for (int i = 0; i < 32; i++) {
    if (i >= u_numModes) break;
    intensity += gaussian(v_uv.x, u_modes[i], sigma);
  }

  // Temporal bandwidth envelope
  float bw = 0.1 + 0.4 * S_t;
  float envelope = gaussian(v_uv.x, 0.5, bw);
  intensity *= envelope;

  // Partition depth fringes
  float phase = v_uv.x * 50.0 * S_e;
  float depth_mod = 1.0 + 0.3 * S_e * sin(phase);
  intensity *= depth_mod;

  // Phase modulation in Y-axis
  float phase_mod = 0.8 + 0.2 * sin(v_uv.y * 6.28318 + u_time * 0.5);
  intensity *= phase_mod;

  intensity = clamp(intensity, 0.0, 1.0);

  // Encode as colour: warm palette
  vec3 lo = vec3(0.02, 0.02, 0.05);
  vec3 mid = vec3(0.1, 0.4, 0.6);
  vec3 hi = vec3(0.3, 0.9, 0.85);
  vec3 col = mix(lo, mid, smoothstep(0.0, 0.3, intensity));
  col = mix(col, hi, smoothstep(0.3, 0.8, intensity));

  fragColor = vec4(col, 1.0);
}`;

const INTERFERENCE_SHADER = `#version 300 es
precision highp float;

uniform sampler2D u_texA;
uniform sampler2D u_texB;
uniform float u_time;

in vec2 v_uv;
out vec4 fragColor;

void main() {
  vec3 a = texture(u_texA, v_uv).rgb;
  vec3 b = texture(u_texB, v_uv).rgb;

  float ampA = length(a);
  float ampB = length(b);
  float phaseA = atan(a.g, a.r);
  float phaseB = atan(b.g, b.r);

  float visibility = 0.5 + 0.5 * ampA * ampB * cos(phaseA + phaseB);

  // Colour: constructive = teal, destructive = dark
  vec3 constructive = vec3(0.34, 0.9, 0.85);
  vec3 destructive = vec3(0.03, 0.03, 0.06);
  vec3 col = mix(destructive, constructive, visibility);

  fragColor = vec4(col, 1.0);
}`;

// ─── WebGL Helpers ──────────────────────────────────────────────────

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function createProgram(gl, vertSrc, fragSrc) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  if (!vs || !fs) return null;
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error("Link error:", gl.getProgramInfoLog(prog));
    return null;
  }
  return prog;
}

function createFBO(gl, w, h) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { fbo, tex };
}

// ─── React Component ────────────────────────────────────────────────

export default function PartitionObserver({
  molA, molB = null, width = 512, height = 256, showInterference = false
}) {
  const canvasRef = useRef(null);
  const glRef = useRef(null);
  const programsRef = useRef({});
  const fboRef = useRef({});
  const quadRef = useRef(null);
  const frameRef = useRef(0);
  const [visibility, setVisibility] = useState(null);

  const setupGL = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false });
    if (!gl) { console.error("WebGL2 not supported"); return; }
    glRef.current = gl;

    // Compile programs
    const obsProg = createProgram(gl, VERTEX_SHADER, OBSERVATION_SHADER);
    const intProg = createProgram(gl, VERTEX_SHADER, INTERFERENCE_SHADER);
    programsRef.current = { obs: obsProg, int: intProg };

    // Fullscreen quad
    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
    quadRef.current = quad;

    // FBOs for offscreen observation textures
    fboRef.current.a = createFBO(gl, width, height);
    fboRef.current.b = createFBO(gl, width, height);
  }, [width, height]);

  const drawObservation = useCallback((gl, prog, mol, fbo, time) => {
    if (!prog || !mol) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo ? fbo.fbo : null);
    gl.viewport(0, 0, width, height);
    gl.useProgram(prog);

    // Bind quad
    gl.bindBuffer(gl.ARRAY_BUFFER, quadRef.current);
    const loc = gl.getAttribLocation(prog, "a_position");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    // Uniforms
    gl.uniform3f(gl.getUniformLocation(prog, "u_sentropy"), mol.S_k, mol.S_t, mol.S_e);
    gl.uniform1i(gl.getUniformLocation(prog, "u_numModes"), mol.modes.length);
    gl.uniform1f(gl.getUniformLocation(prog, "u_time"), time);

    const normModes = new Float32Array(32);
    mol.modes.forEach((w, i) => { if (i < 32) normModes[i] = w / 4401.0; });
    gl.uniform1fv(gl.getUniformLocation(prog, "u_modes"), normModes);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }, [width, height]);

  const drawInterference = useCallback((gl, prog, texA, texB, time) => {
    if (!prog) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.useProgram(prog);

    gl.bindBuffer(gl.ARRAY_BUFFER, quadRef.current);
    const loc = gl.getAttribLocation(prog, "a_position");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texA);
    gl.uniform1i(gl.getUniformLocation(prog, "u_texA"), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, texB);
    gl.uniform1i(gl.getUniformLocation(prog, "u_texB"), 1);

    gl.uniform1f(gl.getUniformLocation(prog, "u_time"), time);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // Read back and integrate visibility
    const pixels = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    let totalVis = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      totalVis += (pixels[i] + pixels[i+1] + pixels[i+2]) / (3 * 255);
    }
    setVisibility(totalVis / (width * height));
  }, [width, height]);

  // Render loop
  useEffect(() => {
    setupGL();
    let animId;
    const render = (t) => {
      const gl = glRef.current;
      const progs = programsRef.current;
      if (!gl || !progs.obs) { animId = requestAnimationFrame(render); return; }

      const time = t * 0.001;

      if (showInterference && molA && molB) {
        // Observe A into FBO A
        drawObservation(gl, progs.obs, molA, fboRef.current.a, time);
        // Observe B into FBO B
        drawObservation(gl, progs.obs, molB, fboRef.current.b, time);
        // Interference to screen
        drawInterference(gl, progs.int, fboRef.current.a.tex, fboRef.current.b.tex, time);
      } else if (molA) {
        // Single observation to screen
        drawObservation(gl, progs.obs, molA, null, time);
      }

      animId = requestAnimationFrame(render);
    };
    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [molA, molB, showInterference, setupGL, drawObservation, drawInterference]);

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="w-full rounded-lg border border-neutral-800"
        style={{ imageRendering: "pixelated" }}
      />
      {showInterference && visibility !== null && (
        <div className="absolute top-2 right-2 bg-black/70 px-3 py-1 rounded text-sm font-mono">
          <span className="text-neutral-400">V̄ = </span>
          <span className="text-[#58E6D9]">{visibility.toFixed(4)}</span>
        </div>
      )}
    </div>
  );
}
