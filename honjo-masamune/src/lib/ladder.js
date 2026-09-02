/**
 * Closed-ladder invariants, in the browser.
 *
 * A port of the two functions the categorical-ladder paper is built on
 * (validate_categorical_ladder.py: `circulation`, `uniformity`). They are pure
 * arithmetic, which is the whole reason this file exists: the charts can
 * RECOMPUTE the paper's quantities from the same power profiles rather than
 * replaying stored outputs, and then compare the two.
 *
 * That comparison is the point. A chart that only draws a committed number
 * cannot tell you the implementation has drifted away from the paper; one that
 * recomputes and diffs can, and `checkAgainst` below reports the residual so a
 * mismatch is visible on the panel instead of silent.
 */

/**
 * Circulation: residue deposited per circuit of a closed ladder.
 *
 *   rho = -sum log(1 - p_i)
 *
 * Additive around the cycle, invariant under rotation, and exactly zero when
 * every rung is inert. A rung of unit power is a total cut, so rho diverges.
 */
export function circulation(powers) {
  let total = 0;
  for (const p of powers) {
    if (p >= 1) return Infinity;
    total += -Math.log(1 - p);
  }
  return total;
}

/**
 * Uniformity: rotational uniformity of a cyclic power profile, in [0,1].
 *
 * 1 exactly when the profile is constant; falling as the profile becomes less
 * even. Defined as 1 - (sd/mean) so it is scale-free.
 *
 * Worth stating because it caused a registered prediction to fail: this reads
 * the DISPERSION of the rung values, not the SYMMETRY of the pattern. That is
 * why cyclohexane scores a perfect 1.000 while not being aromatic, and why u
 * moves monotonically under ring substitution.
 */
export function uniformity(powers) {
  const n = powers.length;
  if (n === 0) return 1;
  const mean = powers.reduce((a, b) => a + b, 0) / n;
  if (mean <= 0) return 1;
  const varr = powers.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  return Math.max(0, 1 - Math.sqrt(varr) / mean);
}

/** Every rotation of a cyclic profile. */
export function rotations(powers) {
  const n = powers.length;
  const out = [];
  for (let i = 0; i < n; i++) out.push([...powers.slice(i), ...powers.slice(0, i)]);
  return out;
}

/** Composite power of an open ladder: the probability of at least one cut. */
export function composite(powers) {
  let survive = 1;
  for (const p of powers) survive *= 1 - p;
  return 1 - survive;
}

/** Per-rung circulation — rho spread evenly over the cycle. */
export function rhoPerRung(powers) {
  return powers.length ? circulation(powers) / powers.length : 0;
}

/**
 * Largest deviation of an invariant across all rotations of a profile.
 * Rotation invariance is a theorem, so this should sit at machine epsilon;
 * anything larger is a defect in the port rather than a property of the ladder.
 */
export function rotationDeviation(powers, fn) {
  const vals = rotations(powers).map(fn);
  return Math.max(...vals) - Math.min(...vals);
}

/**
 * Recompute an invariant and compare with the value committed in the paper's
 * results JSON.
 *
 * Returns the residual and whether it sits within tolerance. Charts show this
 * so that "the browser agrees with the paper" is something the reader can see
 * rather than something the page asserts.
 */
export function checkAgainst(recomputed, committed, tol = 1e-9) {
  if (committed === undefined || committed === null || !Number.isFinite(recomputed)) {
    return { ok: null, residual: null };
  }
  const residual = Math.abs(recomputed - committed);
  return { ok: residual <= tol, residual };
}
