/**
 * Stage 4 of the pipeline: Allocate.
 *
 * Effort is split across steps by water-filling against a scalar shadow price.
 * Two properties matter and both are visible on the page:
 *
 *   - The optimum is UNIQUE, because each yield curve is strictly concave and
 *     the feasible set is a simplex. There is one answer, not a tie broken by
 *     implementation order.
 *
 *   - NO STEP IS EVER DROPPED. There is no heuristic, no priority list, no
 *     blacklist. A step that receives zero effort received zero because the
 *     shadow price exceeded its marginal yield at the origin -- a number the
 *     reader can inspect -- and not because something decided it was
 *     unimportant.
 *
 * Concavity fails for one class of step. A `lookup` against a record store is
 * all-or-nothing: half a retrieval retrieves nothing. Those are step functions,
 * not concave curves, so they are charged FIRST at their fixed cost and the
 * remainder is optimised over the genuinely concave ones.
 *
 * Ported from hfq/allocate.py.
 */

export class YieldSpec {
  constructor(vari, weight, { allOrNothing = false, fixedCost = 1.0 } = {}) {
    this.var = vari;
    this.weight = weight;
    this.allOrNothing = allOrNothing;
    this.fixedCost = fixedCost;
  }

  /** gamma(e) = w * log1p(e). Non-decreasing, smooth, strictly concave,
   *  gamma(0) = 0. Effort is measured in requests, not seconds: seconds are a
   *  property of someone else's hardware. */
  gamma(e) {
    return this.weight * Math.log1p(e);
  }

  /** Marginal yield. Strictly decreasing, so it inverts. */
  dgamma(e) {
    return this.weight / (1.0 + e);
  }

  /** The effort at which marginal yield equals p. */
  invert(p) {
    if (p <= 0) return Infinity;
    return Math.max(0.0, this.weight / p - 1.0);
  }
}

export class Allocation {
  constructor({ effort, shadowPrice, support, spent, budget, fixedSpent }) {
    this.effort = effort;
    this.shadowPrice = shadowPrice;
    this.support = support;
    this.spent = spent;
    this.budget = budget;
    this.fixedSpent = fixedSpent;
  }

  get(vari) {
    return this.effort[vari] ?? 0.0;
  }

  toJSON() {
    return {
      effort: this.effort,
      shadow_price: this.shadowPrice,
      support: this.support,
      spent: this.spent,
      budget: this.budget,
      fixed_spent: this.fixedSpent,
    };
  }
}

/**
 * Solve for the unique optimal split of `budget` across `specs`.
 *
 * All-or-nothing steps are charged first, IN PLAN ORDER. Order is part of the
 * answer here, not an artefact: a plan that cannot afford its third retrieval
 * should be told that, not silently handed a cheaper different third step.
 */
export function solve(specs, budget) {
  const effort = {};
  let remaining = budget;
  let fixedSpent = 0.0;

  for (const s of specs) {
    if (!s.allOrNothing) continue;
    const take = Math.min(s.fixedCost, Math.max(0.0, remaining));
    effort[s.var] = take;
    remaining -= take;
    fixedSpent += take;
  }

  const concave = specs.filter((s) => !s.allOrNothing);

  // Nothing left to optimise: the price is zero because no constraint binds on
  // the concave steps -- not because they were refused.
  if (!concave.length || remaining <= 0) {
    for (const s of concave) effort[s.var] = 0.0;
    return new Allocation({
      effort,
      shadowPrice: 0.0,
      support: [],
      spent: fixedSpent,
      budget,
      fixedSpent,
    });
  }

  const total = (p) => concave.reduce((a, s) => a + s.invert(p), 0.0);

  // Bracket. At p = max(weight) every step wants zero or less, so total <=
  // budget; halving p raises demand monotonically.
  let hi = Math.max(...concave.map((s) => s.weight));
  let lo = 1e-15;
  let guard = 0;
  while (total(hi) > remaining && guard < 200) {
    hi *= 2.0;
    guard += 1;
    if (hi > 1e18) break;
  }

  const tol = 1e-12 * Math.max(1.0, remaining);
  for (let i = 0; i < 400; i += 1) {
    const mid = 0.5 * (lo + hi);
    if (total(mid) > remaining) lo = mid;
    else hi = mid;
    if (hi - lo <= tol) break;
  }
  const p = 0.5 * (lo + hi);

  const support = [];
  let spent = fixedSpent;
  for (const s of concave) {
    const e = s.invert(p);
    effort[s.var] = e;
    spent += e;
    if (e > 0) support.push(s.var);
  }

  return new Allocation({
    effort,
    shadowPrice: p,
    support,
    spent,
    budget,
    fixedSpent,
  });
}

/**
 * The KKT residuals -- the check that the allocation really is optimal rather
 * than merely feasible.
 *
 * Two conditions. Every step ON the support must have marginal yield equal to
 * the shadow price (that is what water-filling means). Every step OFF the
 * support must have marginal yield AT THE ORIGIN no greater than the price:
 * if it were greater, spending the first unit there would have paid better
 * than wherever that unit went, and the split would not be optimal.
 */
export function kktResiduals(specs, alloc) {
  const p = alloc.shadowPrice;
  const byVar = Object.fromEntries(specs.map((s) => [s.var, s]));

  const stationarity = {};
  for (const v of alloc.support) {
    const s = byVar[v];
    if (!s) continue;
    stationarity[v] = Math.abs(s.dgamma(alloc.get(v)) - p);
  }

  let offSupportOk = true;
  const offSupport = {};
  for (const s of specs) {
    if (s.allOrNothing || alloc.support.includes(s.var)) continue;
    const d0 = s.dgamma(0.0);
    offSupport[s.var] = d0;
    if (d0 > p + 1e-9) offSupportOk = false;
  }

  const budgetBinds =
    Math.abs(alloc.spent - alloc.budget) <= 1e-6 * Math.max(1.0, alloc.budget);

  return {
    shadow_price: p,
    stationarity,
    max_stationarity_residual: Object.keys(stationarity).length
      ? Math.max(...Object.values(stationarity))
      : 0.0,
    off_support: offSupport,
    off_support_satisfied: offSupportOk,
    budget_binds: budgetBinds,
    spent: alloc.spent,
    budget: alloc.budget,
  };
}
