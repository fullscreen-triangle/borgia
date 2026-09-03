# Cover Letter — *Hydrogen* (MDPI)

Kundai Farai Sachikonye
Chair of Proteomics and Bioanalytics, Experimental Bioinformatics
Technical University of Munich
Maximus-von-Imhof-Forum 3, 85354 Freising, Germany
kundai.sachikonye@wzw.tum.de

---

To the Editorial Office
*Hydrogen* (ISSN 2673-4141)

Dear Editors,

I am submitting for your consideration a research article entitled **"Four
Routes to a Spectral Line: The Complete Spectra of Hydrogen, Molecular
Hydrogen and Water, Derived Four Ways, with One Route Detecting an Error in
the Others."**

## What the paper does

The manuscript presents a complete spectral atlas of atomic hydrogen,
molecular hydrogen and water — 70 lines and bands across 23 spectroscopic
modalities, spanning ten orders of magnitude in transition energy from the
21 cm hyperfine line to the inner-valence binding of water — and derives
every value by four independent methods, comparing all four against NIST
ASD, HITRAN 2020 and the Shimanouchi compilation.

The four routes share only the shell-capacity relation *C*(*n*) = 2*n*².
Beyond that they have no common machinery: one solves a differential
equation, one resolves a partition address in a physical bounded oscillator,
one deletes the carrier entirely and computes invariants of the resulting
contact sequence, and one takes a minimum cut at rest.

## What it reports, including a null result

I want to be direct about the structure of the claim, because the paper is
in part a correction of an earlier and weaker version of itself.

Three of the four routes agree with each other **bit-for-bit**. The paper
reports this as a null result rather than as a triple validation, because
the three are three derivations of a single closed form and could not have
disagreed; their common 1.1 × 10⁻⁵ residual to experiment is the
reduced-mass and QED content not carried at that order, not a discrepancy
between methods. An earlier two-route version of this work presented such
agreement as confirmation, and that was a mistake I have corrected here
rather than repeated.

The paper's substantive contribution comes from the fourth route, which
computes quantities the other three do not define. One of these — the
additivity of a quantity called circulation along a spectral ladder — is a
constraint relating *three measured lines to one another*, rather than one
line to a formula. It is therefore a check no per-line comparison can
perform.

**Run on the standard tabulated wavelengths, that check fails.** The
residual is 6.4 × 10⁻⁵, exceeding the rounding budget of the quoted digits
by a factor of 73 and rising monotonically with principal quantum number.
The cause is a unit-convention error: the Balmer wavelengths in common
tabulations are standard-air values, and they are routinely used alongside
vacuum Lyman values. Converting them to vacuum drops the residual to
7.1 × 10⁻⁷ — inside the rounding budget — and removes the trend entirely.

I regard this as the most useful result in the paper, and it is a practical
caution for anyone combining hydrogen line data across spectral regions. It
is invisible to per-line validation precisely because each line agrees with
the table it was drawn from.

The hyperfine 21 cm line receives a structural treatment: it is the only
transition in the atlas that rewrites the spin/parity label alone, which
makes it electric-dipole forbidden by a topological argument, and its
measured power sits 5.85 orders of magnitude below the smallest allowed
transition while remaining strictly nonzero. This is a quantitative
statement of what a 10⁷-year lifetime amounts to.

## Relevance to *Hydrogen*

I should be candid that this is a spectroscopic and methodological paper
rather than an energy-technology one, and I would rather say so plainly than
have the Editors discover it. My case for the fit is:

1. **Two of the three systems are hydrogen and molecular hydrogen**, and the
   atlas is hydrogen spectroscopy end to end — Lyman, Balmer, Paschen,
   Brackett and Pfund series, fine structure, the Lamb shift, the 21 cm
   line, H₂ vibrational and rotational structure, the Lyman and Werner
   bands.
2. **The convention finding is directly practical** for any work combining
   hydrogen line data across the UV and visible, including plasma
   diagnostics, combustion spectroscopy and Balmer-line temperature or
   density determination — contexts in which hydrogen spectroscopy serves
   energy-relevant measurement.
3. The journal states its scope as "chemical, physical, and engineering
   developments in hydrogen science and technology" and covers "all aspects
   of hydrogen," with no restriction on paper length so that full
   experimental and theoretical detail can be reported.

If the Editors judge the fit to be outside the journal's practical scope, I
would be grateful for a transfer recommendation rather than a desk
rejection, and I will not contest that judgement.

## Compliance

- The work is original, has not been published previously, and is not under
  consideration elsewhere.
- There is a single author; there are no competing interests to declare.
- No funding supported this work specifically.
- No human or animal subjects, and no ethical approval, are involved.
- All reference data are from public sources (NIST ASD, HITRAN 2020, NIST
  WebBook, CODATA 2022), cited in full.
- **Reproducibility:** every computed quantity is produced by two
  self-contained Python scripts included with the submission, which write a
  machine-readable results file and regenerate every figure. Expectations
  were registered in a separate file *before* the quantities were computed;
  that file is submitted unedited, including the one expectation that was
  refuted before it was diagnosed. The refuted mixed-convention calculation
  is retained in the source so the failure can be reproduced rather than
  merely read about.
- A graphical abstract is supplied in portrait format at 2500 × 1375 px,
  within MDPI's required range, composed specifically for this purpose and
  not duplicating any figure in the manuscript.

## Suggested reviewers

Reviewers with expertise in atomic and molecular spectroscopy, spectroscopic
reference data and its conventions, or hydrogen plasma diagnostics would be
best placed to assess the work. I have no reviewers to exclude.

Thank you for your consideration.

Yours sincerely,

**Kundai Farai Sachikonye**
Technical University of Munich
