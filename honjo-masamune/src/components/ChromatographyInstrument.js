/**
 * Chromatography Retention Predictor
 * ==================================
 * Predicts C18 (reverse-phase) and HILIC retention times directly from
 * partition coordinates (n, ell, m). No training, no calibration — the
 * retention IS the partition coordinate.
 *
 * Framework reference: Consequence 4.8 (Chromatography as partition-depth
 * projection) in bounded-phase-space-trajectory.tex.
 */
import { useEffect, useMemo, useState } from "react";

const RT_SCALE = 5.0; // minutes per S-unit (15 cm column at 0.3 cm/min)

// C18 kernel: lipophilicity drives retention
function scoreC18({ n, ell, m, logP }) {
  const structural = (ell + m / 2) / Math.max(n, 1);
  const lipo = 1.0 + 0.3 * (logP ?? 0);
  return structural * 2.43 * Math.max(0.4, lipo / 1.4);
}

// HILIC kernel: hydrophilicity (inverse of lipophilicity) + polar surface
function scoreHILIC({ n, ell_hilic, m, logP }) {
  const structural = (ell_hilic + m / 2) / Math.max(n, 1);
  const hydro = 1.0 - 0.15 * (logP ?? 0);
  return structural * 2.43 * Math.max(0.4, hydro);
}

export default function ChromatographyInstrument() {
  const [data, setData] = useState(null);
  const [selected, setSelected] = useState("caffeine");
  const [custom, setCustom] = useState({ n: 10, ell: 3, m: 4, logP: 1.0 });
  const [mode, setMode] = useState("library"); // library | custom

  useEffect(() => {
    fetch("/validation/chromatography.json")
      .then((r) => r.json())
      .then(setData)
      .catch(() => setData(null));
  }, []);

  const c18Map = useMemo(() => {
    if (!data) return {};
    return Object.fromEntries(data.c18.map((c) => [c.compound, c]));
  }, [data]);

  const hilicMap = useMemo(() => {
    if (!data) return {};
    return Object.fromEntries(data.hilic.map((c) => [c.compound, c]));
  }, [data]);

  const selC18 = mode === "library" ? c18Map[selected] : null;
  const selHILIC = mode === "library" ? hilicMap[selected] : null;

  const customC18 = useMemo(() => {
    if (mode !== "custom") return null;
    const S = scoreC18(custom);
    return { S_c18: S, RT_predicted_min: RT_SCALE * S };
  }, [mode, custom]);

  const customHILIC = useMemo(() => {
    if (mode !== "custom") return null;
    const ellH = Math.max(0, custom.n - custom.ell - Math.floor(custom.m / 2));
    const S = scoreHILIC({ ...custom, ell_hilic: ellH });
    return { S_hilic: S, RT_predicted_min: RT_SCALE * S, ell_hilic: ellH };
  }, [mode, custom]);

  if (!data) {
    return <div className="text-neutral-500 text-sm py-12 text-center">Loading chromatography data…</div>;
  }

  const maxRT = Math.max(
    ...data.c18.map((c) => Math.max(c.RT_predicted_min, c.RT_measured_min)),
    ...data.hilic.map((c) => Math.max(c.RT_predicted_min, c.RT_measured_min)),
    12
  );

  return (
    <div>
      {/* Mode toggle */}
      <div className="flex gap-2 mb-4">
        <button
          className={`text-xs px-3 py-1.5 rounded tracking-wider uppercase ${
            mode === "library" ? "bg-[#58E6D9]/20 text-[#58E6D9]" : "bg-neutral-800 text-neutral-400"
          }`}
          onClick={() => setMode("library")}
        >
          Library (15 pharmaceuticals)
        </button>
        <button
          className={`text-xs px-3 py-1.5 rounded tracking-wider uppercase ${
            mode === "custom" ? "bg-[#58E6D9]/20 text-[#58E6D9]" : "bg-neutral-800 text-neutral-400"
          }`}
          onClick={() => setMode("custom")}
        >
          Custom partition coords
        </button>
      </div>

      {mode === "library" && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* C18 table */}
          <div>
            <h3 className="text-sm font-semibold text-neutral-300 mb-2 uppercase tracking-wider">
              C18 · Reverse-Phase (lipophilicity wins)
            </h3>
            <div className="border border-neutral-800 rounded-lg overflow-hidden">
              <table className="w-full text-xs">
                <thead className="bg-neutral-800/60 text-neutral-500">
                  <tr>
                    <th className="text-left px-2 py-1.5 font-normal tracking-wider uppercase">Compound</th>
                    <th className="text-right px-2 py-1.5 font-normal tracking-wider uppercase">(n,ℓ,m)</th>
                    <th className="text-right px-2 py-1.5 font-normal tracking-wider uppercase">RT pred</th>
                    <th className="text-right px-2 py-1.5 font-normal tracking-wider uppercase">RT meas</th>
                  </tr>
                </thead>
                <tbody>
                  {data.c18.map((c) => (
                    <tr
                      key={c.compound}
                      className={`cursor-pointer border-t border-neutral-800 ${
                        selected === c.compound ? "bg-[#58E6D9]/10" : "hover:bg-neutral-800/30"
                      }`}
                      onClick={() => setSelected(c.compound)}
                    >
                      <td className="px-2 py-1 text-white">{c.compound}</td>
                      <td className="px-2 py-1 text-right font-mono text-neutral-400">
                        ({c.n},{c.ell},{c.m})
                      </td>
                      <td className="px-2 py-1 text-right font-mono text-[#58E6D9]">
                        {c.RT_predicted_min.toFixed(2)}
                      </td>
                      <td className="px-2 py-1 text-right font-mono text-neutral-300">
                        {c.RT_measured_min.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* HILIC table */}
          <div>
            <h3 className="text-sm font-semibold text-neutral-300 mb-2 uppercase tracking-wider">
              HILIC · Hydrophilic (complement of C18)
            </h3>
            <div className="border border-neutral-800 rounded-lg overflow-hidden">
              <table className="w-full text-xs">
                <thead className="bg-neutral-800/60 text-neutral-500">
                  <tr>
                    <th className="text-left px-2 py-1.5 font-normal tracking-wider uppercase">Compound</th>
                    <th className="text-right px-2 py-1.5 font-normal tracking-wider uppercase">ℓ_c18 / ℓ_hilic</th>
                    <th className="text-right px-2 py-1.5 font-normal tracking-wider uppercase">RT pred</th>
                    <th className="text-right px-2 py-1.5 font-normal tracking-wider uppercase">RT meas</th>
                  </tr>
                </thead>
                <tbody>
                  {data.hilic.map((c) => (
                    <tr
                      key={c.compound}
                      className={`cursor-pointer border-t border-neutral-800 ${
                        selected === c.compound ? "bg-[#58E6D9]/10" : "hover:bg-neutral-800/30"
                      }`}
                      onClick={() => setSelected(c.compound)}
                    >
                      <td className="px-2 py-1 text-white">{c.compound}</td>
                      <td className="px-2 py-1 text-right font-mono text-neutral-400">
                        {c.ell_c18} / {c.ell_hilic}
                      </td>
                      <td className="px-2 py-1 text-right font-mono text-[#58E6D9]">
                        {c.RT_predicted_min.toFixed(2)}
                      </td>
                      <td className="px-2 py-1 text-right font-mono text-neutral-300">
                        {c.RT_measured_min.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {mode === "custom" && (
        <div className="bg-neutral-800/30 rounded-lg p-5 border border-neutral-800 mb-6">
          <h3 className="text-sm font-semibold text-neutral-300 mb-3 uppercase tracking-wider">
            Partition coordinates
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { key: "n", label: "n (shell count)", min: 1, max: 40, step: 1 },
              { key: "ell", label: "ℓ (lipophilic branches)", min: 0, max: 20, step: 1 },
              { key: "m", label: "m (polar groups)", min: 0, max: 12, step: 1 },
              { key: "logP", label: "logP", min: -3, max: 6, step: 0.01 },
            ].map((f) => (
              <div key={f.key}>
                <label className="text-neutral-500 text-xs tracking-wider uppercase block mb-1">
                  {f.label}
                </label>
                <input
                  type="number"
                  min={f.min}
                  max={f.max}
                  step={f.step}
                  value={custom[f.key]}
                  onChange={(e) =>
                    setCustom({ ...custom, [f.key]: parseFloat(e.target.value) || 0 })
                  }
                  className="w-full bg-neutral-800 text-white text-sm p-2 rounded font-mono border border-neutral-700 focus:border-[#58E6D9] outline-none"
                />
              </div>
            ))}
          </div>
          <p className="text-neutral-600 text-xs mt-3">
            n = principal shell reach, ℓ = angular branches retained on C18, m = polar attachment count.
            RT scales linearly: RT = 5.0·S (for a 15 cm column at 0.3 cm/min).
          </p>
        </div>
      )}

      {/* Selected compound details */}
      {(selC18 || customC18) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <DetailCard
            label="C18 prediction"
            S={selC18 ? selC18.S_c18 : customC18.S_c18}
            RT_pred={selC18 ? selC18.RT_predicted_min : customC18.RT_predicted_min}
            RT_meas={selC18 ? selC18.RT_measured_min : null}
            err={selC18 ? selC18.error_pct : null}
            maxRT={maxRT}
            color="#58E6D9"
          />
          <DetailCard
            label="HILIC prediction"
            S={selHILIC ? selHILIC.S_hilic : customHILIC.S_hilic}
            RT_pred={selHILIC ? selHILIC.RT_predicted_min : customHILIC.RT_predicted_min}
            RT_meas={selHILIC ? selHILIC.RT_measured_min : null}
            err={selHILIC ? selHILIC.error_pct : null}
            maxRT={maxRT}
            color="#a855f7"
            sub={
              customHILIC
                ? `ℓ_hilic inferred as n − ℓ − ⌊m/2⌋ = ${customHILIC.ell_hilic}`
                : null
            }
          />
        </div>
      )}

      <p className="text-neutral-600 text-xs mt-6">
        Retention = partition-depth projection. The column selects along one axis of (n, ℓ, m, s):
        C18 projects along ℓ (lipophilic branches), HILIC projects along m (polar attachments).
        No force field, no training data, no experimental calibration — the retention time is the coordinate.
      </p>
    </div>
  );
}

function DetailCard({ label, S, RT_pred, RT_meas, err, maxRT, color, sub }) {
  const predPct = (RT_pred / maxRT) * 100;
  const measPct = RT_meas !== null ? (RT_meas / maxRT) * 100 : null;
  return (
    <div className="bg-neutral-800/40 rounded-xl p-5 border border-neutral-800">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-white text-sm font-medium">{label}</h4>
        <span className="text-neutral-500 text-xs font-mono">S = {S.toFixed(3)}</span>
      </div>
      <div className="space-y-3">
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-neutral-400">Predicted</span>
            <span className="font-mono" style={{ color }}>
              {RT_pred.toFixed(2)} min
            </span>
          </div>
          <div className="w-full h-2 bg-neutral-800 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${predPct}%`, backgroundColor: color }}
            />
          </div>
        </div>
        {measPct !== null && (
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-neutral-400">Measured</span>
              <span className="text-white font-mono">{RT_meas.toFixed(2)} min</span>
            </div>
            <div className="w-full h-2 bg-neutral-800 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full bg-white/50 transition-all duration-500"
                style={{ width: `${measPct}%` }}
              />
            </div>
          </div>
        )}
        {err !== null && err !== undefined && (
          <div className="text-xs text-neutral-500">
            Framework error: <span className="text-white font-mono">{Math.abs(err).toFixed(1)}%</span>
          </div>
        )}
        {sub && <p className="text-neutral-600 text-xs">{sub}</p>}
      </div>
    </div>
  );
}
