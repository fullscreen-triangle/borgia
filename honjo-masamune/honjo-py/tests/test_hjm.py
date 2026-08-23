"""Conformance tests.

Organised by the property each one checks.  Several are negative
controls: their pass condition is a refusal.
"""

from __future__ import annotations

import os

import pytest

from hjm import Prov, Verdict, run_honjo, run_plan
from hjm.core.chem import covalent_valence, shell_capacity, vacancy
from hjm.core.verdict import Label
from hjm.masamune.capability import capability, missing
from hjm.masamune.translate import translate_smiles

EX = os.path.join(os.path.dirname(__file__), "..", "examples")


# ------------------------------------------------------------ chemistry


def test_shell_capacity():
    assert [shell_capacity(n) for n in range(1, 6)] == [2, 8, 18, 32, 50]


@pytest.mark.parametrize(
    "z,nu", [(1, 1), (2, 0), (6, 4), (7, 3), (8, 2), (9, 1), (10, 0), (18, 0)]
)
def test_vacancy(z, nu):
    assert vacancy(z) == nu


def test_noble_iff_zero_vacancy():
    nobles = {2, 10, 18}
    for z in range(1, 19):
        assert (vacancy(z) == 0) == (z in nobles), z


def test_covalent_valence_second_period():
    # C 4, N 3, O 2, F 1, Ne 0 -- read off the vacancy arithmetic
    assert [covalent_valence(z) for z in (6, 7, 8, 9, 10)] == [4, 3, 2, 1, 0]


# ------------------------------------------------------- P1: no sharp cut


def test_no_zero_residue_value():
    out = run_honjo("floor 1.0\nC := cut 6")
    for b in out["bindings"]:
        assert b["residue"] >= b["floor"] > 0


def test_resolution_gate_refuses():
    """A floor finer than the target resolution is refused, not run."""
    out = run_honjo("floor 1e-12\nC := cut 6", eps_target=1e-9)
    assert out["status"] == "refused"
    assert any(l.get("level") == "refusal" for l in out["log"])


# ------------------------------------------------- P2: provenance monotone


def test_supplied_never_becomes_stated():
    v = translate_smiles("CCO")
    g = v.value
    assert g.provenance() is Prov.SUPPLIED
    # the graph tag is the max over its own elements
    assert any(a.prov is Prov.SUPPLIED for a in g.atoms.values())


def test_prov_join_is_max():
    assert Prov.join([Prov.STATED, Prov.STATED]) is Prov.STATED
    assert Prov.join([Prov.STATED, Prov.SUPPLIED]) is Prov.SUPPLIED
    assert Prov.join([]) is Prov.STATED


def test_bracket_hydrogens_are_stated():
    """Same molecule, two spellings, different provenance."""
    implicit = translate_smiles("CCO").value.supplied_fraction()
    explicit = translate_smiles("[CH3][CH2][OH]").value.supplied_fraction()
    assert implicit > explicit
    assert explicit < 0.2


def test_fully_stated_record_has_zero_supplied():
    """The control: where nothing is supplied, the fraction is exactly 0."""
    g = translate_smiles("O=C=O").value
    assert g.supplied_fraction() == 0.0


def test_medium_edges_excluded_from_denominator():
    """Medium edges are supplied by construction; counting them would make
    the statistic a property of the representation, not the source."""
    g = translate_smiles("O=C=O").value
    assert any(c.is_medium_edge for c in g.contacts.values())
    assert all(
        c.prov is Prov.SUPPLIED for c in g.contacts.values() if c.is_medium_edge
    )
    assert g.supplied_fraction() == 0.0


# -------------------------------------------------- P3: no empty-as-failure


def test_only_value_bearing_labels_carry_a_value():
    with pytest.raises(ValueError):
        Verdict(Label.UNCLOSED, {}, value="something")


def test_four_failures_are_distinguishable():
    """The conflation control: four different failures, four labels."""
    unclosed = run_honjo("floor 1.0\nN := cut 7\nH := cut 1\nX := close N(H,H)")
    inert = run_honjo("floor 1.0\nNe := cut 10\nH := cut 1\nX := close Ne(H)")
    underprov = run_honjo(
        'floor 1.0\ng := import graph "one.smi" require stated', base_dir=EX
    )
    subfloor = run_honjo("floor 1e-12\nC := cut 6", eps_target=1e-9)

    labels = []
    for out in (unclosed, inert, underprov, subfloor):
        vs = [l for l in out["log"] if "verdict" in l]
        assert vs, out
        v = vs[0]["verdict"]
        labels.append(v if isinstance(v, str) else v["verdict"])

    assert labels == ["unclosed", "inert", "underprovenanced", "subfloor"]
    assert len(set(labels)) == 4


def test_value_or_nothing_wrapper_conflates_them():
    """Negative control: the impoverished interface loses the distinction."""

    def value_or_nothing(out):
        return out["bindings"] or None

    outs = [
        run_honjo("floor 1.0\nN := cut 7\nH := cut 1\nX := close N(H,H)"),
        run_honjo("floor 1.0\nNe := cut 10\nH := cut 1\nX := close Ne(H)"),
    ]
    # both produce bindings for their atoms but no X; the wrapper cannot
    # tell why X is missing
    assert all("X" not in [b["name"] for b in o["bindings"]] for o in outs)


def test_inert_is_a_positive_claim():
    out = run_honjo("floor 1.0\nNe := cut 10\nH := cut 1\nX := close Ne(H)")
    v = [l for l in out["log"] if "verdict" in l][0]
    assert v["verdict"] == "inert"
    assert v["payload"]["certified_vacancy_zero"] is True


# ---------------------------------------------------------- cell counts


def test_cell_counts_distinguish():
    """C=O twice is not the same value as two C-O singles."""
    dbl = run_honjo("floor 1.0\nC := cut 6\nO := cut 8\nX := close C(O:2, O:2)")
    assert dbl["status"] == "ok"
    x = [b for b in dbl["bindings"] if b["name"] == "X"][0]
    assert x["closed"] is True
    assert [l["cells"] for l in x["ligands"]] == [2, 2]
    assert x["geometry"]["shape"] == "linear"


def test_single_cells_do_not_close_carbon_with_two_oxygens():
    out = run_honjo("floor 1.0\nC := cut 6\nO := cut 8\nX := close C(O:1, O:1)")
    v = [l for l in out["log"] if "verdict" in l][0]
    assert v["verdict"] == "unclosed"


def test_deloc_carries_no_per_pair_count():
    src = "floor 1.0\n" + "\n".join(
        f"c{i} := cut 6" for i in range(1, 7)
    ) + "\nring := deloc ring(c1,c2,c3,c4,c5,c6) cells: 9"
    out = run_honjo(src)
    r = [b for b in out["bindings"] if b["name"] == "ring"][0]
    assert r["total_cells"] == 9
    assert r["per_pair_cells"] is None
    assert r["n_centres"] == 6


def test_deloc_needs_three_centres():
    out = run_honjo("floor 1.0\na := cut 6\nb := cut 6\nr := deloc ring(a,b)")
    v = [l for l in out["log"] if "verdict" in l][0]
    assert v["verdict"] == "unclosed"


def test_aromatic_ring_is_delocalised_not_kekule():
    g = translate_smiles("c1ccccc1").value
    assert len(g.delocs) == 1
    sysid = next(iter(g.delocs))
    assert g.delocs[sysid]["per_bond_cells"] is None
    assert g.delocs[sysid]["n_centres" if "n_centres" in g.delocs[sysid] else "members"]


# ---------------------------------------------------------- capability


def test_capability_missing_is_decidable_without_reading():
    assert missing("smiles", {"coords3d"}) == {"coords3d"}
    assert missing("smiles", {"element", "connectivity"}) == set()
    assert "stereo" not in capability("smiles")  # deliberately under-declared


def test_unsupported_request_carries_no_graph():
    v = translate_smiles("CCO", required={"coords3d"})
    assert v.label is Label.UNSUPPORTED
    assert v.value is None
    assert "coords3d" in v.payload["missing_features"]


def test_malformed_is_distinct_from_empty():
    """A record that says nothing is not a record that says something
    ill-formed.  The reader once returned MALFORMED for both, because the
    parser raises on empty input before the (V2) check is reached; that
    conflation is the defect this test now guards against."""
    bad = translate_smiles("C(")
    empty = translate_smiles("   ")
    assert bad.label is Label.MALFORMED
    assert empty.label is Label.EMPTY
    assert bad.label is not empty.label
    assert bad.value is None
    assert empty.value is None


def test_plan_refuses_before_reading_on_capability():
    src = """
    plan p {
      source lib : smiles at "compounds.smi"
      let raw  := read lib
      let mols := translate raw require element, coords3d
      emit mols
    }
    """
    out = run_plan(src, base_dir=EX)
    assert out["status"] == "refused"
    assert out["records_read"] == 0          # nothing was opened
    assert "coords3d" in out["refusal"]["missing_features"]


# ------------------------------------------------------------- the clock


def test_cut_count_is_monotone():
    out = run_honjo("floor 1.0\nA := cut 6\nB := cut 6\nC := cut 6")
    assert out["cut_count"] == 3
    idx = [b["cut_index"] for b in out["bindings"]]
    assert idx == sorted(idx) and len(set(idx)) == 3


def test_reevaluation_is_a_new_cut():
    """Not a cached recomputation: the same expression twice costs twice."""
    one = run_honjo("floor 1.0\nA := cut 6")
    two = run_honjo("floor 1.0\nA := cut 6\nB := cut 6")
    assert two["cut_count"] == one["cut_count"] * 2


# --------------------------------------------------------------- cuts


def test_separation_cost_positive_and_medium_adjacent():
    g = translate_smiles("O=C=O").value
    assert g.validate() == []
    for key in g.atoms:
        sigma, side, _p = g.separation_cost(key)
        assert sigma >= g.floor > 0
        assert key in side


def test_burial_depth_is_at_least_one():
    g = translate_smiles("CCO").value
    for key in g.atoms:
        assert g.burial_depth(key) >= 1


# ---------------------------------------------------------------- plans


def test_plan_runs_and_selects():
    with open(os.path.join(EX, "audit.ms"), encoding="utf-8") as fh:
        out = run_plan(fh.read(), base_dir=EX)
    assert out["status"] == "ok"
    sel = [s for s in out["steps"] if s["step"] == "select"][0]
    assert sel["kept"] >= 1
    assert sel["kept"] + sel["dropped"] == out["records_read"]


def test_plan_expect_threshold_downgrades_verdict():
    src = """
    plan p {
      source lib : smiles at "compounds.smi"
      let raw  := read lib
      let mols := translate raw require element, connectivity
                    expect supplied < 0.01
      emit mols
    }
    """
    out = run_plan(src, base_dir=EX)
    tally = [s for s in out["steps"] if s["step"] == "translate"][0]["tally"]
    assert tally.get("incomplete", 0) > 0


def test_plan_assertion_can_fail():
    src = """
    plan p {
      source lib : smiles at "compounds.smi"
      let raw  := read lib
      let mols := translate raw require element, connectivity
      let core := select mols where supplied == 0.0
      assert core.count > 999  emit "not enough"
      emit core
    }
    """
    out = run_plan(src, base_dir=EX)
    assert out["status"] == "assertion-failed"


# ------------------------------------------------------------ examples


@pytest.mark.parametrize(
    "name,expect",
    [("atoms.hj", "ok"), ("water.hj", "ok"), ("oxides.hj", "ok"),
     ("benzene.hj", "ok"), ("refuse.hj", "refused")],
)
def test_example_scripts(name, expect):
    with open(os.path.join(EX, name), encoding="utf-8") as fh:
        out = run_honjo(fh.read(), base_dir=EX)
    assert out["status"] == expect, out.get("log")


def test_water_geometry():
    with open(os.path.join(EX, "water.hj"), encoding="utf-8") as fh:
        out = run_honjo(fh.read(), base_dir=EX)
    w = [b for b in out["bindings"] if b["name"] == "W"][0]
    assert w["geometry"]["regions"] == 4
    assert w["geometry"]["lone_pairs"] == 2
    assert w["geometry"]["ideal_angle_deg"] == 109.47
    assert w["geometry"]["compressed_by_lone_pairs"] is True
    # the framework gives the maximal-separation configuration, not the
    # quantitative correction
    assert w["geometry"]["quantitative_correction"] is None


# ------------------------------------------------------------ json shape


def test_output_is_json_serialisable():
    import json

    out = run_honjo("floor 1.0\nO := cut 8\nH := cut 1\nW := close O(H,H)")
    json.dumps(out)  # must not raise

    v = translate_smiles("c1ccccc1")
    json.dumps(v.to_dict())
