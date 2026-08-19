-- audit.ms : how much of this corpus is record, and how much convention?
plan provenance_audit {
  source lib : smiles at "compounds.smi"
  budget 100 records

  let raw  := read lib
  let mols := translate raw
                require element, connectivity, cellcount
                expect supplied < 0.25
                else report

  let core := select mols where supplied == 0.0

  assert core.count > 1  emit "too few fully-stated structures"
  emit core with provenance
}
