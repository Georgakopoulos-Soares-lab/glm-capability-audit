#!/bin/bash
# A. Base nucleotide search per benchmark at the most permissive identity (0.30).
# --max-seqs 20 matches the frozen manuscript pipeline (nn_lookup_probe.search_hits).
# The default of 300 finds marginal low-coverage hits the frozen run did not, which
# is the entire source of an earlier 11,992-vs-11,994 discrepancy on HT hard_70.
# capturing pident so higher thresholds can be derived. A true re-run at two
# thresholds validates that derivation; see validate_threshold_derivation.py.
set -u
MM=/home/nvidia/tools/mmseqs/bin/mmseqs
O=/home/nvidia/glm-capability-audit/reports/threshold_sensitivity/hits
mkdir -p $O /data/nvidia/tmp_mmseqs/thr
D=/data/nvidia
run () { # name  train.csv  test.csv  seqcol
  local n=$1 tr=$2 te=$3
  [ -s "$O/$n.tsv" ] && { echo "  $n cached"; return; }
  /home/nvidia/miniconda3/envs/biojepa/bin/python - "$tr" "$te" "$O/$n" <<'PY'
import sys, pandas as pd
tr,te,out=sys.argv[1],sys.argv[2],sys.argv[3]
for src,tag in ((tr,"train"),(te,"test")):
    d=pd.read_csv(src); c=[x for x in d.columns if x in ("sequence","text")][0]
    with open(f"{out}.{tag}.fa","w") as f:
        for i,s in enumerate(d[c].astype(str)): f.write(f">{i}\n{s}\n")
PY
  $MM easy-search "$O/$n.test.fa" "$O/$n.train.fa" "$O/$n.tsv" /data/nvidia/tmp_mmseqs/thr/$n \
      --search-type 3 --min-seq-id 0.30 -c 0.10 --cov-mode 2 -e 1e-3 --max-seqs 20 \
      --threads 48 -v 1 --format-output "query,target,pident,qcov,alnlen,evalue,bits" \
    && echo "  $n hits=$(wc -l < $O/$n.tsv)"
  rm -f "$O/$n.test.fa" "$O/$n.train.fa"; rm -rf /data/nvidia/tmp_mmseqs/thr/$n
}
H=$D/hvue_v2
run HVUE_HT_hard_70   $H/Host_Tropism/hard_70_1000bp/train.csv          $H/Host_Tropism/hard_70_1000bp/test.csv
run HVUE_HT_std_95    $H/Host_Tropism/standard_95_1000bp/train.csv      $H/Host_Tropism/standard_95_1000bp/test.csv
run HVUE_Patho_hard   $H/Pathogenicity/hard_capped_1000bp/train.csv     $H/Pathogenicity/hard_capped_1000bp/test.csv
run HVUE_Patho_std    $H/Pathogenicity/standard_capped_1000bp/train.csv $H/Pathogenicity/standard_capped_1000bp/test.csv
run HVUE_Transm_hard  $H/Transmissibility/hard_capped_1000bp/train.csv  $H/Transmissibility/hard_capped_1000bp/test.csv
run HVUE_Transm_std   $H/Transmissibility/standard_capped_1000bp/train.csv $H/Transmissibility/standard_capped_1000bp/test.csv
run VHDB_v2           $D/vhdb_tropism/within_family_1000bp/train.csv    $D/vhdb_tropism/within_family_1000bp/test.csv
run GUE_phage_v2      $D/gue_devcarve/leannmlindsey_GUE_phage_fragments/train.csv $D/gue_devcarve/leannmlindsey_GUE_phage_fragments/test.csv
run GUE_covid         $D/gue_devcarve/leannmlindsey_GUE_virus_covid/train.csv     $D/gue_devcarve/leannmlindsey_GUE_virus_covid/test.csv
echo "BASE SEARCHES DONE"
