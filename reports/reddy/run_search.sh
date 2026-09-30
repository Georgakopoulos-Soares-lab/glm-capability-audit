#!/bin/bash
# Frozen L1/L2 protocol, applied with HVUE train as QUERY so that --cov-mode 2
# (coverage of the query) still refers to a 1 kb sequence, as it does in the
# manuscript. Reddy genomes are whole genomes and cannot be the coverage
# reference without changing what the threshold means.
set -u
MM=/home/nvidia/tools/mmseqs/bin/mmseqs
cd /home/nvidia/glm-capability-audit/reports/reddy
for tag in hard70 std95; do
  for mode in "3:nt" "2:tr"; do
    st=${mode%%:*}; nm=${mode##*:}
    extra=""; [ "$st" = "2" ] && extra="-s 7.5"
    echo "== $tag $nm (search-type $st)"
    $MM easy-search hvue_${tag}_train.fasta test_genomes.fasta \
        hits_${tag}_${nm}.tsv tmp/${tag}_${nm} \
        --search-type $st $extra --min-seq-id 0.30 -c 0.10 --cov-mode 2 -e 1e-3 \
        --threads 48 -v 1 --format-output "query,target,pident,qcov,alnlen,evalue,bits" \
      && echo "   hits: $(wc -l < hits_${tag}_${nm}.tsv)"
    rm -rf tmp/${tag}_${nm}
  done
done
echo "ALL DONE"
