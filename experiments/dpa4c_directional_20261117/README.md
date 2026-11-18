# DPA4C directional ACVR on Cumulene

This directory preserves the training code deployed to HHU on 17 November 2026. The directional model is in `src/experiments/dpa4c_directional`. The q-EFA adapter, proposal and DPA4C representation are in `src/experiments/periodic_qefa` and `unified/curator`. Four Curator modules in the repository root use the versions from this run. The main framework branch is `acvr-unified-framework`.

Run the CPU directional transport check from this directory with `python src/experiments/dpa4c_directional/check_transport.py`. It compares the factored computation with a dense reference and checks mixed gradients and orthogonal transformations. For the original HHU training layout, supply the parent checkpoint, data and split under `inputs/` and use the container launcher referenced in `job.sh`. The deployment manifest records the input hashes. Training settings and arm definitions are in `src/experiments/dpa4c_directional/PROTOCOL.md`.
