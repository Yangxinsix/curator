"""Export only held-out SA-SA DFT molecular dipoles, aligned to the prepared ASE source."""

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io.trajectory import Trajectory


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main(args):
    # Identity of the originally released log table as independently recorded in the dipole audit.
    if sha(args.pkl) != "9cea8d8b77b088547aaf6a7c11330f3a666580400a0d72190ef7da9065b215d1":
        raise ValueError("Source is not the audited original SA-SA DFT log table")
    table = pd.read_pickle(args.pkl)
    split = json.loads(args.split.read_text())
    result = io.StringIO()
    writer = csv.writer(result)
    writer.writerow(("source_index", "split", "source_basename", "DFT_x_D", "DFT_y_D", "DFT_z_D"))
    counts = {}
    with Trajectory(args.trajectory) as trajectory:
        union = set(split["train"]) | set(split["validation"]) | set(split["test"])
        if len(union) != sum(map(len, split.values())) or union != set(range(len(trajectory))):
            raise ValueError("Source indices must partition prepared trajectory")
        for phase in ("validation", "test"):
            count = 0
            for idx in sorted(split[phase]):
                at = trajectory[idx]
                line = table.iloc[int(at.info["source_frame"])]
                orig = line["xyz", "structure"]
                name = at.info["source_basename"]
                if (line["info", "file_basename"] != name or at.info["total_charge"] != 0
                        or not np.array_equal(orig.numbers, at.numbers)):
                    raise ValueError(f"Original geometry identity differs at {idx}")
                np.testing.assert_allclose(orig.positions, at.positions, rtol=0, atol=1e-8)
                vector = np.asarray(line["log", "dipole_moments"], dtype=float)
                if vector.shape != (3,) or not np.isfinite(vector).all():
                    continue
                np.testing.assert_allclose(np.linalg.norm(vector), line["log", "dipole_moment"],
                                           rtol=1e-4, atol=1e-5)
                writer.writerow((idx, phase, name, *vector))
                count += 1
            counts[phase] = {"covered": count, "requested": len(split[phase])}
    args.output.write_bytes(gzip.compress(result.getvalue().encode(), mtime=0))
    print(json.dumps({"splits": counts, "trajectory_sha256": sha(args.trajectory),
                      "split_sha256": sha(args.split), "reference_sha256": sha(args.output)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("pkl", "trajectory", "split", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    main(parser.parse_args())
