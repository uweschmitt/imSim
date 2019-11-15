import coord
import numpy as np
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("infile")
parser.add_argument("outfile")
parser.add_argument("--n", default=1000, type=int)
parser.add_argument("--mag_min", default=16.0, type=float)
parser.add_argument("--mag_max", default=25.0, type=float)
args = parser.parse_args()

with open(args.outfile, 'w') as out_fd:
    with open(args.infile, 'r') as in_fd:
        for line in in_fd:
            if line.startswith("object"):
                break
            if line.startswith("rightascension"):
                ra = float(line.split()[1])
            if line.startswith("declination"):
                dec = float(line.split()[1])
            out_fd.write(line)
    ddec = 2.05
    dra = 2.05/np.cos(np.deg2rad(dec))
    for i in range(args.n):
        ra1 = np.random.uniform(-dra, dra)+ra
        dec1 = np.random.uniform(-ddec, ddec)+dec
        mag = np.random.uniform(args.mag_min, args.mag_max)
        out = f"object {i+10000000:8d} {ra1:20.14f} {dec1:20.14f} {mag:20.14f} starSED/phoSimMLT/lte035-4.5-1.0a+0.4.BT-Settl.spec.gz 0 0 0 0 0 0 point none CCM 0.01014165 3.1\n"
        out_fd.write(out)
