import numpy as np
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("infile")
parser.add_argument("outfile")
parser.add_argument("--min_mag", default=13.0, type=float)
parser.add_argument("--max_mag", default=30.0, type=float)
args = parser.parse_args()

with open(args.infile, 'r') as in_fd:
    with open(args.outfile, 'w') as out_fd:
        for line in in_fd:
            if not line.startswith("object"):
                out_fd.write(line)
            else:
                tokens = line.split(' ')
                magnitude = float(tokens[4])
                magnitude = max(magnitude, args.min_mag)
                magnitude = min(magnitude, args.max_mag)
                tokens[4] = str(magnitude)
                out_fd.write(' '.join(tokens))
