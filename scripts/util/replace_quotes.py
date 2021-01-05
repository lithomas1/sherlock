import sys

file = sys.argv[1]
replaced = []
replace_targets = [(r"\u2018", "'"), (r"\u2019", "'")]

with open(file, "r") as fin:
    for line in fin:
        for targets in replace_targets:
            line = line.replace(*targets)
        replaced.append(line)

with open(file, "w+") as fout:
    print(*replaced, sep="", end="", file=fout)
