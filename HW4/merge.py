import sys
import collections

def main(args):
    in_files = args[1:-1]
    out_file = args[-1]
    sum_preds = collections.defaultdict(int)
    header = None
    n = len(in_files)
    for file_name in in_files:
        with open(file_name, "r") as f:
            header = f.readline().strip()
            for line in f:
                res = line.split(',')
                if len(res) >= 2:
                    ind, count = res
                    sum_preds[int(ind)] += int(count)
    with open(out_file, "w") as f:
        print >>f, header
        for i, c in sorted(sum_preds.items()):
            print >>f, ("%d,%d" % (i, ((c + n/2) / n)))

if __name__ == "__main__":
    main(sys.argv)
