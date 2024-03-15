import sys

def WriteSEP(cnt):
    if cnt < 0:
        return
    # only allow one empty line
    cnt = min(cnt, 1)
    for i in range(cnt):
        sys.stdout.write('\n')

    sys.stdout.write('\n')

cont_nl_cnt = -1

for line in sys.stdin:
    line = line.strip()
    if line:
        WriteSEP(cont_nl_cnt)
        sys.stdout.write(line)
        cont_nl_cnt = 0
    else:
        cont_nl_cnt += 1

WriteSEP(cont_nl_cnt)