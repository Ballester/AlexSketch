with open('full_files.txt') as fid:
    for line in fid:
        name = 'results/' + line.rsplit('\n')[0]
        with open(name) as f:

            preliminar = {}
            for l in f:
                l = l.split(': ')
                if l[0] == 'Top-5':
                    preliminar[l[0]] = int(l[1])
                elif l[0] == 'Top-1':
                    preliminar[l[0]] = int(l[1])
                elif l[0] == 'Trained Top-5':
                    preliminar[l[0]] = int(l[1])
                elif l[0] == 'Trained Top-1':
                    preliminar[l[0]] = int(l[1])
                elif l[0] == 'Not-Trained Top-5':
                    preliminar[l[0]] = int(l[1])
                elif l[0] == 'Not-Trained Top-1':
                    preliminar[l[0]] = int(l[1])
            results[f[0]].append(preliminar)
