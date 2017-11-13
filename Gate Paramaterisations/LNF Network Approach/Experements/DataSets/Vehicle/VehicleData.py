import numpy as np
files = ['xaa', 'xab', 'xac', 'xad', 'xae', 'xaf', 'xag', 'xah', 'xai']

def read_single_file(name):
    lines = []
    f = open('./' + name, 'r+')
    for line in f:
        lines.append(line)
    f.close()

    examples = []
    targets = []

    for line in lines:
        splt = line.strip().split(' ')
        target = splt[-1]
        example = [int(splt[x]) for x in range(0, len(splt)-1)]

        target_actual = [0, 0, 0, 0]
        if target == 'opel':
            target_actual[0] = 1
        elif target == 'saab':
            target_actual[1] = 1
        elif target == 'bus':
            target_actual[2] = 1
        elif target== 'van':
            target_actual[3] = 1

        targets.append(target_actual)
        examples.append(example)
    return [np.array(examples), np.array(targets)]

def read_data():
    all_files = [read_single_file(name + '.dat') for name in files]
    examples = np.concatenate([x[0] for x in all_files])
    targets = np.concatenate([x[1] for x in all_files])

    ex_max = np.amax(examples, 0)
    ex_min = np.amin(examples, 0)

    examples = (examples - ex_min)/(ex_max - ex_min)

    return examples, targets

read_data()
