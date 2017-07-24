import numpy as np

def entropy(S, nc):
    """Assuming format of each problem instance is
        [..., class]
    """

    if len(S) == 0:
        return 0

    count = np.zeros(nc).tolist()

    for instance in S:
        count[int(instance[-1] - 1)] += 1

    probs = map(lambda x: float(x)/float(len(S)), count)

    return -sum(map(lambda x: 0 if x == 0 else x * np.log2(x), probs))

def induced_partition_entropy(S, left, right, nc):
    return (float(len(left))/float(len(S))) * entropy(left, nc) + (float(len(right))/float(len(S))) * entropy(right, nc)


def compute_partition(S, A, T):
    left = list(filter(lambda x: x[A] < T, S))
    right = list(filter(lambda x: x[A] >= T, S))

    return left, right

def classes_represented(S):
    dc = []
    
    for ins in S:
        cls = ins[-1]
        if cls not in dc:
            dc.append(cls)

    return len(dc)

def partition(S, A, nc):
    T_best = 0
    partition_best = None
    induced_ent_best = np.inf

    # Find best way to partition
    for ins in S:
        T = ins[A]

        left, right = compute_partition(S, A, T)
        ent = induced_partition_entropy(S, left, right, nc)
        #print(ent)
        if ent < induced_ent_best:
            induced_ent_best = ent
            T_best = T
            partition_best = (left, right)


    # Compute gain
    gain = entropy(S, nc) - induced_ent_best

    # Determin if partition should be made
    S_cr = classes_represented(S)
    l_cr = classes_represented(partition_best[0])
    r_cr = classes_represented(partition_best[1])

    S_ent = entropy(S, nc)
    l_ent = entropy(partition_best[0], nc)
    r_ent = entropy(partition_best[1], nc)
    
    delta = np.log2(3**S_cr - 2) - (S_cr * S_ent - l_cr * l_ent - r_cr * r_ent)
    apply = (gain >= (np.log2(len(S) - 1)/float(len(S))) + (delta/float(len(S))))

    if apply:
        lp = partition(partition_best[0], A, nc)
        rp = partition(partition_best[1], A, nc)

        return lp + [T_best] + rp
    else:
        return []
    

