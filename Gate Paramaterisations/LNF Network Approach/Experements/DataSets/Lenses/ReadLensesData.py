def read_data():
    instances = []
    targets = []
    
    f = open('lenses.data', 'r+')
    for line in f:
        orig_instance = [int(x) for x in line.strip().split('  ')]

        target = orig_instance[-1]
        instance = [0,0,0,0,0,0]

        instance[orig_instance[0] - 1] = 1
        for i in range(3, 6):
            instance[i] = orig_instance[i - 2] - 1

        instances.append(instance)
        targets.append(target)

    f.close()
    return instances, targets


if __name__ == '__main__':
    i, t = read_data()
    print(i)
    print(t)
        
    

