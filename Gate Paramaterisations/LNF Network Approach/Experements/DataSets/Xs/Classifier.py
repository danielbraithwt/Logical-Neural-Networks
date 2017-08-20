def load_data():
    f = open('image.txt', 'r+')
    examples = []
    targets = []

    lines = []
    for line in f:
        lines.append(line)

    print(lines)

    for i in range(0, len(lines), 4):
        t = lines[i+1]
        img = lines[i+2]

        print(img.split('\n'))

        #target = (t == "#Yes")
        #example = [int(x) for x in img]

        #examples.append(example)
        #targets.append(1 if target else 0)

print(load_data())
        
