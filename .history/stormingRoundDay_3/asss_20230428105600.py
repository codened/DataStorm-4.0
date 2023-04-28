L1 = {'a', 'b', 'cc'}
L3 = set()

for w1 in L1:
    for w2 in L1:
        for w3 in L1:
            L3.add(w1 + w2 + w3)

print("L13:", L3)
