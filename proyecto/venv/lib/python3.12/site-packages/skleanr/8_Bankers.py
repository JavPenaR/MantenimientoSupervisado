def display(mx, alloc, need, avail):
    print('Available Resources: ', avail)
    print('Allocated Matrix:')
    for row in alloc:
        print(*row)
    print()
    print('Max Matrix:')
    for row in mx:
        print(*row)
    print()
    print('Need Matrix:')
    for row in need:
        print(*row)
    print()


def bankers(n, r, mx, alloc, need, avail):
    print('***** Initially *****')
    display(mx, alloc, need, avail)

    safe_sequence = []

    while True:
        found = False

        for i in range(n):
            if i + 1 in safe_sequence:
                continue

            can_allocate = True
            for j in range(r):
                if need[i][j] > avail[j]:
                    can_allocate = False
                    break

            if can_allocate:
                safe_sequence.append(i + 1)
                for j in range(r):
                    avail[j] += alloc[i][j]
                    alloc[i][j] = 0
                    need[i][j] = '-'  # Mark as done

                print(f'***** After allocating resources to P{i + 1} *****')
                print(f'P{i + 1} can be allocated resources for execution..')
                display(mx, alloc, need, avail)
                print()
                found = True
                break

        if not found:
            print("System is NOT in a safe state !!")
            break

        if len(safe_sequence) == n:
            print("System is in a safe state !!")
            print("Safe Sequence is:", end=" ")
            for i in safe_sequence:
                print(f"P{i}", end=" ")
            print()
            break


n = 5
r = 3
alloc = [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]]
mx = [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]]
need = [[7, 4, 3], [1, 2, 2], [6, 0, 0], [0, 1, 1], [4, 3, 1]]
avail = [3, 3, 2]

bankers(n, r, mx, alloc, need, avail)
