for _ in range(int(input())):
    x, y, a = map(int, input().split())
    print(["YES", "NO"][a % (x + y) < x])
