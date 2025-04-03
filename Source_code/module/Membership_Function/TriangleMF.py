# ==================== Triangular Membership Function ====================

def triangle_mf(x, params):
    a, b, c = params
    if a <= x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return (c - x) / (c - b)
    return 0

def TriangleMF(x1, label, MFnumber, sigma, centers):
    if 1 <= label <= MFnumber:
        center = centers[int(label) - 1]
        a = center - sigma
        b = center
        c = center + sigma
        return triangle_mf(x1, [a, b, c])
    return 0