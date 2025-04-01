# ==================== Trapezoidal Membership Function ====================

def trapezoid_mf(x, params):
    a, b, c, d = params
    if a <= x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1
    elif c < x <= d:
        return (d - x) / (d - c)
    return 0

def TrapezoidMF(x1, label, MFnumber, sigma, centers):
    if 1 <= label <= MFnumber:
        center = centers[int(label) - 1]
        a = center - 2 * sigma
        b = center - sigma
        c = center + sigma
        d = center + 2 * sigma
        return trapezoid_mf(x1, [a, b, c, d])
    return 0