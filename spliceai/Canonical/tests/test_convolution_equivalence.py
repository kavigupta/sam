import torch


def test():
    N, Cin, Cout, L = 3, 5, 7, 11
    r = 2
    X = torch.randn(N, Cin, L)
    E = torch.randn(Cout, Cin, r * 2 + 1)
    fast_out = torch.nn.functional.conv1d(X, E, padding=r)

    direct_out = torch.zeros(N, Cout, L)
    for bidx in range(N):
        for i in range(Cout):
            for x in range(L):
                direct_out[bidx, i, x] = sum(
                    E[i, j, d + r] * X[bidx, j, x + d]
                    for j in range(Cin)
                    for d in range(-r, r + 1)
                    if 0 <= x + d < L
                )
    assert ((direct_out - fast_out).abs() < 1e-5).all().item()


test()
