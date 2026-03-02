import numpy as np
from BatchNormalization import BatchNorm1d, BatchNorm2d


def grad_check_1d():
    np.random.seed(0)
    N, D = 8, 5
    x = np.random.randn(N, D)
    bn = BatchNorm1d(D)
    y = bn.forward(x, training=True)
    dout = np.random.randn(*y.shape)
    dx = bn.backward(dout)

    # Numerical gradient for dx
    eps = 1e-5
    dx_num = np.zeros_like(x)
    for i in range(N):
        for j in range(D):
            old = x[i, j]
            x[i, j] = old + eps
            y_pos = bn.forward(x, training=True)
            loss_pos = np.sum(y_pos * dout)
            x[i, j] = old - eps
            y_neg = bn.forward(x, training=True)
            loss_neg = np.sum(y_neg * dout)
            x[i, j] = old
            dx_num[i, j] = (loss_pos - loss_neg) / (2 * eps)
    rel_err = np.max(np.abs(dx - dx_num) / (np.maximum(1e-8, np.abs(dx) + np.abs(dx_num))))
    print(f"BatchNorm1d backward grad check rel error: {rel_err:.2e}")


def smoke_test_2d():
    np.random.seed(1)
    N, C, H, W = 4, 3, 5, 5
    x = np.random.randn(N, C, H, W)
    bn = BatchNorm2d(C)
    # Train mode
    y_train = bn.forward(x, training=True)
    assert y_train.shape == x.shape
    # Inference mode should use running stats
    bn.eval()
    y_eval = bn.forward(x, training=False)
    assert y_eval.shape == x.shape
    print("BatchNorm2d smoke test passed.")


if __name__ == "__main__":
    grad_check_1d()
    smoke_test_2d()

