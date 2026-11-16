import torch
import torchvision
import cv2
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def chain_rule():
    """
    Compute df/dz, df/dq, df/dx, and df/dy for f(x,y,z)=xy+z,
    where q=xy, at x=-2, y=5, z=-4.
    Return them in this order: df/dz, df/dq, df/dx, df/dy.
    """
    # inputs
    x, y, z = -2.0, 5.0, -4.0

    # forward pass
    q = x * y              # q = -10
    f = q + z              # f = -14

    # backward pass (chain rule), starting from df/df = 1
    df_df = 1.0
    df_dq = 1.0 * df_df    # f = q + z  -> add passes the gradient through: 1
    df_dz = 1.0 * df_df    #                                              : 1
    df_dx = y * df_dq      # q = x * y  -> multiply "switches" the inputs: y = 5
    df_dy = x * df_dq      #                                             : x = -2

    return float(df_dz), float(df_dq), float(df_dx), float(df_dy)

def ReLU():
    """
    Compute dx and dw, and return them in order.
    Forward:
        y = ReLU(w0 * x0 + w1 * x1 + w2)

    Returns:
        dx -- gradient with respect to input x, as a vector [dx0, dx1]
        dw -- gradient with respect to weights (including the third term w2),
              as a vector [dw0, dw1, dw2]
    """
    # inputs and weights
    x0, x1 = -1.0, -2.0
    w0, w1, w2 = 2.0, -3.0, -3.0

    # forward pass
    s = w0 * x0 + w1 * x1 + w2       # = -2 + 6 - 3 = 1
    y = max(0.0, s)                  # ReLU: 1 (positive, so it stays)

    # backward pass, starting from dy/dy = 1
    dy_dy = 1.0
    dy_ds = dy_dy * (1.0 if s > 0 else 0.0)   # ReLU gate: open (s > 0), so 1

    # the "+" copies the gradient to every term; each multiply "switches" the inputs
    dx = [w0 * dy_ds, w1 * dy_ds]              # [2, -3]
    dw = [x0 * dy_ds, x1 * dy_ds, 1.0 * dy_ds] # [-1, -2, 1]

    return dx, dw

def _sigmoid_graph():
    """Forward pass of f(w,x) = 1/(1+exp(-(w0x0+w1x1+w2))) from the lecture notes."""
    w0 = torch.tensor(2.0, requires_grad=True)
    x0 = torch.tensor(-1.0, requires_grad=True)
    w1 = torch.tensor(-3.0, requires_grad=True)
    x1 = torch.tensor(-2.0, requires_grad=True)
    w2 = torch.tensor(-3.0, requires_grad=True)

    s = w0 * x0 + w1 * x1 + w2     # 1
    a = torch.exp(-s)              # 0.37
    b = 1 + a                      # 1.37
    c = 1 / b                      # 0.73
    return (w0, x0, w1, x1, w2), (a, b, c)

def chain_rule_a():
    """
    In the lecture notes, the last three forward pass values are
    a=0.37, b=1.37, and c=0.73.
    Calculate these numbers to 4 decimal digits and return in order of a, b, c
    """
    _, (a, b, c) = _sigmoid_graph()
    a = round(a.item(), 4)
    b = round(b.item(), 4)
    c = round(c.item(), 4)
    return a, b, c

def chain_rule_b():
    """
    In the lecture notes, the backward pass values are
    ±0.20, ±0.39, -0.59, and -0.53.
    Calculate these numbers to 4 decimal digits
    and return in order of gradients for w0, x0, w1, x1, w2.
    """
    (w0, x0, w1, x1, w2), (a, b, c) = _sigmoid_graph()
    c.backward()                   # autograd fills .grad for every input

    gw0 = round(w0.grad.item(), 4)
    gx0 = round(x0.grad.item(), 4)
    gw1 = round(w1.grad.item(), 4)
    gx1 = round(x1.grad.item(), 4)
    gw2 = round(w2.grad.item(), 4)
    return gw0, gx0, gw1, gx1, gw2

def _tanh_net():
    """f(w,x) = tanh(w0x0 + w1x1 + w2) with w=[5,2], x=[-1,4], w2=-2."""
    w = torch.tensor([5.0, 2.0, -2.0], requires_grad=True)   # w0, w1, w2 (bias)
    x = torch.tensor([-1.0, 4.0])
    y_hat = torch.tanh(w[0] * x[0] + w[1] * x[1] + w[2])
    return w, y_hat

def backprop_a():
    """
    Let f(w,x) = torch.tanh(w0x0+w1x1+w2).
    Assume the weight vector is w = [w0=5, w1=2],
    the input vector is  x = [x0=-1,x1= 4],, and the bias is  w2  =-2.
    Use PyTorch to calculate the forward pass of the network, return y_hat = f(w,x).
    """
    _, y_hat = _tanh_net()
    y_hat = y_hat.item()           # tanh(1) = 0.7616
    return y_hat

def backprop_b():
    """
    Use PyTorch Autograd to calculate the gradients
    for each of the weights, and return the gradient of them
    in order of w0, w1, and w2.
    """
    w, y_hat = _tanh_net()
    loss = (y_hat - 1.0) ** 2      # MSE loss with ground truth 1
    loss.backward()

    gw0, gw1, gw2 = [g.item() for g in w.grad]
    return gw0, gw1, gw2

def backprop_c():
    """
    Assuming a learning rate of 0.1,
    update each of the weights accordingly.
    For simplicity, just do one iteration.
    And return the updated weights in the order of w0, w1, and w2
    """
    lr = 0.1
    w, y_hat = _tanh_net()
    loss = (y_hat - 1.0) ** 2
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad           # one gradient-descent step: w = w - lr * dL/dw

    w0, w1, w2 = [v.item() for v in w]
    return  w0, w1, w2


def constructParaboloid(w=256, h=256):
    img = np.zeros((w, h), np.float32)
    for x in range(w):
        for y in range(h):
            # let's center the paraboloid in the img
            img[y, x] = (x - w / 2) ** 2 + (y - h / 2) ** 2
    return img


def _derivative_images(paraboloid):
    """1st and 2nd derivatives of the image with small convolution kernels (central differences)."""
    def conv(kernel):
        k = torch.tensor(kernel, dtype=torch.float32)
        k = k.unsqueeze(0).unsqueeze(0)                        # (1,1,kh,kw)
        pad = (k.shape[2] // 2, k.shape[3] // 2)
        return F.conv2d(paraboloid, k, padding=pad)[0, 0]       # (H, W)

    fx = conv([[-0.5, 0.0, 0.5]])                  # df/dx
    fy = conv([[-0.5], [0.0], [0.5]])              # df/dy
    fxx = conv([[1.0, -2.0, 1.0]])                 # d2f/dx2
    fyy = conv([[1.0], [-2.0], [1.0]])             # d2f/dy2
    fxy = conv([[0.25, 0.0, -0.25],
                [0.0, 0.0, 0.0],
                [-0.25, 0.0, 0.25]])               # d2f/dxdy
    return fx, fy, fxx, fyy, fxy


def _sample(img, x, y):
    """Value of a 2-D tensor at a non-integer position (bilinear interpolation)."""
    H, W = img.shape
    x = min(max(x, 1.0), W - 2.0)        # stay inside, away from the padded border
    y = min(max(y, 1.0), H - 2.0)
    c0, r0 = int(np.floor(x)), int(np.floor(y))
    c1, r1 = min(c0 + 1, W - 1), min(r0 + 1, H - 1)
    dx, dy = x - c0, y - r0
    top = (1 - dx) * img[r0, c0].item() + dx * img[r0, c1].item()
    bottom = (1 - dx) * img[r1, c0].item() + dx * img[r1, c1].item()
    return (1 - dy) * top + dy * bottom


def newtonMethod(x0, y0):
    paraboloid = torch.tensor([constructParaboloid()]).squeeze()
    paraboloid = torch.unsqueeze(paraboloid, 0)
    paraboloid = torch.unsqueeze(paraboloid, 0)    # -> (1,1,H,W) for conv2d

    """
    Insert your code here
    """
    fx, fy, fxx, fyy, fxy = _derivative_images(paraboloid)
    H, W = fx.shape

    x, y = float(x0), float(y0)
    for _ in range(100):
        # read derivatives at the current position
        g = torch.tensor([_sample(fx, x, y), _sample(fy, x, y)])   # gradient
        hxy = _sample(fxy, x, y)
        Hm = torch.tensor([[_sample(fxx, x, y), hxy],
                           [hxy, _sample(fyy, x, y)]])               # Hessian

        step = torch.linalg.solve(Hm, g)                            # H^-1 * g
        x, y = x - step[0].item(), y - step[1].item()               # Newton update
        x = min(max(x, 0), W - 1)
        y = min(max(y, 0), H - 1)

        if abs(step[0].item()) < 1e-6 and abs(step[1].item()) < 1e-6:
            break

    final_x, final_y = int(round(x)), int(round(y))
    return final_x, final_y


def sgd(x0, y0, lr=0.001):
    paraboloid = torch.tensor([constructParaboloid()]).squeeze()
    paraboloid = torch.unsqueeze(paraboloid, 0)
    paraboloid = torch.unsqueeze(paraboloid, 0)

    """
    Insert your code here
    """
    fx, fy, _, _, _ = _derivative_images(paraboloid)
    H, W = fx.shape

    x, y = float(x0), float(y0)
    for _ in range(100000):
        gx, gy = _sample(fx, x, y), _sample(fy, x, y)   # gradient at current position

        new_x = min(max(x - lr * gx, 0), W - 1)       # step against the gradient,
        new_y = min(max(y - lr * gy, 0), H - 1)       # kept inside the image

        moved = abs(new_x - x) + abs(new_y - y)
        x, y = new_x, new_y
        if moved < 1e-4:                              # converged: barely moving any more
            break

    final_x, final_y = int(round(x)), int(round(y))
    return final_x, final_y


if __name__ == "__main__":
    pass
    # print("Q1  chain_rule:", chain_rule())
    # print("Q2  ReLU:", ReLU())
    # print("Q3a chain_rule_a:", chain_rule_a())
    # print("Q3b chain_rule_b:", chain_rule_b())
    # print("Q4a backprop_a:", backprop_a())
    # print("Q4b backprop_b:", backprop_b())
    # print("Q4c backprop_c:", backprop_c())
    # print("Q5a newtonMethod:", newtonMethod(30, 200))
    # print("Q5b sgd:", sgd(30, 200, lr=0.001))
