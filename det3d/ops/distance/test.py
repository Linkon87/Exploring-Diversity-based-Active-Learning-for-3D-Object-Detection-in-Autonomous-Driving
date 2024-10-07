import torch
import cdist_ext as _ext
import det3d.torchie as torchie

def torch_distance(src: torch.Tensor, dst: torch.Tensor, type: str="L1") -> torch.Tensor:
    src_expand = src[:, None, :] # [N, 1, C]
    dst_expand = dst[None, :, :] # [1, M, C]
    if type == "L1":
        dist = torch.abs(src_expand - dst_expand) # [N, M, C]
        dist = dist.sum(2) # [N, M]
    elif type == "L2":
        dist = (src_expand - dst_expand)**2 # [N, M, C]
        dist = dist.sum(2) # [M, N]
    else:
        raise NotImplementedError
    return dist

if __name__ == "__main__":
    timer = torchie.Timer()

    a = torch.randn(80, 12).cuda()
    b = torch.randn(90, 12).cuda()

    with torchie.Timer("dist1:"):
        for i in range(1000):
            a = torch.randn(80, 512).cuda()
            b = torch.randn(90, 512).cuda()
            dist1 = torch_distance(a, b)
    with torchie.Timer("dist2:"):
        for i in range(1000):
            a = torch.randn(80, 512).cuda()
            b = torch.randn(90, 512).cuda()
            dist2 = _ext.distance(a, b)
    with torchie.Timer("dist3:"):
        for i in range(1000):
            a = torch.randn(80, 512).cuda()
            b = torch.randn(90, 512).cuda()
            dist3 = _ext.fast_distance(a, b)
            # dist3 = dist3.sum(-1)

    a = torch.randn(100, 512).cuda()
    b = torch.randn(100, 512).cuda()
    dist1 = torch_distance(a, b)
    dist2 = _ext.distance(a, b, _ext.distance_type.L1)
    dist3 = _ext.fast_distance(a, b)
    # dist3 = dist3.sum(-1)
    # print("dist1:\n", dist1)
    # print("dist2:\n", dist2)
    # print("dist3:\n", dist3)
    print("err2: ", torch.abs(dist1 - dist2).max())
    print("err3: ", torch.abs(dist1 - dist3).max())
    s_dist1 = torch_distance(a, b, "L2")
    s_dist2 = _ext.distance(a, b, _ext.distance_type.L2)
    print("s_err2: ", torch.abs(s_dist1 - s_dist2).max())


