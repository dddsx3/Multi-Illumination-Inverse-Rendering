"""
评估指标模块（Phase 0 新增）

为逆渲染系统提供标准的量化评估指标，作为 Phase 1 合成数据集
（ShapeNet + BlenderProc）的指标契约。所有指标同时支持 torch.Tensor
与 numpy.ndarray 输入，形状约定 [B, C, H, W] 或 [C, H, W]。

指标清单：
- normal_metrics:  法线平均角度误差（°）、中位角度误差、11.25°/22.5°/30° 准确率
                   （光度立体基准的惯用指标）
- depth_metrics:   深度 RMSE、MAE、尺度不变 RMSE（si-RMSE，Eigen et al. 2014）
                   —— 深度存在全局尺度歧义，si-RMSE 对尺度不变
- albedo_metrics:  反照率 MSE、MAE、尺度不变 MAE
                   —— 反照率与光照强度存在乘积歧义，尺度不变版对歧义鲁棒
- recon_metrics:   重建 PSNR、SSIM（内置 11x11 高斯窗 SSIM，无额外依赖）

Mask 约定：所有 GT 指标支持传入 mask [B,1,H,W] 或 [1,H,W]（bool 或 0/1），
只在有效像素上统计；不传 mask 时使用全图。

数据集协议（Phase 1 契约）：每个场景目录应包含
  light_001..K.png  多光照灰度图
  depth.npy         深度图 [1,H,W]（float32，近大远小的相机深度）
  albedo.npy        反照率图 [1,H,W]（float32，[0,1]）
  normal.npy        法线图 [3,H,W]（float32，单位向量，指向相机）
  mask.npy          有效像素掩码 [1,H,W]（uint8，0/1）
  sh_coeffs.npy     每光照 SH 系数 [K,9]（可选，用于光照监督）
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "normal_metrics", "depth_metrics", "albedo_metrics", "recon_metrics",
    "compute_all", "as_tensor",
]


def as_tensor(x) -> torch.Tensor:
    """统一转换为 [B, C, H, W] 的 float32 torch.Tensor"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(x))
    elif not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if x.dtype != torch.float32:
        x = x.float()
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # [H,W] -> [B=1,C=1,H,W]
    elif x.dim() == 3:
        x = x.unsqueeze(0)               # [C,H,W] -> [B=1,C,H,W]
    return x


def _prep_mask(mask, b: int, c: int, h: int, w: int) -> torch.Tensor:
    """将 mask 统一为 [B,1,H,W] float，未提供时全 1"""
    if mask is None:
        return torch.ones(b, 1, h, w, dtype=torch.float32)
    mask = as_tensor(mask)
    if mask.dim() == 4 and mask.shape[1] > 1:
        mask = mask[:, :1]
    if mask.shape[2:] != (h, w):
        mask = F.interpolate(mask, size=(h, w), mode="nearest")
    return (mask > 0).float()


def _eps():
    return 1e-8


def normal_metrics(pred, gt, mask=None, allow_flip: bool = True) -> dict:
    """
    法线评估：平均角度误差（MAE, °）、中位角度误差、阈值准确率

    Args:
        pred: 预测法线 [B,3,H,W] 或 [3,H,W]（单位向量）
        gt: GT 法线，同 pred
        mask: 有效像素掩码
        allow_flip: 是否允许符号翻转（取 min(θ, 180°-θ)）。
                    深度派生的法线面朝相机，通常无需翻转；设 True 更保守。

    Returns:
        {'mae_deg', 'median_deg', 'acc_11_25', 'acc_22_5', 'acc_30'}
    """
    pred, gt = as_tensor(pred), as_tensor(gt)
    b, _, h, w = pred.shape
    m = _prep_mask(mask, b, 1, h, w)

    pred_n = F.normalize(pred, p=2, dim=1)
    gt_n = F.normalize(gt, p=2, dim=1)

    # 角度误差（弧度）
    dot = (pred_n * gt_n).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)  # [B,1,H,W]
    if allow_flip:
        theta = torch.minimum(theta, torch.pi - theta)

    valid = (m > 0) & ~torch.isnan(pred_n).any(dim=1, keepdim=True)
    theta = theta[valid]

    if theta.numel() == 0:
        return {'mae_deg': float('nan'), 'median_deg': float('nan'),
                'acc_11_25': float('nan'), 'acc_22_5': float('nan'),
                'acc_30': float('nan')}

    theta_deg = theta * 180.0 / torch.pi
    return {
        'mae_deg': float(theta_deg.mean().item()),
        'median_deg': float(theta_deg.median().item()),
        'acc_11_25': float((theta_deg <= 11.25).float().mean().item()),
        'acc_22_5': float((theta_deg <= 22.5).float().mean().item()),
        'acc_30': float((theta_deg <= 30.0).float().mean().item()),
    }


def depth_metrics(pred, gt, mask=None) -> dict:
    """
    深度评估：RMSE、MAE、尺度不变 RMSE（si-RMSE, Eigen et al. CVPR 2014）

    深度存在全局尺度歧义（近大远小的绝对值不可观），si-RMSE 在
    log 空间去除全局偏移后计算，对乘性尺度不变（scale-invariant）；
    线性亮度偏移不保证不变。
    """
    pred, gt = as_tensor(pred), as_tensor(gt)
    b, _, h, w = pred.shape
    m = _prep_mask(mask, b, 1, h, w)

    # 防止 log 域 NaN：将无效像素与极值像素遮掉
    valid_gt = (gt > _eps()) & (m > 0)
    if not valid_gt.any():
        return {'rmse': float('nan'), 'mae': float('nan'), 'si_rmse': float('nan')}

    err = (pred - gt) * m
    rmse = torch.sqrt((err ** 2).sum() / m.sum())
    mae = err.abs().sum() / m.sum()

    # 尺度不变 RMSE（log 空间去均值）
    log_pred = torch.log(pred.clamp_min(_eps()))
    log_gt = torch.log(gt.clamp_min(_eps()))
    d = (log_pred - log_gt) * m
    d_centered = d - (d.sum() / m.sum())
    si_rmse = torch.sqrt((d_centered ** 2).sum() / m.sum())

    return {'rmse': float(rmse.item()), 'mae': float(mae.item()),
            'si_rmse': float(si_rmse.item())}


def albedo_metrics(pred, gt, mask=None, scale_invariant: bool = True) -> dict:
    """
    反照率评估：MSE、MAE，以及尺度不变 MAE

    反照率与光照强度存在乘积歧义（ρ 与光照能量互换不改变渲染结果），
    尺度不变版用最小二乘估计全局尺度 s = argmin ||s·pred - gt||² 后计算 MAE。
    """
    pred, gt = as_tensor(pred), as_tensor(gt)
    b, _, h, w = pred.shape
    m = _prep_mask(mask, b, 1, h, w)

    err = (pred - gt) * m
    mse = (err ** 2).sum() / m.sum()
    mae = err.abs().sum() / m.sum()

    out = {'mse': float(mse.item()), 'mae': float(mae.item())}

    if scale_invariant:
        p, g = pred * m, gt * m
        denom = (p ** 2).sum()
        if denom > _eps():
            s = (p * g).sum() / denom
            si_mae = (s * pred - gt).abs().mul(m).sum() / m.sum()
            out['si_mae'] = float(si_mae.item())
        else:
            out['si_mae'] = float('nan')

    return out


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """1D 高斯核 -> 2D 归一化窗口 [1,1,size,size]"""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g1 = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    win = torch.outer(g1, g1)
    return (win / win.sum()).view(1, 1, size, size)


def _ssim_torch(x: torch.Tensor, y: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """通道级 SSIM，返回 [B,C,H,W]（边界为有效反射 padding）"""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    pad = win.shape[-1] // 2
    x = F.pad(x, [pad] * 4, mode="reflect")
    y = F.pad(y, [pad] * 4, mode="reflect")

    ux = F.conv2d(x, win)
    uy = F.conv2d(y, win)
    uxx = F.conv2d(x * x, win)
    uyy = F.conv2d(y * y, win)
    uxy = F.conv2d(x * y, win)

    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy

    ssim = ((2 * ux * uy + c1) * (2 * vxy + c2)) / \
           ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    return ssim


def recon_metrics(pred, target) -> dict:
    """
    重建图像评估：PSNR、SSIM

    输入 [B,K,H,W] 的多光照图像时逐光照计算后取平均。
    """
    pred, target = as_tensor(pred), as_tensor(target)

    if pred.dim() == 4 and pred.shape[1] != 1:
        # 多光照 [B,K,H,W]：逐 K 计算再平均
        psnr_list, ssim_list = [], []
        for k in range(pred.shape[1]):
            m = recon_metrics(pred[:, k:k+1], target[:, k:k+1])
            psnr_list.append(m['psnr'])
            ssim_list.append(m['ssim'])
        return {'psnr': float(np.mean(psnr_list)), 'ssim': float(np.mean(ssim_list))}

    # [B,1,H,W] 或 [B,3,H,W]：逐通道/逐样本平均
    mse = ((pred - target) ** 2).mean(dim=(2, 3))
    psnr = 10.0 * torch.log10(1.0 / (mse + _eps()))

    win = _gaussian_window().to(device=pred.device, dtype=pred.dtype)
    ssim = _ssim_torch(pred, target, win).mean(dim=(2, 3))

    return {
        'psnr': float(psnr.mean().item()),
        'ssim': float(ssim.mean().item()),
    }


def compute_all(pred: dict, gt: dict, mask=None) -> dict:
    """
    汇总评估：对同时具备的键计算指标

    Args:
        pred: {'depth'|'albedo'|'normal'|'image': tensor}
        gt:   同结构
        mask: 有效像素掩码（可选）

    Returns:
        合并后的指标字典（键带前缀，如 'depth_rmse'）
    """
    results = {}
    if 'normal' in pred and 'normal' in gt:
        for k, v in normal_metrics(pred['normal'], gt['normal'], mask).items():
            results[f'normal_{k}'] = v
    if 'depth' in pred and 'depth' in gt:
        for k, v in depth_metrics(pred['depth'], gt['depth'], mask).items():
            results[f'depth_{k}'] = v
    if 'albedo' in pred and 'albedo' in gt:
        for k, v in albedo_metrics(pred['albedo'], gt['albedo'], mask).items():
            results[f'albedo_{k}'] = v
    if 'image' in pred and 'image' in gt:
        for k, v in recon_metrics(pred['image'], gt['image']).items():
            results[f'image_{k}'] = v
    return results


def _main():
    parser = argparse.ArgumentParser(description='指标计算 CLI（快速检查用）')
    parser.add_argument('--pred', required=True, help='预测文件 .npy')
    parser.add_argument('--gt', required=True, help='GT 文件 .npy')
    parser.add_argument('--kind', required=True,
                        choices=['normal', 'depth', 'albedo', 'image'],
                        help='指标类型')
    parser.add_argument('--mask', default=None, help='掩码文件 .npy（可选）')
    args = parser.parse_args()

    pred = np.load(args.pred)
    gt = np.load(args.gt)
    mask = np.load(args.mask) if args.mask else None

    if args.kind == 'normal':
        out = normal_metrics(pred, gt, mask)
    elif args.kind == 'depth':
        out = depth_metrics(pred, gt, mask)
    elif args.kind == 'albedo':
        out = albedo_metrics(pred, gt, mask)
    else:
        out = recon_metrics(pred, gt)

    for k, v in out.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    _main()
