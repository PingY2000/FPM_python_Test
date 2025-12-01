# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 15:33:05 2025

@author: 22568
"""

# fpm_reconstruct.py
# 说明：
#  - 读取目录中匹配 snapshot_*.tif 的图片（文件名包含 LED 状态字符串，如 ..._000000001000... .tif）
#  - 从文件名解析 LED 状态字符串（0/1 序列），并把“1”对应的 LED 视为该张图的照明源
#  - 支持用户提供 LED 位置文件（csv） 或 按圆环自动生成 LED 坐标（默认）
#  - 执行简单的 FPM 迭代重建（傅里叶域拼接 + 强度投影；ePIE-style）
#  - 输出重建的幅值和相位图像（tif/png）
#
# 依赖：
#   pip install numpy scipy tifffile matplotlib
#
# 使用：
#   python fpm_reconstruct.py --data_dir ./snapshots --out_prefix result --iters 50
#
# 注意事项：
#   请务必在脚本开头配置你的光学参数（波长、像素尺寸、放大、NA、LED高度/位置等）
#   这不是最复杂的 FPM 实现（没有加入像差估计、相位 unwrap、位移补偿或深度反演），但能作为可运行的起点。

import os
import re
import glob
import argparse
import numpy as np
from tifffile import imread, imsave
from scipy.fft import fftshift, ifftshift, fft2, ifft2
import matplotlib.pyplot as plt

# ---------------- USER SET: 需要你根据系统填写这些参数 ----------------
WAVELENGTH = 0.530e-6           # 波长，单位 m（示例 530 nm）
CAM_PIXEL_SIZE = 3.45e-6        # 相机像素尺寸，单位 m（相机像素pitch）
MAGNIFICATION = 10.0            # 物镜放大倍数（例如10x）
OBJ_PIXEL_SIZE = CAM_PIXEL_SIZE / MAGNIFICATION  # 物平面采样间距 m/pixel
OBJ_NX = None                   # 如果已知目标输出尺寸想覆盖更大频域可设置，否则自动取图片大小
OBJ_NY = None
OBJ_UPSAMPLE = 2                # 频域扩展因子（>1 表示在频域使用更大的画布）
OBJ_NA = 0.1                    # 物镜数值孔径（示例）
LED_HEIGHT = 0.06               # LED 阵列到样品的距离，单位 m（例如 60 mm）
LED_LAYOUT = 'ring'             # 'ring' 或 'grid' 或 'from_csv'
LED_COUNT = 37                  # 如果自动生成圆环，则 LED 数量
LED_RADIUS = 0.02               # 环半径 m（用于自动圆环布局）
LED_POS_CSV = None              # 若有 LED 坐标文件 (index,x(m),y(m),z(m)) 可指定路径
# ---------------------------------------------------------------------

# algorithm parameters
ALPHA = 0.8          # 更新步长（0~1）
NUM_ITERS = 60       # 迭代次数
PUPIL_PADDING = OBJ_UPSAMPLE  # 频域画布放大倍数

# helper: parse filename to get LED bitstring
def extract_led_string(fname):
    # 寻找最后一个连续的 0/1 长串
    m = re.search(r'([01]{10,})', os.path.basename(fname))
    if m:
        return m.group(1)
    # 备用方式：寻找形如 _LEDxxxxx_
    m2 = re.search(r'_LED([01]+)', os.path.basename(fname), re.IGNORECASE)
    if m2:
        return m2.group(1)
    return None

# helper: load all matching tif files and parse
def load_images(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, 'snapshot_*.tif')))
    imgs = []
    led_strings = []
    filenames = []
    for f in files:
        s = extract_led_string(f)
        if s is None:
            print("跳过未匹配 LED 字符串的文件:", f)
            continue
        im = imread(f)
        if im.ndim == 3:
            # 若是RGB，转换为灰度
            im = np.mean(im, axis=2).astype(np.float32)
        else:
            im = im.astype(np.float32)
        imgs.append(im)
        led_strings.append(s)
        filenames.append(os.path.basename(f))
    return filenames, np.array(imgs), led_strings

# helper: generate LED positions (x,y,z) in meters
def generate_led_positions(n_leds, layout='ring', radius=LED_RADIUS, height=LED_HEIGHT):
    if layout == 'ring':
        angles = np.linspace(0, 2*np.pi, n_leds, endpoint=False)
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        zs = np.ones_like(xs) * height
        pos = np.stack([xs, ys, zs], axis=1)
        return pos
    elif layout == 'grid':
        # generate roughly square grid
        side = int(np.ceil(np.sqrt(n_leds)))
        coords = []
        pitch = radius * 2 / (side-1) if side>1 else 0
        for i in range(side):
            for j in range(side):
                coords.append(((i-(side-1)/2)*pitch, (j-(side-1)/2)*pitch, height))
                if len(coords) >= n_leds: break
            if len(coords) >= n_leds: break
        return np.array(coords[:n_leds])
    else:
        raise ValueError("未知 layout")

# if LED csv provided, parse it: csv columns (index,x(m),y(m),z(m))
def load_led_csv(path):
    data = np.loadtxt(path, delimiter=',')
    # expect rows like: index,x,y,z
    # if only x,y given, assume z=LED_HEIGHT
    if data.shape[1] == 2:
        xs = data[:,0]; ys = data[:,1]; zs = np.ones_like(xs)*LED_HEIGHT
    elif data.shape[1] >= 3:
        xs = data[:,1]; ys = data[:,2]; zs = data[:,3] if data.shape[1]>3 else np.ones_like(xs)*LED_HEIGHT
    else:
        raise ValueError("CSV 列数不符合预期。")
    return np.stack([xs, ys, zs], axis=1)

# compute shift in Fourier bins for given LED pos (meters)
def compute_shift_pixels(led_xy_z, Nx, Ny, dx, wavelength):
    # led_xy_z: (x,y,z)
    x, y, z = led_xy_z
    # sin(theta_x) = x / sqrt(x^2 + y^2 + z^2)  (approx for plane wave arriving)
    denom = np.sqrt(x*x + y*y + z*z)
    sinx = x / denom
    siny = y / denom
    # shift in Fourier pixels derived earlier:
    # shift_pix = (N * dx / lambda) * sin(theta)
    shift_x = (Nx * dx / wavelength) * sinx
    shift_y = (Ny * dx / wavelength) * siny
    return shift_x, shift_y

# make pupil mask in frequency bins (Nx,Ny canvas)
def make_pupil(Nx, Ny, dx, wavelength, NA, padding=1):
    # cutoff bins radius
    radius = (Nx * dx / wavelength) * NA * padding
    cx = Nx//2; cy = Ny//2
    Y, X = np.ogrid[:Ny, :Nx]
    R = np.sqrt((X-cx)**2 + (Y-cy)**2)
    P = R <= radius
    return P.astype(np.complex64)

# main FPM reconstruction (basic ePIE-style)
def fpm_reconstruct(imgs, led_strings, led_positions, wavelength, dx, NA, iters=40, alpha=0.8, upsample=2):
    n_imgs, H, W = imgs.shape
    # decide canvas size
    canvas_Nx = int(W * upsample)
    canvas_Ny = int(H * upsample)
    print("image size:", W, H, "canvas:", canvas_Nx, canvas_Ny)

    # build pupil on canvas
    P = make_pupil(canvas_Nx, canvas_Ny, dx, wavelength, NA, padding=1.0)
    # initialize object spectrum (complex) random or small
    obj_spec = np.zeros((canvas_Ny, canvas_Nx), dtype=np.complex64)
    # initialize with low-res FT average to help convergence
    avg = np.mean(imgs, axis=0)
    # zero-pad avg to canvas and set initial magnitude
    pad_x = (canvas_Nx - W)//2; pad_y = (canvas_Ny - H)//2
    tmp = np.zeros((canvas_Ny, canvas_Nx), dtype=np.float32)
    tmp[pad_y:pad_y+H, pad_x:pad_x+W] = avg
    obj_spec = fftshift(fft2(ifftshift(tmp))).astype(np.complex64)

    # precompute shifts and pupil slices indices
    shifts = []
    # map each led_string to a (kx,ky) shift by averaging positions indicated by '1'
    # but in typical FPM each image corresponds to one LED ON; your filenames may have single '1' or multiple ones.
    for s in led_strings:
        # find the index of the '1' which is the lit LED(s)
        # take the first '1' occurrence as representative (if multiple ones appear, we average their positions)
        idxs = [i for i,ch in enumerate(s) if ch=='1']
        if len(idxs)==0:
            shifts.append((0.0, 0.0))
            continue
        # average positions of these idxs
        pos = np.mean(led_positions[idxs,:], axis=0)
        sx, sy = compute_shift_pixels(pos, canvas_Nx, canvas_Ny, dx, wavelength)
        shifts.append((sx, sy))

    # iterative loop
    for it in range(iters):
        print(f"Iter {it+1}/{iters}")
        # optionally randomize order
        order = np.arange(n_imgs)
        np.random.shuffle(order)
        for ii in order:
            I = imgs[ii]
            # center crop indices for this shift
            sx, sy = shifts[ii]
            # compute integer pixel shifts (may include fractional part for better accuracy with interpolation; here we use roll + phase ramp for fractional)
            cx = canvas_Nx//2; cy = canvas_Ny//2
            # desired center in obj_spec (in pixel coords)
            center_x = int(np.round(cx + sx))
            center_y = int(np.round(cy + sy))

            # extract window of size W x H from obj_spec centered at (center_x, center_y)
            x0 = center_x - W//2
            y0 = center_y - H//2
            # handle boundaries with padding
            # create a temp spec patch initialized zero
            spec_patch = np.zeros((canvas_Ny, canvas_Nx), dtype=np.complex64)
            # copy the relevant region using modular arithmetic or clipping
            x1 = max(0, x0); x2 = min(canvas_Nx, x0+W)
            y1 = max(0, y0); y2 = min(canvas_Ny, y0+H)
            sx1 = x1 - x0; sy1 = y1 - y0
            sx2 = sx1 + (x2-x1); sy2 = sy1 + (y2-y1)

            spec_patch[y1:y2, x1:x2] = obj_spec[y1:y2, x1:x2] * P[y1:y2, x1:x2]

            # inverse FFT to spatial field (pad reduces aliasing)
            field = fftshift(ifft2(ifftshift(spec_patch)))
            # crop to image region
            im_field = np.zeros((H,W), dtype=np.complex64)
            im_field[sy1:sy2, sx1:sx2] = field[y1:y2, x1:x2]

            # enforce intensity constraint: replace amplitude with sqrt(I)
            eps = 1e-6
            current_amp = np.abs(im_field)
            desired_amp = np.sqrt(I + eps)
            # avoid division by zero
            mask = current_amp > 0
            updated_field = np.copy(im_field)
            updated_field[mask] = im_field[mask] * (desired_amp[mask] / current_amp[mask])
            # if zero amplitude, set based on desired amplitude and zero phase
            updated_field[~mask] = desired_amp[~mask]

            # get updated spectrum patch
            updated_spec_patch = fftshift(fft2(ifftshift(pad_to_canvas(updated_field, canvas_Nx, canvas_Ny, x0, y0))))

            # difference and update object spectrum (within pupil support)
            delta = updated_spec_patch - spec_patch
            # update only where pupil exists to avoid adding outside support
            obj_spec = obj_spec + alpha * delta * P

        # end for images
    # end iterations

    # final object in spatial domain
    final_obj_img = fftshift(ifft2(ifftshift(obj_spec)))
    amplitude = np.abs(crop_to_sensor_plane(final_obj_img, W, H, pad_x, pad_y))
    phase = np.angle(crop_to_sensor_plane(final_obj_img, W, H, pad_x, pad_y))
    return amplitude, phase

# utility: pad a HxW spatial field into canvas at offset x0,y0 then return full canvas array (spatial)
def pad_to_canvas(spatial_field, canvas_Nx, canvas_Ny, x0, y0):
    H, W = spatial_field.shape
    canvas = np.zeros((canvas_Ny, canvas_Nx), dtype=np.complex64)
    x1 = max(0, x0); y1 = max(0, y0)
    x2 = min(canvas_Nx, x0+W); y2 = min(canvas_Ny, y0+H)
    sx1 = x1 - x0; sy1 = y1 - y0
    sx2 = sx1 + (x2-x1); sy2 = sy1 + (y2-y1)
    canvas[y1:y2, x1:x2] = spatial_field[sy1:sy2, sx1:sx2]
    return canvas

# utility: crop final object back to original sensor region (inverse of pad)
def crop_to_sensor_plane(spatial_canvas, W, H, pad_x, pad_y):
    return spatial_canvas[pad_y:pad_y+H, pad_x:pad_x+W]

# ----------------- main runnable CLI ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.', help='包含 snapshot_*.tif 的目录')
    parser.add_argument('--out_prefix', type=str, default='fpm_result', help='输出文件前缀')
    parser.add_argument('--led_csv', type=str, default=LED_POS_CSV, help='可选 LED csv 文件路径')
    parser.add_argument('--layout', type=str, default=LED_LAYOUT, help='自动生成 LED 布局: ring 或 grid')
    parser.add_argument('--led_count', type=int, default=LED_COUNT, help='自动布局时 LED 数量')
    parser.add_argument('--led_radius', type=float, default=LED_RADIUS, help='自动布局半径 (m)')
    parser.add_argument('--wavelength', type=float, default=WAVELENGTH, help='波长 (m)')
    parser.add_argument('--pixel_size', type=float, default=CAM_PIXEL_SIZE, help='相机像素尺寸 (m)')
    parser.add_argument('--mag', type=float, default=MAGNIFICATION, help='物镜放大倍数')
    parser.add_argument('--NA', type=float, default=OBJ_NA, help='物镜 NA')
    parser.add_argument('--led_height', type=float, default=LED_HEIGHT, help='LED 高度 (m)')
    parser.add_argument('--iters', type=int, default=NUM_ITERS, help='迭代次数')
    parser.add_argument('--alpha', type=float, default=ALPHA, help='更新步长')
    args = parser.parse_args()

    fnames, imgs, led_strings = load_images(args.data_dir)
    if len(imgs)==0:
        print("没有找到符合的图片。请检查 data_dir 与文件命名。")
        return
    H, W = imgs.shape[1], imgs.shape[2]
    dx = args.pixel_size / args.mag
    print("读取图片数量:", len(imgs), "单张尺寸:", W, H, "对象面采样 dx=", dx)

    # prepare LED positions
    if args.led_csv:
        led_positions = load_led_csv(args.led_csv)
    else:
        # if led_strings length known, use that length
        maxlen = max(len(s) for s in led_strings)
        led_positions = generate_led_positions(maxlen, layout=args.layout, radius=args.led_radius, height=args.led_height)
        if led_positions.shape[0] < maxlen:
            raise RuntimeError("生成的 LED 数目小于文件里 LED 字符串长度")

    amp, ph = fpm_reconstruct(imgs, led_strings, led_positions, args.wavelength, dx, args.NA, iters=args.iters, alpha=args.alpha, upsample=OBJ_UPSAMPLE)

    # 保存结果
    amp_norm = (amp - amp.min()) / (amp.max()-amp.min() + 1e-12)
    ph_wrapped = np.angle(np.exp(1j*ph))  # [-pi, pi]
    out_amp = (amp_norm*65535).astype(np.uint16)
    out_phase = ((ph_wrapped+np.pi)/(2*np.pi)*65535).astype(np.uint16)
    imsave(f"{args.out_prefix}_amp.tif", out_amp)
    imsave(f"{args.out_prefix}_phase.tif", out_phase)
    print("重建完成，已保存:", f"{args.out_prefix}_amp.tif", f"{args.out_prefix}_phase.tif")

    # quick show
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.title('Amplitude'); plt.imshow(amp_norm, cmap='gray'); plt.axis('off')
    plt.subplot(1,2,2); plt.title('Phase'); plt.imshow(ph_wrapped, cmap='jet'); plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
