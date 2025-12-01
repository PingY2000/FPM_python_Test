# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 09:20:09 2025

@author: YUP2CHA
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.patches import Circle
from tqdm import tqdm

def generate_fpm_principle_gif(
    output_filename="fpm_principle.gif",
    na_obj=0.2,
    na_illum=0.4,
    led_array_size=15,
    img_size=512,
    fps=20
):
    """
    生成一个FPM频谱拼接原理的GIF动画。
    """
    synth_na = na_obj + na_illum
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    images = []

    # 绘制坐标轴和背景
    ax.set_facecolor('black')
    ax.set_xlim(-synth_na * 1.1, synth_na * 1.1)
    ax.set_ylim(-synth_na * 1.1, synth_na * 1.1)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel(r'$k_x/k_0$', fontsize=14)
    ax.set_ylabel(r'$k_y/k_0$', fontsize=14)
    ax.set_title("FPM Synthetic Aperture Stitching", fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # 绘制合成孔径范围
    synthetic_aperture = Circle((0, 0), synth_na, color='gray', fill=False, linestyle='--', label=f'Synthetic NA ({synth_na:.2f})')
    ax.add_patch(synthetic_aperture)
    
    # 绘制物镜孔径范围
    objective_aperture = Circle((0, 0), na_obj, color='cyan', fill=False, lw=2, label=f'Objective NA ({na_obj:.2f})')
    ax.add_patch(objective_aperture)
    ax.legend()
    
    # 生成螺旋形的LED扫描顺序
    positions = []
    half_size = led_array_size // 2
    if half_size == 0: half_size = 1
    x, y = 0, 0
    dx, dy = 0, -1
    for _ in range(led_array_size**2):
        if (-half_size <= x <= half_size) and (-half_size <= y <= half_size):
            kx_na = x * na_illum / half_size
            ky_na = y * na_illum / half_size
            if np.sqrt(kx_na**2 + ky_na**2) <= na_illum:
                if (kx_na, ky_na) not in positions:
                    positions.append((kx_na, ky_na))
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x + dx, y + dy
        
    # 累积的频谱覆盖图
    coverage = np.zeros((img_size, img_size))
    
    for i, (kx, ky) in tqdm(enumerate(positions), total=len(positions), desc="Generating frames"):
            # 绘制当前的子孔径
            sub_aperture = Circle((kx, ky), na_obj, color='lime', alpha=0.2)
            p = ax.add_patch(sub_aperture)
            
            # 绘制当前照明点
            led_dot, = ax.plot(kx, ky, 'ro', markersize=8)
            
            # --- 修改部分 1 ---
            # 将画布转换为图像 (兼容新版 matplotlib)
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            frame_rgba = np.array(buf)
            frame_rgb = frame_rgba[:, :, :3] # 从 RGBA 转换为 RGB
            images.append(frame_rgb)
            # --- 修改结束 ---
            
            # 在下一帧移除动态元素
            led_dot.remove()
            p.set_color('blue')
            p.set_alpha(0.1)

    # 添加结尾帧
    end_text = ax.text(0, 0, 'Full Spectrum\nRecovered!', 
                       ha='center', va='center', fontsize=24, color='white',
                       bbox=dict(facecolor='green', alpha=0.7))
    
    # 生成GIF
    print(f"Saving GIF to {output_filename}...")
    imageio.mimsave(output_filename, images, fps=fps)
    print("Done.")
    plt.close(fig)

if __name__ == "__main__":
    generate_fpm_principle_gif()