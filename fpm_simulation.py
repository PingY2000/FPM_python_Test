# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 11:43:10 2025

@author: YUP2CHA
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from skimage.data import camera, horse
from skimage.transform import resize
from skimage.util import random_noise
from tqdm import tqdm
import time

# =============================================================================
# 1. FPM 重建引擎 (从之前的回答整合而来)
# =============================================================================
class FPM_System:
    """完整的傅里叶叠层显微镜系统"""
    
    def __init__(self, params):
        self.params = params
        self.wavelength = params['wavelength']
        self.NA_obj = params['NA_obj']
        self.pixel_size = params['pixel_size']
        self.image_size = params['image_size']
        self.cutoff_freq = self.NA_obj / self.wavelength

    def reconstruct_2D(self, images, led_positions):
        """2D高分辨率重建（使用ePIE算法）"""
        print("\n" + "="*70)
        print(" "*25 + "2D FPM RECONSTRUCTION")
        print("="*70)
        start_time = time.time()
        
        O_hr, P, error_curve = self._ePIE_reconstruction(images, led_positions)
        
        elapsed_time = time.time() - start_time
        metrics = {
            'error_curve': error_curve,
            'reconstruction_time': elapsed_time
        }
        
        print(f"\n  Reconstruction completed in {elapsed_time:.2f} seconds")
        print(f"  Final RMSE: {error_curve[-1]:.6e}")
        print("="*70 + "\n")
        
        return O_hr, P, metrics

    def _ePIE_reconstruction(self, images, led_positions):
        """ePIE重建核心算法"""
        N_images, H_lr, W_lr = images.shape
        iterations = self.params.get('iterations', 50)
        alpha = self.params.get('alpha', 1.0)
        beta = self.params.get('beta', 1.0)
        upsampling = self.params.get('upsampling', 4)
        
        H_hr = H_lr * upsampling
        W_hr = W_lr * upsampling
        
        print(f"  Low-res size : {H_lr} x {W_lr}")
        print(f"  High-res size: {H_hr} x {W_hr}")
        print(f"  Iterations   : {iterations}\n")
        
        # 初始化高分辨率样品频谱
        center_idx = N_images // 2
        init_image = zoom(images[center_idx], upsampling)
        O_hr_spectrum = fftshift(fft2(init_image))
        
        # 初始化光瞳函数
        P = self._initialize_pupil(H_lr, W_lr)
        
        error_curve = []
        
        print("  Starting ePIE iterations...")
        for iteration in tqdm(range(iterations), desc="  Progress"):
            order = np.random.permutation(N_images)
            total_error = 0
            
            for idx in order:
                kx_pixel, ky_pixel = led_positions[idx]
                
                # 1. 裁剪高分辨率频谱以匹配光瞳大小
                O_hr_cropped = self._crop_spectrum(O_hr_spectrum, (ky_pixel, kx_pixel), (H_lr, W_lr))
                
                # 2. 通过光瞳
                G = O_hr_cropped * P
                
                # 3. 逆FFT到空间域
                g = ifft2(ifftshift(G))
                
                # 4. 振幅替换
                measured_amp = np.sqrt(images[idx])
                g_updated = measured_amp * np.exp(1j * np.angle(g))
                
                total_error += np.sum((np.abs(g) - measured_amp)**2)
                
                # 5. FFT回频域
                G_updated = fftshift(fft2(g_updated))
                
                # 6. 计算更新量
                delta_G = G_updated - G
                
                # 7. 更新样品频谱
                P_conj = np.conj(P)
                update_term_O = alpha * P_conj * delta_G / (np.max(np.abs(P)**2) + 1e-8)
                O_hr_spectrum = self._update_spectrum(O_hr_spectrum, update_term_O, (ky_pixel, kx_pixel))

                # 8. 更新光瞳函数
                O_hr_cropped_conj = np.conj(O_hr_cropped)
                update_term_P = beta * O_hr_cropped_conj * delta_G / (np.max(np.abs(O_hr_cropped)**2) + 1e-8)
                P += update_term_P

            error_curve.append(np.sqrt(total_error / (N_images * H_lr * W_lr)))
        
        return ifft2(ifftshift(O_hr_spectrum)), P, np.array(error_curve)

    def _initialize_pupil(self, H, W):
        fy, fx = np.meshgrid(
            fftfreq(W, d=self.pixel_size),
            fftfreq(H, d=self.pixel_size),
            indexing='xy'
        )
        f_radius = np.sqrt(fx**2 + fy**2)
        return (f_radius <= self.cutoff_freq).astype(complex)

    def _crop_spectrum(self, spectrum, center_shift, size):
        """从大频谱中根据中心偏移裁剪出一块区域，带边界处理。"""
        cy, cx = center_shift
        H_crop, W_crop = size
        H_hr, W_hr = spectrum.shape
        center_y_hr, center_x_hr = H_hr // 2, W_hr // 2
        
        # 计算在大频谱中的实际裁剪区域
        start_y_hr = center_y_hr - H_crop // 2 + cy
        end_y_hr = start_y_hr + H_crop
        start_x_hr = center_x_hr - W_crop // 2 + cx
        end_x_hr = start_x_hr + W_crop
        
        # 计算在目标 patch 中的填充区域
        start_y_patch = 0
        end_y_patch = H_crop
        start_x_patch = 0
        end_x_patch = W_crop
        
        # 处理边界情况
        if start_y_hr < 0:
            start_y_patch = -start_y_hr
            start_y_hr = 0
        if end_y_hr > H_hr:
            end_y_patch = H_crop - (end_y_hr - H_hr)
            end_y_hr = H_hr
        if start_x_hr < 0:
            start_x_patch = -start_x_hr
            start_x_hr = 0
        if end_x_hr > W_hr:
            end_x_patch = W_crop - (end_x_hr - W_hr)
            end_x_hr = W_hr
            
        # 创建一个正确尺寸的零矩阵
        cropped_patch = np.zeros(size, dtype=spectrum.dtype)
        
        # 将有效区域的数据复制过来
        if start_y_patch < end_y_patch and start_x_patch < end_x_patch:
            cropped_patch[start_y_patch:end_y_patch, start_x_patch:end_x_patch] = \
                spectrum[start_y_hr:end_y_hr, start_x_hr:end_x_hr]
                
        return cropped_patch

    def _update_spectrum(self, spectrum, update_patch, center_shift):
        """将更新块加回到大频谱的相应位置，带边界处理。"""
        cy, cx = center_shift
        H_patch, W_patch = update_patch.shape
        H_hr, W_hr = spectrum.shape
        center_y_hr, center_x_hr = H_hr // 2, W_hr // 2
        
        # 计算在大频谱中的实际更新区域
        start_y_hr = center_y_hr - H_patch // 2 + cy
        end_y_hr = start_y_hr + H_patch
        start_x_hr = center_x_hr - W_patch // 2 + cx
        end_x_hr = start_x_hr + W_patch
        
        # 计算在 update_patch 中要裁剪的区域
        start_y_patch = 0
        end_y_patch = H_patch
        start_x_patch = 0
        end_x_patch = W_patch
    
        # 处理边界情况
        if start_y_hr < 0:
            start_y_patch = -start_y_hr
            start_y_hr = 0
        if end_y_hr > H_hr:
            end_y_patch = H_patch - (end_y_hr - H_hr)
            end_y_hr = H_hr
        if start_x_hr < 0:
            start_x_patch = -start_x_hr
            start_x_hr = 0
        if end_x_hr > W_hr:
            end_x_patch = W_patch - (end_x_hr - W_hr)
            end_x_hr = W_hr
    
        # 将有效部分的更新应用到大频谱上
        if start_y_hr < end_y_hr and start_x_hr < end_x_hr:
            spectrum[start_y_hr:end_y_hr, start_x_hr:end_x_hr] += \
                update_patch[start_y_patch:end_y_patch, start_x_patch:end_x_patch]
                
        return spectrum

# =============================================================================
# 2. 仿真辅助工具
# =============================================================================
def create_test_object(size):
    """创建高分辨率的测试样品（振幅和相位）"""
    # 处理振幅图像 (camera是灰度图，没问题)
    amp = resize(camera(), size, anti_aliasing=True)
    amp = amp / amp.max()
    
    # --- 解决方案在这里 ---
    # 1. 先加载 horse() 图像
    horse_img = horse()
    # 2. 将其从布尔类型转换为浮点数类型
    horse_img_float = horse_img.astype(float)
    # 3. 现在可以安全地使用 anti_aliasing=True 进行缩放了
    phase = resize(horse_img_float, size, anti_aliasing=True)
    # --- 修改结束 ---
    
    phase = (1 - phase) * np.pi  # 让马的轮廓有pi的相移
    
    return amp * np.exp(1j * phase)

def generate_led_positions_2d(params):
    """根据参数生成LED位置（单位：高分辨率频谱中的像素偏移）"""
    NA_illum = params['NA_illum']
    wavelength = params['wavelength']
    upsampling = params['upsampling']
    pixel_size = params['pixel_size']
    H_lr, W_lr = params['image_size']
    
    # 高分辨率网格的频率步长
    df_hr = 1 / (H_lr * upsampling * pixel_size)
    
    # 计算最大照明频率对应的像素偏移
    max_k_illum = NA_illum / wavelength
    max_pixel_shift = int(np.floor(max_k_illum / df_hr))
    
    # 设定LED阵列在像素偏移空间中的范围和步长
    led_array_size = params.get('led_array_size', 15)
    half_size = led_array_size // 2
    
    positions = []
    
    # 生成一个螺旋形的LED扫描顺序，从中心到外围
    x, y = 0, 0
    dx, dy = 0, -1
    for _ in range(led_array_size**2):
        if (-half_size <= x <= half_size) and (-half_size <= y <= half_size):
            kx_pixel = int(x * max_pixel_shift / half_size)
            ky_pixel = int(y * max_pixel_shift / half_size)
            if (kx_pixel, ky_pixel) not in positions:
                positions.append((kx_pixel, ky_pixel))
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x + dx, y + dy
        
    return np.array(positions)

class FPM_Simulator:
    """FPM前向模型模拟器"""
    def __init__(self, ground_truth, params):
        self.ground_truth = ground_truth
        self.params = params
        self.H_hr, self.W_hr = ground_truth.shape
        self.H_lr, self.W_lr = params['image_size']
        
        # 创建物镜光瞳函数（低通滤波器）
        fy, fx = np.meshgrid(
            fftfreq(self.W_lr, d=params['pixel_size']),
            fftfreq(self.H_lr, d=params['pixel_size']),
            indexing='xy'
        )
        f_radius = np.sqrt(fx**2 + fy**2)
        self.pupil = (f_radius <= params['NA_obj'] / params['wavelength']).astype(complex)
    
    def generate_low_res_data(self, led_positions):
        """模拟生成低分辨率图像序列"""
        print("Simulating data acquisition...")
        N_images = len(led_positions)
        lr_images = np.zeros((N_images, self.H_lr, self.W_lr))
        
        O_hr_spectrum = fftshift(fft2(self.ground_truth))
        
        for i, (kx, ky) in tqdm(enumerate(led_positions), total=N_images, desc="  Simulating"):
            # 1. 裁剪高分辨率频谱
            O_hr_cropped = self._crop_spectrum(O_hr_spectrum, (ky, kx), (self.H_lr, self.W_lr))
            
            # 2. 通过光瞳
            G = O_hr_cropped * self.pupil
            
            # 3. 得到相机平面的场
            g = ifft2(ifftshift(G))
            
            # 4. 计算强度
            intensity = np.abs(g)**2
            
            # 5. 添加噪声 (可选)
            if self.params.get('add_noise', False):
                intensity = random_noise(intensity, mode='poisson')
            
            lr_images[i, :, :] = intensity
        
        print("Simulation complete.")
        return lr_images

    def _crop_spectrum(self, spectrum, center_shift, size):
        cy, cx = center_shift
        H_crop, W_crop = size
        
        center_y_hr, center_x_hr = self.H_hr // 2, self.W_hr // 2
        
        start_y = center_y_hr - H_crop // 2 + cy
        start_x = center_x_hr - W_crop // 2 + cx
        
        # 边界检查
        if not (0 <= start_y and start_y + H_crop <= self.H_hr and 
                0 <= start_x and start_x + W_crop <= self.W_hr):
            # 如果超出边界，返回一个零矩阵
            return np.zeros(size, dtype=spectrum.dtype)
            
        return spectrum[start_y : start_y + H_crop, start_x : start_x + W_crop]

# =============================================================================
# 3. 主执行脚本
# =============================================================================
if __name__ == "__main__":
    
    # --- 仿真参数定义 ---
    params = {
        'wavelength': 532e-9,          # 波长 (m)
        'NA_obj': 0.1,                 # 物镜NA (低NA，更容易看到效果)
        'NA_illum': 0.4,               # 照明NA
        'pixel_size': 1.67e-6,         # 相机像素尺寸 (m)
        'image_size': (64, 64),        # 低分辨率图像尺寸
        'upsampling': 4,               # 上采样倍数 (决定了高分辨率图像的尺寸)
        'led_array_size': 15,          # 虚拟LED阵列边长
        'add_noise': True,             # 是否添加噪声
        
        # 重建参数
        'iterations': 20,              # 2D重建迭代次数
        'alpha': 1.0,                  # 样品更新步长
        'beta': 1.0,                   # 光瞳更新步长
    }

    # 1. 创建Ground Truth
    hr_size = (params['image_size'][0] * params['upsampling'], 
               params['image_size'][1] * params['upsampling'])
    ground_truth_object = create_test_object(hr_size)
    
    # 2. 生成LED位置
    led_positions = generate_led_positions_2d(params)
    
    # 3. 模拟数据采集
    simulator = FPM_Simulator(ground_truth_object, params)
    low_res_images = simulator.generate_low_res_data(led_positions)
    
    # 4. 执行FPM重建
    fpm_system = FPM_System(params)
    O_recon, P_recon, metrics = fpm_system.reconstruct_2D(low_res_images, led_positions)
    
    # 5. 可视化结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("FPM Simulation and Reconstruction", fontsize=16, fontweight='bold')
    
    # 原始低分辨率图像
    ax = axes[0, 0]
    im = ax.imshow(low_res_images[len(low_res_images)//2], cmap='gray')
    ax.set_title("Input: Low-Res Image (Center)", fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Ground Truth
    ax = axes[0, 1]
    im = ax.imshow(np.abs(ground_truth_object), cmap='gray')
    ax.set_title("Ground Truth (Amplitude)", fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[0, 2]
    im = ax.imshow(np.angle(ground_truth_object), cmap='twilight')
    ax.set_title("Ground Truth (Phase)", fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 重建结果
    ax = axes[1, 1]
    im = ax.imshow(np.abs(O_recon), cmap='gray')
    ax.set_title("Reconstruction (Amplitude)", fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[1, 2]
    # 对重建相位进行相位解包裹以获得更好的可视化效果
    from skimage.restoration import unwrap_phase
    phase_recon = unwrap_phase(np.angle(O_recon))
    im = ax.imshow(phase_recon, cmap='twilight')
    ax.set_title("Reconstruction (Phase)", fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 误差曲线和光瞳
    ax = axes[1, 0]
    ax.semilogy(metrics['error_curve'], 'b-', linewidth=2)
    ax.set_title('Convergence Curve (RMSE)', fontweight='bold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('RMSE')
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
