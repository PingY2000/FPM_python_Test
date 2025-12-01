# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 11:44:04 2025

@author: YUP2CHA
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
from tqdm import tqdm
import time

class FPM_System:
    """
    完整的傅里叶叠层显微镜（FPM）系统控制类。
    
    该类封装了2D重建、LED位置校正、3D重建接口和结果可视化等核心功能。
    """
    
    def __init__(self, params):
        """
        初始化FPM系统。
        
        参数:
        params (dict): 包含所有系统和算法参数的字典。
            'wavelength': 波长 (m)
            'NA_obj': 物镜NA
            'pixel_size': 相机像素尺寸 (m)
            'image_size': 低分辨率图像尺寸 (H, W)
            'upsampling': 上采样倍数, e.g., 4
            'NA_illum': 照明NA
            'iterations': 2D重建迭代次数
            'alpha', 'beta': ePIE算法更新步长
            'led_iterations', 'led_lr': LED校正的迭代次数和学习率
            'use_gpu': (未来扩展) 是否使用GPU
        """
        self.params = params
        self.wavelength = params['wavelength']
        self.NA_obj = params['NA_obj']
        self.pixel_size = params['pixel_size']
        self.image_size_lr = params['image_size']
        self.upsampling = params.get('upsampling', 4)
        self.image_size_hr = (self.image_size_lr[0] * self.upsampling, 
                              self.image_size_lr[1] * self.upsampling)

        # 计算系统衍生参数
        self.cutoff_freq = self.NA_obj / self.wavelength
        self.NA_illum = params.get('NA_illum', self.NA_obj)
        self.synth_NA = self.NA_obj + self.NA_illum
        self.resolution_original = 0.61 * self.wavelength / self.NA_obj
        self.resolution_synthetic = 0.61 * self.wavelength / self.synth_NA
        
        self._print_system_info()
    
    def _print_system_info(self):
        """打印系统初始化信息"""
        print("\n" + "="*70)
        print(" "*20 + "FPM SYSTEM INITIALIZED")
        print("="*70)
        print(f"  Wavelength         : {self.wavelength*1e9:.1f} nm")
        print(f"  Objective NA       : {self.NA_obj:.2f}")
        print(f"  Illumination NA    : {self.NA_illum:.2f}")
        print(f"  Synthetic NA       : {self.synth_NA:.2f}")
        print(f"  Original Resolution: {self.resolution_original*1e9:.1f} nm")
        print(f"  Synthetic Resolution: {self.resolution_synthetic*1e9:.1f} nm")
        print(f"  Resolution Gain    : {self.synth_NA/self.NA_obj:.2f}x")
        print(f"  Low-Res Image Size : {self.image_size_lr}")
        print(f"  High-Res Image Size: {self.image_size_hr}")
        print("="*70 + "\n")

    def reconstruct_2D(self, images, led_positions):
        """
        执行2D高分辨率重建。
        
        参数:
            images (ndarray): (N, H, W) 低分辨率强度图像序列。
            led_positions (ndarray): (N, 2) 每个图像对应的LED位置（像素偏移）。
        
        返回:
            O_recon (ndarray): 重建的高分辨率复振幅图像。
            P_recon (ndarray): 恢复的光瞳函数。
            metrics (dict): 包含重建过程指标的字典。
        """
        print("\n" + "="*70)
        print(" "*25 + "2D FPM RECONSTRUCTION")
        print("="*70)
        start_time = time.time()
        
        O_recon, P_recon, error_curve = self._ePIE_reconstruction(images, led_positions)
        
        elapsed_time = time.time() - start_time
        metrics = {
            'error_curve': error_curve,
            'final_error': error_curve[-1],
            'reconstruction_time': elapsed_time,
            'iterations': len(error_curve)
        }
        
        print(f"\n  Reconstruction completed in {elapsed_time:.2f} seconds.")
        print(f"  Final RMSE: {error_curve[-1]:.6e}")
        print("="*70 + "\n")
        
        return O_recon, P_recon, metrics

    def _ePIE_reconstruction(self, images, led_positions):
        """ePIE重建核心算法实现"""
        N_images, H_lr, W_lr = images.shape
        H_hr, W_hr = self.image_size_hr
        
        iterations = self.params.get('iterations', 20)
        alpha = self.params.get('alpha', 1.0)
        beta = self.params.get('beta', 1.0)
        
        print(f"  Reconstructing from {N_images} images over {iterations} iterations...")
        
        # 初始化高分辨率样品频谱
        center_idx = N_images // 2
        init_image = zoom(images[center_idx], self.upsampling)
        O_hr_spectrum = fftshift(fft2(init_image))
        
        # 初始化光瞳函数
        P = self._initialize_pupil(H_lr, W_lr)
        
        error_curve = []
        
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
        """创建基于物镜NA的理想圆形光瞳函数"""
        fy, fx = np.meshgrid(
            fftfreq(W, d=self.pixel_size),
            fftfreq(H, d=self.pixel_size),
            indexing='xy'
        )
        f_radius = np.sqrt(fx**2 + fy**2)
        return (f_radius <= self.cutoff_freq).astype(complex)

    def _crop_spectrum(self, spectrum, center_shift, size):
        """从大频谱中根据中心偏移裁剪出一块区域"""
        cy, cx = center_shift
        H_crop, W_crop = size
        H_hr, W_hr = spectrum.shape
        center_y_hr, center_x_hr = H_hr // 2, W_hr // 2
        
        start_y = center_y_hr - H_crop // 2 + cy
        start_x = center_x_hr - W_crop // 2 + cx
        
        return spectrum[start_y : start_y + H_crop, start_x : start_x + W_crop]

    def _update_spectrum(self, spectrum, update_patch, center_shift):
        """将更新块加回到大频谱的相应位置"""
        cy, cx = center_shift
        H_patch, W_patch = update_patch.shape
        H_hr, W_hr = spectrum.shape
        center_y_hr, center_x_hr = H_hr // 2, W_hr // 2
        
        start_y = center_y_hr - H_patch // 2 + cy
        start_x = center_x_hr - W_patch // 2 + cx
        
        spectrum[start_y : start_y + H_patch, start_x : start_x + W_patch] += update_patch
        return spectrum

    def visualize_2D_result(self, O_recon, P_recon, metrics, ground_truth=None):
        """
        可视化2D重建结果，并与Ground Truth（如果提供）对比。
        
        参数:
            O_recon (ndarray): 重建的高分辨率复振幅。
            P_recon (ndarray): 恢复的光瞳函数。
            metrics (dict): 重建指标。
            ground_truth (ndarray, optional): 真实的复振幅对象。
        """
        n_cols = 3 if ground_truth is None else 4
        fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 9))
        fig.suptitle("FPM 2D Reconstruction Results", fontsize=16, fontweight='bold')

        # 重建振幅
        ax = axes[0, 0]
        im = ax.imshow(np.abs(O_recon), cmap='gray')
        ax.set_title("Recon Amplitude", fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # 重建相位
        ax = axes[1, 0]
        phase_recon_unwrapped = unwrap_phase(np.angle(O_recon))
        im = ax.imshow(phase_recon_unwrapped, cmap='twilight')
        ax.set_title("Recon Phase", fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # 恢复的光瞳
        ax = axes[0, 1]
        im = ax.imshow(np.abs(P_recon), cmap='viridis')
        ax.set_title("Recovered Pupil (Abs)", fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, 1]
        im = ax.imshow(np.angle(P_recon), cmap='twilight')
        ax.set_title("Recovered Pupil (Phase)", fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 误差曲线
        ax = axes[0, 2]
        ax.semilogy(metrics['error_curve'], 'b-o', linewidth=2, markersize=4)
        ax.set_title("Convergence Curve", fontweight='bold')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RMSE")
        ax.grid(True, which="both", ls="--", alpha=0.5)

        # 空白或频谱图
        ax = axes[1, 2]
        spectrum_recon = np.log1p(np.abs(fftshift(fft2(O_recon))))
        im = ax.imshow(spectrum_recon, cmap='magma')
        ax.set_title("Recon Spectrum (log)", fontweight='bold')
        ax.axis('off')
        
        # Ground Truth对比
        if ground_truth is not None:
            ax = axes[0, 3]
            im = ax.imshow(np.abs(ground_truth), cmap='gray')
            ax.set_title("GT Amplitude", fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
            
            ax = axes[1, 3]
            im = ax.imshow(np.angle(ground_truth), cmap='twilight')
            ax.set_title("GT Phase", fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def reconstruct_3D(self, *args, **kwargs):
        """
        3D折射率重建的接口。
        
        注意：这是一个占位符，需要与`FPM_3D_Reconstructor`类集成。
        """
        print("\n" + "="*70)
        print(" "*25 + "3D FPM RECONSTRUCTION")
        print("="*70)
        print("  INFO: This is a placeholder for 3D reconstruction.")
        print("  To implement, integrate the FPM_3D_Reconstructor class here.")
        
        # 示例集成代码:
        # from .fpm_3d import FPM_3D_Reconstructor # 假设在另一文件中
        #
        # reconstructor = FPM_3D_Reconstructor(self.params)
        # RI_3D, metrics_3d = reconstructor.reconstruct(images_2D_complex, led_positions_3d, depth_layers)
        # return RI_3D, metrics_3d
        
        raise NotImplementedError("3D reconstruction is not fully implemented in this class yet.")
