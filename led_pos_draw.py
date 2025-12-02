import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# --- 1. 定义布局参数 ---

# 各圈LED数量 (从内到外)
led_counts = [1, 6, 12, 24]

# 更新后的半径 (单位: mm)
dim = [0, 22, 44, 84] 
radii  = [number / 2 for number in dim]

# 各圈Z轴高度 (单位: mm)
z_layer4 = 120
z_layer3 = z_layer4 + 12
z_layer2 = z_layer3 + 10
z_layer1 = z_layer2

z_levels = [z_layer1, z_layer2, z_layer3, z_layer4]

# --- 2. 计算每个LED的坐标 ---

all_coords = []
led_index = 1 # LED从1开始编号
coords_by_layer = {i+1: [] for i in range(len(led_counts))}

# 第1圈: 中心 (1颗LED)
x, y, z = radii[0], radii[0], z_levels[0]
coords_by_layer[1].append((x, y, z, led_index))
all_coords.append({'index': led_index, 'layer': 1, 'x': x, 'y': y, 'z': z})
led_index += 1

# 第2圈: 6颗LED, 从右水平开始, 顺时针
n_leds, radius, z = led_counts[1], radii[1], z_levels[1]
angle_step = 2 * np.pi / n_leds
start_angle = 0  # 0弧度代表X轴正方向 (右侧水平)
for i in range(n_leds):
    angle = start_angle - i * angle_step # 顺时针: 角度为负
    x, y = radius * np.cos(angle), radius * np.sin(angle)
    coords_by_layer[2].append((x, y, z, led_index))
    all_coords.append({'index': led_index, 'layer': 2, 'x': x, 'y': y, 'z': z})
    led_index += 1

# 第3圈: 12颗LED, 从左水平开始, 逆时针
n_leds, radius, z = led_counts[2], radii[2], z_levels[2]
angle_step = 2 * np.pi / n_leds
start_angle = np.pi  # pi弧度代表X轴负方向 (左侧水平)
for i in range(n_leds):
    angle = start_angle + i * angle_step # 逆时针: 角度为正
    x, y = radius * np.cos(angle), radius * np.sin(angle)
    coords_by_layer[3].append((x, y, z, led_index))
    all_coords.append({'index': led_index, 'layer': 3, 'x': x, 'y': y, 'z': z})
    led_index += 1

# 第4圈: 24颗LED, 从左水平开始, 逆时针
n_leds, radius, z = led_counts[3], radii[3], z_levels[3]
angle_step = 2 * np.pi / n_leds
start_angle = np.pi  # pi弧度代表X轴负方向 (左侧水平)
for i in range(n_leds):
    angle = start_angle + i * angle_step # 逆时针: 角度为正
    x, y = radius * np.cos(angle), radius * np.sin(angle)
    coords_by_layer[4].append((x, y, z, led_index))
    all_coords.append({'index': led_index, 'layer': 4, 'x': x, 'y': y, 'z': z})
    led_index += 1

# --- 3. 创建示意图 ---
fig = plt.figure(figsize=(18, 9))
fig.suptitle('LED Layout Schematic (Total 43 LEDs: 1-6-12-24)', fontsize=16)

# 子图1: 2D 俯视图
ax1 = fig.add_subplot(1, 2, 1)
colors = ['red', 'green', 'blue', 'purple']
for layer_num, layer_coords in coords_by_layer.items():
    x_coords = [c[0] for c in layer_coords]
    y_coords = [c[1] for c in layer_coords]
    indices = [c[3] for c in layer_coords]
    ax1.scatter(x_coords, y_coords, c=colors[layer_num-1], s=60, label=f'Layer {layer_num} (R={radii[layer_num-1]}mm)')
    for i, txt in enumerate(indices):
        ax1.annotate(txt, (x_coords[i], y_coords[i]), textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)

ax1.set_title('2D Top-Down View')
ax1.set_xlabel('X (mm)')
ax1.set_ylabel('Y (mm)')
ax1.set_aspect('equal', adjustable='box')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()
# 绘制参考圆环
for r in radii[1:]:
    ax1.add_artist(plt.Circle((0, 0), r, color='gray', fill=False, linestyle=':', linewidth=1))

# 子图2: 3D 视图
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
for layer_num, layer_coords in coords_by_layer.items():
    x_coords = [c[0] for c in layer_coords]
    y_coords = [c[1] for c in layer_coords]
    z_coords = [c[2] for c in layer_coords]
    ax2.scatter(x_coords, y_coords, z_coords, c=colors[layer_num-1], s=60, depthshade=True, label=f'Layer {layer_num} (Z={z_levels[layer_num-1]}mm)')

ax2.set_title('3D Perspective View')
ax2.set_xlabel('X (mm)')
ax2.set_ylabel('Y (mm)')
ax2.set_zlabel('Z (mm)')
ax2.set_zticks(np.unique(z_levels)) # 自动设置Z轴刻度
ax2.legend()

# ======================= 关键代码 =======================
# 通过设置box_aspect，使得X,Y,Z轴的视觉比例与数据范围一致，从而实现1:1:1的真实物理比例。
# 这样，图中的物体就不会因为坐标轴的自动拉伸而变形。
ax2.set_box_aspect([
    np.ptp(np.array([c['x'] for c in all_coords])), # X-range
    np.ptp(np.array([c['y'] for c in all_coords])), # Y-range
    np.ptp(np.array([c['z'] for c in all_coords]))  # Z-range
])
# ========================================================

ax2.view_init(elev=25, azim=-50) # 设置一个好的观察视角

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# --- (可选) 4. 打印所有LED的坐标列表 ---
print("Generated LED Coordinates (Index, Layer, X, Y, Z)")
print("-" * 50)
print(f"{'Index':<6} {'Layer':<6} {'X (mm)':<12} {'Y (mm)':<12} {'Z (mm)':<10}")
print("-" * 50)
for led in all_coords:
    x_val = round(led['x'], 2)
    y_val = round(led['y'], 2)
    z_val = round(led['z'], 2)
    print(f"{led['index']:<6} {led['layer']:<6} {x_val:<12.2f} {y_val:<12.2f} {z_val:<10.2f}")
