import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

# --- НАЛАШТУВАННЯ ---
filename = 'density_profile.dat'
output_image = 'result_angle.png'

# --- 1. ЧИТАННЯ ---
print(f"Reading {filename}...")
try:
    data = np.loadtxt(filename, skiprows=4)
except:
    print("Error reading file.")
    exit()

x_raw = data[:, 1]
z_raw = data[:, 2]
rho_raw = data[:, 4]

# Поріг густини
max_rho = np.max(rho_raw)
threshold = max_rho * 0.5
mask_liquid = rho_raw > threshold
x_liq = x_raw[mask_liquid]
z_liq = z_raw[mask_liquid]

# --- 2. КОНТУР ---
unique_zs = np.unique(z_liq)
x_contour = []
z_contour = []
for z_val in unique_zs:
    xs = x_liq[z_liq == z_val]
    if len(xs) > 0:
        x_contour.append(np.min(xs))
        x_contour.append(np.max(xs))
        z_contour.append(z_val)
        z_contour.append(z_val)

x_contour = np.array(x_contour)
z_contour = np.array(z_contour)

# --- 3. ФІТИНГ КОЛА ---
z_water_bottom = np.min(z_liq) # Реальне дно води
# Беремо точки вище дна на 4А, щоб уникнути спотворень
fit_mask = z_contour > (z_water_bottom + 4.0)
x_fit = x_contour[fit_mask]
z_fit = z_contour[fit_mask]

def residuals(p, x, z):
    xc, zc, R = p
    return np.sqrt((x-xc)**2 + (z-zc)**2) - R

p0 = [np.mean(x_fit), np.mean(z_fit), (np.max(x_fit)-np.min(x_fit))/2]
res = least_squares(residuals, p0, args=(x_fit, z_fit))
xc, zc, R = res.x

print(f"Fit: Center=({xc:.1f}, {zc:.1f}), R={R:.1f}")
print(f"Water Bottom Z: {z_water_bottom:.1f}")

# --- 4. РОЗРАХУНОК КУТА (ГЕОМЕТРИЧНИЙ) ---
# Ми рахуємо кут перетину кола з площиною дна води (z_water_bottom)
# Delta H - відстань від центру кола до дна води
dist_to_bottom = zc - z_water_bottom

if dist_to_bottom >= R:
    # Центр кола вище, ніж радіус -> Коло ледь торкається або висить
    theta = 180.0
    print("Status: Superhydrophobic (Droplet sits like a ball)")
elif dist_to_bottom <= -R:
    theta = 0.0
else:
    # Формула: cos(theta) = - (dist / R)
    # Якщо центр високо, dist > 0, cos < 0 -> кут > 90
    cos_theta = - (dist_to_bottom / R)
    theta_rad = np.arccos(cos_theta)
    theta = np.degrees(theta_rad)

print("="*40)
print(f"FINAL CONTACT ANGLE: {theta:.2f}°")
print("="*40)

# --- 5. МАЛЮНОК ---
plt.figure(figsize=(10, 8))
plt.title(f'Result: {theta:.1f}° (Superhydrophobic SPC/E on LJ-Carbon)')

plt.scatter(x_liq, z_liq, c='lightblue', alpha=0.3, label='Water')
plt.scatter(x_fit, z_fit, c='red', s=10, label='Fit Points')

t = np.linspace(0, 2*np.pi, 200)
plt.plot(xc+R*np.cos(t), zc+R*np.sin(t), 'g-', lw=2, label='Circle Fit')

plt.axhline(z_water_bottom, c='black', ls='--', label='Contact Line (Water Bottom)')
plt.legend()
plt.axis('equal')
plt.ylim(z_water_bottom - 10, zc + R + 10)
plt.savefig(output_image)
print(f"Saved {output_image}")