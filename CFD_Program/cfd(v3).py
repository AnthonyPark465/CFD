import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from PIL import Image
import os

NX, NY = 160, 80
X_MIN, X_MAX = 0.0, 12.0
Y_MIN, Y_MAX = 0.0, 6.0
dx = (X_MAX - X_MIN) / (NX - 1)
dy = (Y_MAX - Y_MIN) / (NY - 1)

GRID_COLS, GRID_ROWS = 30, 15
domain_w = X_MAX - X_MIN
domain_h = Y_MAX - Y_MIN
cell_w = domain_w / GRID_COLS
cell_h = domain_h / GRID_ROWS
cell_max_len = 0.8 * min(cell_w, cell_h)

VEL_MIN, VEL_MAX = 0.0, 12.0
VEL_INIT = 4.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

x_lin = np.linspace(X_MIN, X_MAX, NX)
y_lin = np.linspace(Y_MIN, Y_MAX, NY)
Xg, Yg = np.meshgrid(x_lin, y_lin)

x_centers = X_MIN + (np.arange(GRID_COLS) + 0.5) * cell_w
y_centers = Y_MIN + (np.arange(GRID_ROWS) + 0.5) * cell_h
Xa, Ya = np.meshgrid(x_centers, y_centers)


def velocity_to_length(speed):
    k = cell_max_len / max(VEL_MAX, 1e-9)
    return np.minimum(speed * k, cell_max_len)

def bilerp(field, xq, yq):
    xi = (xq - X_MIN) / dx
    yi = (yq - Y_MIN) / dy
    i0 = np.clip(np.floor(xi).astype(int), 0, NX - 2)
    j0 = np.clip(np.floor(yi).astype(int), 0, NY - 2)
    tx = xi - i0
    ty = yi - j0
    f00 = field[j0, i0]
    f10 = field[j0, i0 + 1]
    f01 = field[j0 + 1, i0]
    f11 = field[j0 + 1, i0 + 1]
    return (1 - tx)*(1 - ty)*f00 + tx*(1 - ty)*f10 + (1 - tx)*ty*f01 + tx*ty*f11

def analytic_cylinder_velocity(Uinf, cx, cy, a):

    X = Xg - cx
    Y = Yg - cy
    r2 = X*X + Y*Y
    r = np.sqrt(r2)
    theta = np.arctan2(Y, X)

    eps = 1e-9
    inv_r2 = 1.0 / np.maximum(r2, eps)

    ur = Uinf * (1.0 - (a*a) * inv_r2) * np.cos(theta)
    uth = -Uinf * (1.0 + (a*a) * inv_r2) * np.sin(theta)

    Ux = ur * np.cos(theta) - uth * np.sin(theta)
    Uy = ur * np.sin(theta) + uth * np.cos(theta)

    inside = r2 <= a*a
    Ux = np.where(inside, 0.0, Ux)
    Uy = np.where(inside, 0.0, Uy)
    mask = inside
    return Ux, Uy, mask

def load_mask_from_png(path):
    if not os.path.exists(path):
        return None
    arr = np.array(Image.open(path).convert("L").resize((NX, NY), Image.NEAREST))
    solid = arr < 128
    return np.flipud(solid)

def solve_potential_vectorized(mask_solid, U_in, iters=300, tol=1e-5, omega=1.6):

    Lx = X_MAX - X_MIN
    phi = np.linspace(0.0, U_in * Lx, NX)[None, :].repeat(NY, axis=0)

    fluid = ~mask_solid
    inv_dx2 = 1.0 / dx**2
    inv_dy2 = 1.0 / dy**2
    denom = 2*inv_dx2 + 2*inv_dy2

    for _ in range(iters):

        phi_l = np.roll(phi, 1, axis=1)
        phi_r = np.roll(phi, -1, axis=1)
        phi_b = np.roll(phi, 1, axis=0)
        phi_t = np.roll(phi, -1, axis=0)

        phi_l = np.where(fluid, phi_l, phi)
        phi_r = np.where(fluid, phi_r, phi)
        phi_b = np.where(fluid, phi_b, phi)
        phi_t = np.where(fluid, phi_t, phi)


        phi_new = ((phi_l + phi_r) * inv_dx2 + (phi_b + phi_t) * inv_dy2) / denom

        phi = np.where(fluid, (1-omega)*phi + omega*phi_new, phi)

        phi[:, 0] = 0.0
        phi[:, -1] = U_in * Lx
        phi[0, :] = phi[1, :]
        phi[-1, :] = phi[-2, :]

    return phi

def grad_centered(phi):
    Ux = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2*dx)
    Uy = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2*dy)
    return Ux, Uy

U0 = VEL_INIT
cx0, cy0, a0 = (X_MIN + X_MAX)/2, (Y_MIN + Y_MAX)/2, 0.8
Ux0, Uy0, mask0 = analytic_cylinder_velocity(U0, cx0, cy0, a0)

Ua = bilerp(Ux0, Xa, Ya)
Va = bilerp(Uy0, Xa, Ya)
spd = np.sqrt(Ua**2 + Va**2)
L = velocity_to_length(spd)
nx = np.where(spd > 1e-9, Ua / spd, 1.0)
ny = np.where(spd > 1e-9, Va / spd, 0.0)
Uar, Var = nx * L, ny * L

solid_at_arrows = bilerp(mask0.astype(float), Xa, Ya) > 0.5
Uar = np.where(solid_at_arrows, 0.0, Uar)
Var = np.where(solid_at_arrows, 0.0, Var)

fig, ax = plt.subplots(figsize=(10, 5.2))
plt.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.30)

for i in range(GRID_COLS + 1):
    ax.axvline(X_MIN + i * cell_w, linewidth=0.5, alpha=0.15)
for j in range(GRID_ROWS + 1):
    ax.axhline(Y_MIN + j * cell_h, linewidth=0.5, alpha=0.15)

quiv = ax.quiver(Xa, Ya, Uar, Var, angles='xy', scale_units='xy', scale=1.0, width=0.003, pivot='middle')
ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Y_MIN, Y_MAX); ax.set_aspect('equal', adjustable='box')
ax.set_title("FAST Airflow Visualizer — Analytic Cylinder / Vectorized Mask")
ax.set_xlabel("x"); ax.set_ylabel("y")

mask_img = ax.imshow(mask0, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin='lower', alpha=0.15, cmap='gray_r')
ax.set_navigate(False)

slider_ax = fig.add_axes([0.12, 0.16, 0.55, 0.05])
vel_slider = Slider(slider_ax, "Velocity", VEL_MIN, VEL_MAX, valinit=VEL_INIT)

mode_ax = fig.add_axes([0.83, 0.15, 0.14, 0.12])
mode_radio = RadioButtons(mode_ax, ("Analytic Cylinder", "Numerical Mask"), active=0)

path_ax = fig.add_axes([0.12, 0.08, 0.55, 0.05])
path_box = TextBox(path_ax, "Mask PNG:", initial=os.path.join(SCRIPT_DIR, "mask_circle.png"))

load_ax = fig.add_axes([0.70, 0.08, 0.10, 0.05])
load_btn = Button(load_ax, "Load")

reset_ax = fig.add_axes([0.70, 0.16, 0.10, 0.05])
reset_btn = Button(reset_ax, "Reset")

info_txt = ax.text(0.01, 1.02, "", transform=ax.transAxes, ha='left', va='bottom')

try:
    sample = np.ones((NY, NX), dtype=np.uint8)*255
    cy, cx = NY//2, NX//2
    YI, XI = np.ogrid[:NY, :NX]
    sample[(XI-cx)**2 + (YI-cy)**2 <= (NY*0.22)**2] = 0
    Image.fromarray(np.flipud(sample)).save(os.path.join(SCRIPT_DIR, "mask_circle.png"))
except Exception:
    pass

current_mode = "Analytic Cylinder"
current_mask = mask0

def update_arrows(Ux, Uy, mask_bool):
    Ua = bilerp(Ux, Xa, Ya)
    Va = bilerp(Uy, Xa, Ya)
    spd = np.sqrt(Ua**2 + Va**2)
    L = velocity_to_length(spd)
    nx = np.where(spd > 1e-9, Ua / spd, 1.0)
    ny = np.where(spd > 1e-9, Va / spd, 0.0)
    Uar, Var = nx * L, ny * L
    solid = bilerp(mask_bool.astype(float), Xa, Ya) > 0.5
    Uar = np.where(solid, 0.0, Uar)
    Var = np.where(solid, 0.0, Var)
    quiv.set_UVC(Uar, Var)
    mask_img.set_data(mask_bool)
    fig.canvas.draw_idle()

def recompute():
    U = vel_slider.val
    if current_mode == "Analytic Cylinder":
        Ux, Uy, m = analytic_cylinder_velocity(U, cx0, cy0, a0)
        update_arrows(Ux, Uy, m)
        info_txt.set_text(f"Mode: Analytic cylinder (a={a0:.2f}); U={U:.2f}")
    else:
        m = current_mask
        phi = solve_potential_vectorized(m, U, iters=220, omega=1.7)
        Ux, Uy = grad_centered(phi)
        update_arrows(Ux, Uy, m)
        info_txt.set_text(f"Mode: Numerical mask; U={U:.2f}")

def on_vel(val):
    recompute()
vel_slider.on_changed(on_vel)

def on_mode(label):
    global current_mode
    current_mode = label
    recompute()
mode_radio.on_clicked(on_mode)

def on_load(event):
    global current_mask, current_mode
    path = path_box.text.strip()
    m = load_mask_from_png(path)
    if m is not None:
        current_mask = m
        current_mode = "Numerical Mask"
        mode_radio.set_active(1)
        recompute()
load_btn.on_clicked(on_load)

def on_reset(event):
    vel_slider.reset()
reset_btn.on_clicked(on_reset)


copied_mask = None

copy_ax = fig.add_axes([0.12, 0.00, 0.10, 0.05])
paste_ax = fig.add_axes([0.23, 0.00, 0.10, 0.05])
clear_ax = fig.add_axes([0.34, 0.00, 0.10, 0.05])
save_ax = fig.add_axes([0.45, 0.00, 0.10, 0.05])

copy_btn = Button(copy_ax, "Copy Mask")
paste_btn = Button(paste_ax, "Paste Mask")
clear_btn = Button(clear_ax, "Clear Mask")
save_btn = Button(save_ax, "Save Mask")

def on_copy(event):
    global copied_mask
    copied_mask = current_mask.copy()
    info_txt.set_text("Mask copied to memory")
    fig.canvas.draw_idle()

def on_paste(event):
    global current_mask, copied_mask
    if copied_mask is not None:
        current_mask = copied_mask.copy()
        current_mode = "Numerical Mask"
        mode_radio.set_active(1)
        recompute()
        info_txt.set_text("Mask pasted")
    else:
        info_txt.set_text("No mask copied yet")
    fig.canvas.draw_idle()

def on_clear(event):
    global current_mask
    current_mask = np.zeros_like(current_mask, dtype=bool)
    current_mode = "Numerical Mask"
    mode_radio.set_active(1)
    recompute()
    info_txt.set_text("Mask cleared (empty flow field)")
    fig.canvas.draw_idle()

def on_save(event):
    path = os.path.join(SCRIPT_DIR, "saved_mask.png")
    img = (255 * (~current_mask).astype(np.uint8)) 
    Image.fromarray(np.flipud(img)).save(path)
    info_txt.set_text(f"Saved to {path}")
    fig.canvas.draw_idle()

copy_btn.on_clicked(on_copy)
paste_btn.on_clicked(on_paste)
clear_btn.on_clicked(on_clear)
save_btn.on_clicked(on_save)


def on_key(event):
    if event.key == "a" and event.guiEvent.metaKey:
        on_clear(None)
    elif event.key == "c" and event.guiEvent.metaKey:
        on_copy(None)
    elif event.key == "v" and event.guiEvent.metaKey:
        on_paste(None)
fig.canvas.mpl_connect('key_press_event', on_key)


plt.show()

with open('/mnt/data/cfd_fast.py', 'w') as f:
    f.write(open(__file__, 'r').read())


