import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from PIL import Image
import os

#Domain Setup
NX, NY = 120, 60
X_MIN, X_MAX = 0.0, 12.0
Y_MIN, Y_MAX = 0.0, 6.0
dx = (X_MAX - X_MIN) / (NX - 1)
dy = (Y_MAX - Y_MIN) / (NY - 1)

GRID_COLS, GRID_ROWS = 24, 12
domain_w = X_MAX - X_MIN
domain_h = Y_MAX - Y_MIN
cell_w = domain_w / GRID_COLS
cell_h = domain_h / GRID_ROWS
cell_max_len = 0.8 * min(cell_w, cell_h)

VEL_MIN, VEL_MAX = 0.0, 10.0
VEL_INIT = 3.0

# Auto-detect your local folder (no /mnt/data/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Grid coordinates
x_lin = np.linspace(X_MIN, X_MAX, NX)
y_lin = np.linspace(Y_MIN, Y_MAX, NY)
Xg, Yg = np.meshgrid(x_lin, y_lin)

x_centers = X_MIN + (np.arange(GRID_COLS) + 0.5) * cell_w
y_centers = Y_MIN + (np.arange(GRID_ROWS) + 0.5) * cell_h
Xa, Ya = np.meshgrid(x_centers, y_centers)

#Mask Builders
def build_mask_circle(cx, cy, r):
    return (Xg - cx) ** 2 + (Yg - cy) ** 2 <= r ** 2

def build_mask_rect(x0, x1, y0, y1):
    return (Xg >= x0) & (Xg <= x1) & (Yg >= y0) & (Yg <= y1)

def load_mask_from_png(path):
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert("L").resize((NX, NY), Image.NEAREST)
    arr = np.array(img)
    solid = arr < 128
    solid = np.flipud(solid)
    return solid


def velocity_to_length(speed):
    k = cell_max_len / max(VEL_MAX, 1e-9)
    return np.minimum(speed * k, cell_max_len)

def solve_potential(mask_solid, U_in, iters=800):
    """Laplace solver: ∇²φ = 0 with no-through and inflow boundaries"""
    Lx = X_MAX - X_MIN
    phi = np.linspace(0.0, U_in * Lx, NX)[None, :].repeat(NY, axis=0)
    fluid = ~mask_solid
    for _ in range(iters):
        for j in range(1, NY - 1):
            for i in range(1, NX - 1):
                if not fluid[j, i]:
                    continue
                phi_l = phi[j, i - 1] if fluid[j, i - 1] else phi[j, i]
                phi_r = phi[j, i + 1] if fluid[j, i + 1] else phi[j, i]
                phi_b = phi[j - 1, i] if fluid[j - 1, i] else phi[j, i]
                phi_t = phi[j + 1, i] if fluid[j + 1, i] else phi[j, i]
                phi[j, i] = ((phi_l + phi_r) / dx**2 + (phi_b + phi_t) / dy**2) / (2/dx**2 + 2/dy**2)
        # Boundary conditions
        phi[:, 0] = 0.0
        phi[:, -1] = U_in * Lx
        phi[0, :] = phi[1, :]
        phi[-1, :] = phi[-2, :]
    return phi

def grad_centered(phi):
    Ux = np.zeros_like(phi)
    Vy = np.zeros_like(phi)
    Ux[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * dx)
    Vy[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * dy)
    return Ux, Vy

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

#Default Mask
default_mask = build_mask_circle((X_MIN + X_MAX) / 2, (Y_MIN + Y_MAX) / 2, 0.8)

#Figure Setup
fig, ax = plt.subplots(figsize=(10, 5.2))
plt.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.26)
for i in range(GRID_COLS + 1):
    ax.axvline(X_MIN + i * cell_w, linewidth=0.5, alpha=0.15)
for j in range(GRID_ROWS + 1):
    ax.axhline(Y_MIN + j * cell_h, linewidth=0.5, alpha=0.15)

phi = solve_potential(default_mask, VEL_INIT)
Ux, Vy = grad_centered(phi)
Ua = bilerp(Ux, Xa, Ya)
Va = bilerp(Vy, Xa, Ya)
spd = np.sqrt(Ua**2 + Va**2)
L = velocity_to_length(spd)
nx = np.where(spd > 1e-9, Ua / spd, 1.0)
ny = np.where(spd > 1e-9, Va / spd, 0.0)
Uar, Var = nx * L, ny * L

solid_at_arrows = bilerp(default_mask.astype(float), Xa, Ya) > 0.5
Uar = np.where(solid_at_arrows, 0.0, Uar)
Var = np.where(solid_at_arrows, 0.0, Var)

quiv = ax.quiver(Xa, Ya, Uar, Var, angles='xy', scale_units='xy', scale=1.0, width=0.003, pivot='middle')
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_aspect('equal', adjustable='box')
ax.set_title("Airflow Around Obstacles")
ax.set_xlabel("x")
ax.set_ylabel("y")

obs_img = ax.imshow(default_mask, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin='lower', alpha=0.15, cmap='gray_r')
ax.set_navigate(False)

#Widgets
slider_ax = fig.add_axes([0.12, 0.12, 0.55, 0.05])
vel_slider = Slider(ax=slider_ax, label="Inflow Velocity", valmin=VEL_MIN, valmax=VEL_MAX, valinit=VEL_INIT)
reset_ax = fig.add_axes([0.70, 0.12, 0.10, 0.05])
reset_btn = Button(reset_ax, "Reset")
radio_ax = fig.add_axes([0.83, 0.11, 0.13, 0.12])
radio = RadioButtons(radio_ax, ("Circle", "Rectangle", "Import PNG"), active=0)

path_ax = fig.add_axes([0.12, 0.05, 0.55, 0.05])
path_box = TextBox(path_ax, "Mask path:", initial=os.path.join(SCRIPT_DIR, "mask_circle.png"))
load_ax = fig.add_axes([0.70, 0.05, 0.10, 0.05])
load_btn = Button(load_ax, "Load")

vel_readout = ax.text(0.01, 1.02, "", transform=ax.transAxes, ha='left', va='bottom')

current_mask = default_mask.copy()

#Functions
def recompute(field_mask, U_in):
    phi = solve_potential(field_mask, U_in)
    Ux, Vy = grad_centered(phi)
    Ua = bilerp(Ux, Xa, Ya)
    Va = bilerp(Vy, Xa, Ya)
    spd = np.sqrt(Ua**2 + Va**2)
    L = velocity_to_length(spd)
    nx = np.where(spd > 1e-9, Ua / spd, 1.0)
    ny = np.where(spd > 1e-9, Va / spd, 0.0)
    Uar, Var = nx * L, ny * L
    solid = bilerp(field_mask.astype(float), Xa, Ya) > 0.5
    Uar = np.where(solid, 0.0, Uar)
    Var = np.where(solid, 0.0, Var)
    quiv.set_UVC(Uar, Var)
    obs_img.set_data(field_mask)
    vel_readout.set_text(f"U_in = {U_in:.2f} (units/s); arrow cap = {cell_max_len:.3f}")
    fig.canvas.draw_idle()

def on_slide(val):
    recompute(current_mask, vel_slider.val)
vel_slider.on_changed(on_slide)

def on_reset(event):
    vel_slider.reset()
reset_btn.on_clicked(on_reset)

def on_radio(label):
    global current_mask
    if label == "Circle":
        current_mask = build_mask_circle((X_MIN + X_MAX) / 2, (Y_MIN + Y_MAX) / 2, 0.8)
    elif label == "Rectangle":
        current_mask = build_mask_rect(5.0, 7.0, 2.3, 3.7)
    else:
        m = load_mask_from_png(path_box.text.strip())
        if m is not None:
            current_mask = m
    recompute(current_mask, vel_slider.val)
radio.on_clicked(on_radio)

def on_load(event):
    m = load_mask_from_png(path_box.text.strip())
    if m is not None:
        global current_mask
        current_mask = m
        radio.set_active(2)
        recompute(current_mask, vel_slider.val)
load_btn.on_clicked(on_load)

plt.show()
