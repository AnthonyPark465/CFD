
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox, CheckButtons
from PIL import Image
import os, time, hashlib


NX, NY = 160, 80
X_MIN, X_MAX = 0.0, 12.0
Y_MIN, Y_MAX = 0.0, 6.0
dx = (X_MAX - X_MIN) / (NX - 1)
dy = (Y_MAX - Y_MIN) / (NY - 1)


ARROW_STEP_X = 6 
ARROW_STEP_Y = 6
ARROW_OFFSET_X = 3 
ARROW_OFFSET_Y = 3

cell_w = dx * ARROW_STEP_X
cell_h = dy * ARROW_STEP_Y
ARROW_CAP = 0.8 * min(cell_w, cell_h)

VEL_MIN, VEL_MAX, VEL_INIT = 0.0, 12.0, 4.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
x_lin = np.linspace(X_MIN, X_MAX, NX)
y_lin = np.linspace(Y_MIN, Y_MAX, NY)

ix = np.arange(ARROW_OFFSET_X, NX, ARROW_STEP_X)
iy = np.arange(ARROW_OFFSET_Y, NY, ARROW_STEP_Y)
Xi, Yi = np.meshgrid(ix, iy)
Xa, Ya = x_lin[Xi], y_lin[Yi]

def mask_from_png(path, threshold=128, invert=False):
    if not os.path.exists(path):
        return np.zeros((NY, NX), dtype=bool)
    A = np.array(Image.open(path).convert("L").resize((NX, NY), Image.NEAREST))
    solid = (A < threshold)
    if invert: solid = ~solid
    return np.flipud(solid)

def mask_hash(mask):
    return hashlib.sha1(mask.view(np.uint8)).hexdigest()

def analytic_cylinder(U, cx, cy, a):
    X, Y = np.meshgrid(x_lin, y_lin)
    X = X - cx; Y = Y - cy
    r2 = X*X + Y*Y
    th = np.arctan2(Y, X)
    eps = 1e-9
    invr2 = 1.0/np.maximum(r2, eps)
    ur = U * (1 - a*a*invr2) * np.cos(th)
    uth = -U * (1 + a*a*invr2) * np.sin(th)
    Ux = ur*np.cos(th) - uth*np.sin(th)
    Uy = ur*np.sin(th) + uth*np.cos(th)
    inside = r2 <= a*a
    Ux[inside] = 0.0; Uy[inside] = 0.0
    return Ux, Uy, inside


_phi_cache = None
_phi_mask_sig = None
_phi_U = None

def solve_phi(mask, U, iters_small=40, iters_full=220, omega=1.7):
    global _phi_cache, _phi_mask_sig, _phi_U
    Lx = X_MAX - X_MIN
    fluid = ~mask
    sig = mask_hash(mask)
    if _phi_cache is not None and sig == _phi_mask_sig:
        phi = _phi_cache.copy()

        iters = iters_small
    else:

        phi = np.linspace(0.0, U*Lx, NX)[None, :].repeat(NY, axis=0)
        iters = iters_full
        _phi_mask_sig = sig

    inv_dx2 = 1.0/dx**2; inv_dy2 = 1.0/dy**2; denom = 2*inv_dx2 + 2*inv_dy2
    for _ in range(iters):
        l = np.roll(phi, 1, 1); r = np.roll(phi, -1, 1)
        b = np.roll(phi, 1, 0); t = np.roll(phi, -1, 0)
        l = np.where(fluid, l, phi); r = np.where(fluid, r, phi)
        b = np.where(fluid, b, phi); t = np.where(fluid, t, phi)
        phi_new = ((l+r)*inv_dx2 + (b+t)*inv_dy2)/denom
        phi = np.where(fluid, (1-omega)*phi + omega*phi_new, phi)
        phi[:, 0] = 0.0; phi[:, -1] = U*Lx
        phi[0, :] = phi[1, :]; phi[-1, :] = phi[-2, :]
    _phi_cache = phi.copy(); _phi_U = U
    return phi

def grad_centered(phi):
    Ux = (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / (2*dx)
    Uy = (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) / (2*dy)
    return Ux, Uy


def arrow_vectors(Ux, Uy, mask):
    Ua = Uy[::ARROW_STEP_Y, ::ARROW_STEP_X]

    Ua = Ux[iy][:, ix]
    Va = Uy[iy][:, ix]
    spd = np.sqrt(Ua*Ua + Va*Va) + 1e-12
    scale = np.minimum(spd*(ARROW_CAP/VEL_MAX), ARROW_CAP)/spd
    Ua = Ua*scale; Va = Va*scale
    solid = mask[iy][:, ix]
    Ua[solid] = 0.0; Va[solid] = 0.0
    return Ua, Va


N_PART = 180
px = np.full(N_PART, X_MIN + 0.02)
py = np.random.uniform(Y_MIN+0.2, Y_MAX-0.2, size=N_PART)

def sample_nn(field, x, y):
    i = np.clip(((x - X_MIN)/dx).astype(int), 0, NX-1)
    j = np.clip(((y - Y_MIN)/dy).astype(int), 0, NY-1)
    return field[j, i]

def step_particles(dt, Ux, Uy, mask):
    global px, py
    u = sample_nn(Ux, px, py)
    v = sample_nn(Uy, px, py)
    px += u*dt*0.5; py += v*dt*0.5
    i = np.clip(((px - X_MIN)/dx).astype(int), 0, NX-1)
    j = np.clip(((py - Y_MIN)/dy).astype(int), 0, NY-1)
    hit = (px<X_MIN)|(px>X_MAX)|(py<Y_MIN)|(py>Y_MAX)| (mask[j, i])
    if np.any(hit):
        px[hit] = X_MIN + 0.02
        py[hit] = np.random.uniform(Y_MIN+0.2, Y_MAX-0.2, size=hit.sum())

U = VEL_INIT
Ux0, Uy0, mask0 = analytic_cylinder(U, (X_MIN+X_MAX)/2, (Y_MIN+Y_MAX)/2, 0.8)
Ua0, Va0 = arrow_vectors(Ux0, Uy0, mask0)

fig, ax = plt.subplots(figsize=(10.5, 5.2))
plt.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.30)
quiv = ax.quiver(Xa, Ya, Ua0, Va0, angles='xy', scale_units='xy', scale=1.0, width=0.003, pivot='middle')
mask_img = ax.imshow(mask0, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin='lower', alpha=0.15, cmap='gray_r')
ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Y_MIN, Y_MAX); ax.set_aspect('equal')
ax.set_title("CFD Lite — fast arrows & tracers (strided, NN sampling)")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_navigate(False)

vel_ax = fig.add_axes([0.12, 0.20, 0.55, 0.05])
vel_slider = Slider(vel_ax, "Velocity", VEL_MIN, VEL_MAX, valinit=VEL_INIT)

mode_ax = fig.add_axes([0.70, 0.20, 0.22, 0.10])
mode_radio = RadioButtons(mode_ax, ("Analytic Cylinder", "Numerical Mask"), active=0)

path_ax = fig.add_axes([0.12, 0.12, 0.55, 0.05])
path_box = TextBox(path_ax, "Mask PNG:", initial=os.path.join(SCRIPT_DIR, "mask_circle.png"))
th_ax = fig.add_axes([0.12, 0.06, 0.25, 0.05])
th_slider = Slider(th_ax, "Threshold", 1, 254, valinit=128, valstep=1)
inv_ax = fig.add_axes([0.39, 0.06, 0.10, 0.05])
inv_btn = CheckButtons(inv_ax, ["Invert"], [False])
load_ax = fig.add_axes([0.51, 0.06, 0.08, 0.05]); load_btn = Button(load_ax, "Load")
clear_ax = fig.add_axes([0.61, 0.06, 0.08, 0.05]); clear_btn = Button(clear_ax, "Clear")

info_txt = ax.text(0.01, 1.02, "", transform=ax.transAxes, ha='left', va='bottom', fontsize=9)

sc = ax.scatter(px, py, s=8, alpha=0.75)
last = time.time()
current_mode = "Analytic Cylinder"
current_mask = mask0.copy()

def field_from_mode(U):
    if current_mode == "Analytic Cylinder":
        return analytic_cylinder(U, (X_MIN+X_MAX)/2, (Y_MIN+Y_MAX)/2, 0.8)
    else:
        phi = solve_phi(current_mask, U, iters_small=30, iters_full=160, omega=1.8)
        Ux, Uy = grad_centered(phi)
        return Ux, Uy, current_mask

def recompute():
    U = vel_slider.val
    Ux, Uy, m = field_from_mode(U)
    Ua, Va = arrow_vectors(Ux, Uy, m)
    quiv.set_UVC(Ua, Va)
    mask_img.set_data(m)
    info_txt.set_text(f"Mode: {current_mode} | U={U:.2f} | Arrows={Ua.size}")
    fig.canvas.draw_idle()

def on_vel(_): recompute()
vel_slider.on_changed(on_vel)

def on_mode(label):
    global current_mode
    current_mode = label
    recompute()
mode_radio.on_clicked(on_mode)

def on_load(_):
    global current_mask, current_mode
    current_mask = mask_from_png(path_box.text.strip(), int(th_slider.val), inv_btn.get_status()[0])
    current_mode = "Numerical Mask"; mode_radio.set_active(1)
    global _phi_cache, _phi_mask_sig
    _phi_cache, _phi_mask_sig = None, None
    recompute()
load_btn.on_clicked(on_load)

def on_clear(_):
    global current_mask, _phi_cache, _phi_mask_sig
    current_mask[:] = False
    _phi_cache, _phi_mask_sig = None, None
    recompute()
clear_btn.on_clicked(on_clear)

def timer_cb(_evt):
    global last
    now = time.time(); dt = min(0.05, now-last); last = now
    Ux, Uy, m = field_from_mode(vel_slider.val)
    step_particles(dt, Ux, Uy, m)
    sc.set_offsets(np.column_stack([px, py]))
    fig.canvas.draw_idle()

timer = fig.canvas.new_timer(interval=30)
timer.add_callback(timer_cb, None)
timer.start()

try:
    A = np.ones((NY, NX), dtype=np.uint8)*255
    cy, cx = NY//2, NX//2
    YI, XI = np.ogrid[:NY, :NX]
    A[(XI-cx)**2 + (YI-cy)**2 <= (NY*0.22)**2] = 0
    Image.fromarray(np.flipud(A)).save(os.path.join(SCRIPT_DIR, "mask_circle.png"))
except Exception:
    pass

plt.show()
