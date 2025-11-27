import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox, CheckButtons
from PIL import Image
import os
import time

# -----------------------------
# Grid/domain configuration
# -----------------------------
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

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

# -----------------------------
# Coordinate arrays
# -----------------------------
x_lin = np.linspace(X_MIN, X_MAX, NX)
y_lin = np.linspace(Y_MIN, Y_MAX, NY)
Xg, Yg = np.meshgrid(x_lin, y_lin)

x_centers = X_MIN + (np.arange(GRID_COLS) + 0.5) * cell_w
y_centers = Y_MIN + (np.arange(GRID_ROWS) + 0.5) * cell_h
Xa, Ya = np.meshgrid(x_centers, y_centers)

# -----------------------------
# Utilities
# -----------------------------
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

def load_mask_from_png(path, threshold=128):
    if not os.path.exists(path):
        return None
    arr = np.array(Image.open(path).convert("L").resize((NX, NY), Image.NEAREST))
    solid = arr < threshold
    return np.flipud(solid)

def fill_internal_holes(mask_solid):
    fluid = ~mask_solid
    reachable = np.zeros_like(fluid, dtype=bool)

    reachable[0, :] |= fluid[0, :]
    reachable[-1, :] |= fluid[-1, :]
    reachable[:, 0] |= fluid[:, 0]
    reachable[:, -1] |= fluid[:, -1]

    changed = True
    while changed:
        nb = (np.roll(reachable, 1, axis=0) |
              np.roll(reachable, -1, axis=0) |
              np.roll(reachable, 1, axis=1) |
              np.roll(reachable, -1, axis=1))
        new_reach = nb & fluid
        new = new_reach | reachable
        changed = np.any(new != reachable)
        reachable = new

        reachable[0, :] |= fluid[0, :]
        reachable[-1, :] |= fluid[-1, :]
        reachable[:, 0] |= fluid[:, 0]
        reachable[:, -1] |= fluid[:, -1]

    fixed_mask = ~(reachable)
    return fixed_mask

def analytic_cylinder_velocity(Uinf, cx, cy, a):
    X = Xg - cx
    Y = Yg - cy
    r2 = X*X + Y*Y
    theta = np.arctan2(Y, X)
    eps = 1e-9
    inv_r2 = 1.0 / np.maximum(r2, eps)
    ur  = Uinf * (1.0 - (a*a) * inv_r2) * np.cos(theta)
    uth = -Uinf * (1.0 + (a*a) * inv_r2) * np.sin(theta)
    Ux  = ur * np.cos(theta) - uth * np.sin(theta)
    Uy  = ur * np.sin(theta) + uth * np.cos(theta)
    inside = r2 <= a*a
    Ux = np.where(inside, 0.0, Ux)
    Uy = np.where(inside, 0.0, Uy)
    mask = inside
    return Ux, Uy, mask

def solve_potential_vectorized(mask_solid, U_in, iters=220, omega=1.7):
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

        phi[:, 0]  = 0.0
        phi[:, -1] = U_in * Lx
        phi[0, :]  = phi[1, :]
        phi[-1, :] = phi[-2, :]

    return phi

def grad_centered(phi):
    Ux = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2*dx)
    Uy = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2*dy)
    return Ux, Uy

def sample_arrows(Ux, Uy, mask_bool):
    Ua  = bilerp(Ux, Xa, Ya)
    Va  = bilerp(Uy, Xa, Ya)
    spd = np.sqrt(Ua**2 + Va**2)
    L   = velocity_to_length(spd)
    nx  = np.where(spd > 1e-9, Ua / spd, 1.0)
    ny  = np.where(spd > 1e-9, Va / spd, 0.0)
    Uar, Var = nx * L, ny * L
    solid = bilerp(mask_bool.astype(float), Xa, Ya) > 0.5
    Uar = np.where(solid, 0.0, Uar)
    Var = np.where(solid, 0.0, Var)
    return Uar, Var, spd

# -----------------------------
# Initial field/mask
# -----------------------------
U0 = VEL_INIT
cx0, cy0, a0 = (X_MIN + X_MAX)/2, (Y_MIN + Y_MAX)/2, 0.8
Ux0, Uy0, mask0 = analytic_cylinder_velocity(U0, cx0, cy0, a0)
Uar0, Var0, spd0 = sample_arrows(Ux0, Uy0, mask0)

# -----------------------------
# Figure and artists
# -----------------------------
fig, ax = plt.subplots(figsize=(11, 5.6))
plt.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.34)

for i in range(GRID_COLS + 1):
    ax.axvline(X_MIN + i * cell_w, linewidth=0.5, alpha=0.12)
for j in range(GRID_ROWS + 1):
    ax.axhline(Y_MIN + j * cell_h, linewidth=0.5, alpha=0.12)

quiv = ax.quiver(Xa, Ya, Uar0, Var0, angles='xy', scale_units='xy', scale=1.0,
                 width=0.003, pivot='middle')
ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Y_MIN, Y_MAX); ax.set_aspect('equal', adjustable='box')
ax.set_title("Airflow Sandbox — Arrows - Streamlines - Colormap - Tracers")

bg_img   = ax.imshow(spd0, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin='lower', alpha=0.0)
mask_img = ax.imshow(mask0, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin='lower', alpha=0.15)

ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_navigate(False)

# -----------------------------
# Widgets
# -----------------------------
slider_ax = fig.add_axes([0.12, 0.22, 0.55, 0.04])
vel_slider = Slider(slider_ax, "Velocity", VEL_MIN, VEL_MAX, valinit=VEL_INIT)

mode_ax = fig.add_axes([0.83, 0.21, 0.14, 0.10])
mode_radio = RadioButtons(mode_ax, ("Analytic Cylinder", "Numerical Mask"), active=0)

path_ax = fig.add_axes([0.12, 0.16, 0.55, 0.04])
path_box = TextBox(path_ax, "Mask PNG:", initial=os.path.join(SCRIPT_DIR, "mask_circle.png"))

th_ax = fig.add_axes([0.12, 0.11, 0.20, 0.04])
th_slider = Slider(th_ax, "Threshold", 1, 254, valinit=128, valstep=1)

load_ax = fig.add_axes([0.46, 0.11, 0.08, 0.05])
load_btn = Button(load_ax, "Load")

holes_ax = fig.add_axes([0.55, 0.11, 0.12, 0.05])
holes_btn = Button(holes_ax, "Fix Holes")

copy_ax  = fig.add_axes([0.12, 0.03, 0.10, 0.05])
paste_ax = fig.add_axes([0.23, 0.03, 0.10, 0.05])
clear_ax = fig.add_axes([0.34, 0.03, 0.10, 0.05])
save_ax  = fig.add_axes([0.45, 0.03, 0.10, 0.05])

copy_btn  = Button(copy_ax,  "Copy Mask")
paste_btn = Button(paste_ax, "Paste Mask")
clear_btn = Button(clear_ax, "Clear Mask")
save_btn  = Button(save_ax,  "Save Mask")

checks_ax  = fig.add_axes([0.83, 0.07, 0.14, 0.14])
vis_checks = CheckButtons(checks_ax, ["Show Arrows", "Show Streamlines", "Show Colormap"], [True, False, False])

alpha_ax = fig.add_axes([0.67, 0.03, 0.15, 0.05])
alpha_slider = Slider(alpha_ax, "Colormap α", 0.0, 1.0, valinit=0.5)

# --- Draw UI (NEW) ---
draw_ax = fig.add_axes([0.67, 0.11, 0.10, 0.05])
draw_check = CheckButtons(draw_ax, ["Draw Mode"], [False])

brush_ax = fig.add_axes([0.78, 0.11, 0.19, 0.04])
brush_slider = Slider(brush_ax, "Brush Radius", 1, 30, valinit=6, valstep=1)

# -----------------------------
# Text and state
# -----------------------------
info_txt = ax.text(0.50, 1.02, "", transform=ax.transAxes, ha='left', va='bottom', fontsize=9)

current_mode = "Analytic Cylinder"
current_mask = mask0.copy()
copied_mask  = None
stream_obj   = None

# Particles
N_PART = 220
px = np.full(N_PART, X_MIN + 0.02)
py = np.random.uniform(Y_MIN+0.2, Y_MAX-0.2, size=N_PART)
last_update = time.time()
part_scatter = ax.scatter(px, py, s=8, alpha=0.8)

# -----------------------------
# Field helpers
# -----------------------------
def vel_field(U):
    if current_mode == "Analytic Cylinder":
        Ux, Uy, m = analytic_cylinder_velocity(U, (X_MIN+X_MAX)/2, (Y_MIN+Y_MAX)/2, 0.8)
    else:
        m = current_mask
        phi = solve_potential_vectorized(m, U, iters=200, omega=1.7)
        Ux, Uy = grad_centered(phi)
    spd = np.sqrt(Ux**2 + Uy**2)
    return Ux, Uy, m, spd

def recompute():
    global stream_obj
    U = vel_slider.val
    Ux, Uy, m, spd = vel_field(U)

    # Arrows
    Uar, Var, _ = sample_arrows(Ux, Uy, m)
    quiv.set_UVC(Uar, Var)
    quiv.set_visible(vis_checks.get_status()[0])

    # Colormap
    bg_img.set_data(spd)
    bg_img.set_alpha(alpha_slider.val if vis_checks.get_status()[2] else 0.0)

    # Streamlines: remove old safely, then redraw if enabled
    if stream_obj is not None:
        try:
            if getattr(stream_obj, "lines", None) is not None:
                stream_obj.lines.remove()
            if getattr(stream_obj, "arrows", None) is not None:
                stream_obj.arrows.remove()
        except Exception:
            pass
        stream_obj = None

    if vis_checks.get_status()[1]:
        stream_obj = ax.streamplot(x_lin, y_lin, Ux, Uy, density=1.4)

    # Mask overlay
    mask_img.set_data(m.astype(float))

    fig.canvas.draw_idle()

def restart_sim():
    """Reset particles and recompute flow with the current mask."""
    global px, py, last_update
    px[:] = X_MIN + 0.02
    py[:] = np.random.uniform(Y_MIN+0.2, Y_MAX-0.2, size=py.shape[0])
    last_update = time.time()
    recompute()

# -----------------------------
# UI callbacks
# -----------------------------
def on_vel(_):
    recompute()
vel_slider.on_changed(on_vel)

def on_mode(label):
    global current_mode
    current_mode = label
    recompute()
mode_radio.on_clicked(on_mode)

def on_load(_):
    global current_mask, current_mode
    m = load_mask_from_png(path_box.text.strip(), int(th_slider.val))
    if m is not None:
        current_mask = m
        current_mode = "Numerical Mask"
        mode_radio.set_active(1)
        restart_sim()
load_btn.on_clicked(on_load)

def on_fix_holes(_):
    global current_mask, current_mode
    current_mask = fill_internal_holes(current_mask)
    current_mode = "Numerical Mask"; mode_radio.set_active(1)
    restart_sim()
holes_btn.on_clicked(on_fix_holes)

def on_alpha(_):
    recompute()
alpha_slider.on_changed(on_alpha)

def on_vis_checks(_):
    quiv.set_visible(vis_checks.get_status()[0])
    recompute()
vis_checks.on_clicked(on_vis_checks)

def on_copy(_):
    global copied_mask
    copied_mask = current_mask.copy()
    info_txt.set_text("Mask copied")
    fig.canvas.draw_idle()

def on_paste(_):
    global current_mask, current_mode
    if copied_mask is not None:
        current_mask = copied_mask.copy()
        current_mode = "Numerical Mask"; mode_radio.set_active(1)
        restart_sim()

def on_clear(_):
    global current_mask, current_mode
    current_mask = np.zeros_like(current_mask, dtype=bool)
    current_mode = "Numerical Mask"; mode_radio.set_active(1)
    restart_sim()

def on_save(_):
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
    try:
        meta = bool(event.guiEvent.metaKey)
    except Exception:
        meta = False
    if meta and event.key == "c":
        on_copy(None)
    elif meta and event.key == "v":
        on_paste(None)
    elif meta and event.key == "a":
        on_clear(None)
fig.canvas.mpl_connect('key_press_event', on_key)

# -----------------------------
# Particles
# -----------------------------
def step_particles(dt, Ux, Uy, mask_bool):
    global px, py
    Upx = bilerp(Ux, px, py)
    Upy = bilerp(Uy, px, py)
    px = px + Upx * dt * 0.5
    py = py + Upy * dt * 0.5
    out = (px < X_MIN) | (px > X_MAX) | (py < Y_MIN) | (py > Y_MAX) | (bilerp(mask_bool.astype(float), px, py) > 0.5)
    if np.any(out):
        px[out] = X_MIN + 0.02
        py[out] = np.random.uniform(Y_MIN+0.2, Y_MAX-0.2, size=out.sum())

def timer_callback(event):
    global last_update
    now = time.time()
    dt = min(0.05, now - last_update)
    last_update = now
    U = vel_slider.val
    Ux, Uy, m, _ = vel_field(U)
    step_particles(dt, Ux, Uy, m)
    part_scatter.set_offsets(np.column_stack([px, py]))
    fig.canvas.draw_idle()

timer = fig.canvas.new_timer(interval=30)
timer.add_callback(timer_callback, None)
timer.start()

# -----------------------------
# Drawing tools + "restart on toggle-off"
# -----------------------------
is_drawing = False
erase_mode = False
shift_held = False
pending_draw_changes = False   # set true whenever user paints/erases

def world_to_ij(x, y):
    i = int(round((x - X_MIN) / dx))
    j = int(round((y - Y_MIN) / dy))
    i = np.clip(i, 0, NX - 1)
    j = np.clip(j, 0, NY - 1)
    return i, j

def paint_circle(mask, i_c, j_c, r_pix, value):
    if r_pix <= 0:
        return
    i0 = max(i_c - r_pix, 0); i1 = min(i_c + r_pix, NX - 1)
    j0 = max(j_c - r_pix, 0); j1 = min(j_c + r_pix, NY - 1)
    ii = np.arange(i0, i1 + 1)
    jj = np.arange(j0, j1 + 1)
    II, JJ = np.meshgrid(ii, jj)
    disk = (II - i_c)**2 + (JJ - j_c)**2 <= r_pix**2
    sub = mask[j0:j1 + 1, i0:i1 + 1]
    if value:
        sub[disk] = True
    else:
        sub[disk] = False

def set_numerical_mode():
    global current_mode
    if current_mode != "Numerical Mask":
        current_mode = "Numerical Mask"
        mode_radio.set_active(1)

def refresh_mask_only():
    mask_img.set_data(current_mask.astype(float))
    fig.canvas.draw_idle()

def on_draw_toggled(_):
    global pending_draw_changes
    enabled = draw_check.get_status()[0]
    ax.set_navigate(not enabled)
    fig.canvas.draw_idle()
    if not enabled and pending_draw_changes:
        set_numerical_mode()
        restart_sim()
        pending_draw_changes = False
draw_check.on_clicked(on_draw_toggled)

def on_mouse_press(event):
    global is_drawing, erase_mode, shift_held, pending_draw_changes
    if not draw_check.get_status()[0]:
        return
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return
    erase_mode = (event.button == 3) or shift_held
    r = int(brush_slider.val)
    i, j = world_to_ij(event.xdata, event.ydata)
    paint_circle(current_mask, i, j, r, value=not erase_mode)
    pending_draw_changes = True
    refresh_mask_only()
    is_drawing = True

def on_mouse_move(event):
    global pending_draw_changes
    if not is_drawing:
        return
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return
    r = int(brush_slider.val)
    i, j = world_to_ij(event.xdata, event.ydata)
    paint_circle(current_mask, i, j, r, value=not erase_mode)
    pending_draw_changes = True
    refresh_mask_only()

def on_mouse_release(event):
    global is_drawing
    if not is_drawing:
        return
    is_drawing = False
    # Optional: immediate recompute on stroke end; keep for snappier feedback
    set_numerical_mode()
    recompute()

def on_key_press_draw(event):
    global shift_held, erase_mode
    if event.key in ("shift", "shiftleft", "shiftright"):
        shift_held = True
        if is_drawing:
            erase_mode = True

def on_key_release_draw(event):
    global shift_held, erase_mode
    if event.key in ("shift", "shiftleft", "shiftright"):
        shift_held = False
        if is_drawing:
            erase_mode = False

fig.canvas.mpl_connect('button_press_event',   on_mouse_press)
fig.canvas.mpl_connect('motion_notify_event',  on_mouse_move)
fig.canvas.mpl_connect('button_release_event', on_mouse_release)
fig.canvas.mpl_connect('key_press_event',      on_key_press_draw)
fig.canvas.mpl_connect('key_release_event',    on_key_release_draw)


try:
    sample = np.ones((NY, NX), dtype=np.uint8)*255
    cy, cx = NY//2, NX//2
    YI, XI = np.ogrid[:NY, :NX]
    sample[(XI-cx)**2 + (YI-cy)**2 <= (NY*0.22)**2] = 0
    Image.fromarray(np.flipud(sample)).save(os.path.join(SCRIPT_DIR, "mask_circle.png"))
except Exception:
    pass

plt.show()
