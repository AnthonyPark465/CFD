import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, TextBox, CheckButtons
from PIL import Image
import os
import time
from datetime import datetime

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

MASK_DIR ="mask_circle.png" #os.path.join(SCRIPT_DIR, "mask")
os.makedirs(MASK_DIR, exist_ok=True)

# This variable will always point to the PNG that BOTH modes reference.
current_mask_path = os.path.join(MASK_DIR, "active_mask.png")

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
    solid = arr < threshold      # True = solid
    return np.flipud(solid)

def save_mask_png(mask_bool, path):
    # solid=True -> 0 (black), fluid=False -> 255 (white)
    img = (255 * (~mask_bool).astype(np.uint8))
    Image.fromarray(np.flipud(img)).save(path)

def new_versioned_mask_path():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(MASK_DIR, f"mask_{ts}.png")

def analytic_cylinder_velocity(Uinf, cx, cy, a):
    # Analytic velocity field (geometry is NOT from here; both modes use PNG geometry)
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
    return Ux, Uy

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

        # BCs
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
# Ensure an initial PNG exists
# -----------------------------
if not os.path.exists(current_mask_path):
    m0 = np.zeros((NY, NX), dtype=bool)
    cy, cx = NY//2, NX//2
    YI, XI = np.ogrid[:NY, :NX]
    m0[(XI-cx)**2 + (YI-cy)**2 <= (NY*0.22)**2] = True  # default solid circle
    save_mask_png(m0, current_mask_path)

png_mask = load_mask_from_png(current_mask_path)
if png_mask is None:
    png_mask = np.zeros((NY, NX), dtype=bool)

# -----------------------------
# Initial field/mask
# -----------------------------
U0 = VEL_INIT
UxA0, UyA0 = analytic_cylinder_velocity(U0, (X_MIN + X_MAX)/2, (Y_MIN + Y_MAX)/2, 0.8)
Ux0 = np.where(png_mask, 0.0, UxA0)
Uy0 = np.where(png_mask, 0.0, UyA0)
Uar0, Var0, spd0 = sample_arrows(Ux0, Uy0, png_mask)

# -----------------------------
# Figure and artists
# -----------------------------
fig, ax = plt.subplots(figsize=(11, 5.6))
plt.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.25)  # two rows of controls

for i in range(GRID_COLS + 1):
    ax.axvline(X_MIN + i * cell_w, linewidth=0.5, alpha=0.12)
for j in range(GRID_ROWS + 1):
    ax.axhline(Y_MIN + j * cell_h, linewidth=0.5, alpha=0.12)

quiv = ax.quiver(Xa, Ya, Uar0, Var0, angles='xy', scale_units='xy', scale=1.0,
                 width=0.003, pivot='middle')
ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Y_MIN, Y_MAX); ax.set_aspect('equal', adjustable='box')
ax.set_title("Airflow Sandbox — One PNG used by both modes")

bg_img   = ax.imshow(spd0, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin='lower', alpha=0.0)
mask_img = ax.imshow(png_mask, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin='lower', alpha=0.18)

ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_navigate(False)

# -----------------------------
# Layout helper (12-col grid)
# -----------------------------
def ui_slot(col, span=1, *, n=12, y=0.05, h=0.08, left=0.06, right=0.98, gutter=0.008):
    total_w = right - left
    col_w   = (total_w - (n - 1) * gutter) / n
    x = left + col * (col_w + gutter)
    w = col_w * span + gutter * (span - 1)
    return [x, y, w, h]

# -----------------------------
# Widgets (2-row clean layout)
# -----------------------------
# Row 1
vel_box_ax = fig.add_axes(ui_slot(0, span=3, y=0.13, h=0.08))
vel_box = TextBox(vel_box_ax, "Velocity", initial=str(VEL_INIT))

mode_ax = fig.add_axes(ui_slot(3, span=4, y=0.13, h=0.08))
mode_radio = RadioButtons(mode_ax, ("Analytic Cylinder", "Numerical Mask"), active=0)

draw_ax = fig.add_axes(ui_slot(7, span=2, y=0.13, h=0.08))
draw_check = CheckButtons(draw_ax, ["Draw"], [False])

# Row 2
checks_ax = fig.add_axes(ui_slot(0, span=4, y=0.05, h=0.08))
vis_checks = CheckButtons(checks_ax, ["Arrows", "Streamlines", "Colormap"], [True, False, False])

alpha_ax = fig.add_axes(ui_slot(4, span=3, y=0.05, h=0.08))
alpha_slider = Slider(alpha_ax, "Colormap α", 0.0, 1.0, valinit=0.5)

brush_ax = fig.add_axes(ui_slot(7, span=5, y=0.05, h=0.08))
brush_slider = Slider(brush_ax, "Brush Radius", 1, 30, valinit=6, valstep=1)

# -----------------------------
# State
# -----------------------------
info_txt = ax.text(0.01, 1.02, f"Mask: {os.path.basename(current_mask_path)}", transform=ax.transAxes,
                   ha='left', va='bottom', fontsize=9)

current_mode   = "Analytic Cylinder"
working_mask   = None
stream_obj     = None

# Particles
N_PART = 220
px = np.full(N_PART, X_MIN + 0.02)
py = np.random.uniform(Y_MIN+0.2, Y_MAX-0.2, size=N_PART)
last_update = time.time()
part_scatter = ax.scatter(px, py, s=8, alpha=0.8)

# -----------------------------
# Field helpers (both modes reference png_mask)
# -----------------------------
def get_current_velocity():
    try:
        U = float(vel_box.text)
    except Exception:
        U = VEL_INIT
    return max(VEL_MIN, min(VEL_MAX, U))

def vel_field(U):
    global png_mask
    if current_mode == "Analytic Cylinder":
        UxA, UyA = analytic_cylinder_velocity(U, (X_MIN+X_MAX)/2, (Y_MIN+Y_MAX)/2, 0.8)
        Ux = np.where(png_mask, 0.0, UxA)
        Uy = np.where(png_mask, 0.0, UyA)
        m  = png_mask
    else:
        m = png_mask
        phi = solve_potential_vectorized(m, U, iters=200, omega=1.7)
        Ux, Uy = grad_centered(phi)
        Ux = np.where(m, 0.0, Ux)
        Uy = np.where(m, 0.0, Uy)

    spd = np.sqrt(Ux**2 + Uy**2)
    return Ux, Uy, m, spd

def recompute():
    global stream_obj
    U = get_current_velocity()
    Ux, Uy, m, spd = vel_field(U)

    Uar, Var, _ = sample_arrows(Ux, Uy, m)
    quiv.set_UVC(Uar, Var)
    quiv.set_visible(vis_checks.get_status()[0])

    bg_img.set_data(spd)
    bg_img.set_alpha(alpha_slider.val if vis_checks.get_status()[2] else 0.0)

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

    mask_img.set_data(m.astype(float))
    fig.canvas.draw_idle()

def restart_sim():
    global px, py, last_update
    px[:] = X_MIN + 0.02
    py[:] = np.random.uniform(Y_MIN+0.2, Y_MAX-0.2, size=py.shape[0])
    last_update = time.time()
    recompute()

# -----------------------------
# UI callbacks
# -----------------------------
def on_mode(label):
    global current_mode
    current_mode = label
    recompute()
mode_radio.on_clicked(on_mode)

def on_alpha(_):
    recompute()
alpha_slider.on_changed(on_alpha)

def on_vis_checks(_):
    quiv.set_visible(vis_checks.get_status()[0])
    recompute()
vis_checks.on_clicked(on_vis_checks)

def on_vel_submit(text):
    try:
        U = float(text)
    except Exception:
        U = VEL_INIT
    vel_box.set_val(str(max(VEL_MIN, min(VEL_MAX, U))))
    recompute()
vel_box.on_submit(on_vel_submit)

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

def timer_callback(_):
    global last_update
    now = time.time()
    dt = min(0.05, now - last_update)
    last_update = now
    U = get_current_velocity()
    Ux, Uy, m, _ = vel_field(U)
    step_particles(dt, Ux, Uy, m)
    part_scatter.set_offsets(np.column_stack([px, py]))
    fig.canvas.draw_idle()

timer = fig.canvas.new_timer(interval=30)
timer.add_callback(timer_callback, None)
timer.start()

# -----------------------------
# Drawing: create a NEW PNG when finishing
# -----------------------------
is_drawing = False
erase_mode = False
shift_held = False
pending_draw_changes = False
working_mask = None

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
    sub[disk] = value

def begin_drawing():
    global working_mask
    working_mask = png_mask.copy()
    mask_img.set_data(working_mask.astype(float))
    fig.canvas.draw_idle()

def finish_drawing_create_new_png():
    """Create a brand-new PNG from edits and make BOTH modes reference it."""
    global working_mask, png_mask, current_mask_path
    if working_mask is None:
        return
    # 1) Create a new unique PNG file
    new_path = new_versioned_mask_path()
    save_mask_png(working_mask, new_path)
    # 2) Update the 'active' symlink/file so future runs also use latest
    try:
        # Overwrite active_mask.png with the new content
        save_mask_png(working_mask, current_mask_path)
    except Exception:
        pass
    # 3) Update in-memory mask to the new image
    loaded = load_mask_from_png(new_path)
    png_mask = loaded if loaded is not None else working_mask.copy()
    working_mask = None
    # 4) Update UI label and restart
    info_txt.set_text(f"Mask: {os.path.basename(new_path)}")
    restart_sim()

def on_draw_toggled(_):
    global pending_draw_changes
    enabled = draw_check.get_status()[0]
    ax.set_navigate(not enabled)
    fig.canvas.draw_idle()
    if enabled:
        pending_draw_changes = False
        begin_drawing()
    else:
        if pending_draw_changes:
            finish_drawing_create_new_png()
            pending_draw_changes = False
        else:
            recompute()
draw_check.on_clicked(on_draw_toggled)

def on_mouse_press(event):
    global is_drawing, erase_mode, shift_held, pending_draw_changes
    if not draw_check.get_status()[0]:
        return
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return
    if working_mask is None:
        begin_drawing()
    erase_mode = (event.button == 3) or shift_held
    r = int(brush_slider.val)
    i, j = world_to_ij(event.xdata, event.ydata)
    paint_circle(working_mask, i, j, r, value=not erase_mode)
    pending_draw_changes = True
    mask_img.set_data(working_mask.astype(float))
    fig.canvas.draw_idle()
    is_drawing = True

def on_mouse_move(event):
    global pending_draw_changes
    if not is_drawing or working_mask is None:
        return
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return
    r = int(brush_slider.val)
    i, j = world_to_ij(event.xdata, event.ydata)
    paint_circle(working_mask, i, j, r, value=not erase_mode)
    pending_draw_changes = True
    mask_img.set_data(working_mask.astype(float))
    fig.canvas.draw_idle()

def on_mouse_release(event):
    global is_drawing
    if not is_drawing:
        return
    is_drawing = False
    if working_mask is not None:
        mask_img.set_data(working_mask.astype(float))
        fig.canvas.draw_idle()

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

plt.show()
