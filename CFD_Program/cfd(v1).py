import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


GRID_COLS, GRID_ROWS = 24, 12
X_MIN, X_MAX = 0.0, 12.0
Y_MIN, Y_MAX = 0.0, 6.0
VEL_MIN, VEL_MAX = 0.0, 10.0
VEL_INIT = 3.0


domain_w = X_MAX - X_MIN
domain_h = Y_MAX - Y_MIN
cell_w = domain_w / GRID_COLS
cell_h = domain_h / GRID_ROWS
cell_max_len = 0.8 * min(cell_w, cell_h)


x_centers = X_MIN + (np.arange(GRID_COLS) + 0.5) * cell_w
y_centers = Y_MIN + (np.arange(GRID_ROWS) + 0.5) * cell_h
Xc, Yc = np.meshgrid(x_centers, y_centers)


def velocity_to_length(v):
   k = cell_max_len / max(VEL_MAX, 1e-9)
   return np.minimum(v * k, cell_max_len)


L0 = velocity_to_length(VEL_INIT)
U = np.full_like(Xc, L0, dtype=float)
V = np.zeros_like(Yc, dtype=float)


fig, ax = plt.subplots(figsize=(9, 4.8))
plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.22)


quiv = ax.quiver(
   Xc, Yc, U, V,
   angles='xy',
   scale_units='xy',
   scale=1.0,
   width=0.002,
   pivot='middle'
)


ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_aspect('equal', adjustable='box')
ax.set_title("Cell-Based Airflow Arrows — Length capped at 80% of cell")
ax.set_xlabel("x (units)")
ax.set_ylabel("y (units)")


for i in range(GRID_COLS + 1):
   ax.axvline(X_MIN + i * cell_w, linewidth=0.5, alpha=0.2)
for j in range(GRID_ROWS + 1):
   ax.axhline(Y_MIN + j * cell_h, linewidth=0.5, alpha=0.2)


ax.set_navigate(False)


vel_text = ax.text(0.01, 1.02, f"Velocity: {VEL_INIT:.2f} (units/s); Max arrow len = {cell_max_len:.3f}",
                  transform=ax.transAxes, ha='left', va='bottom')


slider_ax = fig.add_axes([0.10, 0.10, 0.70, 0.05])
vel_slider = Slider(ax=slider_ax, label="Flow Velocity", valmin=VEL_MIN, valmax=VEL_MAX, valinit=VEL_INIT)


def on_slide(val):
   v = vel_slider.val
   L = velocity_to_length(v)
   quiv.set_UVC(np.full_like(Xc, L), np.zeros_like(Yc))
   vel_text.set_text(f"Velocity: {v:.2f} (units/s); Max arrow len = {cell_max_len:.3f}")
   fig.canvas.draw_idle()


vel_slider.on_changed(on_slide)


reset_ax = fig.add_axes([0.83, 0.10, 0.1, 0.05])
reset_btn = Button(reset_ax, "Reset")
def on_reset(event):
   vel_slider.reset()
reset_btn.on_clicked(on_reset)


plt.show()
