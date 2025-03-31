import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import itertools
# import functools # No longer needed after removing calculate_sums decorator use
import math

# --- Helper Function Removed (calculate_sums) ---
# We will calculate sums on the fly

class InteractiveDiceVisualizer:
    def __init__(self, max_dice=8, sides=6, initial_dice=1):
        if not isinstance(initial_dice, int) or initial_dice < 1 or initial_dice > max_dice:
            print(f"Warning: initial_dice ({initial_dice}) invalid. Setting to 1.")
            initial_dice = 1

        self.max_dice = max_dice
        self.sides = sides
        self.fixed_dims = {}
        self.slice_sliders = {}
        # --- Removed self.outcome_sums ---
        self.num_dice = 0
        self.target_sums_set = set()
        self.min_sum = 1
        self.max_sum = 1

        # --- Colormap ---
        cmap_colors = [(1.0, 0.0, 0.0), (1.0, 0.5, 0.0), (0.1, 0.8, 0.1)]
        self.prob_cmap = LinearSegmentedColormap.from_list("ProbCmap", cmap_colors, N=256)

        # --- Base Colors ---
        self.base_color = (0.9, 0.9, 0.9, 0.15) # Keep some transparency
        self.bar_edge_color = None
        self.bar_edge_linewidth = 0
        self.prob_bar_bg_color = '#444444'
        self.target_selector_bg = '#333333'
        self.target_selector_selected_color = 'lightcoral'
        self.target_selector_tick_color = 'white'
        # Option: Slightly faster rendering, potentially flatter look
        self.use_shade = True # Set to False to potentially speed up rendering

        self.fig = plt.figure(figsize=(10, 9))
        self.fig.patch.set_facecolor('black')

        self.main_ax_rect = [0.05, 0.40, 0.9, 0.55]
        self.ax_main = None

        # --- UI Element Setup ---
        axcolor = '#333333'
        slider_text_color = 'white'
        element_h = 0.03
        slider_spacing = 0.01
        bottom_start = 0.28

        # Num Dice Slider
        self.ax_num_dice = self.fig.add_axes([0.15, bottom_start, 0.65, element_h], facecolor=axcolor)
        self.slider_num_dice = Slider(self.ax_num_dice, 'Num Dice', 1, self.max_dice, valinit=initial_dice, valstep=1, color='skyblue')
        self.slider_num_dice.label.set_color(slider_text_color)
        self.slider_num_dice.valtext.set_color(slider_text_color)

        # Target Sum Selector Axes
        target_sel_y = bottom_start - (element_h + slider_spacing)
        self.ax_target_selector = self.fig.add_axes([0.15, target_sel_y, 0.65, element_h], facecolor=self.target_selector_bg)
        self.ax_target_selector.set_yticks([])
        self.ax_target_selector.tick_params(axis='x', colors=self.target_selector_tick_color)
        self.ax_target_selector.set_title("Target Sum(s) - Click to Select/Deselect", color=slider_text_color, fontsize=9, pad=3)
        self.target_selector_cid = self.fig.canvas.mpl_connect('button_press_event', self._on_target_select)

        # Slice Slider setup
        self.slice_slider_axes = []
        self.slice_slider_text_color = slider_text_color
        max_slice_sliders = self.max_dice - 3
        current_slider_y = target_sel_y - (element_h + slider_spacing)
        for i in range(max_slice_sliders):
            y_pos = current_slider_y - i * (element_h + slider_spacing)
            ax = self.fig.add_axes([0.15, y_pos, 0.65, element_h], facecolor=axcolor, visible=False)
            self.slice_slider_axes.append(ax)

        # Probability Bar Axes
        prob_bar_y = current_slider_y - max_slice_sliders * (element_h + slider_spacing) - (element_h + slider_spacing*1.5)
        self.ax_prob_bar = self.fig.add_axes([0.15, prob_bar_y, 0.65, element_h], facecolor='black')
        self.ax_prob_bar.set_xticks([])
        self.ax_prob_bar.set_yticks([])
        self.ax_prob_bar.set_xlim(0, 1); self.ax_prob_bar.set_ylim(0, 1)
        self.prob_bar_title = self.ax_prob_bar.set_title("Likelihood", color=slider_text_color, fontsize=10, pad=5)

        # --- Initial Update ---
        self._update_num_dice(self.slider_num_dice.val)
        self.slider_num_dice.on_changed(self._update_num_dice)

        plt.show()

    # --- NEW: Helper Generator for Visible Outcomes ---
    def _iterate_visible_outcomes(self):
        """
        Generator that yields details for outcomes visible in the current view (1D, 2D, or 3D slice).

        Yields:
            tuple: (coords, current_sum) where:
                   coords (tuple): The (x, y, z) coordinates for plotting (0-based index).
                   current_sum (int): The sum of dice rolls for this outcome.
                   dice_rolls (tuple): The specific dice rolls (1-based) for this outcome.
        """
        if self.num_dice <= 0:
            return # Yield nothing if no dice

        sides = self.sides
        num_dice = self.num_dice

        # Build the full dice roll tuple based on fixed dimensions and iteration
        base_rolls = [0] * num_dice # 0-based for internal calculation
        fixed_indices = set(self.fixed_dims.keys())

        # Pre-fill fixed dimensions (adjusting to 0-based index)
        fixed_sum_part = 0
        for idx, val in self.fixed_dims.items():
            if 0 <= idx < num_dice:
                 # Ensure fixed val is valid before assigning
                if 1 <= val <= sides:
                    base_rolls[idx] = val - 1 # Store 0-based index
                    fixed_sum_part += val     # Add 1-based value to sum part
                else:
                    # If a fixed dimension is invalid (shouldn't happen with sliders),
                    # this entire slice is invalid. Stop iteration.
                    return

        if num_dice == 1:
            # Only dimension 0 varies
            if 0 in fixed_indices: # If the only die is fixed
                 # Check validity one last time
                if 1 <= self.fixed_dims[0] <= sides:
                    i = self.fixed_dims[0] - 1
                    current_sum = self.fixed_dims[0]
                    dice_rolls_1based = tuple(r + 1 for r in base_rolls)
                    yield (i, 0, 0), current_sum, dice_rolls_1based
            else: # Iterate the single die
                for i in range(sides):
                    base_rolls[0] = i
                    current_sum = i + 1 # Simple sum for one die
                    dice_rolls_1based = tuple(r + 1 for r in base_rolls)
                    yield (i, 0, 0), current_sum, dice_rolls_1based

        elif num_dice == 2:
            # Dimensions 0 and 1 vary (potentially)
            iter_indices = [idx for idx in range(2) if idx not in fixed_indices]

            if len(iter_indices) == 0: # Both dice fixed
                 # Check validity one last time
                if 1 <= self.fixed_dims[0] <= sides and 1 <= self.fixed_dims[1] <= sides:
                    i, j = self.fixed_dims[0] - 1, self.fixed_dims[1] - 1
                    current_sum = self.fixed_dims[0] + self.fixed_dims[1]
                    dice_rolls_1based = tuple(r + 1 for r in base_rolls)
                    yield (i, j, 0), current_sum, dice_rolls_1based
            elif len(iter_indices) == 1: # One die fixed
                fixed_idx = 1 - iter_indices[0]
                iter_idx = iter_indices[0]
                if not (1 <= self.fixed_dims[fixed_idx] <= sides): return # Invalid fixed value

                for val in range(sides):
                    base_rolls[iter_idx] = val
                    current_sum = fixed_sum_part + (val + 1)
                    plot_x = base_rolls[0]
                    plot_y = base_rolls[1]
                    dice_rolls_1based = tuple(r + 1 for r in base_rolls)
                    yield (plot_x, plot_y, 0), current_sum, dice_rolls_1based
            else: # Both dice iterate
                for i, j in itertools.product(range(sides), repeat=2):
                    base_rolls[0] = i
                    base_rolls[1] = j
                    current_sum = (i + 1) + (j + 1) # Simple sum for two dice
                    dice_rolls_1based = tuple(r + 1 for r in base_rolls)
                    yield (i, j, 0), current_sum, dice_rolls_1based

        else: # num_dice >= 3
            # Dimensions 0, 1, 2 vary (potentially)
            iter_indices = [idx for idx in range(3) if idx not in fixed_indices]

            if len(iter_indices) == 0: # All first 3 dice fixed
                if not (1 <= self.fixed_dims[0] <= sides and \
                        1 <= self.fixed_dims[1] <= sides and \
                        1 <= self.fixed_dims[2] <= sides):
                    return # Invalid fixed value(s)
                i, j, k = self.fixed_dims[0]-1, self.fixed_dims[1]-1, self.fixed_dims[2]-1
                current_sum = fixed_sum_part # Sum comes entirely from fixed dice
                dice_rolls_1based = tuple(r + 1 for r in base_rolls)
                yield (i, j, k), current_sum, dice_rolls_1based

            elif len(iter_indices) == 1: # Two of first 3 fixed
                # Determine which of first 3 axes iterates
                iter_idx = iter_indices[0]
                for val in range(sides):
                   base_rolls[iter_idx] = val
                   current_sum = fixed_sum_part + (val + 1)
                   plot_x, plot_y, plot_z = base_rolls[0], base_rolls[1], base_rolls[2]
                   dice_rolls_1based = tuple(r + 1 for r in base_rolls)
                   yield (plot_x, plot_y, plot_z), current_sum, dice_rolls_1based

            elif len(iter_indices) == 2: # One of first 3 fixed
                 # Determine which two of first 3 axes iterate
                for val1, val2 in itertools.product(range(sides), repeat=2):
                    base_rolls[iter_indices[0]] = val1
                    base_rolls[iter_indices[1]] = val2
                    current_sum = fixed_sum_part + (val1 + 1) + (val2 + 1)
                    plot_x, plot_y, plot_z = base_rolls[0], base_rolls[1], base_rolls[2]
                    dice_rolls_1based = tuple(r + 1 for r in base_rolls)
                    yield (plot_x, plot_y, plot_z), current_sum, dice_rolls_1based

            else: # First 3 dice iterate (standard 3D case)
                for i, j, k in itertools.product(range(sides), repeat=3):
                    base_rolls[0] = i
                    base_rolls[1] = j
                    base_rolls[2] = k
                    # Sum includes fixed part + iterated part
                    current_sum = fixed_sum_part + (i + 1) + (j + 1) + (k + 1)
                    dice_rolls_1based = tuple(r + 1 for r in base_rolls)
                    yield (i, j, k), current_sum, dice_rolls_1based


    # --- Modified _calculate_probability_and_counts ---
    def _calculate_probability_and_counts(self):
        """ Calculates probability based on the target_sums_set using on-the-fly sums. """
        if self.num_dice == 0:
            return 0.0, 0, 0

        current_target_set = self.target_sums_set
        total_visible = 0
        target_visible = 0

        # Use the generator to iterate through visible outcomes
        # The generator now calculates the sum internally
        for _, current_sum, _ in self._iterate_visible_outcomes():
            # Only count outcomes that are potentially visible (alpha > tiny)
            # Note: base_color alpha check is now primarily visual in _redraw_plot
            # Here we count all mathematically possible outcomes in the slice.
            total_visible += 1
            if current_sum in current_target_set:
                target_visible += 1

        probability = target_visible / total_visible if total_visible > 0 else 0.0
        return probability, total_visible, target_visible

    # --- Target Selector Click Handler (Unchanged) ---
    def _on_target_select(self, event):
        """Handles clicks on the target sum selector axes."""
        if event.inaxes != self.ax_target_selector: return
        if event.button != 1: return

        clicked_sum_float = event.xdata
        if clicked_sum_float is None: return

        clicked_sum_int = int(math.floor(clicked_sum_float + 0.5))

        if self.min_sum <= clicked_sum_int <= self.max_sum:
            if clicked_sum_int in self.target_sums_set:
                self.target_sums_set.remove(clicked_sum_int)
            else:
                self.target_sums_set.add(clicked_sum_int)
            self._redraw_target_selector()
            self._redraw_plot()

    # --- Redraw Target Selector Axes (Unchanged) ---
    def _redraw_target_selector(self):
        """Clears and redraws the visual state of the target selector axes."""
        ax = self.ax_target_selector
        ax.clear()
        ax.set_facecolor(self.target_selector_bg)

        ax.set_xlim(self.min_sum - 0.5, self.max_sum + 0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([])

        for selected_sum in self.target_sums_set:
            rect = patches.Rectangle((selected_sum - 0.4, 0.1), 0.8, 0.8,
                                     facecolor=self.target_selector_selected_color,
                                     edgecolor=None)
            ax.add_patch(rect)

        num_ticks = min(15, self.max_sum - self.min_sum + 1)
        tick_step = max(1, (self.max_sum - self.min_sum + 1) // num_ticks) if self.max_sum > self.min_sum else 1
        ticks = np.arange(self.min_sum, self.max_sum + 1, step=tick_step)
        ax.set_xticks(ticks)
        ax.tick_params(axis='x', colors=self.target_selector_tick_color, labelsize=8)

        ax.set_title("Target Sum(s) - Click to Select/Deselect", color=self.slice_slider_text_color, fontsize=9, pad=3)
        self.fig.canvas.draw_idle()


    # --- Modified _update_num_dice (No outcome_sums calculation) ---
    def _update_num_dice(self, val):
        new_num_dice = int(val)
        # Check current_num_dice correctly
        if new_num_dice == self.num_dice and hasattr(self, '_init_done'): return # Avoid redundant updates after init

        self.num_dice = new_num_dice
        # --- Removed: self._calculate_sums_internal() ---

        self.min_sum = self.num_dice * 1
        self.max_sum = self.num_dice * self.sides
        if self.max_sum < self.min_sum: self.max_sum = self.min_sum

        # Clear or filter target sums
        self.target_sums_set.clear() # Simplest approach
        # self.target_sums_set = {s for s in self.target_sums_set if self.min_sum <= s <= self.max_sum} # Alternative

        # Update Slice Sliders (Logic remains similar, ensure positioning)
        self.fixed_dims.clear()
        for slider in self.slice_sliders.values():
             if hasattr(slider, 'disconnect_events'): slider.disconnect_events()
        self.slice_sliders.clear()
        num_slice_sliders_needed = max(0, self.num_dice - 3)

        target_sel_y = self.ax_target_selector.get_position().y0
        element_h = self.ax_num_dice.get_position().height
        slider_spacing = 0.01
        current_slider_y = target_sel_y - (element_h + slider_spacing)

        for i in range(len(self.slice_slider_axes)):
            ax = self.slice_slider_axes[i]
            y_pos = current_slider_y - i * (element_h + slider_spacing)
            ax.set_position([0.15, y_pos, 0.65, element_h]) # Reposition just in case

            if i < num_slice_sliders_needed:
                fixed_idx = i + 3 # Axis index (0-based) for the die being fixed
                ax.clear()
                slice_init_val = 1
                slider = Slider(ax, f'Fix Die {fixed_idx + 1}', 1, self.sides, valinit=slice_init_val, valstep=1, color='lightgreen')
                # Use lambda with default argument capture for index
                slider.on_changed(lambda val, idx=fixed_idx: self._update_slice(idx, val))
                slider.label.set_color(self.slice_slider_text_color)
                slider.valtext.set_color(self.slice_slider_text_color)
                ax.set_facecolor(self.target_selector_bg)
                self.slice_sliders[fixed_idx] = slider
                self.fixed_dims[fixed_idx] = slice_init_val # Store initial fixed value
                ax.set_visible(True)
            else:
                ax.clear(); ax.set_visible(False)

        # Redraw Everything
        self._redraw_target_selector()
        self._redraw_plot()
        self._init_done = True # Mark initial setup as done


    # --- Modified _update_slice (No outcome_sums access) ---
    def _update_slice(self, fixed_idx, val):
        self.fixed_dims[fixed_idx] = int(val)
        # Redrawing plot recalculates probability based on the new slice constraints
        self._redraw_plot()


    # --- Modified _redraw_plot (Uses generator, on-the-fly sums) ---
    def _redraw_plot(self):
        # Calculate probability first (now uses generator)
        probability, total_cubes_in_view, target_cubes_in_view = self._calculate_probability_and_counts()
        dynamic_target_color = self.prob_cmap(probability)

        # Axis Management (same)
        if self.ax_main is None or getattr(self.ax_main, 'name', None) != '3d':
            if self.ax_main: self.ax_main.remove()
            self.ax_main = self.fig.add_subplot(111, projection='3d', facecolor='black')
            self.ax_main.set_position(self.main_ax_rect)
        else:
            self.ax_main.clear(); self.ax_main.set_facecolor('black')

        current_target_set = self.target_sums_set
        sides = self.sides
        num_dice = self.num_dice

        # Prepare lists for bar3d - consider pre-allocation if needed later
        xpos_list, ypos_list, zpos_list, colors_list = [], [], [], []
        dx = dy = dz = 0.95 # Bar size

        # Use the generator for iteration and on-the-fly sum calculation
        for coords, current_sum, _ in self._iterate_visible_outcomes():
            plot_x, plot_y, plot_z = coords # Get 0-based coords from generator

            is_target = current_sum in current_target_set
            color = dynamic_target_color if is_target else self.base_color

            # Check alpha value for visibility before adding to plot lists
            # Use math.isclose for float comparison robustness
            if not math.isclose(color[3], 0.0): # Check alpha component (index 3)
                # Center the bar visually by adding half the gap
                xpos_list.append(plot_x + (1-dx)/2)
                ypos_list.append(plot_y + (1-dy)/2)
                zpos_list.append(plot_z + (1-dz)/2)
                colors_list.append(color)

        # Plotting Bars
        if not xpos_list:
            # Display message if no visible bars (e.g., invalid slice or alpha=0)
            plot_center = (sides-1)/2 # Adjust center based on 0-based indexing
            self.ax_main.text(plot_center, plot_center, plot_center,
                              "No outcomes for this slice/view",
                              ha='center', va='center', color='gray', fontsize=10)
        else:
            # Convert lists to numpy arrays for bar3d
            xpos = np.array(xpos_list); ypos = np.array(ypos_list); zpos = np.array(zpos_list)
            colors_arr = np.array(colors_list); size = len(xpos)
            dx_arr = np.full(size, dx); dy_arr = np.full(size, dy); dz_arr = np.full(size, dz)

            self.ax_main.bar3d(xpos, ypos, zpos, dx_arr, dy_arr, dz_arr, color=colors_arr,
                               edgecolor=self.bar_edge_color, linewidth=self.bar_edge_linewidth,
                               shade=self.use_shade, # Use the configurable shade setting
                               zsort='average') # 'average' often looks best with transparency


        # Update Probability Bar (Logic mostly unchanged, uses pre-calculated prob)
        self.ax_prob_bar.clear(); self.ax_prob_bar.set_facecolor('black')
        self.ax_prob_bar.set_xticks([]); self.ax_prob_bar.set_yticks([])
        self.ax_prob_bar.set_xlim(0, 1); self.ax_prob_bar.set_ylim(0, 1)
        prob_bar_dynamic_color = self.prob_cmap(probability)
        self.ax_prob_bar.add_patch(patches.Rectangle((0, 0.1), 1, 0.8, color=self.prob_bar_bg_color, zorder=1))

        if total_cubes_in_view > 0:
            self.ax_prob_bar.add_patch(patches.Rectangle((0, 0.1), probability, 0.8, color=prob_bar_dynamic_color, zorder=2))
            if probability < 0.1: label_text = "Very Unlikely"
            elif probability < 0.3: label_text = "Unlikely"
            elif probability < 0.7: label_text = "Possible"
            elif probability < 0.9: label_text = "Likely"
            else: label_text = "Very Likely"
            display_text = f"{label_text} ({probability * 100:.1f}%)"
            self.ax_prob_bar.text(0.5, 0.5, display_text, ha='center', va='center', color='white', fontsize=10, zorder=3, bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2', edgecolor='none'))
        else:
             self.ax_prob_bar.text(0.5, 0.5, "N/A", ha='center', va='center', color='gray', fontsize=10, zorder=3)

        self.ax_prob_bar.set_title("Likelihood", color=self.slice_slider_text_color, fontsize=10, pad=5)

        # 3D Axis Styling - Update Title (Unchanged content, just formatting)
        # Use (sides-1) for limits because coordinates are 0-based indices
        self.ax_main.set_xlim(-0.5, sides - 0.5);
        self.ax_main.set_ylim(-0.5, sides - 0.5);
        self.ax_main.set_zlim(-0.5, sides - 0.5)

        fixed_dims_str = ", Fixed: " + ", ".join([f"D{k+1}={v}" for k,v in sorted(self.fixed_dims.items())]) if self.fixed_dims else ""
        title_color = 'white'
        target_str = self._format_target_set()
        plot_title = f'{num_dice} Dice Outcomes (Target: {target_str}{fixed_dims_str})'

        if self.ax_main and hasattr(self.ax_main, 'text2D'):
            self.ax_main.text2D(0.5, 0.98, plot_title, transform=self.ax_main.transAxes,
                                ha="center", va="top", color=title_color, fontsize=12)
            self.ax_main.axis('off') # Turn off grid/panes/ticks

        # Final Refresh
        try:
            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error during draw_idle: {e}")

    # --- Helper to format target set for title (Unchanged) ---
    def _format_target_set(self):
        """Formats the target_sums_set into a concise string."""
        if not self.target_sums_set:
            return "None"
        sorted_sums = sorted(list(self.target_sums_set))
        if len(sorted_sums) > 5: # Limit display length for title
             # Could implement range detection here (e.g., "5-7, 10")
             # For now, just indicate count
            return f"{len(sorted_sums)} values"
        return ", ".join(map(str, sorted_sums))


    # --- disconnect_events (Unchanged) ---
    def disconnect_events(self):
        if hasattr(self, 'slider_num_dice') and self.slider_num_dice: self.slider_num_dice.disconnect_events()
        if hasattr(self, 'slice_sliders') and self.slice_sliders:
            for slider in self.slice_sliders.values():
                if hasattr(slider, 'disconnect_events'): slider.disconnect_events()
        if hasattr(self, 'target_selector_cid') and self.target_selector_cid:
            try: self.fig.canvas.mpl_disconnect(self.target_selector_cid)
            except Exception: pass

# --- Main execution (Unchanged) ---
if __name__ == "__main__":
    # Example: Set max_dice to 7 for slightly better performance baseline
    visualizer = InteractiveDiceVisualizer(max_dice=7, sides=6, initial_dice=3)

    def on_close(event):
        print("Closing window...")
        if 'visualizer' in globals() and visualizer:
            visualizer.disconnect_events()

    if hasattr(visualizer, 'fig') and visualizer.fig:
        fig_id = visualizer.fig.canvas.mpl_connect('close_event', on_close)
    else:
        print("Error: Figure object not found for connecting close event.")