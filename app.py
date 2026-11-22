import sys
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QSpinBox, QComboBox, QGroupBox, QSlider, 
    QFormLayout, QCheckBox, QFileDialog
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
import mplcursors
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# --- Dark Theme Settings ---
DARK_BG = "#1e1e1e"
DARK_FG = "#e0e0e0"
GRID_COLOR = "#444444"

class ReductionWorker(QThread):
    """Runs data generation and dimensionality reduction in a background thread."""
    finished = Signal(object, object, str, object, dict)
    error = Signal(str)

    def __init__(self, n_dims, n_clusters, samples_per_cluster, cluster_std, random_state, method, target_dims, perplexity, standardize, view_mode):
        super().__init__()
        self.n_dims = n_dims
        self.n_clusters = n_clusters
        self.samples_per_cluster = samples_per_cluster
        self.cluster_std = cluster_std
        self.random_state = random_state
        self.method = method
        self.target_dims = target_dims
        self.perplexity = perplexity
        self.standardize = standardize
        self.view_mode = view_mode

    def run(self):
        try:
            n_total_samples = self.samples_per_cluster * self.n_clusters
            
            # 1. Generate N-dimensional blobs
            X, y = make_blobs(
                n_samples=n_total_samples, 
                n_features=self.n_dims, 
                centers=self.n_clusters, 
                random_state=self.random_state, 
                cluster_std=self.cluster_std
            )

            # Standardize data if requested
            if self.standardize:
                X = StandardScaler().fit_transform(X)

            # 2. Reduce dimensionality (or skip if Pairplot/Parallel)
            method_name = self.method
            X_reduced = None
            extra_info = {}
            
            # Calculate Cluster Quality Metrics (Subsample if huge to save time)
            if n_total_samples > 2000:
                sample_idx = np.random.choice(n_total_samples, 2000, replace=False)
                X_sample, y_sample = X[sample_idx], y[sample_idx]
            else:
                X_sample, y_sample = X, y
                
            if len(np.unique(y_sample)) > 1:
                extra_info['silhouette'] = silhouette_score(X_sample, y_sample)
                extra_info['davies_bouldin'] = davies_bouldin_score(X_sample, y_sample)
            
            if self.view_mode in ["Pairplot", "Parallel"]:
                X_reduced = X
                method_name = f"Raw {self.n_dims}D Data"
            else:
                if self.n_dims <= self.target_dims:
                    X_reduced = X[:, :self.target_dims]
                    method_name = f"Raw {self.target_dims}D Data"
                elif self.method == "PCA":
                    reducer = PCA(n_components=self.target_dims, random_state=self.random_state)
                    X_reduced = reducer.fit_transform(X)
                    total_variance = np.sum(reducer.explained_variance_ratio_) * 100
                    extra_info['variance'] = total_variance
                    method_name += f" ({total_variance:.1f}% Variance)"
                elif self.method == "t-SNE":
                    perplexity = min(self.perplexity, n_total_samples - 1)
                    reducer = TSNE(n_components=self.target_dims, perplexity=perplexity, random_state=self.random_state, init='pca', learning_rate='auto')
                    X_reduced = reducer.fit_transform(X)
                elif self.method == "MDS":
                    reducer = MDS(n_components=self.target_dims, random_state=self.random_state, normalized_stress='auto')
                    X_reduced = reducer.fit_transform(X)
                elif self.method == "Isomap":
                    n_neighbors = min(5, n_total_samples - 1)
                    reducer = Isomap(n_components=self.target_dims, n_neighbors=n_neighbors)
                    X_reduced = reducer.fit_transform(X)

            self.finished.emit(X, y, method_name, X_reduced, extra_info)
        except Exception as e:
            self.error.emit(str(e))


class BlobCanvas(FigureCanvas):
    """Custom Matplotlib Canvas widget"""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 6), facecolor=DARK_BG, constrained_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, facecolor=DARK_BG)
        self.show_placeholder()
        
        # Auto-rotation timer
        self.rotate_timer = QTimer(self)
        self.rotate_timer.timeout.connect(self._do_rotate)
        self.is_3d = False
        self.azim_angle = -60

    def _do_rotate(self):
        if self.is_3d:
            self.azim_angle = (self.azim_angle + 1) % 360
            self.ax.view_init(elev=20, azim=self.azim_angle)
            self.draw_idle()

    def set_auto_rotate(self, enabled):
        if enabled and self.is_3d:
            self.rotate_timer.start(30)
        else:
            self.rotate_timer.stop()

    def show_placeholder(self):
        self.ax.clear()
        self.fig.clear()
        self.ax = self.fig.add_subplot(111, facecolor=DARK_BG)
        self.ax.set_facecolor(DARK_BG)
        self.ax.set_title("N-Dimensional Blob Visualizer", color=DARK_FG)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_color(DARK_FG)
        self.ax.text(0.5, 0.5, "Configure parameters and click 'Generate'", 
                     ha='center', va='center', fontsize=12, color=DARK_FG, transform=self.ax.transAxes)
        self.draw()

    def plot_blobs(self, X_raw, X_reduced, y, method_name, point_size, alpha, cmap, view_mode, show_hulls, show_kde, show_centroids, use_legend):
        self.fig.clear() # Clear all axes to support grids
        
        if view_mode == "Pairplot":
            self.is_3d = False
            self._plot_pairplot(X_raw, y, point_size, alpha, cmap)
        elif view_mode == "Parallel":
            self.is_3d = False
            self._plot_parallel(X_raw, y, alpha, cmap)
        else:
            is_3d = X_reduced.shape[1] == 3
            self.is_3d = is_3d
            self._plot_scatter(X_reduced, y, method_name, point_size, alpha, cmap, is_3d, show_hulls, show_kde, show_centroids, use_legend)
        
        self.draw()

    def _plot_pairplot(self, X, y, point_size, alpha, cmap):
        dims_to_plot = min(3, X.shape[1])
        for i in range(dims_to_plot):
            for j in range(dims_to_plot):
                ax = self.fig.add_subplot(dims_to_plot, dims_to_plot, i*dims_to_plot + j + 1, facecolor=DARK_BG)
                ax.tick_params(axis='both', colors=DARK_FG, labelsize=8)
                
                if i == j:
                    ax.hist(X[:, i], bins=20, color=DARK_FG, alpha=0.7)
                else:
                    ax.scatter(X[:, j], X[:, i], c=y, cmap=cmap, s=point_size, alpha=alpha, edgecolors='none')
                
                if i == dims_to_plot - 1: ax.set_xlabel(f"Dim {j+1}", color=DARK_FG)
                if j == 0: ax.set_ylabel(f"Dim {i+1}", color=DARK_FG)
                
                ax.grid(True, linestyle='--', alpha=0.3, color=GRID_COLOR)
                for spine in ax.spines.values(): spine.set_color(DARK_FG)
                    
        self.fig.suptitle("Pairplot of First 3 Dimensions", color=DARK_FG, fontsize=14)

    def _plot_parallel(self, X, y, alpha, cmap):
        ax = self.fig.add_subplot(111, facecolor=DARK_BG)
        ax.set_title("Parallel Coordinates Plot", color=DARK_FG, fontsize=14, pad=10)
        
        # Limit to 15 dimensions for visual clarity
        dims_to_plot = min(15, X.shape[1])
        x_data = np.arange(1, dims_to_plot + 1)
        
        unique_clusters = np.unique(y)
        for cluster in unique_clusters:
            points = X[y == cluster, :dims_to_plot]
            color = plt.cm.get_cmap(cmap)(cluster / max(1, len(unique_clusters)-1))
            
            # Plot each line
            for point in points:
                ax.plot(x_data, point, alpha=alpha, color=color, linewidth=1.5)
                
        ax.set_xticks(x_data)
        ax.set_xticklabels([f"D{i}" for i in x_data], color=DARK_FG)
        ax.set_ylabel("Value", color=DARK_FG)
        ax.tick_params(axis='y', colors=DARK_FG)
        ax.grid(True, linestyle='--', alpha=0.3, color=GRID_COLOR)
        for spine in ax.spines.values(): spine.set_color(DARK_FG)

    def _plot_scatter(self, X_reduced, y, method_name, point_size, alpha, cmap, is_3d, show_hulls, show_kde, show_centroids, use_legend):
        if is_3d:
            self.ax = self.fig.add_subplot(111, projection='3d', facecolor=DARK_BG)
            self.ax.set_zlabel("Component 3", color=DARK_FG)
            self.ax.xaxis.set_pane_color((0, 0, 0, 0))
            self.ax.yaxis.set_pane_color((0, 0, 0, 0))
            self.ax.zaxis.set_pane_color((0, 0, 0, 0))
            self.ax.xaxis.grid(True, color=GRID_COLOR, linestyle='--', alpha=0.3)
            self.ax.yaxis.grid(True, color=GRID_COLOR, linestyle='--', alpha=0.3)
            self.ax.zaxis.grid(True, color=GRID_COLOR, linestyle='--', alpha=0.3)
            self.azim_angle = -60
            self.ax.view_init(elev=20, azim=self.azim_angle)
        else:
            self.ax = self.fig.add_subplot(111, facecolor=DARK_BG)
            self.ax.grid(True, linestyle='--', alpha=0.3, color=GRID_COLOR)
            self.ax.set_ylabel("Component 2", color=DARK_FG)

        self.ax.set_xlabel("Component 1", color=DARK_FG)
        self.ax.tick_params(axis='both', colors=DARK_FG)
        
        if not is_3d:
            for spine in self.ax.spines.values(): spine.set_color(DARK_FG)

        x_data = X_reduced[:, 0]
        y_data = X_reduced[:, 1]
        z_data = X_reduced[:, 2] if is_3d else None

        scatter = self.ax.scatter(
            x_data, y_data, z_data if is_3d else None, 
            c=y, cmap=cmap, s=point_size, alpha=alpha, edgecolors='none', picker=True
        )

        # 2D Enhancements (Hulls, KDE, Centroids)
        if not is_3d:
            unique_clusters = np.unique(y)
            cmap_func = plt.cm.get_cmap(cmap)
            
            if show_hulls:
                for cluster in unique_clusters:
                    points = X_reduced[y == cluster]
                    if len(points) >= 3:
                        try:
                            hull = ConvexHull(points)
                            hull_points = points[hull.vertices]
                            polygon = Polygon(hull_points, alpha=0.2, fill=False, edgecolor=DARK_FG, linewidth=1.5, linestyle='--')
                            self.ax.add_patch(polygon)
                        except Exception: pass

            if show_kde:
                # Create a grid for the contour
                xmin, xmax = self.ax.get_xlim()
                ymin, ymax = self.ax.get_ylim()
                xi, yi = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
                
                for cluster in unique_clusters:
                    points = X_reduced[y == cluster]
                    if len(points) >= 5: # Need enough points for KDE
                        try:
                            kde = gaussian_kde(points.T)
                            zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                            color = cmap_func(cluster / max(1, len(unique_clusters)-1))
                            # Draw 3 contour levels
                            self.ax.contour(xi, yi, zi, colors=[color], alpha=0.6, linewidths=1.5)
                        except Exception: pass

            if show_centroids:
                for cluster in unique_clusters:
                    points = X_reduced[y == cluster]
                    center = np.mean(points, axis=0)
                    self.ax.scatter(center[0], center[1], marker='X', color='white', s=150, edgecolors='black', linewidth=1.5, zorder=5)

        elif is_3d and show_centroids:
            unique_clusters = np.unique(y)
            for cluster in unique_clusters:
                points = X_reduced[y == cluster]
                center = np.mean(points, axis=0)
                self.ax.scatter(center[0], center[1], center[2], marker='X', color='white', s=150, edgecolors='black', linewidth=1.5, zorder=5)

        self.ax.set_title(f"{method_name} Reduction ({'3D' if is_3d else '2D'})", color=DARK_FG, fontsize=14, pad=10)
        
        if use_legend:
            handles, labels = scatter.legend_elements()
            self.ax.legend(handles, [f"Cluster {int(l)}" for l in labels], 
                           title="Clusters", facecolor=DARK_BG, edgecolor=DARK_FG, labelcolor=DARK_FG)
            leg = self.ax.get_legend()
            if leg: leg.get_title().set_color(DARK_FG)
        else:
            cb = self.fig.colorbar(scatter, ax=self.ax, shrink=0.6 if is_3d else 1.0)
            cb.set_label("Cluster ID", color=DARK_FG)
            cb.ax.yaxis.set_tick_params(color=DARK_FG)
            plt_out = cb.ax.get_yticklabels()
            [plt_label.set_color(DARK_FG) for plt_label in plt_out]
            cb.outline.set_edgecolor(DARK_FG)

        mplcursors.cursor(scatter, hover=True).connect(
            "add", lambda sel: sel.annotation.set_text(
                f"Cluster: {y[sel.target.index]}\n"
                f"X: {sel.target[0]:.2f}\n"
                f"Y: {sel.target[1]:.2f}" + (f"\nZ: {sel.target[2]:.2f}" if is_3d else "")
            )
        )

# Need to import plt for colormap fetching inside canvas
import matplotlib.pyplot as plt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced N-Dimensional Blob Visualizer & EDA Tool")
        self.resize(1200, 800)
        self.worker = None
        
        self.current_X_raw = None
        self.current_X_reduced = None
        self.current_y = None
        self.current_method_name = ""

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- LEFT PANEL ---
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_widget.setFixedWidth(320)
        
        # Data Generation Group
        gen_group = QGroupBox("Data Generation")
        gen_layout = QFormLayout(gen_group)
        
        self.dim_spin = QSpinBox()
        self.dim_spin.setRange(2, 100)
        self.dim_spin.setValue(15)
        gen_layout.addRow("Dimensions (N):", self.dim_spin)

        self.cluster_spin = QSpinBox()
        self.cluster_spin.setRange(2, 15)
        self.cluster_spin.setValue(5)
        gen_layout.addRow("Clusters:", self.cluster_spin)

        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(10, 1000)
        self.sample_spin.setValue(150)
        gen_layout.addRow("Samples/Cluster:", self.sample_spin)

        self.std_slider = QSlider(Qt.Horizontal)
        self.std_slider.setRange(5, 50)
        self.std_slider.setValue(15)
        self.std_label = QLabel("1.5")
        self.std_slider.valueChanged.connect(lambda v: self.std_label.setText(f"{v/10:.1f}"))
        gen_layout.addRow("Cluster Spread:", self.std_slider)
        gen_layout.addRow("", self.std_label)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 9999)
        self.seed_spin.setValue(42)
        
        # Seed layout with "Next" button
        seed_widget = QWidget()
        seed_layout = QHBoxLayout(seed_widget)
        seed_layout.setContentsMargins(0,0,0,0)
        seed_layout.addWidget(self.seed_spin)
        self.btn_next_seed = QPushButton("→")
        self.btn_next_seed.setFixedWidth(30)
        self.btn_next_seed.setToolTip("Increment seed and regenerate")
        self.btn_next_seed.clicked.connect(self.next_seed)
        seed_layout.addWidget(self.btn_next_seed)
        gen_layout.addRow("Random Seed:", seed_widget)
        
        self.chk_standardize = QCheckBox("Standardize Data (Z-Score)")
        gen_layout.addRow(self.chk_standardize)

        controls_layout.addWidget(gen_group)

        # Algorithm Group
        algo_group = QGroupBox("Algorithm & View")
        algo_layout = QFormLayout(algo_group)

        self.view_combo = QComboBox()
        self.view_combo.addItems(["Reduction (2D/3D)", "Pairplot (Raw N-Dims)", "Parallel Coordinates"])
        self.view_combo.currentTextChanged.connect(self.toggle_algorithm_visibility)
        algo_layout.addRow("View Mode:", self.view_combo)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["PCA", "t-SNE", "MDS", "Isomap"])
        self.method_combo.currentTextChanged.connect(self.toggle_perplexity_visibility)
        algo_layout.addRow("Method:", self.method_combo)

        self.perplexity_spin = QSpinBox()
        self.perplexity_spin.setRange(5, 100)
        self.perplexity_spin.setValue(30)
        self.perplexity_spin.setEnabled(False)
        algo_layout.addRow("t-SNE Perplexity:", self.perplexity_spin)

        controls_layout.addWidget(algo_group)
        
        # Visual Features Group
        vis_group = QGroupBox("Visual Features (Live Update)")
        vis_layout = QFormLayout(vis_group)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["viridis", "plasma", "inferno", "magma", "cividis", "tab10", "rainbow"])
        self.cmap_combo.currentTextChanged.connect(self.update_plot_visuals)
        vis_layout.addRow("Colormap:", self.cmap_combo)

        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(5, 100)
        self.size_slider.setValue(30)
        self.size_slider.valueChanged.connect(self.update_plot_visuals)
        vis_layout.addRow("Point Size:", self.size_slider)

        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(10, 100)
        self.alpha_slider.setValue(80)
        self.alpha_slider.valueChanged.connect(self.update_plot_visuals)
        vis_layout.addRow("Opacity:", self.alpha_slider)

        controls_layout.addWidget(vis_group)

        # Plot Toggles Group
        feat_group = QGroupBox("Plot Toggles")
        feat_layout = QVBoxLayout(feat_group)
        
        self.chk_3d = QCheckBox("Enable 3D Plot (Reduces to 3 dims)")
        self.chk_3d.toggled.connect(self.update_plot_visuals)
        feat_layout.addWidget(self.chk_3d)
        
        self.chk_rotate = QCheckBox("Auto-Rotate 3D View")
        self.chk_rotate.toggled.connect(self.toggle_auto_rotate)
        feat_layout.addWidget(self.chk_rotate)
        
        self.chk_hulls = QCheckBox("Show Convex Hulls (2D Only)")
        self.chk_hulls.setChecked(False)
        self.chk_3d.toggled.connect(self.chk_hulls.setDisabled)
        self.chk_hulls.toggled.connect(self.update_plot_visuals)
        feat_layout.addWidget(self.chk_hulls)
        
        self.chk_kde = QCheckBox("Show Density Contours (KDE) (2D Only)")
        self.chk_kde.setChecked(True)
        self.chk_3d.toggled.connect(self.chk_kde.setDisabled)
        self.chk_kde.toggled.connect(self.update_plot_visuals)
        feat_layout.addWidget(self.chk_kde)
        
        self.chk_centroids = QCheckBox("Show Cluster Centroids")
        self.chk_centroids.setChecked(True)
        self.chk_centroids.toggled.connect(self.update_plot_visuals)
        feat_layout.addWidget(self.chk_centroids)

        self.chk_legend = QCheckBox("Use Discrete Legend (Instead of Colorbar)")
        self.chk_legend.toggled.connect(self.update_plot_visuals)
        feat_layout.addWidget(self.chk_legend)
        
        controls_layout.addWidget(feat_group)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_generate = QPushButton("⏵ Generate")
        self.btn_generate.setStyleSheet("font-weight: bold; background-color: #2d8c3c; color: white; padding: 12px; border-radius: 4px;")
        self.btn_generate.clicked.connect(self.start_generation)
        btn_layout.addWidget(self.btn_generate)

        self.btn_export = QPushButton("💾 Export CSV")
        self.btn_export.setStyleSheet("font-weight: bold; background-color: #007acc; color: white; padding: 12px; border-radius: 4px;")
        self.btn_export.clicked.connect(self.export_data)
        self.btn_export.setEnabled(False)
        btn_layout.addWidget(self.btn_export)
        
        controls_layout.addLayout(btn_layout)
        controls_layout.addStretch()
        main_layout.addWidget(controls_widget)

        # --- RIGHT PANEL ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.canvas = BlobCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet(f"QToolBar {{ background: {DARK_BG}; border: none; spacing: 5px; }}"
                                   f"QToolButton {{ background: {DARK_BG}; color: {DARK_FG}; padding: 4px; }}"
                                   f"QToolButton:hover {{ background: #333333; }}")

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        main_layout.addWidget(right_panel, 1)

        self.statusBar().showMessage("Ready")

    def next_seed(self):
        self.seed_spin.setValue(self.seed_spin.value() + 1)
        self.start_generation()

    def toggle_auto_rotate(self, checked):
        self.canvas.set_auto_rotate(checked)

    def toggle_algorithm_visibility(self, view_mode):
        is_reduction = (view_mode == "Reduction (2D/3D)")
        self.method_combo.setEnabled(is_reduction)
        self.perplexity_spin.setEnabled(is_reduction and self.method_combo.currentText() == "t-SNE")
        self.chk_3d.setEnabled(is_reduction)
        self.chk_rotate.setEnabled(is_reduction)

    def toggle_perplexity_visibility(self, method):
        is_tsne = (method == "t-SNE")
        is_reduction = (self.view_combo.currentText() == "Reduction (2D/3D)")
        self.perplexity_spin.setEnabled(is_tsne and is_reduction)

    def start_generation(self):
        self.btn_generate.setEnabled(False)
        self.btn_generate.setText("⏳ Computing...")
        self.btn_export.setEnabled(False)
        self.statusBar().showMessage("Generating data and reducing dimensions in background...")
        QApplication.processEvents()

        view_mode = "Reduction" if "Reduction" in self.view_combo.currentText() else ("Pairplot" if "Pairplot" in self.view_combo.currentText() else "Parallel")
        is_3d = self.chk_3d.isChecked()
        target_dims = 3 if is_3d else 2

        params = {
            "n_dims": self.dim_spin.value(),
            "n_clusters": self.cluster_spin.value(),
            "samples_per_cluster": self.sample_spin.value(),
            "cluster_std": int(self.std_slider.value()) / 10.0,
            "random_state": self.seed_spin.value(),
            "method": self.method_combo.currentText(),
            "target_dims": target_dims,
            "perplexity": self.perplexity_spin.value(),
            "standardize": self.chk_standardize.isChecked(),
            "view_mode": view_mode
        }

        self.worker = ReductionWorker(**params)
        self.worker.finished.connect(self.on_generation_finished)
        self.worker.error.connect(self.on_generation_error)
        self.worker.start()

    def on_generation_finished(self, X_raw, y, method_name, X_reduced, extra_info):
        self.current_X_raw = X_raw
        self.current_y = y
        self.current_method_name = method_name
        self.current_X_reduced = X_reduced if X_reduced is not None else X_raw
        
        self.update_plot_visuals()
        
        n_dims = self.dim_spin.value()
        n_clusters = self.cluster_spin.value()
        n_samples = self.sample_spin.value() * n_clusters
        dims_str = "3D" if self.chk_3d.isChecked() else "2D"
        view_str = self.view_combo.currentText()
        
        status_msg = f"Success: {n_samples} samples in {n_dims}D | View: {view_str} | Method: {method_name}."
        if 'variance' in extra_info:
            status_msg += f" | Variance: {extra_info['variance']:.1f}%"
        if 'silhouette' in extra_info:
            status_msg += f" | Silhouette: {extra_info['silhouette']:.3f} (Higher is better)"
        if 'davies_bouldin' in extra_info:
            status_msg += f" | Davies-Bouldin: {extra_info['davies_bouldin']:.3f} (Lower is better)"
            
        self.statusBar().showMessage(status_msg)
        
        self.btn_generate.setEnabled(True)
        self.btn_generate.setText("⏵ Generate")
        self.btn_export.setEnabled(True)

    def update_plot_visuals(self):
        if self.current_X_raw is None:
            return

        point_size = self.size_slider.value()
        alpha = int(self.alpha_slider.value()) / 100.0
        cmap = self.cmap_combo.currentText()
        is_3d = self.chk_3d.isChecked()
        show_hulls = self.chk_hulls.isChecked() and not is_3d
        show_kde = self.chk_kde.isChecked() and not is_3d
        show_centroids = self.chk_centroids.isChecked()
        use_legend = self.chk_legend.isChecked()
        view_mode = "Pairplot" if "Pairplot" in self.view_combo.currentText() else ("Parallel" if "Parallel" in self.view_combo.currentText() else "Reduction")

        self.canvas.plot_blobs(
            self.current_X_raw, self.current_X_reduced, self.current_y, self.current_method_name,
            point_size, alpha, cmap, view_mode, show_hulls, show_kde, show_centroids, use_legend
        )
        
        self.canvas.set_auto_rotate(self.chk_rotate.isChecked())

    def export_data(self):
        if self.current_X_raw is None:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Data as CSV", "blob_data.csv", "CSV Files (*.csv)")
        if not file_path:
            return
            
        try:
            n_samples = self.current_X_raw.shape[0]
            n_raw_dims = self.current_X_raw.shape[1]
            n_red_dims = self.current_X_reduced.shape[1] if self.current_X_reduced is not None else 0
            
            raw_headers = [f"Raw_Dim_{i+1}" for i in range(n_raw_dims)]
            red_headers = [f"Reduced_Dim_{i+1}" for i in range(n_red_dims)]
            header = ",".join(raw_headers + red_headers + ["Cluster_ID"])
            
            export_data = np.hstack((self.current_X_raw, self.current_X_reduced, self.current_y.reshape(-1, 1)))
            
            np.savetxt(file_path, export_data, delimiter=",", header=header, comments="", fmt="%.6f")
            self.statusBar().showMessage(f"Successfully exported {n_samples} samples to {file_path}")
        except Exception as e:
            self.statusBar().showMessage(f"Error saving file: {str(e)}")

    def on_generation_error(self, error_msg):
        self.statusBar().showMessage(f"Error: {error_msg}")
        self.btn_generate.setEnabled(True)
        self.btn_generate.setText("⏵ Generate")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    app.setStyleSheet(f"""
        QWidget {{ background-color: {DARK_BG}; color: {DARK_FG}; font-size: 10pt; }}
        QGroupBox {{ font-weight: bold; border: 1px solid #3a3a3a; border-radius: 4px; margin-top: 10px; padding-top: 10px; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }}
        QSpinBox, QComboBox {{ background-color: #2d2d2d; border: 1px solid #3a3a3a; border-radius: 2px; padding: 3px; }}
        QComboBox QAbstractItemView {{ background-color: #2d2d2d; selection-background-color: #2d8c3c; }}
        QPushButton {{ background-color: #007acc; color: white; border: none; padding: 5px; border-radius: 3px; }}
        QPushButton:hover {{ background-color: #0098ff; }}
        QPushButton:disabled {{ background-color: #333333; color: #888888; }}
        QSlider::groove:horizontal {{ background: #2d2d2d; height: 6px; border-radius: 3px; }}
        QSlider::handle:horizontal {{ background: #e0e0e0; width: 14px; margin: -4px 0; border-radius: 7px; }}
        QCheckBox {{ spacing: 8px; }}
        QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 3px; border: 1px solid #555555; background: #2d2d2d; }}
        QCheckBox::indicator:checked {{ background: #2d8c3c; border: 1px solid #2d8c3c; }}
        QStatusBar {{ background-color: #007acc; color: white; }}
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())