import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet
from matplotlib.patches import RegularPolygon
text_size = 25  # 全局字体大小

# Function to generate vertices of an equilateral triangle
def generate_triangle_vertices():
    L = 2 / np.sqrt(3)  # Side length
    return np.array([[0, 0], [L, 0], [L / 2, 1]])  # Vertices (B, A, C)

# Function to generate barycentric grid and convert to Cartesian coordinates
def generate_barycentric_grid(resolution=400):
    b_values = np.linspace(0, 1, resolution)
    d_values = np.linspace(0, 1, resolution)
    grid_b, grid_d = np.meshgrid(b_values, d_values)
    u_values = 1 - grid_b - grid_d  # u = 1 - b - d

    # Mask valid barycentric coordinates where all b, d, and u > 0
    mask = (u_values > 0) & (grid_b > 0) & (grid_d > 0)
    grid_b, grid_d, grid_u = grid_b[mask], grid_d[mask], u_values[mask]

    # Convert to Cartesian coordinates
    L = 2 / np.sqrt(3)
    x = grid_b * L + grid_d * 0 + grid_u * (L / 2)
    y = grid_d * 0 + grid_u * 1
    return x, y, np.vstack([grid_b, grid_d, grid_u]).T  # Cartesian and barycentric

# Function to compute Dirichlet density
def compute_dirichlet_density(barycentric_coords, alpha):
    return np.array([dirichlet.pdf(coord, alpha) for coord in barycentric_coords])

# Function to project a point to a line segment (find perpendicular foot)
def project_to_line(p, a, b):
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)  # Ensure projection is within the segment
    return a + t * ab

# Function to plot Barycentric triangle with perpendicular lines (First Subplot)
def plot_barycentric_triangle(ax):
    vertices = generate_triangle_vertices()
    L = 2 / np.sqrt(3)

    # Draw triangle
    ax.fill(*zip(*vertices), edgecolor='black', facecolor='white', linewidth=2)

    # Annotate vertices
    ax.text(0, -0.05, r'$d$', fontsize=text_size, ha='center', va='top')
    ax.text(L, -0.05, r'$u$', fontsize=text_size, ha='center', va='top')
    ax.text(L / 2, 1.05, r'$b$', fontsize=text_size, ha='center')

    # Define barycentric coordinates
    disbelief = 0.3
    belief = 0.4
    uncertainty = 1 - disbelief - belief

    # Convert to Cartesian coordinates
    bx = disbelief * 0 + belief * L + uncertainty * (L / 2)
    by = disbelief * 0 + belief * 0 + uncertainty * 1
    barycenter = np.array([bx, by])

    # Plot barycentric point
    ax.plot(bx, by, 'o', color='red', markersize=8, label=r'$C_x$', zorder=5)

    # Draw perpendicular lines in two segments to leave space for text labels
    edge_pairs = [
        (vertices[0], vertices[1], (118/ 255,181/ 255,197/ 255), r'$b_x$'),
        (vertices[1], vertices[2], (234/ 255,182/ 255,118/ 255), r'$d_x$'),
        (vertices[2], vertices[0], (135/ 255,62/ 255,35/ 255), r'$u_x$')
    ]
    for a, b, color, label in edge_pairs:
        foot = project_to_line(barycenter, a, b)
        line_x = [barycenter[0], foot[0]]
        line_y = [barycenter[1], foot[1]]

        # Divide the line into two segments
        mid_x = (barycenter[0] + foot[0]) / 2
        mid_y = (barycenter[1] + foot[1]) / 2

        # Draw first segment
        ax.plot([line_x[0], (line_x[0] + mid_x) / 2], [line_y[0], (line_y[0] + mid_y) / 2],
                '--', color=color, linewidth=1.5)
        # Draw second segment
        ax.plot([(line_x[1] + mid_x) / 2, line_x[1]], [(line_y[1] + mid_y) / 2, line_y[1]],
                '--', color=color, linewidth=1.5)

        # Plot foot of the perpendicular
        ax.plot(foot[0], foot[1], 'o', color=color, markersize=6, markeredgecolor='black')

        # Add label at the center of the line
        ax.text(mid_x, mid_y, label, fontsize=text_size, color=color, ha='center', va='center')

    # # Add custom arrow and pattern
    # ax.annotate('', xy=(1.1, 0.5), xytext=(L / 2, 0.5),  # Arrow start and end points
    #             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

    # Draw circles using RegularPolygon
    circle_x = 2 / np.sqrt(3) + 0.25  # Align with colorbar x-coordinate
    radius = 0.05
    circle_centers = [0.5, 0.65, 0.35]
    circle_colors = ['blue', 'green', 'purple']
    for center, color in zip(circle_centers, circle_colors):
        patch = RegularPolygon((circle_x, center), numVertices=50, radius=radius, color=color, zorder=10)
        ax.add_patch(patch)

    # Add label below the triangle
    ax.text(L / 2, -0.3, r'$\omega_x=(b_x, d_x, u_x)$', fontsize=text_size, ha='center')

    # Axis settings
    ax.set_aspect('equal')
    ax.axis('off')



# Function to plot Dirichlet heatmaps and the Barycentric triangle
def plot_combined_heatmaps():
    vertices = generate_triangle_vertices()
    x, y, barycentric_coords = generate_barycentric_grid()

    # Define alphas and titles
    alphas = [
        [4, 2, 2]
    ]
    labels = [
        "Beta Distribution"
    ]

    # Create figure with subplots in a vertical layout (2 rows, 1 column)
    fig, axes = plt.subplots(len(alphas) + 1, 1, figsize=(6, 10),  # Adjust size for vertical layout
                             gridspec_kw={'hspace': 0.1},  # Reduce vertical space
                             constrained_layout=True)

    # First subplot: Barycentric triangle
    plot_barycentric_triangle(axes[0])

    # Compute densities for all alphas and apply log transformation
    all_densities = []
    for alpha in alphas:
        density = compute_dirichlet_density(barycentric_coords, alpha)
        log_density = density  # Log transformation
        all_densities.append(log_density)

    vmin = min(np.min(d) for d in all_densities)
    vmax = max(np.max(d) for d in all_densities)

    # Remaining subplots: Dirichlet heatmaps with log-scaled densities and contour lines
    for i, (alpha, ax, label) in enumerate(zip(alphas, axes[1:], labels)):
        log_density = all_densities[i]
        # Heatmap scatter plot
        sc = ax.scatter(x, y, c=log_density, cmap='inferno', s=1, edgecolor=None, vmin=vmin, vmax=vmax)
        ax.plot(*zip(*np.vstack([vertices, vertices[0]])), 'k-', linewidth=1.5)

        # Annotate vertices
        ax.text(0, -0.05, r'$d$', fontsize=text_size, ha='center', va='top')
        ax.text(2 / np.sqrt(3), -0.05, r'$b$', fontsize=text_size, ha='center', va='top')
        ax.text(1 / np.sqrt(3), 1.05, r'$u$', fontsize=text_size, ha='center')

        # Add alpha label below the triangle
        ax.text(1 / np.sqrt(3), -0.3, label, fontsize=text_size, ha='center')

        ax.set_aspect('equal')
        ax.axis('off')

        # Add a colorbar only for the heatmap
        cbar = fig.colorbar(sc, ax=ax, location='right', shrink=0.8)
        cbar.ax.set_ylabel("Distribution Density", fontsize=text_size)
        cbar.ax.tick_params(labelsize=20)

    # Set consistent axis limits for all subplots
    for ax in axes:
        ax.set_xlim(-0.1, 2 / np.sqrt(3) + 0.1)
        ax.set_ylim(-0.1, 1.1)

    plt.savefig("beta_heatmaps_tight_vertical.jpg", dpi=300)
    plt.show()

# Execute the combined function
plot_combined_heatmaps()
