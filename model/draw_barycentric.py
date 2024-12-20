import numpy as np
import matplotlib.pyplot as plt


# Function to generate vertices of an equilateral triangle
def generate_triangle_vertices():
    L = 2 / np.sqrt(3)  # Side length
    return np.array([[0, 0], [L, 0], [L / 2, 1]])  # Vertices in Cartesian coordinates


# Function to project a point to a line segment (find perpendicular foot)
def project_to_line(p, a, b):
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)  # Ensure projection is within the segment
    return a + t * ab


# Function to plot a beautiful Barycentric triangle
def plot_barycentric_triangle():
    fig, ax = plt.subplots(figsize=(8, 8))


    # Generate triangle vertices
    vertices = generate_triangle_vertices()
    L = 2 / np.sqrt(3)

    # Draw triangle with a nice edge color
    ax.fill(*zip(*vertices), edgecolor='black', facecolor='white', linewidth=2)

    # Annotate triangle vertices with modern fonts
    ax.text(0, -0.05, r'$\bar{x}$ vertex (disbelief)', fontsize=12, ha='center', va='top', color='black')
    ax.text(L, -0.05, r'$x$ vertex (belief)', fontsize=12, ha='center', va='top', color='black')
    ax.text(L / 2, 1.05, r'$u$ vertex (uncertainty)', fontsize=12, ha='center', color='black')

    # Define barycentric coordinates
    disbelief = 0.3
    belief = 0.4
    uncertainty = 1 - disbelief - belief

    # Convert to Cartesian coordinates
    bx = disbelief * 0 + belief * L + uncertainty * (L / 2)
    by = disbelief * 0 + belief * 0 + uncertainty * 1
    barycenter = np.array([bx, by])

    # Plot barycentric point with size and style
    ax.plot(bx, by, 'o', color='red', markersize=8, label=r'$C_x=(b_x, d_x, u_x)$', zorder=5)

    # Draw perpendicular lines with soft dashed lines
    edge_pairs = [
        (vertices[0], vertices[1], r'$u_x$', 'blue'),
        (vertices[1], vertices[2], r'$d_x$', 'green'),
        (vertices[2], vertices[0], r'$b_x$', 'purple')
    ]
    for a, b, label, color in edge_pairs:
        foot = project_to_line(barycenter, a, b)
        ax.plot([barycenter[0], foot[0]], [barycenter[1], foot[1]], linestyle='--', color=color, linewidth=1.5)
        ax.plot(foot[0], foot[1], 'o', color=color, markersize=6, markeredgecolor='black', zorder=4)

        # Add label to the middle of the line
        mid_x = (barycenter[0] + foot[0]) / 2
        mid_y = (barycenter[1] + foot[1]) / 2
        ax.text(mid_x, mid_y, label, fontsize=12, color=color, ha='center', va='center', backgroundcolor='white')

    # Add barycenter label
    ax.text(bx + 0.02, by + 0.02, r'$C_x$', fontsize=12, color='red')

    # Add grid for a more polished look
    ax.grid(True, linestyle='--', alpha=0.3)

    # Axis settings
    ax.set_xlim(-0.1, L + 0.1)
    ax.set_ylim(-0.1, 1.1)
    ax.axis('equal')
    ax.axis('off')
    plt.title("Barycentric Triangle", fontsize=15, color='black')
    plt.legend()
    plt.show()


# Execute the improved function
plot_barycentric_triangle()
