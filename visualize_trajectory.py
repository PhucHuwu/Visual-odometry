"""Visualize 3D Trajectory

Đọc trajectory từ file và hiển thị 3D plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


def load_trajectory(filename):
    """Load trajectory từ file"""
    data = np.loadtxt(filename)
    return data


def plot_trajectory_3d(trajectory, title="Visual Odometry Trajectory"):
    """Plot 3D trajectory"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]

    # Plot trajectory line
    ax.plot(x, y, z, 'b-', linewidth=2, label='Camera Path')

    # Plot start and end points
    ax.scatter(x[0], y[0], z[0], c='green', marker='o', s=100, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='red', marker='s', s=100, label='End')

    # Labels
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Legend
    ax.legend()

    # Grid
    ax.grid(True, alpha=0.3)

    # Equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Stats
    stats_text = f"Total poses: {len(trajectory)}\n"
    stats_text += f"Path length: {np.linalg.norm(trajectory[-1] - trajectory[0]):.2f} m\n"
    stats_text += f"X range: [{x.min():.2f}, {x.max():.2f}]\n"
    stats_text += f"Y range: [{y.min():.2f}, {y.max():.2f}]\n"
    stats_text += f"Z range: [{z.min():.2f}, {z.max():.2f}]"

    plt.figtext(0.02, 0.98, stats_text, fontsize=10,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def plot_trajectory_2d(trajectory):
    """Plot 2D trajectory (top view)"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]

    # XY plane (top view)
    axes[0].plot(x, y, 'b-', linewidth=2)
    axes[0].scatter(x[0], y[0], c='green', marker='o', s=100, label='Start')
    axes[0].scatter(x[-1], y[-1], c='red', marker='s', s=100, label='End')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title('Top View (XY Plane)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].axis('equal')

    # XZ plane (side view)
    axes[1].plot(x, z, 'g-', linewidth=2)
    axes[1].scatter(x[0], z[0], c='green', marker='o', s=100, label='Start')
    axes[1].scatter(x[-1], z[-1], c='red', marker='s', s=100, label='End')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Z (m)')
    axes[1].set_title('Side View (XZ Plane)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].axis('equal')

    # YZ plane (front view)
    axes[2].plot(y, z, 'r-', linewidth=2)
    axes[2].scatter(y[0], z[0], c='green', marker='o', s=100, label='Start')
    axes[2].scatter(y[-1], z[-1], c='red', marker='s', s=100, label='End')
    axes[2].set_xlabel('Y (m)')
    axes[2].set_ylabel('Z (m)')
    axes[2].set_title('Front View (YZ Plane)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].axis('equal')

    plt.tight_layout()
    return fig


def plot_top_view(trajectory):
    """Plot top view only (XY plane - nhìn từ trên xuống)"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    x = trajectory[:, 0]
    y = trajectory[:, 1]

    # Plot trajectory
    ax.plot(x, y, 'b-', linewidth=2, label='Camera Path')

    # Plot start and end
    ax.scatter(x[0], y[0], c='green', marker='o', s=150, label='Start', zorder=5)
    ax.scatter(x[-1], y[-1], c='red', marker='s', s=150, label='End', zorder=5)

    # Labels
    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Y (m)', fontsize=14)
    ax.set_title('Top View - Camera Trajectory (Bird\'s Eye View)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.axis('equal')

    # Stats
    stats_text = f"Total poses: {len(trajectory)}\n"
    stats_text += f"X range: [{x.min():.2f}, {x.max():.2f}] m\n"
    stats_text += f"Y range: [{y.min():.2f}, {y.max():.2f}] m\n"
    stats_text += f"Horizontal distance: {np.linalg.norm([x[-1] - x[0], y[-1] - y[0]]):.2f} m"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize Visual Odometry Trajectory')
    parser.add_argument(
        '--input',
        type=str,
        default='output/trajectory.txt',
        help='Path tới trajectory file (default: output/trajectory.txt)'
    )
    parser.add_argument(
        '--2d',
        action='store_true',
        dest='plot_2d',
        help='Hiển thị 2D views thay vì 3D'
    )
    parser.add_argument(
        '--top-view',
        action='store_true',
        dest='top_view',
        help='Hiển thị top view only (XY plane - nhìn từ trên xuống, bỏ qua Z)'
    )

    args = parser.parse_args()

    # Load trajectory
    print(f"Loading trajectory từ: {args.input}")
    trajectory = load_trajectory(args.input)
    print(f"Loaded {len(trajectory)} poses")

    # Plot
    if args.top_view:
        print("Plotting top view (XY plane)...")
        fig = plot_top_view(trajectory)
    elif args.plot_2d:
        print("Plotting 2D views...")
        fig = plot_trajectory_2d(trajectory)
    else:
        print("Plotting 3D trajectory...")
        fig = plot_trajectory_3d(trajectory)

    print("Hiển thị plot. Đóng cửa sổ để thoát.")
    plt.show()


if __name__ == '__main__':
    main()
