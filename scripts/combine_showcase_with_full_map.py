#!/usr/bin/env python3
"""
Combine the showcase plot with the full X-ray map.

Creates a figure with the full map as the main panel and the showcase as an inset.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from PIL import Image
import numpy as np

# Paths
BASE_DIR = Path('/Users/gozalig1/Projects/compact-groups-xray-analysis')
SHOWCASE_IMG = BASE_DIR / 'outputs' / 'figures' / 'cw_all' / 'showcase_cw_all.png'
FULL_MAP_IMG = BASE_DIR / 'cosmos-web_galaxy-groups-X-ray-properties' / 'figures' / 'xray_map_full_simple.png'
OUTPUT_DIR = BASE_DIR / 'cosmos-web_galaxy-groups-X-ray-properties' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def combine_images():
    """Combine showcase and full map into a single figure."""
    print("Loading images...")
    
    # Load images
    showcase = Image.open(SHOWCASE_IMG)
    full_map = Image.open(FULL_MAP_IMG)
    
    print(f"Showcase size: {showcase.size}")
    print(f"Full map size: {full_map.size}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Main panel: full map (larger, left side)
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(full_map)
    ax1.axis('off')
    ax1.set_title('Full COSMOS X-ray Map', fontsize=14, fontweight='bold', pad=10)
    
    # Inset panel: showcase (smaller, right side)
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(showcase)
    ax2.axis('off')
    ax2.set_title('Showcase: Highest SNR Detected Group', fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout(pad=2.0)
    
    output_path = OUTPUT_DIR / 'xray_map_with_showcase.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Alternative: Create with showcase as inset on the full map
    fig2 = plt.figure(figsize=(14, 10))
    ax_main = plt.subplot(1, 1, 1)
    ax_main.imshow(full_map)
    ax_main.axis('off')
    ax_main.set_title('COSMOS X-ray Map with Showcase Inset', fontsize=16, fontweight='bold', pad=15)
    
    # Add showcase as inset in upper right corner
    # Position: [left, bottom, width, height] in figure coordinates
    ax_inset = fig2.add_axes([0.55, 0.55, 0.4, 0.4])  # Upper right corner
    ax_inset.imshow(showcase)
    ax_inset.axis('off')
    
    # Add border around inset
    rect = Rectangle((0, 0), 1, 1, transform=ax_inset.transAxes,
                     fill=False, edgecolor='black', linewidth=2)
    ax_inset.add_patch(rect)
    
    output_path2 = OUTPUT_DIR / 'xray_map_with_showcase_inset.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path2}")
    plt.close()


if __name__ == '__main__':
    combine_images()
