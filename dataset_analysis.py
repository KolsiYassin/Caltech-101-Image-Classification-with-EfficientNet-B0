from torchvision.datasets import Caltech101
from collections import Counter
import matplotlib.pyplot as plt
import random
import numpy as np

from pathlib import Path

# Create directory for plots
plots_dir = Path("./plots")
plots_dir.mkdir(parents=True, exist_ok=True)


# import dataset
dataset = Caltech101(
    root="./data",
    download=True
)

# first glimpse 
print(f"Number of samples: {len(dataset)}") #8677
print(f"Number of classes: {len(dataset.categories)}") #101
print(f"First 10 classes: {dataset.categories[:10]}")


# inspect one sample
image, label = dataset[0]

print(f"Image type: {type(image)}") # PIL.JpegImagePlugin
print(f"Image size: {image.size}") # (510, 337)
print(f"Label: {label}")
print(f"Class name: {dataset.categories[label]}")

# understand dataset structure
print(dataset)
print(dataset.categories[:20])
print(dataset.annotation_categories)

print(dataset.target_type)

# data distibution 

labels = [dataset[i][1] for i in range(len(dataset))]
class_counts = Counter(labels)

for class_id, count in sorted(class_counts.items()):
    print(f"{class_id:3d} | {dataset.categories[class_id]:20s} | {count}")


counts = np.array(list(class_counts.values()))

print(f"Minimum images per class: {counts.min()}")
print(f"Maximum images per class: {counts.max()}")
print(f"Mean images per class:    {counts.mean():.2f}")
print(f"Median images per class:  {np.median(counts):.2f}")

# visualize dataset distribution

class_names = [dataset.categories[i] for i in range(len(dataset.categories))]
class_counts_sorted = [class_counts[i] for i in range(len(dataset.categories))]

### Normal plot
plt.figure(figsize=(20, 6)) 
plt.bar(class_names, class_counts_sorted) 
plt.xlabel("Class") 
plt.ylabel("Number of images") 
plt.title("Caltech-101 Class Distribution") 
plt.xticks(rotation=90) 
plt.tight_layout() 
# Save plot 
plt.savefig(plots_dir / "class_distribution.png", dpi=300, bbox_inches="tight")

### Sorted Horizontal plot
# Sort classes by number of images
sorted_data = sorted(
    zip(class_names, class_counts_sorted),
    key=lambda x: x[1]
)

class_names_sorted = [x[0] for x in sorted_data]
counts_sorted = [x[1] for x in sorted_data]

plt.figure(figsize=(14, 30))

plt.barh(class_names_sorted, counts_sorted)

plt.xlabel("Number of images")
plt.ylabel("Class")
plt.title("Caltech-101 Class Distribution")

plt.yticks(fontsize=7)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    plots_dir / "class_distribution_horizontal.png",
    dpi=300,
    bbox_inches="tight"
)

### Sorted Vertical bar chart
plt.figure(figsize=(30, 10))

plt.bar(class_names_sorted, counts_sorted)

plt.xlabel("Class")
plt.ylabel("Number of images")
plt.title("Caltech-101 Class Distribution")

plt.xticks(
    rotation=90,
    fontsize=7
)

plt.tight_layout()

plt.savefig(
    plots_dir / "class_distribution_vertical.png",
    dpi=300,
    bbox_inches="tight"
)

# Ispect images
fig, axes = plt.subplots(3, 5, figsize=(15, 9))

for ax in axes.flat:
    index = random.randint(0, len(dataset) - 1)
    
    image, label = dataset[index]
    
    ax.imshow(image)
    ax.set_title(dataset.categories[label])
    ax.axis("off")

plt.tight_layout()
# Save plot
plt.savefig(
    plots_dir / "random_samples.png",
    dpi=300,
    bbox_inches="tight"
)

# images dimenstions
sizes = []

for i in range(len(dataset)):
    image, _ = dataset[i]
    sizes.append(image.size)

size_counts = Counter(sizes)

print(f"Number of unique image sizes: {len(size_counts)}")

print("\nMost common image sizes:")
for size, count in size_counts.most_common(10):
    print(f"{size}: {count} images")

widths = [size[0] for size in sizes]
heights = [size[1] for size in sizes]

mean_width = np.mean(widths)
mean_height = np.mean(heights)

# Width and height distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Width
axes[0].hist(widths, bins=30)
axes[0].axvline(
    mean_width,
    linestyle="--",
    label=f"Mean = {mean_width:.1f}px"
)
axes[0].set_xlabel("Width (pixels)")
axes[0].set_ylabel("Number of images")
axes[0].set_title("Distribution of Image Widths")
axes[0].legend()

# Height
axes[1].hist(heights, bins=30)
axes[1].axvline(
    mean_height,
    linestyle="--",
    label=f"Mean = {mean_height:.1f}px"
)
axes[1].set_xlabel("Height (pixels)")
axes[1].set_ylabel("Number of images")
axes[1].set_title("Distribution of Image Heights")
axes[1].legend()

plt.tight_layout()

plt.savefig(
    plots_dir / "image_size_distributions.png",
    dpi=300,
    bbox_inches="tight"
)


print(f"Width  - min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.2f}")
print(f"Height - min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.2f}")

#images mode
modes = Counter()

for i in range(len(dataset)):
    image, _ = dataset[i]
    modes[image.mode] += 1

print(modes)

# Visualize image modes

mode_names = list(modes.keys())
mode_counts = list(modes.values())

plt.figure(figsize=(7, 5))

plt.bar(mode_names, mode_counts)

plt.xlabel("Image mode")
plt.ylabel("Number of images")
plt.title("Caltech-101 Image Color Modes")

plt.tight_layout()

plt.savefig(
    plots_dir / "image_modes.png",
    dpi=300,
    bbox_inches="tight"
)
