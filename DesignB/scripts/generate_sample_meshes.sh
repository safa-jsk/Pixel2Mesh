#!/bin/bash
# Generate sample meshes from each ShapeNet category for poster

# Ensure we're in the project root
if [ ! -d "datasets/data/shapenet" ]; then
    echo "ERROR: Must run from project root directory"
    echo "Usage: ./scripts/evaluation/generate_sample_meshes.sh"
    exit 1
fi

echo "Creating sample images directory..."
mkdir -p datasets/examples_for_poster

# ShapeNet categories (13 classes)
declare -A categories=(
    ["02691156"]="airplane"
    ["02828884"]="bench"
    ["02933112"]="cabinet"
    ["02958343"]="car"
    ["03001627"]="chair"
    ["03211117"]="display"
    ["03636649"]="lamp"
    ["03691459"]="loudspeaker"
    ["04090263"]="rifle"
    ["04256520"]="sofa"
    ["04379243"]="table"
    ["04401088"]="telephone"
    ["04530566"]="watercraft"
)

# Collect 2 sample images from each category
for cat_id in "${!categories[@]}"; do
    cat_name="${categories[$cat_id]}"
    echo "Collecting samples for $cat_name ($cat_id)..."
    
    # Find first 2 objects in this category
    find datasets/data/shapenet/data_tf/$cat_id -name "00.png" | head -2 | while read img; do
        # Extract object ID from path
        obj_id=$(echo $img | cut -d'/' -f6)
        # Copy image with descriptive name
        cp "$img" "datasets/examples_for_poster/${cat_name}_${obj_id}.png"
        echo "  Added: ${cat_name}_${obj_id}.png"
    done
done

echo ""
echo "Sample images collected: $(ls datasets/examples_for_poster/*.png | wc -l)"
echo ""
echo "Now run prediction in Docker to generate meshes..."
