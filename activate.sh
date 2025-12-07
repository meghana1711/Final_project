#!/bin/bash
# Save this as: /vast.mnt/home/20236193/projects/Final_project/activate.sh
# Usage: source activate.sh

echo "🔧 Loading modules..."
module purge
module load Python/3.12.3-GCCcore-13.3.0  

echo "Activating virtual environment..."
cd /vast.mnt/home/20236193/projects/Final_project
source venv/bin/activate

echo "✅ Environment ready!"
echo "Python: $(python --version)"
echo "Location: $(which python)"
echo ""
echo "To deactivate: type 'deactivate'"