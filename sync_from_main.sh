#!/bin/bash
# Sync script to copy changes from main Groundeep repo to this standalone repo

MAIN_REPO="/home/student/Desktop/Groundeep"
UNIMODAL_REPO="/home/student/Desktop/groundeep-unimodal-training"

echo "🔄 Syncing files from main repo to unimodal repo..."

# Copy core files
cp "$MAIN_REPO/src/classes/gdbn_model.py" "$UNIMODAL_REPO/src/classes/gdbn_model.py"
cp "$MAIN_REPO/src/main_scripts/train.py" "$UNIMODAL_REPO/src/main_scripts/train.py"
cp "$MAIN_REPO/src/datasets/uniform_dataset.py" "$UNIMODAL_REPO/src/datasets/uniform_dataset.py"
cp "$MAIN_REPO/src/utils/wandb_utils.py" "$UNIMODAL_REPO/src/utils/wandb_utils.py"
cp "$MAIN_REPO/src/utils/probe_utils.py" "$UNIMODAL_REPO/src/utils/probe_utils.py"
cp "$MAIN_REPO/src/configs/training_config.yaml" "$UNIMODAL_REPO/src/configs/training_config.yaml"

# Clean gdbn_model.py to remove multimodal code (iMDBN class)
# This requires manual intervention - the script only copies, you need to manually clean

echo "✅ Files copied!"
echo ""
echo "⚠️  IMPORTANT: You need to manually clean gdbn_model.py to remove:"
echo "   - iMDBN class (lines 692+)"
echo "   - energy_utils imports"
echo "   - conditional_steps imports"
echo "   - imdbn_logging imports"
echo ""
echo "📝 Next steps:"
echo "   1. Review changes: git status"
echo "   2. Commit: git add . && git commit -m 'Sync from main repo'"
echo "   3. Push: git push"
