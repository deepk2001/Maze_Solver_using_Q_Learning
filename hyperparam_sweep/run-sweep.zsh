#!/bin/zsh
#
# Hyperparameter sweep: 3 learning rates × 3 discount rates = 9 runs
# Results in: hyperparam_sweep/results/
#

# 1. Path to your venv (from project root)
VENV_PATH=".venv/bin/activate"

# 2. Activate venv
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
    echo "Using virtual environment: $VENV_PATH"
else
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# 3. Headless: dummy SDL driver to avoid black screen
export SDL_VIDEODRIVER=dummy

# Run from project root
cd "$(dirname "$0")/.."

RESULTS_DIR="hyperparam_sweep/results"
echo "Starting hyperparameter sweep (3×3 grid)..."
echo "Results: $RESULTS_DIR/"

# 4. Loop through configs
find hyperparam_sweep/configs -name "*.yaml" | while read config; do
    echo "------------------------------------------------"
    echo "Running config: $config"
    python run.py --config "$config" --results-dir "$RESULTS_DIR" --no-eval-display
    echo "Completed: $config"
    # 5. Critical for Mac: Wait 3 seconds to let the window fully close
    sleep 3
done

echo "------------------------------------------------"
echo "All experiments finished!"
echo "Run: python visualize_hyperparam_results.py"
