#!/bin/zsh

# 1. Path to your venv
VENV_PATH=".venv/bin/activate"

# 2. Activate venv
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
    echo "Using virtual environment: $VENV_PATH"
else
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# 3. macOS Fix: Ensure we ARE using the real video driver
# We explicitly unset the dummy driver in case it was set in your current session
unset SDL_VIDEODRIVER

echo "Starting visible experiment batch..."
echo "TIP: Do not click away from the Pygame windows once they start!"

# 4. Loop through configs
find configs -name "*.yaml" | while read config; do
    echo "------------------------------------------------"
    echo "Running config: $config"
    
    # IMPORTANT: We REMOVED --no-eval-display to make it visible
    python run.py --config "$config"
    
    echo "Completed: $config"
    
    # 5. Critical for Mac: Wait 3 seconds to let the window fully销毁
    # This prevents the 'Black Window' hang on the next run
    sleep 3
done

echo "------------------------------------------------"
echo "All experiments finished!"