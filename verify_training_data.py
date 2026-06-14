import sys
import numpy as np
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_training_data.py <path_to_npz_file>")
        sys.exit(1)
        
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        sys.exit(1)
        
    print(f"Loading data from {filepath}...")
    try:
        data = np.load(filepath)
    except Exception as e:
        print(f"Error loading npz file: {e}")
        sys.exit(1)
        
    # Check keys
    required_keys = ['frames', 'sensors', 'actions', 'timestamps']
    for key in required_keys:
        if key not in data:
            print(f"Error: Required key '{key}' missing from npz file.")
            sys.exit(1)
            
    frames = data['frames']
    sensors = data['sensors']
    actions = data['actions']
    timestamps = data['timestamps']
    
    print("\n--- Array Shapes ---")
    print(f"frames:     {frames.shape} ({frames.dtype})")
    print(f"sensors:    {sensors.shape} ({sensors.dtype})")
    print(f"actions:    {actions.shape} ({actions.dtype})")
    print(f"timestamps: {timestamps.shape} ({timestamps.dtype})")
    
    n_samples = len(timestamps)
    if n_samples == 0:
        print("\nSession is empty (0 frames).")
        sys.exit(0)
        
    duration = timestamps[-1] - timestamps[0]
    print(f"\nDuration: {duration:.2f} seconds")
    print(f"Frames:   {n_samples}")
    print(f"Average Frequency: {n_samples / duration:.2f} Hz" if duration > 0 else "Frequency: N/A")
    
    print("\n--- Sensors (Min/Max values) ---")
    sensor_names = [
        "1. Position X (mm)",
        "2. Position Y (mm)",
        "3. Rotation Cap (deg)",
        "4. Pose Pitch (deg)",
        "5. Left Wheel Speed (mmps)",
        "6. Right Wheel Speed (mmps)",
        "7. Cliff Detected (0/1)",
        "8. Lift Height (mm)",
        "9. Head Angle (deg)",
        "10. Battery Voltage (V)",
        "11. Quaternion q0",
        "12. Moving Flag (0/1)"
    ]
    for i, name in enumerate(sensor_names):
        col = sensors[:, i]
        print(f"- {name:28}: min={col.min():8.2f}, max={col.max():8.2f}")
        
    print("\n--- Action Summary ---")
    # actions: left_wheel, right_wheel, head_angle
    left_actions = actions[:, 0]
    right_actions = actions[:, 1]
    
    fwd_count = 0
    bwd_count = 0
    rot_count = 0
    stop_count = 0
    
    for i in range(n_samples):
        left = left_actions[i]
        right = right_actions[i]
        
        if left == 0 and right == 0:
            stop_count += 1
        elif left > 0 and right > 0:
            fwd_count += 1
        elif left < 0 and right < 0:
            bwd_count += 1
        else: # left/right have different signs or one is zero (rotation)
            rot_count += 1
            
    print(f"Stop:     {stop_count:4d} frames ({stop_count / n_samples * 100:.1f}%)")
    print(f"Forward:  {fwd_count:4d} frames ({fwd_count / n_samples * 100:.1f}%)")
    print(f"Backward: {bwd_count:4d} frames ({bwd_count / n_samples * 100:.1f}%)")
    print(f"Rotation: {rot_count:4d} frames ({rot_count / n_samples * 100:.1f}%)")

if __name__ == "__main__":
    main()
