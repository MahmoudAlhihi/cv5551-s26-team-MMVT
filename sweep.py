# neeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed python sweep.py --port /dev/ttyACM0

from xarm.wrapper import XArmAPI
from read_mic import find_arduino_port, serial_reader, latest_values, data_lock
import threading
import time
import matplotlib.pyplot as plt
import numpy as np
from audio_localize import localize_source

def collect_mic_data(duration=2.0, port=None, baud=115200, window=500):
    port = port or find_arduino_port()
    t = threading.Thread(target=serial_reader, args=(port, baud, window), daemon=True)
    t.start()
    time.sleep(duration)
    with data_lock:
        return list(latest_values)

def plot_mic_data(data):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data, color="#00e5ff", linewidth=1.0)
    ax.fill_between(range(len(data)), data, alpha=0.18, color="#00e5ff")
    ax.set_xlabel("Sample")
    ax.set_ylabel("2.5 kHz Magnitude")
    ax.set_title(f"Recorded Signal  |  {len(data)} samples  |  peak: {max(data):.1f}")
    fig.tight_layout()
    plt.show()

GRIPPER_LENGTH = 0.067 * 1000

ROBOT_IP = '192.168.1.182'

# Table/sheet bounds in mm (robot frame)
X_MIN, X_MAX = 110.0, 320.0
Y_MIN, Y_MAX = -275.0, 406.0

SWEEP_Z = 150.0 # fixed height above table (mm)
SWEEP_SPEED = 100 # mm/s
TRAVEL_SPEED = 100 # mm/s
ROW_STEP = 40.0 # spacing between parallel sweep passes (mm)

# def sweep_table(arm: XArmAPI, port=None) -> None:
#     # Move to start position (wait for this one)
#     arm.set_position(
#         x=X_MAX, y=Y_MIN, z=SWEEP_Z,
#         roll=180, pitch=0, yaw=0,
#         wait=True, speed=TRAVEL_SPEED, mvacc=500
#     )

#     # Start the sweep move WITHOUT waiting
#     arm.set_position(
#         x=X_MIN, y=Y_MIN, z=SWEEP_Z,
#         roll=180, pitch=0, yaw=0,
#         wait=False, speed=SWEEP_SPEED, mvacc=500
#     )

#     arm.set_position(
#         x=X_MIN, y=Y_MAX, z=SWEEP_Z,
#         roll=180, pitch=0, yaw=0,
#         wait=True, speed=TRAVEL_SPEED, mvacc=500
#     )

#     # Record mic while the arm is moving
#     move_duration = (X_MAX - X_MIN) / SWEEP_SPEED  # ~2.1s
#     data = collect_mic_data(duration=move_duration + 1.0, port=port)

#     # Wait for arm to finish just in case
#     time.sleep(1.0)

#     plot_mic_data(data)

#     print("Sweep complete.")

def sweep_table(arm: XArmAPI, port=None) -> None:
    # Move to start position
    arm.set_position(
        x=X_MAX, y=Y_MIN, z=SWEEP_Z,
        roll=180, pitch=0, yaw=0,
        wait=True, speed=TRAVEL_SPEED, mvacc=500
    )

    # Sweep X_MAX -> X_MIN along Y_MIN (record while moving)
    arm.set_position(
        x=X_MIN, y=Y_MIN, z=SWEEP_Z,
        roll=180, pitch=0, yaw=0,
        wait=False, speed=SWEEP_SPEED, mvacc=500
    )
    x_duration = (X_MAX - X_MIN) / SWEEP_SPEED
    data_x = collect_mic_data(duration=x_duration + 1.0, port=port)
    x_coord, smooth_x = localize_source(data_x, start_coord=X_MAX, end_coord=X_MIN)
    time.sleep(1.0)

    # Clear the shared deque for the next recording
    with data_lock:
        latest_values.clear()

    # Sweep X_MIN,Y_MIN -> X_MIN,Y_MAX (record while moving)
    arm.set_position(
        x=X_MIN, y=Y_MAX, z=SWEEP_Z,
        roll=180, pitch=0, yaw=0,
        wait=False, speed=SWEEP_SPEED, mvacc=500
    )
    y_duration = (Y_MAX - Y_MIN) / SWEEP_SPEED
    data_y = collect_mic_data(duration=y_duration + 1.0, port=port)
    y_coord, smooth_y = localize_source(data_y, start_coord=Y_MIN, end_coord=Y_MAX)
    time.sleep(1.0)

    print(f"Sound source estimated at: X={x_coord:.1f} mm, Y={y_coord:.1f} mm")

    plt.style.use('dark_background')
    # Plot both sweeps
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # X sweep
    ax1.plot(data_x, color="#00e5ff", linewidth=0.5, alpha=0.4, label="Raw")
    ax1.plot(smooth_x, color="#ffffff", linewidth=2.0, label="Smoothed")
    peak_x_idx = int(np.argmax(smooth_x))
    ax1.axvline(peak_x_idx, color="#ff4081", linestyle="--", label=f"Peak → X={x_coord:.1f} mm")
    ax1.set_ylabel("2.5 kHz Magnitude")
    ax1.set_title(f"X Sweep (X_MAX→X_MIN)")
    ax1.legend()

    # Y sweep
    ax2.plot(data_y, color="#ff6e40", linewidth=0.5, alpha=0.4, label="Raw")
    ax2.plot(smooth_y, color="#ffffff", linewidth=2.0, label="Smoothed")
    peak_y_idx = int(np.argmax(smooth_y))
    ax2.axvline(peak_y_idx, color="#ff4081", linestyle="--", label=f"Peak → Y={y_coord:.1f} mm")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("2.5 kHz Magnitude")
    ax2.set_title(f"Y Sweep (Y_MIN→Y_MAX)")
    ax2.legend()

    fig.tight_layout()
    plt.show()

    print("Sweep complete.")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None)
    args = parser.parse_args()

    arm = XArmAPI(ROBOT_IP)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)

    try:
        sweep_table(arm, port=args.port)
        arm.move_gohome(wait=True)
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()