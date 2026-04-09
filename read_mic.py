"""
Reads 2.5kHz Goertzel magnitude values from an Arduino Uno over USB serial
and displays a live scrolling graph of the signal intensity.

Requirements:
    pip install pyserial matplotlib

Usage:
    python read_mic.py                  # auto-detect port
    python read_mic.py --port COM3      # specify port explicitly
    python read_mic.py --port /dev/ttyUSB0 --baud 115200
    python read_mic.py --window 200     # number of samples shown at once (default: 100)
    neeeeeeeed python read_mic.py --port /dev/ttyACM0
"""

import argparse
import collections
import threading
import serial
import serial.tools.list_ports
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Shared state between serial thread and plot
data_lock = threading.Lock()
latest_values = collections.deque()
sample_count = 0
arduino_info = ""


def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = (p.description or "").lower()
        if "arduino" in desc or "ch340" in desc or "usb serial" in desc:
            return p.device
    if ports:
        return ports[0].device
    return None


def serial_reader(port, baud, window):
    """Background thread: reads serial lines and appends magnitudes to the deque."""
    global sample_count, arduino_info
    try:
        ser = serial.Serial(port, baud, timeout=2)
    except serial.SerialException as e:
        print(f"ERROR: Could not open port: {e}")
        sys.exit(1)

    time.sleep(2)
    ser.reset_input_buffer()

    try:
        while True:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("#"):
                arduino_info = line[1:].strip()
                continue
            try:
                magnitude = float(line)
                with data_lock:
                    latest_values.append(magnitude)
                    if len(latest_values) > window:
                        latest_values.popleft()
                    sample_count += 1
            except ValueError:
                pass
    except Exception:
        pass
    finally:
        ser.close()


def main():
    parser = argparse.ArgumentParser(description="Live graph of 2.5kHz intensity from Arduino")
    parser.add_argument("--port",   default=None,    help="Serial port (e.g. COM3)")
    parser.add_argument("--baud",   type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--window", type=int, default=100,    help="Samples visible on graph (default: 100)")
    args = parser.parse_args()

    port = args.port or find_arduino_port()
    if port is None:
        print("ERROR: No serial port found. Plug in the Arduino or use --port <port>.")
        sys.exit(1)

    print(f"Connecting to {port} at {args.baud} baud ...")

    # Start serial reading in background thread
    t = threading.Thread(target=serial_reader, args=(port, args.baud, args.window), daemon=True)
    t.start()

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#0e0e0e")

    line_plot, = ax.plot([], [], color="#00e5ff", linewidth=1.5)
    fill = ax.fill_between([], [], alpha=0.25, color="#00e5ff")

    ax.set_xlim(0, args.window)
    ax.set_ylim(0, 1)          # auto-scales once data arrives
    ax.set_xlabel("Sample", color="#aaaaaa")
    ax.set_ylabel("2.5 kHz Magnitude", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    title = ax.set_title("Waiting for Arduino...", color="#ffffff", pad=10)
    fig.tight_layout()

    def update(_frame):
        nonlocal fill
        with data_lock:
            values = list(latest_values)
            count  = sample_count
            info   = arduino_info

        if not values:
            return line_plot,

        xs = list(range(len(values)))
        line_plot.set_data(xs, values)

        # Re-draw fill under the line
        fill.remove()
        fill = ax.fill_between(xs, values, alpha=0.18, color="#00e5ff")

        # Auto-scale Y with a bit of headroom
        peak = max(values)
        ax.set_ylim(0, max(peak * 1.2, 10))
        ax.set_xlim(0, args.window)

        lbl = info if info else f"port: {port}"
        title.set_text(
            f"2.5 kHz Signal Intensity  |  sample #{count}  |  {lbl}\n"
            f"current: {values[-1]:.1f}   peak: {peak:.1f}"
        )
        return line_plot,

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)

    plt.show()


if __name__ == "__main__":
    main()
