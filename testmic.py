#!/usr/bin/env python3
"""
test_microphone.py

Quick diagnostic to check if the microphone rig is reading data.
"""

import time
import serial

# Configuration
MIC_PORT = "/dev/ttyACM1"
BAUD_RATE = 9600
TEST_DURATION = 5  # seconds

def test_microphone():
    print("=" * 60)
    print("MICROPHONE TEST")
    print("=" * 60)
    print(f"Port: {MIC_PORT}")
    print(f"Baud rate: {BAUD_RATE}")
    print(f"Test duration: {TEST_DURATION} seconds")
    print()
    
    try:
        # Open serial connection
        print("Opening serial port...")
        ser = serial.Serial(MIC_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset
        print("✓ Port opened successfully\n")
        
        # Flush any old data
        ser.reset_input_buffer()
        
        print(f"Reading data for {TEST_DURATION} seconds...")
        print("(Make some noise near the microphones!)")
        print("-" * 60)
        
        start_time = time.time()
        line_count = 0
        error_count = 0
        
        while (time.time() - start_time) < TEST_DURATION:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    line_count += 1
                    
                    # Parse the data
                    parts = line.split(',')
                    if len(parts) >= 2:
                        mic1 = parts[0].strip()
                        mic2 = parts[1].strip()
                        print(f"[{line_count:3d}] Mic1: {mic1:>6s}  Mic2: {mic2:>6s}  Raw: {line}")
                    else:
                        print(f"[{line_count:3d}] Unexpected format: {line}")
                        error_count += 1
                        
                except Exception as e:
                    error_count += 1
                    print(f"[ERROR] Failed to parse line: {e}")
            
            time.sleep(0.01)  # Small delay
        
        ser.close()
        
        print("-" * 60)
        print()
        print("=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"Total lines read: {line_count}")
        print(f"Parse errors: {error_count}")
        print(f"Data rate: {line_count / TEST_DURATION:.1f} lines/sec")
        
        if line_count == 0:
            print()
            print("✗ FAILED - No data received from microphone!")
            print("  Possible issues:")
            print("  - Arduino not connected")
            print("  - Wrong port (check /dev/ttyACM* or /dev/ttyUSB*)")
            print("  - Arduino not programmed with microphone sketch")
            print("  - Wrong baud rate")
        elif error_count > line_count * 0.5:
            print()
            print("⚠ WARNING - Many parse errors!")
            print("  Check Arduino sketch output format")
        else:
            print()
            print("✓ SUCCESS - Microphone is reading data!")
        
        print("=" * 60)
        
    except serial.SerialException as e:
        print(f"\n✗ FAILED - Cannot open serial port: {e}")
        print("\nTroubleshooting:")
        print("1. Check if device exists:")
        print(f"   ls -l {MIC_PORT}")
        print("2. Check available serial ports:")
        print("   ls -l /dev/ttyACM* /dev/ttyUSB*")
        print("3. Check permissions:")
        print(f"   sudo chmod 666 {MIC_PORT}")
        print("4. Check if another process is using it:")
        print("   lsof | grep ttyACM")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    test_microphone()