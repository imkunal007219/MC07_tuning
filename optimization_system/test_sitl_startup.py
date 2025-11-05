#!/usr/bin/env python3
"""
Simple SITL Startup Script for 30kg Drone
Tests if SITL can start properly and accept connections
"""

import subprocess
import time
import os
import signal
import sys
from pymavlink import mavutil

# Configuration
ARDUPILOT_PATH = os.path.expanduser("~/Documents/MC07_tuning/ardupilot")
FRAME_TYPE = "drone-30kg"
INSTANCE_ID = 0
# When using MAVProxy (default sim_vehicle.py behavior):
# - SITL runs on port 5760 (MAVProxy connects here)
# - MAVProxy outputs on port 14550 (we connect here)
# When using --no-mavproxy:
# - SITL runs on port 5760 (we connect directly here)
USE_MAVPROXY = True  # Set to False to connect directly to SITL
if USE_MAVPROXY:
    SITL_PORT = 14550 + (INSTANCE_ID * 10)  # MAVProxy output port
else:
    SITL_PORT = 5760 + (INSTANCE_ID * 10)   # Direct SITL port
SPEEDUP = 1

def cleanup_old_processes():
    """Kill any existing SITL processes"""
    print("Cleaning up old SITL processes...")
    os.system("pkill -9 arducopter 2>/dev/null")
    os.system("pkill -9 sim_vehicle 2>/dev/null")
    time.sleep(2)
    print("✓ Cleanup complete")

def start_sitl():
    """Start SITL with minimal options"""
    print(f"\nStarting SITL for {FRAME_TYPE}...")
    print(f"  ArduPilot path: {ARDUPILOT_PATH}")
    print(f"  Instance ID: {INSTANCE_ID}")
    print(f"  SITL port: {SITL_PORT}")
    print(f"  Speedup: {SPEEDUP}")

    # Change to ArduCopter directory
    work_dir = os.path.join(ARDUPILOT_PATH, "ArduCopter")
    sim_vehicle_path = os.path.join(ARDUPILOT_PATH, "Tools/autotest/sim_vehicle.py")


    # Build command - simple version that works
    cmd = [
        "python3",
        sim_vehicle_path,
        "-v", "ArduCopter",
        "-f", FRAME_TYPE,
#        "--no-rebuild",
#        "--no-mavproxy",
        "--console",
        "-I", str(INSTANCE_ID),
        "--speedup", str(SPEEDUP),
    ]

    #print(f"\nCommand: {' '.join(cmd)}\n")
    print("../Tools/autotest/sim_vehicle.py -v ArduCopter -f drone-30kg")

    # Start SITL process
    # Don't capture stdout/stderr so we can see MAVProxy console
    try:
        process = subprocess.Popen(
            cmd,
            cwd=work_dir,
            preexec_fn=os.setsid
        )

        print(f"✓ SITL process started (PID: {process.pid})")
        print(f"✓ MAVProxy console should appear in a separate window")
        return process

    except Exception as e:
        print(f"✗ Failed to start SITL: {e}")
        return None

def wait_for_sitl(port, timeout=90):
    """Wait for SITL to be ready and accepting connections"""
    print(f"\nWaiting for SITL to be ready on port {port}...")
    print(f"  (Giving SITL 15 seconds to boot up...)")

    # Give SITL time to start before attempting connection
    time.sleep(15)

    # MAVProxy --out parameter uses UDP, not TCP!
    connection_string = f"udp:127.0.0.1:{port}"
    print(f"  Connecting via UDP to {connection_string}")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            elapsed = int(time.time() - start_time)
            print(f"  Attempt connection ({elapsed}s)...", end='\r')

            # Try to connect
            conn = mavutil.mavlink_connection(connection_string, timeout=2)

            # Wait for heartbeat
            msg = conn.wait_heartbeat(timeout=3)

            if msg:
                print(f"\n✓ Heartbeat received after {elapsed}s!")
                print(f"  System ID: {msg.get_srcSystem()}")
                print(f"  Component ID: {msg.get_srcComponent()}")
                print(f"  Type: {msg.type}")
                return conn

        except Exception as e:
            # Connection failed, retry
            time.sleep(1)
            continue

    print(f"\n✗ Timeout after {timeout}s waiting for SITL")
    return None

def test_position_data(conn, timeout=30):
    """Test if we can get position data (EKF initialized)"""
    print(f"\nTesting position data (EKF initialization)...")

    start_time = time.time()
    while time.time() - start_time < timeout:
        # Try to get position
        msg = conn.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)

        if msg:
            lat = msg.lat / 1e7
            lon = msg.lon / 1e7
            alt = msg.relative_alt / 1000.0

            print(f"✓ Position data received!")
            print(f"  Latitude: {lat:.6f}")
            print(f"  Longitude: {lon:.6f}")
            print(f"  Altitude: {alt:.2f} m")
            return True

        elapsed = int(time.time() - start_time)
        print(f"  Waiting for position data ({elapsed}s)...", end='\r')
        time.sleep(0.5)

    print(f"\n✗ No position data after {timeout}s")
    return False

def test_ekf_status(conn):
    """Check EKF status"""
    print(f"\nChecking EKF status...")

    for i in range(20):
        # Request EKF status
        msg = conn.recv_match(type='EKF_STATUS_REPORT', blocking=True, timeout=1)

        if msg:
            print(f"✓ EKF Status Report received!")
            print(f"  Flags: {msg.flags}")
            print(f"  Velocity variance: {msg.velocity_variance:.4f}")
            print(f"  Position variance: {msg.pos_horiz_variance:.4f}")
            print(f"  Compass variance: {msg.compass_variance:.4f}")
            return True

        time.sleep(0.5)

    print(f"✗ No EKF status report received")
    return False

def main():
    """Main test sequence"""
    print("="*60)
    print("SITL Startup Test for 30kg Drone")
    print("="*60)

    # Step 1: Cleanup
    cleanup_old_processes()

    # Step 2: Start SITL
    process = start_sitl()
    if not process:
        print("\n✗ Failed to start SITL")
        return 1

    try:
        # Step 3: Wait for connection
        conn = wait_for_sitl(SITL_PORT)
        if not conn:
            print("\n✗ Failed to connect to SITL")
            return 1

        # Step 4: Test position data
        if not test_position_data(conn):
            print("\n⚠ Warning: No position data (EKF might not be initialized)")

        # Step 5: Test EKF status
        test_ekf_status(conn)

        print("\n" + "="*60)
        print("✓ SITL STARTUP TEST SUCCESSFUL!")
        print("="*60)
        print("\nSITL is running and ready for connections.")
        print(f"Connection string: tcp:127.0.0.1:{SITL_PORT}")
        print("\nPress Ctrl+C to stop SITL...")

        # Keep running until interrupted
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nShutting down...")

    finally:
        # Cleanup
        if process:
            print("Stopping SITL process...")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
            except:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            print("✓ SITL stopped")

    return 0

if __name__ == "__main__":
    sys.exit(main())
