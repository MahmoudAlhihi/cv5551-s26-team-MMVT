from xarm.wrapper import XArmAPI

GRIPPER_LENGTH = 0.067 * 1000

ROBOT_IP = '192.168.1.182'

# Table/sheet bounds in mm (robot frame)
X_MIN, X_MAX = 200.0, 500.0
Y_MIN, Y_MAX = -200.0, 200.0

SWEEP_Z = 50.0 # fixed height above table (mm)
SWEEP_SPEED = 100 # mm/s
TRAVEL_SPEED = 200 # mm/s
ROW_STEP = 40.0 # spacing between parallel sweep passes (mm)


def sweep_table(arm: XArmAPI) -> None:
    """
    Boustrophedon (back-and-forth) sweep over the table at a fixed Z height
    Sweeps along X for each Y row, stepping in Y between rows
    """
    y = Y_MIN
    row = 0

    while y <= Y_MAX + 1e-6:
        # Alternate sweep direction each row
        x_start, x_end = (X_MIN, X_MAX) if row % 2 == 0 else (X_MAX, X_MIN)

        # Reposition above row start at travel speed
        arm.set_position(
            x=x_start, y=y, z=SWEEP_Z,
            roll=180, pitch=0, yaw=0,
            wait=True, speed=TRAVEL_SPEED, mvacc=500
        )

        # Sweep across the row
        arm.set_position(
            x=x_end, y=y, z=SWEEP_Z,
            roll=180, pitch=0, yaw=0,
            wait=True, speed=SWEEP_SPEED, mvacc=200
        )

        y   += ROW_STEP
        row += 1

    print("Sweep complete.")


def main():
    arm = XArmAPI(ROBOT_IP)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)

    try:
        sweep_table(arm)
        arm.move_gohome(wait=True, speed=TRAVEL_SPEED, mvacc=500)
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()