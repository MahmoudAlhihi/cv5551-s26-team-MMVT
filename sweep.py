from xarm.wrapper import XArmAPI

GRIPPER_LENGTH = 0.067 * 1000

ROBOT_IP = '192.168.1.182'

# Table/sheet bounds in mm (robot frame)
X_MIN, X_MAX = 110.0, 320.0
Y_MIN, Y_MAX = -275.0, 406.0

SWEEP_Z = 150.0 # fixed height above table (mm)
SWEEP_SPEED = 100 # mm/s
TRAVEL_SPEED = 100 # mm/s
ROW_STEP = 40.0 # spacing between parallel sweep passes (mm)


def sweep_table(arm: XArmAPI) -> None:
    

    # Reposition above row start at travel speed
    arm.set_position(
        x=X_MAX, y=Y_MIN, z=SWEEP_Z,
        roll=180, pitch=0, yaw=0,
        wait=True, speed=TRAVEL_SPEED, mvacc=500
    )

    # Reposition above row start at travel speed
    arm.set_position(
        x=X_MIN, y=Y_MIN, z=SWEEP_Z,
        roll=180, pitch=0, yaw=0,
        wait=True, speed=TRAVEL_SPEED, mvacc=500
    )

    # Reposition above row start at travel speed
    arm.set_position(
        x=X_MIN, y=Y_MAX, z=SWEEP_Z,
        roll=180, pitch=0, yaw=0,
        wait=True, speed=TRAVEL_SPEED, mvacc=500
    )


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
        arm.move_gohome(wait=True)
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()