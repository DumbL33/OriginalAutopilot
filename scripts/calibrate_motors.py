from Rosmaster_Lib import Rosmaster
import time

car = Rosmaster(debug=True)
#car.create_receive_threading()
time.sleep(0.5)

car.set_motor(0, 50, 0, 50)
time.sleep(10)

car.set_car_motion(0, 0, 0)



# Set motors to exact same power (no PID compensation)
# set_motor(speed_1, speed_2, speed_3, speed_4)
# Values: -100 to 100

# Example: All wheels at 50% power
#car.set_motor(50, 50, 50, 50)
#time.sleep(3)
#car.set_motor(0, 0, 0, 0)  # Stop

# For turning left with equal power on both sides
# (this might not turn well, but both sides will have equal power)
#car.set_motor(30, 30, -30, -30)  # Left wheels forward, right wheels backward
##time.sleep(2)
#car.set_motor(0, 0, 0, 0)