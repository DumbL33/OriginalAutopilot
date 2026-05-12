from Rosmaster_Lib import Rosmaster
import time

car = Rosmaster(debug=True)

time.sleep(0.5)

pid = car.get_motion_pid()
print("Current PID:", pid) 


time.sleep(0.5)

car.set_pid_param(kp=0.5, ki=0.1, kd=0.3, forever=False)
time.sleep(0.5)

print("\n=== Reading PID values back ===")
pid = car.get_motion_pid()
print("Current PID:", pid)