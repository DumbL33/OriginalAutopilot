from Rosmaster_Lib import Rosmaster

#rm = Rosmaster(car_type=1, com='/dev/myserial')
#rm.create_receive_threading()
#rm.set_auto_report_state(True)
#voltage = rm.get_battery_voltage()

def volt_to_perc(v):
    v_min, v_max = 9.6, 12.6
    return max(0, min(100, (v-v_min) / (v_max - v_min) * 100))

#percent = volt_to_perc(voltage)

#print(percent)

rm = Rosmaster()

rm.create_receive_threading()

print("Reported:", rm.get_battery_voltage())
