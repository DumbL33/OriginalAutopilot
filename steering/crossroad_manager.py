import time


class CrossroadManager:
    
    def __init__(self, stop_duration=5.0, command_reset_duration=3.0, cooldown_duration=3.0):
        
        self.stop_duration = stop_duration
        self.command_reset_duration = command_reset_duration  # When to clear the sign command
        self.cooldown_duration = cooldown_duration
        
        self.on_crossroad = False
        self.was_on_crossroad = False
        self.stop_processed = False
        self.stop_start_time = None
        self.cooldown_start_time = None
        self.in_cooldown = False
        self.stopping = False
        
        # Track when we started processing the stop command
        self.command_received_time = None
        
        self.crossroads_detected = 0
        self.stops_executed = 0
        
        print(f"✓ Crossroad Manager initialized")
        print(f"  Stop duration: {stop_duration}s")
        print(f"  Command reset after: {command_reset_duration}s")
        print(f"  Cooldown duration: {cooldown_duration}s")
    
    def update(self, is_crossroad, sign_command=None):
        current_time = time.time()
        should_stop = False
        ignore_stop_sign = False
        updated_sign_command = sign_command
        
        # Check cooldown
        if self.in_cooldown:
            time_since_cooldown = current_time - self.cooldown_start_time
            if time_since_cooldown >= self.cooldown_duration:
                self.in_cooldown = False
                print("✓ Cooldown period ended, ready for next crossroad")
        
        # If we're currently stopping, maintain the stop for full duration
        if self.stopping and self.stop_start_time is not None:
            time_stopped = current_time - self.stop_start_time
            
            if time_stopped < self.stop_duration:
                should_stop = True
                remaining = self.stop_duration - time_stopped
                
                # Reset sign_command after command_reset_duration (3 seconds)
                if time_stopped >= self.command_reset_duration and updated_sign_command == 'stop':
                    updated_sign_command = None
                    print(f"✓ Sign command reset after {self.command_reset_duration}s")
                
                # Print progress every 0.5s
                if int(time_stopped * 2) != int((time_stopped - 0.1) * 2):
                    print(f"   ⏸️  Stopping... {remaining:.1f}s remaining")
            else:
                # Stop duration complete
                print("✓ Stop duration complete, releasing")
                self.stop_start_time = None
                self.stopping = False
                self.stop_processed = False  # Reset for next crossroad
                self.in_cooldown = True
                self.cooldown_start_time = current_time
                updated_sign_command = None  # Ensure command is cleared
                print(f"   Starting {self.cooldown_duration}s cooldown period")
            
            return should_stop, ignore_stop_sign, updated_sign_command
        
        # Detect crossroad and check for stop command
        if is_crossroad and not self.in_cooldown:
            # Track crossroad entry
            if not self.was_on_crossroad:
                self.on_crossroad = True
                self.crossroads_detected += 1
                print(f"🚦 CROSSROAD DETECTED (#{self.crossroads_detected})")
            
            # Check if we should stop (crossroad + stop command + not already processed)
            if sign_command == 'stop' and not self.stop_processed and not self.stopping:
                should_stop = True
                self.stopping = True
                self.stop_processed = True
                self.stop_start_time = current_time
                self.command_received_time = current_time
                self.stops_executed += 1
                print(f"🛑 STOPPING AT CROSSROAD for {self.stop_duration}s")
                print(f"   Sign command will reset after {self.command_reset_duration}s")
                print(f"   Total stops executed: {self.stops_executed}")
        
        # Track crossroad exit (but don't reset anything based on this)
        if not is_crossroad and self.was_on_crossroad and not self.stopping:
            print("🚦 Exited crossroad area")
            self.on_crossroad = False
            # Don't reset stop_processed here - let it reset after stop completes
        
        self.was_on_crossroad = is_crossroad
        
        return should_stop, ignore_stop_sign, updated_sign_command
    
    def is_stopping(self):
        return self.stopping
    
    def get_state_info(self):
        
        return {
            'on_crossroad': self.on_crossroad,
            'stop_processed': self.stop_processed,
            'in_cooldown': self.in_cooldown,
            'stopping': self.stopping,
            'crossroads_detected': self.crossroads_detected,
            'stops_executed': self.stops_executed
        }
    
    def reset(self):
        self.on_crossroad = False
        self.was_on_crossroad = False
        self.stop_processed = False
        self.stop_start_time = None
        self.cooldown_start_time = None
        self.command_received_time = None
        self.in_cooldown = False
        self.stopping = False
        print("Crossroad manager state reset")