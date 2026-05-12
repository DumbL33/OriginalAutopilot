import socket
import threading

def listen_udp(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', port))
    print(f"Listening on UDP port {port}")
    
    while True:
        data, addr = sock.recvfrom(1024)
        print(f"Port {port} received from {addr}: {data.decode()}")
        print("")

# Create threads for each port
ports = [50001, 50002, 50003]
threads = []

for port in ports:
    thread = threading.Thread(target=listen_udp, args=(port,)) # Thread will exit when main program exits
    thread.start()
    threads.append(thread)

print("Listening on ports 50001, 50002, and 50003...")
print("Press Ctrl+C to stop")

# Keep the main thread alive
try:
    for thread in threads:
        thread.join()
except KeyboardInterrupt:
    print("\nStopping listeners...")