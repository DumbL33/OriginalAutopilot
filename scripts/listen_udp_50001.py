import socket

def listen_udp(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', port))
    print(f"Listening on UDP port {port}")
    
    while True:
        data, addr = sock.recvfrom(1024)
        print(f"Port {port} received from {addr}: {data.decode()}")

listen_udp(50001)
