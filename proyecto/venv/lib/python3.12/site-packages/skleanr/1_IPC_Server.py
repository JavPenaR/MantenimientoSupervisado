import socket

# Set up server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('127.0.0.1', 8888))
server.listen(5)
print("\n Server : ", server, "\n" )

print("Server listening on port 8888")

while True:
    # Accept connection and handle data
    client, _ = server.accept()

    print("\n Client : ", client, "\n" )

    data = client.recv(1024)
    print("Received:", data.decode())
    client.send(data)
    client.close()
