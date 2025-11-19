import socket
import threading

# Function to handle each client connection
def handle_client(client_socket, clients):
    while True:
        # Receive message from client
        message = client_socket.recv(1024).decode()
        if not message:  # If message is empty, client has disconnected
            break

        # Broadcast message to all clients
        for client in clients:
            if client != client_socket:  # Don't send the message back to the sender
                client.send(message.encode())

    # Remove client from list after they disconnect
    clients.remove(client_socket)
    client_socket.close()

# Set up server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('127.0.0.1', 8888))
server.listen(5)

print("Server listening on port 8888")

# List to store client sockets
clients = []

# Accept incoming connections and handle them in separate threads
while True:
    client_socket, _ = server.accept()  # Accept connection
    clients.append(client_socket)  # Add client socket to list
    client_handler = threading.Thread(target=handle_client, args=(client_socket, clients))  # Start thread to handle client
    client_handler.start()
