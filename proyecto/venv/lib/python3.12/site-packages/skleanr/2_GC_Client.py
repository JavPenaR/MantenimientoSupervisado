import socket
import threading

# Function to receive messages from server
def receive_messages(client_socket):
    while True:
        message = client_socket.recv(1024).decode()
        print(message)

# Set up client
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('127.0.0.1', 8888))

# Start a thread to receive messages from server
receive_thread = threading.Thread(target=receive_messages, args=(client,))
receive_thread.start()

# Main thread to send messages to server
while True:
    message = input()
    client.send(message.encode())
