import socket

# Set up client
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("\n", client, "\n")
client.connect(('127.0.0.1', 8888))

# Send data to server
message = b"Hello, server!"
sent = client.send(message)
print("\n", client, "\n")

# Check if message has been sent
if sent == len(message):
    print("Message sent successfully!")
else:
    print("Message not sent completely.")

# Receive data from server
response = client.recv(1024)
print("\n", client, "\n")
print("\nResponse: ", response.decode(), "\n")

# Close client connection
client.close()
