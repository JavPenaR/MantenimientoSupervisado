import socket
import threading

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 1234

def handle_client(client_socket, address):
    try:
        # Receive message from client
        message = client_socket.recv(1024).decode()
        print(f"Received message from {address}: {message}")

        # Send response back to client
        response = f"Response to {message}"
        client_socket.sendall(response.encode())
    except socket.error as e:
        print(f"Socket error: {e}")
    finally:
        client_socket.close()

def main():
    try:
        # Create server socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            # Bind server socket to address
            server_socket.bind((SERVER_HOST, SERVER_PORT))
            print("Server started")

            # Listen for incoming connections
            server_socket.listen(5)

            while True:
                # Accept client connection
                client_socket, address = server_socket.accept()
                print(f"Client connected: {address}")

                # Create and start client handler thread
                client_handler = threading.Thread(target=handle_client, args=(client_socket, address))
                client_handler.start()

    except socket.error as e:
        print(f"Socket error: {e}")

if __name__ == "__main__":
    main()
