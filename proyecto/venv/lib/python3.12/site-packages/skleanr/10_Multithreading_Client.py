import socket
import threading

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 1234

class ClientThread(threading.Thread):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def run(self):
        try:
            # Connect to server
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.connect((SERVER_HOST, SERVER_PORT))

                # Send message to server
                message = f"{self.name} is sending a message"
                client_socket.sendall(message.encode())

                # Receive response from server
                response = client_socket.recv(1024).decode()
                print(f"{self.name} received response: {response}")

        except socket.error as e:
            print(f"Socket error: {e}")

def main():
    # Create and start client threads
    threads = []
    for i in range(5):
        thread = ClientThread(f"Thread {i}")
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
