def print_server_load(num_servers, num_processes):
    processes_per_server = num_processes // num_servers
    extra_processes = num_processes % num_servers

    for i in range(extra_processes):
        print(f"Server {i + 1} has {processes_per_server + 1} processes")

    for i in range(extra_processes, num_servers):
        print(f"Server {i + 1} has {processes_per_server} processes")


def display_menu():
    print("1. Add Server")
    print("2. Remove Server")
    print("3. Add Processes")
    print("4. Remove Processes")
    print("5. Exit")


def main():
    num_servers = int(input("Enter number of servers: "))
    num_processes = int(input("Enter number of processes: "))

    while True:
        print_server_load(num_servers, num_processes)
        display_menu()
        choice = int(input("> "))

        if choice == 1:
            num_servers += int(input("Enter number of servers to be added: "))
        elif choice == 2:
            num_servers -= int(input("Enter number of servers to be removed: "))
        elif choice == 3:
            num_processes += int(input("Enter number of processes to be added: "))
        elif choice == 4:
            num_processes -= int(input("Enter number of processes to be removed: "))
        elif choice == 5:
            return
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
