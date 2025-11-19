def main():
    num_processes = int(input("Enter the number of processes: "))
    num_cs = int(input("Enter the number of processes that want to enter the critical section: "))

    # Dictionary to store process IDs and their corresponding timestamps
    process_map = {}
    timestamps = []

    # Input process IDs and timestamps
    for _ in range(num_cs):
        process, timestamp = input("Enter the Process ID and Timestamp of process (e.g., '2 10'): ").split()
        process_map[int(timestamp)] = int(process)
        timestamps.append(int(timestamp))

    print()  # For better output formatting

    # Sort timestamps to ensure requests are processed in order
    timestamps.sort()

    # Simulate process requests
    for time in timestamps:
        process_cs = process_map[time]
        for i in range(num_processes):
            if process_cs != i:
                print(f"Process {process_cs} has requested Process {i}")
        print()  # Empty line for better output formatting

    # Simulate process acceptances and entry to critical section
    for time in timestamps:
        process_cs = process_map[time]
        for i in range(num_processes):
            if process_cs != i:
                print(f"Process {i} has accepted the request of process {process_cs}")
        print()  # Empty line for better output formatting
        print(f'Process {process_cs} has now entered the critical section')
        print(f'Process {process_cs} has now exited the critical section')
        print()

if __name__ == "__main__":
    main()
