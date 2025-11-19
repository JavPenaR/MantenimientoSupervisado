# Code to synchronize time

import time
import random

# Time server node index
TIME_SERVER_NODE = 0

# Node class representing a node in the network
class Node:
    def __init__(self, id, offset):
        self.id = id
        self.offset = offset
        # Initialize the clock with the current time in seconds
        self.clock = time.time() + random.uniform(-1, 1)  # Add a random offset

    # Method to adjust the clock offset of the node
    def adjust_clock(self, offset):
        self.offset += offset
        self.clock += offset  # Update the clock value

# Function to synchronize all nodes in the network
def synchronize_clocks(nodes):
    # Calculate the time of the time server node
    server_time = nodes[TIME_SERVER_NODE].clock
    
    # Adjust offset to each node
    for node in nodes:
        if node.id != TIME_SERVER_NODE:
            node_time = node.clock
            offset = server_time - node_time
            node.adjust_clock(offset)

# Main function
def main():
    # Take input for the number of nodes
    num_nodes = int(input("Enter the number of nodes: "))

    # Create nodes in the network with user-defined offsets
    nodes = []
    for i in range(num_nodes):
        offset = float(input(f"Enter the offset for node {i + 1} (in seconds): "))
        nodes.append(Node(i + 1, offset))

    # Print initial clock values of all nodes
    print("\nBefore synchronization:")
    for node in nodes:
        print(f"Node {node.id} clock: {node.clock}")

    return nodes

if __name__ == "__main__":
    nodes = main()

    # Synchronize clocks of all nodes after the main process completes
    synchronize_clocks(nodes)

    # Print clock values of all nodes after synchronization
    print("\nAfter synchronization:")
    for node in nodes:
        print(f"Node {node.id} clock: {node.clock}")
