import pygame
import heapq
import math
import random

class Planner:
    def is_safe_distance_from_walls(self, surface, pos, world_width, world_height, safety_buffer=5):
        """
        Check if a position is at least safety_buffer pixels away from any brown wall.
        
        Args:
            surface: The pygame Surface representing the world map
            pos: The position to check (x, y)
            world_width: Width of the world
            world_height: Height of the world
            safety_buffer: Minimum distance to maintain from walls (default 5 pixels)
            
        Returns:
            Boolean indicating if the position is a safe distance from walls
        """
        WHITE = (255, 255, 255)
        BROWN = (181, 101, 29)
        
        # Check if the position itself is valid
        x, y = int(pos[0]), int(pos[1])
        if x < 0 or x >= world_width or y < 0 or y >= world_height:
            return False
            
        try:
            if surface.get_at((x, y)) != WHITE:
                return False
        except IndexError:
            return False
            
        # Check all points in a square around the position
        for dx in range(-safety_buffer, safety_buffer + 1):
            for dy in range(-safety_buffer, safety_buffer + 1):
                check_x, check_y = x + dx, y + dy
                
                # Skip if out of bounds
                if (check_x < 0 or check_x >= world_width or 
                    check_y < 0 or check_y >= world_height):
                    continue
                
                # Check if this point is a brown wall
                try:
                    pixel_color = surface.get_at((check_x, check_y))
                    # If any nearby pixel is a brown wall, the position is unsafe
                    if pixel_color[:3] == BROWN:
                        return False
                except IndexError:
                    continue
                    
        # If we reached here, the position is safe
        return True

    def get_path(self, surface, world_height, world_width, start, goal):
        """
        Implements Dijkstra's algorithm to find the shortest path between start and goal points.
        Maintains a safety buffer of 5 pixels from all walls.
        
        Args:
            surface: The pygame Surface representing the world map
            world_height: Height of the world
            world_width: Width of the world
            start: Starting coordinates (x, y)
            goal: Goal coordinates (x, y)
            
        Returns:
            A list of (x, y) coordinates representing the path from start to goal
        """
        # Ensure start and goal are tuples with integer coordinates
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        
        # Constants
        WHITE = (255, 255, 255)
        SAFETY_BUFFER = 5
        
        # Verify that start and goal are valid positions and not too close to walls
        # Note: We allow start and goal to be close to walls if necessary since they're fixed positions
        try:
            if start[0] < 0 or start[0] >= world_width or start[1] < 0 or start[1] >= world_height:
                print(f"Invalid start position: {start}")
                return []
            if goal[0] < 0 or goal[0] >= world_width or goal[1] < 0 or goal[1] >= world_height:
                print(f"Invalid goal position: {goal}")
                return []
            
            # Check if start or goal is a wall
            if surface.get_at(start) != WHITE:
                print(f"Start position is a wall: {start}")
                return []
            if surface.get_at(goal) != WHITE:
                print(f"Goal position is a wall: {goal}")
                return []
        except IndexError:
            print(f"Index error checking start/goal positions: {start}, {goal}")
            return []
        
        # If start and goal are the same, return a single-point path
        if start == goal:
            return [start]
        
        # Define directions (8-directional movement)
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),  # Cardinal directions
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal directions
        ]
        
        # Create visited set to keep track of evaluated nodes
        visited = set()
        
        # Create priority queue with starting node
        # Format: (distance, count, node)
        pq = []
        count = 0  # Tiebreaker for nodes with same distance
        heapq.heappush(pq, (0, count, start))
        
        # Dictionary to store distances from start to each node
        distances = {start: 0}
        
        # Dictionary to store the parent of each node (for path reconstruction)
        came_from = {}
        
        # Set a maximum number of iterations to prevent infinite loops
        max_iterations = world_width * world_height
        iteration = 0
        
        while pq and iteration < max_iterations:
            iteration += 1
            
            # Get node with smallest distance from start
            current_dist, _, current = heapq.heappop(pq)
            
            # Skip if already visited
            if current in visited:
                continue
                
            # Mark as visited
            visited.add(current)
            
            # Check if we've reached the goal
            if current[0] == goal[0] and current[1] == goal[1]:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            # Check all neighbors
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Skip if out of bounds
                if (neighbor[0] < 0 or neighbor[0] >= world_width or
                    neighbor[1] < 0 or neighbor[1] >= world_height):
                    continue
                
                # Skip if already visited
                if neighbor in visited:
                    continue
                
                # Skip if neighbor is not white or too close to a wall
                # (except for goal point which is allowed to be near walls)
                if neighbor != goal:
                    if not self.is_safe_distance_from_walls(surface, neighbor, world_width, world_height, SAFETY_BUFFER):
                        continue
                else:
                    # For the goal, just make sure it's not a wall
                    try:
                        if surface.get_at((int(neighbor[0]), int(neighbor[1]))) != WHITE:
                            continue
                    except IndexError:
                        continue
                
                # Calculate movement cost (diagonal movement costs more)
                movement_cost = 1.0 if dx == 0 or dy == 0 else 1.414
                new_distance = distances[current] + movement_cost
                
                # Update distance if this path is better
                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    came_from[neighbor] = current
                    
                    # Add to priority queue
                    count += 1
                    heapq.heappush(pq, (new_distance, count, neighbor))
        
        # No path found
        if iteration >= max_iterations:
            print("Dijkstra's algorithm exceeded maximum iterations")
        else:
            print("No path found to goal with safety buffer. Attempting to find path without buffer...")
            
            # If no path was found with safety buffer, try again with a reduced buffer
            # This fallback ensures we can still find a path even if it means getting
            # closer to walls than we'd like
            return self._get_path_without_buffer(surface, world_height, world_width, start, goal)
    
    def _get_path_without_buffer(self, surface, world_height, world_width, start, goal):
        """
        Fallback pathfinding method that doesn't enforce the wall safety buffer.
        Used when no path can be found with the buffer constraint.
        """
        # Constants
        WHITE = (255, 255, 255)
        
        # Define directions (8-directional movement)
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),  # Cardinal directions
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal directions
        ]
        
        # Create visited set to keep track of evaluated nodes
        visited = set()
        
        # Create priority queue with starting node
        pq = []
        count = 0
        heapq.heappush(pq, (0, count, start))
        
        # Dictionary to store distances from start to each node
        distances = {start: 0}
        
        # Dictionary to store the parent of each node (for path reconstruction)
        came_from = {}
        
        # Set a maximum number of iterations to prevent infinite loops
        max_iterations = world_width * world_height
        iteration = 0
        
        while pq and iteration < max_iterations:
            iteration += 1
            
            # Get node with smallest distance from start
            current_dist, _, current = heapq.heappop(pq)
            
            # Skip if already visited
            if current in visited:
                continue
                
            # Mark as visited
            visited.add(current)
            
            # Check if we've reached the goal
            if current[0] == goal[0] and current[1] == goal[1]:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            # Check all neighbors
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Skip if out of bounds
                if (neighbor[0] < 0 or neighbor[0] >= world_width or
                    neighbor[1] < 0 or neighbor[1] >= world_height):
                    continue
                
                # Skip if already visited
                if neighbor in visited:
                    continue
                
                # Skip if neighbor is a wall
                try:
                    if surface.get_at((int(neighbor[0]), int(neighbor[1]))) != WHITE:
                        continue
                except IndexError:
                    continue
                
                # Calculate movement cost (diagonal movement costs more)
                movement_cost = 1.0 if dx == 0 or dy == 0 else 1.414
                new_distance = distances[current] + movement_cost
                
                # Update distance if this path is better
                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    came_from[neighbor] = current
                    
                    # Add to priority queue
                    count += 1
                    heapq.heappush(pq, (new_distance, count, neighbor))
        
        # No path found even without buffer
        print("No path found, even without safety buffer")
        return []
        
    def get_meeting_point(self, surface, world_height, world_width, pos1, pos2):
        """
        Finds a suitable meeting point between two agents that is at least 5 pixels
        away from any brown wall.
        
        Args:
            surface: The pygame Surface representing the world map
            world_height: Height of the world
            world_width: Width of the world
            pos1: Position of agent 1 (x, y)
            pos2: Position of agent 2 (x, y)
        
        Returns:
            A (x, y) coordinate for the meeting point
        """
        WHITE = (255, 255, 255)
        SAFETY_BUFFER = 5
        
        # First try the exact midpoint
        midpoint_x = (pos1[0] + pos2[0]) / 2.0
        midpoint_y = (pos1[1] + pos2[1]) / 2.0
        midpoint = (int(midpoint_x), int(midpoint_y))
        
        # Check if midpoint is valid and safe distance from walls
        if self.is_safe_distance_from_walls(surface, midpoint, world_width, world_height, SAFETY_BUFFER):
            return midpoint
        
        # If midpoint isn't suitable, try to find a nearby safe space
        found_points = []
        
        for radius in range(1, 100, 2):  # Try increasingly larger circles
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # Only check points near the circle (to reduce computation)
                    if abs(dx*dx + dy*dy - radius*radius) > radius:
                        continue
                    
                    check_point = (int(midpoint_x + dx), int(midpoint_y + dy))
                    
                    # Skip if out of bounds
                    if (check_point[0] < 0 or check_point[0] >= world_width or
                        check_point[1] < 0 or check_point[1] >= world_height):
                        continue
                    
                    # Check if this point is safe from walls
                    if self.is_safe_distance_from_walls(surface, check_point, world_width, world_height, SAFETY_BUFFER):
                        # Calculate the "goodness" of this point (how equidistant it is from both agents)
                        dist1 = math.sqrt((check_point[0] - pos1[0])**2 + (check_point[1] - pos1[1])**2)
                        dist2 = math.sqrt((check_point[0] - pos2[0])**2 + (check_point[1] - pos2[1])**2)
                        
                        # Prefer points that are more equidistant
                        diff = abs(dist1 - dist2)
                        
                        # Add to our candidates list
                        found_points.append((diff, check_point))
                        
                        # If we found some points, we can stop the extensive search
                        if len(found_points) >= 10:
                            # Get the best point (lowest difference in distances)
                            found_points.sort(key=lambda x: x[0])
                            return found_points[0][1]
            
            # If we found any valid points in this radius, use the best one
            if found_points:
                found_points.sort(key=lambda x: x[0])
                return found_points[0][1]
        
        # If we couldn't find a safe point, try fallback approaches
        print("Warning: Could not find meeting point with safety buffer. Trying without buffer...")
        
        # Try again but without the safety buffer requirement
        # First try the midpoint if it's just free (not a wall)
        try:
            if midpoint[0] >= 0 and midpoint[0] < world_width and midpoint[1] >= 0 and midpoint[1] < world_height:
                if surface.get_at(midpoint) == WHITE:
                    return midpoint
        except IndexError:
            pass
            
        # Try a few points between agents
        for t in [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8]:
            check_point = (int(pos1[0] + t * (pos2[0] - pos1[0])), 
                        int(pos1[1] + t * (pos2[1] - pos1[1])))
            
            if check_point[0] >= 0 and check_point[0] < world_width and check_point[1] >= 0 and check_point[1] < world_height:
                try:
                    if surface.get_at(check_point) == WHITE:
                        return check_point
                except IndexError:
                    pass
        
        # Last resort: find any valid point near either agent
        for agent_pos in [pos1, pos2]:
            for radius in range(1, 20):
                for _ in range(10):
                    angle = random.uniform(0, 2 * math.pi)
                    dx = int(radius * math.cos(angle))
                    dy = int(radius * math.sin(angle))
                    
                    check_point = (int(agent_pos[0] + dx), int(agent_pos[1] + dy))
                    
                    if check_point[0] >= 0 and check_point[0] < world_width and check_point[1] >= 0 and check_point[1] < world_height:
                        try:
                            if surface.get_at(check_point) == WHITE:
                                return check_point
                        except IndexError:
                            pass
        
        # Ultimate fallback: just return position of agent 1
        print("Warning: Could not find any suitable meeting point. Using agent 1's position.")
        return pos1