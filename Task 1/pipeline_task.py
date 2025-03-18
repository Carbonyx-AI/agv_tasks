"""
API for agent: (Read robot_api.py for more detailed documentation)
agent.move(dist)     : moves the agent in forward direction for dist units. Avoid using values more than 5 the value of distance for one 
					   iteration of work. It will still work but you won't see it being animated (was too much effort implementing that)
agent.scan()         : returns a list of size 180 which gives distance of nearest obstacle/wall for every 2 degrees 
						with 0th element being the distance at degree the agent is facing - 180
agent.rotate(deg)    : rotates the agent by deg degrees
agent.get_imu_data() : gives the direction the agent is heading towards.

(For Subtasks)
agent.get_world() :  gives pygame.Surface of the world, if this.get_at((x, y)) != WHITE then its a wall
agent.get_pos()   : returns coordinates of the agent. Usable only once. (x, y) = (pos[0], pos[1])

Task : This class is supposed to make the two agents meet somewhere on the map. You can create the map yourself in the pygame GUI, you can
save and load maps from your system. Your code will be tested on different maps which I will make randomly lol. You are 
supposed to document your progress. Your approach and effort to solve this problem matters more than the final solution. 
One approach to solve this is to implement localization and mapping to get the map for both agents and then use some kind of map merging algorithm 
to get a transform between the maps and use that to get relative displacement between the two agents and use that to make them meet. Subtasks 
given below are supposed to reduce the number of things you need to do make the agents meet. You can use any resource (pre-written code, LLMs, 
books etc) but you are supposed to mention what you used in your documentation and have a basic understanding of what you did. 

Subtask 1: You can use initial coordinates of both agents and the world surface map
Subtask 2: In conditions of subtask 1, ensure a smooth path. Read about Dubin Curves or Reeds-Shepp
Subtask 3: You can use initial coordinates of both agents but not the map
Subtask 4: You can use the map but not coordinates of either of agents
Subtask 5: You can use only scan and imu data.

Subtask 6 (Bonus): Do this optimally without scanning the full map if possible. (Some image processing techniques may be used)

If you do Subtask x, you get points for all Subtask y such that y <= x

GUI:
You can use WASD and arrow keys to control the agents. It will be helpful in debugging (or you can just play around lol)
Add Walls : duh, it adds walls. You can use a brush to add walls to the map.
Remove Walls : You can use a brush to remove walls from the map.
Start Pipeline : It basically runs the work function in this class
View : There are two views 
		Full -> in this mode the normal map with brown walls and white background is shown
		Explored -> this shows the area that the scan has explored till now in white color rest is in black
Upload Map : This can be used to upload a .png file as map. Note: stuff might get buggy if you upload something that you didn't save from the GUI but essentially anything with brown rgb(181, 101, 29) walls and white backgruond should work. 
Save Map : This saves the current map as a .png file. 

Note: If you find any bugs, try to fix them and write about them in your documentation and DM me (aryanr_) on discord
"""

from localization import Localization
from planning import Planner
import math

class Pipeline:
	def __init__(self, world_width, world_height, map):
		# initial position will have to be assumed to be (world_width, world_height)
		self.explored_map1 = Localization(2 * world_width, 2 * world_height) # map explored by agent 1
		self.explored_map2 = Localization(2 * world_width, 2 * world_height) # map explored by agent 2
		self.planner = Planner() # gets the path once the map has been finalized
		self.map_merger = None
		self.path1 = None  # Path for agent1
		self.path2 = None  # Path for agent2
		self.ind1 = 0      # Index for agent1's path
		self.ind2 = 0      # Index for agent2's path
		self.meeting_point = None  # The point where the agents should meet
		
		# For tracking positions (since get_pos() can only be called once)
		self.pos_agent1 = None
		self.pos_agent2 = None
		self.initialized = False
		
		self.world_height = world_height
		self.world_width = world_width
		self.world_map = map  # Store reference to the world map
		return

	def reset(self):
		self.path1 = None
		self.path2 = None
		self.ind1 = 0
		self.ind2 = 0
		self.meeting_point = None
		self.pos_agent1 = None
		self.pos_agent2 = None
		self.initialized = False
		return

	def work(self, agent1, agent2):
		# Initialize on first call (get_pos can only be called once)
		if not self.initialized:
			# Get the positions of both agents (only once!)
			self.pos_agent1 = agent1.get_pos()
			self.pos_agent2 = agent2.get_pos()
			self.initialized = True
			
			# Find a suitable meeting point
			self.meeting_point = self.planner.get_meeting_point(
				self.world_map, 
				self.world_height, 
				self.world_width, 
				self.pos_agent1, 
				self.pos_agent2
			)
			
			# Find paths for both agents to the meeting point
			self.path1 = self.planner.get_path(
				self.world_map, 
				self.world_height,
				self.world_width, 
				self.pos_agent1, 
				self.meeting_point
			)
			
			self.path2 = self.planner.get_path(
				self.world_map, 
				self.world_height,
				self.world_width, 
				self.pos_agent2, 
				self.meeting_point
			)
			
			# Reset indices and check if paths are valid
			self.ind1 = 0 if self.path1 else -1  # -1 indicates no valid path
			self.ind2 = 0 if self.path2 else -1
			
			print(f"Agent 1 position: {self.pos_agent1}")
			print(f"Agent 2 position: {self.pos_agent2}")
			print(f"Meeting point: {self.meeting_point}")
			print(f"Path 1 length: {len(self.path1) if self.path1 else 0}")
			print(f"Path 2 length: {len(self.path2) if self.path2 else 0}")
		
		# Check if we have valid paths
		if not self.path1 or not self.path2:
			print("No valid paths found to the meeting point")
			return
			
		# Move agents along their paths
		moved1 = self._move_agent(agent1, self.ind1 < len(self.path1))
		moved2 = self._move_agent(agent2, self.ind2 < len(self.path2), is_agent1=False)
		
		# Update our tracked positions based on agent movements
		current_angle1 = agent1.get_imu_data()
		current_angle2 = agent2.get_imu_data()
		
		# Only update positions if we've moved (based on return from _move_agent)
		if moved1['rotated'] or moved1['moved']:
			# If agent moved, update our tracked position
			if moved1['moved']:
				move_distance = moved1['distance']
				angle_rad = math.radians(current_angle1)
				self.pos_agent1 = [
					self.pos_agent1[0] + move_distance * math.cos(angle_rad),
					self.pos_agent1[1] + move_distance * math.sin(angle_rad)
				]
		
		if moved2['rotated'] or moved2['moved']:
			# If agent moved, update our tracked position
			if moved2['moved']:
				move_distance = moved2['distance']
				angle_rad = math.radians(current_angle2)
				self.pos_agent2 = [
					self.pos_agent2[0] + move_distance * math.cos(angle_rad),
					self.pos_agent2[1] + move_distance * math.sin(angle_rad)
				]
		
		# Check if agents have reached or are close to their current waypoints
		if self.ind1 < len(self.path1):
			target1 = self.path1[self.ind1]
			if self._is_close_to_target(self.pos_agent1, target1):
				if self.ind1 < len(self.path1) - 1:
					self.ind1 += 1
				    
		if self.ind2 < len(self.path2):
			target2 = self.path2[self.ind2]
			if self._is_close_to_target(self.pos_agent2, target2):
				if self.ind2 < len(self.path2) - 1:
					self.ind2 += 1
		
		# Check if both agents have reached the meeting point
		if (self._is_close_to_target(self.pos_agent1, self.meeting_point) and 
			self._is_close_to_target(self.pos_agent2, self.meeting_point)):
			print("Agents have successfully met!")
		
		return
	
	def _move_agent(self, agent, has_target, is_agent1=True):
		"""
		Helper method to move an agent along its path
		Returns dict with info about movement
		"""
		result = {'rotated': False, 'moved': False, 'distance': 0}
		
		if not has_target:
			return result
		
		# Get current position and path index
		current_pos = self.pos_agent1 if is_agent1 else self.pos_agent2
		path = self.path1 if is_agent1 else self.path2
		path_index = self.ind1 if is_agent1 else self.ind2
		
		if path_index >= len(path):
			# If we've reached the end of the path, target the meeting point
			target = self.meeting_point
		else:
			target = path[path_index]
		
		# Calculate angle to target
		dx = target[0] - current_pos[0]
		dy = target[1] - current_pos[1]
		target_angle = math.degrees(math.atan2(dy, dx))
		
		# Get current orientation
		current_angle = agent.get_imu_data()
		
		# Calculate angle difference (consider the shorter rotation)
		angle_diff = (target_angle - current_angle + 180) % 360 - 180
		
		# Rotate towards target
		if abs(angle_diff) > 5:  # Threshold for rotation
			rotation_step = min(5, abs(angle_diff))  # Max rotation per step
			if angle_diff > 0:
				agent.rotate(rotation_step)
			else:
				agent.rotate(-rotation_step)
			result['rotated'] = True
		else:
			# Move towards target if aligned properly
			distance = math.sqrt(dx*dx + dy*dy)
			if distance > 2:  # Don't try to move if very close
				move_step = min(2, distance)  # Max movement per step
				if agent.move(move_step):  # Only update if move was successful
					result['moved'] = True
					result['distance'] = move_step
		
		return result
	
	def _is_close_to_target(self, pos, target, threshold=5):
		"""Check if position is close to target within threshold"""
		dx = pos[0] - target[0]
		dy = pos[1] - target[1]
		return (dx*dx + dy*dy) <= threshold*threshold
		

