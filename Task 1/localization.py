import pygame

WHITE = (255, 255, 255)

class Localization:
    def __init__(self, world_width, world_height):
        self.map = pygame.Surface((world_width, world_height)) 
        self.map.fill((0, 0, 0))
        self.world_height = world_height
        self.world_width = world_width

    def update(self, agent):
        return
    
    def at(self, x, y):
        return self.map.get_at((int(x), int(y)))

