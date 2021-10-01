import pygame
import random

HEIGTH, WIDTH, FPS = 360, 360, 30

#initialize pygame and create window
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGTH))
pygame.display.set_caption("RL Game")
clock = pygame.time.Clock()

#colors 
WHITE = (255,255,255)
RED = (255,0,0)
BLACK = (0,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
LIGHT_BLUE = (82,219,255)

#game loop
running = True

while running:
    #keep look running at right speed
    clock.tick(FPS)

    #process input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    #update

    #draw/render screen
    screen.fill(LIGHT_BLUE)

    #after drwing fill display
    pygame.display.flip()


pygame.quit()