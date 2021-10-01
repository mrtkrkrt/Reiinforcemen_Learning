import pygame
import random

HEIGTH, WIDTH, FPS = 360, 360, 30

#colors 
WHITE = (255,255,255)
RED = (255,0,0)
BLACK = (0,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
LIGHT_BLUE = (82,219,255)

class Player(pygame.sprite.Sprite):
    #sprite for the player
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20,20))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.radius = 10
        pygame.draw.circle(self.image, GREEN, self.rect.center, self.radius)
        self.rect.center = (WIDTH/2, 350)
        self.y_speed = 0
        self.x_speed = 0
    
    def update(self):
        self.x_speed = 0

        event = pygame.key.get_pressed()

        if event[pygame.K_RIGHT]:
            self.x_speed = 4
        elif event[pygame.K_LEFT]:
            self.x_speed = -4
        else:
            self.x_speed = 0

        self.rect.x += self.x_speed

        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
        if self.rect.left < 0:
            self.rect.left = 0  


class Enemy(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10,10))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        
        self.radius = 5
        pygame.draw.circle(self.image, BLACK, self.rect.center, self.radius)

        self.rect.x = random.randrange(0, WIDTH - self.rect.width)
        self.rect.y = random.randrange(2, 6)

        self.x_speed = 0
        self.y_speed = 3

    def update(self):

        self.rect.x += self.x_speed
        self.rect.y += self.y_speed

        if self.rect.top > HEIGTH + 10:
            self.rect.x = random.randrange(0, WIDTH - self.rect.width)
            self.rect.y = random.randrange(2, 6)
            self.y_speed = 3

    def get_coordinates(self):
        return (self.rect.x, self.rect.y)



#initialize pygame and create window
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGTH))
pygame.display.set_caption("RL Game")
clock = pygame.time.Clock()

#game loop
running = True

#sprite 
all_sprite = pygame.sprite.Group()
enemy = pygame.sprite.Group()
player = Player()
en1 = Enemy()
en2 = Enemy()
all_sprite.add(player)
all_sprite.add(en1)
all_sprite.add(en2)
enemy.add(en1)
enemy.add(en2)

while running:
    #keep look running at right speed
    clock.tick(FPS)

    #process input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    #draw/render screen
    screen.fill(RED)

    #update
    all_sprite.update()
    all_sprite.draw(screen)

    hits = pygame.sprite.spritecollide(player, enemy, False, pygame.sprite.collide_circle)

    if hits:
        running = False
        print("Game Over!!!")
    #after drwing fill display
    pygame.display.flip()


pygame.quit()