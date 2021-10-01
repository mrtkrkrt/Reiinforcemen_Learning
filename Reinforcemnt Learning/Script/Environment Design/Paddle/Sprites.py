import pygame
import random

class Enemy(pygame.sprite.Sprite):

    def __init__(self, width, height):
        self.width = width
        self.height = height
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10,10))
        self.image.fill((0,0,255))
        self.rect = self.image.get_rect()
        self.radius = 5
        pygame.draw.circle(self.image, (0,0,0), self.rect.center,self.radius)
        self.rect.center = (random.randint(0, self.width), 0)
        self.x_speed = 0
        self.y_speed = 5

    def update(self):
        self.rect.x += self.x_speed
        self.rect.y += self.y_speed

        if self.rect.top > self.height + 10:
            self.rect.x = random.randrange(0, self.width - self.rect.width)
            self.rect.y = random.randrange(2, 6)
            self.y_speed = 3

    def get_coordinates(self):
        return (self.rect.x, self.rect.y)
    
    def reset(self):
        self.rect.center = (random.randint(0, self.width), 0)

class Paddle(pygame.sprite.Sprite):
    
    def __init__(self, width, height):
        self.height = height
        self.width = width
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((75,20))
        self.image.fill((0,0,0))
        self.rect = self.image.get_rect()
        self.rect.center = (width/2, height-10)
        self.x_speed = 0
        self.y_speed = 0

    def update(self, action):
        self.x_speed = 0
        event = pygame.key.get_pressed()

        if event[pygame.K_LEFT] or action == 0:
            self.x_speed = -4
        elif event[pygame.K_RIGHT] or action == 1:
            self.x_speed = 4
        else:
            self.x_speed = 0

        self.rect.x +=self.x_speed
        
        if self.rect.right > self.height:
            self.rect.right = self.height
        if self.rect.left < 0:
            self.rect.left = 0

    def get_coordinates(self):
        return (self.rect.x, self.rect.y)

    def reset(self):
        self.rect.center = (self.width/2, self.height-10)


def main():
    pygame.init()
    screen = pygame.display.set_mode((360, 360))
    pygame.display.set_caption("Paddle")
    clock = pygame.time.Clock()

    enemy = Enemy(360,360)
    spritess = pygame.sprite.Group()
    spritess.add(enemy)

    running = True

    while running:
    #keep look running at right speed
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((255,255,255))

        spritess.update()
        spritess.draw(screen)

        #after drwing fill display
        pygame.display.flip()

    pygame.quit()
