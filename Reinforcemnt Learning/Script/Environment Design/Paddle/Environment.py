import pygame
from Sprites import Paddle, Enemy
from DQL_Agent import Agent

class Environment():

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.paddle = Paddle(self.width, self.height)
        self.m1 = Enemy(self.width, self.height)
        self.m2 = Enemy(self.width, self.height)
        self.enemy_sprites = pygame.sprite.Group()
        self.enemy_sprites.add(self.m1)
        self.enemy_sprites.add(self.m2)
        self.all_sprite = pygame.sprite.Group()
        self.all_sprite.add(self.paddle)
        self.all_sprite.add(self.m1)
        self.all_sprite.add(self.m2)

        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        self.reward = 0
        self.total_reward = 0
        self.done = False
        self.agent = Agent(3, 4)

    def findDistance(self, a, b):
        d = a-b
        return d
    
    def reset(self):
        self.width = 360
        self.height = 360
        self.paddle = Paddle(self.width, self.height)
        self.m1 = Enemy(self.width, self.height)
        self.m2 = Enemy(self.width, self.height)
        self.enemy_sprites = pygame.sprite.Group()
        self.enemy_sprites.add(self.m1)
        self.enemy_sprites.add(self.m2)
        self.all_sprite = pygame.sprite.Group()
        self.all_sprite.add(self.paddle)
        self.all_sprite.add(self.m1)
        self.all_sprite.add(self.m2)

        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        self.reward = 0
        self.total_reward = 0
        self.done = False

        state_list = []
        
        # get coordinate
        player_state = self.paddle.get_coordinates()
        m1_state = self.m1.get_coordinates()
        m2_state = self.m2.get_coordinates()
        
        # find distance
        state_list.append(self.findDistance(player_state[0],m1_state[0]))
        state_list.append(self.findDistance(player_state[1],m1_state[1]))
        state_list.append(self.findDistance(player_state[0],m2_state[0]))
        state_list.append(self.findDistance(player_state[1],m2_state[1]))

        return [state_list]

    def step(self, action):
        state_list = []
        
        # update
        self.paddle.update(action)
        self.enemy_sprites.update()
        
        # get coordinate
        next_player_state = self.paddle.get_coordinates()
        next_m1_state = self.m1.get_coordinates()
        next_m2_state = self.m2.get_coordinates()
        
        # find distance
        state_list.append(self.findDistance(next_player_state[0],next_m1_state[0]))
        state_list.append(self.findDistance(next_player_state[1],next_m1_state[1]))
        state_list.append(self.findDistance(next_player_state[0],next_m2_state[0]))
        state_list.append(self.findDistance(next_player_state[1],next_m2_state[1]))
        
        return [state_list]

    def run(self):

        state = self.reset()
        running = True
        time = 0

        while running:
            self.reward = .001
            self.clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action = self.agent.act(state)
            next_state = self.step(action)
            self.total_reward += self.reward

            """-300 total_reward olunca done=True olsun
            her birini yakalayınca 75 alsın
            kaçırınca 75 kaybetsin
            """

            hits = pygame.sprite.spritecollide(self.paddle,self.enemy_sprites,False, pygame.sprite.collide_circle)   

            for enemy in self.enemy_sprites.sprites():
                temp = pygame.sprite.Group()
                temp.add(enemy)
                if pygame.sprite.spritecollide(self.paddle, temp, False, pygame.sprite.collide_circle):
                    self.reward = 75
                    self.total_reward += self.reward
                    enemy.reset()
                if enemy.get_coordinates()[1] > self.paddle.get_coordinates()[1]:
                    self.reward = -75
                    self.total_reward += self.reward
                    enemy.reset()

            if self.total_reward <= -300:
                running = False
                self.done = True
                self.paddle.reset()
                print("Total Reward : ", self.total_reward, " Time : ", time)
                time = 0
            
            self.agent.remember(state, action, self.reward, next_state, self.done)
            state = next_state
            self.agent.replay()
            self.agent.adaptiveEGreedy()

            self.screen.fill((0,255,0))
            self.enemy_sprites.update()
            self.paddle.update(2)
            self.all_sprite.draw(self.screen)

            pygame.display.flip()
            time += 1

        self.agent.model.save("Model.model")
        pygame.quit()


def main():
    env = Environment(360, 360)
    liste = []
    t = 0

    while True:
        t += 1

        print("Episode : ", t)
        liste.append(env.total_reward)

        pygame.init()
        screen = pygame.display.set_mode((360,360))
        pygame.display.set_caption("RL Game")
        clock = pygame.time.Clock()

        env.run()

main()



