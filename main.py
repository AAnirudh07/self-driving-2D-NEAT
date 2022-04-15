import pygame
import neat
import math
import os
import sys

map_width=1244
map_height=1016

screen = pygame.display.set_mode((map_width,map_height))
car_map = pygame.image.load(os.path.join("Assets","map.png"))

class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.org_img=pygame.image.load(os.path.join("Assets","car.png"))
        self.image = self.org_img #image displayed on the screen
        self.rect=self.image.get_rect(center=(490,820)) #'bounding box' for the image
        self.velocity=pygame.math.Vector2(0.8,0)
        self.angle=0
        self.rotation = 10
        self.direction = 0 #left or right flag
        self.alive=True
        self.radars=[]
    
    def update(self):
        self.radars.clear()
        self.drive()
        self.rotate()
        for radar_angle in (-45, 0, 45):
            self.radar(radar_angle)
        self.collision()
        self.data()
    
    def collision(self):
        length=40
        collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]
        collision_point_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]
        if screen.get_at(collision_point_right) == pygame.Color(2, 105, 31, 255) or screen.get_at(collision_point_left) == pygame.Color(2, 105, 31, 255):
            self.alive = False
        pygame.draw.circle(screen, (0, 255, 255, 0), collision_point_right, 4)
        pygame.draw.circle(screen, (0, 255, 255, 0), collision_point_left, 4)
    
    
    def drive(self):
        self.rect.center += self.velocity*16
    
    def rotate(self):
        if self.direction==1:
            self.angle-=self.rotation
            self.velocity.rotate_ip(self.rotation)

        elif self.direction==-1:
            self.angle+=self.rotation
            self.velocity.rotate_ip(-self.rotation)

        
        self.image = pygame.transform.rotozoom(self.org_img, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)


    def radar(self, radar_angle):
        length=0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])
        
        while not screen.get_at((x,y)) == pygame.Color(2, 105, 31, 255) and length<200:
            length+=1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle+radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle+radar_angle)) * length) 
            
        pygame.draw.line(screen, (255, 255, 255, 255), self.rect.center, (x, y), 1)
        pygame.draw.circle(screen, (0, 255, 0, 0), (x, y), 3)

        dist = int(math.sqrt(math.pow(self.rect.center[0] - x, 2) + math.pow(self.rect.center[1] - y, 2)))
        self.radars.append([radar_angle, dist])
    
    def data(self):
        input = [0, 0, 0]
        for i, radar in enumerate(self.radars):
            input[i] = int(radar[1])
        return input

def remove(index): #remove cars that run out of the track
    cars.pop(index)
    ge.pop(index)
    nets.pop(index)

def eval_genomes(genomes, config): #evaluate the fitness of the cars going around the track
    global cars, ge, nets
    cars=[]
    ge=[]
    nets=[]

    for genome_id, genome in genomes:
            cars.append(pygame.sprite.GroupSingle(Car()))
            ge.append(genome)
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append(net)
            genome.fitness = 0
    
    
    run = True
    while run:
        #exit if the X button is pressed 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        screen.blit(car_map,(0,0))
        
        if len(cars) == 0: break

        for i, car in enumerate(cars):
            ge[i].fitness += 1
            if not car.sprite.alive:
                remove(i)

        for i, car in enumerate(cars):
            output = nets[i].activate(car.sprite.data())
            if output[0] > 0.7:
                car.sprite.direction = 1
            if output[1] > 0.7:
                car.sprite.direction = -1
            if output[0] <= 0.7 and output[1] <= 0.7:
                car.sprite.direction = 0

        for car in cars:
            car.draw(screen)
            car.update()
        pygame.display.update()

def run(config_path):
    global pop
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    pop.run(eval_genomes, 50)

if __name__ == '__main__':
    run('config.txt')

