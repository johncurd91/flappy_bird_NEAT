import pygame
import neat
import os
import random
pygame.font.init()

WIN_WIDTH = 500
WIN_HEIGHT = 800

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join('images', 'bird1.png'))),
             pygame.transform.scale2x(pygame.image.load(os.path.join('images', 'bird2.png'))),
             pygame.transform.scale2x(pygame.image.load(os.path.join('images', 'bird3.png')))]

PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join('images', 'pipe.png')))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join('images', 'base.png')))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join('images', 'bg.png')))

STAT_FONT = pygame.font.SysFont('comicsans', 50)

GEN = 0

class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1

        d = self.vel * self.tick_count + 1.5 * self.tick_count**2  # equation for jump displacement

        if d >= 16:
            d = 16  # set terminal vel at 16

        if d < 0:
            d -= 2

        self.y = self.y + d

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL  # nose dive at tilt over 90

    def draw(self, win):
        # loop through images to animate bird
        self.img_count += 1

        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]  # set image when in nose dive
            self.img_count = self.ANIMATION_TIME * 2

        # rotate image based on tilt
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(round(self.x), round(self.y))).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = 200
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0

        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)  # flips pipe image for top pipe
        self.PIPE_BOTTOM = PIPE_IMG

        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)  # random range for pipe placement
        self.top = self.height - self.PIPE_TOP.get_height()  # negative value beyond top of window
        self.bottom = self.height + self.GAP  # positive value within window

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()  # get mask of image for pixel perfect collision
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)  # function returns None if no points overlap
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True  # returns True if collision detected

        return False  # otherwise return False


class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


# Main game
def draw_window(win, birds, pipes, base, score):
    win.blit(BG_IMG, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render('Score: ' + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    text = STAT_FONT.render('Gen: ' + str(GEN), 1, (255, 255, 255))
    win.blit(text, (10, 10))

    text = STAT_FONT.render('Alive: ' + str(len(birds)), 1, (255, 255, 255))
    win.blit(text, (10, 50))



    base.draw(win)

    for bird in birds:
        bird.draw(win)
    pygame.display.update()


def eval_genomes(genomes, config):
    global GEN
    GEN += 1
    nets = []
    ge = []
    birds = []

    for _, g in genomes:  # because genomes is a tuple (index, object)
        net = neat.nn.FeedForwardNetwork.create(g, config)  # set up NN for genome
        nets.append(net)  # add to list
        birds.append(Bird(230, 350))  # add corresponding bird
        g.fitness = 0
        ge.append(g)

    base = Base(730)
    pipes = [Pipe(700)]
    score = 0
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()  # clock object to control framerate

    run = True
    while run:
        clock.tick(30)  # set framerate
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        # Set which pipe in list for bird to look at based on x
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height),
                                      abs(bird.y - pipes[pipe_ind].bottom)))  # give NN bird y, and position of pipes

            if output[0] > 0.5:
                bird.jump()

        add_pipe = False
        rem = []  # list to hold pipes that have left the screen
        for pipe in pipes:
            for x, bird in enumerate(birds):  # get index and bird
                if pipe.collide(bird):  # check if bird has collided with pipe
                    ge[x].fitness -= 1  # remove fitness from bird that has collided

                    # remove corresponding bird, NN, and genome:
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)


                if not pipe.passed and pipe.x < bird.x:  # check if bird had passed pipe
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:  # check if pipe is off screen
                rem.append(pipe)  # move pipe to temp rem list

            pipe.move()

        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5  # to favor birds that pass pipes, rather than collide
            pipes.append(Pipe(700))

        for r in rem:
            pipes.remove(r)  # removes pipes from temp rem list

        for x, bird in enumerate(birds):  # get index and bird
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:  # check if bird has collided with base or top
                # remove corresponding bird, NN, and genome:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        base.move()

        draw_window(win, birds, pipes, base, score)




def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)  # generated population based on config file

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 50)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)
