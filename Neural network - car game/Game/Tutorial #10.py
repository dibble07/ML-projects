import pygame
pygame.init()

# initialise game
win_sz = (600,400)
win = pygame.display.set_mode(win_sz)
pygame.display.set_caption("First Game")
clock = pygame.time.Clock()
score = 0
bg = pygame.image.load('bg.jpg')

class course(object):
    def __init__(self,x,y,width,height,track_width):
    	self.inner = pygame.Rect(x + track_width, y + track_width, width - 2*track_width, height - 2*track_width)
    	self.outer = pygame.Rect(x, y, width, height)

    def draw(self, win):
        pygame.draw.rect(win, (255,0,0), self.inner,2)
        pygame.draw.rect(win, (255,0,0), self.outer,2)

class player(object):
    def __init__(self,x,y,width,height):
        self.hitbox = pygame.Rect(x, y, width, height)
        self.xvel = 5
        self.yvel = 5
        self.functioning = True
        self.img = pygame.transform.scale(pygame.image.load("car.png") , (self.hitbox.w,self.hitbox.h))

    def draw(self, win):
        win.blit(self.img, (self.hitbox.x, self.hitbox.y))
        pygame.draw.rect(win, (255,0,0), self.hitbox,2)

    # def hit(self):
        # self.x = 100
        # self.y = 410
        # i = 0
        # while i < 200:
        #     pygame.time.delay(10)
        #     i += 1
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             i = 201
        #             pygame.quit()

def redrawGameWindow():
    win.blit(bg, (0,0))
    track.draw(win)
    car.draw(win)
    pygame.display.update()

track = course(50, 50, 500, 300, 100)
car = player(1, 1, 20, 20)

run = True
while run:
    clock.tick(27)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    if track.outer.colliderect(car.hitbox) and not(track.inner.colliderect(car.hitbox)):
    	print("good")
    else:
    	print("bad")

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT] and car.hitbox.x >= car.xvel:
        car.hitbox.x -= car.xvel
    if keys[pygame.K_RIGHT] and car.hitbox.x <= win_sz[0] - car.hitbox.width - car.xvel:
        car.hitbox.x += car.xvel
    if keys[pygame.K_UP] and car.hitbox.y >= car.yvel:
        car.hitbox.y -= car.yvel
    if keys[pygame.K_DOWN] and car.hitbox.y <= win_sz[1] - car.hitbox.width - car.yvel:
        car.hitbox.y += car.yvel
            
    redrawGameWindow()

pygame.quit()