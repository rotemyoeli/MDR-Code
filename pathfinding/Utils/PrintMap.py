import pygame
from pygame.locals import *

# NEW CODE
RED = (255, 0, 0)
GRAY = (150, 150, 150)
pygame.init()
w, h = 600, 860
scale = 1


screen = pygame.display.set_mode((w, h))
running = True
img0 = pygame.image.load('manhattan-map-1.png')
img0.convert()
img = img0

rect = img.get_rect()
rect.center = w // 2, h // 2
moving = False
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == MOUSEBUTTONDOWN:
            if rect.collidepoint(event.pos):
                moving = True
        elif event.type == MOUSEBUTTONUP:
            moving = False
        elif event.type == MOUSEMOTION and moving:
            rect.move_ip(event.rel)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_MINUS:
                scale /= 1.1
            elif event.key == pygame.K_EQUALS:
                scale *= 1.1
            img = pygame.transform.rotozoom(img0, 0, scale)
    screen.fill(GRAY)
    screen.blit(img, rect)
    pygame.draw.rect(screen, RED, rect, 1)
    pygame.display.update()
pygame.quit()
# END Of New Code