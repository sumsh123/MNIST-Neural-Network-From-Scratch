import pygame

pygame.init()

WIDTH = 1400
HEIGHT = 800

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Network Visualizer")

clock = pygame.time.Clock()

TITLE = pygame.font.SysFont("Segoe UI", 36, bold=True)
TEXT = pygame.font.SysFont("Segoe UI", 22)

WHITE = (245,245,245)
BLACK = (18,18,18)
PINK = (255,105,180)
LIGHT = (60,60,60)
GRAY = (35,35,35)

drawing_surface = pygame.Surface((350,350))
drawing_surface.fill(BLACK)

running = True

while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

    if pygame.mouse.get_pressed()[0]:

        mx,my = pygame.mouse.get_pos()

        if 40 < mx < 390 and 120 < my < 470:

            pygame.draw.circle(
                drawing_surface,
                WHITE,
                (mx-40,my-120),
                8
            )

    screen.fill((12,12,18))

    title = TITLE.render(
        "Neural Network From Scratch",
        True,
        WHITE
    )

    screen.blit(title,(40,30))

    pygame.draw.rect(
        screen,
        GRAY,
        (40,120,350,350),
        border_radius=20
    )

    screen.blit(
        drawing_surface,
        (40,120)
    )

    pygame.draw.rect(
        screen,
        PINK,
        (40,520,150,50),
        border_radius=12
    )

    pygame.draw.rect(
        screen,
        LIGHT,
        (240,520,150,50),
        border_radius=12
    )

    screen.blit(
        TEXT.render("Predict",True,WHITE),
        (75,535)
    )

    screen.blit(
        TEXT.render("Clear",True,WHITE),
        (285,535)
    )

    pygame.display.flip()

    clock.tick(60)

pygame.quit()