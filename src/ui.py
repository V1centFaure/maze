import pygame

class NumericStepper:
    def __init__(self, pos, size, min_val=0, max_val=100, initial=0, step=1):
        self.rect = pygame.Rect(pos[0], pos[1], size[0], size[1])
        self.value = initial
        self.min = min_val
        self.max = max_val
        self.step = step

        # Définir les rectangles pour les flèches
        arrow_w, arrow_h = size[1], size[1] // 2
        self.up_rect = pygame.Rect(self.rect.right, self.rect.top, arrow_w, arrow_h)
        self.down_rect = pygame.Rect(self.rect.right, self.rect.bottom - arrow_h, arrow_w, arrow_h)

        self.font = pygame.font.SysFont(None, size[1] - 4)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.up_rect.collidepoint(event.pos):
                self.value = min(self.value + self.step, self.max)
            elif self.down_rect.collidepoint(event.pos):
                self.value = max(self.value - self.step, self.min)

    def draw(self, screen):
        # Fond du champ numérique
        pygame.draw.rect(screen, (230, 230, 230), self.rect)
        pygame.draw.rect(screen, (100,100,100), self.rect, 2)
        # Valeur numérique
        txt = self.font.render(str(self.value), True, (0,0,0))
        screen.blit(txt, (self.rect.x + 5, self.rect.y + 2))
        # Flèche haut
        pygame.draw.rect(screen, (200,200,250), self.up_rect)
        pygame.draw.polygon(screen, (70,70,200), [
            (self.up_rect.centerx, self.up_rect.top + 3),
            (self.up_rect.left + 5, self.up_rect.bottom - 3),
            (self.up_rect.right - 5, self.up_rect.bottom - 3)
        ])
        # Flèche bas
        pygame.draw.rect(screen, (200,200,250), self.down_rect)
        pygame.draw.polygon(screen, (70,70,200), [
            (self.down_rect.left + 5, self.down_rect.top + 3),
            (self.down_rect.right - 5, self.down_rect.top + 3),
            (self.down_rect.centerx, self.down_rect.bottom - 3)
        ])

    def get_value(self):
        return self.value