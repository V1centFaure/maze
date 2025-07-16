import pygame

class Slider:
    def __init__(self, pos, size, min_val, max_val, initial_val):
        self.pos = pos  # (x, y)
        self.size = size  # (width, height)
        self.min = min_val
        self.max = max_val
        self.value = initial_val
        self.slider_rect = pygame.Rect(pos[0], pos[1], size[0], size[1])
        self.knob_radius = size[1] // 2 + 2
        self.dragging = False

    def draw(self, screen):
        # Draw bar
        pygame.draw.rect(screen, (180, 180, 180), self.slider_rect)
        # Position du bouton
        knob_x = int(self.pos[0] + ((self.value - self.min) / (self.max - self.min)) * self.size[0])
        knob_y = self.pos[1] + self.size[1] // 2
        pygame.draw.circle(screen, (70, 70, 255), (knob_x, knob_y), self.knob_radius)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            knob_x = int(self.pos[0] + ((self.value - self.min) / (self.max - self.min)) * self.size[0])
            knob_y = self.pos[1] + self.size[1] // 2
            # Si clic sur le bouton
            if (mouse_x - knob_x) ** 2 + (mouse_y - knob_y) ** 2 < self.knob_radius ** 2:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mouse_x = event.pos[0]
            rel_x = min(max(mouse_x, self.pos[0]), self.pos[0] + self.size[0])
            percent = (rel_x - self.pos[0]) / self.size[0]
            self.value = int(self.min + percent * (self.max - self.min))

    def get_value(self):
        return self.value