"""
Slider widget module for Pygame graphical interface.

This module provides a Slider class for creating slider controls
to select values within a given range via an interactive
graphical interface.
"""

import pygame


class Slider:
    """
    Interactive slider widget for Pygame.
    
    Allows the user to select a value within a defined range
    by dragging a knob along a horizontal bar.
    """
    
    def __init__(self, pos, size, min_val, max_val, initial_val):
        """
        Initialize a new slider.
        
        Args:
            pos (tuple): Position (x, y) of the slider's top-left corner
            size (tuple): Dimensions (width, height) of the slider
            min_val (int/float): Minimum slider value
            max_val (int/float): Maximum slider value
            initial_val (int/float): Initial slider value
        """
        self.pos = pos          # Slider position (x, y)
        self.size = size        # Slider dimensions (width, height)
        self.min = min_val      # Minimum value
        self.max = max_val      # Maximum value
        self.value = initial_val # Current value
        
        # Rectangle representing the slider bar
        self.slider_rect = pygame.Rect(pos[0], pos[1], size[0], size[1])
        
        # Control knob radius
        self.knob_radius = size[1] // 2 + 2
        
        # Knob dragging state
        self.dragging = False

    def draw(self, screen):
        """
        Draw the slider on the screen.
        
        Displays the gray background bar and blue control knob
        at the position corresponding to the current value.
        
        Args:
            screen: Pygame surface on which to draw the slider
        """
        # Draw background bar
        pygame.draw.rect(screen, (180, 180, 180), self.slider_rect)
        
        # Calculate knob position based on current value
        if self.max != self.min:  # Avoid division by zero
            knob_x = int(self.pos[0] + ((self.value - self.min) / (self.max - self.min)) * self.size[0])
        else:
            knob_x = self.pos[0]
        
        knob_y = self.pos[1] + self.size[1] // 2
        
        # Draw control knob
        pygame.draw.circle(screen, (70, 70, 255), (knob_x, knob_y), self.knob_radius)

    def handle_event(self, event):
        """
        Handle mouse events for slider interaction.
        
        Detects clicks on the knob, dragging and release
        to update the slider value accordingly.
        
        Args:
            event: Pygame event to process
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Get mouse position
            mouse_x, mouse_y = event.pos
            
            # Calculate current knob position
            if self.max != self.min:
                knob_x = int(self.pos[0] + ((self.value - self.min) / (self.max - self.min)) * self.size[0])
            else:
                knob_x = self.pos[0]
            knob_y = self.pos[1] + self.size[1] // 2
            
            # Check if click is on knob (circular detection)
            distance_squared = (mouse_x - knob_x) ** 2 + (mouse_y - knob_y) ** 2
            if distance_squared < self.knob_radius ** 2:
                self.dragging = True
                
        elif event.type == pygame.MOUSEBUTTONUP:
            # Stop dragging when mouse button is released
            self.dragging = False
            
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            # Update value during dragging
            mouse_x = event.pos[0]
            
            # Constrain x position within slider bounds
            rel_x = min(max(mouse_x, self.pos[0]), self.pos[0] + self.size[0])
            
            # Calculate position percentage on slider
            if self.size[0] > 0:  # Avoid division by zero
                percent = (rel_x - self.pos[0]) / self.size[0]
                
                # Convert percentage to value in [min, max] range
                self.value = int(self.min + percent * (self.max - self.min))

    def get_value(self):
        """
        Return the current slider value.
        
        Returns:
            int/float: Current slider value
        """
        return self.value
    
    def set_value(self, value):
        """
        Set a new value for the slider.
        
        The value is automatically constrained within the [min, max] range.
        
        Args:
            value (int/float): New value to assign to the slider
        """
        self.value = max(self.min, min(self.max, value))
