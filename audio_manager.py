import pygame


class AudioManager:
    def __init__(self, pop_path, music_path):
        pygame.mixer.init()
        pygame.init()
        self.pop_sound = pygame.mixer.Sound(pop_path)
        self.music_sound = pygame.mixer.Sound(music_path)
        self.pop_channel = pygame.mixer.Channel(1)
        self.music_channel = pygame.mixer.Channel(2)
        self.start_time_ms = None

    def ensure_music(self):
        if not self.music_channel.get_busy():
            self.music_channel.play(self.music_sound)
            self.start_time_ms = pygame.time.get_ticks()
        return self.start_time_ms

    def play_pop(self):
        if not self.pop_channel.get_busy():
            self.pop_channel.play(self.pop_sound)
