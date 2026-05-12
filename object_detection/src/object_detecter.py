import pygame
import math
import sys
import numpy as np
import os


class ObstacleUI:
    def __init__(self, visualize=True):

        self.visualize = visualize
        self.running = True
        self.obstacles = []

        self.scale = 250
        self.sector_angle = 30
        self.sector_distance = 2.0
        self.lidar_offset = 0.15

        self.check = 0
        self.check2 = 0

        if not self.visualize:
            return

        pygame.init()
        self.WIDTH, self.HEIGHT = 600, 800
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Obstacle Detection Visualization")

        self.car_x, self.car_y = self.WIDTH // 2, self.HEIGHT - 150
        
        # Find the image file relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, "image", "car_icon.png")
        
        # Fallback to parent directory if not found
        if not os.path.exists(image_path):
            image_path = os.path.join(script_dir, "..", "image", "car_icon.png")
        
        self.car_image = pygame.image.load(image_path)

        car_width = 0.1911 * 2.5
        car_height = 0.337 * 2.5
        car_width_px = int(car_width * self.scale)
        car_height_px = int(car_height * self.scale)
        self.car_image = pygame.transform.scale(self.car_image, (car_width_px, car_height_px))

        self.RED = (220, 80, 80)
        self.GRAY = (200, 200, 200, 100)
        self.BG = (250, 245, 230)
        self.GREEN = (120, 200, 120)
        self.WARNING_TEXT = (255, 0, 0)
        self.DISTANCE_TEXT = (60, 40, 30)
        self.ORANGE = (255, 165, 0)
        self.CLUSTER_BOX = (50, 50, 50)

        self.clock = pygame.time.Clock()

    def lidar_to_screen(self, x, y):
        screen_x = int(self.car_x - y * self.scale)
        screen_y = int((self.car_y - self.lidar_offset * self.scale) - x * self.scale)
        return screen_x, screen_y

    def is_in_warning_zone(self, x, y):
        x_adj = x - self.lidar_offset
        distance = math.sqrt(x_adj ** 2 + y ** 2)
        if distance > self.sector_distance:
            return False
        angle_deg = math.degrees(math.atan2(y, x_adj))
        return -self.sector_angle <= angle_deg <= self.sector_angle

    def is_in_front_box(self, x, y, max_distance=0.2, width=0.19):
        x_adj = x - self.lidar_offset

        return 0 < x_adj < max_distance + self.lidar_offset and abs(y) < width / 2

    def any_dangerous_point(self):
        found = False
        for cluster in self.obstacles:
            for x, y in cluster:
                if self.is_in_front_box(x, y):
                    found = True
                    break
        if not found:
            self.check = False
        return found

    def set_obstacles(self, new_obstacles):
        self.obstacles = new_obstacles

    def run(self, fps=30):

        if not self.visualize:
            return

        self.clock.tick(fps)
        self.screen.fill(self.BG)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False

        self.draw_sector_filled()
        self.draw_car()
        self.draw_obstacles()
        self.draw_warning()
        self.draw_legend()

        pygame.display.flip()

    def draw_car(self):
        car_rect = self.car_image.get_rect(center=(self.car_x, self.car_y))
        self.screen.blit(self.car_image, car_rect)

    def draw_sector_filled(self):
        sector_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        origin_x = self.car_x
        origin_y = self.car_y - (0.337 * self.scale) / 2
        origin_y -= self.lidar_offset * self.scale
        points = [(origin_x, origin_y)]
        for angle in range(-self.sector_angle, self.sector_angle + 1):
            rad = math.radians(angle)
            x = origin_x + self.sector_distance * self.scale * math.sin(rad)
            y = origin_y - self.sector_distance * self.scale * math.cos(rad)
            points.append((x, y))
        pygame.draw.polygon(sector_surface, self.GRAY, points)
        self.screen.blit(sector_surface, (0, 0))

    def draw_obstacles(self):
        for cluster in self.obstacles:
            for x, y in cluster:
                px, py = self.lidar_to_screen(x, y)
                if self.is_in_front_box(x, y):
                    pygame.draw.circle(self.screen, self.RED, (px, py), 6)
                elif self.is_in_warning_zone(x, y):
                    pygame.draw.circle(self.screen, self.ORANGE, (px, py), 5)
                else:
                    pygame.draw.circle(self.screen, self.GREEN, (px, py), 4)
        self.draw_cluster_boxes()

    def draw_cluster_boxes(self):
        for cluster in self.obstacles:
            if not cluster:
                continue
            xs = np.array([p[0] for p in cluster])
            ys = np.array([p[1] for p in cluster])
            min_x, max_x = xs.min(), xs.max()
            min_y, max_y = ys.min(), ys.max()
            top_left = self.lidar_to_screen(min_x, min_y)
            bottom_right = self.lidar_to_screen(max_x, max_y)
            width = bottom_right[0] - top_left[0]
            height = bottom_right[1] - top_left[1]
            pygame.draw.rect(self.screen, self.CLUSTER_BOX, (top_left[0], top_left[1], width, height), 1)

    def draw_warning(self):
        if self.any_dangerous_point():
            font = pygame.font.SysFont("Arial", 32, bold=True)
            text = font.render("WARNING: Obstacle ahead!", True, self.WARNING_TEXT)
            self.screen.blit(text, (self.WIDTH // 2 - 180, 50))

    def draw_legend(self):
        font = pygame.font.SysFont("Arial", 18)
        self.screen.blit(font.render("(RED POINT) Danger", True, self.RED), (20, 20))
        self.screen.blit(font.render("(ORANGE POINT) Warning zone", True, self.ORANGE), (20, 40))
        self.screen.blit(font.render("(GREEN POINT) Safe", True, self.GREEN), (20, 60))

    def is_running(self):
        return self.running

    def shutdown(self):
        self.running = False
        if self.visualize:
            pygame.quit()


def main():
    obstacle_ui = ObstacleUI(visualize=True)
    lidar_data = [[(0.18, -0.1), (0.19, 0), (0.20, 0), (0.21, 0), (0.22, 0)]]
    obstacle_ui.set_obstacles(lidar_data)
    while obstacle_ui.is_running():
        obstacle_ui.run()
    obstacle_ui.shutdown()
    sys.exit()


if __name__ == "__main__":
    main()