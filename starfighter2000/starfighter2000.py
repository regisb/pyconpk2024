#! /usr/bin/env python
import math
import random
import sys
import typing as t
from time import time

import cv2
import pyray as ray
from cffi import FFI
from ultralytics import YOLO

SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 720
WEBCAM_ID = 0 # 4 for logitech, 0 for laptop
# 78 degrees for the logitech c920s, 70 for my crappy laptop webcam
WEBCAM_FOV = 70


def main():
    if sys.argv[1:2] == ["detect"]:
        detect()
    else:
        play()


def detect():
    """
    Detect people and display results
    """
    while True:
        frame = Player.tracker.read_frame()
        for player_id, x1, y1, x2, y2 in Player.tracker.iter_players_in_frame(frame):
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
            )
            cv2.putText(
                frame,
                f"{player_id}",
                (int(x1), int(y1)),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
            )
        cv2.imshow("I see you :)", frame)
        cv2.waitKey(1)


def play():
    load()
    current_time = time()
    while not ray.window_should_close():
        dt = time() - current_time
        current_time = time()
        update(dt)
        draw()
    ray.close_window()


def load():
    Scene.load()
    Player.load()
    Bonus.load()


def update(dt):
    Player.update_all(dt)
    Bonus.update_all(dt)


def draw():
    ray.begin_drawing()
    ray.clear_background(ray.RAYWHITE)
    ray.begin_mode_3d(Scene.camera)
    Player.draw_all()
    Bonus.draw_all()
    ray.end_mode_3d()
    Player.draw_high_scores()
    ray.end_drawing()


class Scene:
    camera = None

    @classmethod
    def load(cls):
        ray.set_trace_log_level(ray.TraceLogLevel.LOG_ERROR)
        ray.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, "Let's play")
        cls.camera = ray.Camera3D(
            ray.Vector3(0, 0, -1),  # position
            ray.Vector3(0, 0, 0),  # target
            ray.Vector3(0, 1, 0),  # up -- remember y is up!
            WEBCAM_FOV,  # field of view in degrees
            ray.CAMERA_PERSPECTIVE,  # projection
        )


class Bonus:
    model = None
    total_count = 100
    instances = []

    @classmethod
    def load(cls):
        cls.model = ray.load_model("./assets/ball.gltf.glb")

        # generate bonuses
        while len(cls.instances) < cls.total_count:
            angle = random.uniform(-WEBCAM_FOV / 2, WEBCAM_FOV / 2)
            z = random.uniform(2, 30)
            x = (1 + z) * math.tan(angle * math.pi / 180) * SCREEN_WIDTH / SCREEN_HEIGHT
            y = random.uniform(4, 50)
            cls.instances.append(cls(x, y, z))

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.axis = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        self.angle = random.uniform(0, 360)
        self.caught = False

    @classmethod
    def iter_all(cls):
        yield from cls.instances

    @classmethod
    def update_all(cls, dt):
        # update individual bonuses
        for bonus in cls.iter_all():
            bonus.update(dt)

        cls.instances = list(
            filter(lambda b: b.y >= -10 and not b.caught, cls.instances)
        )

    def update(self, dt):
        falling_speed = 2
        rotation_speed = 0.2 * 360
        self.y -= falling_speed * dt
        self.angle += dt * rotation_speed

        for player in Player.iter_visible():
            if (player.x - self.x) ** 2 + (player.y - self.y) ** 2 + (
                player.z - self.z
            ) ** 2 < 1:
                self.caught = True
                player.score += 1
                break

    @classmethod
    def draw_all(cls):
        for bonus in cls.iter_all():
            bonus.draw()

    def draw(self):
        ray.draw_model_ex(
            self.model,
            ray.Vector3(self.x, self.y, self.z),
            ray.Vector3(*self.axis),
            self.angle,
            ray.Vector3(1, 1, 1),  # scale
            ray.WHITE,
        )


class Tracker:

    def __init__(self, webcam: int = 0, model="yolov8n.pt"):
        """
        Awesome object tracking model. We don't even have to think about the model we're
        using, I love it.
        """
        self.webcam = cv2.VideoCapture(webcam)
        self.tracker = YOLO(model)

    def read_frame(self):
        _ret, frame = self.webcam.read()
        return frame

    def iter_players(self) -> t.Iterator["Result"]:
        """
        Find objects of type "person" in webcam frame and yield their normalized coordinates.
        """
        frame = self.read_frame()
        for player_id, x1, y1, x2, y2 in self.iter_players_in_frame(frame):
            yield Result(frame, player_id, x1, y1, x2, y2)

    def iter_players_in_frame(self, frame):
        for results in self.tracker.track(
            frame, stream=True, persist=False, verbose=False
        ):
            for box in results.boxes:
                if results.names[int((box.cls))] == "person" and box.id:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    yield int(box.id), x1, y1, x2, y2


class Result:
    """
    Detection result
    """

    ffi = FFI()

    def __init__(self, frame, player_id, x1, y1, x2, y2):
        # Extract window from frame
        detected = cv2.UMat(frame, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

        # Resize to a fixed square size. This is useful mostly to display scores
        detected = cv2.resize(detected, (400, 400))

        # Rotate, otherwise the texture is applied upside down
        detected = cv2.rotate(detected, cv2.ROTATE_180)

        # Convert detected window to raylib image. This is tricky... we convert to png
        # before re-loading into raylib. If anyone has a better solution, I'm all ears.
        _ret, png = cv2.imencode(".png", detected)
        image = ray.load_image_from_memory(
            ".png",
            self.ffi.cast("unsigned char*", png.ctypes.data),
            png.size,
        )

        # Convert to object
        self.id = player_id
        self.x1 = x1 / frame.shape[1]
        self.y1 = y1 / frame.shape[0]
        self.x2 = x2 / frame.shape[1]
        self.y2 = y2 / frame.shape[0]
        self.image = image


def back_project(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float, float]:
    """
    Back-project coordinates yielded by the Tracker to space coordinates.

    We just make one assumption: that all humans have the same size. Yes, I know, this
    is not reality. Video games are an illusion!
    """
    # This is the human size width and height, in meters
    human_size_x = 0.8
    human_size_y = 1.7

    # tan(FOVx/2)
    tanfovy2 = math.tan(WEBCAM_FOV / 2 * math.pi / 180)
    tanfovx2 = tanfovy2 * SCREEN_WIDTH / SCREEN_HEIGHT

    # Guess the depth first, based on human size
    z = human_size_x * human_size_y / (4 * (x2 - x1) * (y2 - y1) * tanfovx2 * tanfovy2)

    xp = (x2 + x1) / 2 - 1 / 2
    yp = (1 - (y2 + y1)) / 2
    x = 2 * xp * z * tanfovx2
    y = 2 * yp * z * tanfovy2

    return x, y, z


class Player:
    model = None
    instances = {}
    visible = set()
    tracker = Tracker(webcam=WEBCAM_ID)

    @classmethod
    def load(cls):
        cube_mesh = ray.gen_mesh_cube(1, 1, 1)
        cls.model = ray.load_model_from_mesh(cube_mesh)

    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.score = 0
        self.texture = None
        self.image = None

    @classmethod
    def iter_all(cls):
        yield from cls.instances.values()

    @classmethod
    def iter_visible(cls):
        for player_id in cls.visible:
            yield cls.instances[player_id]

    @classmethod
    def update_all(cls, _dt):
        # detect players
        cls.visible.clear()
        for result in cls.tracker.iter_players():
            # add player to set of visible players
            player_id = result.id
            cls.visible.add(player_id)
            if player_id not in cls.instances:
                cls.instances[player_id] = Player()
            player = cls.instances[player_id]
            # update player position and image
            x, y, z = back_project(result.x1, result.y1, result.x2, result.y2)
            player.set_position(x, y, z)
            player.set_image(result.image)

    def set_position(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def set_image(self, image):
        """
        We need to unload the texture and the image, otherwise we have a memory leak.
        """
        if self.texture is not None:
            ray.unload_texture(self.texture)
            ray.unload_image(self.image)
        self.image = image
        self.texture = ray.load_texture_from_image(self.image)

    @classmethod
    def draw_all(cls):
        for player in cls.iter_visible():
            player.draw()

    def draw(self):
        ray.set_material_texture(
            self.model.materials[0], ray.MATERIAL_MAP_DIFFUSE, self.texture
        )
        ray.draw_model(
            self.model,
            ray.Vector3(self.x, self.y, self.z),
            1,  # scale
            ray.WHITE,
        )

    @classmethod
    def draw_high_scores(cls):
        # Note that we display high scores for all players, not just those who are
        # visible
        top_players = sorted(cls.iter_all(), key=lambda p: -p.score)
        for p, player in enumerate(top_players[:10]):
            ray.draw_texture_ex(
                player.texture,
                ray.Vector2(110, (p + 1) * 110),
                180,
                0.25,
                ray.WHITE,
            )
            ray.draw_text(
                f"{player.score}",
                120,
                p * 110 + 20,
                50,
                # ray.GREEN,
                ray.MAGENTA,
                # (92, 9, 91, 255), # pink
            )


if __name__ == "__main__":
    main()
