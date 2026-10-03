import cv2
import mediapipe as mp
import pyautogui
import time
import math


# ============================================================
# CONFIGURATION
# ============================================================

# Camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Active area / ROI
# Hand movement inside this box controls the full screen.
FRAME_MARGIN = 90

# Cursor smoothing
BASE_SMOOTHING = 7

# Cursor sensitivity
CURSOR_SENSITIVITY = 1.30

# Ignore tiny hand movements
DEADZONE = 3


# Adaptive smoothing
SMOOTHING_FAST = 3
SMOOTHING_MEDIUM = 5
SMOOTHING_SLOW = 8

# Distance thresholds for adaptive smoothing
FAST_DISTANCE = 70
MEDIUM_DISTANCE = 25

# Gesture thresholds
CLICK_THRESHOLD_Y = 45
THUMB_UP_DOWN_THRESHOLD = 50

# Gesture cooldowns
RIGHT_CLICK_COOLDOWN = 0.5
SCROLL_COOLDOWN = 0.10

# Scroll
SCROLL_AMOUNT = 7
THUMB_SCROLL_THRESHOLD = 50

# Drawing
CIRCLE_RADIUS = 5
CIRCLE_COLOR = (0, 255, 0)
CIRCLE_THICKNESS = -1

LINE_COLOR = (0, 255, 0)
LINE_THICKNESS = 2

# Display
SHOW_LANDMARKS = True
SHOW_ROI = True
SHOW_FPS = True


# ============================================================
# PYAUTOGUI OPTIMIZATION
# ============================================================

pyautogui.PAUSE = 0
pyautogui.MINIMUM_DURATION = 0


# ============================================================
# WEBCAM INITIALIZATION
# ============================================================

def init_webcam():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video device.")

    return cap


# ============================================================
# FRAME PROCESSING
# ============================================================

def process_frame(frame):
    # Mirror the camera
    frame = cv2.flip(frame, 1)

    # OpenCV uses BGR, MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame, rgb_frame


# ============================================================
# DRAW HAND LANDMARKS
# ============================================================

def draw_landmarks(frame, hands, drawing_utils):

    last_landmarks = None

    for hand in hands:

        last_landmarks = hand.landmark

        if SHOW_LANDMARKS:

            # Draw MediaPipe's standard hand connections
            drawing_utils.draw_landmarks(
                frame,
                hand
            )

            # Draw landmark circles
            for landmark in last_landmarks:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    CIRCLE_RADIUS,
                    CIRCLE_COLOR,
                    CIRCLE_THICKNESS
                )

    return last_landmarks


# ============================================================
# GET LANDMARK COORDINATES
# ============================================================

def get_landmark_coordinates(
    landmarks,
    frame_width,
    frame_height
):

    coords = {}

    for landmark_id, landmark in enumerate(landmarks):

        x = int(
            landmark.x * frame_width
        )

        y = int(
            landmark.y * frame_height
        )

        coords[landmark_id] = (x, y)

    return coords


# ============================================================
# DRAW ACTIVE ROI
# ============================================================

def draw_active_area(
    frame,
    frame_width,
    frame_height
):

    if not SHOW_ROI:
        return

    cv2.rectangle(
        frame,
        (
            FRAME_MARGIN,
            FRAME_MARGIN
        ),
        (
            frame_width - FRAME_MARGIN,
            frame_height - FRAME_MARGIN
        ),
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Active Area",
        (
            FRAME_MARGIN + 5,
            FRAME_MARGIN - 10
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
        cv2.LINE_AA
    )


# ============================================================
# MAP CAMERA COORDINATES TO SCREEN
# ============================================================

def map_to_screen(
    coords,
    screen_width,
    screen_height,
    frame_width,
    frame_height
):

    mapped_coords = {}

    # Active camera region
    min_x = FRAME_MARGIN
    max_x = frame_width - FRAME_MARGIN

    min_y = FRAME_MARGIN
    max_y = frame_height - FRAME_MARGIN

    for landmark_id, (x, y) in coords.items():

        # Keep coordinates inside active area
        x = max(min_x, min(x, max_x))
        y = max(min_y, min(y, max_y))

        # Camera ROI -> Full screen
        mapped_x = (
            (x - min_x)
            / (max_x - min_x)
            * screen_width
        )

        mapped_y = (
            (y - min_y)
            / (max_y - min_y)
            * screen_height
        )

        # Keep cursor safely inside screen
        mapped_x = max(
            0,
            min(mapped_x, screen_width - 1)
        )

        mapped_y = max(
            0,
            min(mapped_y, screen_height - 1)
        )

        mapped_coords[landmark_id] = (
            mapped_x,
            mapped_y
        )

    return mapped_coords


# ============================================================
# MOVE CURSOR
# ============================================================

def move_cursor(
    index_coords,
    plocx,
    plocy,
    smoothening
):

    index_x, index_y = index_coords

    # Difference between target and previous cursor position
    dx = index_x - plocx
    dy = index_y - plocy

    # --------------------------------------------------------
    # DEADZONE
    # --------------------------------------------------------

    if abs(dx) < DEADZONE:
        dx = 0

    if abs(dy) < DEADZONE:
        dy = 0

    # --------------------------------------------------------
    # APPLY CURSOR SENSITIVITY
    # --------------------------------------------------------
    # Makes cursor travel farther for the same hand movement.

    dx *= CURSOR_SENSITIVITY
    dy *= CURSOR_SENSITIVITY

    # --------------------------------------------------------
    # MOVEMENT DISTANCE
    # --------------------------------------------------------

    distance = math.sqrt(
        dx * dx + dy * dy
    )

    # --------------------------------------------------------
    # ADAPTIVE SMOOTHING
    # --------------------------------------------------------

    if distance < MEDIUM_DISTANCE:

        current_smoothing = SMOOTHING_SLOW

    elif distance < FAST_DISTANCE:

        current_smoothing = SMOOTHING_MEDIUM

    else:

        current_smoothing = SMOOTHING_FAST

    # --------------------------------------------------------
    # SMOOTH CURSOR MOVEMENT
    # --------------------------------------------------------

    clocx = (
        plocx
        + dx / current_smoothing
    )

    clocy = (
        plocy
        + dy / current_smoothing
    )

    # --------------------------------------------------------
    # MOVE ACTUAL MOUSE
    # --------------------------------------------------------

    pyautogui.moveTo(
        int(clocx),
        int(clocy),
        _pause=False
    )

    return clocx, clocy

# ============================================================
# GESTURE DETECTION
# ============================================================

def detect_gestures(
    coords,
    thumb_coords,
    click_time,
    click_threshold,
    single_click_flag,
    left_dragging,
    last_right_click_time,
    last_scroll_time
):

    thumb_x, thumb_y = thumb_coords

    current_time = time.time()

    # ========================================================
    # LEFT CLICK
    # ========================================================

    index_thumb_y_distance = abs(
        coords[8][1] - thumb_y
    )

    if index_thumb_y_distance < CLICK_THRESHOLD_Y:

        if (
            current_time - click_time
            < click_threshold
        ):

            pyautogui.doubleClick()

            click_time = 0

        else:

            if not single_click_flag:

                pyautogui.click()

                single_click_flag = True

            click_time = current_time

    else:

        single_click_flag = False

    # ========================================================
    # DRAG
    # ========================================================

    ring_thumb_y_distance = abs(
        coords[16][1] - thumb_y
    )

    if ring_thumb_y_distance < CLICK_THRESHOLD_Y:

        if not left_dragging:

            pyautogui.mouseDown()

            left_dragging = True

    else:

        if left_dragging:

            pyautogui.mouseUp()

            left_dragging = False

    # ========================================================
    # RIGHT CLICK
    # ========================================================

    middle_thumb_y_distance = abs(
        coords[12][1] - thumb_y
    )

    if middle_thumb_y_distance < CLICK_THRESHOLD_Y:

        if (
            current_time - last_right_click_time
            > RIGHT_CLICK_COOLDOWN
        ):

            pyautogui.rightClick()

            last_right_click_time = current_time

   # ========================================================
   # SCROLL GESTURE DETECTION
   # ========================================================

   # Check whether four main fingers are folded
    index_folded = coords[8][1] > coords[6][1]
    middle_folded = coords[12][1] > coords[10][1]
    ring_folded = coords[16][1] > coords[14][1]
    pinky_folded = coords[20][1] > coords[18][1]

    all_fingers_folded = (
        index_folded
        and middle_folded
        and ring_folded
        and pinky_folded
    )

    # ========================================================
    # THUMBS UP / DOWN
    # ========================================================

    if all_fingers_folded:

        thumb_tip_y = coords[4][1]
        wrist_y = coords[0][1]

    # ----------------------------------------------------
    # THUMBS UP → SCROLL UP
    # ----------------------------------------------------

        if thumb_tip_y < wrist_y - THUMB_SCROLL_THRESHOLD:

            if  current_time - last_scroll_time > SCROLL_COOLDOWN:
                pyautogui.scroll(SCROLL_AMOUNT)
                last_scroll_time = current_time

    # ----------------------------------------------------
    # THUMBS DOWN → SCROLL DOWN
    # ----------------------------------------------------

        elif thumb_tip_y > wrist_y + THUMB_SCROLL_THRESHOLD:

            if current_time - last_scroll_time > SCROLL_COOLDOWN:
                pyautogui.scroll( -SCROLL_AMOUNT  )
                last_scroll_time = current_time

    return (
        click_time,
        single_click_flag,
        left_dragging,
        last_right_click_time,
        last_scroll_time
    )


# ============================================================
# USER INSTRUCTIONS
# ============================================================

def add_user_instructions(frame):

    instructions = [

        "Virtual Mouse",

        "Index Finger  : Move Cursor",

        "Thumb + Index : Left Click",

        "Thumb + Middle: Right Click",

        "Thumb + Ring  : Drag",

        "Thumb Up/Down : Scroll",

        "ESC           : Exit"

    ]

    y0 = 25
    dy = 28

    for i, line in enumerate(instructions):

        y = y0 + i * dy

        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


# ============================================================
# FPS CALCULATION
# ============================================================

def draw_fps(frame, fps):

    if not SHOW_FPS:
        return

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 245),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # INITIALIZE CAMERA
    # --------------------------------------------------------

    cap = init_webcam()

    # --------------------------------------------------------
    # INITIALIZE MEDIAPIPE
    # --------------------------------------------------------

    hand_detector = mp.solutions.hands.Hands(

        static_image_mode=False,

        max_num_hands=1,

        model_complexity=0,

        min_detection_confidence=0.6,

        min_tracking_confidence=0.6

    )

    drawing_utils = mp.solutions.drawing_utils

    # --------------------------------------------------------
    # SCREEN SIZE
    # --------------------------------------------------------

    screen_width, screen_height = pyautogui.size()

    # --------------------------------------------------------
    # CURSOR STATE
    # --------------------------------------------------------

    smoothening = BASE_SMOOTHING

    plocx = screen_width / 2
    plocy = screen_height / 2

    # --------------------------------------------------------
    # CLICK STATE
    # --------------------------------------------------------

    click_time = 0

    click_threshold = 0.3

    single_click_flag = False

    # --------------------------------------------------------
    # DRAG STATE
    # --------------------------------------------------------

    left_dragging = False

    # --------------------------------------------------------
    # RIGHT CLICK STATE
    # --------------------------------------------------------

    last_right_click_time = 0

    # --------------------------------------------------------
    # SCROLL STATE
    # --------------------------------------------------------

    last_scroll_time = 0

    # --------------------------------------------------------
    # FPS VARIABLES
    # --------------------------------------------------------

    previous_time = time.time()

    fps = 0

    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while True:

            # ------------------------------------------------
            # READ CAMERA FRAME
            # ------------------------------------------------

            ret, frame = cap.read()

            if not ret:

                print(
                    "Error: Failed to capture image."
                )

                break

            # ------------------------------------------------
            # PROCESS FRAME
            # ------------------------------------------------

            frame, rgb_frame = process_frame(
                frame
            )

            frame_height, frame_width, _ = (
                frame.shape
            )

            # ------------------------------------------------
            # DRAW ACTIVE AREA
            # ------------------------------------------------

            draw_active_area(
                frame,
                frame_width,
                frame_height
            )

            # ------------------------------------------------
            # MEDIAPIPE HAND DETECTION
            # ------------------------------------------------

            output = hand_detector.process(
                rgb_frame
            )

            hands = output.multi_hand_landmarks

            # ------------------------------------------------
            # HAND FOUND
            # ------------------------------------------------

            if hands:

                # --------------------------------------------
                # LANDMARKS
                # --------------------------------------------

                landmarks = draw_landmarks(
                    frame,
                    hands,
                    drawing_utils
                )

                if landmarks is not None:

                    # ----------------------------------------
                    # CAMERA COORDINATES
                    # ----------------------------------------

                    coords = get_landmark_coordinates(
                        landmarks,
                        frame_width,
                        frame_height
                    )

                    # ----------------------------------------
                    # SCREEN COORDINATES
                    # ----------------------------------------

                    mapped_coords = map_to_screen(
                        coords,
                        screen_width,
                        screen_height,
                        frame_width,
                        frame_height
                    )

                    # ----------------------------------------
                    # MOVE CURSOR
                    # ----------------------------------------

                    clocx, clocy = move_cursor(
                        mapped_coords[8],
                        plocx,
                        plocy,
                        smoothening
                    )

                    # ----------------------------------------
                    # SAVE CURRENT POSITION
                    # ----------------------------------------

                    plocx = clocx
                    plocy = clocy

                    # ----------------------------------------
                    # DETECT GESTURES
                    # ----------------------------------------

                    (
                        click_time,
                        single_click_flag,
                        left_dragging,
                        last_right_click_time,
                        last_scroll_time
                    ) = detect_gestures(

                        mapped_coords,

                        mapped_coords[4],

                        click_time,

                        click_threshold,

                        single_click_flag,

                        left_dragging,

                        last_right_click_time,

                        last_scroll_time
                    )

            # ------------------------------------------------
            # USER INSTRUCTIONS
            # ------------------------------------------------

            add_user_instructions(
                frame
            )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            current_time = time.time()

            time_difference = (
                current_time - previous_time
            )

            if time_difference > 0:

                current_fps = (
                    1 / time_difference
                )

                # Smooth FPS display
                fps = (
                    0.9 * fps
                    + 0.1 * current_fps
                )

            previous_time = current_time

            draw_fps(
                frame,
                fps
            )

            # ------------------------------------------------
            # SHOW CAMERA
            # ------------------------------------------------

            cv2.imshow(
                "Virtual Mouse",
                frame
            )

            # ------------------------------------------------
            # ESC TO EXIT
            # ------------------------------------------------

            if cv2.waitKey(1) & 0xFF == 27:

                break

    finally:

        # ----------------------------------------------------
        # RELEASE DRAG IF PROGRAM EXITS DURING DRAG
        # ----------------------------------------------------

        if left_dragging:

            pyautogui.mouseUp()

        # ----------------------------------------------------
        # RELEASE CAMERA
        # ----------------------------------------------------

        cap.release()

        # ----------------------------------------------------
        # CLOSE WINDOWS
        # ----------------------------------------------------

        cv2.destroyAllWindows()

        # ----------------------------------------------------
        # CLOSE MEDIAPIPE
        # ----------------------------------------------------

        hand_detector.close()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()