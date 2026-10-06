import cv2
import mediapipe as mp
import pyautogui
import time
import math


# ============================================================
# CONFIGURATION
# ============================================================

# ---------------- Camera ----------------

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


# ---------------- Active Area / ROI ----------------

FRAME_MARGIN = 90


# ---------------- Cursor ----------------

BASE_SMOOTHING = 7
CURSOR_SENSITIVITY = 1.30
DEADZONE = 3

SMOOTHING_FAST = 3
SMOOTHING_MEDIUM = 5
SMOOTHING_SLOW = 8

FAST_DISTANCE = 70
MEDIUM_DISTANCE = 25


# ---------------- Gesture ----------------

# Pinch threshold is relative to palm size.
# This makes the system adaptive when hand moves
# closer/farther from the camera.
PINCH_RATIO = 0.38

# Minimum distance from camera required for gesture detection
MIN_PALM_SIZE = 25


# ---------------- Click ----------------

CLICK_COOLDOWN = 0.25
RIGHT_CLICK_COOLDOWN = 0.50


# ---------------- Scroll ----------------

SCROLL_AMOUNT = 7
SCROLL_COOLDOWN = 0.15
THUMB_SCROLL_THRESHOLD = 50


# ---------------- Display ----------------

SHOW_LANDMARKS = True
SHOW_ROI = True
SHOW_FPS = True
SHOW_HAND_LABEL = True
SHOW_GESTURE = True


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

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Error: Could not open video device."
        )

    return cap


# ============================================================
# FRAME PROCESSING
# ============================================================

def process_frame(frame):

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # BGR -> RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    return frame, rgb_frame


# ============================================================
# DRAW HAND LANDMARKS
# ============================================================

def draw_landmarks(
    frame,
    hands,
    drawing_utils
):

    last_landmarks = None

    for hand in hands:

        last_landmarks = hand.landmark

        if SHOW_LANDMARKS:

            drawing_utils.draw_landmarks(
                frame,
                hand
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

    for landmark_id, landmark in enumerate(
        landmarks
    ):

        x = int(
            landmark.x * frame_width
        )

        y = int(
            landmark.y * frame_height
        )

        coords[landmark_id] = (
            x,
            y
        )

    return coords


# ============================================================
# DRAW ACTIVE AREA
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

    min_x = FRAME_MARGIN
    max_x = frame_width - FRAME_MARGIN

    min_y = FRAME_MARGIN
    max_y = frame_height - FRAME_MARGIN

    for landmark_id, (x, y) in coords.items():

        # Keep inside active area
        x = max(
            min_x,
            min(x, max_x)
        )

        y = max(
            min_y,
            min(y, max_y)
        )

        # Camera -> Screen
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

        # Safety limits
        mapped_x = max(
            0,
            min(
                mapped_x,
                screen_width - 1
            )
        )

        mapped_y = max(
            0,
            min(
                mapped_y,
                screen_height - 1
            )
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
    plocy
):

    index_x, index_y = index_coords

    # Difference
    dx = index_x - plocx
    dy = index_y - plocy

    # Deadzone
    if abs(dx) < DEADZONE:
        dx = 0

    if abs(dy) < DEADZONE:
        dy = 0

    # Sensitivity
    dx *= CURSOR_SENSITIVITY
    dy *= CURSOR_SENSITIVITY

    # Movement distance
    distance = math.hypot(
        dx,
        dy
    )

    # Adaptive smoothing
    if distance < MEDIUM_DISTANCE:

        smoothing = SMOOTHING_SLOW

    elif distance < FAST_DISTANCE:

        smoothing = SMOOTHING_MEDIUM

    else:

        smoothing = SMOOTHING_FAST

    # Smooth movement
    clocx = (
        plocx
        + dx / smoothing
    )

    clocy = (
        plocy
        + dy / smoothing
    )

    # Move mouse
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
    click_active,
    right_click_active,
    dragging,
    last_click_time,
    last_right_click_time,
    last_scroll_time
):

    current_time = time.time()

    gesture_name = "None"

    # ========================================================
    # DISTANCE FUNCTION
    # ========================================================

    def distance(
        point1,
        point2
    ):

        return math.hypot(
            point1[0] - point2[0],
            point1[1] - point2[1]
        )

    # ========================================================
    # PALM SIZE
    # ========================================================

    palm_size = distance(
        coords[0],   # Wrist
        coords[9]    # Middle MCP
    )

    # Hand too small / too far
    if palm_size < MIN_PALM_SIZE:

        return (
            click_active,
            right_click_active,
            dragging,
            last_click_time,
            last_right_click_time,
            last_scroll_time,
            gesture_name
        )

    # ========================================================
    # ADAPTIVE PINCH THRESHOLD
    # ========================================================

    pinch_threshold = (
        palm_size * PINCH_RATIO
    )

    # ========================================================
    # THUMB
    # ========================================================

    thumb = coords[4]

    # ========================================================
    # FINGER DISTANCES
    # ========================================================

    index_distance = distance(
        thumb,
        coords[8]
    )

    middle_distance = distance(
        thumb,
        coords[12]
    )

    ring_distance = distance(
        thumb,
        coords[16]
    )

    # ========================================================
    # PINCH STATES
    # ========================================================

    index_pinch = (
        index_distance
        < pinch_threshold
    )

    middle_pinch = (
        middle_distance
        < pinch_threshold
    )

    ring_pinch = (
        ring_distance
        < pinch_threshold
    )

    # ========================================================
    # GESTURE PRIORITY
    #
    # Drag
    #   ↓
    # Right Click
    #   ↓
    # Left Click
    #   ↓
    # Scroll
    # ========================================================


    # ========================================================
    # DRAG
    # Thumb + Ring
    # ========================================================

    if ring_pinch:

        gesture_name = "DRAG"

        if not dragging:

            pyautogui.mouseDown()

            dragging = True

        # Disable click states while dragging
        click_active = False
        right_click_active = False

        return (
            click_active,
            right_click_active,
            dragging,
            last_click_time,
            last_right_click_time,
            last_scroll_time,
            gesture_name
        )

    else:

        if dragging:

            pyautogui.mouseUp()

            dragging = False


    # ========================================================
    # RIGHT CLICK
    # Thumb + Middle
    # ========================================================

    if middle_pinch:

        gesture_name = "RIGHT CLICK"

        if not right_click_active:

            if (
                current_time
                - last_right_click_time
                > RIGHT_CLICK_COOLDOWN
            ):

                pyautogui.rightClick()

                last_right_click_time = (
                    current_time
                )

                right_click_active = True

        click_active = False

        return (
            click_active,
            right_click_active,
            dragging,
            last_click_time,
            last_right_click_time,
            last_scroll_time,
            gesture_name
        )

    else:

        right_click_active = False


    # ========================================================
    # LEFT CLICK
    # Thumb + Index
    # ========================================================

    if index_pinch:

        gesture_name = "LEFT CLICK"

        if not click_active:

            if (
                current_time
                - last_click_time
                > CLICK_COOLDOWN
            ):

                pyautogui.click()

                last_click_time = (
                    current_time
                )

                click_active = True

    else:

        click_active = False


    # ========================================================
    # FINGER FOLD DETECTION
    # ========================================================

    index_folded = (
        coords[8][1]
        > coords[6][1]
    )

    middle_folded = (
        coords[12][1]
        > coords[10][1]
    )

    ring_folded = (
        coords[16][1]
        > coords[14][1]
    )

    pinky_folded = (
        coords[20][1]
        > coords[18][1]
    )

    all_fingers_folded = (
        index_folded
        and middle_folded
        and ring_folded
        and pinky_folded
    )


    # ========================================================
    # SCROLL
    # ========================================================

    if (
        all_fingers_folded
        and not index_pinch
        and not middle_pinch
        and not ring_pinch
    ):

        thumb_y = coords[4][1]
        wrist_y = coords[0][1]

        # ----------------------------------------------------
        # THUMB UP
        # ----------------------------------------------------

        if (
            thumb_y
            < wrist_y - THUMB_SCROLL_THRESHOLD
        ):

            gesture_name = "SCROLL UP"

            if (
                current_time
                - last_scroll_time
                > SCROLL_COOLDOWN
            ):

                pyautogui.scroll(
                    SCROLL_AMOUNT
                )

                last_scroll_time = (
                    current_time
                )

        # ----------------------------------------------------
        # THUMB DOWN
        # ----------------------------------------------------

        elif (
            thumb_y
            > wrist_y + THUMB_SCROLL_THRESHOLD
        ):

            gesture_name = "SCROLL DOWN"

            if (
                current_time
                - last_scroll_time
                > SCROLL_COOLDOWN
            ):

                pyautogui.scroll(
                    -SCROLL_AMOUNT
                )

                last_scroll_time = (
                    current_time
                )

    return (
        click_active,
        right_click_active,
        dragging,
        last_click_time,
        last_right_click_time,
        last_scroll_time,
        gesture_name
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

    for i, line in enumerate(
        instructions
    ):

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
# FPS
# ============================================================

def draw_fps(
    frame,
    fps
):

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
# HAND INFORMATION
# ============================================================

def draw_hand_info(
    frame,
    hand_label,
    gesture_name
):

    if SHOW_HAND_LABEL:

        cv2.putText(
            frame,
            f"Hand: {hand_label}",
            (10, 275),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    if SHOW_GESTURE:

        cv2.putText(
            frame,
            f"Gesture: {gesture_name}",
            (10, 305),
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

    # ========================================================
    # CAMERA
    # ========================================================

    cap = init_webcam()


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    hand_detector = mp.solutions.hands.Hands(

        static_image_mode=False,

        # One hand is enough.
        # Either Left OR Right hand can control the mouse.
        max_num_hands=1,

        model_complexity=0,

        min_detection_confidence=0.6,

        min_tracking_confidence=0.6
    )

    drawing_utils = (
        mp.solutions.drawing_utils
    )


    # ========================================================
    # SCREEN
    # ========================================================

    screen_width, screen_height = (
        pyautogui.size()
    )


    # ========================================================
    # CURSOR STATE
    # ========================================================

    plocx = screen_width / 2
    plocy = screen_height / 2


    # ========================================================
    # GESTURE STATE
    # ========================================================

    click_active = False

    right_click_active = False

    dragging = False

    last_click_time = 0

    last_right_click_time = 0

    last_scroll_time = 0


    # ========================================================
    # FPS
    # ========================================================

    previous_time = time.time()

    fps = 0


    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while True:

            # ==================================================
            # READ CAMERA
            # ==================================================

            ret, frame = cap.read()

            if not ret:

                print(
                    "Error: Failed to capture image."
                )

                break


            # ==================================================
            # PROCESS FRAME
            # ==================================================

            frame, rgb_frame = (
                process_frame(frame)
            )

            frame_height, frame_width, _ = (
                frame.shape
            )


            # ==================================================
            # ACTIVE AREA
            # ==================================================

            draw_active_area(
                frame,
                frame_width,
                frame_height
            )


            # ==================================================
            # MEDIAPIPE
            # ==================================================

            output = hand_detector.process(
                rgb_frame
            )

            hands = (
                output.multi_hand_landmarks
            )

            handedness = (
                output.multi_handedness
            )


            # ==================================================
            # HAND FOUND
            # ==================================================

            gesture_name = "None"

            if hands:

                # ----------------------------------------------
                # GET LANDMARKS
                # ----------------------------------------------

                landmarks = draw_landmarks(
                    frame,
                    hands,
                    drawing_utils
                )


                if landmarks is not None:

                    # ------------------------------------------
                    # RAW CAMERA COORDINATES
                    # ------------------------------------------

                    coords = (
                        get_landmark_coordinates(
                            landmarks,
                            frame_width,
                            frame_height
                        )
                    )


                    # ------------------------------------------
                    # HAND LABEL
                    # ------------------------------------------

                    hand_label = "Unknown"

                    if handedness:

                        hand_label = (
                            handedness[0]
                            .classification[0]
                            .label
                        )


                    # ------------------------------------------
                    # SCREEN COORDINATES
                    #
                    # ONLY used for cursor movement.
                    # ------------------------------------------

                    mapped_coords = (
                        map_to_screen(
                            coords,
                            screen_width,
                            screen_height,
                            frame_width,
                            frame_height
                        )
                    )


                    # ------------------------------------------
                    # MOVE CURSOR
                    #
                    # Index fingertip = landmark 8
                    # ------------------------------------------

                    clocx, clocy = (
                        move_cursor(
                            mapped_coords[8],
                            plocx,
                            plocy
                        )
                    )

                    plocx = clocx
                    plocy = clocy


                    # ------------------------------------------
                    # GESTURE DETECTION
                    #
                    # IMPORTANT:
                    # RAW coords are passed here.
                    #
                    # NOT mapped_coords.
                    # ------------------------------------------

                    (
                        click_active,
                        right_click_active,
                        dragging,
                        last_click_time,
                        last_right_click_time,
                        last_scroll_time,
                        gesture_name
                    ) = detect_gestures(

                        coords,

                        click_active,

                        right_click_active,

                        dragging,

                        last_click_time,

                        last_right_click_time,

                        last_scroll_time
                    )


                    # ------------------------------------------
                    # DISPLAY HAND + GESTURE
                    # ------------------------------------------

                    draw_hand_info(
                        frame,
                        hand_label,
                        gesture_name
                    )


            # ==================================================
            # INSTRUCTIONS
            # ==================================================

            add_user_instructions(
                frame
            )


            # ==================================================
            # FPS
            # ==================================================

            current_time = time.time()

            time_difference = (
                current_time
                - previous_time
            )

            if time_difference > 0:

                current_fps = (
                    1 / time_difference
                )

                fps = (
                    0.9 * fps
                    + 0.1 * current_fps
                )

            previous_time = current_time

            draw_fps(
                frame,
                fps
            )


            # ==================================================
            # DISPLAY
            # ==================================================

            cv2.imshow(
                "Virtual Mouse",
                frame
            )


            # ==================================================
            # ESC
            # ==================================================

            if (
                cv2.waitKey(1) & 0xFF
                == 27
            ):

                break


    finally:

        # ======================================================
        # RELEASE MOUSE IF DRAGGING
        # ======================================================

        if dragging:

            pyautogui.mouseUp()


        # ======================================================
        # RELEASE CAMERA
        # ======================================================

        cap.release()


        # ======================================================
        # CLOSE WINDOWS
        # ======================================================

        cv2.destroyAllWindows()


        # ======================================================
        # CLOSE MEDIAPIPE
        # ======================================================

        hand_detector.close()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()