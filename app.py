import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Initialize MediaPipe pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# Store 3D landmarks across frames
landmark_3d_list = []

# Load your video
cap = cv2.VideoCapture('Archery3.mp4')  # Replace with your video path

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    # Use original frame for pose detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks and results.pose_world_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        pose_landmarks = results.pose_world_landmarks.landmark
        current_frame_kps = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks])
        landmark_3d_list.append(current_frame_kps)

    # Optionally flip for display only
    display_frame = cv2.flip(frame, 1)
    cv2.imshow("Archery Pose Tracking", display_frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ----------- ANALYSIS FUNCTIONS ------------ #

def plot_3d_frame(points, title="3D Pose Visualization"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points = np.array(points)

    xs, ys, zs = points[:, 0], points[:, 2], -points[:, 1]  # YZ flipped for natural view
    ax.scatter(xs, ys, zs, c='r')

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    plt.show()

def shoulder_alignment_error(landmarks):

    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # 3D vector from left to right shoulder
    vec_3d = np.array([
        right_shoulder[0] - left_shoulder[0],
        right_shoulder[1] - left_shoulder[1],
        right_shoulder[2] - left_shoulder[2]
    ])

    # Calculate horizontal (XZ) angle for alignment
    vec_xz = np.array([vec_3d[0], vec_3d[2]])
    angle_xz = np.degrees(np.arctan2(vec_xz[1], vec_xz[0]))

    print(f"Shoulder line 3D XZ angle: {angle_xz:.2f} degrees")

    if abs(angle_xz) > 10:
        print("⚠️  Posture Alert: Shoulders are misaligned in horizontal plane!")
    else:
        print("✅ Good shoulder alignment in horizontal plane.")

    # Optional: Visualize shoulder line in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(left_shoulder[0], left_shoulder[2], -left_shoulder[1], c='b', label='Left Shoulder')
    ax.scatter(right_shoulder[0], right_shoulder[2], -right_shoulder[1], c='g', label='Right Shoulder')
    ax.plot([left_shoulder[0], right_shoulder[0]],
            [left_shoulder[2], right_shoulder[2]],
            [-left_shoulder[1], -right_shoulder[1]], 'r-', label='Shoulder Line')
    ax.set_title('Shoulder Alignment (3D)')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.legend()
    plt.show()

def detect_archery_phase(landmarks):
    lh = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    rh = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

    draw_distance = np.linalg.norm(np.array(rh[:2]) - np.array(ls[:2]))

    if draw_distance < 0.15:
        return "Nocking / Set-up"
    elif draw_distance > 0.4 and draw_distance < 0.6:
        return "Draw Phase"
    elif draw_distance >= 0.6:
        return "Anchor / Aiming"
    else:
        return "Release / Follow-through"

# ----------- APPLY TO ONE FRAME ------------ #


# Select the last available frame for analysis
if len(landmark_3d_list) > 0:
    frame_index = len(landmark_3d_list) - 1
    sample = landmark_3d_list[frame_index]

    print("\n--- ARCHERY FORM ANALYSIS ---")
    shoulder_alignment_error(sample)
    phase = detect_archery_phase(sample)
    print(f"Detected Phase: {phase}")

    plot_3d_frame(sample, title=f"3D Skeleton - {phase}")
else:
    print("No valid frames found for analysis.")
