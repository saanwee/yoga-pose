import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models, datasets
import torch.nn.functional as F

from collections import deque
import time
import threading
import random

from pose_recommender import PoseRecommender

valid_modes = ["Standing-Only", "Hard", "Class"]
print("Available modes:", ", ".join(valid_modes))
selected_mode = input("Enter the mode you'd like to use: ").strip()

while selected_mode not in valid_modes:
    print("Invalid mode. Please choose from:", ", ".join(valid_modes))
    selected_mode = input("Enter the mode you'd like to use: ").strip()

pose_manager = PoseRecommender(mode=selected_mode)


def achieved_pose(target_pose: int, detected_poses: list[int], recent_poses: deque, threshold: int) -> bool:
    detected = False
    for pose in detected_poses:
        if pose == target_pose:
            detected = True

    if len(recent_poses) >= threshold * 1.75:
        recent_poses.popleft()

    if detected:
        recent_poses.append(True)
    else:
        recent_poses.append(False)

    num_correct = 0
    for pose in recent_poses:
        if pose:
            num_correct += 1
    
    return num_correct >= threshold


def load_model(model, model_path, device, num_classes):
    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    model.to(device)


def get_topk_predictions(model, image, k, transform, device):
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output = output.squeeze()
        probs = F.softmax(output, dim=0)

        topk_probs, topk_classes = torch.topk(probs, k, dim=0)

        return topk_classes.tolist()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2()
    model_path = "models/mobilenet_15_e-3_e-4.pt"
    load_model(model, model_path, device, 12)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    classes = ["Boat Pose", "Chair Pose", "Child Pose", "Downward Facing Dog", "Fish Pose", "Lord of the Dance Pose", "Side Plank Pose", "Sitting Pose", "Tree Pose", "Warrior 3", "Warrior 2", "Warrior 1"]
        
    index_to_class = {i: name for i, name in enumerate(classes)}
    class_to_index = {name: i for i, name in enumerate(classes)}

    cap = cv2.VideoCapture(0)

    total_time = 0
    total_frames = 0

    curr_pose = class_to_index[pose_manager.get_next_pose("Boat Pose")]
    recent_poses = deque()
    waiting = False
    
    def switch_pose(pose):
        global curr_pose, waiting, recent_poses
        print("switched pose called")

        curr_pose = pose
        waiting = False
        recent_poses.clear()

    counter = 0

    while True:
        start = time.time()

        ret, frame = cap.read()

        cv2.putText(frame, f'Target: {classes[curr_pose]}', (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, 
            (255, 255, 255), 5, cv2.LINE_AA)      

        if waiting:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0),
                10, cv2.LINE_AA)

        cv2.imshow('Yoga Pose', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end = time.time()
        total_time += end - start
        total_frames += 1

        # do model inference once every 10 frames
        if counter != 10:
            counter += 1
            continue
        else:
            counter = 0

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predicted_poses = get_topk_predictions(model, frame, 2, transform, device)

        achieved = achieved_pose(curr_pose, predicted_poses, recent_poses, 5)
        if achieved and not waiting:
            curr_pose_name = index_to_class[curr_pose]
            next_pose_name = pose_manager.get_next_pose(curr_pose_name)
            next_pose = class_to_index[next_pose_name]

            print(f"Finished {classes[curr_pose]}, next pose is {classes[next_pose]}")
            timer = threading.Timer(10, switch_pose, args=[next_pose])
            timer.start()

            waiting = True
        
        print(classes[curr_pose], " | " , classes[predicted_poses[0]], " | " , classes[predicted_poses[1]])

    cap.release()
    cv2.destroyAllWindows()

    average_time = 1000 * (total_time / total_frames)
    print("Average time per frame: ", average_time, "ms")
    print(1000 / average_time, "fps")
