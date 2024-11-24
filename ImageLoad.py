import os
import cv2
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



def load_data(folder_paths, verbose=True):
    """
    Load images and corresponding landmark annotations from given folder paths.

    Args:
        folder_paths (list of str): List of folder paths containing .jpg and .mat files.
        verbose (bool): Whether to show progress bar and prints. Defaults to True.

    Returns:
        images (list of np.ndarray): List of image arrays.
        landmarks (list of np.ndarray): List of landmark arrays.
    """
    images = []
    landmarks = []

    # 전체 파일 수 계산
    total_files = sum(len([f for f in os.listdir(folder) if f.endswith('.jpg')])
                      for folder in folder_paths if os.path.exists(folder))

    with tqdm(total=total_files, disable=not verbose) as pbar:
        for folder in folder_paths:
            if not os.path.exists(folder):
                if verbose:
                    print(f"Warning: Folder does not exist: {folder}")
                continue

            try:
                for file in os.listdir(folder):
                    if file.endswith(".jpg"):
                        # Image path
                        img_path = os.path.join(folder, file)
                        mat_path = os.path.splitext(img_path)[0] + ".mat"

                        try:
                            # Load image
                            img = cv2.imread(img_path)
                            if img is None:
                                if verbose:
                                    print(f"Failed to load image: {img_path}")
                                continue

                            # Load landmarks from .mat file
                            if os.path.exists(mat_path):
                                try:
                                    mat_data = loadmat(mat_path)
                                    if 'pt2d' in mat_data:
                                        pt2d = mat_data['pt2d']
                                        landmarks.append(pt2d)
                                        images.append(img)
                                    else:
                                        if verbose:
                                            print(f"No 'pt2d' data in {mat_path}")
                                except Exception as e:
                                    if verbose:
                                        print(f"Error loading .mat file {mat_path}: {str(e)}")
                            else:
                                if verbose:
                                    print(f"Missing .mat file for: {img_path}")

                        except Exception as e:
                            if verbose:
                                print(f"Error processing file {file}: {str(e)}")

                        pbar.update(1)

            except Exception as e:
                if verbose:
                    print(f"Error accessing folder {folder}: {str(e)}")

    if verbose:
        print(f"\nLoaded {len(images)} images with landmarks")

    return images, landmarks


class FacialLandmarkDataset(Dataset):
    def __init__(self, images, landmarks, input_size=224, original_size=450):
        """
        Custom Dataset for facial landmarks

        Args:
            images (list): List of image arrays
            landmarks (list): List of landmark arrays
            input_size (int): Target size for image resizing
            original_size (int): Original size of the images
        """
        self.images = images
        self.landmarks = landmarks
        self.input_size = input_size
        self.original_size = original_size
        self.scale_factor = input_size / original_size

        # 랜드마크 인덱스 매핑 정의
        self.landmark_indices = {
            'eyebrow_left': list(range(17, 22)),
            'eyebrow_right': list(range(22, 27)),
            'nose': list(range(27, 36)),
            'eye_left': list(range(36, 42)),
            'eye_right': list(range(42, 48)),
            'mouth': list(range(48, 68))
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        landmark = self.landmarks[idx]

        # Resize image to 224x224
        image = cv2.resize(image, (self.input_size, self.input_size))

        # 흑백 이미지인 경우 3채널로 변환
        if len(image.shape) == 2:  # 흑백 이미지
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:  # 단일 채널 이미지
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # BGR to RGB 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Scale landmark coordinates
        landmark = landmark * self.scale_factor

        # Convert to torch tensors
        image = torch.FloatTensor(image) / 255.0  # normalize to [0, 1]
        landmark = torch.FloatTensor(landmark)

        return image, landmark


def prepare_datasets(folder_paths, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                     batch_size=32, num_workers=8, input_size=224):
    """
    데이터 로드 및 데이터셋 준비

    Args:
        folder_paths (list): 데이터 폴더 경로 리스트
        train_ratio (float): 학습 데이터 비율
        val_ratio (float): 검증 데이터 비율
        test_ratio (float): 테스트 데이터 비율
        batch_size (int): 배치 크기
        num_workers (int): 데이터 로더 워커 수
        input_size (int): 입력 이미지 크기

    Returns:
        train_loader, val_loader, test_loader: PyTorch 데이터로더들
    """
    # 데이터 로드
    images, landmarks = load_data(folder_paths)

    # 데이터 분할
    train_val_images, test_images, train_val_landmarks, test_landmarks = train_test_split(
        images, landmarks, test_size=test_ratio, random_state=42
    )

    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_images, val_images, train_landmarks, val_landmarks = train_test_split(
        train_val_images, train_val_landmarks, test_size=val_ratio_adjusted, random_state=42
    )

    # 데이터셋 생성
    train_dataset = FacialLandmarkDataset(train_images, train_landmarks, input_size=input_size)
    val_dataset = FacialLandmarkDataset(val_images, val_landmarks, input_size=input_size)
    test_dataset = FacialLandmarkDataset(test_images, test_landmarks, input_size=input_size)

    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    folder_paths = [
        r"C:/Users/dhkim/OneDrive/Desktop/300W_LP/AFW_Flip",
        r"C:/Users/dhkim/OneDrive/Desktop/300W_LP/HELEN_Flip",
        r"C:/Users/dhkim/OneDrive/Desktop/300W_LP/IBUG_Flip",
        r"C:/Users/dhkim/OneDrive/Desktop/300W_LP/LFPW_Flip"
    ]

    # 샘플 데이터 로드
    train_loader, val_loader, test_loader = prepare_datasets(
        folder_paths=folder_paths,
        batch_size=4,  # 테스트를 위해 작은 배치 사이즈 사용
        num_workers=0  # 디버깅을 위해 단일 워커 사용
    )

    # 첫 번째 배치 가져오기
    images, landmarks = next(iter(train_loader))

    print(f"\nImage shape: {images.shape}")  # 예상: torch.Size([4, 3, 224, 224])
    print(f"Landmark shape: {landmarks.shape}")  # 예상: torch.Size([4, 68, 2])

