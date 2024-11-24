import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import FacialLandmarkModel
from ImageLoad import prepare_datasets


def load_training_data(folder_paths, batch_size=32, num_workers=0):
    """
    학습에 필요한 데이터 로더를 준비하는 함수

    Args:
        folder_paths (list): 데이터가 있는 폴더 경로 리스트
        batch_size (int): 배치 크기
        num_workers (int): 데이터 로딩에 사용할 워커 수

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    try:
        train_loader, val_loader, test_loader = prepare_datasets(
            folder_paths=folder_paths,
            batch_size=batch_size,
            num_workers=num_workers
        )

        # 데이터 로드 확인
        print("\nData Loading Summary:")
        print(f"Train batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")

        return train_loader, val_loader, test_loader

    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {str(e)}")
        raise

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, path='checkpoint.pt'):
        """
        Early stopping handler

        Args:
            patience (int): 성능 개선을 기다리는 에폭 수
            min_delta (float): 성능 개선으로 인정할 최소 변화량
            path (str): 모델 저장 경로
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """최상의 모델을 저장"""
        torch.save(model.state_dict(), self.path)


def train_model(train_loader, val_loader, model, num_epochs=200, device='cuda',
                checkpoint_dir='checkpoints'):
    """
    모델 학습 함수

    Args:
        train_loader: 학습 데이터로더
        val_loader: 검증 데이터로더
        model: 학습할 모델
        num_epochs (int): 총 에폭 수
        device (str): 학습에 사용할 장치 ('cuda' or 'cpu')
        checkpoint_dir (str): 체크포인트 저장 디렉토리
    """
    # 디렉토리 생성
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Loss function과 optimizer 설정
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Early stopping 설정
    early_stopping = EarlyStopping(
        patience=10,
        path=os.path.join(checkpoint_dir, 'best_model.pt')
    )

    # 학습 기록용
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }

    # 학습 시작
    print("Training started...")
    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        # 학습 루프
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for images, landmarks in train_loop:
            # 이미지 차원 순서 변경 (B, H, W, C) -> (B, C, H, W)
            images = images.permute(0, 3, 1, 2).to(device)
            landmarks = landmarks.reshape(landmarks.shape[0], -1).to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, landmarks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_loop.set_postfix({'loss': f'{loss.item():.4f}'})

        # 검증 단계
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, landmarks in val_loader:
                # 이미지 차원 순서 변경 (B, H, W, C) -> (B, C, H, W)
                images = images.permute(0, 3, 1, 2).to(device)
                landmarks = landmarks.reshape(landmarks.shape[0], -1).to(device)
                outputs = model(images)
                val_loss = criterion(outputs, landmarks)
                val_losses.append(val_loss.item())

        # 평균 loss 계산
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        # 현재 learning rate 저장
        current_lr = optimizer.param_groups[0]['lr']

        # History 업데이트
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rates'].append(current_lr)

        # 결과 출력
        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Learning Rate: {current_lr}')

        # Learning rate 조정
        scheduler.step(avg_val_loss)

        # Early stopping 체크
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # 학습 그래프 그리기
    plt.figure(figsize=(12,4))

    # Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Learning rate 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history['learning_rates'])
    plt.title('Learning Rate over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')

    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'training_history.png'))
    plt.close()

    return model, history


if __name__ == "__main__":
    # 학습 설정
    BATCH_SIZE = 32
    NUM_WORKERS = 0  # 워커 수를 0으로 변경
    NUM_EPOCHS = 200
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 경로 설정
    folder_paths = [
        r"C:/Users/dhkim/OneDrive/Desktop/300W_LP/AFW_Flip",
        r"C:/Users/dhkim/OneDrive/Desktop/300W_LP/HELEN_Flip",
        r"C:/Users/dhkim/OneDrive/Desktop/300W_LP/IBUG_Flip",
        r"C:/Users/dhkim/OneDrive/Desktop/300W_LP/LFPW_Flip"
    ]

    print("=== 학습 시작 ===")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of workers: {NUM_WORKERS}")
    print(f"Number of epochs: {NUM_EPOCHS}")

    try:
        # 데이터 로드
        print("\n1. 데이터 로딩 중...")
        train_loader, val_loader, test_loader = load_training_data(
            folder_paths=folder_paths,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )

        # 모델 생성
        print("\n2. 모델 생성 중...")
        model = FacialLandmarkModel(use_pretrained=True).to(DEVICE)
        print("모델 생성 완료")

        # 학습 실행
        print("\n3. 학습 시작...")
        model, history = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            num_epochs=NUM_EPOCHS,
            device=DEVICE
        )

        print("\n=== 학습 완료 ===")

    except Exception as e:
        print(f"\nError: {str(e)}")
        raise