import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import numpy as np


class FacialLandmarkModel(nn.Module):
    def __init__(self, use_pretrained=True):
        """
        ResNet18 기반의 얼굴 랜드마크 검출 모델

        Args:
            use_pretrained (bool): ImageNet 사전학습 가중치 사용 여부
        """
        super(FacialLandmarkModel, self).__init__()

        # ResNet18 백본 로드
        if use_pretrained:
            self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet18(weights=None)

        # 마지막 fully connected 층 제거
        in_features = self.resnet.fc.in_features

        # 새로운 회귀 헤드 생성
        self.regression_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 136)  # 68개 랜드마크 * 2(x, y 좌표)
        )

        # 기존 FC 층을 identity로 변경
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): 입력 이미지 배치 (B, 3, 224, 224)

        Returns:
            torch.Tensor: 랜드마크 좌표 (B, 136)
        """
        features = self.resnet(x)
        landmarks = self.regression_head(features)
        return landmarks


if __name__ == "__main__":
    # 모델 테스트
    def test_model():
        print("=== 모델 테스트 시작 ===")

        # 1. 장치 설정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"사용 장치: {device}")

        # 2. 모델 생성
        model = FacialLandmarkModel(use_pretrained=True).to(device)
        print("\n모델 구조:")
        print(model)

        # 3. 샘플 입력 생성
        batch_size = 4
        sample_input = torch.randn(batch_size, 3, 224, 224).to(device)

        # 4. Forward pass 테스트
        print("\nForward pass 테스트:")
        with torch.no_grad():
            output = model(sample_input)

        print(f"입력 크기: {sample_input.shape}")
        print(f"출력 크기: {output.shape}")

        # 5. 출력값 범위 확인
        print(f"\n출력값 통계:")
        print(f"최소값: {output.min().item():.2f}")
        print(f"최대값: {output.max().item():.2f}")
        print(f"평균값: {output.mean().item():.2f}")
        print(f"표준편차: {output.std().item():.2f}")

        # 6. Backward pass 테스트
        print("\nBackward pass 테스트:")
        sample_target = torch.randn(batch_size, 136).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Forward
        output = model(sample_input)
        loss = criterion(output, sample_target)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient 확인
        has_gradient = all(param.grad is not None for param in model.parameters() if param.requires_grad)
        print(f"Loss 값: {loss.item():.4f}")
        print(f"그래디언트 계산 완료: {has_gradient}")

        print("\n=== 모델 테스트 완료 ===")
        return has_gradient and output.shape == (batch_size, 136)


    # 테스트 실행
    try:
        success = test_model()
        print(f"\n테스트 결과: {'성공' if success else '실패'}")
    except Exception as e:
        print(f"테스트 중 오류 발생: {str(e)}")