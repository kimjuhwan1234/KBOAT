from sklearn.metrics import f1_score
import numpy as np


def custom_f1_score(y_true, y_pred, average='weighted'):
    """
    맞춤형 F1 스코어 계산 함수
    멀티클래스 멀티아웃풋을 지원하기 위해 각 타겟에 대해 F1 스코어를 계산하고 이를 평균합니다.

    Parameters:
    - y_true: 실제 값, shape (n_samples, n_outputs)
    - y_pred: 예측 값, shape (n_samples, n_outputs)
    - average: 'weighted', 'macro', 'micro' 중 하나로, 평균 계산 방법을 지정합니다.

    Returns:
    - mean_f1: 평균 F1 스코어
    """
    n_outputs = y_true.shape[1]  # 출력(타겟) 수
    f1_scores = []

    for i in range(n_outputs):
        f1 = f1_score(y_true[:, i], y_pred[:, i], average=average)
        f1_scores.append(f1)

    mean_f1 = np.mean(f1_scores)
    return mean_f1
