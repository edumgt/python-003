# plot_training_log.py
# -----------------------------------------------------
# train_cats.py 학습 결과를 그래프로 시각화하는 스크립트
# -----------------------------------------------------

import json
import matplotlib.pyplot as plt
import os

LOG_PATH = "logs/train_log.jsonl"

def load_logs(path=LOG_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 로그 파일을 찾을 수 없습니다: {path}")

    epochs, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = eval(line.strip())  # train_cats.py가 dict 문자열로 기록함
            epochs.append(data["epoch"])
            train_loss.append(data["train_loss"])
            val_loss.append(data["val_loss"])
            train_acc.append(data["train_acc"])
            val_acc.append(data["val_acc"])

    return epochs, train_loss, val_loss, train_acc, val_acc


def plot_metrics(epochs, train_loss, val_loss, train_acc, val_acc):
    plt.figure(figsize=(10, 5))

    # Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss", marker="o")
    plt.plot(epochs, val_loss, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Accuracy 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train Accuracy", marker="o")
    plt.plot(epochs, val_acc, label="Val Accuracy", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    epochs, train_loss, val_loss, train_acc, val_acc = load_logs()
    plot_metrics(epochs, train_loss, val_loss, train_acc, val_acc)
