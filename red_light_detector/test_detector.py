#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import RedLightDetector

def test_detector():
    print("Тестирование детектора...")
    detector = RedLightDetector()

    test_img = "test_image.jpg"
    if not os.path.exists(test_img):
        print(f"Тестовое изображение {test_img} не найдено")
        return False

    img, stats = detector.detect(test_img)

    if img is None:
        print("Ошибка: изображение не обработано")
        return False

    print(f"Результаты: {stats}")
    print(f"Размер изображения: {img.shape}")
    print("Тест пройден!")
    return True

if __name__ == "__main__":
    success = test_detector()
    sys.exit(0 if success else 1)
