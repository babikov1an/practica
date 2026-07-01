import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from datetime import datetime
import os


class RedLightDetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.car_class_id = 2
        self.traffic_light_class_id = 9
        self._font_path = self._find_font()

    def _find_font(self):
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        ]
        for p in font_paths:
            if os.path.exists(p):
                return p
        return None

    def _put_text_cyrillic(self, img, text, position, font_size=24, color=(255, 255, 255), bg_color=None):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        if self._font_path:
            font = ImageFont.truetype(self._font_path, font_size)
        else:
            font = ImageFont.load_default()
        bbox = draw.textbbox(position, text, font=font)
        if bg_color:
            draw.rectangle([bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2], fill=bg_color)
        draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def detect(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None, {"error": "Не удалось загрузить изображение"}

        h_img, w_img = img.shape[:2]
        cx_img, cy_img = w_img // 2, h_img // 2

        # Run detection at higher resolution
        results = self.model(img, imgsz=1280, conf=0.15)
        result = results[0]

        cars = []
        traffic_lights = []

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_id == self.car_class_id:
                car_h = y2 - y1
                car_w = x2 - x1
                if car_h < 35 or car_w < 35:
                    continue
                cars.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "label": "Автомобиль"
                })
            elif cls_id == self.traffic_light_class_id:
                color = self._detect_traffic_light_color(img, x1, y1, x2, y2)
                # Distance from center of image
                tl_cx = (x1 + x2) // 2
                tl_cy = (y1 + y2) // 2
                dist_from_center = ((tl_cx - cx_img)**2 + (tl_cy - cy_img)**2) ** 0.5
                traffic_lights.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "label": f"Светофор ({color})",
                    "color": color,
                    "dist_from_center": dist_from_center
                })

        # Also run a second pass on center crop for better traffic light detection
        crop_margin = 0.35
        crop_x1 = int(w_img * crop_margin)
        crop_x2 = int(w_img * (1 - crop_margin))
        crop_y1 = int(h_img * 0.1)
        crop_y2 = int(h_img * 0.6)
        center_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

        results_crop = self.model(center_crop, imgsz=640, conf=0.10)
        for box in results_crop[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id == self.traffic_light_class_id:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Map back to full image coordinates
                x1 += crop_x1
                x2 += crop_x1
                y1 += crop_y1
                y2 += crop_y1
                color = self._detect_traffic_light_color(img, x1, y1, x2, y2)
                tl_cx = (x1 + x2) // 2
                tl_cy = (y1 + y2) // 2
                dist_from_center = ((tl_cx - cx_img)**2 + (tl_cy - cy_img)**2) ** 0.5
                # Only add if not a duplicate (within 50px of existing)
                is_duplicate = any(
                    abs((tl["bbox"][0] + tl["bbox"][2])//2 - tl_cx) < 50 and
                    abs((tl["bbox"][1] + tl["bbox"][3])//2 - tl_cy) < 50
                    for tl in traffic_lights
                )
                if not is_duplicate:
                    traffic_lights.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "label": f"Светофор ({color})",
                        "color": color,
                        "dist_from_center": dist_from_center
                    })

        # Sort traffic lights: closest to center first
        traffic_lights.sort(key=lambda tl: tl["dist_from_center"])

        # Supplemental: color-based red light detection (find bright red spots)
        # Only in upper 50% of image — traffic lights are never at road level
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        upper_half = hsv_img[:h_img // 2, :]

        red_mask1 = cv2.inRange(upper_half, np.array([0, 100, 180]), np.array([12, 255, 255]))
        red_mask2 = cv2.inRange(upper_half, np.array([165, 100, 180]), np.array([180, 255, 255]))
        red_mask = red_mask1 | red_mask2
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 150 or area > 5000:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            if aspect < 0.3 or aspect > 2.5:
                continue

            tl_cx = x + w // 2
            tl_cy = y + h // 2
            is_duplicate = any(
                abs((tl["bbox"][0] + tl["bbox"][2])//2 - tl_cx) < 80 and
                abs((tl["bbox"][1] + tl["bbox"][3])//2 - tl_cy) < 80
                for tl in traffic_lights
            )
            if not is_duplicate:
                pad = max(w, h)
                bx1 = max(0, x - pad)
                by1 = max(0, y - pad)
                bx2 = min(w_img, x + w + pad)
                by2 = min(h_img, y + h + pad)
                dist_from_center = ((tl_cx - cx_img)**2 + (tl_cy - cy_img)**2) ** 0.5
                traffic_lights.append({
                    "bbox": [bx1, by1, bx2, by2],
                    "confidence": 0.5,
                    "label": "Светофор (красный)",
                    "color": "красный",
                    "dist_from_center": dist_from_center
                })

        traffic_lights.sort(key=lambda tl: tl["dist_from_center"])

        output_img = img.copy()

        # Draw all detected cars (green boxes)
        for car in cars:
            x1, y1, x2, y2 = car["bbox"]
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"Car {car['confidence']:.0%}"
            output_img = self._put_text_cyrillic(output_img, label, (x1 + 2, y1 - 22),
                                                  font_size=18, color=(255, 255, 255), bg_color=(0, 160, 0))

        # Draw all traffic lights
        for tl in traffic_lights:
            x1, y1, x2, y2 = tl["bbox"]
            if tl["color"] == "красный":
                box_color = (0, 0, 255)
            elif tl["color"] == "зеленый":
                box_color = (0, 200, 0)
            else:
                box_color = (0, 200, 255)
            cv2.rectangle(output_img, (x1, y1), (x2, y2), box_color, 2)
            label = tl["label"]
            output_img = self._put_text_cyrillic(output_img, label, (x1 + 2, y1 - 22),
                                                  font_size=18, color=(255, 255, 255), bg_color=box_color)

        # Find violations - if ANY red light is active, cars in the intersection are violators
        violations = []
        has_red_light = any(tl["color"] == "красный" for tl in traffic_lights)
        red_lights = [tl for tl in traffic_lights if tl["color"] == "красный"]

        h_img, w_img = img.shape[:2]

        # Find the stop line position: lowest car bottom edge (closest to camera)
        # Cars waiting at the line have their wheels at the bottom of the image
        car_bottoms = [(car["bbox"][3], car) for car in cars]
        if car_bottoms:
            # The stop line is roughly at the level of the lowest (closest) car's bottom
            stop_line_y = max(bottom for bottom, _ in car_bottoms)
        else:
            stop_line_y = h_img

        violation_count = 0
        for car in cars:
            x1, y1, x2, y2 = car["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            car_bottom = y2
            car_h = y2 - y1
            car_w = x2 - x1

            past_stop_line = car_bottom < stop_line_y - 50

            # Car should be in a reasonable range (not too far in the background)
            in_range = h_img * 0.25 < cy < h_img * 0.75

            if has_red_light and past_stop_line and in_range:
                # Find the closest red light to connect with a line
                closest_tl = None
                min_dist = float('inf')
                for tl in red_lights:
                    tl_cx = (tl["bbox"][0] + tl["bbox"][2]) // 2
                    tl_cy = (tl["bbox"][1] + tl["bbox"][3]) // 2
                    dist = ((cx - tl_cx)**2 + (cy - tl_cy)**2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        closest_tl = tl

                violation_count += 1
                violations.append({
                    "car_bbox": car["bbox"],
                    "traffic_light_bbox": closest_tl["bbox"] if closest_tl else None,
                    "description": f"Нарушитель #{violation_count}: Проезд на красный свет"
                })

                # Semi-transparent red overlay on the car
                overlay = output_img.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.3, output_img, 0.7, 0, output_img)

                # Thick red border (multiple layers for visibility)
                for i in range(5):
                    cv2.rectangle(output_img,
                                (x1 - i, y1 - i),
                                (x2 + i, y2 + i),
                                (0, 0, 255), 2)

                # Corner markers for extra visibility
                corner_len = 20
                cv2.line(output_img, (x1, y1), (x1 + corner_len, y1), (0, 255, 255), 3)
                cv2.line(output_img, (x1, y1), (x1, y1 + corner_len), (0, 255, 255), 3)
                cv2.line(output_img, (x2, y1), (x2 - corner_len, y1), (0, 255, 255), 3)
                cv2.line(output_img, (x2, y1), (x2, y1 + corner_len), (0, 255, 255), 3)
                cv2.line(output_img, (x1, y2), (x1 + corner_len, y2), (0, 255, 255), 3)
                cv2.line(output_img, (x1, y2), (x1, y2 - corner_len), (0, 255, 255), 3)
                cv2.line(output_img, (x2, y2), (x2 - corner_len, y2), (0, 255, 255), 3)
                cv2.line(output_img, (x2, y2), (x2, y2 - corner_len), (0, 255, 255), 3)

                # "НАРУШЕНИЕ #N" label with background
                label = f"НАРУШЕНИЕ #{violation_count}"
                lx = x1
                ly = y1 - 30
                if ly < 30:
                    ly = y2 + 30
                output_img = self._put_text_cyrillic(output_img, label, (lx, ly),
                                                      font_size=26, color=(255, 255, 255), bg_color=(0, 0, 200))

                # Red dashed line from car to closest red traffic light
                if closest_tl:
                    car_center = (cx, cy)
                    tl_center = ((closest_tl["bbox"][0] + closest_tl["bbox"][2]) // 2,
                                (closest_tl["bbox"][1] + closest_tl["bbox"][3]) // 2)
                    self._draw_dashed_line(output_img, car_center, tl_center, (0, 0, 255), 2, 10)

                    # Red circle around the traffic light
                    tl_cx, tl_cy = tl_center
                    radius = max(closest_tl["bbox"][2] - closest_tl["bbox"][0],
                                closest_tl["bbox"][3] - closest_tl["bbox"][1]) // 2 + 10
                    cv2.circle(output_img, (tl_cx, tl_cy), radius, (0, 0, 255), 3)

        # Banner at top if violations found
        if violations:
            h, w = output_img.shape[:2]
            banner_h = 60
            banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
            banner[:] = (0, 0, 180)
            banner_text = f"!!! ОБНАРУЖЕНО НАРУШЕНИЙ: {len(violations)} !!!"
            text_x = w // 2 - 280
            banner_pil = Image.fromarray(cv2.cvtColor(banner, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(banner_pil)
            if self._font_path:
                font = ImageFont.truetype(self._font_path, 36)
            else:
                font = ImageFont.load_default()
            draw.text((text_x, 10), banner_text, font=font, fill=(255, 255, 255))
            banner = cv2.cvtColor(np.array(banner_pil), cv2.COLOR_RGB2BGR)
            output_img = np.vstack([banner, output_img])

        # Legend box at bottom-right
        h, w = output_img.shape[:2]
        legend_w, legend_h = 240, 130
        lx, ly = w - legend_w - 10, h - legend_h - 10
        legend_overlay = output_img.copy()
        cv2.rectangle(legend_overlay, (lx, ly), (lx + legend_w, ly + legend_h), (30, 30, 30), -1)
        cv2.addWeighted(legend_overlay, 0.7, output_img, 0.3, 0, output_img)
        cv2.rectangle(output_img, (lx, ly), (lx + legend_w, ly + legend_h), (200, 200, 200), 2)

        output_img = self._put_text_cyrillic(output_img, "Легенда:", (lx + 10, ly + 8),
                                              font_size=20, color=(255, 255, 255))
        cv2.rectangle(output_img, (lx + 10, ly + 38), (lx + 30, ly + 53), (0, 200, 0), -1)
        output_img = self._put_text_cyrillic(output_img, "Авто", (lx + 40, ly + 36),
                                              font_size=18, color=(255, 255, 255))
        cv2.rectangle(output_img, (lx + 10, ly + 62), (lx + 30, ly + 77), (0, 0, 255), -1)
        output_img = self._put_text_cyrillic(output_img, "Нарушитель", (lx + 40, ly + 60),
                                              font_size=18, color=(255, 255, 255))
        cv2.rectangle(output_img, (lx + 10, ly + 86), (lx + 30, ly + 101), (0, 0, 255), -1)
        output_img = self._put_text_cyrillic(output_img, "Красный свет", (lx + 40, ly + 84),
                                              font_size=18, color=(255, 255, 255))

        stats = {
            "total_cars": len(cars),
            "total_traffic_lights": len(traffic_lights),
            "violations": len(violations),
            "red_lights": sum(1 for tl in traffic_lights if tl["color"] == "красный"),
            "green_lights": sum(1 for tl in traffic_lights if tl["color"] == "зеленый"),
            "yellow_lights": sum(1 for tl in traffic_lights if tl["color"] == "желтый"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return output_img, stats

    def _draw_dashed_line(self, img, pt1, pt2, color, thickness=2, dash_length=10):
        dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        dashes = int(dist / dash_length)
        for i in range(dashes):
            start = i / dashes
            end = (i + 0.5) / dashes
            x1 = int(pt1[0] + (pt2[0] - pt1[0]) * start)
            y1 = int(pt1[1] + (pt2[1] - pt1[1]) * start)
            x2 = int(pt1[0] + (pt2[0] - pt1[0]) * end)
            y2 = int(pt1[1] + (pt2[1] - pt1[1]) * end)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    def _detect_traffic_light_color(self, img, x1, y1, x2, y2):
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return "неизвестный"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Only consider BRIGHT pixels (the actual lit LED, not the dark housing)
        bright_mask = v > 150

        # Wide red ranges
        red_mask1 = cv2.inRange(h, 0, 15)
        red_mask2 = cv2.inRange(h, 160, 180)
        red_mask = (red_mask1 | red_mask2) & (s > 50) & bright_mask

        # Green range
        green_mask = cv2.inRange(h, 30, 90) & (s > 50) & bright_mask

        # Yellow/orange range
        yellow_mask = cv2.inRange(h, 10, 40) & (s > 50) & bright_mask

        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)

        max_pixels = max(red_pixels, green_pixels, yellow_pixels)

        if max_pixels < 3:
            return "неизвестный"

        if max_pixels == red_pixels:
            return "красный"
        elif max_pixels == green_pixels:
            return "зеленый"
        else:
            return "желтый"
