import cv2
import numpy as np
import pyautogui
import time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
from PIL import Image, ImageTk
import heapq
from typing import List, Tuple, Optional
import logging
from dataclasses import dataclass
import json
import os
import pytesseract
from cnocr import CnOcr
import sys
from io import StringIO

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
ocr = CnOcr(rec_model_name="number-densenet_lite_136-fc", det_model_name="ch_PP-OCRv4_det_server", cand_alphabet=["1", "2", "3", "4", "5", "6", "7", "8"]) 

# 禁用pyautogui的安全机制
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.1

class LogRedirector:
    """重定向print输出到GUI"""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.original_stdout = sys.stdout
        
    def write(self, text):
        # 写入到GUI
        if self.text_widget and text.strip():
            timestamp = time.strftime("%H:%M:%S")
            self.text_widget.insert(tk.END, f"[{timestamp}] {text}\n")
            self.text_widget.see(tk.END)
            self.text_widget.update()
        
        # 也写入到原始stdout
        self.original_stdout.write(text)
        
    def flush(self):
        if hasattr(self.original_stdout, 'flush'):
            self.original_stdout.flush()

@dataclass
class PuzzleRegion:
    """拼图区域信息"""
    x: int
    y: int
    width: int
    height: int
    cell_width: int
    cell_height: int

class PuzzleState:
    """拼图状态类"""
    def __init__(self, board: List[List[int]], empty_pos: Tuple[int, int] = None):
        self.board = [row[:] for row in board]
        self.empty_pos = empty_pos or self.find_empty_position()
        self.g_cost = 0
        self.h_cost = 0
        self.parent = None
        self.move = None
    
    def find_empty_position(self) -> Tuple[int, int]:
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    return (i, j)
        return (2, 2)  # 默认右下角为空格
    
    def get_neighbors(self) -> List['PuzzleState']:
        neighbors = []
        row, col = self.empty_pos
        directions = [(-1, 0, 'UP'), (1, 0, 'DOWN'), (0, -1, 'LEFT'), (0, 1, 'RIGHT')]
        
        for dr, dc, move in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = PuzzleState(self.board)
                new_state.board[row][col] = self.board[new_row][new_col]
                new_state.board[new_row][new_col] = 0
                new_state.empty_pos = (new_row, new_col)
                new_state.parent = self
                new_state.move = move
                neighbors.append(new_state)
        return neighbors
    
    def manhattan_distance(self) -> int:
        distance = 0
        for i in range(3):
            for j in range(3):
                if self.board[i][j] != 0:
                    value = self.board[i][j]
                    target_row = (value - 1) // 3
                    target_col = (value - 1) % 3
                    distance += abs(i - target_row) + abs(j - target_col)
        return distance
    
    def is_goal(self) -> bool:
        target = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        return self.board == target
    
    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.board))
    
    def __eq__(self, other):
        return self.board == other.board
    
    def __lt__(self, other):
        return (self.g_cost + self.h_cost) < (other.g_cost + other.h_cost)

class PuzzleSolver:
    """A*算法拼图求解器"""
    
    @staticmethod
    def solve(initial_state: PuzzleState) -> List[str]:
        if initial_state.is_goal():
            return []
        
        open_set = []
        closed_set = set()
        
        initial_state.h_cost = initial_state.manhattan_distance()
        heapq.heappush(open_set, initial_state)
        
        nodes_explored = 0
        max_nodes = 100000  # 防止无限循环
        
        while open_set and nodes_explored < max_nodes:
            current = heapq.heappop(open_set)
            nodes_explored += 1
            
            if current.is_goal():
                moves = []
                while current.parent is not None:
                    moves.append(current.move)
                    current = current.parent
                return moves[::-1]
            
            closed_set.add(current)
            
            for neighbor in current.get_neighbors():
                if neighbor in closed_set:
                    continue
                
                neighbor.g_cost = current.g_cost + 1
                neighbor.h_cost = neighbor.manhattan_distance()
                heapq.heappush(open_set, neighbor)
        
        return []  # 无解或超出探索限制

class ScreenPuzzleDetector:
    """屏幕拼图检测器"""
    
    def __init__(self):
        self.puzzle_region = None
        self.debug_mode = True
    
    def capture_screen(self, region: Tuple[int, int, int, int] = None) -> Optional[np.ndarray]:
        """截取屏幕并进行错误检查"""
        try:
            if region:
                x, y, width, height = region
                # 检查区域是否有效
                screen_width, screen_height = pyautogui.size()
                
                if x < 0 or y < 0 or x + width > screen_width or y + height > screen_height:
                    print(f"截图区域超出屏幕范围: {region}, 屏幕大小: {screen_width}x{screen_height}")
                    return None
                
                if width <= 0 or height <= 0:
                    print(f"截图区域大小无效: {region}")
                    return None
                
                screenshot = pyautogui.screenshot(region=region)
            else:
                screenshot = pyautogui.screenshot()

            if screenshot is None:
                print("截图失败，返回None")
                return None
            
            # 转换为OpenCV格式
            screenshot_array = np.array(screenshot)
            
            if screenshot_array.size == 0:
                print("截图数组为空")
                return None
            
            screenshot_cv = cv2.cvtColor(screenshot_array, cv2.COLOR_RGB2BGR)
            
            # 保存调试图像
            if self.debug_mode:
                debug_path = "debug_screenshot.png"
                cv2.imwrite(debug_path, screenshot_cv)
                print(f"调试图像已保存: {debug_path}")
            
            return screenshot_cv
            
        except Exception as e:
            print(f"截图异常: {e}")
            return None
    
    def detect_puzzle_region_auto(self, image: np.ndarray) -> Optional[PuzzleRegion]:
        """自动检测拼图区域 - 改进版，针对分离图块结构"""
        try:
            if image is None or image.size == 0:
                print("输入图像为空")
                return None
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
            # 预处理 - 降噪
            denoised = cv2.medianBlur(gray, 3)
            blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
            
            # 多种阈值方法结合
            _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                   cv2.THRESH_BINARY, 15, 5)
            
            # 结合阈值结果
            combined_thresh = cv2.bitwise_or(thresh_otsu, adaptive_thresh)
            
            # 改进的边缘检测 - 使用多种参数
            edges1 = cv2.Canny(blurred, 30, 100)  # 较低阈值，捕获更多边缘
            edges2 = cv2.Canny(combined_thresh, 50, 150)  # 标准阈值
            
            # 合并边缘
            edges_combined = cv2.bitwise_or(edges1, edges2)
            
            # === 关键改进：多步骤形态学操作 ===
            
            # 步骤1: 先用小核进行闭运算，连接近距离的断裂
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges_closed = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel_small)
            
            # 步骤2: 膨胀操作，进一步连接断裂的边缘
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            edges_dilated = cv2.dilate(edges_closed, kernel_dilate, iterations=1)
            
            # 步骤3: 较大核的闭运算，连接更远距离的断裂
            kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            edges_final = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel_large)
            
            # 步骤4: 轻微的腐蚀，恢复形状但保持连接性
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            edges_final = cv2.erode(edges_final, kernel_erode, iterations=1)
            
            # 保存调试图像
            if self.debug_mode:
                cv2.imwrite("debug_edges_original.png", edges_combined)
                cv2.imwrite("debug_edges_closed.png", edges_closed)
                cv2.imwrite("debug_edges_dilated.png", edges_dilated)
                cv2.imwrite("debug_edges_processed.png", edges_final)
                print("边缘处理调试图像已保存")
            
            # 查找轮廓
            contours, _ = cv2.findContours(edges_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print("未找到任何轮廓")
                return self._get_default_region(image)
            
            # 筛选可能的拼图块轮廓
            puzzle_blocks = []
            min_area = 0.6 * min(image.shape[0], image.shape[1]) * min(image.shape[0], image.shape[1]) / 9  # 最小面积阈值
            max_area = 1.5 * min(image.shape[0], image.shape[1]) * min(image.shape[0], image.shape[1]) / 9  # 最大面积阈值
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    # 计算轮廓的边界矩形
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 筛选接近正方形的轮廓
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.8 < aspect_ratio < 1.2:  # 宽高比在合理范围内
                        puzzle_blocks.append({
                            'contour': contour,
                            'bbox': (x, y, w, h),
                            'area': area,
                            'center': (x + w // 2, y + h // 2)
                        })
            
            print(f"找到 {len(puzzle_blocks)} 个候选拼图块")
            print(puzzle_blocks)
            
            if len(puzzle_blocks) < 4:  # 至少需要4个块才能确定网格
                print("候选拼图块数量不足，尝试降低阈值")
                return self._detect_with_lower_threshold(image)
            
            # 按面积排序，取最大的几个
            puzzle_blocks.sort(key=lambda x: x['area'], reverse=True)
            puzzle_blocks = puzzle_blocks[:min(12, len(puzzle_blocks))]  # 最多取12个
            
            # 尝试找到3x3网格结构
            grid_region = self._find_grid_structure(puzzle_blocks)
            
            if grid_region:
                print(f"检测到3x3网格结构: {grid_region}")
                return grid_region
            else:
                print("未能检测到3x3网格结构，使用聚类方法")
                return self._cluster_based_detection(puzzle_blocks, image)
                
        except Exception as e:
            print(f"检测拼图区域失败: {e}")
            return self._get_default_region(image)

    def _detect_with_lower_threshold(self, image: np.ndarray) -> Optional[PuzzleRegion]:
        """使用更低阈值重新检测"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
            # 预处理 - 降噪
            denoised = cv2.medianBlur(gray, 3)
            blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
            
            # 多种阈值方法结合
            _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                   cv2.THRESH_BINARY, 15, 5)
            
            # 结合阈值结果
            combined_thresh = cv2.bitwise_or(thresh_otsu, adaptive_thresh)
            
            # 改进的边缘检测 - 使用多种参数
            edges1 = cv2.Canny(blurred, 30, 100)  # 较低阈值，捕获更多边缘
            edges2 = cv2.Canny(combined_thresh, 50, 150)  # 标准阈值
            
            # 合并边缘
            edges_combined = cv2.bitwise_or(edges1, edges2)
            
            # === 关键改进：多步骤形态学操作 ===
            
            # 步骤1: 先用小核进行闭运算，连接近距离的断裂
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges_closed = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel_small)
            
            # 步骤2: 膨胀操作，进一步连接断裂的边缘
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            edges_dilated = cv2.dilate(edges_closed, kernel_dilate, iterations=1)
            
            # 步骤3: 较大核的闭运算，连接更远距离的断裂
            kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            edges_final = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel_large)
            
            # 步骤4: 轻微的腐蚀，恢复形状但保持连接性
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            edges_final = cv2.erode(edges_final, kernel_erode, iterations=1)
            
            # 保存调试图像
            if self.debug_mode:
                cv2.imwrite("debug_edges_original.png", edges_combined)
                cv2.imwrite("debug_edges_closed.png", edges_closed)
                cv2.imwrite("debug_edges_dilated.png", edges_dilated)
                cv2.imwrite("debug_edges_processed.png", edges_final)
                print("边缘处理调试图像已保存")
            
            # 查找轮廓
            contours, _ = cv2.findContours(edges_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print("未找到任何轮廓")
                return self._get_default_region(image)
            
            # 筛选可能的拼图块轮廓
            puzzle_blocks = []
            min_area = 0.3 * min(image.shape[0], image.shape[1]) * min(image.shape[0], image.shape[1]) / 9  # 最小面积阈值
            max_area = 3 * min(image.shape[0], image.shape[1]) * min(image.shape[0], image.shape[1]) / 9  # 最大面积阈值
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    # 计算轮廓的边界矩形
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 筛选接近正方形的轮廓
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.5 < aspect_ratio < 2:  # 宽高比在合理范围内
                        puzzle_blocks.append({
                            'contour': contour,
                            'bbox': (x, y, w, h),
                            'area': area,
                            'center': (x + w // 2, y + h // 2)
                        })
            
            print(f"找到 {len(puzzle_blocks)} 个候选拼图块")
            print(puzzle_blocks)
            
            if len(puzzle_blocks) >= 4:
                return self._find_grid_structure(puzzle_blocks) or self._cluster_based_detection(puzzle_blocks, image)
            else:
                return self._get_default_region(image)
                
        except Exception as e:
            print(f"低阈值检测失败: {e}")
            return self._get_default_region(image)

    def _find_grid_structure(self, blocks: List[dict]) -> Optional[PuzzleRegion]:
        """尝试找到3x3网格结构"""
        try:
            if len(blocks) < 4:
                return None
            
            # 提取所有中心点
            centers = [block['center'] for block in blocks]
            
            # 按y坐标分组（行）
            centers_by_y = {}
            y_tolerance = 50  # y坐标容差
            
            for x, y in centers:
                found_group = False
                for group_y in centers_by_y:
                    if abs(y - group_y) <= y_tolerance:
                        centers_by_y[group_y].append((x, y))
                        found_group = True
                        break
                if not found_group:
                    centers_by_y[y] = [(x, y)]
            
            # 筛选出有足够点的行
            valid_rows = []
            for y, points in centers_by_y.items():
                if len(points) >= 2:  # 至少2个点才能构成一行
                    points.sort(key=lambda p: p[0])  # 按x坐标排序
                    valid_rows.append((y, points))
            
            if len(valid_rows) < 2:
                print("未找到足够的行")
                return None
            
            # 按y坐标排序行
            valid_rows.sort(key=lambda r: r[0])
            
            # 计算网格参数
            all_x = []
            all_y = []
            
            for y, points in valid_rows:
                all_y.append(y)
                for x, _ in points:
                    all_x.append(x)
            
            if len(all_x) < 4 or len(all_y) < 2:
                print("网格点数量不足")
                return None
            
            # 计算边界
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            
            # 估算单元格大小
            x_coords = sorted(list(set(all_x)))
            y_coords = sorted(list(set(all_y)))
            
            # 计算平均间距
            if len(x_coords) > 1:
                x_spacing = sum(x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)) / (len(x_coords)-1)
            else:
                x_spacing = 100
            
            if len(y_coords) > 1:
                y_spacing = sum(y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)) / (len(y_coords)-1)
            else:
                y_spacing = 100
            
            # 扩展边界以包含完整的3x3网格
            cell_size = int((x_spacing + y_spacing) / 2)
            
            # 计算网格区域
            grid_x = min_x - cell_size // 2
            grid_y = min_y - cell_size // 2
            grid_width = max_x - min_x + cell_size
            grid_height = max_y - min_y + cell_size
            
            # 确保区域为3x3
            if grid_width < grid_height:
                diff = grid_height - grid_width
                grid_x -= diff // 2
                grid_width = grid_height
            elif grid_height < grid_width:
                diff = grid_width - grid_height
                grid_y -= diff // 2
                grid_height = grid_width
            
            print(f"计算出的网格区域: ({grid_x}, {grid_y}, {grid_width}x{grid_height})")
            print(f"估算的单元格大小: {cell_size}")
            
            return PuzzleRegion(
                x=max(0, grid_x),
                y=max(0, grid_y),
                width=grid_width,
                height=grid_height,
                cell_width=(grid_width) // 3,
                cell_height=(grid_height) // 3
            )
            
        except Exception as e:
            print(f"网格结构检测失败: {e}")
            return None

    def _cluster_based_detection(self, blocks: List[dict], image: np.ndarray) -> Optional[PuzzleRegion]:
        """基于聚类的检测方法"""
        try:
            if len(blocks) < 4:
                return None
            
            # 使用所有块的边界框来估算整体区域
            all_x = []
            all_y = []
            all_x2 = []
            all_y2 = []
            
            for block in blocks:
                x, y, w, h = block['bbox']
                all_x.append(x)
                all_y.append(y)
                all_x2.append(x + w)
                all_y2.append(y + h)
            
            # 计算总边界
            min_x = min(all_x)
            min_y = min(all_y)
            max_x = max(all_x2)
            max_y = max(all_y2)
            
            # 计算区域大小
            width = max_x - min_x
            height = max_y - min_y
            
            # 扩展边界以确保包含所有块
            padding = max(width, height) // 10
            min_x -= padding
            min_y -= padding
            width += 2 * padding
            height += 2 * padding
            
            # 使区域更接近正方形
            if width > height:
                diff = width - height
                min_y -= diff // 2
                height = width
            elif height > width:
                diff = height - width
                min_x -= diff // 2
                width = height
            
            print(f"聚类方法检测结果: ({min_x}, {min_y}, {width}x{height})")
            
            return PuzzleRegion(
                x=max(0, min_x),
                y=max(0, min_y),
                width=width,
                height=height,
                cell_width=width // 3,
                cell_height=height // 3
            )
            
        except Exception as e:
            print(f"聚类检测失败: {e}")
            return None

    def _get_default_region(self, image: np.ndarray) -> PuzzleRegion:
        """返回默认区域"""
        h, w = image.shape[:2]
        size = min(w, h) // 2
        x = (w - size) // 2
        y = (h - size) // 2
        
        print(f"使用默认区域: ({x}, {y}, {size}x{size})")
        
        return PuzzleRegion(
            x=x, y=y, width=size, height=size,
            cell_width=size // 3, cell_height=size // 3
        )
    
    def analyze_puzzle_state(self, image: np.ndarray, region: PuzzleRegion) -> Optional[List[List[int]]]:
        """分析拼图状态"""
        try:
            if image is None or image.size == 0:
                print("输入图像为空")
                return None
            
            # 提取拼图区域
            puzzle_area = image
            
            if puzzle_area.size == 0:
                print("提取的拼图区域为空")
                return None
            
            # 保存调试图像
            if self.debug_mode:
                cv2.imwrite("debug_puzzle_area.png", puzzle_area)
                print("拼图区域调试图像已保存: debug_puzzle_area.png")
            
            # 转换为灰度图
            if len(puzzle_area.shape) == 3:
                gray = cv2.cvtColor(puzzle_area, cv2.COLOR_BGR2GRAY)
            else:
                gray = puzzle_area
            
            puzzle_state = [[0 for _ in range(3)] for _ in range(3)]
            
            for i in range(3):
                for j in range(3):
                    # 提取单个格子
                    cell_x = int((j + 4/5) * region.cell_width)
                    cell_y = int((i + 1/10) * region.cell_height)
                    cell_x2 = int(min((j + 9/10) * region.cell_width, gray.shape[1]))
                    cell_y2 = int(min((i + 1/5) * region.cell_height, gray.shape[0]))
                    
                    if cell_x >= cell_x2 or cell_y >= cell_y2:
                        continue
                    
                    cell = gray[cell_y:cell_y2, cell_x:cell_x2]
                    
                    if cell.size == 0:
                        continue
                    
                    # 识别数字
                    number = self.recognize_number_improved(cell, i, j)
                    puzzle_state[i][j] = number
                    
                    # 保存调试格子图像
                    if self.debug_mode:
                        cv2.imwrite(f"debug_cell_{i}_{j}.png", cell)
            
            print(f"识别的拼图状态: {puzzle_state}")
            return puzzle_state
            
        except Exception as e:
            print(f"分析拼图状态失败: {e}")
            return None
    
    def recognize_number_improved(self, cell: np.ndarray, row: int, col: int) -> int:
        """改进的数字识别"""
        try:
            if cell.size == 0:
                return 0
            
            # 预处理
            if len(cell.shape) == 3:
                cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            
            # 检查是否为空格（通过颜色判断）
            mean_intensity = np.mean(cell)
            
            # 应用高斯模糊前检查
            if cell.shape[0] > 0 and cell.shape[1] > 0:
                blurred = cv2.GaussianBlur(cell, (5, 5), 0)
            else:
                blurred = cell
            
            # 二值化
            _, binary = cv2.threshold(blurred, 175, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            digit_str = ocr.ocr_for_single_line(binary)
            print([digit_str['text'],digit_str['score']]) if digit_str and digit_str['score']>0.5 and digit_str['text'] != '' else print(0)
        
            return int(digit_str['text']) if digit_str and digit_str['score']>0.5 and digit_str['text'] != '' else 0
                
        except Exception as e:
            print(f"数字识别失败: {e}")
            # 返回基于位置的默认值
            default_number = row * 3 + col + 1
            return default_number if default_number < 9 else 0

class PuzzleSolverGUI:
    """拼图求解器图形界面"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Microsoft Rewards 拼图自动求解器 v2.1")
        self.root.geometry("900x900")
        self.root.resizable(False, False)
        
        self.detector = ScreenPuzzleDetector()
        self.solver = PuzzleSolver()
        self.puzzle_region = None
        self.is_solving = False
        self.current_empty_pos = (2, 2)  # 跟踪空格位置
        self.current_puzzle_state = None  # 当前拼图状态
        
        self.setup_ui()
        self.load_settings()
        
        # 设置日志重定向
        self.log_redirector = LogRedirector(self.log_text)
        sys.stdout = self.log_redirector
    
    def setup_ui(self):
        """设置用户界面"""
        # 创建主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧框架
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 右侧框架（日志）
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # 标题
        title_label = tk.Label(left_frame, text="Microsoft Rewards 拼图自动求解器 v2.1", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # 状态显示
        self.status_var = tk.StringVar(value="就绪")
        status_label = tk.Label(left_frame, textvariable=self.status_var, 
                               font=("Arial", 12), fg="blue")
        status_label.pack(pady=5)
        
        # 控制按钮框架
        control_frame = tk.Frame(left_frame)
        control_frame.pack(pady=20)
        
        # 第一行按钮
        button_row1 = tk.Frame(control_frame)
        button_row1.pack(pady=5)
        
        # 自动检测按钮
        self.auto_detect_btn = tk.Button(button_row1, text="自动检测拼图", 
                                        command=self.auto_detect_puzzle,
                                        font=("Arial", 10), bg="lightyellow")
        self.auto_detect_btn.pack(side=tk.LEFT, padx=2)
        
        # 手动设置区域按钮
        self.setup_region_btn = tk.Button(button_row1, text="手动设置区域", 
                                         command=self.setup_puzzle_region,
                                         font=("Arial", 10), bg="lightblue")
        self.setup_region_btn.pack(side=tk.LEFT, padx=2)
        
        # 测试识别按钮
        self.test_btn = tk.Button(button_row1, text="测试识别", 
                                 command=self.test_recognition,
                                 font=("Arial", 10), bg="lightgray")
        self.test_btn.pack(side=tk.LEFT, padx=2)
        
        # 第二行按钮
        button_row2 = tk.Frame(control_frame)
        button_row2.pack(pady=5)
        
        # 开始自动解密按钮
        self.start_btn = tk.Button(button_row2, text="开始自动解密", 
                                  command=self.start_auto_solve,
                                  font=("Arial", 10, "bold"), bg="lightgreen")
        self.start_btn.pack(side=tk.LEFT, padx=2)
        
        # 停止按钮
        self.stop_btn = tk.Button(button_row2, text="停止", 
                                 command=self.stop_solving,
                                 font=("Arial", 10), bg="lightcoral", state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        # 设置框架
        settings_frame = tk.LabelFrame(left_frame, text="设置", font=("Arial", 12))
        settings_frame.pack(pady=20, padx=20, fill="x")
        
        # 点击延迟设置
        delay_frame = tk.Frame(settings_frame)
        delay_frame.pack(pady=5, fill="x")
        
        tk.Label(delay_frame, text="点击延迟(秒):").pack(side=tk.LEFT)
        self.delay_var = tk.DoubleVar(value=0.2)
        delay_spin = tk.Spinbox(delay_frame, from_=0.1, to=3.0, increment=0.1, 
                               textvariable=self.delay_var, width=10)
        delay_spin.pack(side=tk.LEFT, padx=10)
        
        # 调试模式
        debug_frame = tk.Frame(settings_frame)
        debug_frame.pack(pady=5, fill="x")
        
        self.debug_var = tk.BooleanVar(value=True)
        debug_check = tk.Checkbutton(debug_frame, text="调试模式（保存识别图像）", 
                                 variable=self.debug_var,
                                 command=self.toggle_debug_mode)
        debug_check.pack(side=tk.LEFT)
        
        # 拼图预览
        preview_frame = tk.LabelFrame(left_frame, text="拼图预览（点击数字可修改）", font=("Arial", 12))
        preview_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        self.preview_canvas = tk.Canvas(preview_frame, width=300, height=300, bg="white")
        self.preview_canvas.pack(pady=10)
        self.preview_canvas.bind("<Button-1>", self.on_canvas_click)
        
        # 区域信息显示
        info_frame = tk.Frame(left_frame)
        info_frame.pack(pady=10)
        
        self.region_info_var = tk.StringVar(value="未设置拼图区域")
        region_info_label = tk.Label(info_frame, textvariable=self.region_info_var, 
                                   font=("Arial", 10), fg="gray")
        region_info_label.pack()
        
        # 底部按钮
        bottom_frame = tk.Frame(left_frame)
        bottom_frame.pack(pady=5)
        
        save_btn = tk.Button(bottom_frame, text="保存设置", command=self.save_settings)
        save_btn.pack(side=tk.LEFT, padx=5)
        save_btn = tk.Button(bottom_frame, text="加载设置", command=self.load_settings)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        # 右侧日志区域
        log_frame = tk.LabelFrame(right_frame, text="程序日志", font=("Arial", 12))
        log_frame.pack(fill="both", expand=True)
        
        # 创建日志文本框和滚动条
        log_container = tk.Frame(log_frame)
        log_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(log_container, width=50, wrap=tk.WORD, font=("Consolas", 9))
        log_scrollbar = tk.Scrollbar(log_container, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill="both", expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill="y")
        
        # 日志控制按钮
        log_control_frame = tk.Frame(log_frame)
        log_control_frame.pack(pady=5)
        
        clear_log_btn = tk.Button(log_control_frame, text="清空日志", command=self.clear_log)
        clear_log_btn.pack(side=tk.LEFT, padx=5)
        
        save_log_btn = tk.Button(log_control_frame, text="保存日志", command=self.save_log)
        save_log_btn.pack(side=tk.LEFT, padx=5)
    
    def on_canvas_click(self, event):
        """处理预览画布点击事件"""
        if not self.current_puzzle_state:
            messagebox.showwarning("警告", "请先进行拼图识别")
            return
        
        # 计算点击位置对应的格子
        cell_size = 90
        start_x = 15
        start_y = 15
        
        col = (event.x - start_x) // cell_size
        row = (event.y - start_y) // cell_size
        
        if 0 <= row < 3 and 0 <= col < 3:
            current_value = self.current_puzzle_state[row][col]
            
            # 弹出输入对话框
            new_value = simpledialog.askinteger(
                "修改数字", 
                f"当前值: {current_value}\n请输入新值 (0-8):",
                minvalue=0, maxvalue=8, initialvalue=current_value
            )
            
            if new_value is not None:
                self.current_puzzle_state[row][col] = new_value
                self.show_puzzle_preview(self.current_puzzle_state)
                print(f"手动修改: 位置({row},{col}) {current_value} -> {new_value}")
    
    def toggle_debug_mode(self):
        """切换调试模式"""
        self.detector.debug_mode = self.debug_var.get()
        print(f"调试模式: {'开启' if self.debug_var.get() else '关闭'}")
    
    def clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)
        print("日志已清空")
    
    def save_log(self):
        """保存日志到文件"""
        try:
            log_content = self.log_text.get(1.0, tk.END)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"puzzle_log_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(log_content)
            
            print(f"日志已保存到: {filename}")
            messagebox.showinfo("成功", f"日志已保存到: {filename}")
        except Exception as e:
            print(f"保存日志失败: {e}")
            messagebox.showerror("错误", f"保存日志失败: {e}")
    
    def auto_detect_puzzle(self):
        """自动检测拼图"""
        print("开始自动检测拼图...")
        
        try:
            # 截取整个屏幕
            self.root.withdraw()  # 隐藏主窗口
            time.sleep(1)
            screenshot = self.detector.capture_screen()
            time.sleep(1)
            self.root.deiconify()  # 显示主窗口
            
            if screenshot is None:
                print("截图失败")
                return

            print(f"截图成功，图像大小: {screenshot.shape}")
            
            # 自动检测拼图区域
            detected_region = self.detector.detect_puzzle_region_auto(screenshot)
            
            if detected_region:
                self.puzzle_region = detected_region
                self.update_region_info()
                print(f"自动检测成功: ({detected_region.x}, {detected_region.y}, {detected_region.width}x{detected_region.height})")
                
                # 测试识别
                self.test_recognition()
            else:
                print("自动检测失败，请尝试手动设置区域")
                
        except Exception as e:
            print(f"自动检测出错: {e}")
    
    def setup_puzzle_region(self):
        """手动设置拼图区域"""
        print("请在3秒后点击拼图区域的左上角...")
        self.root.after(3000, self.capture_region_start)
    
    def capture_region_start(self):
        """开始捕获区域"""
        print("请点击拼图区域的左上角")
        #self.root.withdraw()  # 隐藏主窗口
        
        try:
            # 等待用户点击
            print("等待点击左上角...")
            time.sleep(1)
            x1, y1 = pyautogui.position()
            
            print(f"左上角坐标: ({x1}, {y1})")
            print("现在请点击拼图区域的右下角")
            
            time.sleep(4)
            x2, y2 = pyautogui.position()
            print(f"右下角坐标: ({x2}, {y2})")
            
            # 计算区域
            region_x = min(x1, x2)
            region_y = min(y1, y2)
            region_width = abs(x2 - x1)
            region_height = abs(y2 - y1)
            
            if region_width > 0 and region_height > 0:
                self.puzzle_region = PuzzleRegion(
                    x=region_x, y=region_y, 
                    width=region_width, height=region_height,
                    cell_width=region_width // 3, cell_height=region_height // 3
                )
                
                self.update_region_info()
                print(f"拼图区域设置完成: ({region_x}, {region_y}, {region_width}x{region_height})")
                
                # 自动测试识别
                self.test_recognition()
            else:
                print("区域大小无效，请重新设置")
                
        except Exception as e:
            print(f"设置区域失败: {e}")
        
        finally:
            #self.root.deiconify()  # 显示主窗口
            print("已设置区域")
    
    def update_region_info(self):
        """更新区域信息显示"""
        if self.puzzle_region:
            info_text = f"拼图区域: ({self.puzzle_region.x}, {self.puzzle_region.y}) {self.puzzle_region.width}x{self.puzzle_region.height}"
            self.region_info_var.set(info_text)
        else:
            self.region_info_var.set("未设置拼图区域")
    
    def test_recognition(self):
        """测试识别"""
        if not self.puzzle_region:
            print("请先设置拼图区域")
            return
        
        try:
            print("测试拼图识别...")
            
            # 截取拼图区域
            self.root.withdraw()  # 隐藏主窗口
            time.sleep(1)
            screenshot = self.detector.capture_screen(
                (self.puzzle_region.x, self.puzzle_region.y, 
                 self.puzzle_region.width, self.puzzle_region.height)
            )
            time.sleep(1)
            self.root.deiconify()  # 显示主窗口
            
            if screenshot is None:
                print("截图失败")
                return
            
            print(f"截图成功，大小: {screenshot.shape}")
            
            # 分析拼图状态
            puzzle_state = self.detector.analyze_puzzle_state(screenshot, self.puzzle_region)
            
            if puzzle_state:
                print("识别成功！拼图状态:")
                for i, row in enumerate(puzzle_state):
                    print(f"  第{i+1}行: {row}")
                
                # 保存当前状态
                self.current_puzzle_state = puzzle_state
                
                # 显示预览
                self.show_puzzle_preview(puzzle_state)
                
                # 检查是否已完成
                initial_state = PuzzleState(puzzle_state)
                if initial_state.is_goal():
                    print("拼图已经完成！")
                else:
                    print(f"距离完成还需 {initial_state.manhattan_distance()} 步")
            else:
                print("识别失败")
                
        except Exception as e:
            print(f"测试识别失败: {e}")
    
    def start_auto_solve(self):
        """开始自动解密"""
        if not self.puzzle_region:
            messagebox.showwarning("警告", "请先设置拼图区域")
            return
        
        if not self.current_puzzle_state:
            messagebox.showwarning("警告", "请先进行拼图识别")
            return
        
        if self.is_solving:
            return
        
        self.is_solving = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("正在解密...")
        
        # 在新线程中执行解密
        solve_thread = threading.Thread(target=self.solve_puzzle)
        solve_thread.daemon = True
        solve_thread.start()
    
    def solve_puzzle(self):
        """解决拼图"""
        try:
            print("=== 开始自动解密 ===")
            
            # 使用当前保存的拼图状态
            puzzle_state = self.current_puzzle_state
            
            print("使用的拼图状态:")
            for i, row in enumerate(puzzle_state):
                print(f"  第{i+1}行: {row}")
            
            # 创建拼图状态对象
            initial_state = PuzzleState(puzzle_state)
            self.current_empty_pos = initial_state.empty_pos
            
            if initial_state.is_goal():
                print("拼图已经完成！")
                return
            
            print(f"开始计算解决方案...（预估需要 {initial_state.manhattan_distance()} 步）")
            
            # 求解
            solution = self.solver.solve(initial_state)
            
            if not solution:
                print("无法找到解决方案，可能拼图无解或识别有误")
                return
            
            print(f"找到解决方案！共需要 {len(solution)} 步")
            print(f"解决步骤: {' -> '.join(solution)}")
            
            # 执行解决方案
            self.execute_solution(solution)
            
        except Exception as e:
            print(f"解密失败: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
        finally:
            self.is_solving = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.status_var.set("就绪")
    
    def show_puzzle_preview(self, puzzle_state: List[List[int]]):
        """显示拼图预览"""
        self.preview_canvas.delete("all")
        
        cell_size = 90
        start_x = 15
        start_y = 15
        
        for i in range(3):
            for j in range(3):
                x1 = start_x + j * cell_size
                y1 = start_y + i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                # 绘制格子
                if puzzle_state[i][j] == 0:
                    # 空格
                    self.preview_canvas.create_rectangle(x1, y1, x2, y2, 
                                                       fill="lightgray", outline="black", width=2)
                    self.preview_canvas.create_text((x1+x2)/2, (y1+y2)/2, 
                                                  text="空", font=("Arial", 16))
                else:
                    # 数字格子
                    color = "lightgreen" if puzzle_state[i][j] == i*3+j+1 else "lightcoral"
                    self.preview_canvas.create_rectangle(x1, y1, x2, y2, 
                                                       fill=color, outline="black", width=2)
                    self.preview_canvas.create_text((x1+x2)/2, (y1+y2)/2, 
                                                  text=str(puzzle_state[i][j]), 
                                                  font=("Arial", 24, "bold"))
        
        # 添加提示文本
        self.preview_canvas.create_text(150, 280, text="点击数字可修改", 
                                       font=("Arial", 10), fill="gray")
    
    def execute_solution(self, solution: List[str]):
        """执行解决方案"""
        print("开始执行解决方案...")
        
        for i, move in enumerate(solution):
            if not self.is_solving:  # 检查是否被停止
                print("解密被用户停止")
                break
            
            print(f"执行第 {i+1}/{len(solution)} 步: {move}")
            
            # 计算点击位置
            click_x, click_y = self.calculate_click_position(move)
            
            print(f"点击坐标: ({click_x}, {click_y})")
            
            # 执行点击
            try:
                pyautogui.click(click_x, click_y)
                
                # 更新空格位置
                self.update_empty_position(move)
                
                # 等待延迟
                delay = self.delay_var.get()
                print(f"等待 {delay} 秒...")
                time.sleep(delay)
                
            except Exception as e:
                print(f"点击失败: {e}")
                break
        
        if self.is_solving:
            print("=== 解决方案执行完成！===")
        
    def calculate_click_position(self, move: str) -> Tuple[int, int]:
        """计算点击位置"""
        empty_row, empty_col = self.current_empty_pos
        
        # 根据移动方向计算要点击的格子
        if move == 'UP':
            target_row, target_col = empty_row - 1, empty_col
        elif move == 'DOWN':
            target_row, target_col = empty_row + 1, empty_col
        elif move == 'LEFT':
            target_row, target_col = empty_row, empty_col - 1
        elif move == 'RIGHT':
            target_row, target_col = empty_row, empty_col + 1
        else:
            target_row, target_col = empty_row, empty_col
        
        # 计算在屏幕上的坐标
        click_x = (self.puzzle_region.x + 
                  target_col * self.puzzle_region.cell_width + 
                  self.puzzle_region.cell_width // 2)
        click_y = (self.puzzle_region.y + 
                  target_row * self.puzzle_region.cell_height + 
                  self.puzzle_region.cell_height // 2)
        
        return click_x, click_y
    
    def update_empty_position(self, move: str):
        """更新空格位置"""
        empty_row, empty_col = self.current_empty_pos
        
        if move == 'UP':
            self.current_empty_pos = (empty_row - 1, empty_col)
        elif move == 'DOWN':
            self.current_empty_pos = (empty_row + 1, empty_col)
        elif move == 'LEFT':
            self.current_empty_pos = (empty_row, empty_col - 1)
        elif move == 'RIGHT':
            self.current_empty_pos = (empty_row, empty_col + 1)
    
    def stop_solving(self):
        """停止解密"""
        self.is_solving = False
        print("正在停止解密...")
    
    def save_settings(self):
        """保存设置"""
        settings = {
            'delay': self.delay_var.get(),
            'debug_mode': self.debug_var.get(),
            'puzzle_region': {
                'x': self.puzzle_region.x if self.puzzle_region else 0,
                'y': self.puzzle_region.y if self.puzzle_region else 0,
                'width': self.puzzle_region.width if self.puzzle_region else 0,
                'height': self.puzzle_region.height if self.puzzle_region else 0,
            } if self.puzzle_region else None
        }
        
        try:
            with open('puzzle_settings.json', 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            print("设置已保存到 puzzle_settings.json")
        except Exception as e:
            print(f"保存设置失败: {e}")
    
    def load_settings(self):
        """加载设置"""
        try:
            if os.path.exists('puzzle_settings.json'):
                with open('puzzle_settings.json', 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                self.delay_var.set(settings.get('delay', 0.2))
                self.debug_var.set(settings.get('debug_mode', True))
                self.detector.debug_mode = self.debug_var.get()
                
                if settings.get('puzzle_region'):
                    region_data = settings['puzzle_region']
                    if all(region_data.values()):
                        self.puzzle_region = PuzzleRegion(
                            x=region_data['x'],
                            y=region_data['y'],
                            width=region_data['width'],
                            height=region_data['height'],
                            cell_width=region_data['width'] // 3,
                            cell_height=region_data['height'] // 3
                        )
                        self.update_region_info()
                        print("设置已从文件加载")
        except Exception as e:
            print(f"加载设置失败: {e}")
    
    def run(self):
        """运行应用"""
        print("=== Microsoft Rewards 拼图自动求解器 v2.1 ===")
        print("新功能：")
        print("- 点击预览区域的数字可以手动修改识别结果")
        print("- 右侧显示程序运行日志")
        print("使用步骤：")
        print("1. 点击'自动检测拼图'或'手动设置区域'")
        print("2. 点击'测试识别'验证识别效果")
        print("3. 如需要，点击预览区域的数字进行手动调整")
        print("4. 点击'开始自动解密'执行求解")
        print("5. 调试模式开启时会保存识别过程的图像文件")
        
        try:
            self.root.mainloop()
        finally:
            # 恢复标准输出
            sys.stdout = self.log_redirector.original_stdout

def main():
    """主函数"""
    try:
        app = PuzzleSolverGUI()
        app.run()
    except Exception as e:
        print(f"程序运行出错: {e}")
        messagebox.showerror("错误", f"程序运行出错: {e}")

if __name__ == "__main__":
    main()
