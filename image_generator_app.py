import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, colorchooser, messagebox
from PIL import Image, ImageDraw, ImageFont, ImageTk, ImageFilter
import numpy as np
import random
import os
import sys
import threading
import glob
import json
import re
from typing import Tuple, Dict, Any

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def smart_wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
    wrapped_lines = []; user_lines = text.split('\n')
    for line in user_lines:
        if not line: wrapped_lines.append(''); continue
        words, current_line = line.split(), ""
        for word in words:
            if draw.textlength(current_line + word, font=font) <= max_width: current_line += word + " "
            else: wrapped_lines.append(current_line.strip()); current_line = word + " "
        wrapped_lines.append(current_line.strip())
    return "\n".join(wrapped_lines)
def apply_water_ripple(image: Image.Image, freq: float, amp: float) -> Image.Image:
    if amp == 0 or freq == 0: return image
    img_array = np.array(image); rows, cols, _ = img_array.shape
    x_map, y_map = np.meshgrid(np.arange(cols), np.arange(rows))
    offset_x = (amp * np.sin(y_map * freq)).astype(int); offset_y = (amp * np.sin(x_map * freq)).astype(int)
    map_x = np.clip(x_map + offset_x, 0, cols - 1); map_y = np.clip(y_map + offset_y, 0, rows - 1)
    return Image.fromarray(img_array[map_y, map_x])
def _create_gradient_image(width: int, height: int, colors: list[tuple[int, int, int]], direction: str = 'vertical') -> Image.Image:
    """Создает изображение с линейным градиентом."""
    if not colors:
        return Image.new('RGBA', (width, height), (0, 0, 0, 0))
    if len(colors) == 1:
        return Image.new('RGB', (width, height), colors[0])

    if direction == 'horizontal':
        gradient_array = np.zeros((height, width, 3), dtype=np.uint8)
        num_segments = len(colors) - 1
        segment_width = width / num_segments
        
        for i in range(num_segments):
            color_start = np.array(colors[i])
            color_end = np.array(colors[i+1])
            
            x_start = int(i * segment_width)
            x_end = int((i + 1) * segment_width)
            if i == num_segments - 1:
                x_end = width

            for x in range(x_start, x_end):
                ratio = (x - x_start) / segment_width
                inter_color = color_start * (1 - ratio) + color_end * ratio
                gradient_array[:, x] = inter_color.astype(np.uint8)
                
        return Image.fromarray(gradient_array, 'RGB')
    
    else: # Вертикальный градиент (старая логика)
        gradient_array = np.zeros((height, width, 3), dtype=np.uint8)
        num_segments = len(colors) - 1
        segment_height = height / num_segments
        
        for i in range(num_segments):
            color_start = np.array(colors[i])
            color_end = np.array(colors[i+1])
            
            y_start = int(i * segment_height)
            y_end = int((i + 1) * segment_height)
            if i == num_segments - 1:
                y_end = height

            for y in range(y_start, y_end):
                ratio = (y - y_start) / segment_height
                inter_color = color_start * (1 - ratio) + color_end * ratio
                gradient_array[y, :] = inter_color.astype(np.uint8)
                
        return Image.fromarray(gradient_array, 'RGB')

def generate_image_base(**params: Any) -> Image.Image:
    text, angle = params.get('text', 'Пример текста (start)'), params.get('angle', 0.0)
    wave_amplitude, wave_frequency = params.get('wave_amplitude', 10.0), params.get('wave_frequency', 0.07)
    bars_enabled, bars_count, bars_opacity = params.get('bars_enabled', False), int(params.get('bars_count', 5)), int(params.get('bars_opacity', 128))
    diag_bars_enabled, diag_bars_count, diag_bars_opacity = params.get('diag_bars_enabled', False), int(params.get('diag_bars_count', 4)), int(params.get('diag_bars_opacity', 100))
    circles_enabled, circles_count, circles_opacity = params.get('circles_enabled', False), int(params.get('circles_count', 6)), int(params.get('circles_opacity', 80))
    lens_enabled, lens_cell_size, lens_blur_radius = params.get('lens_enabled', False), int(params.get('lens_cell_size', 20)), int(params.get('lens_blur_radius', 1))
    text_color, bg_color = params.get('text_color', (240, 240, 240)), params.get('bg_color', (20, 20, 20))
    font_path, font_size = params.get('font_path', resource_path('font1.ttf')), int(params.get('font_size', 40))
    text_offset_x, text_offset_y = params.get('text_offset_x', 0), params.get('text_offset_y', 0)
    water_enabled, water_freq, water_amp = params.get('water_enabled', False), params.get('water_freq', 0.05), params.get('water_amp', 10)
    is_color_font = params.get('is_color_font', False)
    gradient_enabled = params.get('gradient_enabled', False)
    gradient_colors = params.get('gradient_colors', [])
    gradient_direction = params.get('gradient_direction', 'vertical') # Получаем новое значение
    IMAGE_SIZE = (512, 512)
    try:
        background = Image.new('RGBA', IMAGE_SIZE, bg_color)
        overlay_layer = Image.new('RGBA', IMAGE_SIZE, (0, 0, 0, 0)); draw_overlay = ImageDraw.Draw(overlay_layer)
        if circles_enabled and circles_count > 0:
            for _ in range(circles_count):
                radius = random.randint(30, 150); x, y = random.randint(-radius, IMAGE_SIZE[0]), random.randint(-radius, IMAGE_SIZE[1])
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), circles_opacity)
                draw_overlay.ellipse([x, y, x + radius*2, y + radius*2], fill=color)
        if bars_enabled and bars_count > 0:
            bar_height = max(1, IMAGE_SIZE[1] // (bars_count * 2))
            for _ in range(bars_count):
                bar_y = random.randint(0, IMAGE_SIZE[1] - bar_height)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), bars_opacity)
                draw_overlay.rectangle([0, bar_y, IMAGE_SIZE[0], bar_y + bar_height], fill=color)
        if diag_bars_enabled and diag_bars_count > 0:
            for _ in range(diag_bars_count):
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), diag_bars_opacity)
                start = (random.randint(-100, IMAGE_SIZE[0]+100), random.randint(-100, IMAGE_SIZE[1]+100))
                end = (random.randint(-100, IMAGE_SIZE[0]+100), random.randint(-100, IMAGE_SIZE[1]+100))
                draw_overlay.line([start, end], fill=color, width=random.randint(15, 60))
        background = Image.alpha_composite(background, overlay_layer)
        try:
            font = ImageFont.truetype(font_path, font_size, layout_engine=ImageFont.Layout.RAQM)
        except OSError:
            font = ImageFont.truetype(font_path, font_size)
        temp_draw = ImageDraw.Draw(Image.new('RGBA', (1,1)))
        wrapped_text = smart_wrap_text(temp_draw, text, font, IMAGE_SIZE[0] - 40)
        
        text_layer = Image.new('RGBA', IMAGE_SIZE, (0, 0, 0, 0)); draw_text = ImageDraw.Draw(text_layer)

        use_gradient = gradient_enabled and len(gradient_colors) >= 2
        if use_gradient and not is_color_font:
            draw_text.multiline_text((IMAGE_SIZE[0]/2 + text_offset_x, IMAGE_SIZE[1]/2 + text_offset_y), wrapped_text, font=font, fill=(255, 255, 255), anchor="mm", align="center")
            
            bbox_float = draw_text.multiline_textbbox((IMAGE_SIZE[0]/2 + text_offset_x, IMAGE_SIZE[1]/2 + text_offset_y), wrapped_text, font=font, anchor="mm", align="center")
            bbox = tuple(map(int, bbox_float))

            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                gradient_fill = _create_gradient_image(text_width, text_height, gradient_colors, direction=gradient_direction)
                
                gradient_text_layer = Image.new('RGBA', IMAGE_SIZE, (0, 0, 0, 0))
                gradient_text_layer.paste(gradient_fill, (bbox[0], bbox[1]), text_layer.crop(bbox))
                text_layer = gradient_text_layer

        elif is_color_font:
            draw_text.multiline_text((IMAGE_SIZE[0]/2 + text_offset_x, IMAGE_SIZE[1]/2 + text_offset_y), wrapped_text, font=font, anchor="mm", align="center")
        else:
            draw_text.multiline_text((IMAGE_SIZE[0]/2 + text_offset_x, IMAGE_SIZE[1]/2 + text_offset_y), wrapped_text, font=font, fill=text_color, anchor="mm", align="center")

        rotated_text_layer = text_layer.rotate(angle, expand=False, resample=Image.BICUBIC)
        
        # ---НАЧАЛО ИЗМЕНЕНИЙ---
        # Оптимизация эффекта волны с помощью векторизации NumPy
        img_array = np.array(rotated_text_layer)
        rows, cols, _ = img_array.shape

        if wave_amplitude > 0 and wave_frequency > 0:
            # Создаем сетку координат для каждого пикселя
            x_grid, y_grid = np.meshgrid(np.arange(cols), np.arange(rows))
            
            # Рассчитываем смещение по оси Y для каждой колонки (зависит от X)
            y_offsets = (wave_amplitude * np.sin(x_grid * wave_frequency)).astype(int)
            
            # Вычисляем новые Y-координаты для выборки пикселей из исходного изображения.
            # Это обратное отображение: для пикселя (y, x) в конечном изображении
            # мы берем пиксель (source_y, x) из исходного.
            source_y = y_grid + y_offsets
            
            # Ограничиваем координаты, чтобы не выйти за пределы изображения
            np.clip(source_y, 0, rows - 1, out=source_y)
            
            # Используем продвинутую индексацию NumPy для создания искаженного изображения за одну операцию
            distorted_array = img_array[source_y, x_grid]
        else:
            # Если эффект волны отключен, просто используем исходный массив
            distorted_array = img_array
            
        distorted_layer = Image.fromarray(distorted_array, 'RGBA')
        # ---КОНЕЦ ИЗМЕНЕНИЙ---
        
        final_image = Image.alpha_composite(background, distorted_layer).convert("RGB")
        if water_enabled: final_image = apply_water_ripple(final_image, water_freq, water_amp)
        if lens_enabled and lens_cell_size > 0 and lens_blur_radius > 0:
            for x in range(0, IMAGE_SIZE[0], lens_cell_size):
                for y in range(0, IMAGE_SIZE[1], lens_cell_size):
                    box = (x, y, x + lens_cell_size, y + lens_cell_size)
                    cell = final_image.crop(box).filter(ImageFilter.GaussianBlur(radius=lens_blur_radius))
                    final_image.paste(cell, box)
        return final_image
    except Exception as e:
        print(f"Ошибка генерации базового изображения: {e}"); error_img = Image.new('RGB', IMAGE_SIZE, (50, 0, 0))
        draw = ImageDraw.Draw(error_img); error_font = ImageFont.load_default(size=20)
        error_text = smart_wrap_text(draw, f"Критическая ошибка:\n{e}", error_font, IMAGE_SIZE[0] - 20)
        draw.multiline_text((10, 10), error_text, fill=(255, 50, 50), font=error_font); return error_img
    
def generate_noise_layer(strength, colorized, caliber):
    IMAGE_SIZE = (512, 512);
    if strength <= 0: return None
    caliber = max(1, int(caliber)); small_size = (IMAGE_SIZE[0] // caliber, IMAGE_SIZE[1] // caliber)
    if colorized:
        noise_array = np.random.randint(0, 255, (small_size[1], small_size[0], 3), dtype=np.uint8)
        noise_layer = Image.fromarray(noise_array, 'RGB').convert('RGBA')
    else:
        noise_array = np.random.randint(0, 45, small_size, dtype=np.uint8)
        noise_layer = Image.fromarray(noise_array, 'L').convert('RGBA')
    noise_layer = noise_layer.resize(IMAGE_SIZE, Image.Resampling.NEAREST)
    noise_layer.putalpha(Image.new('L', IMAGE_SIZE, int(strength))); return noise_layer

class TutorialWindow(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Добро пожаловать!"); self.transient(master); self.grab_set()
        master_x, master_y, master_w, master_h = master.winfo_x(), master.winfo_y(), master.winfo_width(), master.winfo_height()
        win_w, win_h = 450, 600; x, y = master_x + (master_w - win_w) // 2, master_y + (master_h - win_h) // 2
        self.geometry(f"{win_w}x{win_h}+{x}+{y}")
        try:
            self.bg_image = ctk.CTkImage(Image.open(resource_path("tutorial.png")), size=(400, 400))
            ctk.CTkLabel(self, image=self.bg_image, text="").pack(pady=10)
        except (FileNotFoundError, IOError): ctk.CTkLabel(self, text="[Изображение tutorial.png не найдено]").pack(pady=(20, 10))
        text = ("Привет! Рада приветствовать тебя в генераторе пикч. Генерируй любой текст так, чтобы ни одна нейронка его не распознала!\n\nИспользуй слайдеры и вкладки для настройки эффектов. Можно генерировать много пикч за раз. Все изменения сразу отображаются справа. Удачи!\n\n- Создано в ТГАЧЕ @dvach_chatbot -")
        ctk.CTkLabel(self, text=text, wraplength=400, justify="center").pack(padx=20, pady=10)
        ctk.CTkButton(self, text="Понятно!", command=self.close_window).pack(pady=20, padx=20, fill="x")
    def close_window(self):
        try:
            with open(".first_run_complete", "w") as f: f.write("done")
        except IOError: print("Не удалось создать файл .first_run_complete")
        self.destroy()

class BatchConfigWindow(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.master, self.title("Настройки пакетной генерации"), self.geometry("750x750"), self.transient(master), self.grab_set()
        self.active_preview_param, self.preview_animation_state, self.preview_animation_job, self.config = 'angle', 0, None, {}
        self.config_path = "batch_config.json"

        self.main_frame = ctk.CTkFrame(self); self.main_frame.pack(expand=True, fill="both", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=3); self.main_frame.grid_columnconfigure(1, weight=2)
        self.main_frame.grid_rowconfigure(0, weight=1)

        self.scrollable_frame = ctk.CTkScrollableFrame(self.main_frame); self.scrollable_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10)); self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.controls = {}
        row, bold_font = 0, ("Verdana", 12, "bold")
        for key, p in self.master.param_definitions.items():
            frame = ctk.CTkFrame(self.scrollable_frame, fg_color="transparent"); frame.grid(row=row, column=0, sticky="ew", padx=10, pady=5); frame.grid_columnconfigure(1, weight=1)
            # ---НАЧАЛО ИЗМЕНЕНИЙ---
            # Убираем chk.select(), чтобы состояние загружалось из файла
            check = ctk.CTkCheckBox(frame, text="", width=20); check.grid(row=0, column=0, sticky="w")
            # ---КОНЕЦ ИЗМЕНЕНИЙ---
            label = ctk.CTkLabel(frame, text=p['label'], font=bold_font); label.grid(row=0, column=0, padx=(30, 0), sticky="w")
            min_val_label = ctk.CTkLabel(frame, text=f"Min: {p['default_min']}", width=80); min_val_label.grid(row=1, column=0, padx=(30, 0), sticky="w")
            min_slider = ctk.CTkSlider(frame, from_=p['range'][0], to=p['range'][1]); min_slider.set(p['default_min']); min_slider.grid(row=1, column=1, sticky="ew")
            max_val_label = ctk.CTkLabel(frame, text=f"Max: {p['default_max']}", width=80); max_val_label.grid(row=2, column=0, padx=(30, 0), sticky="w")
            max_slider = ctk.CTkSlider(frame, from_=p['range'][0], to=p['range'][1]); max_slider.set(p['default_max']); max_slider.grid(row=2, column=1, sticky="ew")
            def make_cmd(k, lbl, is_flt): return lambda val: self.set_active_preview(k, val, lbl, is_flt)
            min_slider.configure(command=make_cmd(key, min_val_label, p['is_float'])); max_slider.configure(command=make_cmd(key, max_val_label, p['is_float']))
            self.controls[key] = {'check': check, 'min': min_slider, 'max': max_slider}; row += 1

        checkboxes_frame = ctk.CTkFrame(self.scrollable_frame, fg_color="transparent"); checkboxes_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=5); row+=1
        self.boolean_checkboxes = {} 
        for key, text in [('gradient_enabled', "Градиент"), ('noise_colorized', "Цветной шум"), ('bars_enabled', "Гориз. полосы"), ('diag_bars_enabled', "Диаг. полосы"), ('circles_enabled', "Круги"), ('water_enabled', "Водяная рябь"), ('lens_enabled', "Поле линз"), ('random_colors', "Случайные цвета")]:
            chk = ctk.CTkCheckBox(checkboxes_frame, text=text, font=bold_font, command=self.animate_preview); chk.pack(anchor="w")
            self.boolean_checkboxes[key] = chk

        self.preview_label = ctk.CTkLabel(self.main_frame, text=""); self.preview_label.grid(row=0, column=1, padx=10, pady=(0, 10), sticky="nsew")

        button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent"); button_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10,0))
        button_frame.grid_columnconfigure((0,1), weight=1)
        ctk.CTkButton(button_frame, text="Начать генерацию", command=self.start_batch).grid(row=0, column=0, padx=5, sticky="ew")
        ctk.CTkButton(button_frame, text="Отмена", command=self.destroy, fg_color="gray50").grid(row=0, column=1, padx=5, sticky="ew")
        
        self._load_config() 
        self.animate_preview()

    def _load_config(self): 
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Загрузка состояний слайдеров и их чекбоксов
            param_config = config.get('params', {})
            for key, item in self.controls.items():
                if key in param_config:
                    cfg = param_config[key]
                    if cfg.get('check', True): item['check'].select()
                    else: item['check'].deselect()
                    
                    # ---НАЧАЛО ИЗМЕНЕНИЙ---
                    # Получаем определения для текущего параметра, чтобы взять из них диапазон по умолчанию
                    p_def = self.master.param_definitions[key]
                    item['min'].set(cfg.get('min', p_def['range'][0]))
                    item['max'].set(cfg.get('max', p_def['range'][1]))
                    # ---КОНЕЦ ИЗМЕНЕНИЙ---

            # Загрузка состояний булевых чекбоксов
            boolean_config = config.get('booleans', {})
            for key, chk in self.boolean_checkboxes.items():
                if key in boolean_config:
                    if boolean_config[key]: chk.select()
                    else: chk.deselect()

        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            print("Файл конфигурации пакетной генерации не найден или поврежден. Используются значения по умолчанию.")

    def _save_config(self): # ---НАЧАЛО ИЗМЕНЕНИЙ---
        config_to_save = {'params': {}, 'booleans': {}}
        # Сохранение состояний слайдеров и их чекбоксов
        for key, item in self.controls.items():
            config_to_save['params'][key] = {
                'check': bool(item['check'].get()),
                'min': item['min'].get(),
                'max': item['max'].get()
            }
        # Сохранение состояний булевых чекбоксов
        for key, chk in self.boolean_checkboxes.items():
            config_to_save['booleans'][key] = bool(chk.get())
            
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=4)
        except IOError as e:
            print(f"Не удалось сохранить конфигурацию пакетной генерации: {e}") # ---КОНЕЦ ИЗМЕНЕНИЙ---

    def set_active_preview(self, param_key, value, label_widget, is_float):
        prefix = "Min: " if "Min" in label_widget.cget("text") else "Max: "
        label_widget.configure(text=f"{prefix}{value:.2f}" if is_float else f"{prefix}{int(value)}")
        if self.active_preview_param != param_key: self.active_preview_param = param_key; self.preview_animation_state = 0
    def animate_preview(self):
        # Отменяем предыдущую задачу, если она есть, чтобы избежать множественных вызовов
        if self.preview_animation_job:
            self.after_cancel(self.preview_animation_job)
            self.preview_animation_job = None

        params = self.master.get_current_params()
        params['text'] = "ПРИМЕР\nТЕКСТА\n(start)" # Используем текст из вашего скриншота

        # Применяем состояние всех флажков к параметрам превью
        for key, chk in self.boolean_checkboxes.items():
            if chk.get():
                params[key] = True
            else:
                params[key] = False # Явно отключаем эффект, если флажок снят

        # Специальная обработка для цветов
        if self.boolean_checkboxes['random_colors'].get():
            if self.boolean_checkboxes['gradient_enabled'].get():
                # Рандомизируем цвета градиента для превью
                # ---НАЧАЛО ИЗМЕНЕНИЙ---
                num_colors = self.controls['gradient_colors_count']['min'].get() if self.preview_animation_state == 0 else self.controls['gradient_colors_count']['max'].get()
                # ---КОНЕЦ ИЗМЕНЕНИЙ---
                params['gradient_colors'] = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(int(num_colors))]
            else:
                # Рандомизируем цвет текста для превью
                params['text_color'] = tuple(random.randint(150, 255) for _ in range(3))
            params['bg_color'] = tuple(random.randint(0, 100) for _ in range(3))

        param_key = self.active_preview_param
        if param_key in self.controls:
            min_val, max_val = self.controls[param_key]['min'].get(), self.controls[param_key]['max'].get()
            if min_val > max_val: min_val, max_val = max_val, min_val
            # Анимация значения активного слайдера
            val = min_val if self.preview_animation_state == 0 else (min_val + max_val) / 2 if self.preview_animation_state == 1 else max_val
            params[param_key] = val

        preview_img = generate_image_base(**params)
        ctk_img = ctk.CTkImage(light_image=preview_img, size=(250, 250))
        self.preview_label.configure(image=ctk_img)
        self.preview_animation_state = (self.preview_animation_state + 1) % 3
        self.preview_animation_job = self.after(700, self.animate_preview)
    def start_batch(self):
        self.config = {'params': {}}
        try:
            for key, item in self.controls.items():
                min_val, max_val = item['min'].get(), item['max'].get()
                if min_val > max_val: min_val, max_val = max_val, min_val
                self.config['params'][key] = {'randomize': bool(item['check'].get()), 'min': min_val, 'max': max_val}
            
            # ---НАЧАЛО ИЗМЕНЕНИЙ---
            for key, chk in self.boolean_checkboxes.items():
                self.config[key] = {'randomize': bool(chk.get())}
            # ---КОНЕЦ ИЗМЕНЕНИЙ---
            self.master.start_batch_generation_thread(self.config); self.destroy()
        except ValueError: messagebox.showerror("Ошибка", "Ошибка при чтении параметров.")
    def destroy(self):
        self._save_config() # ---НАЧАЛО ИЗМЕНЕНИЙ---
        if self.preview_animation_job: self.after_cancel(self.preview_animation_job)
        super().destroy()

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Генератор v1.1"); self.geometry("1200x850")
        self.batch_config_win = None # Добавляем атрибут для хранения ссылки на окно
        self.current_image: Image.Image = None; self.cached_noise_layer: Image.Image = None
        self.last_noise_params: Dict[str, Any] = {}; self.force_noise_regeneration = True
        self._update_job_id = None
        
        self.font_mapping = {
            "Шрифт 1": resource_path("font1.ttf"), "Шрифт 2": resource_path("font2.ttf"), 
            "Шрифт 3": resource_path("font3.ttf"), "Шрифт 4": resource_path("font4.ttf"), 
            "Рукописный 1": resource_path("font5.ttf"), "Рукописный 2": resource_path("font6.ttf"),
            "Цветной (TTF) (English Only)": resource_path("color1.ttf"), "Цветной 1 (OTF)": resource_path("color2.otf"),
            "Цветной 2 (OTF) (English Only)": resource_path("color3.otf")
        }
        self.color_font_names = {"Цветной 1 (OTF)", "Цветной 2 (OTF) (English Only)"}
        self.available_fonts = [name for name, path in self.font_mapping.items() if os.path.exists(path)]
        self.save_folder_path = None
        
        self.text_color, self.bg_color = (240, 240, 240), (20, 20, 20)
        self.gradient_colors = [(230, 50, 50), (50, 50, 230), (50, 230, 50), (230, 230, 50), (230, 50, 230)]
        self.gradient_color_buttons = []

        self.param_definitions = {
        'angle': {'label': 'Угол', 'range': (-45, 45), 'default_min': -20, 'default_max': 20, 'is_float': True}, 'font_size': {'label': 'Размер шрифта', 'range': (24, 72), 'default_min': 36, 'default_max': 48, 'is_float': False},
        'wave_amplitude': {'label': 'Ампл. волны', 'range': (0, 30), 'default_min': 5, 'default_max': 20, 'is_float': True}, 'wave_frequency': {'label': 'Част. волны', 'range': (0, 0.2), 'default_min': 0.05, 'default_max': 0.15, 'is_float': True},
        'noise_strength': {'label': 'Сила шума', 'range': (0, 100), 'default_min': 20, 'default_max': 80, 'is_float': False}, 'noise_caliber': {'label': 'Калибр шума', 'range': (1, 26), 'default_min': 1, 'default_max': 10, 'is_float': False},
        'bars_count': {'label': 'Кол-во гор. полос', 'range': (0, 20), 'default_min': 2, 'default_max': 10, 'is_float': False}, 'bars_opacity': {'label': 'Прозр. гор. полос', 'range': (0, 255), 'default_min': 50, 'default_max': 150, 'is_float': False},
        'diag_bars_count': {'label': 'Кол-во диаг. полос', 'range': (0, 10), 'default_min': 2, 'default_max': 6, 'is_float': False}, 'diag_bars_opacity': {'label': 'Прозр. диаг. полос', 'range': (0, 255), 'default_min': 40, 'default_max': 120, 'is_float': False},
        'circles_count': {'label': 'Кол-во кругов', 'range': (0, 15), 'default_min': 3, 'default_max': 8, 'is_float': False}, 'circles_opacity': {'label': 'Прозр. кругов', 'range': (0, 255), 'default_min': 30, 'default_max': 100, 'is_float': False},
        'water_freq': {'label': 'Частота ряби', 'range': (0, 0.1), 'default_min': 0.02, 'default_max': 0.08, 'is_float': True}, 'water_amp': {'label': 'Амплитуда ряби', 'range': (0, 30), 'default_min': 5, 'default_max': 20, 'is_float': True},
        'lens_cell_size': {'label': 'Размер линзы', 'range': (5, 50), 'default_min': 15, 'default_max': 35, 'is_float': False}, 'lens_blur_radius': {'label': 'Размытие линзы', 'range': (0, 5), 'default_min': 1, 'default_max': 3, 'is_float': True},
        'gradient_colors_count': {'label': 'Кол-во цветов градиента', 'range': (2, 5), 'default_min': 2, 'default_max': 5, 'is_float': False},
    }

        if not self.available_fonts:
            messagebox.showerror("Критическая ошибка", "Файлы шрифтов (font*.ttf, color*.otf) не найдены!\n\nУбедитесь, что шрифты находятся в той же папке, что и .exe файл.")
            self.after(100, self.destroy)
            return
            
        try: 
            self.iconbitmap(resource_path("icon.ico"))
        except tk.TclError: print("Файл иконки 'icon.ico' не найден.")
        ctk.set_appearance_mode("dark")
        self.grid_columnconfigure(0, weight=1); self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)
        self.protocol("WM_DELETE_WINDOW", self.destroy) # Добавляем перехват закрытия
        self.setup_ui_layout()
        self.setup_controls()
        self.update_image()
        if not os.path.exists(".first_run_complete"): self.after(200, self.show_tutorial_window)

    def setup_ui_layout(self):
        left_frame = ctk.CTkFrame(self)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        left_frame.grid_rowconfigure(1, weight=1); left_frame.grid_columnconfigure(0, weight=1)
        utility_button_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        utility_button_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ctk.CTkButton(utility_button_frame, text="💾 Сохранить", command=self.save_settings, fg_color="gray40", height=26).pack(side="left", padx=5, fill="x", expand=True)
        ctk.CTkButton(utility_button_frame, text="📂 Загрузить", command=self.load_settings, fg_color="gray40", height=26).pack(side="left", padx=5, fill="x", expand=True)
        ctk.CTkButton(utility_button_frame, text="🔄 Сбросить", command=self.reset_settings, fg_color="gray40", height=26).pack(side="left", padx=5, fill="x", expand=True)
        ctk.CTkButton(utility_button_frame, text="🎓 Обучение", command=self.show_tutorial_window, fg_color="gray40", height=26).pack(side="left", padx=5, fill="x", expand=True)
        tab_font = ctk.CTkFont(family="Verdana", size=14, weight="bold")
        self.tab_view = ctk.CTkTabview(left_frame, fg_color=self.cget("fg_color"), segmented_button_selected_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"], segmented_button_unselected_color="gray25", height=40)
        self.tab_view._segmented_button.configure(font=tab_font, height=40)
        self.tab_view.grid(row=1, column=0, sticky="nsew")
        self.tab_view.add("Основные"); self.tab_view.add("Искажения"); self.tab_view.add("Пакетная генерация")
        self.controls_frame = self.tab_view.tab("Основные"); self.distortions_frame = self.tab_view.tab("Искажения"); self.batch_frame = self.tab_view.tab("Пакетная генерация")
        self.controls_frame.grid_columnconfigure(1, weight=1); self.distortions_frame.grid_columnconfigure(1, weight=1); self.batch_frame.grid_columnconfigure(0, weight=1)
        self.image_frame = ctk.CTkFrame(self)
        self.image_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        top_image_bar = ctk.CTkFrame(self.image_frame, fg_color="transparent")
        top_image_bar.pack(side="top", fill="x", pady=5, padx=5)
        change_folder_font = ctk.CTkFont(family="Verdana", size=12, underline=True)
        self.change_folder_button = ctk.CTkButton(top_image_bar, text="Выбрать папку", command=self._select_save_folder, fg_color="transparent", text_color="gray70", hover=False, font=change_folder_font, width=120)
        self.change_folder_button.pack(side="right", padx=10)
        self.save_image_button = ctk.CTkButton(top_image_bar, text="💾 Сохранить текущее", command=self.save_image)
        self.save_image_button.pack(side="right")
        self.image_label = ctk.CTkLabel(self.image_frame, text="")
        self.image_label.pack(expand=True, fill="both")

    def _update_ui_states(self):
        is_color_font = self.font_combo.get() in self.color_font_names
        gradient_enabled = self.gradient_checkbox.get() and not is_color_font
        num_gradient_colors = int(self.gradient_colors_slider.get())

        # Блокируем всё, если выбран цветной шрифт
        self.text_color_button.configure(state="disabled" if is_color_font else "normal")
        self.gradient_checkbox.configure(state="disabled" if is_color_font else "normal")
        
        # Управляем видимостью элементов градиента
        self.gradient_colors_slider.configure(state="disabled" if not gradient_enabled else "normal")
        for i, btn in enumerate(self.gradient_color_buttons):
            btn.configure(state="normal" if gradient_enabled and i < num_gradient_colors else "disabled")
        
        if is_color_font:
            self.gradient_checkbox.deselect()

        self._schedule_update()

    def copy_text(self, event=None):
        try:
            selected_text = self.textbox.get("sel.first", "sel.last")
            self.clipboard_clear()
            self.clipboard_append(selected_text)
        except tk.TclError:
            # Ничего не выделено, копировать нечего
            pass
        return "break" # Предотвращаем дальнейшую обработку события

    def paste_text(self, event=None):
        try:
            text_to_paste = self.clipboard_get()
            # Удаляем выделенный текст (если есть) перед вставкой
            if self.textbox.tag_ranges("sel"):
                self.textbox.delete("sel.first", "sel.last")
            
            self.textbox.insert(tk.INSERT, text_to_paste)
            self._schedule_update() # Обновляем картинку после вставки
        except tk.TclError:
            # В буфере обмена нет текста
            pass
        return "break"

    def cut_text(self, event=None):
        self.copy_text() # Сначала копируем выделенный текст
        try:
            # Затем удаляем его
            self.textbox.delete("sel.first", "sel.last")
            self._schedule_update() # Обновляем картинку после вырезания
        except tk.TclError:
            pass
        return "break"
    
    def pick_gradient_color(self, index: int):
            color = colorchooser.askcolor(title=f"Выберите цвет градиента №{index+1}", initialcolor=self.gradient_colors[index])
            if color and color[0]:
                self.gradient_colors[index] = tuple(int(c) for c in color[0])
                self._schedule_update()

    def randomize_gradient(self):
            count = int(self.gradient_colors_slider.get())
            for i in range(count):
                self.gradient_colors[i] = tuple(random.randint(0, 255) for _ in range(3))
            self._schedule_update()

    def _create_slider(self, master, text, from_, to, initial, row):
        label = ctk.CTkLabel(master, text=text, font=("Verdana", 12, "bold")); label.grid(row=row, column=0, padx=10, sticky="w")
        # ---НАЧАЛО ИЗМЕНЕНИЙ---
        slider = ctk.CTkSlider(master, from_=from_, to=to, command=self._schedule_update); slider.set(initial); slider.grid(row=row, column=1, padx=10, pady=5, sticky="ew")
        # ---КОНЕЦ ИЗМЕНЕНИЙ---
        return slider

    def setup_controls(self):
        bold_font = ("Verdana", 12, "bold"); row = 0
        ctk.CTkLabel(self.controls_frame, text="Текст:", font=bold_font).grid(row=row, column=0, padx=10, pady=(10,0), sticky="w"); row+=1
        self.textbox = ctk.CTkTextbox(self.controls_frame, height=80); self.textbox.grid(row=row, column=0, columnspan=2, padx=10, pady=5, sticky="ew"); self.textbox.insert("0.0", "Пример\nтекста (start)")
        self.textbox.bind("<KeyRelease>", self._schedule_update)
        self.textbox.bind("<Control-c>", self.copy_text)
        self.textbox.bind("<Control-v>", self.paste_text)
        self.textbox.bind("<Control-x>", self.cut_text)
        row+=1
        ctk.CTkLabel(self.controls_frame, text="Шрифт:", font=bold_font).grid(row=row, column=0, padx=10, sticky="w")
        self.font_combo = ctk.CTkComboBox(self.controls_frame, values=list(self.available_fonts), command=lambda val: (self._update_ui_states(), self._schedule_update(val))); self.font_combo.grid(row=row, column=1, padx=10, pady=5, sticky="ew"); row+=1
        self.font_combo.configure(state="readonly")
        if self.available_fonts: self.font_combo.set(random.choice(list(self.available_fonts)))
        self.font_size_slider = self._create_slider(self.controls_frame, "Размер шрифта:", 24, 72, 40, row); row+=1
        text_transform_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent"); text_transform_frame.grid(row=row, column=0, columnspan=2, sticky="ew"); text_transform_frame.grid_columnconfigure(1, weight=1); row+=1
        ctk.CTkLabel(text_transform_frame, text="Трансформации текста", font=bold_font).grid(row=0, column=0, sticky="w", padx=10)
        ctk.CTkButton(text_transform_frame, text="🎲", width=40, command=self.randomize_text_transforms).grid(row=0, column=1, sticky="e", padx=10)
        self.text_offset_x_slider = self._create_slider(self.controls_frame, "Смещение X:", -100, 100, 0, row); row+=1
        self.text_offset_y_slider = self._create_slider(self.controls_frame, "Смещение Y:", -100, 100, 0, row); row+=1
        self.angle_slider = self._create_slider(self.controls_frame, "Угол наклона:", -45, 45, 0, row); row+=1
        self.wave_amp_slider = self._create_slider(self.controls_frame, "Амплитуда волны:", 0, 30, 10, row); row+=1
        self.wave_freq_slider = self._create_slider(self.controls_frame, "Частота волны:", 0, 0.2, 0.07, row); row+=1
        
        # Удаляем старый color_frame и создаем единый новый блок
        
        # Основной фрейм для всех настроек цвета и градиента
        color_grad_frame = ctk.CTkFrame(self.controls_frame); 
        color_grad_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=10, pady=10); row+=1
        color_grad_frame.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(color_grad_frame, text="Цвет и Градиент", font=bold_font).grid(row=0, column=0, columnspan=2, padx=10, pady=(5,5))

        # Кнопки выбора цвета текста и фона
        self.text_color_button = ctk.CTkButton(color_grad_frame, text="Цвет Текста", command=self.pick_text_color)
        self.text_color_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.bg_color_button = ctk.CTkButton(color_grad_frame, text="Цвет Фона", command=self.pick_bg_color)
        self.bg_color_button.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        # Флажок включения градиента
        self.gradient_checkbox = ctk.CTkCheckBox(color_grad_frame, text="Включить градиент", font=bold_font, command=self._update_ui_states)
        self.gradient_checkbox.grid(row=2, column=0, columnspan=2, padx=10, pady=(5,0), sticky="w")
        
        # Переключатель направления градиента
        self.gradient_direction_switch = ctk.CTkSegmentedButton(color_grad_frame, values=["↓ Вертикальный", "→ Горизонтальный"], command=self._schedule_update)
        self.gradient_direction_switch.set("↓ Вертикальный")
        self.gradient_direction_switch.grid(row=3, column=0, columnspan=2, padx=5, pady=10, sticky="ew")

        # Слайдер количества цветов
        self.gradient_colors_slider = self._create_slider(color_grad_frame, "Кол-во цветов:", 2, 5, 2, 4)
        self.gradient_colors_slider.configure(command=lambda val: (self._update_ui_states(), self._schedule_update()))
        
        # Фрейм для кнопок выбора цветов градиента
        self.gradient_buttons_frame = ctk.CTkFrame(color_grad_frame, fg_color="transparent")
        self.gradient_buttons_frame.grid(row=5, column=0, columnspan=2, sticky="ew", padx=5, pady=0)
        self.gradient_buttons_frame.grid_columnconfigure((0,1,2,3,4), weight=1)

        self.gradient_color_buttons.clear()
        for i in range(5):
            btn = ctk.CTkButton(self.gradient_buttons_frame, text=f"{i+1}", width=30, command=lambda i=i: self.pick_gradient_color(i))
            btn.grid(row=0, column=i, padx=3, pady=3, sticky="ew")
            self.gradient_color_buttons.append(btn)
        
        # Кнопки рандомизации
        random_buttons_frame = ctk.CTkFrame(color_grad_frame, fg_color="transparent")
        random_buttons_frame.grid(row=6, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        random_buttons_frame.grid_columnconfigure((0,1), weight=1)
        
        ctk.CTkButton(random_buttons_frame, text="🎲 Случайные Цвета", command=self.randomize_colors).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(random_buttons_frame, text="🎲 Случайный Градиент", command=self.randomize_gradient).grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.controls_frame.grid_rowconfigure(row, weight=1)
        ctk.CTkLabel(self.controls_frame, text="Создано в ТГАЧЕ @dvach_chatbot Тг", text_color="gray50").grid(row=row, column=0, columnspan=2, padx=10, pady=10, sticky="s")
    
        scroll_frame = ctk.CTkScrollableFrame(self.distortions_frame, label_text="Эффекты", label_font=bold_font); scroll_frame.pack(expand=True, fill="both", padx=5, pady=5); scroll_frame.grid_columnconfigure(1, weight=1)
        s_row=0; add_separator = lambda r: ctk.CTkFrame(scroll_frame, height=1, border_width=1).grid(row=r, column=0, columnspan=2, sticky="ew", padx=10, pady=10) or r+1
        ctk.CTkLabel(scroll_frame, text="Шум", font=bold_font).grid(row=s_row, column=0, sticky="w", padx=10); ctk.CTkButton(scroll_frame, text="🎲", width=40, command=self.randomize_noise).grid(row=s_row, column=1, sticky="e", padx=10); s_row+=1
        self.noise_slider = self._create_slider(scroll_frame, "Сила шума:", 0, 100, 50, s_row); s_row+=1
        self.noise_caliber_slider = self._create_slider(scroll_frame, "Калибр шума:", 1, 26, 1, s_row); s_row+=1
        self.check_noise_colorized = ctk.CTkCheckBox(scroll_frame, text="Цветной шум", command=self.update_image, font=bold_font); self.check_noise_colorized.grid(row=s_row, column=0, columnspan=2, padx=10, pady=5, sticky="w"); s_row+=1; s_row=add_separator(s_row)
        ctk.CTkLabel(scroll_frame, text="Горизонтальные полосы:", font=bold_font).grid(row=s_row, column=0, sticky="w", padx=10); ctk.CTkButton(scroll_frame, text="🎲", width=40, command=self.randomize_bars).grid(row=s_row, column=1, sticky="e", padx=10); s_row+=1
        self.check_bars_enabled = ctk.CTkCheckBox(scroll_frame, text="Включить", command=self.update_image, font=bold_font); self.check_bars_enabled.grid(row=s_row, column=0, columnspan=2, padx=10, pady=5, sticky="w"); s_row+=1
        self.bars_count_slider = self._create_slider(scroll_frame, "Количество:", 0, 20, 5, s_row); s_row+=1
        self.bars_opacity_slider = self._create_slider(scroll_frame, "Прозрачность:", 0, 255, 128, s_row); s_row+=1; s_row=add_separator(s_row)
        ctk.CTkLabel(scroll_frame, text="Диагональные полосы:", font=bold_font).grid(row=s_row, column=0, sticky="w", padx=10); ctk.CTkButton(scroll_frame, text="🎲", width=40, command=self.randomize_diag_bars).grid(row=s_row, column=1, sticky="e", padx=10); s_row+=1
        self.check_diag_bars_enabled = ctk.CTkCheckBox(scroll_frame, text="Включить", command=self.update_image, font=bold_font); self.check_diag_bars_enabled.grid(row=s_row, column=0, columnspan=2, padx=10, pady=5, sticky="w"); s_row+=1
        self.diag_bars_count_slider = self._create_slider(scroll_frame, "Количество:", 0, 10, 4, s_row); s_row+=1
        self.diag_bars_opacity_slider = self._create_slider(scroll_frame, "Прозрачность:", 0, 255, 100, s_row); s_row+=1; s_row=add_separator(s_row)
        ctk.CTkLabel(scroll_frame, text="Цветные круги:", font=bold_font).grid(row=s_row, column=0, sticky="w", padx=10); ctk.CTkButton(scroll_frame, text="🎲", width=40, command=self.randomize_circles).grid(row=s_row, column=1, sticky="e", padx=10); s_row+=1
        self.check_circles_enabled = ctk.CTkCheckBox(scroll_frame, text="Включить", command=self.update_image, font=bold_font); self.check_circles_enabled.grid(row=s_row, column=0, columnspan=2, padx=10, pady=5, sticky="w"); s_row+=1
        self.circles_count_slider = self._create_slider(scroll_frame, "Количество:", 0, 15, 6, s_row); s_row+=1
        self.circles_opacity_slider = self._create_slider(scroll_frame, "Прозрачность:", 0, 255, 80, s_row); s_row+=1; s_row=add_separator(s_row)
        ctk.CTkLabel(scroll_frame, text="Водяная рябь:", font=bold_font).grid(row=s_row, column=0, sticky="w", padx=10); ctk.CTkButton(scroll_frame, text="🎲", width=40, command=self.randomize_water).grid(row=s_row, column=1, sticky="e", padx=10); s_row+=1
        self.check_water_enabled = ctk.CTkCheckBox(scroll_frame, text="Включить", command=self.update_image, font=bold_font); self.check_water_enabled.grid(row=s_row, column=0, columnspan=2, padx=10, pady=5, sticky="w"); s_row+=1
        self.water_freq_slider = self._create_slider(scroll_frame, "Частота:", 0, 0.1, 0.05, s_row); s_row+=1
        self.water_amp_slider = self._create_slider(scroll_frame, "Амплитуда:", 0, 30, 10, s_row); s_row+=1; s_row=add_separator(s_row)
        ctk.CTkLabel(scroll_frame, text="Поле линз:", font=bold_font).grid(row=s_row, column=0, sticky="w", padx=10); ctk.CTkButton(scroll_frame, text="🎲", width=40, command=self.randomize_lens).grid(row=s_row, column=1, sticky="e", padx=10); s_row+=1
        self.check_lens_enabled = ctk.CTkCheckBox(scroll_frame, text="Включить", command=self.update_image, font=bold_font); self.check_lens_enabled.grid(row=s_row, column=0, columnspan=2, padx=10, pady=5, sticky="w"); s_row+=1
        self.lens_cell_size_slider = self._create_slider(scroll_frame, "Размер ячейки:", 5, 50, 20, s_row); s_row+=1
        self.lens_blur_radius_slider = self._create_slider(scroll_frame, "Размытие:", 0, 5, 1, s_row); s_row+=1
        row=0
        ctk.CTkLabel(self.batch_frame, text="Количество изображений (1-1000):", font=bold_font).grid(row=row, column=0, columnspan=2, padx=10, pady=(10,0), sticky="w"); row+=1
        self.batch_count_entry = ctk.CTkEntry(self.batch_frame); self.batch_count_entry.grid(row=row, column=0, columnspan=2, padx=10, pady=5, sticky="ew"); row+=1
        self.batch_button = ctk.CTkButton(self.batch_frame, text="Настроить и запустить", command=self.open_batch_config_window); self.batch_button.grid(row=row, column=0, columnspan=2, padx=10, pady=10, sticky="ew"); row+=1
        self.progress_bar = ctk.CTkProgressBar(self.batch_frame); self.progress_bar.grid(row=row, column=0, columnspan=2, padx=10, pady=5, sticky="ew"); self.progress_bar.set(0); row+=1
        button_frame = ctk.CTkFrame(self.batch_frame, fg_color="transparent"); button_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=10); button_frame.grid_columnconfigure((0,1), weight=1)
        self.random_button = ctk.CTkButton(button_frame, text="🎲 Рандомизировать всё", command=self.randomize_params); self.random_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

    def _schedule_update(self, event=None):
        if self._update_job_id:
            self.after_cancel(self._update_job_id)
        self._update_job_id = self.after(250, self.update_image)

    def show_tutorial_window(self): TutorialWindow(self)
    def pick_text_color(self):
        color = colorchooser.askcolor(title="Цвет текста", initialcolor=self.text_color)
        if color and color[0]:
            self.text_color = tuple(int(c) for c in color[0])
            self.update_image()
    def pick_bg_color(self):
        color = colorchooser.askcolor(title="Цвет фона", initialcolor=self.bg_color)
        if color and color[0]:
            self.bg_color = tuple(int(c) for c in color[0])
            self.update_image()
    def force_regenerate_noise(self): self.force_noise_regeneration = True; self.update_image()
    def get_current_params(self) -> Dict[str, Any]:
        font_name = self.font_combo.get()
        is_color_font = font_name in self.color_font_names
        num_gradient_colors = int(self.gradient_colors_slider.get())

        params = {
            'text': self.textbox.get("0.0", "end-1c"), 'angle': self.angle_slider.get(), 
            'wave_amplitude': self.wave_amp_slider.get(), 'wave_frequency': self.wave_freq_slider.get(), 
            'noise_strength': self.noise_slider.get(), 'noise_colorized': bool(self.check_noise_colorized.get()), 
            'noise_caliber': self.noise_caliber_slider.get(), 'bars_enabled': bool(self.check_bars_enabled.get()), 
            'bars_count': self.bars_count_slider.get(), 'bars_opacity': self.bars_opacity_slider.get(), 
            'diag_bars_enabled': bool(self.check_diag_bars_enabled.get()), 'diag_bars_count': self.diag_bars_count_slider.get(), 
            'diag_bars_opacity': self.diag_bars_opacity_slider.get(), 'circles_enabled': bool(self.check_circles_enabled.get()), 
            'circles_count': self.circles_count_slider.get(), 'circles_opacity': self.circles_opacity_slider.get(), 
            'water_enabled': bool(self.check_water_enabled.get()), 'water_freq': self.water_freq_slider.get(), 
            'water_amp': self.water_amp_slider.get(), 'lens_enabled': bool(self.check_lens_enabled.get()), 
            'lens_cell_size': self.lens_cell_size_slider.get(), 'lens_blur_radius': self.lens_blur_radius_slider.get(), 
            'text_color': self.text_color, 'bg_color': self.bg_color, 
            'font_path': self.font_mapping[font_name], 'font_size': self.font_size_slider.get(), 
            'text_offset_x': self.text_offset_x_slider.get(), 'text_offset_y': self.text_offset_y_slider.get(),
            'is_color_font': is_color_font,
            'gradient_enabled': self.gradient_checkbox.get() and not is_color_font,
            'gradient_colors': self.gradient_colors[:num_gradient_colors],
            'gradient_direction': 'horizontal' if self.gradient_direction_switch.get() == "→ Горизонтальный" else 'vertical'
        }
        return params

    def randomize_colors(self):
        # Эта функция теперь рандомизирует либо цвет текста, либо градиент
        if self.gradient_checkbox.get():
             self.randomize_gradient()
        else:
             self.text_color = tuple(random.randint(150, 255) for _ in range(3))
        self.bg_color = tuple(random.randint(0, 100) for _ in range(3))
        self._schedule_update()

    def update_image(self, event=None):
        params = self.get_current_params()
        noise_params = {'strength': params['noise_strength'], 'colorized': params['noise_colorized'], 'caliber': params['noise_caliber']}
        if self.force_noise_regeneration or noise_params != self.last_noise_params: self.cached_noise_layer = generate_noise_layer(**noise_params); self.last_noise_params = noise_params; self.force_noise_regeneration = False
        base_image = generate_image_base(**params); final_image = base_image
        if self.cached_noise_layer: final_image = Image.alpha_composite(base_image.convert("RGBA"), self.cached_noise_layer).convert("RGB")
        self.current_image = final_image; ctk_image = ctk.CTkImage(light_image=self.current_image, size=(512, 512)); self.image_label.configure(image=ctk_image)
    def save_settings(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if not file_path: return
        try:
            with open(file_path, 'w', encoding='utf-8') as f: json.dump(self.get_current_params(), f, indent=4)
        except Exception as e: messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")
    def load_settings(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not file_path: return
        try:
            with open(file_path, 'r', encoding='utf-8') as f: params = json.load(f)
            self.set_params_from_dict(params)
        except Exception as e: messagebox.showerror("Ошибка", f"Не удалось загрузить:\n{e}")
    def set_params_from_dict(self, params: Dict[str, Any]):
        self.textbox.delete("0.0", "end"); self.textbox.insert("0.0", params.get('text', 'Пример\nтекста'))
        for s_name, key, default in [('angle_slider', 'angle', 0), ('wave_amp_slider', 'wave_amplitude', 10), ('wave_freq_slider', 'wave_frequency', 0.07), ('noise_slider', 'noise_strength', 50), ('noise_caliber_slider', 'noise_caliber', 1), ('bars_count_slider', 'bars_count', 5), ('bars_opacity_slider', 'bars_opacity', 128), ('diag_bars_count_slider', 'diag_bars_count', 4), ('diag_bars_opacity_slider', 'diag_bars_opacity', 100), ('circles_count_slider', 'circles_count', 6), ('circles_opacity_slider', 'circles_opacity', 80), ('water_freq_slider', 'water_freq', 0.05), ('water_amp_slider', 'water_amp', 10), ('lens_cell_size_slider', 'lens_cell_size', 20), ('lens_blur_radius_slider', 'lens_blur_radius', 1), ('text_offset_x_slider', 'text_offset_x', 0), ('text_offset_y_slider', 'text_offset_y', 0), ('font_size_slider', 'font_size', 40)]:
            if hasattr(self, s_name): getattr(self, s_name).set(params.get(key, default))
        for key, checkbox in [('noise_colorized', self.check_noise_colorized), ('bars_enabled', self.check_bars_enabled), ('diag_bars_enabled', self.check_diag_bars_enabled), ('circles_enabled', self.check_circles_enabled), ('water_enabled', self.check_water_enabled), ('lens_enabled', self.check_lens_enabled)]:
            if params.get(key, False): checkbox.select()
            else: checkbox.deselect()
        font_path = params.get('font_path', random.choice(list(self.font_mapping.values())))
        display_name = next((name for name, path in self.font_mapping.items() if path == font_path), list(self.available_fonts)[0])
        self.font_combo.set(display_name)
        self.text_color, self.bg_color = tuple(params.get('text_color', [240,240,240])), tuple(params.get('bg_color', [20,20,20]))
        self.force_noise_regeneration = True; self.update_image()
    def reset_settings(self): self.set_params_from_dict({})
    def randomize_text_transforms(self): self.text_offset_x_slider.set(random.uniform(-50, 50)); self.text_offset_y_slider.set(random.uniform(-50, 50)); self.angle_slider.set(random.uniform(-45, 45)); self.wave_amp_slider.set(random.uniform(5, 20)); self.wave_freq_slider.set(random.uniform(0.05, 0.15)); self.font_size_slider.set(random.randint(30, 60)); self.update_image()
    def randomize_noise(self): self.noise_slider.set(random.randint(20, 80)); self.noise_caliber_slider.set(random.randint(1, 15)); (self.check_noise_colorized.select() if random.random() > 0.5 else self.check_noise_colorized.deselect()); self.force_regenerate_noise()
    def randomize_bars(self): self.bars_count_slider.set(random.randint(2, 10)); self.bars_opacity_slider.set(random.randint(50, 150)); self.update_image()
    def randomize_diag_bars(self): self.diag_bars_count_slider.set(random.randint(1, 8)); self.diag_bars_opacity_slider.set(random.randint(40, 120)); self.update_image()
    def randomize_circles(self): self.circles_count_slider.set(random.randint(3, 12)); self.circles_opacity_slider.set(random.randint(30, 100)); self.update_image()
    def randomize_water(self): self.water_freq_slider.set(random.uniform(0.01, 0.1)); self.water_amp_slider.set(random.uniform(5, 25)); self.update_image()
    def randomize_lens(self): self.lens_cell_size_slider.set(random.randint(10, 40)); self.lens_blur_radius_slider.set(random.uniform(0.5, 3)); self.update_image()
    def randomize_params(self):
        self.randomize_text_transforms()
        self.randomize_noise()
        self.randomize_bars()
        self.randomize_diag_bars()
        self.randomize_circles()
        self.randomize_water()
        self.randomize_lens()

        for checkbox in [self.check_bars_enabled, self.check_diag_bars_enabled, self.check_circles_enabled, self.check_water_enabled, self.check_lens_enabled]:
            if random.random() > 0.5:
                checkbox.select()
            else:
                checkbox.deselect()

        # ---НАЧАЛО ИЗМЕНЕНИЙ---
        # Случайно решаем, включать ли градиент
        if random.random() > 0.5:
            self.gradient_checkbox.select()
        else:
            self.gradient_checkbox.deselect()
        
        # Случайно выбираем количество цветов для градиента
        self.gradient_colors_slider.set(random.randint(2, 5))

        # Вызываем randomize_colors(), который теперь "умный"
        self.randomize_colors()
        
        # Обновляем состояние UI, чтобы все изменения применились визуально
        self._update_ui_states()
    def _show_toast(self, message, duration=2000):
        toast = ctk.CTkToplevel(self)
        toast.overrideredirect(True)
        toast.geometry(f"+{self.winfo_x() + (self.winfo_width() // 2) - 100}+{self.winfo_y() + (self.winfo_height() // 2) - 20}")
        label = ctk.CTkLabel(toast, text=message, corner_radius=10, fg_color="gray20", padx=20, pady=10)
        label.pack()
        toast.after(duration, toast.destroy)
    def _select_save_folder(self):
        folder_path = filedialog.askdirectory(title="Выберите папку для сохранения изображений")
        if folder_path:
            self.save_folder_path = folder_path
            self._show_toast(f"Выбрана папка:\n{os.path.basename(folder_path)}")
            return True
        return False
    def save_image(self):
        if not self.current_image: return
        if not self.save_folder_path or not os.path.isdir(self.save_folder_path):
            if not self._select_save_folder():
                return
        try:
            existing_files = glob.glob(os.path.join(self.save_folder_path, 'image_*.png'))
            max_num = 0
            for file in existing_files:
                basename = os.path.basename(file)
                match = re.search(r'image_(\d+)\.png', basename)
                if match:
                    max_num = max(max_num, int(match.group(1)))
            next_num = max_num + 1
            filename = f"image_{next_num:05d}.png"
            file_path = os.path.join(self.save_folder_path, filename)
            self.current_image.save(file_path)
            self._show_toast(f"Сохранено как {filename}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить файл:\n{e}")
    def open_batch_config_window(self):
        if self.batch_config_win is not None and self.batch_config_win.winfo_exists():
            self.batch_config_win.focus()
            return
        self.batch_config_win = BatchConfigWindow(self)
    def destroy(self):
        if self.batch_config_win is not None and self.batch_config_win.winfo_exists():
            self.batch_config_win.destroy()
        super().destroy()
    def start_batch_generation_thread(self, config):
        thread = threading.Thread(target=self.run_batch_generation, args=(config,)); thread.daemon = True; thread.start()
    def run_batch_generation(self, config):
        try: count = int(self.batch_count_entry.get())
        except (ValueError, TypeError): count = 0
        if not 1 <= count <= 1000: messagebox.showerror("Ошибка", "Количество должно быть от 1 до 1000"); return
        folder_path = filedialog.askdirectory(title="Выберите папку для сохранения серии")
        if not folder_path: return
        self.batch_button.configure(state="disabled"); self.progress_bar.set(0)
        current_params = self.get_current_params()
        for i in range(count):
            batch_params = current_params.copy()
            # 1. Применяем рандомизацию для всех параметров со слайдерами (включая gradient_colors_count)
            for key, p_config in config['params'].items():
                if p_config['randomize']:
                    if key in self.param_definitions:
                        is_float = self.param_definitions[key]['is_float']
                        batch_params[key] = random.uniform(p_config['min'], p_config['max']) if is_float else random.randint(int(p_config['min']), int(p_config['max']))
            
            # ---НАЧАЛО ИЗМЕНЕНИЙ---
            # 2. Динамически применяем рандомизацию для всех булевых флажков (включая gradient_enabled)
            for key, p_config in config.items():
                if key == 'params': continue # Пропускаем словарь со слайдерами
                if p_config.get('randomize'):
                    batch_params[key] = random.choice([True, False])

            # 3. Обрабатываем логику цветов на основе флагов, установленных выше
            # Если включена рандомизация цветов, то рандомизируем базовые цвета (фон и простой текст)
            if batch_params.get('random_colors'):
                batch_params['bg_color'] = tuple(random.randint(0, 127) for _ in range(3))
                batch_params['text_color'] = tuple(random.randint(128, 255) for _ in range(3))
            
            # 4. Независимо от 'random_colors', если для этой картинки случайно включился градиент,
            # мы ОБЯЗАТЕЛЬНО должны рандомизировать его цвета.
            if batch_params.get('gradient_enabled'):
                num_colors = batch_params.get('gradient_colors_count', 2) # Кол-во цветов берем из рандомизированного слайдера
                batch_params['gradient_colors'] = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(num_colors)]
            # ---КОНЕЦ ИЗМЕНЕНИЙ---

            image_to_save = generate_image_base(**batch_params)
            noise_layer = generate_noise_layer(batch_params['noise_strength'], batch_params['noise_colorized'], batch_params['noise_caliber'])
            if noise_layer: image_to_save = Image.alpha_composite(image_to_save.convert("RGBA"), noise_layer).convert("RGB")
            image_to_save.save(os.path.join(folder_path, f"image_{i+1:04d}.png"))
            self.progress_bar.set((i + 1) / count)
        self.batch_button.configure(state="normal")
        messagebox.showinfo("Завершено", f"Пакетная генерация завершена.\nСохранено {count} изображений в папку:\n{folder_path}")

if __name__ == "__main__":
    app = App()
    app.mainloop()