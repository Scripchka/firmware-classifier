import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import models, transforms
import torch.nn as nn
import os
import numpy as np

class FirmwareClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("IoT Firmware Classifier")
        self.root.geometry("600x750")
        

        self.model = None
        self.model_path = "model/efficientnet_best.pth"
        

        self.class_names = ["benignware", "hackware", "malware"]  
        self.num_classes = len(self.class_names)  
        
        self.current_file = None
        

        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        tk.Label(self.root, text="🔬 IoT Firmware Security Classifier", 
                font=("Arial", 18, "bold"), fg="#2c3e50").pack(pady=10)
        

        
        frame_buttons = tk.Frame(self.root)
        frame_buttons.pack(pady=15)
        
        tk.Button(frame_buttons, text="📁 Загрузить изображение", 
                 command=self.load_image, width=24, height=2,
                 bg="#ecf0f1", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(frame_buttons, text="🔧 Загрузить бинарный файл", 
                 command=self.load_binary, width=24, height=2,
                 bg="#ecf0f1", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        preview_frame = tk.LabelFrame(self.root, text="Предпросмотр файла", 
                                     padx=10, pady=10, font=("Arial", 11))
        preview_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        self.preview_label = tk.Label(preview_frame, text="Файл не загружен\n\n(здесь будет предпросмотр)", 
                                     bg="#f8f9fa", width=45, height=15,
                                     relief=tk.SUNKEN, font=("Arial", 9))
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        tk.Button(self.root, text="🎯 ЗАПУСТИТЬ КЛАССИФИКАЦИЮ", 
                 command=self.classify,
                 bg="#3498db", fg="white",
                 font=("Arial", 12, "bold"),
                 width=30, height=2).pack(pady=15)
        

        result_frame = tk.LabelFrame(self.root, text="Результаты классификации", 
                                    padx=15, pady=15, font=("Arial", 11, "bold"))
        result_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        self.result_text = tk.Text(result_frame, height=12, width=60,
                                  font=("Consolas", 10), wrap=tk.WORD)
        scrollbar = tk.Scrollbar(result_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        

        self.status_label = tk.Label(self.root, text="Готов к работе", 
                                    relief=tk.SUNKEN, anchor=tk.W, padx=15,
                                    font=("Arial", 9), bg="#f1f2f6")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_model(self):

        print(f"Загрузка модели: {self.model_path}")
        print(f"Ожидаемые классы: {self.class_names}")
        
        if not os.path.exists(self.model_path):
            error_msg = f"Файл модели не найден:\n{os.path.abspath(self.model_path)}"
            self.status_label.config(text=error_msg, fg="red")
            messagebox.showerror("Ошибка", 
                f"Модель не найдена по пути:\n{self.model_path}\n\n"
                f"Убедитесь, что файл модели находится в папке 'model'")
            return
        

        self.status_label.config(text="Загрузка модели...", fg="orange")
        self.root.update()

        state_dict = torch.load(self.model_path, map_location='cpu')
        print(f"✅ Загружено {len(state_dict)} параметров модели")
        self.model = models.efficientnet_b0(pretrained=False)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, self.num_classes)
            

        self.model.load_state_dict(state_dict)
        self.model.eval()
            
        status_text = f"✅ Модель загружена | Классы: {', '.join(self.class_names)}"
        self.status_label.config(text=status_text, fg="green")
        print(f"✅ Модель успешно загружена: {self.num_classes} классов")
            
    
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.tiff")],
            title="Выберите изображение"
        )
        if path:
            self.current_file = path
            self.display_image(path)
            self.status_label.config(text=f"📸 Загружено изображение: {os.path.basename(path)}")
            self.clear_results()
    
    def load_binary(self):
        path = filedialog.askopenfilename(
            filetypes=[("Бинарные файлы", "*.bin *.hex *.rom *.img *.dat *.raw")],
            title="Выберите файл прошивки"
        )
        if path:
            self.current_file = path
            self.display_binary_preview(path)
            self.status_label.config(text=f"🔧 Загружен бинарный файл: {os.path.basename(path)}")
            self.clear_results()
    
    def display_image(self, path):
        try:
            img = Image.open(path)
            img.thumbnail((350, 350))
            photo = ImageTk.PhotoImage(img)
            
            self.preview_label.config(image=photo, text="")
            self.preview_label.image = photo
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")
    
    def display_binary_preview(self, path):
        try:
            with open(path, 'rb') as f:
                data = f.read()
            
            size = min(int(len(data) ** 0.5), 300)
            if size < 32:
                size = 128  
            

            target_size = min(size * size, len(data))
            img_data = data[:target_size]
            

            if len(img_data) < size * size:
                img_data = img_data + b'\x00' * (size * size - len(img_data))
            

            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img_array = img_array.reshape((size, size))
            

            if img_array.max() > img_array.min():
                img_array = ((img_array - img_array.min()) * 255.0 / 
                            (img_array.max() - img_array.min()))
            else:
                img_array = img_array * 255 // 256  
            
            img_array = img_array.astype(np.uint8)
            

            img = Image.fromarray(img_array).convert('RGB')
            img = img.resize((350, 350), Image.Resampling.NEAREST)
            photo = ImageTk.PhotoImage(img)
            
            self.preview_label.config(image=photo)
            self.preview_label.image = photo
            

            file_size = len(data)
            size_text = self.format_size(file_size)
            self.preview_label.config(text=f"Бинарный файл\n{size_text}\n{len(data):,} байт")
            
        except Exception as e:
            self.preview_label.config(
                image=None,
                text=f"Бинарный файл\n{os.path.basename(path)}\n\nНе удалось создать предпросмотр"
            )
    
    def format_size(self, size_bytes):

        for unit in ['Б', 'КБ', 'МБ', 'ГБ']:
            if size_bytes < 1024.0:
                return f"Размер: {size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"Размер: {size_bytes:.1f} ТБ"
    
    def clear_results(self):

        self.result_text.delete(1.0, tk.END)
    
    def classify(self):
        if not self.current_file:
            messagebox.showwarning("Внимание", "Сначала загрузите файл для классификации!")
            return
        
        if not self.model:
            messagebox.showerror("Ошибка", "Модель не загружена!")
            return
        
        try:
            self.status_label.config(text=" Выполняется классификация...", fg="orange")
            self.root.update()
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                   [0.229, 0.224, 0.225])
            ])
            

            file_ext = os.path.splitext(self.current_file)[1].lower()
            
            if file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:

                img = Image.open(self.current_file).convert('RGB')
                file_type = "изображение"
            else:

                with open(self.current_file, 'rb') as f:
                    data = f.read()

                size = min(int(len(data) ** 0.5), 224)
                if size < 32:
                    size = 224
                
                target_size = min(size * size, len(data))
                img_data = data[:target_size]
                
                if len(img_data) < size * size:
                    img_data = img_data + b'\x00' * (size * size - len(img_data))
                
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                img_array = img_array.reshape((size, size))

                if img_array.max() > img_array.min():
                    img_array = ((img_array - img_array.min()) * 255.0 / 
                                (img_array.max() - img_array.min()))
                img_array = img_array.astype(np.uint8)
                
                img = Image.fromarray(img_array).convert('RGB')
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                file_type = "бинарный файл прошивки"
            
            img_tensor = transform(img).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probs, self.num_classes)
            
            self.result_text.delete(1.0, tk.END)
            
            results_text = "═" * 60 + "\n"
            results_text += "          РЕЗУЛЬТАТЫ АНАЛИЗА ПРОШИВКИ\n"
            results_text += "═" * 60 + "\n\n"
            
            results_text += f" Анализируемый файл: {os.path.basename(self.current_file)}\n"
            results_text += f" Тип данных: {file_type}\n"
            results_text += f" Модель: EfficientNet-B0\n"
            results_text += "─" * 60 + "\n\n"
            
            sorted_indices = torch.argsort(probs, dim=1, descending=True)[0]
            
            for rank, idx in enumerate(sorted_indices):
                idx = idx.item()
                prob = probs[0][idx].item() * 100
                class_name = self.class_names[idx]
                

                if class_name == "benignware":
                    icon = "✅"
                    color_code = "#27ae60" 
                    description = "Безопасная прошивка"
                elif class_name == "hackware":
                    icon = "⚠️"
                    color_code = "#f39c12"  
                    description = "Взломанная/модифицированная прошивка"
                else:  # malware
                    icon = "❌"
                    color_code = "#e74c3c" 
                    description = "Вредоносная прошивка"
                

                bar_length = int(prob / 5)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                
                if prob > 85:
                    confidence = "ОЧЕНЬ ВЫСОКАЯ"
                elif prob > 70:
                    confidence = "ВЫСОКАЯ"
                elif prob > 50:
                    confidence = "СРЕДНЯЯ"
                elif prob > 30:
                    confidence = "НИЗКАЯ"
                else:
                    confidence = "ОЧЕНЬ НИЗКАЯ"
                
                results_text += f"{icon} #{rank+1}: {class_name.upper()}\n"
                results_text += f"   📋 {description}\n"
                results_text += f"   📊 Вероятность: {prob:6.2f}%\n"
                results_text += f"   🎯 Уверенность: {confidence}\n"
                results_text += f"   {bar}\n\n"
        
            best_idx = sorted_indices[0].item()
            best_class = self.class_names[best_idx]
            best_prob = probs[0][best_idx].item() * 100
            
            results_text += "═" * 60 + "\n"
            results_text += "               ИТОГОВЫЙ ВЕРДИКТ\n"
            results_text += "═" * 60 + "\n\n"
            
            if best_class == "benignware":
                verdict = "✅ БЕЗОПАСНО"
                verdict_color = "#27ae60"
                recommendation = "Прошивка безопасна для использования."
            elif best_class == "hackware":
                verdict = "⚠️ ПОДОЗРИТЕЛЬНО"
                verdict_color = "#f39c12"
                recommendation = "Рекомендуется дополнительная проверка."
            else:  # malware
                verdict = "❌ ОПАСНО"
                verdict_color = "#e74c3c"
                recommendation = "НЕ ИСПОЛЬЗОВАТЬ! Прошивка содержит вредоносный код."
            
            results_text += f"Вердикт: {verdict}\n"
            results_text += f"Класс: {best_class.upper()} ({best_prob:.1f}% уверенности)\n"
            results_text += f"Рекомендация: {recommendation}\n"
            
            self.result_text.insert(1.0, results_text)
            
            self.highlight_text()
            
            status_text = f"✅ Анализ завершен: {best_class.upper()} ({best_prob:.1f}%)"
            self.status_label.config(text=status_text, fg=verdict_color)
            
        except Exception as e:
            error_msg = f"Ошибка классификации: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            self.status_label.config(text=error_msg, fg="red")
            messagebox.showerror("Ошибка анализа", error_msg)
    
    def highlight_text(self):
        self.result_text.tag_configure("benignware", foreground="#27ae60")
        self.result_text.tag_configure("hackware", foreground="#f39c12")
        self.result_text.tag_configure("malware", foreground="#e74c3c")
        self.result_text.tag_configure("safe", foreground="#27ae60", font=("Consolas", 10, "bold"))
        self.result_text.tag_configure("warning", foreground="#f39c12", font=("Consolas", 10, "bold"))
        self.result_text.tag_configure("danger", foreground="#e74c3c", font=("Consolas", 10, "bold"))
        

        content = self.result_text.get(1.0, tk.END)
        
        for class_name in self.class_names:
            start_idx = "1.0"
            while True:
                start_idx = self.result_text.search(class_name, start_idx, tk.END)
                if not start_idx:
                    break
                end_idx = f"{start_idx}+{len(class_name)}c"
                self.result_text.tag_add(class_name, start_idx, end_idx)
                start_idx = end_idx
        
        for tag, word in [("safe", "БЕЗОПАСНО"), ("warning", "ПОДОЗРИТЕЛЬНО"), ("danger", "ОПАСНО")]:
            start_idx = "1.0"
            while True:
                start_idx = self.result_text.search(word, start_idx, tk.END)
                if not start_idx:
                    break
                end_idx = f"{start_idx}+{len(word)}c"
                self.result_text.tag_add(tag, start_idx, end_idx)
                start_idx = end_idx

def main():
    root = tk.Tk()
    
    try:
        root.iconbitmap('icon.ico')
    except:
        pass
    
    app = FirmwareClassifierGUI(root)
    
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    root.minsize(550, 700)
    
    root.mainloop()

if __name__ == "__main__":
    main()