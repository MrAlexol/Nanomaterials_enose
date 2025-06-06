import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from gui.browser import FileListPanel
import joblib

from gui.classificationWindow import ClassificationWindow
from domain.calc import Classifier
from infrastructure.fileparser import FileParser


class MainGui(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.train_data = None
        self.file_parser = FileParser()

        self.active_classifier = None
        self.feature_names = []
        self.target_name = None
        
        self.classifier = Classifier()
        self.loaded_files = []
        self.title("Enose")
        self.geometry("400x600")

        self.csv_delimiter = tk.StringVar(value=',')
        self.csv_decimal = tk.StringVar(value='.')

        self.tabs = {}
        self.tab_buttons = {}
        self.current_tab = None

        main_frame = tk.Frame(self)
        main_frame.pack(fill="both", expand=True)

        tab_buttons_frame = tk.Frame(main_frame, bg="#e0e0e0")
        tab_buttons_frame.pack(side="top", fill="x")

        bottom_border = tk.Frame(tab_buttons_frame, height=1, bg="gray")
        bottom_border.pack(side="bottom", fill="x")

        self.content_frame = tk.Frame(main_frame)
        self.content_frame.pack(side="top", fill="both", expand=True)

        def add_tab(name, callback):
            btn = tk.Button(
                tab_buttons_frame, 
                text=name, 
                anchor="center",
                bg="#e0e0e0",
                padx=10,
                pady=3,
                font=('Arial', 9),
                command=lambda: self.switch_tab(name)
            )
            btn.pack(side="left", fill="none", expand=False)
            self.tab_buttons[name] = btn
            callback()

        def create_main():
            frame = tk.Frame(self.content_frame)
            self.tabs["Главная"] = frame
            self.create_main_tab(frame)

        def create_settings():
            frame = tk.Frame(self.content_frame)
            self.tabs["Настройки"] = frame
            self.create_settings_tab(frame)

        add_tab("Главная", create_main)
        add_tab("Настройки", create_settings)

        self.switch_tab("Главная")
    
    def set_icon(self):
        icon_path = os.path.join(os.path.dirname(__file__), "..", "assets", "iconlogo.png")
        icon_path = os.path.abspath(icon_path)
        icon = tk.PhotoImage(file=icon_path)
        self.iconphoto(False, icon)

    def switch_tab(self, name):
        if self.current_tab:
            self.tabs[self.current_tab].pack_forget()
            self.tab_buttons[self.current_tab].configure(bg="#e0e0e0")

        self.current_tab = name
        self.tabs[name].pack(fill="both", expand=True)
        self.tab_buttons[name].configure(bg="#d0d0ff")

    def create_main_tab(self, frame):
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(3, weight=1)

        label_title = tk.Label(frame, text="Enose", font=("Arial", 24))
        label_title.grid(row=0, column=0, columnspan=2, pady=20)

        # Кнопки: загрузка классификатора, файлов, обучение
        buttons_frame = ttk.Frame(frame)
        buttons_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        buttons_frame.grid_columnconfigure(0, weight=1)
        buttons_frame.grid_columnconfigure(1, weight=1)

        # Загрузить классификатор (первая строка — на всю ширину)
        self.load_model_btn = ttk.Button(buttons_frame, text="Загрузить классификатор", command=self.load_classifier)
        self.load_model_btn.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))

        # Кнопка "Перейти к классификатору" (вторая строка — на всю ширину, скрыта по умолчанию)
        self.goto_classifier_btn = ttk.Button(buttons_frame, text="Вернуться к классификатору", command=self.open_classification_window)
        self.goto_classifier_btn.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.goto_classifier_btn.grid_remove()

        # Загрузить файлы — слева (третья строка)
        load_data_btn = ttk.Button(buttons_frame, text="Выбрать файлы для обучения", command=self.show_browser)
        load_data_btn.grid(row=2, column=0, sticky="ew", padx=(0, 5))

        # Начать обучение — справа
        self.train_btn = ttk.Button(buttons_frame, text="Обучить модель", command=self.train_model, state="disabled")
        self.train_btn.grid(row=2, column=1, sticky="ew")

        # Заголовок и список файлов
        self.descr_label = tk.Label(
            frame,
            text="📂 Загруженные файлы",
            font=("Arial", 12, "bold"),
            anchor="w",
            fg="#333333"
        )
        self.descr_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=10)

        # Контейнер для сменного контента
        self.content_wrapper = ttk.Frame(frame, height=180)
        self.content_wrapper.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 5))
        self.content_wrapper.grid_propagate(False)

        # Панель со списком файлов
        self.file_list_panel = FileListPanel(self.content_wrapper, files_owner=self)
        self.file_list_panel.pack(fill="both", expand=True)

        # Панель информации о модели (скрыта)
        self.model_info_panel = tk.Text(self.content_wrapper, height=10, wrap="word", bg="#f0f0f0", bd=0, font=("Arial", 10))
        self.model_info_panel.tag_configure("bold", font=("Arial", 10, "bold"))
        self.model_info_panel.config(state="disabled")
        self.model_info_panel.pack_forget()

        # Подпись снизу
        label_footer = tk.Label(frame, text="© Лаборатория наноматериалов, 2025", font=("Arial", 10))
        label_footer.grid(row=4, column=0, columnspan=2, sticky="s", pady=10)


    def update_classifier_buttons(self):
        if self.active_classifier:
            self.goto_classifier_btn.grid()
        else:
            self.goto_classifier_btn.grid_remove()

    def show_browser(self):
        self.descr_label.config(text="📂 Загруженные файлы")
        self.model_info_panel.pack_forget()
        self.file_list_panel.pack(fill="both", expand=True)
        self.file_list_panel.add_file()
    
    def set_selected_files(self, selected_files):
        self.loaded_files = selected_files
        if len(selected_files) > 0:
            self.train_btn.config(state="normal")
        else:
            self.train_btn.config(state="disabled")

    def load_classifier(self):
        path = filedialog.askopenfilename(
            filetypes=[("Файлы Joblib", "*.joblib"), ("Все файлы", "*.*")],
            title="Загрузить классификатор"
        )
        if path:
            try:
                data = joblib.load(path)
                self.active_classifier = data["model"]
                self.feature_names = data.get("features", [])
                self.target_name = data.get("target", "Неизвестно")

                self.classifier.set_model(self.active_classifier, self.feature_names, self.target_name)

                self.file_list_panel.pack_forget()
                self.model_info_panel.pack(fill="both", expand=True)
                self.descr_label.config(text="✅ Модель загружена")
                self.model_info_panel.config(state="normal")
                self.model_info_panel.delete("1.0", "end")

                self.model_info_panel.insert("end", "Информация о классификаторе:\n\n", "bold")
                self.model_info_panel.insert("end", "Классификатор:\n", "bold")
                self.model_info_panel.insert("end", f"{self.active_classifier}\n\n")

                self.model_info_panel.insert("end", "Столбцы признаков:\n", "bold")
                self.model_info_panel.insert("end", f"{', '.join(self.feature_names)}\n\n")

                self.model_info_panel.insert("end", "Целевой столбец:\n", "bold")
                self.model_info_panel.insert("end", f"{self.target_name}\n")

                self.model_info_panel.config(state="disabled")

                self.update_classifier_buttons()
                self.open_classification_window()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{e}")


    def create_settings_tab(self, frame):
        frame.grid_rowconfigure(99, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        label_title = tk.Label(frame, text="Enose", font=("Arial", 24))
        label_title.grid(row=0, column=0, pady=20)

        form_frame = tk.Frame(frame)
        form_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nw")
        form_frame.grid_columnconfigure(1, weight=1)

        def on_change(_=None):
            apply_btn.config(state="normal")

        # Разделитель CSV
        delimiter_label = tk.Label(form_frame, text="Разделитель CSV:")
        delimiter_label.grid(row=0, column=0, sticky="w", padx=(0, 10), pady=5)
        delimiter_options = [";", ","]
        delimiter_selector = ttk.Combobox(
            form_frame,
            values=delimiter_options,
            textvariable=self.csv_delimiter,
            state="readonly",
            width=17,
            takefocus=0
        )
        delimiter_selector.grid(row=0, column=1, sticky="w", pady=5)
        delimiter_selector.bind("<<ComboboxSelected>>", on_change)

        # Десятичный знак
        decimal_label = tk.Label(form_frame, text="Десятичный знак:")
        decimal_label.grid(row=1, column=0, sticky="w", padx=(0, 10), pady=5)
        decimal_options = [",", "."]
        decimal_selector = ttk.Combobox(
            form_frame,
            values=decimal_options,
            textvariable=self.csv_decimal,
            state="readonly",
            width=17,
            takefocus=0
        )
        decimal_selector.grid(row=1, column=1, sticky="w", pady=5)
        decimal_selector.bind("<<ComboboxSelected>>", on_change)

        # Модель
        model_label = tk.Label(form_frame, text="Используемая модель:")
        model_label.grid(row=2, column=0, sticky="w", padx=(0, 10), pady=5)
        self.selected_model = tk.StringVar(value="KNN")
        model_options = ["KNN", "SVM", "Decision Tree", "Logistic Regression"]
        model_selector = ttk.Combobox(
            form_frame,
            values=model_options,
            textvariable=self.selected_model,
            state="readonly",
            width=17,
            takefocus=0
        )
        model_selector.grid(row=2, column=1, sticky="w", pady=5)
        model_selector.bind("<<ComboboxSelected>>", on_change)

        # Признаки
        self.feature_columns = tk.StringVar(value="Датчик1,Датчик2,Датчик3,Датчик4,Датчик5,Датчик6,Датчик7,Датчик8,Влажность,Температура")
        self.target_column = tk.StringVar(value="Материал")

        features_label = tk.Label(form_frame, text="Столбцы признаков:")
        features_label.grid(row=3, column=0, sticky="nw", padx=(0, 10), pady=5)

        features_text = tk.Text(form_frame, height=4, width=30, wrap="word")
        features_text.insert("1.0", self.feature_columns.get())
        features_text.grid(row=3, column=1, sticky="w", pady=5)
        features_text.bind("<KeyRelease>", on_change)

        target_label = tk.Label(form_frame, text="Целевой столбец:")
        target_label.grid(row=4, column=0, sticky="w", padx=(0, 10), pady=5)
        target_entry = tk.Entry(form_frame, textvariable=self.target_column, width=25)
        target_entry.grid(row=4, column=1, sticky="w", pady=5)
        target_entry.bind("<KeyRelease>", on_change)

        # Кнопка "Применить"
        def apply_settings():
            features = features_text.get("1.0", "end").strip()
            target = self.target_column.get().strip()

            if not features:
                messagebox.showwarning("Предупреждение", "Поле признаков не может быть пустым.")
                return
            if not target:
                messagebox.showwarning("Предупреждение", "Целевой столбец не может быть пустым.")
                return

            feature_list = [col.strip() for col in features.split(',')]
            if not all(feature_list):
                messagebox.showwarning("Предупреждение", "Некорректный формат в поле признаков.")
                return

            self.feature_columns.set(','.join(feature_list))
            self.target_column.set(target)

            # messagebox.showinfo("Успех", "Настройки успешно применены.")
            apply_btn.config(state="disabled")

        apply_btn = ttk.Button(form_frame, text="Применить", command=apply_settings, state="disabled")
        apply_btn.grid(row=5, column=1, sticky="e", pady=(10, 0))

        label_footer = tk.Label(frame, text="© Лаборатория наноматериалов, 2025", font=("Arial", 10))
        label_footer.grid(row=99, column=0, sticky="s", pady=10)

    def train_model(self):
        try:
            self.file_parser.set_file_params(
                sep=self.csv_delimiter.get(),
                decimal=self.csv_decimal.get()
            )
            self.file_parser.set_columns(
                self.feature_columns.get().split(','), self.target_column.get()
            )
            self.train_data = self.file_parser.load_multiple_csvs(self.loaded_files)

            model_name = self.selected_model.get()
            if model_name == "KNN":
                self.train_with("knn")
            elif model_name == "SVM":
                self.train_with("svm")
            elif model_name == "Decision Tree":
                self.train_with("decision_tree")
            elif model_name == "Logistic Regression":
                self.train_with("logistic_regression")
            else:
                messagebox.showwarning("Модель", f"Неизвестная модель: {model_name}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файлы:\n{e}")


    def train_with(self, model_type):
        if self.train_data is None:
            messagebox.showerror("Ошибка", "Не заданы данные для обучения.")
            return

        try:
            feature_columns = [col.strip() for col in self.feature_columns.get().split(',')]
            target_column = self.target_column.get().strip()

            if model_type == "knn":
                model = self.classifier.train_knn(self.train_data, feature_columns, target_column)
            elif model_type == "svm":
                model = self.classifier.train_svm(self.train_data, feature_columns, target_column)
            elif model_type == "decision_tree":
                model = self.classifier.train_decision_tree(self.train_data, feature_columns, target_column)
            elif model_type == "logistic_regression":
                model = self.classifier.train_logistic_regression(self.train_data, feature_columns, target_column)
            else:
                raise ValueError(f"Неподдерживаемый тип модели: {model_type}")

            self.active_classifier = model
            self.feature_names = self.classifier.feature_names
            self.target_name = self.classifier.target_name

            self.open_classification_window()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось обучить модель:\n{str(e)}")
    
    def open_classification_window(self):
        self.file_parser.set_file_params(
            sep=self.csv_delimiter.get(),
            decimal=self.csv_decimal.get()
        )
        self.file_parser.set_columns(
                self.feature_columns.get().split(','), self.target_column.get()
        )
        self.withdraw()
        self.classification_window = ClassificationWindow(self)


if __name__ == "__main__":
    app = MainGui()
    app.mainloop()
