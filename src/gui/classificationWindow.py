import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import joblib
import csv
from matplotlib.backends import _backend_tk
_backend_tk.Show._mainloop = False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import re

class ClassificationWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.file_parser = master.file_parser
        self.master = master
        self.title("Enose – Классификация")
        self.geometry("600x500")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.active_classifier = master.active_classifier
        self.feature_names = master.feature_names
        self.target_column = master.target_name

        # Заголовок
        title_label = tk.Label(self, text="Классификация", font=("Arial", 16))
        title_label.pack(pady=(20, 10))

        buttons_frame = tk.Frame(self)
        buttons_frame.pack(fill="x", padx=10, pady=5)

        buttons_frame.grid_columnconfigure(0, weight=1)
        buttons_frame.grid_columnconfigure(1, weight=1)

        btn_save = tk.Button(buttons_frame, text="Сохранить классификатор", command=self.save_model)
        btn_save.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        btn_load = tk.Button(buttons_frame, text="Загрузить данные для классификации", command=self.load_data_for_classification)
        btn_load.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        # Контейнер для сменного содержимого
        self.content_wrapper = tk.Frame(self)
        self.content_wrapper.pack(fill="both", expand=True, padx=10, pady=10)

        # Панель информации о модели
        self.model_info_panel = tk.Text(self.content_wrapper, height=10, wrap="word", bg="#f0f0f0", bd=0, font=("Arial", 10))
        self.model_info_panel.tag_configure("bold", font=("Arial", 10, "bold"))
        self.model_info_panel.insert("end", "Информация о классификаторе:\n\n", "bold")
        self.model_info_panel.insert("end", "Классификатор:\n", "bold")
        self.model_info_panel.insert("end", f"{self.active_classifier}\n\n")
        self.model_info_panel.insert("end", "Столбцы признаков:\n", "bold")
        self.model_info_panel.insert("end", f"{', '.join(self.feature_names)}\n\n")
        self.model_info_panel.insert("end", "Целевой столбец:\n", "bold")
        self.model_info_panel.insert("end", f"{self.target_column}\n")
        self.model_info_panel.config(state="disabled")
        self.model_info_panel.pack(fill="both", expand=True)

        self.add_class_list()

        self.canvas_frame = tk.Frame(self.content_wrapper)
        
        self.result_text = tk.Text(
            self.canvas_frame,
            height=1,
            wrap="none",
            bg="#f0f0f0",
            bd=0,
            font=("Arial", 14),
            relief="flat",
        )
        self.result_text.tag_configure("bold", font=("Arial", 14, "bold"))
        self.result_text.tag_configure("center", justify="center")
        self.result_text.pack(fill="x", padx=10, pady=5)

        self.canvas = None
        self.canvas_frame.pack_forget()

        self.bottom_frame = tk.Frame(self)
        self.bottom_frame.pack(fill="x", side="bottom", padx=10, pady=10)

        back_btn = tk.Button(self.bottom_frame, text="Назад", command=self.go_back)
        back_btn.pack(side="left", anchor="w")

        if self.has_conflict_labels:
            self.after(100, lambda: messagebox.showwarning("Предупреждение", "Найдены похожие классы!"))
            # messagebox.showwarning("Предупреждение", "Были использованы обучающие данные с разных устройств.")

    def add_class_list(self):
        if hasattr(self, "class_list") and self.class_list.winfo_exists():
            self.class_list.destroy()

        style = ttk.Style()
        style.configure("Custom.Treeview", rowheight=24)
        style.layout("Custom.Treeview", [('Treeview.treearea', {'sticky': 'nswe'})])

        self.class_list = ttk.Treeview(
            self.content_wrapper,
            columns=("class",),
            show="headings",
            height=5,
            selectmode="none",
            style="Custom.Treeview"
        )
        self.class_list.heading("class", text="Классы в обучающей выборке")
        self.class_list.column("class", anchor="w")

        class_labels = list(getattr(self.active_classifier, "classes_", []))
        self.has_conflict_labels = False
        for i, label in enumerate(class_labels):
            if has_similar_label(label, class_labels):
                tag = "conflict"
                self.has_conflict_labels = True
            else:
                tag = "even" if i % 2 == 0 else "odd"
            self.class_list.insert("", "end", values=(label,), tags=(tag,))

        self.class_list.tag_configure("even", background="#f9f9f9")
        self.class_list.tag_configure("odd", background="#f0f0f0")
        self.class_list.tag_configure("conflict", background="#ffe5e5", foreground="red")

        self.class_list.pack(fill="x", padx=5, pady=(5, 0))


    def go_back(self):
        self.withdraw()
        self.master.deiconify()

    def on_close(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        self.go_back()
    
    def export_result(self):
        if not hasattr(self, "labels") or not hasattr(self, "percentages"):
            messagebox.showwarning("Предупреждение", "Нет данных для экспорта.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV файлы", "*.csv")],
            title="Сохранить результаты классификации"
        )
        if path:
            try:
                with open(path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f, delimiter=";")
                    writer.writerow(["газ", "результат"])
                    for label, perc in zip(self.labels, self.percentages):
                        writer.writerow([label, f"{perc:.2f}%"])
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{e}")

    def save_model(self):
        if hasattr(self.master, "classifier"):
            path = filedialog.asksaveasfilename(
                defaultextension=".joblib",
                filetypes=[("Файлы Joblib", "*.joblib")],
                title="Сохранить классификатор"
            )
            if path:
                joblib.dump({
                    "model": self.master.classifier.model,
                    "features": self.master.classifier.feature_names,
                    "target": self.master.classifier.target_name
                }, path)

    def load_data_for_classification(self):
        file_path = filedialog.askopenfilename(filetypes=[("Файлы CSV", "*.csv"), ("Все файлы", "*.*")])
        if not file_path:
            return

        try:
            df = self.file_parser.load_single_csv(file_path)
            clf = self.master.classifier

            majority_class, avg_proba, all_preds, all_probs = clf.classify_batch(df)

            # Удалить старую диаграмму (если есть)
            if self.canvas:
                self.canvas.get_tk_widget().destroy()

            # Круговая диаграмма
            class_labels = list(clf.model.classes_)
            percentages = avg_proba * 100

            # Убираем классы с 0%
            filtered = [(label, p) for label, p in zip(class_labels, percentages) if p > 0]
            if not filtered:
                raise ValueError("Все вероятности равны 0 — невозможно построить диаграмму.")

            filtered_labels, filtered_percentages = zip(*filtered)

            # Создаём круговую диаграмму
            fig, ax = plt.subplots(figsize=(5, 3))
            base_colors = list(plt.cm.Pastel1.colors)

            labels = list(filtered_labels)
            percentages = list(filtered_percentages)

            self.labels = labels
            self.percentages = percentages

            # Если классов больше 10 → объединяем остальные в "Другое"
            if len(labels) > 10:
                main_labels = labels[:9]
                main_percentages = percentages[:9]

                other_percentage = sum(percentages[9:])
                if other_percentage > 0:
                    main_labels.append("Другое")
                    main_percentages.append(other_percentage)

                labels = main_labels
                percentages = main_percentages

            # Подбираем цвета
            colors = base_colors[:len(labels)]
            if "Другое" in labels:
                colors.append("#F7EAA0")

            # Если много классов (более 2), используем легенду
            if len(labels) > 2:
                wedges, _ = ax.pie(
                    percentages,
                    startangle=90,
                    colors=colors,
                    textprops={"fontsize": 10}
                )
                ax.legend(wedges, labels, title="Классы", loc="center left", bbox_to_anchor=(1.0, 0.5))
            else:
                ax.pie(
                    percentages,
                    labels=labels,
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=colors,
                    textprops={"fontsize": 10}
                )

            ax.axis("equal")
            fig.tight_layout()

            self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack()
            self.restore_icon()

            # Обновляем текст
            self.result_text.config(state="normal")
            self.result_text.delete("1.0", tk.END)
            self.result_text.tag_configure("center", justify='center')
            self.result_text.tag_configure("bold", font=("Arial", 14, "bold"))

            self.result_text.insert("1.0", f"Результат: {majority_class} ({max(filtered_percentages):1.1f}%)", "bold")
            self.result_text.tag_add("center", "1.0", "end")
            self.result_text.config(state="disabled")

            # Скрываем info, показываем результат
            self.model_info_panel.pack_forget()
            self.class_list.pack_forget()
            self.canvas_frame.pack(fill="both", expand=True)

            # Добавляем кнопку экспорта
            if not hasattr(self, "export_btn") or not self.export_btn.winfo_exists():
                self.export_btn = tk.Button(
                    self.bottom_frame,
                    text="Экспортировать результаты",
                    command=self.export_result
                )
                self.export_btn.pack(side="right", anchor="e")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при классификации:\n{e}")

    def restore_icon(self):
        if sys.platform == "darwin":
            self.master.set_icon()

def has_similar_label(label, class_labels):
    """
    Проверяет, есть ли среди class_labels другие элементы с таким же префиксом,
    как у label, но с числовыми суффиксами.
    Пример: label='air_001', class_labels=['air_001', 'air_002', 'ethanol'] → True
    """
    match = re.match(r"^(.*)_\d+$", label)
    if not match:
        return False

    prefix = match.group(1)
    similar = [
        other for other in class_labels
        if other != label and re.fullmatch(f"{re.escape(prefix)}_\\d+", other)
    ]

    return len(similar) > 0
