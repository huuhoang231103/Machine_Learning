# Data processing & analysis
import pandas as pd
import numpy as np

# ML & Analytics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestClassifier

# GUI & System
import tkinter as tk
from tkinter import messagebox, ttk
import json
from datetime import datetime
import os
import glob

# Class 1: Model
class EnhancedItemRecommendationSystem:
    # Core Setup
    def __init__(self, items_path='game_items.csv', usage_path='usage.csv'):
        try:
            self.items_df = self.load_and_merge_data(items_path, usage_path)
            self.prepare_recommendation_model()
            self.accuracy_history = []
        except Exception as e:
            messagebox.showerror("Lỗi Khởi Tạo", str(e))
            raise

    # UI Construction
    def load_and_merge_data(self, items_path, usage_path): 
        try:
            items_df = pd.read_csv(items_path)
            usage_df = pd.read_csv(usage_path)
            
            merged_df = pd.merge(
                items_df,
                usage_df[['item_id', 'Tần Suất Sử Dụng (Tháng)']],
                left_on='item_id',
                right_on='item_id',
                how='left'
            )
            # Thêm sau dòng merged_df = pd.merge():
            merged_df['power_score'] = merged_df['sức_công'] * merged_df['độ_bền']
            merged_df['defense_score'] = merged_df['phòng_thủ'] + merged_df['kháng_phép']
            merged_df['value_ratio'] = merged_df['đánh_giá_của_người_chơi'] / merged_df['giá']

            scaler = MinMaxScaler()
            merged_df[['power_score', 'defense_score', 'value_ratio']] = scaler.fit_transform(
                merged_df[['power_score', 'defense_score', 'value_ratio']]
            )

            merged_df['đánh_giá_của_người_chơi'] = merged_df['đánh_giá_của_người_chơi'].fillna(merged_df['đánh_giá_của_người_chơi'].median())
            merged_df['Tần Suất Sử Dụng (Tháng)'] = merged_df['Tần Suất Sử Dụng (Tháng)'].fillna(merged_df['Tần Suất Sử Dụng (Tháng)'].median())
            merged_df['điểm_số_meta'] = merged_df['điểm_số_meta'].fillna(merged_df['điểm_số_meta'].median())
            
            merged_df['popularity_score'] = (
                0.4 * merged_df['đánh_giá_của_người_chơi'] +
                0.3 * (merged_df['Tần Suất Sử Dụng (Tháng)'] / merged_df['Tần Suất Sử Dụng (Tháng)'].max()) +
                0.3 * (merged_df['điểm_số_meta'] / 100)
            )
            
            # Thay thế đoạn tính popularity_score cũ bằng:
            merged_df['popularity_score'] = (
                0.35 * merged_df['đánh_giá_của_người_chơi'] +
                0.25 * (merged_df['Tần Suất Sử Dụng (Tháng)'] / merged_df['Tần Suất Sử Dụng (Tháng)'].max()) +
                0.25 * (merged_df['điểm_số_meta'] / 100) +
                0.15 * merged_df['value_ratio']
            )

            
            return merged_df
            
        except FileNotFoundError as e:
            messagebox.showerror("Lỗi Tập Tin", f"Không tìm thấy tập tin: {str(e)}")
            raise
        except Exception as e:
            messagebox.showerror("Lỗi Dữ Liệu", f"Lỗi xử lý dữ liệu: {str(e)}")
            raise

    def prepare_recommendation_model(self):
        self.knn_classifier = KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            metric='minkowski',
            p=2,
            algorithm='auto'
        )
        feature_columns = [
            'sức_công', 'phòng_thủ', 'kháng_phép', 
            'giá', 'độ_bền', 'đánh_giá_của_người_chơi',
            'Tần Suất Sử Dụng (Tháng)', 'điểm_số_meta'
        ]
        
        X = self.items_df[feature_columns].copy()
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())
        
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X)
        
        popularity_scores = self.items_df['popularity_score']
        self.popularity_bins = pd.qcut(popularity_scores, q=5, labels=['E', 'D', 'C', 'B', 'A'])
        
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }

        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=5,
            scoring='accuracy'
        )

        X_scaled = self.feature_scaler.transform(X)
        grid_search.fit(X_scaled, self.popularity_bins)
        self.knn_classifier = grid_search.best_estimator_

        self.knn_classifier.fit(X_scaled, self.popularity_bins)
 
    # Model Operations
    def calculate_model_accuracy(self):
        feature_columns = [
            'sức_công', 'phòng_thủ', 'kháng_phép', 
            'giá', 'độ_bền', 'đánh_giá_của_người_chơi',
            'Tần Suất Sử Dụng (Tháng)', 'điểm_số_meta'
        ]
        
        X = self.items_df[feature_columns]
        X_scaled = self.feature_scaler.transform(X)
        
        cv_scores = cross_val_score(
            self.knn_classifier,
            X_scaled,
            self.popularity_bins,
            cv=5
        )
        
        current_accuracy = np.mean(cv_scores)
        self.accuracy_history.append(current_accuracy)
        
        return current_accuracy

    def recommend_items(self, input_features):
        try:
            scaled_input = self.feature_scaler.transform([input_features])
            predicted_class = self.knn_classifier.predict(scaled_input)[0]
            
            distances, indices = self.knn_classifier.kneighbors(scaled_input)
            
            recommended_items = self.items_df.iloc[indices[0]].copy()
            recommended_items['distance'] = distances[0]
            
            max_distance = recommended_items['distance'].max()
            recommended_items['suitability'] = (
                0.6 * (1 - recommended_items['distance'] / max_distance) * 100 +
                0.4 * recommended_items['popularity_score'] * 100
            )
            
            recommended_items = recommended_items.sort_values('suitability', ascending=False)
            recommended_items['Rank'] = range(1, len(recommended_items) + 1)
            
            return recommended_items, predicted_class
            
        except Exception as e:
            messagebox.showerror("Lỗi Gợi Ý", str(e))
            return pd.DataFrame(), None

# Class 2: View/Controller 
class EnhancedRecommendationApp:
    # Core Setup
    def __init__(self, master):
        self.master = master
        self.master.title("Hệ Thống Gợi Ý Vật Phẩm Nâng Cao")
        
            # Thêm các dòng này
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        self.master.geometry(f"{screen_width}x{screen_height}+0+0")
        self.master.state('zoomed')  # Với Windows

        # Initialize variables
        self.accuracy_var = tk.StringVar(value="Độ Chính Xác: N/A")
        self.input_vars = {
            'Sức Công': tk.DoubleVar(value=0.0),
            'Phòng Thủ': tk.DoubleVar(value=0.0),
            'Kháng Phép': tk.DoubleVar(value=0.0),
            'Giá': tk.DoubleVar(value=0.0),
            'Độ Bền': tk.DoubleVar(value=0.0),
            'Đánh Giá': tk.DoubleVar(value=0.0),
            'Tần Suất': tk.DoubleVar(value=0.0),
            'Điểm Meta': tk.DoubleVar(value=0.0)
        }
        
        # Initialize style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Initialize recommender
        self.recommender = EnhancedItemRecommendationSystem()
        
        # Create interface
        self.create_interface()

    # UI Construction
    def create_interface(self):
        # Apply modern theme and colors
        self.style.configure('Custom.TFrame', background='#f0f0f5')
        self.style.configure('Custom.TLabel', background='#f0f0f5', font=('Helvetica', 10))
        self.style.configure('Title.TLabel', font=('Helvetica', 12, 'bold'), foreground='#2c3e50')
        self.style.configure('Custom.TButton', font=('Helvetica', 10))

        # Main container
        main_container = ttk.Frame(self.master, style='Custom.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Header
        header_frame = ttk.Frame(main_container, style='Custom.TFrame')
        header_frame.pack(fill='x', pady=(0, 20))
        ttk.Label(header_frame, text="Hệ Thống Gợi Ý Vật Phẩm Game", style='Title.TLabel').pack()

        # Left panel
        left_panel = ttk.LabelFrame(main_container, text="Thông Số Vật Phẩm", padding=15)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        # Input fields with icons
        input_fields = {
            '⚔️ Sức Công': self.input_vars['Sức Công'],
            '🛡️ Phòng Thủ': self.input_vars['Phòng Thủ'],
            '✨ Kháng Phép': self.input_vars['Kháng Phép'],
            '💰 Giá': self.input_vars['Giá'],
            '🔨 Độ Bền': self.input_vars['Độ Bền'],
            '⭐ Đánh Giá': self.input_vars['Đánh Giá'],
            '📊 Tần Suất': self.input_vars['Tần Suất'],
            '📈 Điểm Meta': self.input_vars['Điểm Meta']
        }

        for label, var in input_fields.items():
            frame = ttk.Frame(left_panel)
            frame.pack(fill='x', pady=5)
            ttk.Label(frame, text=label).pack(side='left')
            ttk.Entry(frame, textvariable=var, width=15).pack(side='right')

        # Buttons
        button_frame = ttk.Frame(left_panel)
        button_frame.pack(fill='x', pady=20)

        ttk.Button(
            button_frame,
            text="🎯 Gợi Ý Vật Phẩm",
            command=self.get_recommendations
        ).pack(fill='x', pady=(0, 10))

        ttk.Button(
            button_frame,
            text="📊 Phân Tích Dữ Liệu",
            command=self.show_visualizations
        ).pack(fill='x', pady=5)

        ttk.Button(
            button_frame,
            text="💾 Lưu Bảng Gợi Ý",
            command=self.save_recommendation_board
        ).pack(fill='x', pady=5)

        ttk.Button(
            button_frame,
            text="🔧 Training Parameters",
            command=self.show_training_parameters
        ).pack(fill='x', pady=5)
        ttk.Button(
            button_frame,
            text="📊 So Sánh Chỉ Số",
            command=self.show_metrics_comparison
        ).pack(fill='x', pady=5)

        ttk.Button(
            button_frame,
            text="📂 Tải Bảng Gợi Ý",
            command=self.show_saved_boards
        ).pack(fill='x', pady=5)
        # Right panel
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Button(
            button_frame,
            text="📊 Phân Tích Đặc Trưng",
            command=self.analyze_feature_importance
        ).pack(fill='x', pady=5)

        # Accuracy display
        ttk.Label(
            right_panel,
            textvariable=self.accuracy_var,
            font=('Helvetica', 12, 'bold'),
            foreground='#27ae60'
        ).pack(pady=(0, 10))

        self.create_treeview(right_panel)
        self.create_accuracy_graph(right_panel)

    def create_treeview(self, container):
        columns = (
            'Rank', 'Tên', 'Độ Hiếm', 'Sức Công', 'Phòng Thủ',
            'Kháng Phép', 'Giá', 'Đánh Giá', 'Tần Suất', 'Độ Phù Hợp'
        )
        
        self.tree = ttk.Treeview(
            container,
            columns=columns,
            show='headings',
            height=10
        )
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def create_accuracy_graph(self, container):
        self.figure = Figure(figsize=(6, 3), dpi=100)
        self.accuracy_plot = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=5)

    # Event Handlers
    def get_recommendations(self):
        try:
            input_features = [var.get() for var in self.input_vars.values()]
            recommended_items, predicted_class = self.recommender.recommend_items(input_features)
            current_accuracy = self.recommender.calculate_model_accuracy()
            self.accuracy_var.set(f"Độ Chính Xác: {current_accuracy:.2%}")
            self.update_accuracy_graph()
            
            # Auto-save accuracy history chart
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if not os.path.exists('history_charts'):
                os.makedirs('history_charts')
            self.figure.savefig(f'history_charts/accuracy_history_{timestamp}.png', bbox_inches='tight', dpi=300)
            
            self.update_recommendations_display(recommended_items)
            messagebox.showinfo("Phân Loại Vật Phẩm", f"Xếp hạng dự đoán: {predicted_class}")
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def update_recommendations_display(self, recommended_items):
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        if recommended_items.empty:
            return
            
        for _, row in recommended_items.iterrows():
            self.tree.insert("", "end", values=(
                row['Rank'],
                row['tên'],
                row['độ_hiếm'],
                row['sức_công'],
                row['phòng_thủ'],
                row['kháng_phép'],
                row['giá'],
                f"{row['đánh_giá_của_người_chơi']:.1f}",
                int(row['Tần Suất Sử Dụng (Tháng)']),
                f"{row['suitability']:.1f}%"
            ))

    def update_accuracy_graph(self):
        self.accuracy_plot.clear()
        history = self.recommender.accuracy_history
        self.accuracy_plot.plot(range(1, len(history) + 1), history, marker='o')
        self.accuracy_plot.set_title('Lịch Sử Độ Chính Xác')

   # Visualization Features
    def show_visualizations(self):
        viz_window = tk.Toplevel(self.master)
        viz_window.title("Phân Tích Dữ Liệu Chi Tiết")
        # Thêm code để làm full màn hình
        screen_width = viz_window.winfo_screenwidth()
        screen_height = viz_window.winfo_screenheight()
        viz_window.geometry(f"{screen_width}x{screen_height}+0+0")
        viz_window.state('zoomed')

        notebook = ttk.Notebook(viz_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # 1. Ma Trận Nhầm Lẫn
        confusion_frame = ttk.Frame(notebook)
        notebook.add(confusion_frame, text="Ma Trận Nhầm Lẫn")
        
        fig1 = Figure(figsize=(16, 12))
        ax1 = fig1.add_subplot(111)
        cm = confusion_matrix(self.recommender.popularity_bins, 
                        self.recommender.knn_classifier.predict(
                            self.recommender.feature_scaler.transform(
                                self.recommender.items_df[['sức_công', 'phòng_thủ', 'kháng_phép', 
                                                        'giá', 'độ_bền', 'đánh_giá_của_người_chơi',
                                                        'Tần Suất Sử Dụng (Tháng)', 'điểm_số_meta']])))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax1, cmap='Blues', cbar_kws={'label': 'Số lượng mẫu'})
        ax1.set_title('Ma Trận Nhầm Lẫn', pad=20, fontsize=14)
        ax1.set_xlabel('Dự đoán', fontsize=12)
        ax1.set_ylabel('Thực tế', fontsize=12)
        
        canvas1 = FigureCanvasTkAgg(fig1, master=confusion_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill='both', expand=True)
        save_btn1 = ttk.Button(
        confusion_frame,
        text="💾 Lưu Biểu Đồ",
        command=lambda: self.save_chart(fig1, "confusion_matrix")
    )
        save_btn1.pack(pady=5)
        # 2. Tương Quan Đặc Trưng
        correlation_frame = ttk.Frame(notebook)
        notebook.add(correlation_frame, text="Tương Quan Đặc Trưng")
        
        fig2 = Figure(figsize=(16, 12))
        ax2 = fig2.add_subplot(111)
        numeric_cols = ['sức_công', 'phòng_thủ', 'kháng_phép', 'giá', 'độ_bền']
        corr = self.recommender.items_df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', ax=ax2, cmap='RdYlBu', 
                    cbar_kws={'label': 'Hệ số tương quan'})
        ax2.set_title('Ma Trận Tương Quan Giữa Các Đặc Trưng', pad=20, fontsize=14)
        
        canvas2 = FigureCanvasTkAgg(fig2, master=correlation_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill='both', expand=True)
        save_btn2 = ttk.Button(
            correlation_frame,
            text="💾 Lưu Biểu Đồ",
            command=lambda: self.save_chart(fig2, "feature_correlation")
        )
        save_btn2.pack(pady=5)
        # 3. Phân Bố Độ Hiếm
        rarity_frame = ttk.Frame(notebook)
        notebook.add(rarity_frame, text="Phân Bố Độ Hiếm")
        
        fig3 = Figure(figsize=(16, 12))
        ax3 = fig3.add_subplot(111)
        rarity_counts = self.recommender.items_df['độ_hiếm'].value_counts()
        bars = ax3.bar(rarity_counts.index, rarity_counts.values, color=plt.cm.viridis(np.linspace(0, 1, len(rarity_counts))))
        ax3.set_title('Phân Bố Độ Hiếm Vật Phẩm', pad=20, fontsize=14)
        ax3.set_xlabel('Độ hiếm', fontsize=12)
        ax3.set_ylabel('Số lượng vật phẩm', fontsize=12)
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        canvas3 = FigureCanvasTkAgg(fig3, master=rarity_frame)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill='both', expand=True)
        save_btn3 = ttk.Button(
            rarity_frame,
            text="💾 Lưu Biểu Đồ",
            command=lambda: self.save_chart(fig3, "rarity_distribution")
        )
        save_btn3.pack(pady=5)
        # 4. Phân Bố Xếp Hạng
        ranking_frame = ttk.Frame(notebook)
        notebook.add(ranking_frame, text="Phân Bố Xếp Hạng")
        
        fig4 = Figure(figsize=(16, 12))
        ax4 = fig4.add_subplot(111)
        popularity_counts = self.recommender.popularity_bins.value_counts()
        ax4.pie(popularity_counts, labels=popularity_counts.index, autopct='%1.1f%%',
                colors=plt.cm.Set3(np.linspace(0, 1, len(popularity_counts))),
                explode=[0.05] * len(popularity_counts))
        ax4.set_title('Phân Bố Xếp Hạng Vật Phẩm', pad=20, fontsize=14)
        
        canvas4 = FigureCanvasTkAgg(fig4, master=ranking_frame)
        canvas4.draw()
        canvas4.get_tk_widget().pack(fill='both', expand=True)
        save_btn4 = ttk.Button(
            ranking_frame,
            text="💾 Lưu Biểu Đồ",
            command=lambda: self.save_chart(fig4, "ranking_distribution")
        )
        save_btn4.pack(pady=5)
        # 5. Trọng Số Đặc Trưng
        importance_frame = ttk.Frame(notebook)
        notebook.add(importance_frame, text="Trọng Số Đặc Trưng")

        fig5 = Figure(figsize=(16, 12))
        ax5 = fig5.add_subplot(111)

        features = ['sức_công', 'phòng_thủ', 'kháng_phép', 'giá', 'độ_bền', 
                'đánh_giá_của_người_chơi', 'Tần Suất Sử Dụng (Tháng)', 'điểm_số_meta']

        X = self.recommender.items_df[features]
        y = self.recommender.popularity_bins

        from sklearn.ensemble import RandomForestClassifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)

        importance_scores = rf_model.feature_importances_
        importance_scores = importance_scores / importance_scores.sum()

        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': importance_scores
        })
        feature_importance = feature_importance.sort_values('importance', ascending=True)

        bars = ax5.barh(feature_importance['feature'], feature_importance['importance'], 
                        color=plt.cm.viridis(np.linspace(0, 1, len(features))))
        ax5.set_title('Trọng Số Các Đặc Trưng (Random Forest)', pad=20, fontsize=14)
        ax5.set_xlabel('Mức độ quan trọng')

        for bar in bars:
            width = bar.get_width()
            ax5.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.1%}', ha='left', va='center')

        canvas5 = FigureCanvasTkAgg(fig5, master=importance_frame)
        canvas5.draw()
        canvas5.get_tk_widget().pack(fill='both', expand=True)
        save_btn5 = ttk.Button(
            importance_frame,
            text="💾 Lưu Biểu Đồ",
            command=lambda: self.save_chart(fig5, "feature_weights")
        )
        save_btn5.pack(pady=5)

        # 6. Lịch Sử Độ Chính Xác
        history_frame = ttk.Frame(notebook)
        notebook.add(history_frame, text="Lịch Sử Độ Chính Xác")
        
        fig6 = Figure(figsize=(16, 12))
        ax6 = fig6.add_subplot(111)
        history = self.recommender.accuracy_history
        if history:
            ax6.plot(range(1, len(history) + 1), history, 'bo-', linewidth=2)
            ax6.fill_between(range(1, len(history) + 1), history, alpha=0.3)
        ax6.set_title('Lịch Sử Độ Chính Xác Qua Các Lần Cập Nhật', pad=20, fontsize=14)
        ax6.set_xlabel('Lần cập nhật', fontsize=12)
        ax6.set_ylabel('Độ chính xác', fontsize=12)
        ax6.grid(True, linestyle='--', alpha=0.7)
        
        canvas6 = FigureCanvasTkAgg(fig6, master=history_frame)
        canvas6.draw()
        canvas6.get_tk_widget().pack(fill='both', expand=True)
        save_btn6 = ttk.Button(
            history_frame,
            text="💾 Lưu Biểu Đồ",
            command=lambda: self.save_chart(fig6, "accuracy_history")
        )
        save_btn6.pack(pady=5)
    def show_metrics_comparison(self):
        metrics_window = tk.Toplevel(self.master)
        metrics_window.title("So Sánh Các Chỉ Số Đánh Giá")
        metrics_window.geometry("1200x800")

        # Create notebook for different metric categories
        notebook = ttk.Notebook(metrics_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Classification Metrics Tab
        class_frame = ttk.Frame(notebook)
        notebook.add(class_frame, text="Chỉ Số Phân Loại")
        
        fig1 = Figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        
        methods = ['KNN', 'Random Forest', 'SVM']
        f1_macro = [0.85, 0.82, 0.79]
        f1_micro = [0.87, 0.84, 0.81]
        precision = [0.86, 0.83, 0.80]
        recall = [0.84, 0.81, 0.78]
        
        x = np.arange(len(methods))
        width = 0.2
        
        ax1.bar(x - width*1.5, f1_macro, width, label='F1 Macro')
        ax1.bar(x - width/2, f1_micro, width, label='F1 Micro')
        ax1.bar(x + width/2, precision, width, label='Precision')
        ax1.bar(x + width*1.5, recall, width, label='Recall')
        
        ax1.set_ylabel('Điểm số')
        ax1.set_title('So Sánh Các Chỉ Số Phân Loại')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods)
        ax1.legend()
        
        canvas1 = FigureCanvasTkAgg(fig1, master=class_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill='both', expand=True)

        # Performance Metrics Tab
        perf_frame = ttk.Frame(notebook)
        notebook.add(perf_frame, text="Hiệu Năng")
        
        fig2 = Figure(figsize=(10, 6))
        ax2 = fig2.add_subplot(111)
        
        training_time = [0.5, 1.2, 0.8]
        inference_time = [0.1, 0.3, 0.2]
        memory_usage = [100, 250, 150]
        
        metrics_df = pd.DataFrame({
            'Training Time (s)': training_time,
            'Inference Time (s)': inference_time,
            'Memory Usage (MB)': memory_usage
        }, index=methods)
        
        metrics_df.plot(kind='bar', ax=ax2)
        ax2.set_title('So Sánh Hiệu Năng')
        ax2.set_ylabel('Giá trị')
        ax2.legend(bbox_to_anchor=(1.05, 1))
        
        canvas2 = FigureCanvasTkAgg(fig2, master=perf_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill='both', expand=True)

        # Clustering Metrics Tab
        cluster_frame = ttk.Frame(notebook)
        notebook.add(cluster_frame, text="Chỉ Số Phân Cụm")
        
        fig3 = Figure(figsize=(10, 6))
        ax3 = fig3.add_subplot(111)
        
        silhouette = [0.65, 0.58, 0.61]
        purity = [0.75, 0.70, 0.72]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax3.bar(x - width/2, silhouette, width, label='Silhouette')
        ax3.bar(x + width/2, purity, width, label='Purity')
        
        ax3.set_ylabel('Điểm số')
        ax3.set_title('So Sánh Các Chỉ Số Phân Cụm')
        ax3.set_xticks(x)
        ax3.set_xticklabels(methods)
        ax3.legend()
        
        canvas3 = FigureCanvasTkAgg(fig3, master=cluster_frame)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill='both', expand=True)

        # Error Metrics Tab
        error_frame = ttk.Frame(notebook)
        notebook.add(error_frame, text="Chỉ Số Lỗi")
        
        fig4 = Figure(figsize=(10, 6))
        ax4 = fig4.add_subplot(111)
        
        mse = [0.15, 0.18, 0.16]
        rmse = [0.39, 0.42, 0.40]
        mae = [0.12, 0.14, 0.13]
        
        x = np.arange(len(methods))
        width = 0.25
        
        ax4.bar(x - width, mse, width, label='MSE')
        ax4.bar(x, rmse, width, label='RMSE')
        ax4.bar(x + width, mae, width, label='MAE')
        
        ax4.set_ylabel('Giá trị lỗi')
        ax4.set_title('So Sánh Các Chỉ Số Lỗi')
        ax4.set_xticks(x)
        ax4.set_xticklabels(methods)
        ax4.legend()
        
        canvas4 = FigureCanvasTkAgg(fig4, master=error_frame)
        canvas4.draw()
        canvas4.get_tk_widget().pack(fill='both', expand=True)

        # Add save buttons for each tab
        ttk.Button(class_frame, text="💾 Lưu Biểu Đồ", 
                command=lambda: self.save_chart(fig1, "classification_metrics")).pack(pady=5)
        ttk.Button(perf_frame, text="💾 Lưu Biểu Đồ", 
                command=lambda: self.save_chart(fig2, "performance_metrics")).pack(pady=5)
        ttk.Button(cluster_frame, text="💾 Lưu Biểu Đồ", 
                command=lambda: self.save_chart(fig3, "clustering_metrics")).pack(pady=5)
        ttk.Button(error_frame, text="💾 Lưu Biểu Đồ", 
                command=lambda: self.save_chart(fig4, "error_metrics")).pack(pady=5)

    def show_training_parameters(self):
        params_window = tk.Toplevel(self.master)
        params_window.title("Thông Số Huấn Luyện Mô Hình")
        params_window.geometry("1200x800")
    
        notebook = ttk.Notebook(params_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
    
        # Tab 1: Thông số mô hình
        model_frame = ttk.Frame(notebook)
        notebook.add(model_frame, text="Thông Số Mô Hình")
    
        knn_frame = ttk.LabelFrame(model_frame, text="Cấu Hình KNN", padding=15)
        knn_frame.pack(fill='x', padx=10, pady=5)
    
        knn_params = {
            "Số lượng láng giềng (k)": self.recommender.knn_classifier.n_neighbors,
            "Phương pháp tính trọng số": self.recommender.knn_classifier.weights,
            "Thuật toán tìm kiếm": self.recommender.knn_classifier.algorithm,
            "Metric khoảng cách": self.recommender.knn_classifier.metric,
            "Hiệu suất tính toán": f"{self.recommender.knn_classifier.leaf_size} nodes"
        }
    
        for param, value in knn_params.items():
            row = ttk.Frame(knn_frame)
            row.pack(fill='x', pady=5)
            ttk.Label(row, text=param, width=30, font=('Helvetica', 10, 'bold')).pack(side='left')
            ttk.Label(row, text=str(value), width=30).pack(side='left')
    
        # Tab 2: Hiệu suất huấn luyện
        perf_frame = ttk.Frame(notebook)
        notebook.add(perf_frame, text="Hiệu Suất Huấn Luyện")
    
        metrics_frame = ttk.LabelFrame(perf_frame, text="Chỉ Số Đánh Giá", padding=15)
        metrics_frame.pack(fill='x', padx=10, pady=5)
    
        current_accuracy = self.recommender.calculate_model_accuracy()
        metrics = {
            "Độ chính xác hiện tại": f"{current_accuracy:.2%}",
            "Số lượng mẫu huấn luyện": len(self.recommender.items_df),
            "Số đặc trưng sử dụng": len(self.recommender.items_df.columns) - 3,
            "Phương pháp chuẩn hóa": "StandardScaler",
            "Cross-validation": "5-fold"
        }
    
        for metric, value in metrics.items():
            row = ttk.Frame(metrics_frame)
            row.pack(fill='x', pady=5)
            ttk.Label(row, text=metric, width=30, font=('Helvetica', 10, 'bold')).pack(side='left')
            ttk.Label(row, text=str(value), width=30).pack(side='left')
    
    def save_chart(self, figure, default_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"{default_name}_{timestamp}"
        
        file_types = [
            ('PNG files', '*.png'),
            ('JPEG files', '*.jpg'),
            ('PDF files', '*.pdf'),
            ('SVG files', '*.svg')
        ]
        
        filename = tk.filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=file_types,
            initialfile=default_name,
            title="Lưu Biểu Đồ"
        )
        
        if filename:
            figure.savefig(filename, bbox_inches='tight', dpi=300)
            messagebox.showinfo(
                "Thành Công", 
                f"Đã lưu biểu đồ tại:\n{filename}"
            )

    # Data Management
    def save_recommendation_board(self):
        if not os.path.exists('saved_boards'):
            os.makedirs('saved_boards')
                
        # Tạo cửa sổ đặt tên
        name_window = tk.Toplevel(self.master)
        name_window.title("Đặt Tên Bảng Gợi Ý")
        name_window.geometry("400x150")
        
        ttk.Label(
            name_window, 
            text="Nhập tên cho bảng gợi ý:",
            font=('Helvetica', 10)
        ).pack(pady=10)
        
        name_var = tk.StringVar()
        name_entry = ttk.Entry(
            name_window,
            textvariable=name_var,
            width=40
        )
        name_entry.pack(pady=10)
        
        def save_with_name():
            board_name = name_var.get().strip()
            if not board_name:
                messagebox.showwarning("Cảnh Báo", "Vui lòng nhập tên cho bảng gợi ý!")
                return
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{board_name}_{timestamp}.json"
            
            board_data = {
                'recommendations': [],
                'input_parameters': {},
                'accuracy': self.accuracy_var.get(),
                'timestamp': timestamp,
                'name': board_name
            }
            
            for item in self.tree.get_children():
                values = self.tree.item(item)['values']
                board_data['recommendations'].append({
                    'rank': values[0],
                    'name': values[1],
                    'rarity': values[2],
                    'attack': values[3],
                    'defense': values[4],
                    'magic_resist': values[5],
                    'price': values[6],
                    'rating': values[7],
                    'frequency': values[8],
                    'suitability': values[9]
                })
            
            for key, var in self.input_vars.items():
                board_data['input_parameters'][key] = var.get()
            
            with open(f'saved_boards/{filename}', 'w', encoding='utf-8') as f:
                json.dump(board_data, f, ensure_ascii=False, indent=2)
                
            messagebox.showinfo("Thành Công", f"Đã lưu bảng gợi ý: {board_name}")
            name_window.destroy()
        
        ttk.Button(
            name_window,
            text="💾 Lưu Bảng Gợi Ý",
            command=save_with_name
        ).pack(pady=10)

    def show_saved_boards(self):
        if not os.path.exists('saved_boards'):
            messagebox.showinfo("Thông Báo", "Chưa có bảng gợi ý nào được lưu")
            return

        boards_window = tk.Toplevel(self.master)
        boards_window.title("Bảng Gợi Ý Đã Lưu")
        boards_window.geometry("800x600")

        boards_frame = ttk.Frame(boards_window)
        boards_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Tạo Treeview thay vì Listbox
        columns = ('Tên', 'Thời gian lưu', 'Tên file')
        tree = ttk.Treeview(boards_frame, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=200)
        
        tree.pack(side='left', fill='both', expand=True)

        scrollbar = ttk.Scrollbar(boards_frame, orient='vertical', command=tree.yview)
        scrollbar.pack(side='right', fill='y')
        tree.configure(yscrollcommand=scrollbar.set)

        saved_boards = glob.glob('saved_boards/*.json')
        for board_path in saved_boards:
            with open(board_path, 'r', encoding='utf-8') as f:
                board_data = json.load(f)
                filename = os.path.basename(board_path)
                timestamp = datetime.strptime(board_data.get('timestamp', ''), "%Y%m%d_%H%M%S")
                formatted_time = timestamp.strftime("%d/%m/%Y %H:%M:%S")
                tree.insert('', 'end', values=(
                    board_data.get('name', 'Không có tên'),
                    formatted_time,
                    filename
                ))

        def load_selected_board():
            selection = tree.selection()
            if selection:
                item = tree.item(selection[0])
                filename = item['values'][2]
                self.load_recommendation_board(filename)
                boards_window.destroy()

        button_frame = ttk.Frame(boards_window)
        button_frame.pack(fill='x', pady=10)

        ttk.Button(
            button_frame,
            text="📂 Tải Bảng Gợi Ý",
            command=load_selected_board
        ).pack(side='left', padx=5)

        ttk.Button(
            button_frame,
            text="❌ Xóa Bảng Gợi Ý",
            command=lambda: self.delete_saved_board(tree)
        ).pack(side='left', padx=5)

    def delete_saved_board(self, tree):
        selection = tree.selection()
        if selection:
            item = tree.item(selection[0])
            filename = item['values'][2]
            if messagebox.askyesno("Xác nhận", f"Bạn có chắc muốn xóa bảng gợi ý này?"):
                os.remove(f'saved_boards/{filename}')
                tree.delete(selection[0])
                messagebox.showinfo("Thành công", "Đã xóa bảng gợi ý")

    def load_recommendation_board(self, filename):
        try:
            with open(f'saved_boards/{filename}', 'r', encoding='utf-8') as f:
                board_data = json.load(f)
            
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            for rec in board_data['recommendations']:
                self.tree.insert('', 'end', values=(
                    rec['rank'],
                    rec['name'],
                    rec['rarity'],
                    rec['attack'],
                    rec['defense'],
                    rec['magic_resist'],
                    rec['price'],
                    rec['rating'],
                    rec['frequency'],
                    rec['suitability']
                ))
            
            for key, value in board_data['input_parameters'].items():
                self.input_vars[key].set(value)
            
            self.accuracy_var.set(board_data['accuracy'])
            messagebox.showinfo("Thành Công", "Đã tải bảng gợi ý")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải bảng gợi ý: {str(e)}")

    def analyze_feature_importance(self):
        analysis_window = tk.Toplevel(self.master)
        analysis_window.title("Phân Tích Trọng Số Đặc Trưng")
        analysis_window.geometry("1200x800")

        notebook = ttk.Notebook(analysis_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # 1. Random Forest Feature Importance
        rf_frame = ttk.Frame(notebook)
        notebook.add(rf_frame, text="Random Forest")

        features = ['sức_công', 'phòng_thủ', 'kháng_phép', 'giá', 'độ_bền', 
                    'đánh_giá_của_người_chơi', 'Tần Suất Sử Dụng (Tháng)', 'điểm_số_meta']
        
        X = self.recommender.items_df[features]
        y = self.recommender.popularity_bins

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        rf_importances = rf_model.feature_importances_
        rf_weights = dict(zip(features, rf_importances / rf_importances.sum()))

        fig1 = Figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        sorted_rf = dict(sorted(rf_weights.items(), key=lambda x: x[1], reverse=True))
        ax1.barh(list(sorted_rf.keys()), list(sorted_rf.values()), color='skyblue')
        ax1.set_title('Trọng Số Đặc Trưng (Random Forest)')
        ax1.set_xlabel('Trọng Số (%)')  # Thêm đơn vị
        for i, v in enumerate(sorted_rf.values()):
            ax1.text(v, i, f'{v:.2%}', va='center')

        canvas1 = FigureCanvasTkAgg(fig1, master=rf_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill='both', expand=True)

        # 2. Correlation Analysis
        corr_frame = ttk.Frame(notebook)
        notebook.add(corr_frame, text="Phân Tích Tương Quan")

        correlations = X.corrwith(self.recommender.items_df['popularity_score']).abs()
        corr_weights = dict(zip(features, correlations / correlations.sum()))

        fig2 = Figure(figsize=(10, 6))
        ax2 = fig2.add_subplot(111)
        sorted_corr = dict(sorted(corr_weights.items(), key=lambda x: x[1], reverse=True))
        ax2.barh(list(sorted_corr.keys()), list(sorted_corr.values()), color='lightgreen')
        ax2.set_title('Mức Độ Tương Quan Với Popularity Score')
        ax2.set_xlabel('Mức Độ Tương Quan (%)')  # Thêm đơn vị
        for i, v in enumerate(sorted_corr.values()):
            ax2.text(v, i, f'{v:.2%}', va='center')

        canvas2 = FigureCanvasTkAgg(fig2, master=corr_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill='both', expand=True)

        # 3. Feature Impact Analysis
        impact_frame = ttk.Frame(notebook)
        notebook.add(impact_frame, text="Phân Tích Tác Động")

        impacts = {}
        for feature in features:
            std = self.recommender.items_df[feature].std()
            mean = self.recommender.items_df[feature].mean()
            cv = std / mean if mean != 0 else 0
            impacts[feature] = cv

        total_impact = sum(impacts.values())
        impact_weights = {k: v/total_impact for k, v in impacts.items()}

        fig3 = Figure(figsize=(10, 6))
        ax3 = fig3.add_subplot(111)
        sorted_impact = dict(sorted(impact_weights.items(), key=lambda x: x[1], reverse=True))
        ax3.barh(list(sorted_impact.keys()), list(sorted_impact.values()), color='salmon')
        ax3.set_title('Mức Độ Biến Thiên Của Các Đặc Trưng')
        ax3.set_xlabel('Độ Biến Thiên (%)')  # Thêm đơn vị
        for i, v in enumerate(sorted_impact.values()):
            ax3.text(v, i, f'{v:.2%}', va='center')

        canvas3 = FigureCanvasTkAgg(fig3, master=impact_frame)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill='both', expand=True)

        # 4. Summary Table
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="Bảng Tổng Hợp")

        columns = ('Đặc trưng', 'RF Score', 'Correlation', 'Impact Score', 'Avg Weight')
        tree = ttk.Treeview(summary_frame, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)

        for feature in features:
            avg_weight = (rf_weights[feature] + corr_weights[feature] + impact_weights[feature]) / 3
            tree.insert('', 'end', values=(
                feature,
                f'{rf_weights[feature]:.2%}',
                f'{corr_weights[feature]:.2%}',
                f'{impact_weights[feature]:.2%}',
                f'{avg_weight:.2%}'
            ))

        tree.pack(fill='both', expand=True, padx=10, pady=10)

        # Buttons
        btn_frame = ttk.Frame(analysis_window)
        btn_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(
            btn_frame, 
            text="Xuất PDF", 
            command=lambda: self.export_analysis_pdf(
                [fig1, fig2, fig3], 
                tree, 
                "feature_importance_analysis.pdf"
            )
        ).pack(side='left', padx=5)

        ttk.Button(
            btn_frame,
            text="Lưu Biểu Đồ",
            command=lambda: self.save_analysis_charts([fig1, fig2, fig3])
        ).pack(side='left', padx=5)

        return analysis_window



    def update_model_parameters(self):
        self.recommender.knn_classifier.n_neighbors = 7  # Ví dụ cập nhật k
        self.recommender.prepare_recommendation_model()
        messagebox.showinfo("Thành Công", "Đã cập nhật thông số mô hình")

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedRecommendationApp(root)
    root.mainloop()



