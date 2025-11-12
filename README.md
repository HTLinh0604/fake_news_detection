#  Fake News Detection using NLP and Deep Learning
*(Phát hiện Tin giả bằng NLP và Học sâu)*

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Transformers](https://img.shields.io/badge/Transformers-BERT-yellow?logo=huggingface)
![NLP](https://img.shields.io/badge/NLP-Preprocessing-6AC045?style=flat&logo=nltk)
![BERT](https://img.shields.io/badge/BERT-Embeddings-7933FF?style=flat&logo=transformers)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?logo=tensorflow)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Models-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-green?logo=xgboost)
![imbalanced-learn](https://img.shields.io/badge/Imbalanced--Learn-Balancing-lightgrey)

---

##  1. Objective & Context *(Mục tiêu & Bối cảnh)*

* **The Problem:** In the age of rapid social media dissemination, fake news poses a direct threat to social stability, public trust, and national security.
    *(**Vấn đề:** Trong bối cảnh thông tin lan truyền nhanh trên mạng xã hội, tin giả (Fake News) là một vấn đề nghiêm trọng, đe dọa trực tiếp đến sự ổn định xã hội và niềm tin cộng đồng.)*
* **The Goal:** This project develops an AI model to automatically classify fake news based on text content (English headlines), comparing traditional NLP methods with modern Deep Learning architectures.
    *(**Mục tiêu:** Đồ án này phát triển một mô hình AI có khả năng tự động phân loại tin giả dựa trên phân tích nội dung văn bản (tiêu đề tiếng Anh), so sánh các phương pháp NLP truyền thống với các kiến trúc Học sâu hiện đại.)*
* **Datasets Used:** GossipCop & PolitiFact.
    *(**Tập dữ liệu sử dụng:** GossipCop & PolitiFact.)*

---

##  2. Methodology & Pipeline *(Phương pháp & Quy trình Thực nghiệm)*

This project follows a systematic, empirical approach to build and validate the detection models.
*(Dự án này tuân theo một phương pháp thực nghiệm có hệ thống để xây dựng và kiểm định các mô hình phát hiện.)*

### Step 1: Exploratory Data Analysis (EDA)
*(Bước 1: Phân tích Dữ liệu Khám phá (EDA))*

* Initial analysis revealed a **severe class imbalance** in the training data: **13,952 Real** samples vs. **4,604 Fake** samples.
    *(Phân tích ban đầu cho thấy sự **mất cân bằng dữ liệu nghiêm trọng** trong tập huấn luyện: **13.952 mẫu Real** so với **4.604 mẫu Fake**.)*

### Step 2: Data Preprocessing & Balancing
*(Bước 2: Tiền xử lý & Cân bằng Dữ liệu)*

1.  **Cleaning:** Standard text cleaning (lowercase, remove punctuation/stopwords, stemming).
    *(**Làm sạch:** Làm sạch văn bản tiêu chuẩn (chuyển chữ thường, loại bỏ dấu câu/stopwords, chuẩn hóa từ).)*
2.  **Balancing (Undersampling):** To address the imbalance, a two-step undersampling process was applied to the training set:
    *(**Cân bằng (Undersampling):** Để xử lý mất cân bằng, một quy trình undersampling hai bước đã được áp dụng trên tập huấn luyện:)*
    * First, **Tomek Links** were used to clean the class boundaries.
        *(*Trước tiên, sử dụng **Tomek Links** để làm sạch ranh giới phân lớp.)*
    * Second, **NearMiss** was applied to reduce the majority class (Real) to match the minority class (Fake).
        *(*Sau đó, áp dụng **NearMiss** để giảm số lượng lớp đa số (Real) cho đến khi bằng lớp thiểu số (Fake).)*
3.  **Final Training Set:** A balanced set of **9,208 samples** (4,604 Real / 4,604 Fake).
    *(**Tập Huấn luyện Cuối cùng:** Một tập dữ liệu cân bằng gồm **9.208 mẫu** (4.604 Real / 4.604 Fake).)*

### Step 3: Feature Extraction Comparison
*(Bước 3: So sánh Trích xuất Đặc trưng)*

We compared three vectorization methods using a baseline Logistic Regression model:
*(Chúng tôi đã so sánh ba phương pháp vector hóa bằng mô hình Hồi quy Logistic cơ sở:)*

| Vectorization Method | Baseline Accuracy | Baseline F1-Score |
| :--- | :--- | :--- |
| TF-IDF | 0.68 | 0.67 |
| Word2Vec (Avg) | 0.71 | 0.72 |
| **BERT (Avg `last_hidden_state`)** | **0.71** | **0.70** |

* **Conclusion:** **BERT embeddings (768 dimensions)** were chosen as the primary feature input for all main experiments due to their robust and superior performance.
    *(**Kết luận:** **Đặc trưng BERT (768 chiều)** được chọn làm đầu vào đặc trưng chính cho tất cả các thí nghiệm mô hình chính thức, vì nó mang lại hiệu suất vượt trội và ổn định.)*

### Step 4: Model Training & Tuning
*(Bước 4: Huấn luyện & Tinh chỉnh Mô hình)*

* **ML Models:** Logistic Regression (LR), Decision Tree (DT), Random Forest (RF), and XGBoost were trained using Stratified 5-Fold Cross-Validation. Hyperparameters were tuned via **GridSearchCV**.
    *(**Mô hình ML:** Hồi quy Logistic (LR), Cây Quyết định (DT), Rừng Ngẫu nhiên (RF), và XGBoost được huấn luyện bằng Đánh giá chéo 5-fold Stratified. Siêu tham số được tinh chỉnh bằng **GridSearchCV**.)*
* **Deep Learning Model:** A hybrid **BERT-LSTM** model (using BERT features as input to a Bi-directional LSTM layer) was trained and tuned manually.
    *(**Mô hình Học sâu:** Một mô hình lai **BERT-LSTM** (sử dụng đặc trưng BERT làm đầu vào cho lớp LSTM hai chiều) đã được huấn luyện và tinh chỉnh thủ công.)*

---

##  3. Results & Key Findings *(Kết quả & Phát hiện Chính)*

After hyperparameter tuning, the models were evaluated on the held-out test set (20% of original data).
*(Sau khi tinh chỉnh siêu tham số, các mô hình được đánh giá trên tập kiểm tra (20% dữ liệu gốc).)*

### Final Model Performance (on Test Set) *(Hiệu suất Mô hình Cuối cùng (trên Tập Kiểm tra))*

| Model | Input Features | Tuned Test Accuracy | Key Observation |
| :--- | :--- | :--- | :--- |
| **LSTM (Hybrid)** | BERT Embeddings | **0.719** | **Highest overall performance.** |
| **XGBoost** | BERT Embeddings | **0.718** | Almost identical performance to LSTM; strong, balanced classification. |
| **Logistic Regression** | BERT Embeddings | 0.709 | Strongest baseline model, proving BERT's feature power. |

* **Key Finding:** The combination of **BERT embeddings** and advanced **undersampling techniques** provided a powerful semantic foundation. The **BERT-LSTM** architecture achieved the highest accuracy, demonstrating the strength of combining Transformer embeddings with sequential models. **XGBoost** proved to be an extremely competitive and well-balanced alternative.
    *(**Phát hiện Chính:** Sự kết hợp của **đặc trưng BERT** và kỹ thuật **undersampling** đã tạo ra một nền tảng ngữ nghĩa mạnh mẽ. Kiến trúc **BERT-LSTM** đạt độ chính xác cao nhất. **XGBoost** cũng chứng tỏ là một giải pháp thay thế có tính cạnh tranh và cân bằng rất cao.)*

---

##  4. Conclusion & Future Work *(Kết luận & Hướng Phát triển)*

### Conclusion *(Kết luận)*

This project successfully demonstrated that models using **BERT embeddings** significantly outperform traditional methods (TF-IDF, Word2Vec) for fake news classification. The **BERT-LSTM** hybrid model (Acc: 0.719) and the tuned **XGBoost** model (Acc: 0.718) were the most effective.
*(Dự án đã chứng minh thành công rằng các mô hình sử dụng **đặc trưng BERT** vượt trội đáng kể so với các phương pháp truyền thống (TF-IDF, Word2Vec). Mô hình lai **BERT-LSTM** (Acc: 0.719) và mô hình **XGBoost** đã tinh chỉnh (Acc: 0.718) là hiệu quả nhất.)*

### Future Work *(Hướng Phát triển Tương lai)*

1.  **Expand Data Scope:** Analyze full article content, not just headlines.
    *(**Mở rộng Phạm vi Dữ liệu:** Phân tích toàn bộ nội dung bài báo thay vì chỉ tiêu đề.)*
2.  **Integrate Context:** Incorporate contextual metadata (user info, propagation networks) using models like Graph Neural Networks (GNNs).
    *(**Tích hợp Ngữ cảnh:** Tích hợp siêu dữ liệu ngữ cảnh (thông tin người dùng, mạng lưới lan truyền) bằng các mô hình như Mạng Nơ-ron Đồ thị (GNN).)*
3.  **Apply to Vietnamese:** Adapt the pipeline for Vietnamese news using models like **PhoBERT**.
    *(**Áp dụng cho Tiếng Việt:** Triển khai các mô hình cho ngôn ngữ Tiếng Việt (ví dụ: **PhoBERT**).)*
4.  **End-to-End Fine-Tuning:** Fine-tune the entire BERT model end-to-end rather than just using it as a static feature extractor.
    *(**Tinh chỉnh End-to-End:** Tinh chỉnh toàn bộ mô hình BERT thay vì chỉ sử dụng nó làm công cụ trích xuất đặc trưng tĩnh.)*

---

##  Authors *(Nhóm Thực hiện)*

**Students:** *(Sinh viên thực hiện)*  
- Hồ Gia Thành  
- Huỳnh Thái Linh  
- Trương Minh Khoa  

**Supervisor:** *(Giảng viên hướng dẫn)* *TS. Lê Cung Tưởng*  
**University:** *(Trường)* Trường Đại học Công nghệ TP. Hồ Chí Minh — *Khoa học Dữ liệu*  
**Year:** *(Năm thực hiện)* 2025

---

> © 2025 — Project: *social media fake news detection*  
> *Developed for academic research and educational purposes.*
