import pandas as pd

# Đọc file CSV
df = pd.read_csv("annotations_train.csv")

# Tính số lượng class duy nhất trong cột 'class_name'
unique_classes_count = df['class_name'].nunique()

# In kết quả
print(f"Số lượng các class duy nhất là: {unique_classes_count}")