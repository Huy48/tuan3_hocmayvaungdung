import numpy as np
import pandas as pd
from flask import Flask, render_template_string

app = Flask(__name__)

# Hàm đọc dữ liệu từ file Excel
def loadExcel(filename) -> pd.DataFrame:
	data = pd.read_excel(filename)
	return data

# Tạo tập train/test
def splitTrainTest(data, target, ratio=0.25):  # data --> frame
	from sklearn.model_selection import train_test_split
	data_X = data.drop([target], axis=1)
	data_y = data[[target]]
	X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=ratio, random_state=42)
	data_train = pd.concat([X_train, y_train], axis=1)
	
	return data_train, X_test, y_test  # đều là dạng frame

# Hàm tính trung bình của từng lớp trong biến target
def mean_class(data_train, target):  # tên cột target, data_train là dạng pandas
	df_group = data_train.groupby(by=target).mean()  # tất cả các cột đều dạng số
	return df_group  # kết quả là dataframe

# Hàm dự đoán dùng khoảng cách Euclid
def target_pred(data_group, data_test):  # data_test ở dạng mảng
	dict_ = dict()
	for index, value in enumerate(data_group.values):
		result = np.sqrt(np.sum(((data_test - value) ** 2), axis=1))  # khoảng cách euclid
		dict_[index] = result
	df = pd.DataFrame(dict_)
	return df.idxmin(axis=1)  # tìm cột chứa giá trị nhỏ nhất

# Trang chủ
@app.route('/')
def home():
	return '<h1>Welcome to the Iris Prediction App!</h1><p>Go to <a href="/data-preview">/data-preview</a> to see the first rows of the dataset or <a href="/predict">/predict</a> to run predictions.</p>'

# Hiển thị những dòng đầu tiên của dữ liệu
@app.route('/data-preview')
def data_preview():
	data = loadExcel('Iris.xls')
	first_rows = data.head().to_html(classes='table table-striped')
	return render_template_string('''<html>
		<head>
			<title>Data Preview</title>
			<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
		</head>
		<body>
			<div class="container">
				<h1>First 5 rows of Iris dataset</h1>
				{{ first_rows|safe }}
			</div>
		</body>
	</html>''', first_rows=first_rows)

# Dự đoán bằng mô hình KNN
@app.route('/predict')
def predict():
	data = loadExcel('Iris.xls')
	data_train, X_test, y_test = splitTrainTest(data, 'iris', ratio=0.3)
	
	# Tính trung bình lớp
	df_group = mean_class(data_train, 'iris')
	
	# Dự đoán lớp cho tập test
	predictions = target_pred(df_group, X_test.values)
	
	# Chuyển đổi thành DataFrame để hiển thị
	pred_df = pd.DataFrame(predictions, columns=['Predict'])
	
	# Đặt lại chỉ số cho y_test
	y_test.index = range(len(y_test))
	y_test.columns = ['Actual']
	
	# Kết hợp kết quả dự đoán với giá trị thực
	result_df = pd.concat([pred_df, y_test], axis=1)
	
	result_html = result_df.to_html(classes='table table-striped')
	
	return render_template_string('''<html>
		<head>
			<title>Predictions</title>
			<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
		</head>
		<body>
			<div class="container">
				<h1>Predictions</h1>
				{{ result_html|safe }}
			</div>
		</body>
	</html>''', result_html=result_html)

if __name__ == '__main__':
	app.run(debug=True)
