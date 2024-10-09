import numpy as np
import pandas as pd
from flask import Flask, render_template_string, Response
import io

app = Flask(__name__)

# Tạo hàm lấy dữ liệu từ file CSV
def loadCsv(filename) -> pd.DataFrame:
	data = pd.read_csv(filename)
	return data

# Tạo hàm biến đổi cột định tính, dùng phương pháp one-hot
def transform(data, columns_trans):
	for i in columns_trans:
		unique = data[i].unique()
		new_columns = [f'{value}-{i}' for value in unique]
		matrix_0 = np.zeros((len(data), len(unique)), dtype=int)
		frame_0 = pd.DataFrame(matrix_0, columns=new_columns)
		for index, value in enumerate(data[i]):
			frame_0.at[index, f'{value}-{i}'] = 1
		data = pd.concat([data, frame_0], axis=1)
	return data

# Tạo hàm scale dữ liệu về [0,1]
def scale_data(data, columns_scale):
	for i in columns_scale:
		_max = data[i].max()
		_min = data[i].min()
		min_max_scaler = lambda x: round((x - _min) / (_max - _min), 3)
		data[i] = data[i].apply(min_max_scaler)
	return data

# Hàm tính khoảng cách Euclid
def cosine_distance(train_X, test_X):
	dict_distance = dict()
	for index, value in enumerate(test_X, start=1):
		for j in train_X:
			result = np.sqrt(np.sum((j - value) ** 2))
			if index not in dict_distance:
				dict_distance[index] = [result]
			else:
				dict_distance[index].append(result)
	return dict_distance

# Hàm gán kết quả theo k (KNN)
def pred_test(k, train_X, test_X, train_y):
	lst_predict = []
	dict_distance = cosine_distance(train_X, test_X)
	train_y = train_y.to_frame(name='target').reset_index(drop=True)
	frame_concat = pd.concat([pd.DataFrame(dict_distance), train_y], axis=1)
	
	for i in range(1, len(dict_distance) + 1):
		sort_distance = frame_concat[[i, 'target']].sort_values(by=i, ascending=True)[:k]
		target_predict = sort_distance['target'].value_counts(ascending=False).index[0]
		lst_predict.append([i, target_predict])
		
	return lst_predict

# Trang chủ
@app.route('/')
def home():
	return '<h1>Welcome to the Drug Prediction App!</h1><p>Go to <a href="/data-preview">/data-preview</a> to see the first rows of the dataset or <a href="/predict">/predict</a> to run KNN predictions.</p>'

# Hiển thị những dòng đầu tiên của dữ liệu
@app.route('/data-preview')
def data_preview():
	data = loadCsv('drug200.csv')
	first_rows = data.head().to_html(classes='table table-striped')
	return render_template_string('''
	<html>
		<head>
			<title>Data Preview</title>
			<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
		</head>
		<body>
			<div class="container">
				<h1>First 5 rows of drug200.csv</h1>
				{{ first_rows|safe }}
			</div>
		</body>
	</html>
	''', first_rows=first_rows)

# Dự đoán bằng mô hình KNN
@app.route('/predict')
def predict():
	data = loadCsv('drug200.csv')
	# Giả sử cột định tính là 'Sex' và 'BP'
	data = transform(data, ['Sex', 'BP'])
	# Giả sử cột cần scale là 'Age' và 'Na_to_K'
	data = scale_data(data, ['Age', 'Na_to_K'])
	
	X_train = data[['Age', 'Na_to_K']].values
	y_train = data['Drug']
	X_test = X_train  # Để đơn giản, sử dụng cùng dữ liệu để làm test
	k = 3  # Chọn giá trị k
	
	predictions = pred_test(k, X_train, X_test, y_train)
	
	pred_str = "<br>".join([f"Test sample {idx}: Predicted {pred}" for idx, pred in predictions])
	return render_template_string('''
	<html>
		<head>
			<title>KNN Predictions</title>
			<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
		</head>
		<body>
			<div class="container">
				<h1>KNN Predictions</h1>
				<p>{{ pred_str|safe }}</p>
			</div>
		</body>
	</html>
	''', pred_str=pred_str)

if __name__ == '__main__':
	app.run(debug=True)
