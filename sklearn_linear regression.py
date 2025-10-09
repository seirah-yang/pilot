import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 예시 데이터 (환자 나이 vs 혈압)
age = np.array([25, 30, 35, 40, 45, 50]).reshape(-1, 1)
bp = np.array([120, 122, 130, 138, 143, 150])

# 모델 학습
model = LinearRegression()
model.fit(age, bp)

# 예측 및 시각화
pred = model.predict(age)
plt.scatter(age, bp, label='Real Data')
plt.plot(age, pred, label='Model Prediction', linewidth=2)
plt.xlabel('Age')
plt.ylabel('Blood Pressure')
plt.legend()
plt.title('Linear Regression: Age vs BP')
plt.show()
