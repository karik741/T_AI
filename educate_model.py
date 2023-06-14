import django
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'T_AI.settings')
django.setup()

from cabinet.models import BtcPrice


def calculate_percentage(part, whole):
    return 100 * float(part) / float(whole)


# Получение упорядоченного списка цен Bitcoin prices = [13000, 14000, 15000... N]
prices = [float(obj.price) for obj in BtcPrice.objects.order_by('time')]

# Создаем MinMaxScaler, который будет нормализовывать наши данные в формат от 0 до 1
scaler = MinMaxScaler(feature_range=(0, 1))

# Нормализация данных prices стал списком списков prices = [[0.161009], [0.158240], ... [n]]
prices = scaler.fit_transform(np.array(prices).reshape(-1, 1))

# подготовка данных для LSTM
n_steps = 168
# x это список списков входных данных [[0.161009, 0.158240 ..], [0.158240, 0.1693259], ... [n]]
# y это список результатов [0.181009, 0.198240, ... [n]]
x = np.array([prices[i:i + n_steps, 0] for i in range(len(prices) - n_steps - 2)])
y = np.array(prices[n_steps + 2:, 0])

# разделение данных на обучающую и тестовую выборку
train_size = int(0.9 * len(x))
# x_train и y_train это списки списков как x [[0.161009, 0.158240 ..], [0.158240, 0.1693259], ... [n]]
# y_train и y_test это списки как y [0.181009, 0.198240, ... [n]]
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# преобразование данных в формат, который можно подать на вход LSTM
# Каждая еденица изменила значение на список - было 0.198240, стало [0.198240]
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# создание модели
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# обучение модели
model.fit(x_train, y_train, epochs=1, verbose=1)

# прогнозирование
y_pred = model.predict(x_test)

# Переводим прогнозированные значения обратно из нормализованного формата
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))


# Вернем x_test к трехмерной форме
x_test_3d = x_test.reshape(-1, n_steps, 1)

# Переводим все элементы обратно из нормализованного формата
x_test_inv = scaler.inverse_transform(x_test_3d.reshape(-1, 1)).reshape(-1, n_steps)

count_of_correct_prediction = 0

# визуализация результатов
for i in range(len(y_pred)):

    print(f'Курс был: {x_test_inv[i][-1]}')
    print(f'Курс стал через 24 часа: {y_test[i][0]}')
    print(f'Предсказано: {y_pred[i][0]}')
    print("Курс упал" if y_test[i][0] < x_test_inv[i][-1] else "Курс поднялся")
    print(f'Предсказано что курс {"упадет" if y_pred[i][0] < x_test_inv[i][-1] else "поднимется"}')
    if (y_test[i][0] > x_test_inv[i][-1] and y_pred[i][0] > x_test_inv[i][-1]) or \
       (y_test[i][0] < x_test_inv[i][-1] and y_pred[i][0] < x_test_inv[i][-1]):
        count_of_correct_prediction += 1
        print("Предсказание верное!")
    else:
        print("Предсказание неверное - отстой")

print(f'Всего предсказаний: {len(y_pred)}')
print(f'Верных предсказаний: {count_of_correct_prediction}')
print(f'На {(count_of_correct_prediction / len(y_pred)) * 100}% предсказано верно')