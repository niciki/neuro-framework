# neuro-framework
Эта фреймворк определяет класс NeuralNetwork, который включает в себя методы для создания, обучения и использования искусственных нейронных сетей.

Содержит методы для обучения нейронной сети. Метод "train()" выполняет обучение сети на заданном наборе данных, метод "predict()" выполняет предсказание на новых данных, а метод "test()" тестирует сеть на заданном наборе данных и вычисляет метрики качества, метод "create()", который инициализирует атрибуты класса, включая количество слоев, количество нейронов в каждом слое, функцию активации для каждого слоя и методы для обучения и использования нейронной сети.

Метод create класса NeuralNetwork используется для создания искусственной нейронной сети. Параметры метода create:

- self: ссылка на текущий экземпляр класса.
- input_dim: размерность входных данных.
- output_dim: размерность выходных данных.
- hidden_layers: список, который содержит количество нейронов в каждом скрытом слое.
- activations: список, который содержит функции активации для каждого скрытого слоя и выходного слоя.
- optimizer: оптимизатор для обучения нейронной сети.
- loss: функция потерь для обучения нейронной сети.
- task: тип задачи: классификация или регрессия.
- batch_size: размер пакета для обучения нейронной сети.
- epochs: количество эпох для обучения нейронной сети.
- learning_rate: скорость обучения для опт
