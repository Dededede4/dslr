import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns;

sns.set(style="ticks", color_codes=True)
iris = pd.read_csv('dataset_train.csv')

iris = iris.dropna()
iris.pop('Index')
g = sns.pairplot(iris, hue="Hogwarts House")

plt.xlabel('Hogwarts House')


plt.show()