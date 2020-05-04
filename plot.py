import pandas as pd
import matplotlib.pyplot as plt
raw_data = pd.read_csv('output.csv')
raw_data2 = pd.read_csv('noisy_input.csv')
raw_data3 = pd.read_csv('clean_input.csv')
df = pd.DataFrame(data=raw_data)
df2 = pd.DataFrame(data=raw_data2)
df3 = pd.DataFrame(data=raw_data3)
df.plot(subplots=True, legend=None, title = "output")
df2.plot(subplots=True, legend=None, title = "noisy input")
df3.plot(subplots=True, legend=None, title = "original_input")


plt.tight_layout()
plt.show()