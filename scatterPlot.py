import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, BoundaryNorm
from urllib.parse import quote

gsheetid = "1BLUATC6DDbnfDTqoXzAXDSOh12Vc8Nw8aEUqjEJWWVU"
sheet_name = quote("Company MetaData-2024")
gsheet_url = f"https://docs.google.com/spreadsheets/d/{gsheetid}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
df = pd.read_csv(gsheet_url, thousands=',', decimal='.')
df = df[:-3]
df.iloc[:, [22, 26, 28, 33, 6]] = df.iloc[:, [22, 26, 28, 33, 6]].fillna(0)
df = df.loc[df.iloc[:, 6] != 0]
def withSplit():
    unsplit_cols = [3, 8, 11, 16, 19, 28, 33]
    split_cols = [4, 10, 12, 18, 20, 30, 35]

    for unsplit_col, split_col in zip(unsplit_cols, split_cols):
        split_values = df.iloc[:, split_col]
        df.iloc[:, unsplit_col] = df.iloc[:, unsplit_col].where(split_values.isna(), split_values)

withSplit()

df['percent'] = (df.iloc[:, 26] + df.iloc[:, 28] + df.iloc[:, 33])/df.iloc[:, 6]
sorted_df = df.sort_values(by='percent', ascending=True)
totalX = sorted_df['percent']
ticks = sorted_df.iloc[:, 2]

#Set shape for each point
markers = ['^' if num <= 0 else 'o' for num in sorted_df.iloc[:, 19]]

sheet_name = quote("2020-2024 with Splits")

gsheet_url = f"https://docs.google.com/spreadsheets/d/{gsheetid}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data_df = pd.read_csv(gsheet_url, thousands=',', decimal='.')
data_df['Transaction Date'] = pd.to_datetime(data_df['Transaction Date'], errors='coerce')
maxHoldTime = []
totalHoldTime = []
for comp in sorted_df['Ticker Symbol']:
    filtered_df = data_df[data_df['Ticker'] == comp]
    maxTime = counter = totalTime = 0
    firstBuyDate = lastDate = pd.Timestamp.now()
    for row in filtered_df.itertuples():
        if row.Action == 'Market buy':
            if counter <= 0:
                firstBuyDate = row[2]
            counter += row[6]
        elif row.Action == 'Market sell':
            counter -= row[6]
        else:
            continue
        if counter <= 0:
            totalTime += (row[2] - firstBuyDate).days
            maxTime = max(maxTime, (row[2] - firstBuyDate).days)
    if counter > 0:
        totalTime += (lastDate - firstBuyDate).days
        maxTime = max(maxTime, (lastDate - firstBuyDate).days)

    maxHoldTime.append(maxTime)
    totalHoldTime.append(totalTime)

#Calculate color for each point
colorData = np.array(totalX)
boundaries = [-30, -20, -10, 0, 10, 20, 30]  # Percentage points
boundaries = np.array(boundaries) / 100

colors = [
    (246/255, 53/255, 56/255),  # -30%
    (191/255, 64/255, 69/255),  # -20%
    (139/255, 68/255, 78/255),  # -10%
    (65/255, 69/255, 84/255),   # 0%
    (53/255, 118/255, 78/255),  # +10%
    (47/255, 158/255, 79/255),  # +20%
    (48/255, 204/255, 90/255),  # +30%
]

cmap = ListedColormap(colors)

norm = BoundaryNorm(boundaries, cmap.N)

sm = ScalarMappable(cmap=cmap, norm=norm)
colors = sm.to_rgba(colorData)

#Calculate size for each point
sharesNum = sorted_df.iloc[:, 19]
sharesNum = sharesNum.where(sharesNum >= 0, 0)
currentSharePrice = sorted_df.iloc[:, 22]
product = sharesNum * currentSharePrice
scaled_sizes = 50 + 150 * (product / product.max())

#Setup scatterplot
fig, ax = plt.subplots()
fig.set_figwidth(10)
ax.set_xlabel('Return on Investment')
ax.set_ylabel('Maximum days of stock ownership')

for x, y, m, c, s in zip(totalX, maxHoldTime, markers, colors, scaled_sizes):
    ax.scatter(x, y, marker=m, c=[c], s=s, edgecolors='black', linewidths=0.5)

#Display color bar
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Return on Investment', size=8)
cbar.set_ticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
cbar.ax.set_yticklabels(['< -30%', '-20%', '-10%', '0%', '+10%', '+20%', '> +30%'])

#Set gap between x-axis ticks
xTicks = np.linspace(min(totalX), max(totalX), 30)
xTicks = np.around(xTicks, decimals=3)
xTickLabels = [f'{tick:.3f}' for tick in xTicks]
ax.set_xticks(xTicks)
#Set x-axis ticks' rotation
ax.set_xticklabels(xTickLabels, rotation=90)
#Draw 0 line
ax.axvline(x=0, color='r', linestyle='--')
ax.set_title('Maximum days of stock ownership vs Return on Investment')

#Draw company ticks and adjust their position
new_ticks = [plt.text(x, y, tick, fontsize = 6) for x, y, tick in zip(totalX, maxHoldTime, ticks)]
adjust_text(new_ticks, expand=(1.5, 1.2),
            arrowprops=dict(arrowstyle='->', color='grey'))

#Mouse interaction
#cursor = mplcursors.cursor(scatter, hover=True)
#@cursor.connect("add")
#def on_add(sel):
#    compName = sorted_df.iloc[sel.index, 1]
#    sel.annotation.set(text=f'x={sel.target[0]:.2f}, y={sel.target[1]:.2f}, Company: {compName}')

plt.show()