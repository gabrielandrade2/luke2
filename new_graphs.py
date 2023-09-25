import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from string import ascii_lowercase


# SMALL_SIZE = 14
# MEDIUM_SIZE = 16
# BIGGER_SIZE = 18
#
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def forceAspect(ax,aspect=1):
    # plt.gca().set_aspect('equal')
    pass

def divide_lists(list1, list2):
    if not list1 or not list2:
        return []
    return [list1[0] / list2[0]] + divide_lists(list1[1:], list2[1:])


df = pd.read_excel('/Users/gabriel-he/Documents/NAIST-PhD/Strawberry score/Book1.xlsx', sheet_name='Sheet2', header=1)

x = df.columns[-11:].to_list()
x[0] = '0'
x[-10:] = ["{:.0f}".format(int(i*100)) for i in x[-10:]]

names2 = ["HPAELDC", "LUKE", "VanillaNER + LUKE"]
# names_EL = ["HPAELDC", "LUKE", "SimpleNER + LUKE"]
names = iter(ascii_lowercase)


#HPAELDC
offsets = []
offsets.append(df.index[df['Model'] == "HPAELDC"].item())

#LUKE
offsets.append(df.index[df['Model'] == "Luke"].item())

#SimpleNER
offsets.append(df.index[df['Model'] == "SimpleNER"].item())

# Plots
fig, axs = plt.subplots(3, 3, figsize=(9, 9))
for i, (offset, ax, name, name2) in enumerate(zip(offsets, axs[0], names, names2)):
    baseline = df.iloc[offset + 4, 5]
    ax.set_xticks(range(len(x)), x)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_ylim([0.0, 1])

    # ax.text(6, baseline - 0.07, 'Original F1', color='darkblue')
    ax.axhline(y=baseline, color='darkblue', linestyle='--')

    ax.plot(df.iloc[offset + 0, 5:].to_list(), label='Strict Precision', marker='x',  markerfacecolor='none',color='C2', linewidth=1.0)
    ax.plot(df.iloc[offset + 2, 5:].to_list(), label='Strict Recall', marker='^',  markerfacecolor='none',color='C1', linewidth=1.0)
    ax.plot(df.iloc[offset + 4, 5:].to_list(), label='Strict F1', marker='o',  markerfacecolor='none',color='C0', linewidth=2.5)
    forceAspect(ax)
    # ax.plot(df.iloc[offset + 7, 5:].to_list(), label='NER Relaxed Precision', marker='.', linestyle='--', color='C2')
    # ax.plot(df.iloc[offset + 9, 5:].to_list(), label='NER Relaxed Recall', marker='.', linestyle='--', color='C1')
    # ax.plot(df.iloc[offset + 11, 5:].to_list(), label='NER Relaxed F1', marker='.', linestyle='--', color='C0')

    # ax.plot(df.iloc[offset + 13, 5:].to_list(), label='EL Precision', marker='.')
    # ax.plot(df.iloc[offset + 15, 5:].to_list(), label='EL Recall', marker='.')
    # ax.plot(df.iloc[offset + 17, 5:].to_list(), label='EL F1', marker='.')

    ax.title.set_text(name2+"\n("+name+")")


    if i == 0:
        ax.set_ylabel('Strict NER Score')
    #     ax.legend(loc="lower left")
    # if i == 1:
    #     ax.set_xlabel('% of boundary-expanded mentions')
# plt.tight_layout()
# plt.savefig('1.png')
# fig.show()


# fig, axs = plt.subplots(1, 3, figsize=(10, 3))
for i, (offset, ax, name) in enumerate(zip(offsets, axs[1], names)):
    baseline = df.iloc[offset + 11, 5]
    ax.set_xticks(range(len(x)), x)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_ylim([0.0, 1])

    # ax.text(6, baseline - 0.07, 'Original F1', color='darkblue')
    ax.axhline(y=baseline, color='darkblue', linestyle='--')

    ax.plot(df.iloc[offset + 7, 5:].to_list(), label='Relaxed Precision', marker='x',  markerfacecolor='none',color='C2', linewidth=1.0)
    ax.plot(df.iloc[offset + 9, 5:].to_list(), label='Relaxed Recall', marker='^',  markerfacecolor='none',color='C1', linewidth=1.0)
    ax.plot(df.iloc[offset + 11, 5:].to_list(), label='Relaxed F1', marker='o',  markerfacecolor='none',color='C0', linewidth=2.5)
    forceAspect(ax)

    ax.title.set_text("("+name+")")


    if i == 0:
        ax.set_ylabel('Relaxed NER Score')
        # ax.legend(loc="lower left")
    # if i == 1:
    #     ax.set_xlabel('% of boundary-expanded mentions')
# plt.tight_layout()
# plt.savefig('2.png')
# fig.show()

# fig, axs = plt.subplots(1, 3, figsize=(10, 3))

sums=[[0.8903,0.8802,0.8651,0.8183,0.7607,0.7172,0.6168,0.5548,0.4524,0.3070,0.0926],
      [0.9845,0.9806,0.9814,0.9732,0.9531,0.7086,0.3950,0.2028,0.1410,0.1576,0.1601],
      [0.9528,0.9531,0.9530,0.9489,0.9461,0.9019,0.8158,0.7784,0.8212,0.9105,0.9599]]

for i, (offset, ax, name) in enumerate(zip(offsets, axs[2], names)):
    baseline = df.iloc[offset + 17, 5]
    ax.set_xticks(range(len(x)), x)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_ylim([0.0, 1])

    # ax.text(6, baseline - 0.07, 'Original F1', color='darkblue')
    ax.axhline(y=baseline, color='darkblue', linestyle='--')
    #
    # ax.plot(sums[i], label='valid detections', marker=None,  markerfacecolor='none',color='C2', linewidth=1.0, linestyle='--')
    ax.plot(df.iloc[offset + 13, 5:].to_list(), label='Precision', marker='x',  markerfacecolor='none',color='C2', linewidth=1.0)
    ax.plot(df.iloc[offset + 15, 5:].to_list(), label='Recall', marker='^',  markerfacecolor='none',color='C1', linewidth=1.0)
    ax.plot(df.iloc[offset + 17, 5:].to_list(), label='F1', marker='o',  markerfacecolor='none',color='C0', linewidth=2.5)
    forceAspect(ax)

    ax.title.set_text("("+name+")")

    if i == 0:
        ax.set_ylabel('ED Score')
        ax.legend(loc="lower left")
    if i == 1:
        ax.set_xlabel('% of boundary-expanded mentions')

# line = plt.Line2D((.35,.35),(0.06,0.93), color="k", linewidth=0.7)
# fig.add_artist(line)
#
# line = plt.Line2D((.67,.67),(0.06,0.93), color="k", linewidth=0.7)
# fig.add_artist(line)


plt.tight_layout(pad=1.5)
plt.savefig('3.png')
plt.savefig('3.svg', format = 'svg', dpi=1800)
plt.savefig('3.pdf', format = 'pdf', dpi=1800)
fig.show()

df = pd.read_excel('/Users/gabriel-he/Documents/NAIST-PhD/Strawberry score/Book1.xlsx', sheet_name='Sheet7', header=1)
offsets = []
offsets.append(df.index[df['Model'] == "HPAELDC"].item())

#LUKE
offsets.append(df.index[df['Model'] == "Luke"].item())

#SimpleNER
offsets.append(df.index[df['Model'] == "SimpleNER"].item())



names = iter(ascii_lowercase)

fig, axs = plt.subplots(1, 3, figsize=(9, 3))
for i, (offset, ax, name, name2) in enumerate(zip(offsets, axs, names, names2)):
    baseline = df.iloc[offset, 5]
    ax.set_xticks(range(len(x)), x)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_ylim([0.0, 1])

    ax.plot(df.iloc[offset + 0, 3:].to_list(), label='Exact',  markerfacecolor='none',marker='o')
    ax.plot(df.iloc[offset + 1, 3:].to_list(), label='Exceeding',  markerfacecolor='none',marker='x')
    ax.plot(df.iloc[offset + 2, 3:].to_list(), label='Partial',  markerfacecolor='none',marker='^')
    ax.plot(df.iloc[offset + 3, 3:].to_list(), label='Incorrect',  markerfacecolor='none',marker='v')
    ax.plot(df.iloc[offset + 4, 3:].to_list(), label='Missing',  markerfacecolor='none',marker='s')
    forceAspect(ax)

    ax.title.set_text(name2+"\n("+name+")")

    if i == 0:
        ax.set_ylabel('% of mentions')
    if i == 1:
        ax.set_xlabel('% of boundary-expanded mentions')

    if i == 2:
        ax.legend(loc="center left", prop={'size': 9})

plt.tight_layout()
plt.savefig('4.png')
plt.savefig('4.svg', format = 'svg', dpi=1800)
plt.savefig('4.pdf', format = 'pdf', dpi=1800)
fig.show()
