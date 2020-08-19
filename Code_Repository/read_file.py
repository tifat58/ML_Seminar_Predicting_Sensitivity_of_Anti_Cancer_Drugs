import matplotlib.pyplot as plt


def plot1(x,y, plot='plot'):
    print(x,y, plot)

# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
# plt.show()


# now create a subplot which represents the top plot of a grid
# with 2 rows and 1 column. Since this subplot will overlap the
# first, the plot (and its axes) previously created, will be removed
plt.figure(1)

plt.subplot(211)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('1st Figure')

plt.plot(range(12), range(2,14), '*-')
plt.legend('first')
plt.subplot(212, facecolor='y') # creates 2nd subplot with yellow background

plt.subplot(211)
plt.ylabel('y axis modified')
plt.plot(range(12), range(5,17))
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

plt.figure(2)
plt.xlabel('An x axis')
plt.ylabel('An y axis')
plt.title('2nd Figure')
plt.plot(range(12), range(2,14), '*-', label='new')
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0))
plt.savefig('test.png', format='png', dpi=900, bbox_inches='tight')
open_file = open('parameters.txt', 'r', encoding='utf-8')

lines = open_file.read().splitlines()

name = lines[0].split('=')[1]
age = lines[1].split('=')[1]
print(name, age)
plot1(100,200, 'a')
with open('parameters.txt', 'r') as fp:
    print(fp.read())
    line = fp.read()
