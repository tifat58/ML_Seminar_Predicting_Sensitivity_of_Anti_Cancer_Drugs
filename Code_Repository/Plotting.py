import matplotlib.pyplot as plt
import pathlib
import os, sys

train = [0.8695, 0.878, 0.8906]
test = [0.666, 0.777, 0.833]
sen = [0.599, 0.63, 0.64]
spec = [0.975, 0.978, 0.984]



plt.figure(1)
plt.subplot(211)
plt.plot(range(1,4), train, '*-', label='train accuracy')

plt.plot(range(1,4), test, '*-', label='test accuracy')


plt.plot(range(1,4), sen, '*-', label='sensitivity')
plt.plot(range(1,4), spec, '*-', label='specificity')
plt.xlabel('No. of hidden layers')
plt.ylabel('Values')
plt.title('Accuracy values vs Hidden Layers for drug KIN001-260')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='center left', bbox_to_anchor=(1, 0))
plt.grid(True)


# for feature number
ntrain = [0.89, .86, .82, .712]
ntest = [0.83, .88, .51, .87]
nsen = [.64, .57, .44, 0]
nspec = [.94, .97, .97, 1]
features = [100,150,200,300]
plt.subplot(212)

plt.plot(features, ntrain, '*-', label='train accuracy')
plt.plot(features, ntest, '*-', label='test accuracy')

plt.plot(features, nsen, '*-', label='sensitivity')
plt.plot(features, nspec, '*-', label='specificity')
plt.xlabel('No. of Features')
plt.ylabel('Values')
plt.title('Accuracy values vs No. of features for drug KIN001-260 ')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='center left', bbox_to_anchor=(1, 0))
plt.grid(True)

plt.savefig('../extra_1.png', format='png', dpi=1000, bbox_inches='tight')
plt.close()
