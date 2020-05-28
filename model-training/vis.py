import pickle
from xgboost import plot_tree
from matplotlib import pyplot
model_name = './no_poly_4900_6.mdl'
# with open(model_name, 'wb') as f:
#     pickle.dump(model, f)
model = pickle.load(open(model_name, 'rb'))
plot_tree(model, num_trees=0)
pyplot.savefig('123.png')
