import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from deslib.des.knora_u import KNORAU
from deslib.des.des_knn import DESKNN
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from scipy.stats import rankdata
#from scipy.stats import ttest_rel
from scipy.stats import ranksums
from tabulate import tabulate
from KnoraeAlg import KnoraeAlg



datasets =['banknote','breastcan','breastcancoimbra','cryotherapy','diabetes','german','heart','ionosphere','sonar','spambase','wine','wisconsin','phpAyyBys','dataset_net','fri_c1_100_5','ex2data2','phphZierv','visualizing_environmental','fri_c4_100_25','fri_c0_100_5']


n_datasets = len(datasets)
n_splits = 5
n_repeats = 10
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)



clfs= {
    'KNORAU':KNORAU(random_state=2137),
    'DESKNN':DESKNN(random_state=2137),
    'ADABOOST':AdaBoostClassifier(base_estimator=GaussianNB(),random_state=2137),
}

scores = np.zeros((len(clfs), n_datasets  ,n_splits * n_repeats))
'''
for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id,data_id, fold_id] = accuracy_score(y[test], y_pred)

np.save('results', scores)
'''

scores = np.load('results.npy')
mean_scores = np.mean(scores, axis=2).T
print("\nMean scores:\n", mean_scores)

mean_scores_1 = np.mean(mean_scores, axis=0)
print("\nMean ranks:\n", mean_scores_1)

ranks = []
for ms in mean_scores:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)



mean_ranks = np.mean(ranks, axis=0)
print("\nMean ranks:\n", mean_ranks)


alfa = .05
w_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

alfa = .05
t_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(ranks.T[i], ranks.T[j])
print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)



headers = list(clfs.keys())
names_column = np.expand_dims(np.array(list(clfs.keys())), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(clfs), len(clfs)))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\nAdvantage:\n", advantage_table)
