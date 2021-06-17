import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from deslib.des.knora_u import KNORAU
from deslib.des.des_knn import DESKNN
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from tabulate import tabulate
from KnoraeAlg import KnoraeAlg
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import ttest_rel

#lista z nazwami wybranymi zbiorami danych
datasets =['fri_c0_100_25','chscase_census4','breastcancoimbra','cryotherapy','chscase_census5','chscase_census3','heart','chscase_census2','sonar','fri_c0_500_10','fri_c4_500_50','pwLinear','phpAyyBys','fri_c3_100_5','fri_c1_100_5','ex2data2','phphZierv','visualizing_environmental','fri_c4_100_25','fri_c0_100_5']

#określenie długości listy zawierającej zbiory danych
n_datasets = len(datasets)
#ilość foldów
n_splits = 5
#ilość powtórzeń
n_repeats = 10
#inicjalizacja stratyfikowanej walidacji krzyżowej
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=2137)


#inicjalizacja zespołów klasyfikatorów
clfs= {
    'KNORAU':KNORAU(random_state=2137),
    'KNORAE': KnoraeAlg(pool_classifiers=[GaussianNB(),DecisionTreeClassifier(random_state=2137),KNeighborsClassifier()]),
    'DESKNN':DESKNN(random_state=2137),
    'ADABOOST':AdaBoostClassifier(base_estimator=GaussianNB(),random_state=2137),

}
#inicjalizacja nagłówków do tabel
headers = ["KNORAU", "KNORAE", "DESKNN","ADABOOST"]
#inicjalizacja nazw kolumn do tabel
names_column = np.array([["KNORAU"], ["KNORAE"], ["DESKNN"],["ADABOOST"]])


# pętla wyznaczająca dokładności poszczególnych zespołów dla konkretnych zbiorów danych
for data_id, dataset in enumerate(datasets):
    #wyświetlenie nazwy zbioru danych
    print("Zbiór danych:",dataset)
    #inicjalizacja macierzy na wyniki
    scores = np.zeros((len(clfs), n_splits * n_repeats))
    #wyodrębnienie zbioru danych z pliku
    dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",")
    #określenie wzoroców z atrybutami
    X = dataset[:, :-1]
    #wyodrębnienie etykiet
    y = dataset[:, -1].astype(int)
    #wyświetlenie liczby wzorców
    print("Liczba wzorców:",X.shape[0])
    #wyświetlenie liczby atrybutów
    print("Liczba atrybutów:",X.shape[1])
    #wyznacznie unikalnych wartości ze zbioru etykiet
    values, counts = np.unique(y, return_counts=True)
    #wyświetlenie klas
    print("Jakie klasy:",values)
    #wyświetlenie liczby wzorców w danej klasie
    print("Liczność klas",counts)
    #pętla do wyznaczenie dokładkości poszczególnych zespołów wraz z stratyfikowaną walidacją krzyżową
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    #dla poszczególnych zespołów
        for clf_id, clf_name in enumerate(clfs):
            #klonowanie zespołów
            clf = clone(clfs[clf_name])
            #dopasowanie danych
            clf.fit(X[train], y[train])
            #predykcja
            y_pred = clf.predict(X[test])
            #wyliczenie dokładności dla każdego folda
            scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)
    #obliczenie średniej dokładoności
    mean = np.mean(scores, axis=1)
    #obliczenie odchylenia standardowego
    std = np.std(scores, axis=1)
    #wyświetlenie średnich dla każdego z zespołów
    print("średnie dla danych zespołów")
    for clf_id, clf_name in enumerate(clfs):
        print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))
    #określenie poziomu ufności do testów statystycznych
    alfa = .05
    #inicjalizacja macierzy na t-statystyki
    t_statistic = np.zeros((len(clfs), len(clfs)))
    #inicjalizacja macierzy na p-wartości
    p_value = np.zeros((len(clfs), len(clfs)))

    #obliczenie t-statystyki oraz p-wartości
    for i in range(len(clfs)):
        for j in range(len(clfs)):
            t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])

    #wyświetlenie sformatowanej tabeli z t-statystykami i p-wartościami
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("t-statystyka:\n", t_statistic_table, "\n\np-wartość:\n", p_value_table)

    #wyświetlenie sformatowanej tabeli przewag

    advantage = np.zeros((len(clfs), len(clfs)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("Przewagi:\n", advantage_table)

    #wyświetlenie zformatowanej tabeli zależności statystycznych

    significance = np.zeros((len(clfs), len(clfs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("Zależności statystyczne (alpha = 0.05):\n", significance_table)

