import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.neighbors import NearestNeighbors
from sklearn.base import clone
from sklearn.metrics import accuracy_score

class KnoraeAlg(BaseEnsemble,ClassifierMixin):
    # zdefinioweanie __init__
    def __init__(self, pool_classifiers=None, random_state=None,k=5):
        #okreslenie zbioru wykorzystywanych klasyfikatorow
        self.pool_classifiers = pool_classifiers
        #okreslenie wielkosci obszaru kompetencji
        self.k=k
        #okreslenie ziarna losowosci
        self.random_state = random_state
        np.random.seed(self.random_state)

    def fit(self,X,y):
        #sprawdzenie zbiorow X oraz y
        X, y = check_X_y(X, y)
        #przechowujemy X i y

        self.X_, self.y_ = X, y
        #okreslenie zespolu klasyfikatorow przez przypisanie in nazw do tablicy ensemble_
        self.ensemble_ = []
        #przypisanie klasyfikatorow ze zbioru pool_classifiers do tablicy ensemble_
        for i in range(len(self.pool_classifiers)):
            self.ensemble_.append(clone(self.pool_classifiers[i]).fit(self.X_, self.y_))

        return self

    def predict(self,X):
        #wyznacznie najblizszych sasiadow przy pomocy algorytmu NearestNeighbors
        #n_neighbors- ile nablizszych sasiadow ma wyznaczyc jest rowne wielkosci obszaru kompetencji k
        #algorithm- jaki algorytm bedzie wykorzystany w NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(self.X_)
        #okreslanie sasiadow naszych sampli
        distances, indices = nbrs.kneighbors(X)
        #inicjacja macierzy przechowujacej dokladnosci poszegolnych klasyfiaktow w obszaru kompetencji k
        scores = np.zeros((len(self.pool_classifiers), indices.shape[0]))
        #inicjacja macierzy przechowujacej informacje o klasyfiaktorach ktore osiagnely 100% dokladnosci w obszaru kompetencji k
        check_oracle=np.zeros((len(self.pool_classifiers), indices.shape[0]))

        #zmienna k do dzialania petli while
        k=0
        while True:
            #tablica do ktorej beda przypisywane wzorce dla ktroych zaden klasyfiaktor nie osiagnal 100 dokladnosci
            no_100 = []
            # zatrzymanie petli gdy region kompetecji bedzie rowny 0
            if indices.shape[1] == 0:
                break
            #okreslanie klasyfikatorow ktore osiagnely 100 dokladnosci w regionie kompetecji
            for j in range(indices.shape[0]):
                #kopia zbiorow X oraz y
                temp_X=self.X_
                temp_y=self.y_
                #przypisanie sasiadow probki dla ktorych dokonujemy predykcji w regionie kompetecji
                test=[]
                test=indices[j]
                #usuniecie probek testowych ze zbioru uczacego
                train_X=np.delete(temp_X,test,0)
                train_y=np.delete(temp_y,test,0)
                #lista do ktorej przypisujemy wyniki predykcji dla sasiadow danej probki
                temporary = []
                #dokonanie predykcji dla sasiadow danej probki
                for i in range(len(self.pool_classifiers)):
                    #okreslanie klasyfikatora
                    clf=self.pool_classifiers[i]
                    #dopasowanie modelu
                    clf.fit(train_X, train_y)
                    #dokonanie predykcji
                    y_pred=clf.predict(temp_X[test])
                    #sprawdzenie czy jakis klasyfikator osiagnal 100 dokladnosci po pierwszej iteracji
                    for z in range(len(self.pool_classifiers)):
                        if scores[z][j] == 1:
                            break
                    #wpisanie dokladnosci jaka uzyskal dany klasyfiaktor
                    scores[i][j]=accuracy_score(temp_y[test],y_pred)
                    #przypisanie wyników gdy dokladnosc osiagnela 100% do check_oracle
                    if 1.0 == scores[i][j]:
                        check_oracle[i][j]=1
                    temporary.append(accuracy_score(temp_y[test],y_pred))
                #przypisanie numerow probek dla ktroych nie osiagnieto 100 dokladnosci w ich regionie
                if not 1.0 in temporary:
                    no_100.append(j)
            #jezeli dla wszysztkich regionow osiagnieto 100 dokladnosci
            if len(no_100) == 0:
                break
            k+=1
            #pomniejszenie regionu kompetecji
            indices=np.delete(indices,indices.shape[1]-1,axis=1)

        #zmiana wartosci float na int w macierzy chceck_oracle
        check_oracle = check_oracle.astype(int)
        #inicjacja macierzy zawięrającej głosy poszczególnych klasyfikatorów w zespole
        pred_=np.empty([check_oracle.T.shape[0],len(self.pool_classifiers)],dtype=np.uint8)


        X_=[]

        # w tej pętli dokonuje się predykcji poszczególnych wzorców
        for j in range(X.shape[0]):
            #zmiana wymiaru z 1D do 2D
            X_= X[j].reshape(1,-1)
            #pętla, w której wyznaczone klasyfikatory dokonują predykcji
            for i,member_clf in enumerate(self.ensemble_):
                #jeżeli żaden klasyfikator nie osiągnął 100 % dokładności predykcji dokonuje pierwszy w zespole
                if np.max(check_oracle.T[j]) == 0:
                    pred_[j]=member_clf.predict(X_)
                    break
                #jeżeli dany klasyfikator jest upoważniony do podjęcia predykji jej wynik zapisywany jest w macierzy pred_
                elif check_oracle.T[j][i] == 1:
                    pred_[j][i] = member_clf.predict(X_)
                # po predykcji wychodzimy z pętli wewnętrznej i przechodzimy do kolejnego wzorca
                else:
                    continue
        #liczenie głosów w obrębie jednego zespołu
        prediction = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=pred_)
        #zwracamy macierz z wynikami predykcji
        return prediction

