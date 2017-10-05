import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)



class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Model selection based on BIC scores
        modelBIC = []
        ## Based on different number of hidden states and their corresponding BIC values, the one with min BIC value is taken
        for num_hidden_states in range(self.min_n_components,self.max_n_components+1):
            try:
                model = self.base_model(num_hidden_states) ## Make a model
                logL = model.score(self.X, self.lengths)
                logN = np.log(len(self.X))
                params = num_hidden_states * num_hidden_states + 2 * num_hidden_states * len(self.X[0]) - 1
                scoreBIC = -2 * logL + params * logN
                # print("word: {}, num_states: {}, BIC: {}".format(self.this_word, num_hidden_states, scoreBIC))
                modelBIC.append((scoreBIC,model))
            
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_hidden_states))
                pass
            
        if modelBIC != []:
            scoreBIC, model = min(modelBIC)
            return model
        else:
            return None


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Model selection based on DIC scores
        modelDIC = []
        ## Based on different number of hidden states and their corresponding DIC values, the one with max DIC value is taken
        for num_hidden_states in range(self.min_n_components,self.max_n_components+1):
            try:
                model = self.base_model(num_hidden_states) ## Make a model
                logLword = model.score(self.X, self.lengths)
                logLothers = []
                for word in self.words:
                    if word == self.this_word:
                        continue
                    other_word_X, other_word_lengths = self.hwords[word]
                    logL_other_score = model.score(other_word_X, other_word_lengths)
                    logLothers.append(logL_other_score)
                    
                avg_logL_others = np.average(logLothers)
                scoreDIC = logLword - avg_logL_others
                # print("word: {}, num_states: {}, BIC: {}".format(self.this_word, num_hidden_states, scoreBIC))
                modelDIC.append((scoreDIC,model))
            
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_hidden_states))
                pass
            
        if modelDIC != []:
            scoreDIC, model = max(modelDIC)
            return model
        else:
            return None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def select(self):
        results = []
        num_splits = len(self.lengths)
        ''' As KFold() require splits for 2 or more, so there is a special case
            for num_splits == 1 where the algorithm wont be processed
        '''
        if num_splits == 1:
            for num_states in range(self.min_n_components, self.max_n_components + 1):
                try:
                    # print("Train fold indices:{}".format(train_idx))  # view indices of the folds
                    model = self.base_model(num_states)
                    logL = model.score(self.X, self.lengths)
                    # print("Word:{}, num_states:{} => logL:{}".format(word, num_states, logL))
                    results.append((logL, model))
                except:
                    # print("Error training model for word: {} with num_states: {}".format(word, num_states))
                    pass
        else:
            split_method = KFold(n_splits=min(3, len(self.lengths)))
            word = self.this_word
            for num_states in range(self.min_n_components, self.max_n_components + 1):
                try:
                    scores = []
                    model = self.base_model(num_states)
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        # print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))  # view indices of the folds
                        try:
                            train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                                    random_state=self.random_state, verbose=False).fit(train_X,
                                                                                                       train_lengths)
                            test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                            logL = model.score(test_X, test_lengths)
                            scores.append(logL)
                        except:
                            if self.verbose:
                                print("failure on {} with {} states".format(self.this_word, num_states))
                            pass
                    if scores:
                        avgLogLikelihood = np.average(scores)
                        # print("Word:{}, num_states:{} => scores:{}, avg:{}".format(word, num_states, scores, avgLogLikelihood))
                        results.append((avgLogLikelihood, model))
                except ValueError as valueError:
                    if self.verbose:
                        print("Error evaluating model for num_states: {} and word: {} - error: {}".format(num_states, word, valueError))
                    pass
        
        if results != []:
            score, model = max(results, key=lambda x:x[0])
            return model
        else:
            return None

