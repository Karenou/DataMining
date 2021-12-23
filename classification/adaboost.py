import numpy as np
np.random.seed(0)

class Adaboost:

    def __init__(self) -> None:
        """
        initialize data and weight
        """
        self.n = 10
        self.x = list(range(self.n))
        self.y = np.array([1,1,-1,-1,-1,1,1,-1,-1,1])
        self.weight = np.ones(self.n) / self.n

    def compute_error(self, weight, y_true, y_pred) -> float:
        """
        @param weight: weight of data
        @param y_true: true label
        @param y_pred: predicted label
        return the weighted error rate
        """
        return np.sum(weight * (y_true != y_pred)) / self.n

    def weak_classifier(self, x, y) -> tuple:
        """
        aim at finding the threshold v to minimize the error rate
        @param x: sampled x
        @param y: true labels of sampled data
        return the optimal threshold and the direction
        """
        min_error = 1
        optimal_threshold = None
        final_pos = None

        for threshold in np.arange(min(x), max(x) + 0.5, 0.5):
            pos_direction = True
            y_pred = np.array([1 if x_ > threshold else -1 for x_ in x])
            y_pred_rev = -1 * y_pred

            error = self.compute_error(self.weight[x], y, y_pred)
            error_rev = self.compute_error(self.weight[x], y, y_pred_rev)
            
            if min(error, error_rev) < min_error:
                if error < error_rev:
                    min_error = error
                    final_pos = pos_direction
                else:
                    min_error = error_rev
                    final_pos = not pos_direction

                optimal_threshold = threshold

        return optimal_threshold, final_pos

    def train(self):
        """
        build and train the adaboost model
        """
        estimator_weights = []   # weight of each weak classifier
        estimator_errors = []   # error of each weak classifier
        estimator_thresholds = []     # threshold of each weak classifier
        sampled_data = []   # sampled data of each weak classifier
        h = None
        T = 10  # the number of weak classifiers

        for i in range(T):
            
            # sample data D_i based on current weights
            sample_x = np.random.choice(a=self.x, size=self.n, replace=True, p=self.weight)
            sample_y = self.y[sample_x]
            sampled_data.append([sample_x, sample_y])

            # obtain the threshold and predicted value of the current optimal weak classifier
            threshold, pos_direction = self.weak_classifier(sample_x, sample_y)
            y_pred = np.array([1 if x_ > threshold else -1 for x_ in self.x])
            if not pos_direction:
                y_pred *= -1

            # compute the error rate for the current weak classifier on D
            error = self.compute_error(self.weight, self.y, y_pred)

            # get the weight of weak classifier
            estimator_weight = np.log((1-error) / error) / 2

            # store the weak classifier
            estimator_weights.append(estimator_weight)
            estimator_errors.append(error)
            estimator_thresholds.append((threshold, y_pred)) 

            # update the weight of sampled data
            self.weight *= np.exp(-1 * estimator_weight * self.y * y_pred)
            self.weight /= np.sum(self.weight)

            # add up the weighted average of each weak classifier
            if i == 0:
                h = estimator_weight * y_pred
            else:
                h += (estimator_weight * y_pred) 

            # early stop if error is zero
            if self.compute_error(self.weight, self.y, np.sign(h)) == 0.:
                break

        # apply sign function
        h = np.sign(h)
        return h, estimator_weights, estimator_errors, estimator_thresholds, sampled_data


adaboost = Adaboost()
final_y_pred, estimator_weights, estimator_errors, estimator_thresholds, sampled_data = adaboost.train()

print("-----details of each weak classifier-----")
for i, weight, error, (threshold, y_pred), data in zip(
                                            list(np.arange(len(estimator_weights))), 
                                            estimator_weights, 
                                            estimator_errors,
                                            estimator_thresholds, 
                                            sampled_data):
    print("sampled_x: [%s]" % ",".join(list(map(str, data[0]))))
    print("sampled_y: [%s]" % ",".join(list(map(str, data[1]))))
    print("y_pred: [%s]" % ",".join(list(map(str,y_pred))))
    print("estimator_weight: %.4f, error: %.4f, threshold: %.1f" % (weight, error, threshold))
    print()

print("-----final predict value-----")
print(final_y_pred)

print("-----error of the adaboost model-----")
print(adaboost.compute_error(adaboost.weight, adaboost.y, final_y_pred))

print("-----final data weight-----")
print(adaboost.weight)