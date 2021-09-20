import abc

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report, precision_score, recall_score

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Evaluator(abc.ABC):

    def __init__(self, threshold=None):
        self.threshold = threshold

    def _transform(self, y_pred):
        if self.threshold is None:
            return y_pred.argmax(1)
        return (y_pred > self.threshold).argmax(1)

    @abc.abstractmethod
    def visualize(self):
        raise NotImplemented()
    
    @abc.abstractmethod
    def evaluate(self, y_true, y_pred):
        raise NotImplemented()


class ClassificationEvaluator(Evaluator):

    def __init__(self, threshold=None):
        self.threshold = threshold

    def visualize(self):
        pass # TODO
    
    def evaluate(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        conf_mat = confusion_matrix(y_true, y_pred)

        avg_prec = precision_score(y_true, y_pred, average=None)
        prec = sum(avg_prec[1:]) / len(avg_prec[1:])
        rec_score = recall_score(y_true, y_pred, average=None)
        rec = sum(rec_score[1:]) / len(rec_score[1:])
        print(conf_mat, "\n", 
                acc*100, "% accuracy\n", 
                prec*100, f"% precision of {len(avg_prec) -1} classes\n",
                rec*100, f"% recall of {len(rec_score) -1} classes\n",
                )
        report = classification_report(y_true, y_pred)
        print(report)
        results = [acc*100, prec*100, rec*100,]
        return results



class RegressionEvaluator(Evaluator): 
    
    def visualize(self):
        pass # TODO
    
    def evaluate(self, y_true, y_pred):
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true=y_true, y_pred=y_pred)
        results = [mae, mse, rmse, r2]
        
        print(
f"""MAE:  {mae:>6.4f} 
MSE:  {mse:>6.4f}
RMSE: {rmse:>6.4f}
R2:   {r2:>6.4f}
""")
        return results