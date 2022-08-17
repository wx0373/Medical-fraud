#!/usr/bin/env python
# coding: utf-8

# In[1]:


class multi_curves(object):
    def __init__(self, X_train, y_train, X_test,y_test,model):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
    

    def plot_ROC_train(self,i):
        from sklearn import metrics
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_auc_score

    
    #plt.figure(0).clf()

        scores = self.model.predict_proba(self.X_train)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(self.y_train, scores)
        plt.plot(fpr,tpr ,label='Train (top '+str(i)+' features)')

        plt.title('Train ROC Curve')
        plt.legend()
    

    def plot_ROC_test(self,i):
    #plt.figure(0).clf()
        from sklearn import metrics
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_auc_score
            
        roc_auc_score_test  = roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        scores = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(self.y_test, scores)
        plt.plot(fpr,tpr ,label='Test (top '+str(i)+' features)'+' AUC: '+str(round(roc_auc_score_test,3)))
        print ('roc_auc_score for the test dataset: {:.3f}'.format(roc_auc_score_test))
        plt.title('Test ROC Curve')
        plt.legend()
        
        
    
    def plot_PR_train(self,i):
        
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import auc,plot_precision_recall_curve
        
        ####test plot
        y_train_proba = self.model.predict_proba(self.X_train)
        y_train_score = y_train_proba[:, 1]
        precision, recall, thresholds = precision_recall_curve(self.y_train, y_train_score)
        auc_precision_recall = auc(recall, precision)
        print('train'+' PR-AUC is {:.3f}'.format(auc_precision_recall))
   
        plt.plot(recall, precision,label="train"+' (top '+str(i)+' features)')
        plt.xlabel('Recall(Positive label:1)')
        plt.ylabel('Precision(Positive label:1)')
        plt.title('Precision-Recall Curve'+' of '+'train')
        plt.legend()
    def plot_PR_test(self,i):
        
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import auc,plot_precision_recall_curve
        
        ####test plot
        y_test_proba = self.model.predict_proba(self.X_test)
        y_test_score = y_test_proba[:, 1]
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_test_score)
        auc_precision_recall = auc(recall, precision)
        print('Test'+' PR-AUC is {:.3f}'.format(auc_precision_recall))
   
        plt.plot(recall, precision,label="Test"+' (top '+str(i)+' features)'+' AUC: '+str(round(auc_precision_recall,3)))
        plt.xlabel('Recall(Positive label:1)')
        plt.ylabel('Precision(Positive label:1)')
        plt.title('Precision-Recall Curve'+' of '+'Test')
        plt.legend()
        

