from sklearn import metrics
def removeDuplicates(arr): 
    res = []
    res_set = set()
    for a in arr: 
        if a not in res_set:
            res_set.add(a)
            res.append(a)
    return res

def roc_curve(label, prob):
    fpr, tpr, thresholds = metrics.roc_curve(label, prob, pos_label=0)
    return fpr, tpr, thresholds

