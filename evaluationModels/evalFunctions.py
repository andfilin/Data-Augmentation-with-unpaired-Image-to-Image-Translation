import numpy as np
import edit_distance
from pathlib import Path
import cv2

####
# add white border at top of image and draw text there.
####
def _annotate_image(image, text):
    image = cv2.copyMakeBorder(image, 12, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
    cv2.putText(image, text, (1,11), cv2.FONT_HERSHEY_PLAIN, 1, 0)
    return image




####
# takes array of predicted-truth pairs where at least one predictionerror occured.
# outputs arrays containing which digits exactly where predicted wrong.
####
# input: mimatches - output from fcsrn.accuracy()
####
# returns: (
#    error_replace-array: (predicted digit, expected digit), predicted label, truthlabel
#    error_extra-array:         extra predicted digit, predicted label, truthlabel
#    error_missing-array:       missing predicted digit, predicted label, truthlabel
#)
####
def list_errors(mismatches):
    error_replace = [] # wrongly predicted digits
    error_extra = [] # errors where too many characters predicted
    error_missing = [] # too few characters predicted
    # iterate every sample with mispredictions
    for pred, truth, diff, image in mismatches:
        # ignore -1/BLANK digits at end of pred
        index_blank = np.argwhere(pred==-1)
        index_blank = index_blank[0,0] if len(index_blank) > 0 else None
        pred = pred[0:index_blank]
        
        # get minimal editdistance with ops
        matcher = edit_distance.SequenceMatcher(pred, truth)
        for op, i1,i2, j1,j2 in matcher.get_opcodes():
            if op == "replace":
                # one or more characters were not predicted correctly
                wrong_predictions = pred[i1:i2]
                expected_predictions = truth[j1:j2]
                for pair in zip(wrong_predictions,expected_predictions):  
                    annotated_image = _annotate_image(image, "Pred: %d Truth: %d"%(pair[0],pair[1]))
                    error_replace.append((pair, pred, truth, annotated_image))   
            elif op == "delete":
                # one or more characters more than necessary predicted
                extra_predictions = pred[i1:i2]            
                for extra_pred in extra_predictions:
                    assert extra_pred != -1, "blank encountered"
                    annotated_image = _annotate_image(image, "Extra: %d"%(extra_pred))
                    error_extra.append((extra_pred, pred, truth, annotated_image))
            elif op == "insert":
                # one or more characters were not predicted at all
                missed_predictions = truth[j1:j2]
                for missing_pred in missed_predictions:
                    annotated_image = _annotate_image(image, "Missing: %d"%(missing_pred))
                    error_missing.append((missing_pred, pred, truth, annotated_image))
    error_replace = np.array(error_replace)
    error_extra = np.array(error_extra)
    error_missing = np.array(error_missing)
    return (error_replace, error_extra, error_missing)


def print_missingErrors(err_missing, sort=True, listErrors = False, printResult=True):
    s = ""
    s += "-------------------------------------\n"
    s += "Missing Digit Errors\n"
    s += "-------------------------------------\n"
    
    s += "\tDigits by #missed:\n"
    digits, counts = np.unique([row[0] for row in err_missing], return_counts=True)
    s += "digit\t#missing(total=%d)\n"%(sum(counts))
    for digit, count in zip(digits, counts):
        s += "%d:\t%d\n"%(digit,count)
    
    if listErrors:
        s += "\n\tList of every missing-error:\n"
        s += "missing\tpredicted\ttruth\n"
        for err in sorted(list(err_missing), key=lambda row:row[0]):
            s += "%d\t%s \t%s\n"%(err[0],err[1],err[2])
        
    if printResult:
        print(s)
    return s

def print_extraErrors(err_extra, sort=True, listErrors = False, printResult=True):
    s = ""
    s += "-------------------------------------\n"
    s += "Extra Digit Errors\n"
    s += "-------------------------------------\n"
    
    s += "\tDigits by #extra:\n"
    digits, counts = np.unique([row[0] for row in err_extra], return_counts=True)
    s += "digit\t#extra(total=%d)\n"%(sum(counts))
    for digit, count in zip(digits, counts):
        s += "%d:\t%d\n"%(digit,count)
    
    if listErrors:
        s += "\n\tList of every extra-error:\n"
        s += "extra\tpredicted\ttruth\n"
        for err in sorted(list(err_extra), key=lambda row:row[0]):
            s += "%d\t%s   \t%s\n"%(err[0],err[1],err[2])
        
    if printResult:
        print(s)
    return s

def print_replaceErrors(err_replace, sort=True, listErrors = False, printResult=True):
    s = ""
    s += "-------------------------------------\n"
    s += "Replaced Digit Errors\n"
    s += "-------------------------------------\n"
    
    s += "\tTruthDigits by #mistaken\n"
    digits, counts = np.unique([row[0][1] for row in err_replace], return_counts=True)
    s += "truthdigit\t#mistaken(total=%d)\n"%(sum(counts))
    for digit, count in zip(digits, counts):
        s += "%d:\t\t%d\n"%(digit,count)
        
        
    s += "\tPredDigits by #mistake\n"
    digits, counts = np.unique([row[0][0] for row in err_replace], return_counts=True)
    s += "preddigit\t#misstake(total=%d)\n"%(sum(counts))
    for digit, count in zip(digits, counts):
        s += "%d:\t\t%d\n"%(digit,count)
        
    s += "\t replaced-pairs by #\n"
    pairCounts = {}
    sumPairs = 0
    for pair in [row[0] for row in err_replace]:
        if pairCounts.get(pair) ==  None:
            pairCounts[pair] = 0                    
        pairCounts[pair] += 1
        sumPairs += 1
    s += "(pred,truth)\t#(total=%d)\n"%(sumPairs)
    for key,value in sorted(pairCounts.items(), key=lambda item: -item[1]):
        s += "%s  \t%d\n"%(key, value)
    
    if printResult:
        print(s)
    return s

def log_errors(mismatches, labels, printResult = True, verbose = True, savePath=None):
    
    # count errors
    error_replace, error_extra, error_missing = list_errors(mismatches)
    # count total characters
    digits, counts = np.unique(labels.flatten(), return_counts=True)
    
    s = ""
    
    s += "-------------------------------------\n"
    s += "Total Characters\n"
    s += "-------------------------------------\n"
    s += "digit\tcount(total=%d)\n"%(sum(counts))
    for digit, count in zip(digits, counts):
        s += "%d\t%d\n"%(digit, count)
    
    s += "-------------------------------------\n"
    s += "Total Errors\n"
    s += "-------------------------------------\n"
    s += "digits missing:\t%d\n"%(len(error_missing))
    s += "digits extra:\t%d\n"%(len(error_extra))
    s += "digits replaced:\t%d\n"%(len(error_replace))
    
    
    s += print_missingErrors(error_missing, sort=True, listErrors = verbose, printResult=False)
    s += print_extraErrors(error_extra, sort=True, listErrors = verbose, printResult=False)
    s += print_replaceErrors(error_replace, sort=True, listErrors = verbose, printResult=False)
    
    
        
    if savePath != None:
        if not isinstance(savePath, Path):
            savePath = Path(savePath)
            
        assert not savePath.exists(), "specified folder for saving errors already exists"
        savePath.mkdir()
        print("created dir: %s"%(str(savePath)))
        logfile = savePath/"errors.txt"
        logfile.touch()
        logfile.write_text(s)
        print("wrote evalfile to: %s"%(str(logfile)))
        
        ####
        # save errorimages
        ####
        # replace-errors
        errReplaceFolder = savePath / "errors_replaced"
        errReplaceFolder.mkdir()
        for index, row in enumerate(error_replace):
            image = row[-1]
            trueDigit = row[0][1]
            predDigit = row[0][0]
            filename = "true_%d_pred_%d_%d.png" % (trueDigit, predDigit, index)
            file = errReplaceFolder / filename
            cv2.imwrite(str(file), image.astype(int))
        ####
        # missing-errors
        errMissingFolder = savePath / "errors_missing"
        errMissingFolder.mkdir()
        for index, row in enumerate(error_missing):
            image = row[-1]
            missingDigit = row[0]
            filename = "missing_%d_%d.png" % (missingDigit, index)
            file = errMissingFolder / filename
            cv2.imwrite(str(file), image.astype(int))
        ####
        # extra-errors
        errExtraFolder = savePath / "errors_extra"
        errExtraFolder.mkdir()
        for index, row in enumerate(error_extra):
            image = row[-1]
            extraDigit = row[0]
            filename = "extra_%d_%d.png" % (extraDigit, index)
            file = errExtraFolder / filename
            cv2.imwrite(str(file), image.astype(int))
            
        
        
    else:
        print("no savepath specified, errors will not be saved")
        
    if printResult:
        print(s)
                
    return s