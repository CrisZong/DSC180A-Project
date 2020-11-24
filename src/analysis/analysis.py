def offset(X, y, offset, f):
    if offset >= 0:
        return f(X.shift(offset)[offset:], y[offset:])
    return f(X.shift(offset)[:offset], y[:offset])
def showLoss(X,y,lower,upper,fs):
    for i in range(lower, upper):
        print(f"Offset: {i}. \n\tSpearman/Pearson Loss Score: \
            {round(offset(X, y, i, fs[0]), 2)}\n\tPearson Loss Score: \
            {round(offset(X, y, i, fs[1]), 2)}\n\tSpearman Loss Score: \
            {round(offset(X, y, i, fs[2]), 2)}")