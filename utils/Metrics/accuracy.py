def compute_accuracy(preds, labels):
    n_correct = 0
    n_total = len(labels)

    for i in range(n_total):
        if preds[i] == labels[i]:
            n_correct += 1

    return n_correct / n_total
