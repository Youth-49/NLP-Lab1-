# -*- coding:utf-8 _*-


def Score(pred: list, groundTrue: list) -> tuple:
    P = 0.0
    R = 0.0
    F = 0.0
    N_word = len(groundTrue)
    N_positive = 0
    N_negative = 0
    if pred is None or groundTrue is None:
        print('prediction list or groundTrue list is None!')
        return (0.0, 0.0, 0.0)

    for itemPred in pred:
        if itemPred in groundTrue:
            N_positive = N_positive + 1
        else:
            N_negative = N_negative + 1

    P = N_positive / (N_positive+N_negative)
    R = N_positive / N_word
    F = 2.0 * P * R / (P + R)
    return (P, R, F)

def Score_hard(pred: list, groundTrue: list) -> tuple:
    P = 0.0
    R = 0.0
    F = 0.0
    N_word = len(groundTrue)
    N_positive = 0
    N_negative = 0
    if pred is None or groundTrue is None:
        print('prediction list or groundTrue list is None!')
        return (0.0, 0.0, 0.0)

    for (itemPred, itemTrue) in zip(pred, groundTrue):
        if itemPred == itemTrue:
            N_positive = N_positive + 1
        else:
            N_negative = N_negative + 1


    P = N_positive / len(pred)
    R = N_positive / N_word
    F = 2.0 * P * R / (P + R)
    return (P, R, F)