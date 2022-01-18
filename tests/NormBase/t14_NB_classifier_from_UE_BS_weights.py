import numpy as np
np.set_printoptions(precision=3, linewidth=200, suppress=True)

"""
function to test the classifyExpression_W_NormBase function within Unreal Engine

run: python -m tests.NormBase.t14_NB_classifier_from_UE_BS_weights
"""


# declare funcgtion
def NB_classifier(x):
    """
    create classifier function to avoid loading the full NormBase class and to mimic UE implementation

    function assume that nu = 1

    :param x: input as 1D array (one frame at a time)
    :return:
    """
    v = np.linalg.norm(tun_vector, ord=2, axis=1) + 1e-7
    print("v", v)
    f = tun_vector @ x.T @ np.diag(np.power(v, -1))
    f[f < 0] = 0  # ReLu activation instead of dividing by 2 and adding 0.5

    return f


# test 1 - control test
print("----------------------------------------------------------")
print("                     Test 1 ")

tun_vector = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 2, 0]])
print("tun_vector")
print(tun_vector)

xs = np.array([[1, 0, 0, 0],
              [.5, 0, 0, 0],
               [0, 1, 0, 0],
               [.5, .2, 0, 0],
               [1, 0, 1, 0],
               [0, 0, 0, 1],
               [.2, .3, .4, 1],
               [2, 3, 0, 0]])
for x in xs:
    preds = NB_classifier(x)
    print("preds (x={}): \t {}".format(x, preds))
print()

# test 2 - dummy test
print("----------------------------------------------------------")
print("                     Test 2 ")

tun_vector = np.array([[1, 1, 0, 0],
                       [0, 2, 2, 2],
                       [0, 2, 1, 0]])
print("tun_vector")
print(tun_vector)

x = np.array([2, 3, 4, 1])

preds = NB_classifier(x)
print("preds (x={}): \t {}".format(x, preds))
print()

# test 3 - test with ARKKit parameters
print("----------------------------------------------------------")
print("                     Test 3 ")
# 0: Neutral, 1:Happy, 2: Sadness, 3: Surprise, 4: Fear, 5: Anger, 6: Disgust
tun_vector = np.zeros((7, 61))
