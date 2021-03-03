import numpy as np
import matplotlib.pyplot as plt

# print(tf.test.is_gpu_available())

np.set_printoptions(precision=4, linewidth=150)
"""
this script simply reproduce first the Matlab version of the NormBase mechanism, I then tried different ways to compute 
the means and tuning vector as to accept batch of data so the model fits better into a common machine learning 
pipeline such as in TensorFlow

run: python -m tests.t14a_test_norm_base_layer
"""

print("Test Norm Base Layer")

n_features = 5
n_samples = 36
n_category = 3

# create random test case
np.random.seed(0)
train_data = np.random.rand(n_samples, n_features)  # mimic one batch input
train_label = np.random.randint(n_category, size=n_samples)
# print("shape train data", np.shape(train_data))
# print("shape train_label", np.shape(train_label))
# print(train_data)
# print(train_label)
# print()

# ------------------------------------------------------------------------
# -----------------           Norm Base model         --------------------
# ------------------------------------------------------------------------
print("shape train_data", np.shape(train_data))
print("shape train_label", np.shape(train_label))

# -----------------               Training            --------------------
print()
print("------------ reference ------------")
# compute references pattern m
m = np.mean(train_data[train_label == 0], axis=0)
print("m")
print(m)
print()
# # build model
# input = Input(shape=(num_features))
# x = NormBase()(input)
# model = Model(inputs=input, output=x)
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])


# compute references direction tuning n
n = np.zeros((n_category, n_features))
# compute difference u - m with ref vector m
diff = train_data - np.repeat(np.expand_dims(m, axis=1), n_samples, axis=1).T
for i in range(n_category):
    # get data for each category
    cat_diff = diff[train_label == i]
    # get mean of each category
    cat_diff_mean = np.mean(cat_diff, axis=0)
    n[i, :] = cat_diff_mean / np.linalg.norm(cat_diff_mean)
print("n", np.shape(n))
print(n)

# --------------------------------------------------------------------------------
# -------------------------       cumulative way       ---------------------------
# --------------------------------------------------------------------------------
print()
print("----------- cumulative batch computing ------------")
batch_size = 12
num_iter = int(n_samples/batch_size)
m = np.zeros(n_features)
m_cumul = 0
for b in range(num_iter):
    # -----------------------------------------------------------------------------
    # get batch entry
    start = b * batch_size
    end = start + batch_size
    batch_data = train_data[start:end, :]
    batch_label = train_label[start:end]

    # -----------------------------------------------------------------------------
    # update ref_vector m
    ref_data = batch_data[batch_label == 0]  # keep only data with ref = 0 (supposedly neutral face)
    n_ref = np.shape(ref_data)[0]
    m = (m_cumul*m + n_ref*np.mean(ref_data, axis=0))/(m_cumul + n_ref)
    m_cumul += n_ref
print("m")
print(m)
print()

n_mean = np.zeros((n_category, n_features))
n = np.zeros((n_category, n_features))
n_cumul = np.zeros(n_category)
m_batch = np.repeat(np.expand_dims(m, axis=1), batch_size, axis=1).T
for b in range(num_iter):
    # -----------------------------------------------------------------------------
    # get batch entry
    start = b * batch_size
    end = start + batch_size
    batch_data = train_data[start:end, :]
    batch_label = train_label[start:end]

    # -----------------------------------------------------------------------------
    # update direction tuning vector n

    # compute difference u - m with ref vector m
    batch_diff = batch_data - m_batch

    # compute direction tuning for each category
    for i in range(n_category):
        # get data for each category
        cat_diff = batch_diff[batch_label == i]
        # get num_of data
        n_cat_diff = np.shape(cat_diff)[0]
        # update cumulative mean for each category
        n_mean[i] = (n_cumul[i]*n_mean[i] + n_cat_diff*np.mean(cat_diff, axis=0))/(n_cumul[i] + n_cat_diff)
        # update cumulative counts
        n_cumul[i] += n_cat_diff

        n[i] = n_mean[i] / np.linalg.norm(n_mean[i])

print("n")
print(n)

# --------------------------------------------------------------------------------
# ----------------------    cumulative epoch way       ---------------------------
# --------------------------------------------------------------------------------
print()
print("----------- cumulative epoch computing ------------")
batch_size = 12
m = np.zeros(n_features)
n_mean = np.zeros((n_category, n_features))
n = np.zeros((n_category, n_features))
n_cumul = np.zeros(n_category)
ref_cumul = 0
num_epoch = 2
num_iter = int(n_samples/batch_size)
for e in range(num_epoch):  # todo, no need for more than 2 epochs...
    for b in range(num_iter):
        # get batch entry
        start = b * batch_size
        end = start + batch_size
        batch_data = train_data[start:end, :]
        batch_label = train_label[start:end]

        if e > 0:
            # update ref_vector n
            batch_diff = batch_data - np.repeat(np.expand_dims(m, axis=1), batch_size, axis=1).T

            # compute direction tuning for each category
            for i in range(n_category):
                # get data for each category
                cat_diff = batch_diff[batch_label == i]
                # get num_of data
                n_cat_diff = np.shape(cat_diff)[0]
                # update cumulative mean for each category
                n_mean[i] = (n_cumul[i] * n_mean[i] + n_cat_diff * np.mean(cat_diff, axis=0)) / (n_cumul[i] + n_cat_diff)
                # update cumulative counts
                n_cumul[i] += n_cat_diff

                n[i] = n_mean[i] / np.linalg.norm(n_mean[i])
        else:
            # update ref_vector m
            ref_data = batch_data[batch_label == 0]  # keep only data with ref = 0 (supposedly neutral face)
            n_ref = np.shape(ref_data)[0]
            m = (ref_cumul * m + n_ref * np.mean(ref_data, axis=0)) / (ref_cumul + n_ref)
            ref_cumul += n_ref

print("m")
print(m)
print()

print("n")
print(n)

# -----------------               Evaluate            --------------------
# compute response of norm-reference
v = np.zeros((n_samples, n_category))
num_iter = int(n_samples/batch_size)
m_batch = np.repeat(np.expand_dims(m, axis=1), batch_size, axis=1).T
for b in range(num_iter):
    # get batch entry
    start = b * batch_size
    end = start + batch_size
    batch_data = train_data[start:end, :]
    batch_label = train_label[start:end]

    # compute answer
    diff = batch_data - m_batch  # compute difference u - m
    norm_diff = np.sqrt(np.diag(diff.dot(diff.T)))  # |u-m|
    resp = np.diag(np.power(norm_diff, -1)).dot(diff)  # (u-m)/|u-m|
    tuning = resp.dot(n.T)  # (u-m)n/|u-m|
    tuning[tuning < 0] = 0  # keep only positive values # todo where is it written?
    v = np.diag(norm_diff).dot(np.power(tuning, n_category))  # |u-m|((u-m)n/|u-m|)  todo why power of n_category?
    # f = f / np.max(f)  # normalize
    # f_itp[:, :, c] = f

# print()
# print(v[:, 0])
# print(v[:, 1])


# -----------------               Plot            --------------------
plt.figure()
plt.plot(v[:, 0], label="class1")
plt.plot(v[:, 1], label="class2")
plt.plot(v[:, 2], label="class3")
plt.legend()
plt.show()
