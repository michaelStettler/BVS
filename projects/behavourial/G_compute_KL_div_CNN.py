import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from projects.behavourial.project_utils import *
from scipy.stats import wilcoxon


np.set_printoptions(precision=3, suppress=True)
"""
run: python -m projects.behavourial.07_compute_KL_div_CNN
"""

def compute_morph_space_KL_div(P, Q, sum=False):
    log = np.log(P / Q)
    if not sum:
        return np.sum(P * log, axis=-1)
    else:
        return np.sum(P * log)


def compute_morph_space_total_variation(P, Q, sum=False):
    res = 0.5 * np.sum(np.abs(P - Q) ** 2, axis=-1)
    if not sum:
        return res
    else:
        return np.sum(res)


def compute_entropy(P, normalize=False, sum=False):
    log = np.log(P)
    entropy = - np.sum(P * log, axis=-1)
    if normalize:
        entropy = entropy / np.sum(entropy)
    if not sum:
        return entropy
    else:
        return np.sum(entropy)

def compute_entropy_difference(P, Q, normalize=False, sum=False):
    A, B = compute_entropy(P), compute_entropy(Q)
    if normalize:
        A = A / np.sum(A)
        B = B / np.sum(B)
    if not sum:
        return np.abs(A - B)
    else:
        return np.sum(np.abs(A - B))

def wilcoxon_comparison(values, condition):
    print('Using condition:', condition)
    # find best NRE model
    nre_min, other_min = np.inf, np.inf
    for model_name, model_dict in values.items():
        if model_name.startswith('NRE') and np.sum(model_dict[condition]) < nre_min:
            best_nre = model_name
            nre_min = np.sum(model_dict[condition])
    for model_name, model_dict in values.items():
        if not model_name.startswith('NRE') and np.sum(model_dict[condition]) < other_min:
            best_other = model_name
            other_min = np.sum(model_dict[condition])
    ###
    # best_nre = 'NRE_frobenius_static'
    ###
    print('Best models:', best_nre, best_other)
    _, p = wilcoxon(values[best_nre][condition].flatten(), values[best_other][condition].flatten())
    print(p)



def get_pred(load_path, model_name, condition):
    path = os.path.join(load_path, f"{model_name}_{condition}_prob_grid.npy")
    print('Getting preds from:', path)
    preds = np.load(path)
    return preds


def main():
    computer = 'windows'
    # computer = 'alex'

    computer_path, computer_letter = get_computer_path(computer)


    show_plots = True
    plot_format = 'svg'
    # model_names = ["NRE_individual_static", "NRE_individual_dynamic",
    #                "NRE_frobenius_static", "NRE_frobenius_dynamic",
    #                "VGG19_imagenet", "VGG19_imagenet_conv33", "Resnet50v2_imagenet",
    #                "VGG19_affectnet", "ResNet50v2_affectnet", "CORNet_affectnet",
    #                "CORNet_imagenet"]
    model_names = ["NRE_individual_static", "NRE_individual_dynamic",
                   "NRE_frobenius_static", "NRE_frobenius_dynamic",
                   "VGG19_imagenet", "Resnet50v2_imagenet",
                   "VGG19_affectnet", "ResNet50v2_affectnet", "CORNet_affectnet",
                   "CORNet_imagenet"]

    ### names for poster
    model_names = ["NRE_frobenius_static", "NRE_frobenius_dynamic",
                   "FORMER_DFER_linear", "ResNet50v2_affectnet_linear", "CORNet_affectnet_linear",
                   "M3DFEL_linear", 'Resnet50v2_imagenet_linear', 'CORNet_imagenet_linear']
    ###


    conditions = ["human_orig", "monkey_orig"]

    # model_names = ['NRE_frobenius_dynamic', 'VGG19_imagenet']
    # model_names = ['Resnet50v2_imagenet']
    # conditions = ["human_orig"]


    pred_dict = {}
    kl_divergences = {}
    total_variations = {}
    entropies = {}
    entropy_diffs = {}
    for k, model_name in enumerate(model_names):
        pred_model_dict = {}
        kl_model_dict = {}
        var_model_dict = {}
        entropy_model_dict = {}
        entropy_diff_model_dict = {}
        for cond, condition in enumerate(conditions):
            load_path = os.path.join(computer_path, 'model_behav_preds')

            # load data
            # load behavioural data
            behavioural_path = os.path.join(computer_path, 'morphing_psychophysics_result')
            if condition == "human_orig":
                behav_data = np.load(os.path.join(behavioural_path, "human_avatar_orig.npy"))
            elif condition == "monkey_orig":
                behav_data = np.load(os.path.join(behavioural_path, "monkey_avatar_orig.npy"))

            behav_data = np.moveaxis(behav_data, 0, -1)
            print("shape behav_data", np.shape(behav_data))
            if k == 0:  # Get entropy of behavioural data on first iteration
                if cond == 0:
                    behav_pred = {}
                    behav_entropy = {}
                behav_entropy[condition] = compute_entropy(behav_data, normalize=False)
                behav_pred[condition] = behav_data
                if cond == 1:
                    entropies['behavioural'] = behav_entropy
                    pred_dict['behavioural'] = behav_pred


            # load model preds
            if 'linear' in model_name:
                preds = get_pred(join(load_path, 'linear_fits'), model_name.replace('_linear', ''), condition)
            else:
                preds = get_pred(load_path, model_name, condition)

            # compute KL-divergence
            kl_div = compute_morph_space_KL_div(behav_data, preds)
            # Compute total variation distance
            var = compute_morph_space_total_variation(behav_data, preds)
            entropy = compute_entropy(preds, normalize=False)
            entropy_diff = compute_entropy_difference(behav_data, preds, normalize=False)
            # Compute entropy

            pred_model_dict[condition] = preds
            kl_model_dict[condition] = kl_div
            var_model_dict[condition] = var
            entropy_model_dict[condition] = entropy
            entropy_diff_model_dict[condition] = entropy_diff

            # print(model_name, np.sum(kl_div), np.sum(var))
            print('kl:', kl_div)
            # print('var:', var)
            print('--------------------')
            # save values
            np.save(os.path.join(load_path, f"{model_names[k]}_{conditions[cond]}_KL_div"), kl_div)
            # np.save(os.path.join(load_path, f"{model_names[k]}_{conditions[cond]}_total_variation"), var)
        pred_dict[model_name] = pred_model_dict
        kl_divergences[model_name] = kl_model_dict
        total_variations[model_name] = var_model_dict
        print('Trying to plot...')
        entropies[model_name] = entropy_model_dict
        entropy_diffs[model_name] = entropy_diff_model_dict



    #%%

    labels = ["NRE-indi-S", "NRE-indi-D",
                   "NRE-cat-S", "NRE-cat-D",
                   "VGG19-IM", "Resnet50v2-IM",
                   "VGG19-AN", "ResNet50v2-AN", "CORNet-AN",
                   "CORNet-IM"]

    ### Labels for the poster
    labels = [ "NRE-Static", "NRE-Dynamic",
                   "FORMER_DFER", "Resnet50v2-AN",
                   "CORNet-AN", "M3DFEL", "ResNet50v2-IM", "CORNet-IM"
               ]
    ####

    colors = ['#EC7357', '#78A1BB']

    legend_dict = {'human_orig': 'Human Avatar', 'monkey_orig': 'Monkey Avatar'}


    def sort_plot_data(data_dict, model_names, labels, fixed_preorder=None):
        arr = np.zeros(len(model_names))
        for i, model in enumerate(model_names):
            tot = 0
            for condition in conditions:
                tot += np.sum(data_dict[model][condition])
            arr[i] = tot
        order = np.argsort(arr)
        if fixed_preorder is not None:
            order = fixed_preorder
        print('Models:', len(model_names))
        print('Models:', model_names)
        model_names = list(np.array(model_names, dtype='object')[order])
        labels = list(np.array(labels, dtype='object')[order])
        return model_names, labels

    def make_bar_plot(data_dict, model_names, labels, colors, title, sort_plot=True, save_name=None):
        try:
            if sort_plot == True:
                model_names, labels = sort_plot_data(data_dict, model_names, labels)
        except:
            model_names, labels = sort_plot_data(data_dict, model_names, labels, fixed_preorder=sort_plot)
        fig, ax = plt.subplots()
        x = np.arange(len(labels))
        width = 0.25
        x = x + (width / 2)     # make sure that xticks are in correct spots
        # plot each condition
        print(model_names, labels)
        for i, model_name in enumerate(model_names):
            for c, condition in enumerate(conditions):
                offset = width * c
                rects = plt.bar(x[i] + offset, np.sum(data_dict[model_name][condition]),
                                width, label=legend_dict[condition], color=colors[c])
                if i == 0:
                    ax.legend()
        ax.set_xticks(x + width / 2, labels)
        plt.xticks(rotation=90)
        plt.title(title)
        plt.tight_layout()
        if save_name is not None:
            plt.savefig(join('plots', save_name) + '.' + plot_format, format=plot_format)
        plt.show()

    ### Fix the plot order for the poster
    fixed_order = np.array([0, 3, 4, 6, 7, 1, 5, 2])
    ###

    make_bar_plot(kl_divergences, model_names, labels, colors, title='KL Divergence', save_name='kl_divergence', sort_plot=fixed_order)
    make_bar_plot(total_variations, model_names, labels, colors, title="Total Variation Distance", save_name='total_var_dist', sort_plot=fixed_order)
    # make_bar_plot(entropy_diffs, model_names, labels, colors, title="Normalized Entropy Difference", save_name='entropy_diffs', sort_plot=fixed_order)
    model_names.insert(0, 'behavioural')
    labels.insert(0, 'Humans')
    entropy_models = ['behavioural', 'NRE_frobenius_static']
    entropy_labels = ['Humans', 'NRE-static']
    print('Entropy -------------------------------------------------')
    make_bar_plot(entropies, entropy_models, entropy_labels, colors, title='Entropy', sort_plot=False, save_name='entropy_bar_plot')


    ### Entropy heat maps

    nre_entropy = entropies['NRE_frobenius_static']['monkey_orig']
    cnn_entropy = entropies['ResNet50v2_affectnet_linear']['monkey_orig']
    behav_entropy = entropies['behavioural']['monkey_orig']
    print(nre_entropy)
    print(cnn_entropy)
    print(behav_entropy)
    max_entropy = np.maximum(np.max(nre_entropy), np.max(cnn_entropy))
    max_entropy = np.maximum(np.max(behav_entropy), max_entropy)
    print('Max entropy:', max_entropy)

    print('Entropy difference -----------------------------------------')
    print(np.sum(np.abs((nre_entropy - behav_entropy))))
    print(np.sum(np.abs((cnn_entropy - behav_entropy))))

    fig, ax = plt.subplots(1, 3)
    plt.title('Entropies')
    entropy_list = [behav_entropy, nre_entropy, cnn_entropy]
    title_list = ['Humans', 'NRE', 'ResNet']
    for i in range(3):
        im = ax[i].imshow(entropy_list[i], vmin=0, vmax=max_entropy)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].title.set_text(title_list[i])
    cax = fig.add_axes([ax[2].get_position().x1 + 0.02, ax[2].get_position().y0,
                        0.02, ax[2].get_position().y1-ax[2].get_position().y0])
    fig.colorbar(im, cax=cax)
    plt.savefig(join('plots', 'entropy_example.') + plot_format, format=plot_format)
    plt.show()

    for condition in conditions:
        print('Condition:', condition)
        wilcoxon_comparison(kl_divergences, condition)
        wilcoxon_comparison(total_variations, condition)
        wilcoxon_comparison(entropy_diffs, condition)
        print()

    print('Pred dict:')
    print(pred_dict['behavioural']['human_orig'][0, :, :])
    print(pred_dict['NRE_frobenius_static']['human_orig'][0, :, :])
    print(pred_dict['CORNet_imagenet_linear']['human_orig'][0, :, :])
    print(pred_dict['Resnet50v2_imagenet_linear']['human_orig'][0, :, :])
    # print(pred_dict['behavioural']['monkey_orig'][4, :, :])
    # print(pred_dict['Resnet50v2_imagenet']['monkey_orig'][4, :, :])

    # print(np.mean(np.abs(pred_dict['behavioural']['human_orig'] - pred_dict['Resnet50v2_imagenet']['human_orig'])))
    # print(np.mean(np.abs(pred_dict['behavioural']['monkey_orig'] - pred_dict['Resnet50v2_imagenet']['monkey_orig'])))

if __name__ == '__main__':
    main()