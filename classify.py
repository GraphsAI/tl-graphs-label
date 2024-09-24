import random
import argparse
import numpy as np

from utils import test, saveIndex
from training.alignment import optimizeModel, getAlign, getDimShared
from training.dataManagement import loadData, getScaler, convertData

random.seed(0)

np.random.seed(0)
np.set_printoptions(suppress=True)

base_f = "./"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify Transfer Learning")

    parser.add_argument('-b', '--base', type=str, required=True,
                        help="Base dataset to align others")

    parser.add_argument('-d', '--datasets', nargs='+', required=True,
                        help='Other datasets usable for training')

    parser.add_argument('-t', '--transfer', action='store_true',
                        help="Joint training on all datasets")

    parser.add_argument('-save_index', '--save_index', action='store_true',
                        help="Save the index of alignment")

    parser.add_argument('-K', '--K', type=int, default=2,
                        help="Maximum number of cycles for alignment")

    parser.add_argument('-m', '--model', type=str, default="RFC",
                        help="Model for classifying (RFC, SVC, KNN, MLP)")

    parser.add_argument('-s', '--scaler', type=str, default="StandardScaler",
                        help="Scaler (StandardScaler, RobustScaler, Normalizer, PowerTransformer)")

    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose output")

    args = parser.parse_args()

    print("Base:", args.base)
    print("Datasets:", args.datasets)
    print("Transfer:", args.transfer)
    print("Model:", args.model)
    print("Pre-processing:", args.scaler)
    print()

    n_dataset = args.base
    aligned = args.datasets
    include_train = args.transfer

    model_name = args.model
    scaler_name = args.scaler

    train_avg = [[], []]
    test_avg = [[], []]
    train_avg_aligned = [[[], []] for i in range(len(aligned))]
    test_avg_aligned = [[[], []] for i in range(len(aligned))]

    for fold in range(0, 10):

        print(f"------- Fold {fold} -------")

        # Get base dataset
        x_base, x_train, y_train, x_test, y_test = loadData(base_f, args.base, fold)

        base_aligned = []
        x_train_aligned = []
        y_train_aligned = []
        x_test_aligned = []
        y_test_aligned = []

        # Get other datasets
        for i in range(len(aligned)):

            name_dataset = aligned[i]
            x_base1, x_train1, y_train1, x_test1, y_test1 = loadData(base_f, name_dataset, fold)

            base_aligned.append(x_base1)
            x_train_aligned.append(x_train1)
            y_train_aligned.append(y_train1)
            x_test_aligned.append(x_test1)
            y_test_aligned.append(y_test1)

        scaler1 = getScaler(scaler_name)

        dim_shared = getDimShared(x_base, base_aligned)

        # Optimize model on base dataset
        model = optimizeModel(x_train, y_train, scaler1, model_name, dim_shared, args.verbose)

        # Get a good alignment
        dist_aligned = getAlign(x_train, y_train, x_train_aligned, y_train_aligned, model, scaler1, args.K, dim_shared)

        if args.save_index:
            saveIndex(args.base, args.dataset, dist_aligned, base_f, fold, args.model)
            continue

        # Convert data in Z representation
        x_train, x_test, x_train_aligned, x_test_aligned = convertData(x_train, x_test,
                                                                       x_train_aligned, x_test_aligned,
                                                                       dim_shared, dist_aligned)

        x_sample = [0 for _ in range(len(x_train))]

        # Include if joint training set for transfer learning
        if include_train:
            for i in range(len(aligned)):
                x_train = np.concatenate((x_train, x_train_aligned[i]), 0)
                y_train = np.concatenate((y_train, y_train_aligned[i]), 0)
                x_sample.extend([i + 1 for _ in range(len(x_train_aligned[i]))])

        samples, sample_weights = np.unique(x_sample, return_counts=True)
        sample_weights = 1 - (sample_weights / sum(sample_weights))

        # Train the model again
        if args.base == "MUTAG" and model_name == "MLP":
            sample_weights = sample_weights/2

        sample_weights[0] = 1

        if args.base == "MUTAG" and model_name == "SVM":
            sample_weights[0] = 1000

        samples_count = sample_weights
        sample_weight_dict = dict(zip(samples, samples_count))

        x_sample = np.array([sample_weight_dict[s] for s in x_sample])
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)

        x_train = x_train[indices]
        x_sample = x_sample[indices]
        y_train = y_train[indices]

        x_train = scaler1.fit_transform(x_train)
        x_test = scaler1.transform(x_test)

        for i in range(len(x_train_aligned)):
            x_train_aligned[i] = scaler1.transform(x_train_aligned[i])

        for i in range(len(x_test_aligned)):
            x_test_aligned[i] = scaler1.transform(x_test_aligned[i])

        if model_name == "KNN":
            model.fit(x_train, y_train)
        elif model_name == "MLP":
            if args.base == "MUTAG":
                model.fit(x_train, y_train, sample_weight=x_sample, batch_=16)
            else:
                model.fit(x_train, y_train, sample_weight=x_sample)

        else:
            model.fit(x_train, y_train, sample_weight=x_sample)

        print(f"Test {n_dataset} (Base)")
        test(model, x_test, y_test, test_avg)

        for i in range(len(x_train_aligned)):
            print(f"Test {aligned[i]}")
            test(model, x_test_aligned[i], y_test_aligned[i], test_avg_aligned[i])
