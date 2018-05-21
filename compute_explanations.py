from __future__ import print_function
import os, sys
import argparse
import pickle
import xgboost
import sklearn
import sklearn.neural_network
import utils
import anchor_tabular
import numpy as np
import warnings


def main():
    parser = argparse.ArgumentParser(description='Compute some explanations.')
    parser.add_argument('-d', dest='dataset', required=True,
                        choices=['adult', 'recidivism', 'lending'],
                        help='dataset to use')
    parser.add_argument('-e', dest='explainer', required=True,
                        choices=['lime', 'anchor', 'counterfactual', 'counterfactual-high-precision'],
                        help='explainer, either anchor or lime or counterfactual')
    parser.add_argument('-m', dest='model', required=True,
                        choices=['xgboost', 'logistic', 'nn'],
                        help='model: xgboost, logistic or nn')
    parser.add_argument('-c', dest='checkpoint', required=False,
                        default=200, type=int,
                        help='checkpoint after this many explanations')
    parser.add_argument('-o', dest='output', required=True)

    args = parser.parse_args()
    dataset = utils.load_dataset(args.dataset, balance=True)
    ret = {}
    ret['dataset'] = args.dataset
    for x in ['train_idx', 'test_idx', 'validation_idx']:
            ret[x] = getattr(dataset, x)

    explainer = anchor_tabular.AnchorTabularExplainer(
        dataset.class_names, dataset.feature_names,
        dataset.data, dataset.categorical_names)
    explainer.fit(dataset.train, dataset.labels_train,
                  dataset.validation, dataset.labels_validation)

    #if 'counterfactual' in args.explainer:
    if False:
        #??? Models fitted fast enough that we don't need to load models from pickle
        # Load classifier from other pickles
        filename = os.path.join(
            'out_pickles', '%s-anchor-%s' % (
            args.dataset, args.model))
        already_fitted_ret = pickle.load(open(filename))
        c = already_fitted_ret['model']
    else:
        if args.model == 'xgboost':
            c = xgboost.XGBClassifier(n_estimators=400, nthread=10, seed=1)
            c.fit(explainer.encoder.transform(dataset.train), dataset.labels_train)
        if args.model == 'logistic':
            c = sklearn.linear_model.LogisticRegression()
            c.fit(explainer.encoder.transform(dataset.train), dataset.labels_train)
        if args.model == 'nn':
            c = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(50,50))
            c.fit(explainer.encoder.transform(dataset.train), dataset.labels_train)



    ret['encoder'] = explainer.encoder
    ret['model'] = c
    ret['model_name'] = args.model

    def predict_fn(x):
        return c.predict(explainer.encoder.transform(x))

    def predict_proba_fn(x):
        return c.predict_proba(explainer.encoder.transform(x))

    print('Train', sklearn.metrics.accuracy_score(dataset.labels_train,
                                                  predict_fn(dataset.train)))
    print('Test', sklearn.metrics.accuracy_score(dataset.labels_test,
                                                 predict_fn(dataset.test)))
    threshold = 0.95
    tau = 0.1
    delta = 0.05
    epsilon_stop = 0.05
    batch_size = 100
    if args.explainer == 'anchor':
        explain_fn = utils.get_reduced_explain_fn(
            explainer.explain_lucb_beam, c.predict, threshold=threshold,
            delta=delta, tau=tau, batch_size=batch_size / 2,
            sample_whole_instances=True,
            beam_size=10, epsilon_stop=epsilon_stop)
    elif args.explainer == 'lime':
        explain_fn = utils.get_reduced_explain_fn(
            explainer.explain_lime, c.predict_proba, num_features=5,
            use_same_dist=True)
    elif args.explainer == 'counterfactual':
        explain_fn = utils.get_reduced_explain_fn(
            explainer.explain_counterfactual, c.predict, 
            threshold=0, max_manifolds=5
        )
    elif args.explainer == 'counterfactual-high-precision':
        explain_fn = utils.get_reduced_explain_fn(
            explainer.explain_counterfactual, c.predict, 
            threshold=threshold, max_manifolds=5
        )

    ret['exps'] = []
    prec_vec = np.nan*np.ones(dataset.validation.shape[0])
    coverage_vec = np.nan*np.ones(dataset.validation.shape[0])
    for i, d in enumerate(dataset.validation, start=1):
        # print(i)
        if i % 100 == 0:
            print(i)
        if i % args.checkpoint == 0:
            print('Checkpointing')
            if 'counterfactual' not in args.explainer:
                pickle.dump(ret, open(args.output + '.checkpoint', 'w'))

        explanation = explain_fn(d)
        if 'counterfactual' in args.explainer:
            # Compute prec and recall immediately
            X_test = dataset.test

            # Save original manifolds and reset to all
            orig_manifold_idx = explanation.selected_manifold_idx_
            explanation.selected_manifold_idx_ = np.arange(X_test.shape[1])
            X_test_proj, _ = explanation._project(X_test)  # Ignore selected_idx output of this function
            explanation.selected_manifold_idx_ = orig_manifold_idx

            # Predict on projection
            y_pred_model = predict_fn(X_test_proj)
            y_pred = explanation.predict(X_test)
            y_correct = 1 - np.abs(y_pred - y_pred_model)
            
            # Compute precision and coverage
            if isinstance(y_pred, np.ma.MaskedArray):
                n_correct = y_correct.sum()
                n_retrieved = np.sum(~y_pred.mask) # Not masked == selected
            else:
                raise RuntimeError('y_pred should be masked')
                n_correct = np.sum(y_correct)
                n_retrieved = len(y_test)
            n_total = len(y_pred_model)
            if n_retrieved == 0:
                prec = 1
                coverage = 0
            else:
                prec = float(n_correct)/float(n_retrieved)
                coverage = float(n_retrieved)/float(n_total)
            prec_vec[i-1] = prec
            coverage_vec[i-1] = coverage
            print('i=%5d, prec = %.5f, rec = %.5f' % (i-1, prec, coverage))
            
        ret['exps'].append(explanation)

    if 'counterfactual' not in args.explainer:
        pickle.dump(ret, open(args.output, 'w'))
    else:
        print('WARNING: counterfactual explanations cannot be pickled giving prec directly')
        prec_mean = np.mean(prec_vec)
        coverage_mean = np.mean(coverage_vec)
        prec_std = np.std(prec_vec)
        coverage_std = np.std(coverage_vec)
        print('Avg prec = %g +/- %g, Avg coverage = %g +/- %g' 
              % (prec_mean, prec_std, coverage_mean, coverage_std))


if __name__ == '__main__':
    main()
