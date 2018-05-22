import subprocess
datasets = ['adult', 'recidivism', 'lending']
#explainers = ['anchor', 'lime']
explainers = ['counterfactual', 'counterfactual-high-precision']
models = ['xgboost', 'logistic', 'nn']
projection = 'counterfactual' # Could be 'none' or 'counterfactual'
out = 'out_pickles'
for dataset in datasets:
    for explainer in explainers:
        for model in models:
            #outfile = '/tmp/%s-%s-%s.log' % (dataset, explainer, model)
            outfile = 'fixed-result-%s-%s-%s.log' % (dataset, explainer, model)
            print 'Outfile:', outfile
            outfile = open(outfile, 'w', 0)
            cmd = 'python compute_explanations.py -d %s -e %s -m %s --projection %s -o %s ' % (
                dataset, explainer, model, projection,
                '%s/%s-%s-%s' % (out, dataset, explainer, model))
            print cmd
            subprocess.Popen(cmd.split(), stdout=outfile, stderr=outfile)
